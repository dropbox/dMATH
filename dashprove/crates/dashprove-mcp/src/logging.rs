//! Request logging and audit trail for MCP server
//!
//! Provides structured logging of HTTP and WebSocket requests for debugging,
//! auditing, and monitoring purposes.
//!
//! ## Features
//!
//! - Structured log entries with timing, status, client IP
//! - Configurable ring buffer size (memory-bounded)
//! - Query/filter capabilities
//! - Export to JSON for analysis
//!
//! ## Example
//!
//! ```ignore
//! use dashprove_mcp::logging::{RequestLogger, LogConfig};
//!
//! let logger = RequestLogger::new(LogConfig::default());
//! logger.log_request(entry).await;
//! let logs = logger.query(filter).await;
//! ```

use std::collections::VecDeque;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

/// Configuration for the request logger
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogConfig {
    /// Maximum number of log entries to retain (ring buffer size)
    pub max_entries: usize,
    /// Whether request logging is enabled
    pub enabled: bool,
    /// Whether to log request bodies (can be large)
    pub log_request_body: bool,
    /// Whether to log response bodies (can be large)
    pub log_response_body: bool,
    /// Maximum body size to log (bytes) - bodies larger than this are truncated
    pub max_body_size: usize,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            enabled: true,
            log_request_body: false,
            log_response_body: false,
            max_body_size: 4096,
        }
    }
}

/// Type of request/transport
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RequestType {
    /// HTTP request
    Http,
    /// WebSocket message
    WebSocket,
    /// JSON-RPC call
    JsonRpc,
}

/// Status of the request
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RequestStatus {
    /// Request completed successfully
    Success,
    /// Request failed with an error
    Error,
    /// Request was rejected (auth, rate limit, etc.)
    Rejected,
}

/// A single request log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestLogEntry {
    /// Unique ID for this log entry
    pub id: u64,
    /// Timestamp when request was received (Unix timestamp ms)
    pub timestamp_ms: u64,
    /// Type of request
    pub request_type: RequestType,
    /// HTTP method (GET, POST, etc.) or WebSocket message type
    pub method: String,
    /// Request path or tool name
    pub path: String,
    /// Client IP address
    pub client_ip: Option<String>,
    /// HTTP status code (for HTTP requests)
    pub status_code: Option<u16>,
    /// Request status
    pub status: RequestStatus,
    /// Duration of request processing in milliseconds
    pub duration_ms: u64,
    /// Request body (if configured to log)
    pub request_body: Option<String>,
    /// Response body (if configured to log)
    pub response_body: Option<String>,
    /// Error message (if request failed)
    pub error: Option<String>,
    /// Additional metadata (request ID, session ID, etc.)
    #[serde(default)]
    pub metadata: std::collections::HashMap<String, String>,
}

impl RequestLogEntry {
    /// Create a new log entry with default values
    pub fn new(
        request_type: RequestType,
        method: impl Into<String>,
        path: impl Into<String>,
    ) -> Self {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            id: 0, // Will be set by logger
            timestamp_ms: now,
            request_type,
            method: method.into(),
            path: path.into(),
            client_ip: None,
            status_code: None,
            status: RequestStatus::Success,
            duration_ms: 0,
            request_body: None,
            response_body: None,
            error: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Set the client IP address
    pub fn with_client_ip(mut self, ip: IpAddr) -> Self {
        self.client_ip = Some(ip.to_string());
        self
    }

    /// Set the HTTP status code
    pub fn with_status_code(mut self, code: u16) -> Self {
        self.status_code = Some(code);
        if code >= 400 {
            self.status = if code == 401 || code == 403 || code == 429 {
                RequestStatus::Rejected
            } else {
                RequestStatus::Error
            };
        }
        self
    }

    /// Set the request status
    pub fn with_status(mut self, status: RequestStatus) -> Self {
        self.status = status;
        self
    }

    /// Set the duration
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration_ms = duration.as_millis() as u64;
        self
    }

    /// Set the error message
    pub fn with_error(mut self, error: impl Into<String>) -> Self {
        self.error = Some(error.into());
        self.status = RequestStatus::Error;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Filter for querying log entries
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LogFilter {
    /// Filter by request type
    pub request_type: Option<RequestType>,
    /// Filter by status
    pub status: Option<RequestStatus>,
    /// Filter by path prefix
    pub path_prefix: Option<String>,
    /// Filter by client IP
    pub client_ip: Option<String>,
    /// Filter by minimum timestamp (Unix timestamp ms)
    pub since_ms: Option<u64>,
    /// Filter by maximum timestamp (Unix timestamp ms)
    pub until_ms: Option<u64>,
    /// Maximum number of entries to return
    pub limit: Option<usize>,
    /// Offset for pagination
    pub offset: Option<usize>,
}

/// Statistics about request logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogStats {
    /// Total requests logged since startup
    pub total_logged: u64,
    /// Current entries in buffer
    pub entries_in_buffer: usize,
    /// Buffer capacity
    pub buffer_capacity: usize,
    /// Count by status
    pub success_count: u64,
    pub error_count: u64,
    pub rejected_count: u64,
    /// Average response time in milliseconds
    pub avg_duration_ms: f64,
    /// Whether logging is enabled
    pub enabled: bool,
}

/// Internal state for the request logger
struct LoggerState {
    /// Configuration
    config: LogConfig,
    /// Ring buffer of log entries
    entries: VecDeque<RequestLogEntry>,
    /// Next entry ID
    next_id: u64,
    /// Statistics
    total_logged: u64,
    success_count: u64,
    error_count: u64,
    rejected_count: u64,
    total_duration_ms: u64,
}

/// Shared request logger type
pub type SharedRequestLogger = Arc<RequestLogger>;

/// Request logger with thread-safe access
pub struct RequestLogger {
    state: RwLock<LoggerState>,
}

impl RequestLogger {
    /// Create a new request logger with the given configuration
    pub fn new(config: LogConfig) -> SharedRequestLogger {
        let state = LoggerState {
            config: config.clone(),
            entries: VecDeque::with_capacity(config.max_entries),
            next_id: 1,
            total_logged: 0,
            success_count: 0,
            error_count: 0,
            rejected_count: 0,
            total_duration_ms: 0,
        };

        Arc::new(Self {
            state: RwLock::new(state),
        })
    }

    /// Log a request entry
    pub async fn log(&self, mut entry: RequestLogEntry) {
        let mut state = self.state.write().await;

        if !state.config.enabled {
            return;
        }

        // Assign ID
        entry.id = state.next_id;
        state.next_id += 1;

        // Truncate bodies if needed
        if let Some(ref mut body) = entry.request_body {
            if body.len() > state.config.max_body_size {
                body.truncate(state.config.max_body_size);
                body.push_str("...[truncated]");
            }
        }
        if let Some(ref mut body) = entry.response_body {
            if body.len() > state.config.max_body_size {
                body.truncate(state.config.max_body_size);
                body.push_str("...[truncated]");
            }
        }

        // Update statistics
        state.total_logged += 1;
        state.total_duration_ms += entry.duration_ms;
        match entry.status {
            RequestStatus::Success => state.success_count += 1,
            RequestStatus::Error => state.error_count += 1,
            RequestStatus::Rejected => state.rejected_count += 1,
        }

        // Add to ring buffer (remove oldest if full)
        if state.entries.len() >= state.config.max_entries {
            state.entries.pop_front();
        }
        state.entries.push_back(entry);
    }

    /// Query log entries with a filter
    pub async fn query(&self, filter: &LogFilter) -> Vec<RequestLogEntry> {
        let state = self.state.read().await;

        let filtered = state.entries.iter().filter(|entry| {
            // Filter by request type
            if let Some(rt) = filter.request_type {
                if entry.request_type != rt {
                    return false;
                }
            }

            // Filter by status
            if let Some(status) = filter.status {
                if entry.status != status {
                    return false;
                }
            }

            // Filter by path prefix
            if let Some(ref prefix) = filter.path_prefix {
                if !entry.path.starts_with(prefix) {
                    return false;
                }
            }

            // Filter by client IP
            if let Some(ref ip) = filter.client_ip {
                if entry.client_ip.as_ref() != Some(ip) {
                    return false;
                }
            }

            // Filter by timestamp range
            if let Some(since) = filter.since_ms {
                if entry.timestamp_ms < since {
                    return false;
                }
            }
            if let Some(until) = filter.until_ms {
                if entry.timestamp_ms > until {
                    return false;
                }
            }

            true
        });

        // Apply offset and limit
        let offset = filter.offset.unwrap_or(0);
        let limit = filter.limit.unwrap_or(usize::MAX);

        filtered.skip(offset).take(limit).cloned().collect()
    }

    /// Get recent log entries (most recent first)
    pub async fn recent(&self, count: usize) -> Vec<RequestLogEntry> {
        let state = self.state.read().await;
        state.entries.iter().rev().take(count).cloned().collect()
    }

    /// Get logging statistics
    pub async fn stats(&self) -> LogStats {
        let state = self.state.read().await;

        let avg_duration = if state.total_logged > 0 {
            state.total_duration_ms as f64 / state.total_logged as f64
        } else {
            0.0
        };

        LogStats {
            total_logged: state.total_logged,
            entries_in_buffer: state.entries.len(),
            buffer_capacity: state.config.max_entries,
            success_count: state.success_count,
            error_count: state.error_count,
            rejected_count: state.rejected_count,
            avg_duration_ms: avg_duration,
            enabled: state.config.enabled,
        }
    }

    /// Get current configuration
    pub async fn config(&self) -> LogConfig {
        self.state.read().await.config.clone()
    }

    /// Update configuration
    ///
    /// If max_entries is reduced, oldest entries are removed to fit.
    pub async fn update_config(&self, config: LogConfig) {
        let mut state = self.state.write().await;

        // If max_entries is reduced, trim the buffer
        while state.entries.len() > config.max_entries {
            state.entries.pop_front();
        }

        state.config = config;
    }

    /// Clear all log entries
    pub async fn clear(&self) {
        let mut state = self.state.write().await;
        state.entries.clear();
        // Note: We keep statistics (total_logged, etc.) as they represent all-time counts
    }

    /// Export all entries as JSON
    pub async fn export_json(&self) -> Result<String, serde_json::Error> {
        let state = self.state.read().await;
        let entries: Vec<_> = state.entries.iter().collect();
        serde_json::to_string_pretty(&entries)
    }
}

/// Builder for creating log entries during request processing
pub struct RequestLogBuilder {
    entry: RequestLogEntry,
    start: Instant,
    config: LogConfig,
}

impl RequestLogBuilder {
    /// Create a new builder for an HTTP request
    pub fn http(method: impl Into<String>, path: impl Into<String>, config: LogConfig) -> Self {
        Self {
            entry: RequestLogEntry::new(RequestType::Http, method, path),
            start: Instant::now(),
            config,
        }
    }

    /// Create a new builder for a WebSocket message
    pub fn websocket(message_type: impl Into<String>, config: LogConfig) -> Self {
        Self {
            entry: RequestLogEntry::new(RequestType::WebSocket, message_type, "ws"),
            start: Instant::now(),
            config,
        }
    }

    /// Create a new builder for a JSON-RPC call
    pub fn jsonrpc(method: impl Into<String>, config: LogConfig) -> Self {
        let method_str = method.into();
        Self {
            entry: RequestLogEntry::new(RequestType::JsonRpc, method_str.clone(), method_str),
            start: Instant::now(),
            config,
        }
    }

    /// Set the client IP
    pub fn client_ip(mut self, ip: IpAddr) -> Self {
        self.entry = self.entry.with_client_ip(ip);
        self
    }

    /// Set request body (if configured)
    pub fn request_body(mut self, body: impl Into<String>) -> Self {
        if self.config.log_request_body {
            self.entry.request_body = Some(body.into());
        }
        self
    }

    /// Add metadata
    pub fn metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.entry = self.entry.with_metadata(key, value);
        self
    }

    /// Complete the log entry with success
    pub fn success(mut self, status_code: u16) -> RequestLogEntry {
        self.entry.duration_ms = self.start.elapsed().as_millis() as u64;
        self.entry.status_code = Some(status_code);
        self.entry.status = RequestStatus::Success;
        self.entry
    }

    /// Complete the log entry with error
    pub fn error(mut self, status_code: u16, error: impl Into<String>) -> RequestLogEntry {
        self.entry.duration_ms = self.start.elapsed().as_millis() as u64;
        self.entry.status_code = Some(status_code);
        self.entry.error = Some(error.into());
        self.entry.status = RequestStatus::Error;
        self.entry
    }

    /// Complete the log entry as rejected
    pub fn rejected(mut self, status_code: u16, reason: impl Into<String>) -> RequestLogEntry {
        self.entry.duration_ms = self.start.elapsed().as_millis() as u64;
        self.entry.status_code = Some(status_code);
        self.entry.error = Some(reason.into());
        self.entry.status = RequestStatus::Rejected;
        self.entry
    }

    /// Set response body (if configured) and complete
    pub fn with_response(mut self, status_code: u16, body: impl Into<String>) -> RequestLogEntry {
        if self.config.log_response_body {
            self.entry.response_body = Some(body.into());
        }
        self.success(status_code)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_basic_logging() {
        let logger = RequestLogger::new(LogConfig::default());

        let entry = RequestLogEntry::new(RequestType::Http, "POST", "/verify")
            .with_status_code(200)
            .with_duration(Duration::from_millis(50));

        logger.log(entry).await;

        let stats = logger.stats().await;
        assert_eq!(stats.total_logged, 1);
        assert_eq!(stats.success_count, 1);
        assert_eq!(stats.entries_in_buffer, 1);
    }

    #[tokio::test]
    async fn test_ring_buffer_overflow() {
        let config = LogConfig {
            max_entries: 5,
            ..Default::default()
        };
        let logger = RequestLogger::new(config);

        // Log 10 entries
        for i in 0..10 {
            let entry = RequestLogEntry::new(RequestType::Http, "GET", format!("/path/{}", i));
            logger.log(entry).await;
        }

        let stats = logger.stats().await;
        assert_eq!(stats.total_logged, 10);
        assert_eq!(stats.entries_in_buffer, 5);

        // Should have entries 5-9 (oldest removed)
        let entries = logger.recent(10).await;
        assert_eq!(entries.len(), 5);
        assert!(entries[0].path.ends_with("/9")); // Most recent
        assert!(entries[4].path.ends_with("/5")); // Oldest remaining
    }

    #[tokio::test]
    async fn test_query_filter() {
        let logger = RequestLogger::new(LogConfig::default());

        // Log mixed entries
        logger
            .log(RequestLogEntry::new(RequestType::Http, "GET", "/health").with_status_code(200))
            .await;
        logger
            .log(RequestLogEntry::new(RequestType::Http, "POST", "/verify").with_status_code(200))
            .await;
        logger
            .log(
                RequestLogEntry::new(RequestType::Http, "POST", "/verify")
                    .with_status_code(500)
                    .with_error("Internal error"),
            )
            .await;

        // Query errors only
        let errors = logger
            .query(&LogFilter {
                status: Some(RequestStatus::Error),
                ..Default::default()
            })
            .await;
        assert_eq!(errors.len(), 1);

        // Query by path prefix
        let verify_requests = logger
            .query(&LogFilter {
                path_prefix: Some("/verify".to_string()),
                ..Default::default()
            })
            .await;
        assert_eq!(verify_requests.len(), 2);
    }

    #[tokio::test]
    async fn test_status_codes() {
        let entry_200 = RequestLogEntry::new(RequestType::Http, "GET", "/").with_status_code(200);
        assert_eq!(entry_200.status, RequestStatus::Success);

        let entry_401 = RequestLogEntry::new(RequestType::Http, "GET", "/").with_status_code(401);
        assert_eq!(entry_401.status, RequestStatus::Rejected);

        let entry_429 = RequestLogEntry::new(RequestType::Http, "GET", "/").with_status_code(429);
        assert_eq!(entry_429.status, RequestStatus::Rejected);

        let entry_500 = RequestLogEntry::new(RequestType::Http, "GET", "/").with_status_code(500);
        assert_eq!(entry_500.status, RequestStatus::Error);
    }

    #[tokio::test]
    async fn test_builder() {
        let config = LogConfig {
            log_request_body: true,
            ..Default::default()
        };

        let entry = RequestLogBuilder::http("POST", "/verify", config)
            .client_ip("127.0.0.1".parse().unwrap())
            .request_body(r#"{"spec": "test"}"#)
            .metadata("request_id", "abc123")
            .success(200);

        assert_eq!(entry.method, "POST");
        assert_eq!(entry.path, "/verify");
        assert_eq!(entry.client_ip, Some("127.0.0.1".to_string()));
        assert!(entry.request_body.is_some());
        assert_eq!(
            entry.metadata.get("request_id"),
            Some(&"abc123".to_string())
        );
        assert_eq!(entry.status, RequestStatus::Success);
    }

    #[tokio::test]
    async fn test_disabled_logging() {
        let config = LogConfig {
            enabled: false,
            ..Default::default()
        };
        let logger = RequestLogger::new(config);

        logger
            .log(RequestLogEntry::new(RequestType::Http, "GET", "/"))
            .await;

        let stats = logger.stats().await;
        assert_eq!(stats.total_logged, 0);
        assert_eq!(stats.entries_in_buffer, 0);
    }

    #[tokio::test]
    async fn test_export_json() {
        let logger = RequestLogger::new(LogConfig::default());

        logger
            .log(RequestLogEntry::new(RequestType::Http, "GET", "/health"))
            .await;

        let json = logger.export_json().await.unwrap();
        assert!(json.contains("/health"));
        assert!(json.contains("http"));
    }

    #[tokio::test]
    async fn test_clear() {
        let logger = RequestLogger::new(LogConfig::default());

        logger
            .log(RequestLogEntry::new(RequestType::Http, "GET", "/"))
            .await;

        logger.clear().await;

        let stats = logger.stats().await;
        assert_eq!(stats.entries_in_buffer, 0);
        assert_eq!(stats.total_logged, 1); // Total count preserved
    }
}
