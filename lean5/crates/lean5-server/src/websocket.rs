//! WebSocket transport for Lean5 JSON-RPC server
//!
//! Provides WebSocket connectivity with streaming support for progress notifications.
//!
//! # Protocol
//!
//! Messages are JSON-RPC 2.0 over WebSocket text frames.
//!
//! # Streaming Notifications
//!
//! For long-running operations, the server sends progress notifications:
//!
//! ```json
//! {"jsonrpc": "2.0", "method": "progress", "params": {"requestId": 1, "message": "Checking..."}}
//! ```
//!
//! The final result is sent as a normal JSON-RPC response.

use crate::handlers::{
    handle_batch_check, handle_batch_verify_cert, handle_check, handle_get_config,
    handle_get_metrics, handle_get_type, handle_prove, handle_server_info, handle_verify_c,
    handle_verify_cert, ServerState,
};
use crate::progress::{ProgressSender, ProgressUpdate};
use crate::rpc::{parse_message, BatchResponse, ParsedMessage, RequestId, Response, RpcError};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Map;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, Semaphore};
use tokio_tungstenite::{accept_async, tungstenite::Message};
use tracing::{debug, error, info, instrument, warn};

/// Progress notification sent during long-running operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressNotification {
    /// JSON-RPC version
    pub jsonrpc: String,
    /// Method name (always "progress")
    pub method: String,
    /// Progress parameters
    pub params: ProgressParams,
}

/// Progress notification parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressParams {
    /// ID of the request this progress relates to
    #[serde(rename = "requestId")]
    pub request_id: RequestId,
    /// Progress message
    pub message: String,
    /// Progress percentage (0-100, if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub percentage: Option<u8>,
    /// Additional details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

impl ProgressNotification {
    /// Create a new progress notification
    pub fn new(request_id: RequestId, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            method: "progress".to_string(),
            params: ProgressParams {
                request_id,
                message: message.into(),
                percentage: None,
                details: None,
            },
        }
    }

    /// Create a progress notification with percentage
    pub fn with_percentage(
        request_id: RequestId,
        message: impl Into<String>,
        percentage: u8,
    ) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            method: "progress".to_string(),
            params: ProgressParams {
                request_id,
                message: message.into(),
                percentage: Some(percentage.min(100)),
                details: None,
            },
        }
    }
}

impl From<ProgressUpdate> for ProgressNotification {
    fn from(update: ProgressUpdate) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            method: "progress".to_string(),
            params: ProgressParams {
                request_id: update.request_id,
                message: update.message,
                percentage: update.percentage,
                details: update.details,
            },
        }
    }
}

/// WebSocket server configuration
#[derive(Clone, Debug)]
pub struct WebSocketConfig {
    /// Address to bind to
    pub addr: SocketAddr,
    /// Enable GPU acceleration
    pub gpu_enabled: bool,
    /// Maximum concurrent connections
    pub max_concurrent: usize,
    /// Default timeout for operations (milliseconds)
    pub default_timeout_ms: u64,
    /// Number of worker threads for batch operations (0 = auto/Rayon default)
    pub worker_threads: usize,
}

impl Default for WebSocketConfig {
    fn default() -> Self {
        Self {
            addr: "127.0.0.1:8081".parse().expect("valid default address"),
            gpu_enabled: false,
            max_concurrent: 100,
            default_timeout_ms: 5000,
            worker_threads: 0,
        }
    }
}

impl WebSocketConfig {
    /// Create a new WebSocket config with the specified address
    #[must_use]
    pub fn with_addr(mut self, addr: SocketAddr) -> Self {
        self.addr = addr;
        self
    }

    /// Enable or disable GPU acceleration
    #[must_use]
    pub fn with_gpu(mut self, enabled: bool) -> Self {
        self.gpu_enabled = enabled;
        self
    }

    /// Set maximum concurrent connections
    #[must_use]
    pub fn with_max_concurrent(mut self, max: usize) -> Self {
        self.max_concurrent = max;
        self
    }

    /// Set number of worker threads for batch operations
    ///
    /// 0 = auto (Rayon default, typically num_cpus)
    #[must_use]
    pub fn with_worker_threads(mut self, threads: usize) -> Self {
        self.worker_threads = threads;
        self
    }
}

/// WebSocket server handle
pub struct WebSocketHandle {
    /// Local address the server is bound to
    pub local_addr: SocketAddr,
    /// Shutdown signal sender
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
}

impl WebSocketHandle {
    /// Get the local address
    #[must_use]
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Shutdown the server
    pub fn shutdown(mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
    }
}

/// Start the WebSocket server
#[instrument(skip(config))]
pub async fn serve_websocket(config: WebSocketConfig) -> Result<WebSocketHandle, WebSocketError> {
    let listener = TcpListener::bind(config.addr)
        .await
        .map_err(|e| WebSocketError::Bind(e.to_string()))?;

    let local_addr = listener.local_addr()?;
    info!("Lean5 WebSocket server listening on {}", local_addr);

    let state = Arc::new(ServerState {
        env: Arc::new(tokio::sync::RwLock::new(lean5_kernel::Environment::new())),
        default_timeout_ms: config.default_timeout_ms,
        gpu_enabled: config.gpu_enabled,
        worker_threads: config.worker_threads,
        metrics: crate::handlers::ServerMetrics::new(),
    });

    let semaphore = Arc::new(Semaphore::new(config.max_concurrent));
    let (shutdown_tx, mut shutdown_rx) = tokio::sync::oneshot::channel::<()>();

    tokio::spawn(async move {
        loop {
            tokio::select! {
                accept_result = listener.accept() => {
                    match accept_result {
                        Ok((stream, peer_addr)) => {
                            debug!("New WebSocket connection from {}", peer_addr);
                            let state = Arc::clone(&state);
                            let permit = semaphore.clone().acquire_owned().await;

                            tokio::spawn(async move {
                                let _permit = permit;
                                if let Err(e) = handle_websocket_connection(stream, state).await {
                                    warn!("WebSocket error from {}: {}", peer_addr, e);
                                }
                                debug!("WebSocket connection closed: {}", peer_addr);
                            });
                        }
                        Err(e) => {
                            error!("Accept error: {}", e);
                        }
                    }
                }
                _ = &mut shutdown_rx => {
                    info!("WebSocket server shutting down");
                    break;
                }
            }
        }
    });

    Ok(WebSocketHandle {
        local_addr,
        shutdown_tx: Some(shutdown_tx),
    })
}

/// Handle a single WebSocket connection
#[instrument(skip(stream, state))]
async fn handle_websocket_connection(
    stream: TcpStream,
    state: Arc<ServerState>,
) -> Result<(), WebSocketError> {
    let ws_stream = accept_async(stream)
        .await
        .map_err(|e| WebSocketError::Handshake(e.to_string()))?;

    let (mut ws_sender, mut ws_receiver) = ws_stream.split();

    // Channel for sending responses back to the WebSocket
    let (tx, mut rx) = mpsc::channel::<String>(32);

    // Spawn a task to send responses
    let sender_task = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            if ws_sender.send(Message::Text(msg)).await.is_err() {
                break;
            }
        }
        ws_sender
    });

    // Process incoming messages
    while let Some(msg) = ws_receiver.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                let tx = tx.clone();
                let state = Arc::clone(&state);

                // Process message and send response(s)
                process_websocket_message(&text, &state, tx).await;
            }
            Ok(Message::Close(_)) => {
                debug!("Client requested close");
                break;
            }
            Ok(Message::Ping(data)) => {
                // Pong is handled automatically by tungstenite
                debug!("Received ping: {:?}", data);
            }
            Ok(Message::Pong(_)) => {
                debug!("Received pong");
            }
            Ok(Message::Binary(_)) => {
                warn!("Received binary message, ignoring");
            }
            Ok(Message::Frame(_)) => {
                // Internal frame, ignore
            }
            Err(e) => {
                error!("WebSocket error: {}", e);
                break;
            }
        }
    }

    // Clean up sender task
    drop(tx);
    let _ = sender_task.await;

    Ok(())
}

/// Process a WebSocket message and send response(s)
async fn process_websocket_message(json: &str, state: &ServerState, tx: mpsc::Sender<String>) {
    match parse_message(json) {
        Ok(ParsedMessage::Single(request)) => {
            if request.is_notification() {
                // Process notification but don't respond
                let _ = dispatch_request_single(
                    &request.method,
                    request.params,
                    RequestId::Null,
                    state,
                    None,
                )
                .await;
            } else {
                let id = request.id.clone().unwrap_or(RequestId::Null);
                dispatch_request_ws(&request.method, request.params, id, state, tx).await;
            }
        }
        Ok(ParsedMessage::Batch(batch)) => {
            let mut responses = Vec::with_capacity(batch.0.len());
            for request in batch.0 {
                if !request.is_notification() {
                    let id = request.id.clone().unwrap_or(RequestId::Null);
                    let response =
                        dispatch_request_single(&request.method, request.params, id, state, None)
                            .await;
                    responses.push(response);
                }
            }
            if !responses.is_empty() {
                let batch_response = BatchResponse(responses);
                if let Ok(json) = serde_json::to_string(&batch_response) {
                    let _ = tx.send(json).await;
                }
            }
        }
        Err(e) => {
            let response = Response::error(RequestId::Null, e);
            if let Ok(json) = serde_json::to_string(&response) {
                let _ = tx.send(json).await;
            }
        }
    }
}

/// Dispatch a request with progress notifications
async fn dispatch_request_ws(
    method: &str,
    params: Option<serde_json::Value>,
    id: RequestId,
    state: &ServerState,
    tx: mpsc::Sender<String>,
) {
    let wants_progress = matches!(
        method,
        "prove" | "batchCheck" | "batchVerifyCert" | "verifyC"
    );
    let can_stream_progress = wants_progress && !matches!(id, RequestId::Null);

    let (progress_sender, mut progress_task) = if can_stream_progress {
        let (progress_tx, mut progress_rx) = mpsc::channel::<ProgressUpdate>(32);
        let forward_tx = tx.clone();
        let handle = tokio::spawn(async move {
            while let Some(update) = progress_rx.recv().await {
                let notification: ProgressNotification = update.into();
                if let Ok(json) = serde_json::to_string(&notification) {
                    if forward_tx.send(json).await.is_err() {
                        break;
                    }
                }
            }
        });
        (
            Some(ProgressSender::new(id.clone(), progress_tx)),
            Some(handle),
        )
    } else {
        (None, None)
    };

    if can_stream_progress {
        if let Some(progress) = progress_sender.as_ref() {
            progress.notify("Starting...", Some(0), None).await;
        }
    }

    // Execute the actual handler
    let response =
        dispatch_request_single(method, params, id, state, progress_sender.clone()).await;

    drop(progress_sender);
    if let Some(task) = progress_task.take() {
        let _ = task.await;
    }

    // Send the response after all progress messages have been forwarded
    if let Ok(json) = serde_json::to_string(&response) {
        let _ = tx.send(json).await;
    }
}

/// Dispatch a single request (optional progress)
async fn dispatch_request_single(
    method: &str,
    params: Option<serde_json::Value>,
    id: RequestId,
    state: &ServerState,
    progress: Option<ProgressSender>,
) -> Response {
    match method {
        "check" => match parse_params::<crate::handlers::CheckParams>(params) {
            Ok(p) => handle_check(state, id, p).await,
            Err(e) => Response::error(id, e),
        },
        "prove" => match parse_params::<crate::handlers::ProveParams>(params) {
            Ok(p) => handle_prove(state, id, p).await,
            Err(e) => Response::error(id, e),
        },
        "getType" => match parse_params::<crate::handlers::GetTypeParams>(params) {
            Ok(p) => handle_get_type(state, id, p).await,
            Err(e) => Response::error(id, e),
        },
        "batchCheck" => match parse_params::<crate::handlers::BatchCheckParams>(params) {
            Ok(p) => handle_batch_check(state, id, p, progress).await,
            Err(e) => Response::error(id, e),
        },
        "verifyCert" => match parse_params::<crate::handlers::VerifyCertParams>(params) {
            Ok(p) => handle_verify_cert(state, id, p).await,
            Err(e) => Response::error(id, e),
        },
        "batchVerifyCert" => match parse_params::<crate::handlers::BatchVerifyCertParams>(params) {
            Ok(p) => handle_batch_verify_cert(state, id, p, progress).await,
            Err(e) => Response::error(id, e),
        },
        "verifyC" => match parse_params::<crate::handlers::VerifyCParams>(params) {
            Ok(p) => handle_verify_c(state, id, p, progress).await,
            Err(e) => Response::error(id, e),
        },
        "serverInfo" => handle_server_info(state, id).await,
        "getConfig" => handle_get_config(state, id).await,
        "getMetrics" => handle_get_metrics(state, id).await,
        _ => Response::error(id, RpcError::method_not_found(method)),
    }
}

/// Parse request parameters
fn parse_params<T: serde::de::DeserializeOwned>(
    params: Option<serde_json::Value>,
) -> Result<T, RpcError> {
    match params {
        Some(v) => serde_json::from_value(v)
            .map_err(|e| RpcError::invalid_params(format!("Invalid parameters: {e}"))),
        None => serde_json::from_value(serde_json::Value::Object(Map::default()))
            .map_err(|e| RpcError::invalid_params(format!("Missing required parameters: {e}"))),
    }
}

#[derive(Debug, thiserror::Error)]
pub enum WebSocketError {
    #[error("Failed to bind to address: {0}")]
    Bind(String),
    #[error("WebSocket handshake failed: {0}")]
    Handshake(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::connect_async;

    async fn connect_and_send(addr: SocketAddr, request: &str) -> Vec<String> {
        let url = format!("ws://{addr}");
        let (mut ws_stream, _) = connect_async(&url).await.unwrap();

        ws_stream
            .send(Message::Text(request.to_string()))
            .await
            .unwrap();

        let mut responses = Vec::new();
        // Give server time to respond
        let timeout = tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while let Some(msg) = ws_stream.next().await {
                match msg {
                    Ok(Message::Text(text)) => {
                        responses.push(text);
                        // For simple requests, one response is enough
                        if !responses.is_empty() {
                            break;
                        }
                    }
                    Ok(Message::Close(_)) => break,
                    _ => {}
                }
            }
        });
        let _ = timeout.await;

        ws_stream.close(None).await.ok();
        responses
    }

    #[tokio::test]
    async fn test_websocket_server_check() {
        let config = WebSocketConfig::default().with_addr("127.0.0.1:0".parse().unwrap());
        let handle = serve_websocket(config).await.unwrap();
        let addr = handle.local_addr();

        // Use fully-typed expression
        let request = r#"{"jsonrpc": "2.0", "method": "check", "params": {"code": "fun (A : Type) (x : A) => x"}, "id": 1}"#;
        let responses = connect_and_send(addr, request).await;

        assert!(
            !responses.is_empty(),
            "Should receive at least one response"
        );
        let response = &responses[0];
        assert!(response.contains("\"result\""), "Response: {response}");
        assert!(response.contains("\"valid\":true"), "Response: {response}");

        handle.shutdown();
    }

    #[tokio::test]
    async fn test_websocket_server_info() {
        let config = WebSocketConfig::default().with_addr("127.0.0.1:0".parse().unwrap());
        let handle = serve_websocket(config).await.unwrap();
        let addr = handle.local_addr();

        let request = r#"{"jsonrpc": "2.0", "method": "serverInfo", "id": 1}"#;
        let responses = connect_and_send(addr, request).await;

        assert!(!responses.is_empty());
        assert!(responses[0].contains("lean5-server"));
        assert!(responses[0].contains("\"methods\""));

        handle.shutdown();
    }

    #[tokio::test]
    async fn test_websocket_method_not_found() {
        let config = WebSocketConfig::default().with_addr("127.0.0.1:0".parse().unwrap());
        let handle = serve_websocket(config).await.unwrap();
        let addr = handle.local_addr();

        let request = r#"{"jsonrpc": "2.0", "method": "unknownMethod", "id": 1}"#;
        let responses = connect_and_send(addr, request).await;

        assert!(!responses.is_empty());
        assert!(responses[0].contains("\"error\""));
        assert!(responses[0].contains("-32601")); // METHOD_NOT_FOUND

        handle.shutdown();
    }

    #[tokio::test]
    async fn test_websocket_batch_progress_streaming() {
        let config = WebSocketConfig::default().with_addr("127.0.0.1:0".parse().unwrap());
        let handle = serve_websocket(config).await.unwrap();
        let addr = handle.local_addr();

        let url = format!("ws://{addr}");
        let (mut ws_stream, _) = connect_async(&url).await.unwrap();

        let request = r#"{"jsonrpc": "2.0", "method": "batchCheck", "params": {"items": [{"id": "1", "code": "fun (A : Type) (x : A) => x"}, {"id": "2", "code": "Type"}]}, "id": 42}"#;
        ws_stream
            .send(Message::Text(request.to_string()))
            .await
            .unwrap();

        let mut saw_progress = false;
        let mut saw_result = false;
        let mut seen = Vec::new();

        let timeout = tokio::time::timeout(std::time::Duration::from_secs(3), async {
            while let Some(msg) = ws_stream.next().await {
                match msg {
                    Ok(Message::Text(text)) => {
                        if text.contains("\"method\":\"progress\"") {
                            saw_progress = true;
                        }
                        if text.contains("\"result\"") && text.contains("\"results\"") {
                            saw_result = true;
                            seen.push(text);
                            break;
                        }
                        seen.push(text);
                    }
                    Ok(Message::Close(_)) => break,
                    _ => {}
                }
            }
        });
        let _ = timeout.await;

        ws_stream.close(None).await.ok();

        assert!(saw_progress, "Expected progress notification, got {seen:?}");
        assert!(saw_result, "Expected final response, got {seen:?}");

        handle.shutdown();
    }

    #[tokio::test]
    async fn test_websocket_invalid_json() {
        let config = WebSocketConfig::default().with_addr("127.0.0.1:0".parse().unwrap());
        let handle = serve_websocket(config).await.unwrap();
        let addr = handle.local_addr();

        let request = r"{invalid json";
        let responses = connect_and_send(addr, request).await;

        assert!(!responses.is_empty());
        assert!(responses[0].contains("\"error\""));
        assert!(responses[0].contains("-32700")); // PARSE_ERROR

        handle.shutdown();
    }

    #[tokio::test]
    async fn test_progress_notification_struct() {
        let progress = ProgressNotification::new(RequestId::Number(42), "Testing...");
        let json = serde_json::to_string(&progress).unwrap();

        assert!(json.contains("\"jsonrpc\":\"2.0\""));
        assert!(json.contains("\"method\":\"progress\""));
        assert!(json.contains("\"requestId\":42"));
        assert!(json.contains("\"message\":\"Testing...\""));
    }

    #[tokio::test]
    async fn test_progress_with_percentage() {
        let progress = ProgressNotification::with_percentage(
            RequestId::String("req-1".into()),
            "Processing",
            50,
        );
        let json = serde_json::to_string(&progress).unwrap();

        assert!(json.contains("\"percentage\":50"));
        assert!(json.contains("\"requestId\":\"req-1\""));
    }

    #[tokio::test]
    async fn test_websocket_batch_verify_cert_progress_streaming() {
        use lean5_kernel::{Expr, Level, ProofCert};

        let config = WebSocketConfig::default().with_addr("127.0.0.1:0".parse().unwrap());
        let handle = serve_websocket(config).await.unwrap();
        let addr = handle.local_addr();

        let url = format!("ws://{addr}");
        let (mut ws_stream, _) = connect_async(&url).await.unwrap();

        // Construct valid certificates for Sort(0) : Sort(1)
        let level = Level::zero();
        let expr = Expr::Sort(level.clone());
        let cert = ProofCert::Sort {
            level: level.clone(),
        };

        let item1 = crate::handlers::BatchVerifyCertItem {
            id: "cert1".to_string(),
            cert: cert.clone(),
            expr: expr.clone(),
        };
        let item2 = crate::handlers::BatchVerifyCertItem {
            id: "cert2".to_string(),
            cert,
            expr,
        };

        let params = crate::handlers::BatchVerifyCertParams {
            items: vec![item1, item2],
            threads: 0,
            timeout_ms: None,
        };
        let params_json = serde_json::to_string(&params).unwrap();
        let request = format!(
            r#"{{"jsonrpc": "2.0", "method": "batchVerifyCert", "params": {params_json}, "id": 42}}"#
        );

        ws_stream.send(Message::Text(request)).await.unwrap();

        let mut saw_progress = false;
        let mut saw_result = false;
        let mut seen = Vec::new();

        let timeout = tokio::time::timeout(std::time::Duration::from_secs(3), async {
            while let Some(msg) = ws_stream.next().await {
                match msg {
                    Ok(Message::Text(text)) => {
                        if text.contains("\"method\":\"progress\"") {
                            saw_progress = true;
                        }
                        if text.contains("\"result\"") && text.contains("\"results\"") {
                            saw_result = true;
                            seen.push(text);
                            break;
                        }
                        seen.push(text);
                    }
                    Ok(Message::Close(_)) => break,
                    _ => {}
                }
            }
        });
        let _ = timeout.await;

        ws_stream.close(None).await.ok();

        assert!(
            saw_progress,
            "Expected progress notification for batchVerifyCert, got {seen:?}"
        );
        assert!(
            saw_result,
            "Expected final response for batchVerifyCert, got {seen:?}"
        );

        handle.shutdown();
    }
}
