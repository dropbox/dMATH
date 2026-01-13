//! Lean5 JSON-RPC Server
//!
//! Provides an AI-native API for theorem proving:
//!
//! # Methods
//!
//! - `check`: Type check an expression or declaration
//! - `prove`: Attempt automatic proof via SMT
//! - `getType`: Get the type of an expression
//! - `batchCheck`: GPU-accelerated batch checking
//! - `batchVerifyCert`: Parallel batch certificate verification (high-throughput)
//! - `compressCert`: Structure-sharing compression for certificates (in-memory)
//! - `decompressCert`: Restore certificate from structure-sharing compression
//! - `archiveCert`: Byte-level compression (LZ4/Zstd) for storage/transmission
//! - `unarchiveCert`: Restore certificate from byte-level archive
//! - `trainDict`: Train a compression dictionary from sample certificates
//! - `archiveCertWithDict`: Dictionary-based Zstd compression for improved ratios
//! - `unarchiveCertWithDict`: Restore certificate from dictionary-compressed archive
//! - `verifyC`: Verify C code with ACSL specifications
//! - `serverInfo`: Get server capabilities
//!
//! # Transports
//!
//! - **TCP**: JSON-RPC 2.0 over line-delimited TCP (`serve`)
//! - **WebSocket**: JSON-RPC 2.0 over WebSocket with streaming progress (`websocket::serve_websocket`)
//!
//! # Streaming Progress
//!
//! WebSocket clients receive progress notifications during long-running operations:
//!
//! ```json
//! {"jsonrpc": "2.0", "method": "progress", "params": {"requestId": 1, "message": "Checking..."}}
//! ```
//!
//! # Design
//!
//! - JSON-RPC 2.0 protocol
//! - Connection pooling for warm starts
//! - Async/await with tokio runtime
//!
//! # Example
//!
//! ```ignore
//! use lean5_server::{ServerConfig, serve};
//!
//! #[tokio::main]
//! async fn main() {
//!     let config = ServerConfig::default();
//!     serve(config).await.unwrap();
//! }
//! ```

pub mod handlers;
pub mod progress;
pub mod rpc;
pub mod websocket;

use handlers::{
    handle_archive_cert, handle_archive_cert_with_dict, handle_batch_check,
    handle_batch_verify_cert, handle_check, handle_compress_cert, handle_decompress_cert,
    handle_get_config, handle_get_metrics, handle_get_type, handle_prove, handle_server_info,
    handle_train_dict, handle_unarchive_cert, handle_unarchive_cert_with_dict, handle_verify_c,
    handle_verify_cert, ServerState,
};
use rpc::{parse_message, BatchResponse, ParsedMessage, Response, RpcError};
use serde_json::Map;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Semaphore;
use tracing::{debug, error, info, instrument, warn};

/// Lean5 server configuration
#[derive(Clone, Debug)]
pub struct ServerConfig {
    /// Address to bind to
    pub addr: SocketAddr,
    /// Enable GPU acceleration
    pub gpu_enabled: bool,
    /// Maximum concurrent requests
    pub max_concurrent: usize,
    /// Default timeout for operations (milliseconds)
    pub default_timeout_ms: u64,
    /// Number of worker threads for batch operations (0 = auto/Rayon default)
    pub worker_threads: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            addr: "127.0.0.1:8080".parse().expect("valid default address"),
            gpu_enabled: false,
            max_concurrent: 100,
            default_timeout_ms: 5000,
            worker_threads: 0, // Auto (Rayon default = num_cpus)
        }
    }
}

impl ServerConfig {
    /// Create a new server config with the specified address
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

    /// Set maximum concurrent requests
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

/// Server handle for managing a running server
pub struct ServerHandle {
    /// Local address the server is bound to
    pub local_addr: SocketAddr,
    /// Shutdown signal sender
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
}

impl ServerHandle {
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

/// Start the Lean5 server
#[instrument(skip(config))]
pub async fn serve(config: ServerConfig) -> Result<ServerHandle, ServerError> {
    let listener = TcpListener::bind(config.addr)
        .await
        .map_err(|e| ServerError::Bind(e.to_string()))?;

    let local_addr = listener.local_addr()?;
    info!("Lean5 server listening on {}", local_addr);

    let state = Arc::new(ServerState {
        env: Arc::new(tokio::sync::RwLock::new(lean5_kernel::Environment::new())),
        default_timeout_ms: config.default_timeout_ms,
        gpu_enabled: config.gpu_enabled,
        worker_threads: config.worker_threads,
        metrics: handlers::ServerMetrics::new(),
    });

    let semaphore = Arc::new(Semaphore::new(config.max_concurrent));
    let (shutdown_tx, mut shutdown_rx) = tokio::sync::oneshot::channel::<()>();

    tokio::spawn(async move {
        loop {
            tokio::select! {
                accept_result = listener.accept() => {
                    match accept_result {
                        Ok((stream, peer_addr)) => {
                            debug!("New connection from {}", peer_addr);
                            let state = Arc::clone(&state);
                            let permit = semaphore.clone().acquire_owned().await;

                            tokio::spawn(async move {
                                let _permit = permit; // Hold permit for duration
                                if let Err(e) = handle_connection(stream, state).await {
                                    warn!("Connection error from {}: {}", peer_addr, e);
                                }
                                debug!("Connection closed: {}", peer_addr);
                            });
                        }
                        Err(e) => {
                            error!("Accept error: {}", e);
                        }
                    }
                }
                _ = &mut shutdown_rx => {
                    info!("Server shutting down");
                    break;
                }
            }
        }
    });

    Ok(ServerHandle {
        local_addr,
        shutdown_tx: Some(shutdown_tx),
    })
}

/// Handle a single connection
#[instrument(skip(stream, state))]
async fn handle_connection(stream: TcpStream, state: Arc<ServerState>) -> Result<(), ServerError> {
    let (reader, mut writer) = stream.into_split();
    let mut reader = BufReader::new(reader);
    let mut line = String::new();

    loop {
        line.clear();
        let bytes_read = reader.read_line(&mut line).await?;

        if bytes_read == 0 {
            // Connection closed
            break;
        }

        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        debug!("Received: {}", line);

        let response = process_message(line, &state).await;

        // Serialize and send response
        let response_json = match &response {
            MessageResponse::Single(r) => serde_json::to_string(r),
            MessageResponse::Batch(b) => serde_json::to_string(b),
            MessageResponse::None => continue, // Notification, no response
        };

        match response_json {
            Ok(json) => {
                debug!("Sending: {}", json);
                writer.write_all(json.as_bytes()).await?;
                writer.write_all(b"\n").await?;
                writer.flush().await?;
            }
            Err(e) => {
                error!("Failed to serialize response: {}", e);
            }
        }
    }

    Ok(())
}

/// Response type (single, batch, or none for notifications)
enum MessageResponse {
    Single(Response),
    Batch(BatchResponse),
    None,
}

/// Process a JSON-RPC message
async fn process_message(json: &str, state: &ServerState) -> MessageResponse {
    match parse_message(json) {
        Ok(ParsedMessage::Single(request)) => {
            if request.is_notification() {
                // Process notification but don't respond
                let _ = dispatch_request(
                    &request.method,
                    request.params.clone(),
                    rpc::RequestId::Null,
                    state,
                )
                .await;
                MessageResponse::None
            } else {
                let id = request.id.clone().unwrap_or(rpc::RequestId::Null);
                let response = dispatch_request(&request.method, request.params, id, state).await;
                MessageResponse::Single(response)
            }
        }
        Ok(ParsedMessage::Batch(batch)) => {
            let mut responses = Vec::with_capacity(batch.0.len());
            for request in batch.0 {
                if !request.is_notification() {
                    let id = request.id.clone().unwrap_or(rpc::RequestId::Null);
                    let response =
                        dispatch_request(&request.method, request.params, id, state).await;
                    responses.push(response);
                }
            }
            if responses.is_empty() {
                MessageResponse::None
            } else {
                MessageResponse::Batch(BatchResponse(responses))
            }
        }
        Err(e) => MessageResponse::Single(Response::error(rpc::RequestId::Null, e)),
    }
}

/// Dispatch a request to the appropriate handler
async fn dispatch_request(
    method: &str,
    params: Option<serde_json::Value>,
    id: rpc::RequestId,
    state: &ServerState,
) -> Response {
    match method {
        "check" => match parse_params::<handlers::CheckParams>(params) {
            Ok(p) => handle_check(state, id, p).await,
            Err(e) => Response::error(id, e),
        },
        "prove" => match parse_params::<handlers::ProveParams>(params) {
            Ok(p) => handle_prove(state, id, p).await,
            Err(e) => Response::error(id, e),
        },
        "getType" => match parse_params::<handlers::GetTypeParams>(params) {
            Ok(p) => handle_get_type(state, id, p).await,
            Err(e) => Response::error(id, e),
        },
        "batchCheck" => match parse_params::<handlers::BatchCheckParams>(params) {
            Ok(p) => handle_batch_check(state, id, p, None).await,
            Err(e) => Response::error(id, e),
        },
        "verifyCert" => match parse_params::<handlers::VerifyCertParams>(params) {
            Ok(p) => handle_verify_cert(state, id, p).await,
            Err(e) => Response::error(id, e),
        },
        "batchVerifyCert" => match parse_params::<handlers::BatchVerifyCertParams>(params) {
            Ok(p) => handle_batch_verify_cert(state, id, p, None).await,
            Err(e) => Response::error(id, e),
        },
        "compressCert" => match parse_params::<handlers::CompressCertParams>(params) {
            Ok(p) => handle_compress_cert(state, id, p).await,
            Err(e) => Response::error(id, e),
        },
        "decompressCert" => match parse_params::<handlers::DecompressCertParams>(params) {
            Ok(p) => handle_decompress_cert(state, id, p).await,
            Err(e) => Response::error(id, e),
        },
        "archiveCert" => match parse_params::<handlers::ArchiveCertParams>(params) {
            Ok(p) => handle_archive_cert(state, id, p).await,
            Err(e) => Response::error(id, e),
        },
        "unarchiveCert" => match parse_params::<handlers::UnarchiveCertParams>(params) {
            Ok(p) => handle_unarchive_cert(state, id, p).await,
            Err(e) => Response::error(id, e),
        },
        "trainDict" => match parse_params::<handlers::TrainDictParams>(params) {
            Ok(p) => handle_train_dict(state, id, p).await,
            Err(e) => Response::error(id, e),
        },
        "archiveCertWithDict" => {
            match parse_params::<handlers::ArchiveCertWithDictParams>(params) {
                Ok(p) => handle_archive_cert_with_dict(state, id, p).await,
                Err(e) => Response::error(id, e),
            }
        }
        "unarchiveCertWithDict" => {
            match parse_params::<handlers::UnarchiveCertWithDictParams>(params) {
                Ok(p) => handle_unarchive_cert_with_dict(state, id, p).await,
                Err(e) => Response::error(id, e),
            }
        }
        "serverInfo" => handle_server_info(state, id).await,
        "saveEnvironment" => match parse_params::<handlers::SaveEnvironmentParams>(params) {
            Ok(p) => handlers::handle_save_environment(state, id, p).await,
            Err(e) => Response::error(id, e),
        },
        "loadEnvironment" => match parse_params::<handlers::LoadEnvironmentParams>(params) {
            Ok(p) => handlers::handle_load_environment(state, id, p).await,
            Err(e) => Response::error(id, e),
        },
        "getEnvironment" => match parse_params::<handlers::GetEnvironmentParams>(params) {
            Ok(p) => handlers::handle_get_environment(state, id, p).await,
            Err(e) => Response::error(id, e),
        },
        "verifyC" => match parse_params::<handlers::VerifyCParams>(params) {
            Ok(p) => handle_verify_c(state, id, p, None).await,
            Err(e) => Response::error(id, e),
        },
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
pub enum ServerError {
    #[error("Failed to bind to address: {0}")]
    Bind(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

// Re-export key types for external consumers
// Note: ServerState is used internally via the handlers module import
// Note: Response and RpcError are re-exported through the rpc module
pub use handlers::{
    ArchiveCertParams, ArchiveCertResult, ArchiveCertWithDictParams, ArchiveCertWithDictResult,
    BatchCheckItemResult, BatchCheckParams, BatchCheckResult, BatchStats, BatchVerifyCertItem,
    BatchVerifyCertItemResult, BatchVerifyCertParams, BatchVerifyCertResult, BatchVerifyCertStats,
    CheckError, CheckParams, CheckResult, CompressCertParams, CompressCertResult,
    CompressCertStats, DecompressCertParams, DecompressCertResult, GetConfigParams,
    GetConfigResult, GetEnvironmentParams, GetEnvironmentResult, GetMetricsResult, GetTypeParams,
    GetTypeResult, LoadEnvironmentParams, LoadEnvironmentResult, MethodCounts, ProveParams,
    ProveResult, SaveEnvironmentParams, SaveEnvironmentResult, ServerInfo, ServerMetrics,
    TimingStats, TrainDictParams, TrainDictResult, UnarchiveCertParams, UnarchiveCertResult,
    UnarchiveCertWithDictParams, UnarchiveCertWithDictResult, VerifyCFunctionResult, VerifyCParams,
    VerifyCResult, VerifyCVCDetail, VerifyCertParams, VerifyCertResult,
};
pub use websocket::{
    serve_websocket, ProgressNotification, ProgressParams, WebSocketConfig, WebSocketError,
    WebSocketHandle,
};

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use tokio::net::TcpStream;

    async fn send_request(addr: SocketAddr, request: &str) -> String {
        let mut stream = TcpStream::connect(addr).await.unwrap();
        stream.write_all(request.as_bytes()).await.unwrap();
        stream.write_all(b"\n").await.unwrap();

        let mut reader = BufReader::new(stream);
        let mut response = String::new();
        reader.read_line(&mut response).await.unwrap();
        response
    }

    #[tokio::test]
    async fn test_server_check() {
        let config = ServerConfig::default().with_addr("127.0.0.1:0".parse().unwrap());
        let handle = serve(config).await.unwrap();
        let addr = handle.local_addr();

        // Use fully-typed expression
        let request = r#"{"jsonrpc": "2.0", "method": "check", "params": {"code": "fun (A : Type) (x : A) => x"}, "id": 1}"#;
        let response = send_request(addr, request).await;

        assert!(response.contains("\"result\""), "Response: {response}");
        assert!(response.contains("\"valid\":true"), "Response: {response}");

        handle.shutdown();
    }

    #[tokio::test]
    async fn test_server_info() {
        let config = ServerConfig::default().with_addr("127.0.0.1:0".parse().unwrap());
        let handle = serve(config).await.unwrap();
        let addr = handle.local_addr();

        let request = r#"{"jsonrpc": "2.0", "method": "serverInfo", "id": 1}"#;
        let response = send_request(addr, request).await;

        assert!(response.contains("lean5-server"));
        assert!(response.contains("\"methods\""));

        handle.shutdown();
    }

    #[tokio::test]
    async fn test_server_method_not_found() {
        let config = ServerConfig::default().with_addr("127.0.0.1:0".parse().unwrap());
        let handle = serve(config).await.unwrap();
        let addr = handle.local_addr();

        let request = r#"{"jsonrpc": "2.0", "method": "unknownMethod", "id": 1}"#;
        let response = send_request(addr, request).await;

        assert!(response.contains("\"error\""));
        assert!(response.contains("-32601")); // METHOD_NOT_FOUND

        handle.shutdown();
    }

    #[tokio::test]
    async fn test_server_batch_request() {
        let config = ServerConfig::default().with_addr("127.0.0.1:0".parse().unwrap());
        let handle = serve(config).await.unwrap();
        let addr = handle.local_addr();

        let request = r#"[{"jsonrpc": "2.0", "method": "serverInfo", "id": 1}, {"jsonrpc": "2.0", "method": "serverInfo", "id": 2}]"#;
        let response = send_request(addr, request).await;

        // Should be an array response
        assert!(response.starts_with('['));
        assert!(response.ends_with("]\n"));

        handle.shutdown();
    }

    #[tokio::test]
    async fn test_server_invalid_json() {
        let config = ServerConfig::default().with_addr("127.0.0.1:0".parse().unwrap());
        let handle = serve(config).await.unwrap();
        let addr = handle.local_addr();

        let request = r"{invalid json";
        let response = send_request(addr, request).await;

        assert!(response.contains("\"error\""));
        assert!(response.contains("-32700")); // PARSE_ERROR

        handle.shutdown();
    }

    #[tokio::test]
    async fn test_server_verify_c() {
        let config = ServerConfig::default().with_addr("127.0.0.1:0".parse().unwrap());
        let handle = serve(config).await.unwrap();
        let addr = handle.local_addr();

        let request = r#"{"jsonrpc": "2.0", "method": "verifyC", "params": {"code": "//@ requires n >= 0;\n//@ ensures \\result >= 0;\nint id(int n) { return n; }"}, "id": 1}"#;
        let response = send_request(addr, request).await;

        assert!(response.contains("\"result\""), "Response: {response}");
        assert!(
            response.contains("\"num_functions\":1"),
            "Response: {response}"
        );
        assert!(response.contains("\"id\""), "Response: {response}");

        handle.shutdown();
    }

    #[tokio::test]
    async fn test_server_batch_verify_cert() {
        use lean5_kernel::{Expr, Level, ProofCert};

        let config = ServerConfig::default().with_addr("127.0.0.1:0".parse().unwrap());
        let handle = serve(config).await.unwrap();
        let addr = handle.local_addr();

        // Construct a valid certificate for Sort(0) : Sort(1)
        let level = Level::zero();
        let expr = Expr::Sort(level.clone());
        let cert = ProofCert::Sort {
            level: level.clone(),
        };

        let item = handlers::BatchVerifyCertItem {
            id: "test1".to_string(),
            cert,
            expr,
        };

        // Serialize the request params
        let params = handlers::BatchVerifyCertParams {
            items: vec![item],
            threads: 0,
            timeout_ms: None,
        };
        let params_json = serde_json::to_string(&params).unwrap();
        let request = format!(
            r#"{{"jsonrpc": "2.0", "method": "batchVerifyCert", "params": {params_json}, "id": 1}}"#
        );

        let response = send_request(addr, &request).await;

        assert!(response.contains("\"result\""), "Response: {response}");
        assert!(
            response.contains("\"success\":true"),
            "Response: {response}"
        );
        assert!(response.contains("\"total\":1"), "Response: {response}");

        handle.shutdown();
    }
}
