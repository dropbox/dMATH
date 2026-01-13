//! DashProve MCP Server Binary
//!
//! Run with: `dashprove-mcp` or `dashprove-mcp --help`
//!
//! ## Transport modes
//!
//! - `stdio` (default): Standard input/output for MCP protocol communication
//! - `http`: HTTP server with JSON-RPC endpoint at POST /jsonrpc
//! - `websocket`: WebSocket server for bidirectional communication at GET /ws
//!
//! ## Cache persistence
//!
//! Use `--cache-file` to enable automatic cache persistence. The cache will be
//! loaded on startup (if the file exists) and saved on graceful shutdown.
//! Set `--cache-save-interval-secs` to periodically auto-save while running.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use clap::{Parser, ValueEnum};
use tokio::signal;
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

use dashprove_mcp::cache::{CacheAutoSaver, SharedVerificationCache, VerificationCache};
use dashprove_mcp::logging::LogConfig;
use dashprove_mcp::metrics::MetricsConfig;
use dashprove_mcp::ratelimit::RateLimitConfig;
use dashprove_mcp::server::McpServer;
use dashprove_mcp::tools::VerifyUslResult;
use dashprove_mcp::transport::{HttpTransport, StdioTransport, WebSocketTransport};

/// Default interval (in seconds) between automatic cache saves
const DEFAULT_CACHE_SAVE_INTERVAL_SECS: u64 = 300;

/// Default rate limit (requests per second)
const DEFAULT_RATE_LIMIT_RPS: f64 = 10.0;

/// Default burst size for rate limiting
const DEFAULT_RATE_LIMIT_BURST: u32 = 50;

/// Default maximum log entries to retain
const DEFAULT_LOG_MAX_ENTRIES: usize = 1000;

/// DashProve MCP Server - Model Context Protocol interface for AI agents
#[derive(Parser, Debug)]
#[command(name = "dashprove-mcp")]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Transport type
    #[arg(short, long, default_value = "stdio")]
    transport: TransportType,

    /// Bind address for HTTP transport (ignored for stdio)
    #[arg(long, default_value = "127.0.0.1:3001")]
    bind: String,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, default_value = "info")]
    log_level: String,

    /// Path to cache file for automatic persistence.
    /// If specified, cache will be loaded on startup and saved on shutdown.
    #[arg(long)]
    cache_file: Option<PathBuf>,

    /// Disable caching entirely (overrides --cache-file)
    #[arg(long, default_value = "false")]
    no_cache: bool,

    /// Interval (in seconds) between automatic cache saves (requires --cache-file).
    /// Set to 0 to disable periodic auto-save.
    #[arg(long, default_value_t = DEFAULT_CACHE_SAVE_INTERVAL_SECS)]
    cache_save_interval_secs: u64,

    /// API token for authentication (optional, HTTP/WebSocket transports only).
    /// If set, all requests must include this token in the Authorization header
    /// (Bearer token) or via query parameter (?token=...).
    #[arg(long, env = "DASHPROVE_API_TOKEN")]
    api_token: Option<String>,

    /// Enable rate limiting (HTTP/WebSocket transports only).
    /// Limits requests per IP address to prevent abuse.
    #[arg(long, default_value = "false")]
    rate_limit: bool,

    /// Requests per second limit per client IP (requires --rate-limit).
    #[arg(long, default_value_t = DEFAULT_RATE_LIMIT_RPS)]
    rate_limit_rps: f64,

    /// Burst size for rate limiting - maximum requests allowed in a burst.
    /// Higher values allow more bursty traffic patterns (requires --rate-limit).
    #[arg(long, default_value_t = DEFAULT_RATE_LIMIT_BURST)]
    rate_limit_burst: u32,

    /// Enable request logging (HTTP/WebSocket transports only).
    /// Logs all requests with timing, status, and client IP for auditing.
    #[arg(long, default_value = "false")]
    request_logging: bool,

    /// Maximum number of log entries to retain in memory (requires --request-logging).
    /// Older entries are evicted when this limit is reached.
    #[arg(long, default_value_t = DEFAULT_LOG_MAX_ENTRIES)]
    log_max_entries: usize,

    /// Log request bodies (requires --request-logging).
    /// Warning: May increase memory usage significantly for large requests.
    #[arg(long, default_value = "false")]
    log_request_body: bool,

    /// Log response bodies (requires --request-logging).
    /// Warning: May increase memory usage significantly for large responses.
    #[arg(long, default_value = "false")]
    log_response_body: bool,

    /// Enable Prometheus metrics endpoint (HTTP/WebSocket transports only).
    /// Exposes metrics at GET /metrics (Prometheus format) and GET /metrics/json (JSON format).
    #[arg(long, default_value = "false")]
    metrics: bool,
}

/// Transport type
#[derive(Debug, Clone, Copy, ValueEnum)]
enum TransportType {
    /// Standard input/output (default for MCP)
    Stdio,
    /// HTTP server with JSON-RPC endpoint
    Http,
    /// WebSocket server for bidirectional communication
    Websocket,
}

/// Load cache from file if it exists
async fn load_cache_if_exists(cache: &SharedVerificationCache<VerifyUslResult>, path: &PathBuf) {
    if path.exists() {
        info!("Loading cache from {:?}", path);
        match cache.load_from_file(path, false).await {
            Ok(result) => {
                info!(
                    "Loaded {} cache entries ({} expired), snapshot age: {}s",
                    result.entries_loaded, result.entries_expired, result.snapshot_age_secs
                );
            }
            Err(e) => {
                warn!("Failed to load cache from {:?}: {}", path, e);
            }
        }
    } else {
        info!(
            "Cache file {:?} does not exist, starting with empty cache",
            path
        );
    }
}

/// Save cache to file
async fn save_cache(cache: &SharedVerificationCache<VerifyUslResult>, path: &PathBuf) {
    info!("Saving cache to {:?}", path);
    match cache.save_to_file(path).await {
        Ok(result) => {
            info!(
                "Saved {} cache entries to {:?} ({} bytes)",
                result.entries_saved, path, result.size_bytes
            );
        }
        Err(e) => {
            error!("Failed to save cache to {:?}: {}", path, e);
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Initialize logging to stderr (stdout is used for MCP protocol in stdio mode)
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&args.log_level)),
        )
        .with_writer(std::io::stderr)
        .init();

    info!(
        "Starting DashProve MCP Server v{}",
        env!("CARGO_PKG_VERSION")
    );

    // Create cache based on configuration
    let cache: SharedVerificationCache<VerifyUslResult> = if args.no_cache {
        info!("Caching disabled");
        Arc::new(VerificationCache::disabled())
    } else {
        Arc::new(VerificationCache::new())
    };

    // Load cache from file if specified
    if let Some(ref cache_file) = args.cache_file {
        if !args.no_cache {
            load_cache_if_exists(&cache, cache_file).await;
        }
    }

    // Create server with the configured cache
    let server = McpServer::with_cache(cache);

    // Shared cache reference for persistence
    let cache_for_shutdown = server.verification_cache();
    let cache_file_for_shutdown = args.cache_file.clone();

    if args.cache_file.is_none() && args.cache_save_interval_secs > 0 && !args.no_cache {
        warn!(
            "cache_save_interval_secs is set to {} but --cache-file is not provided; periodic auto-save will be skipped",
            args.cache_save_interval_secs
        );
    }

    // Start periodic cache auto-save if configured
    let mut cache_autosave: Option<CacheAutoSaver<VerifyUslResult>> = None;
    if let (Some(cache_file), false) = (args.cache_file.as_ref(), args.no_cache) {
        if args.cache_save_interval_secs > 0 {
            let interval = Duration::from_secs(args.cache_save_interval_secs);
            info!(
                "Enabling cache auto-save every {}s to {:?}",
                interval.as_secs(),
                cache_file
            );
            cache_autosave = Some(CacheAutoSaver::start(
                cache_for_shutdown.clone(),
                cache_file.to_path_buf(),
                interval,
            ));
        } else {
            info!("Cache auto-save disabled (interval set to 0)");
        }
    }

    match args.transport {
        TransportType::Stdio => {
            let transport = StdioTransport::new();
            let mut server = server;

            // For stdio, we run until stdin closes
            // Set up a task to save cache on termination signal
            let shutdown_handle = tokio::spawn({
                let cache_file_for_shutdown = cache_file_for_shutdown.clone();
                let cache_for_shutdown = cache_for_shutdown.clone();
                let no_cache = args.no_cache;

                async move {
                    // Wait for termination signal
                    let _ = signal::ctrl_c().await;
                    info!("Received shutdown signal");
                    if let Some(ref path) = cache_file_for_shutdown {
                        if !no_cache {
                            save_cache(&cache_for_shutdown, path).await;
                        }
                    }
                }
            });

            let result = transport.run_async(&mut server).await;

            // Cancel the shutdown handler since we're exiting normally
            shutdown_handle.abort();

            // Stop auto-save before final persistence
            if let Some(autosave) = cache_autosave.take() {
                autosave.stop().await;
            }

            // Save cache on normal exit too
            if let Some(ref cache_file) = args.cache_file {
                if !args.no_cache {
                    save_cache(&server.verification_cache(), cache_file).await;
                }
            }

            result?;
        }
        TransportType::Http => {
            info!("Starting HTTP transport on {}", args.bind);

            // Build rate limit config if enabled
            let rate_limit_config = if args.rate_limit {
                info!(
                    "Rate limiting enabled: {} requests/sec, burst size {}",
                    args.rate_limit_rps, args.rate_limit_burst
                );
                Some(RateLimitConfig::new(
                    args.rate_limit_rps,
                    args.rate_limit_burst,
                ))
            } else {
                None
            };

            // Build log config if enabled
            let log_config = if args.request_logging {
                info!(
                    "Request logging enabled: max {} entries, log_request_body={}, log_response_body={}",
                    args.log_max_entries, args.log_request_body, args.log_response_body
                );
                Some(LogConfig {
                    max_entries: args.log_max_entries,
                    enabled: true,
                    log_request_body: args.log_request_body,
                    log_response_body: args.log_response_body,
                    ..Default::default()
                })
            } else {
                None
            };

            // Build metrics config if enabled
            let metrics_config = if args.metrics {
                info!("Metrics endpoint enabled: GET /metrics, GET /metrics/json");
                Some(MetricsConfig::enabled())
            } else {
                None
            };

            let transport = HttpTransport::with_all_options(
                &args.bind,
                args.api_token.clone(),
                rate_limit_config,
                log_config,
                metrics_config,
            );

            // Get cache reference before moving server
            let cache_for_http_shutdown = server.verification_cache();
            let cache_file_for_http = args.cache_file.clone();
            let no_cache_http = args.no_cache;

            // Run the HTTP server with graceful shutdown via select
            let shutdown_requested = tokio::select! {
                result = transport.run_async(server) => {
                    // Server exited normally
                    result?;
                    false
                }
                _ = signal::ctrl_c() => {
                    // Shutdown signal received
                    info!("Received shutdown signal, saving cache...");
                    if let Some(ref path) = cache_file_for_http {
                        if !no_cache_http {
                            save_cache(&cache_for_http_shutdown, path).await;
                        }
                    }
                    info!("Shutdown complete");
                    true
                }
            };

            // Stop auto-save before final persistence
            if let Some(autosave) = cache_autosave.take() {
                autosave.stop().await;
            }

            // Save cache on normal HTTP shutdown as well
            if !shutdown_requested {
                if let Some(ref path) = cache_file_for_http {
                    if !no_cache_http {
                        save_cache(&cache_for_http_shutdown, path).await;
                    }
                }
            }
        }
        TransportType::Websocket => {
            info!("Starting WebSocket transport on {}", args.bind);

            // Build rate limit config if enabled
            let rate_limit_config = if args.rate_limit {
                info!(
                    "Rate limiting enabled: {} requests/sec, burst size {}",
                    args.rate_limit_rps, args.rate_limit_burst
                );
                Some(RateLimitConfig::new(
                    args.rate_limit_rps,
                    args.rate_limit_burst,
                ))
            } else {
                None
            };

            // Build log config if enabled
            let log_config = if args.request_logging {
                info!(
                    "Request logging enabled: max {} entries, log_request_body={}, log_response_body={}",
                    args.log_max_entries, args.log_request_body, args.log_response_body
                );
                Some(LogConfig {
                    max_entries: args.log_max_entries,
                    enabled: true,
                    log_request_body: args.log_request_body,
                    log_response_body: args.log_response_body,
                    ..Default::default()
                })
            } else {
                None
            };

            // Build metrics config if enabled
            let metrics_config = if args.metrics {
                info!("Metrics endpoint enabled: GET /metrics, GET /metrics/json");
                Some(MetricsConfig::enabled())
            } else {
                None
            };

            let transport = WebSocketTransport::with_all_options(
                &args.bind,
                args.api_token.clone(),
                rate_limit_config,
                log_config,
                metrics_config,
            );

            // Get cache reference before moving server
            let cache_for_ws_shutdown = server.verification_cache();
            let cache_file_for_ws = args.cache_file.clone();
            let no_cache_ws = args.no_cache;

            // Run the WebSocket server with graceful shutdown via select
            let shutdown_requested = tokio::select! {
                result = transport.run_async(server) => {
                    // Server exited normally
                    result?;
                    false
                }
                _ = signal::ctrl_c() => {
                    // Shutdown signal received
                    info!("Received shutdown signal, saving cache...");
                    if let Some(ref path) = cache_file_for_ws {
                        if !no_cache_ws {
                            save_cache(&cache_for_ws_shutdown, path).await;
                        }
                    }
                    info!("Shutdown complete");
                    true
                }
            };

            // Stop auto-save before final persistence
            if let Some(autosave) = cache_autosave.take() {
                autosave.stop().await;
            }

            // Save cache on normal WebSocket shutdown as well
            if !shutdown_requested {
                if let Some(ref path) = cache_file_for_ws {
                    if !no_cache_ws {
                        save_cache(&cache_for_ws_shutdown, path).await;
                    }
                }
            }
        }
    }

    Ok(())
}
