// Crate-level lint configuration for pedantic clippy
#![allow(clippy::must_use_candidate)] // Server handlers don't need must_use
#![allow(clippy::doc_markdown)] // Missing backticks - low priority
#![allow(clippy::uninlined_format_args)] // Named args are clearer
#![allow(clippy::map_unwrap_or)] // Style preference
#![allow(clippy::too_many_lines)] // Main may be long
#![allow(clippy::option_if_let_else)] // Style preference

//! DashProve REST API Server
//!
//! Provides REST API endpoints for verification, corpus search,
//! tactic suggestions, and proof sketch elaboration.
//!
//! Also provides WebSocket streaming at /ws/verify for long-running
//! verifications with real-time progress updates.
//!
//! Features:
//! - API key authentication via `X-API-Key` or `Authorization: Bearer` headers
//! - Configurable rate limiting per API key
//! - Anonymous access with stricter rate limits (when auth not required)
//!
//! See docs/DESIGN.md for API specification.

use dashprove_server::{
    admin, auth, middleware as server_middleware, routes, routes::ShutdownState, ws,
};

use axum::{
    middleware,
    routing::{delete, get, patch, post},
    Router,
};
use clap::Parser;
use dashprove_learning::ProofLearningSystem;
use routes::AppState;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::signal;
use tracing::{error, info, warn};

/// DashProve REST API Server
#[derive(Parser, Debug)]
#[command(name = "dashprove-server")]
#[command(about = "DashProve verification REST API server")]
struct Args {
    /// Port to listen on
    #[arg(short, long, default_value = "3000", env = "DASHPROVE_PORT")]
    port: u16,

    /// Host to bind to
    #[arg(long, default_value = "0.0.0.0", env = "DASHPROVE_HOST")]
    host: String,

    /// Directory for learning data (corpus, tactics)
    #[arg(long, env = "DASHPROVE_DATA_DIR")]
    data_dir: Option<PathBuf>,

    /// Require API key authentication
    #[arg(long, env = "DASHPROVE_REQUIRE_AUTH")]
    require_auth: bool,

    /// API keys (format: "key:name" or "key:name:rate_limit")
    /// Can be specified multiple times via CLI or comma-separated in DASHPROVE_API_KEYS env var
    #[arg(long = "api-key", value_name = "KEY:NAME[:LIMIT]")]
    api_keys: Vec<String>,

    /// Admin API keys (format: "key:name" or "key:name:rate_limit")
    /// Can be specified multiple times via CLI or comma-separated in DASHPROVE_ADMIN_KEYS env var
    #[arg(long = "admin-key", value_name = "KEY:NAME[:LIMIT]")]
    admin_keys: Vec<String>,

    /// Rate limit for anonymous requests (per minute, default: 10)
    #[arg(long, default_value = "10", env = "DASHPROVE_ANONYMOUS_RATE_LIMIT")]
    anonymous_rate_limit: u32,

    /// Path to persist API keys (JSON file)
    /// Keys added via admin API are persisted here and loaded on restart
    #[arg(long, env = "DASHPROVE_KEYS_FILE")]
    keys_file: Option<PathBuf>,

    /// Path to persist proof cache (JSON file)
    /// Cache is loaded on startup and can be saved via API
    #[arg(long, env = "DASHPROVE_CACHE_FILE")]
    cache_file: Option<PathBuf>,

    /// Interval in seconds for automatic cache saves (0 to disable)
    /// The cache is also saved on graceful shutdown regardless of this setting
    #[arg(long, default_value = "300", env = "DASHPROVE_CACHE_AUTOSAVE_INTERVAL")]
    cache_autosave_interval: u64,

    /// Maximum time in seconds to wait for in-flight requests during shutdown
    /// During this period, no new connections are accepted but existing requests
    /// are allowed to complete. Set to 0 to skip drain and shutdown immediately.
    #[arg(long, default_value = "30", env = "DASHPROVE_SHUTDOWN_DRAIN_TIMEOUT")]
    shutdown_drain_timeout: u64,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    // Load learning system if data directory provided
    let mut state = if let Some(data_dir) = &args.data_dir {
        info!(?data_dir, "Loading learning data");
        match ProofLearningSystem::load_from_dir(data_dir) {
            Ok(learning) => {
                info!(
                    corpus_size = learning.proof_count(),
                    "Learning system loaded"
                );
                AppState::with_learning(learning)
            }
            Err(e) => {
                tracing::warn!(?e, "Failed to load learning data, starting fresh");
                AppState::new()
            }
        }
    } else {
        AppState::new()
    };

    // Configure cache persistence if path provided
    if let Some(cache_file) = &args.cache_file {
        info!(?cache_file, "Loading proof cache from file");
        state = state.with_cache_path(cache_file);
        let cache_len = state.proof_cache.try_read().map(|c| c.len()).unwrap_or(0);
        info!(cache_entries = cache_len, "Cache loaded");
    }

    let state = Arc::new(state);

    // Configure authentication
    let mut auth_config = if args.require_auth {
        auth::AuthConfig::required()
    } else {
        auth::AuthConfig::disabled()
    };
    auth_config = auth_config.with_anonymous_rate_limit(args.anonymous_rate_limit);

    // Collect all API key specs from CLI args and environment variable
    let mut all_api_keys = args.api_keys.clone();

    // Parse DASHPROVE_API_KEYS environment variable (comma-separated)
    if let Ok(env_keys) = std::env::var("DASHPROVE_API_KEYS") {
        for key_spec in env_keys.split(',') {
            let trimmed = key_spec.trim();
            if !trimmed.is_empty() {
                all_api_keys.push(trimmed.to_string());
            }
        }
    }

    // Parse and add API keys
    for key_spec in &all_api_keys {
        let parts: Vec<&str> = key_spec.split(':').collect();
        match parts.len() {
            2 => {
                auth_config = auth_config.with_api_key(parts[0], parts[1]);
                info!(key = parts[0], name = parts[1], "Added API key");
            }
            3 => {
                if let Ok(limit) = parts[2].parse::<u32>() {
                    auth_config = auth_config.with_api_key_rate_limit(parts[0], parts[1], limit);
                    info!(
                        key = parts[0],
                        name = parts[1],
                        limit,
                        "Added API key with custom rate limit"
                    );
                } else {
                    tracing::warn!(spec = key_spec, "Invalid API key spec (bad rate limit)");
                }
            }
            _ => {
                tracing::warn!(
                    spec = key_spec,
                    "Invalid API key spec (expected key:name or key:name:limit)"
                );
            }
        }
    }

    // Collect all admin key specs from CLI args and environment variable
    let mut all_admin_keys = args.admin_keys.clone();

    // Parse DASHPROVE_ADMIN_KEYS environment variable (comma-separated)
    if let Ok(env_keys) = std::env::var("DASHPROVE_ADMIN_KEYS") {
        for key_spec in env_keys.split(',') {
            let trimmed = key_spec.trim();
            if !trimmed.is_empty() {
                all_admin_keys.push(trimmed.to_string());
            }
        }
    }

    // Parse and add admin keys
    for key_spec in &all_admin_keys {
        let parts: Vec<&str> = key_spec.split(':').collect();
        match parts.len() {
            2 => {
                auth_config = auth_config.with_admin_key(parts[0], parts[1]);
                info!(key = parts[0], name = parts[1], "Added admin API key");
            }
            3 => {
                if let Ok(limit) = parts[2].parse::<u32>() {
                    auth_config = auth_config.with_admin_key_rate_limit(parts[0], parts[1], limit);
                    info!(
                        key = parts[0],
                        name = parts[1],
                        limit,
                        "Added admin API key with custom rate limit"
                    );
                } else {
                    tracing::warn!(spec = key_spec, "Invalid admin key spec (bad rate limit)");
                }
            }
            _ => {
                tracing::warn!(
                    spec = key_spec,
                    "Invalid admin key spec (expected key:name or key:name:limit)"
                );
            }
        }
    }

    // Create auth state with optional persistence
    let auth_state = if let Some(keys_file) = &args.keys_file {
        info!(?keys_file, "Loading API keys from persistence file");
        auth::AuthState::load_from_file(auth_config.clone(), keys_file)
    } else {
        auth::AuthState::new(auth_config.clone())
    };

    // Re-read config to get updated count after loading from file
    let final_key_count = {
        let config = auth_state.config.read().await;
        config.api_keys.len()
    };

    info!(
        require_auth = auth_config.required,
        api_keys = final_key_count,
        anonymous_limit = auth_config.anonymous_rate_limit,
        keys_file = ?args.keys_file,
        "Authentication configured"
    );

    // Admin routes for API key management (use auth state)
    // Admin middleware is applied to these routes to require admin privileges
    let admin_key_routes = Router::new()
        .route("/admin/keys", get(admin::list_keys))
        .route("/admin/keys", post(admin::add_key))
        .route("/admin/keys/{key}", delete(admin::revoke_key))
        .route("/admin/keys/{key}", patch(admin::update_key))
        .with_state(auth_state.clone())
        .layer(middleware::from_fn(auth::admin_middleware));

    // Admin routes for session management (use app state for session manager access)
    let admin_session_routes = Router::new()
        .route("/admin/sessions", get(ws::list_sessions))
        .with_state(state.clone())
        .layer(middleware::from_fn(auth::admin_middleware));

    let app = Router::new()
        .route("/health", get(routes::health))
        .route("/version", get(routes::version))
        .route("/metrics", get(routes::prometheus_metrics))
        .route("/verify", post(routes::verify))
        .route("/verify/incremental", post(routes::verify_incremental))
        .route("/corpus/search", get(routes::corpus_search))
        .route("/corpus/stats", get(routes::corpus_stats))
        .route("/corpus/history", get(routes::corpus_history))
        .route("/corpus/compare", get(routes::corpus_compare))
        .route("/corpus/suggest", get(routes::corpus_suggest))
        .route(
            "/corpus/counterexamples/search",
            post(routes::counterexample_search),
        )
        .route(
            "/corpus/counterexamples/text-search",
            get(routes::counterexample_text_search),
        )
        .route(
            "/corpus/counterexamples",
            get(routes::counterexample_list).post(routes::counterexample_add),
        )
        .route(
            "/corpus/counterexamples/:id",
            get(routes::counterexample_get),
        )
        .route(
            "/corpus/counterexamples/classify",
            post(routes::counterexample_classify),
        )
        .route(
            "/corpus/counterexamples/clusters",
            post(routes::counterexample_clusters),
        )
        .route("/tactics/suggest", post(routes::tactics_suggest))
        .route("/sketch/elaborate", post(routes::sketch_elaborate))
        .route("/explain", post(routes::explain))
        .route("/proof-search", post(routes::proof_search))
        .route("/backends", get(routes::list_backends))
        // Cache management endpoints
        .route("/cache/stats", get(routes::cache_stats))
        .route("/cache/save", post(routes::cache_save))
        .route("/cache/load", post(routes::cache_load))
        .route("/cache/clear", delete(routes::cache_clear))
        // WebSocket endpoint for streaming verification
        .route("/ws/verify", get(ws::ws_verify_handler))
        .with_state(state.clone())
        // Merge admin routes (admin_middleware applied to these routes)
        .merge(admin_key_routes)
        .merge(admin_session_routes)
        // Auth middleware runs on all routes (including admin routes before admin_middleware)
        .layer(middleware::from_fn_with_state(
            auth_state,
            auth::auth_middleware,
        ))
        // Request tracking middleware for graceful shutdown drain (outermost layer)
        .layer(middleware::from_fn_with_state(
            state.clone(),
            server_middleware::request_tracking_middleware,
        ));

    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    info!("DashProve server listening on http://{}", addr);

    // Spawn background auto-save task if cache persistence is configured
    let autosave_handle = if let Some(ref cache_path) = args.cache_file {
        if args.cache_autosave_interval > 0 {
            let state_clone = state.clone();
            let cache_path_clone = cache_path.clone();
            let interval_secs = args.cache_autosave_interval;

            info!(interval_secs, "Starting cache auto-save background task");

            Some(tokio::spawn(async move {
                let mut interval =
                    tokio::time::interval(std::time::Duration::from_secs(interval_secs));
                interval.tick().await; // First tick completes immediately, skip it

                loop {
                    interval.tick().await;
                    match state_clone
                        .proof_cache
                        .read()
                        .await
                        .save_to_file(&cache_path_clone)
                    {
                        Ok(entries) => {
                            info!(entries, "Auto-save: cache persisted successfully");
                        }
                        Err(e) => {
                            error!(?e, "Auto-save: failed to persist cache");
                        }
                    }
                }
            }))
        } else {
            info!("Cache auto-save disabled (interval=0)");
            None
        }
    } else {
        None
    };

    // Run server with graceful shutdown
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(
            state.clone(),
            args.cache_file.clone(),
            args.shutdown_drain_timeout,
        ))
        .await
        .unwrap();

    // Cancel auto-save task if running
    if let Some(handle) = autosave_handle {
        handle.abort();
    }
}

/// Signal handler for graceful shutdown (SIGTERM, SIGINT)
///
/// On shutdown signal:
/// 1. Set shutdown state to Draining (health endpoint will return 503)
/// 2. Stop accepting new connections (handled by axum's graceful shutdown)
/// 3. Wait for in-flight HTTP requests to drain (up to drain_timeout_secs)
/// 4. Wait for WebSocket sessions to disconnect (up to drain_timeout_secs)
/// 5. Save cache to disk if configured
/// 6. Set shutdown state to ShuttingDown
async fn shutdown_signal(
    state: Arc<AppState>,
    cache_path: Option<PathBuf>,
    drain_timeout_secs: u64,
) {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {
            info!("Received Ctrl+C, initiating graceful shutdown...");
        }
        () = terminate => {
            info!("Received SIGTERM, initiating graceful shutdown...");
        }
    }

    // Set shutdown state to Draining - health endpoint will return 503
    state.set_shutdown_state(ShutdownState::Draining);
    info!("Shutdown state set to Draining - health endpoint now returns 503");

    // Drain in-flight requests and WebSocket sessions
    if drain_timeout_secs > 0 {
        let in_flight = state.active_requests();
        let ws_sessions = state.session_manager.active_count().await;

        if in_flight > 0 || ws_sessions > 0 {
            info!(
                in_flight_requests = in_flight,
                websocket_sessions = ws_sessions,
                drain_timeout_secs,
                "Waiting for in-flight requests and WebSocket sessions to complete..."
            );

            let drain_deadline =
                tokio::time::Instant::now() + std::time::Duration::from_secs(drain_timeout_secs);

            // Poll every 100ms until all requests and sessions complete or timeout
            loop {
                let current_requests = state.active_requests();
                let current_sessions = state.session_manager.active_count().await;

                if current_requests == 0 && current_sessions == 0 {
                    break;
                }

                if tokio::time::Instant::now() >= drain_deadline {
                    warn!(
                        remaining_requests = current_requests,
                        remaining_sessions = current_sessions,
                        "Drain timeout reached, proceeding with shutdown"
                    );
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }

            let final_requests = state.active_requests();
            let final_sessions = state.session_manager.active_count().await;
            if final_requests == 0 && final_sessions == 0 {
                info!("All in-flight requests and WebSocket sessions completed");
            }
        } else {
            info!("No in-flight requests or WebSocket sessions, proceeding with shutdown");
        }
    } else {
        info!("Drain timeout disabled (0), skipping request and session drain");
    }

    // Set state to ShuttingDown before saving cache
    state.set_shutdown_state(ShutdownState::ShuttingDown);

    // Save cache on shutdown if persistence is configured
    if let Some(cache_path) = cache_path {
        info!("Saving cache before shutdown...");
        match state.proof_cache.read().await.save_to_file(&cache_path) {
            Ok(entries) => {
                info!(entries, path = %cache_path.display(), "Cache saved successfully on shutdown");
            }
            Err(e) => {
                warn!(?e, "Failed to save cache on shutdown");
            }
        }
    }

    info!("Shutdown complete");
}
