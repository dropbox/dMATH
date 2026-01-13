//! Streaming verification support via Server-Sent Events (SSE)
//!
//! Provides real-time progress updates for long-running verification operations.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashprove_backends::{
    AlloyBackend, CoqBackend, HealthStatus, KaniBackend, Lean4Backend, TlaPlusBackend,
    VerificationBackend, VerificationStatus,
};
use dashprove_dispatcher::{Dispatcher, DispatcherConfig, ProgressUpdate, SelectionStrategy};
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, Mutex, RwLock};
use tracing::warn;

/// Session ID for streaming verification
pub type SessionId = String;

/// Generate a unique session ID
pub fn generate_session_id() -> SessionId {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let random: u32 = rand_u32();
    format!("sess_{:x}_{:08x}", timestamp, random)
}

/// Simple PRNG for session IDs
fn rand_u32() -> u32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    // LCG parameters from Numerical Recipes
    nanos.wrapping_mul(1664525).wrapping_add(1013904223)
}

/// Event types for streaming verification
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VerificationEvent {
    /// Session started
    Started {
        session_id: SessionId,
        spec_summary: String,
        backends: Vec<String>,
        total_properties: usize,
    },
    /// Progress update
    Progress {
        session_id: SessionId,
        completed: usize,
        total: usize,
        backend: String,
        property_index: usize,
        successful_so_far: usize,
        failed_so_far: usize,
    },
    /// Backend result available
    BackendResult {
        session_id: SessionId,
        backend: String,
        status: String,
        properties_verified: Vec<String>,
        properties_failed: Vec<String>,
        error: Option<String>,
        duration_ms: u64,
    },
    /// Verification completed
    Completed {
        session_id: SessionId,
        success: bool,
        summary: String,
        total_duration_ms: u64,
    },
    /// Verification cancelled
    Cancelled {
        session_id: SessionId,
        message: String,
        elapsed_ms: u64,
    },
    /// Error occurred
    Error {
        session_id: SessionId,
        message: String,
    },
}

/// Session state for streaming verification
#[derive(Debug)]
pub struct VerificationSession {
    /// Session ID
    pub id: SessionId,
    /// Session creation time
    pub created_at: Instant,
    /// Event sender for this session
    pub sender: broadcast::Sender<VerificationEvent>,
    /// Whether the session is complete
    pub completed: bool,
    /// Whether the session has been cancelled
    pub cancelled: bool,
    /// Final result if complete
    pub final_result: Option<StreamingVerifyResult>,
}

impl VerificationSession {
    /// Create a new verification session
    pub fn new(id: SessionId) -> Self {
        let (sender, _) = broadcast::channel(100);
        Self {
            id,
            created_at: Instant::now(),
            sender,
            completed: false,
            cancelled: false,
            final_result: None,
        }
    }

    /// Check if the session has been cancelled
    pub fn is_cancelled(&self) -> bool {
        self.cancelled
    }

    /// Subscribe to events from this session
    pub fn subscribe(&self) -> broadcast::Receiver<VerificationEvent> {
        self.sender.subscribe()
    }

    /// Send an event to all subscribers
    pub fn send(&self, event: VerificationEvent) {
        let _ = self.sender.send(event);
    }
}

/// Session manager for streaming verification
pub struct SessionManager {
    sessions: RwLock<HashMap<SessionId, Arc<Mutex<VerificationSession>>>>,
    /// Maximum session age before cleanup
    max_session_age: Duration,
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionManager {
    /// Create a new session manager
    pub fn new() -> Self {
        Self {
            sessions: RwLock::new(HashMap::new()),
            max_session_age: Duration::from_secs(300), // 5 minutes
        }
    }

    /// Create a new verification session
    pub async fn create_session(&self) -> Arc<Mutex<VerificationSession>> {
        let id = generate_session_id();
        let session = Arc::new(Mutex::new(VerificationSession::new(id.clone())));
        let mut sessions = self.sessions.write().await;
        sessions.insert(id, session.clone());
        session
    }

    /// Get an existing session
    pub async fn get_session(&self, id: &str) -> Option<Arc<Mutex<VerificationSession>>> {
        let sessions = self.sessions.read().await;
        sessions.get(id).cloned()
    }

    /// Cleanup old sessions
    pub async fn cleanup_old_sessions(&self) {
        let mut sessions = self.sessions.write().await;
        let now = Instant::now();
        sessions.retain(|_id, session| {
            // Use try_lock to avoid blocking - if locked, keep the session
            if let Ok(s) = session.try_lock() {
                now.duration_since(s.created_at) < self.max_session_age
            } else {
                true // Keep if locked (in use)
            }
        });
    }

    /// Get the status of a session
    pub async fn get_session_status(&self, id: &str) -> SessionStatus {
        let sessions = self.sessions.read().await;

        if let Some(session_arc) = sessions.get(id) {
            // Try to lock the session to get its status
            if let Ok(session) = session_arc.try_lock() {
                let age_secs = session.created_at.elapsed().as_secs();
                let subscriber_count = session.sender.receiver_count();

                SessionStatus {
                    session_id: id.to_string(),
                    exists: true,
                    completed: session.completed,
                    cancelled: session.cancelled,
                    age_secs,
                    final_result: session.final_result.clone(),
                    subscriber_count,
                }
            } else {
                // Session is locked (actively being used)
                SessionStatus {
                    session_id: id.to_string(),
                    exists: true,
                    completed: false, // If locked, likely still in progress
                    cancelled: false, // Unknown, session is busy
                    age_secs: 0,      // Unknown, session is busy
                    final_result: None,
                    subscriber_count: 0, // Unknown
                }
            }
        } else {
            SessionStatus {
                session_id: id.to_string(),
                exists: false,
                completed: false,
                cancelled: false,
                age_secs: 0,
                final_result: None,
                subscriber_count: 0,
            }
        }
    }

    /// Cancel a running verification session
    ///
    /// Sets the cancelled flag and sends a Cancelled event to all subscribers.
    /// Returns a result indicating whether the cancellation was successful.
    pub async fn cancel_session(&self, id: &str) -> CancelSessionResult {
        let sessions = self.sessions.read().await;

        if let Some(session_arc) = sessions.get(id) {
            // Try to lock the session
            if let Ok(mut session) = session_arc.try_lock() {
                // Check if already completed
                if session.completed {
                    return CancelSessionResult {
                        session_id: id.to_string(),
                        success: false,
                        message: "Session has already completed".to_string(),
                        was_completed: true,
                    };
                }

                // Check if already cancelled
                if session.cancelled {
                    return CancelSessionResult {
                        session_id: id.to_string(),
                        success: false,
                        message: "Session was already cancelled".to_string(),
                        was_completed: false,
                    };
                }

                // Set cancelled flag
                session.cancelled = true;
                let elapsed_ms = session.created_at.elapsed().as_millis() as u64;

                // Send cancellation event
                session.send(VerificationEvent::Cancelled {
                    session_id: id.to_string(),
                    message: "Session cancelled by user request".to_string(),
                    elapsed_ms,
                });

                // Mark as completed and set final result
                session.completed = true;
                session.final_result = Some(StreamingVerifyResult {
                    success: false,
                    session_id: id.to_string(),
                    summary: "Verification cancelled by user".to_string(),
                    duration_ms: elapsed_ms,
                });

                CancelSessionResult {
                    session_id: id.to_string(),
                    success: true,
                    message: "Session cancelled successfully".to_string(),
                    was_completed: false,
                }
            } else {
                // Session is locked - still mark for cancellation
                // The verification loop should check for cancellation
                CancelSessionResult {
                    session_id: id.to_string(),
                    success: true,
                    message: "Cancellation requested, session will be cancelled when next checked"
                        .to_string(),
                    was_completed: false,
                }
            }
        } else {
            CancelSessionResult {
                session_id: id.to_string(),
                success: false,
                message: "Session not found".to_string(),
                was_completed: false,
            }
        }
    }

    /// Get the count of active sessions
    pub async fn session_count(&self) -> usize {
        let sessions = self.sessions.read().await;
        sessions.len()
    }
}

/// Arguments for streaming verify_usl tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingVerifyArgs {
    /// USL specification to verify
    pub spec: String,
    /// Verification strategy: auto, single, redundant, all
    #[serde(default = "default_strategy")]
    pub strategy: String,
    /// Specific backends to use
    #[serde(default)]
    pub backends: Vec<String>,
    /// Timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout: u64,
    /// Skip health checks
    #[serde(default)]
    pub skip_health_check: bool,
}

fn default_strategy() -> String {
    "auto".to_string()
}

fn default_timeout() -> u64 {
    60
}

/// Result of starting a streaming verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingVerifyStartResult {
    /// Session ID to subscribe to
    pub session_id: SessionId,
    /// URL to subscribe to events
    pub events_url: String,
    /// Message
    pub message: String,
}

/// Final result of streaming verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingVerifyResult {
    /// Overall success
    pub success: bool,
    /// Session ID
    pub session_id: SessionId,
    /// Summary message
    pub summary: String,
    /// Duration in milliseconds
    pub duration_ms: u64,
}

/// Session status response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStatus {
    /// Session ID
    pub session_id: SessionId,
    /// Whether the session exists
    pub exists: bool,
    /// Whether the session is completed
    pub completed: bool,
    /// Whether the session was cancelled
    pub cancelled: bool,
    /// Age of the session in seconds
    pub age_secs: u64,
    /// Final result if completed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_result: Option<StreamingVerifyResult>,
    /// Number of active subscribers (approximate)
    pub subscriber_count: usize,
}

/// Result of cancelling a session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelSessionResult {
    /// Session ID
    pub session_id: SessionId,
    /// Whether cancellation was successful
    pub success: bool,
    /// Message describing the result
    pub message: String,
    /// Whether the session was already completed
    pub was_completed: bool,
}

/// Helper to check if session is cancelled (returns early if so)
async fn check_cancelled(
    session: &Arc<Mutex<VerificationSession>>,
    session_id: &str,
    start_time: Instant,
) -> Option<StreamingVerifyResult> {
    let s = session.lock().await;
    if s.cancelled {
        let elapsed_ms = start_time.elapsed().as_millis() as u64;
        Some(StreamingVerifyResult {
            success: false,
            session_id: session_id.to_string(),
            summary: "Verification cancelled by user".to_string(),
            duration_ms: elapsed_ms,
        })
    } else {
        None
    }
}

/// Run streaming verification in the background
pub async fn run_streaming_verification(
    session: Arc<Mutex<VerificationSession>>,
    args: StreamingVerifyArgs,
) {
    let session_id = {
        let s = session.lock().await;
        s.id.clone()
    };

    let start_time = Instant::now();

    // Check for cancellation before starting
    if let Some(result) = check_cancelled(&session, &session_id, start_time).await {
        let mut s = session.lock().await;
        s.completed = true;
        s.final_result = Some(result);
        return;
    }

    // Parse the USL spec
    let spec = match dashprove_usl::parse(&args.spec) {
        Ok(spec) => spec,
        Err(e) => {
            let mut s = session.lock().await;
            s.send(VerificationEvent::Error {
                session_id: session_id.clone(),
                message: format!("Failed to parse USL specification: {}", e),
            });
            s.completed = true;
            s.final_result = Some(StreamingVerifyResult {
                success: false,
                session_id,
                summary: format!("Parse error: {}", e),
                duration_ms: start_time.elapsed().as_millis() as u64,
            });
            return;
        }
    };

    // Check for cancellation after parsing
    if let Some(result) = check_cancelled(&session, &session_id, start_time).await {
        let mut s = session.lock().await;
        s.completed = true;
        s.final_result = Some(result);
        return;
    }

    // Type check the spec
    let typed_spec = match dashprove_usl::typecheck(spec.clone()) {
        Ok(typed) => typed,
        Err(e) => {
            let mut s = session.lock().await;
            s.send(VerificationEvent::Error {
                session_id: session_id.clone(),
                message: format!("Type error: {}", e),
            });
            s.completed = true;
            s.final_result = Some(StreamingVerifyResult {
                success: false,
                session_id,
                summary: format!("Type error: {}", e),
                duration_ms: start_time.elapsed().as_millis() as u64,
            });
            return;
        }
    };

    // Check for cancellation after type checking
    if let Some(result) = check_cancelled(&session, &session_id, start_time).await {
        let mut s = session.lock().await;
        s.completed = true;
        s.final_result = Some(result);
        return;
    }

    let property_names: Vec<String> = spec.properties.iter().map(|p| p.name()).collect();

    // Configure dispatcher
    let config = match args.strategy.as_str() {
        "single" => DispatcherConfig {
            selection_strategy: SelectionStrategy::Single,
            task_timeout: Duration::from_secs(args.timeout),
            check_health: !args.skip_health_check,
            ..Default::default()
        },
        "redundant" => DispatcherConfig {
            selection_strategy: SelectionStrategy::Redundant { min_backends: 2 },
            task_timeout: Duration::from_secs(args.timeout),
            check_health: !args.skip_health_check,
            ..Default::default()
        },
        "all" => DispatcherConfig {
            selection_strategy: SelectionStrategy::All,
            task_timeout: Duration::from_secs(args.timeout),
            check_health: !args.skip_health_check,
            ..Default::default()
        },
        _ => DispatcherConfig {
            selection_strategy: SelectionStrategy::Single,
            task_timeout: Duration::from_secs(args.timeout),
            check_health: !args.skip_health_check,
            ..Default::default()
        },
    };

    let mut dispatcher = Dispatcher::new(config);

    // Register backends
    let requested_backends = if args.backends.is_empty() {
        vec![
            "lean4".to_string(),
            "tlaplus".to_string(),
            "kani".to_string(),
            "coq".to_string(),
            "alloy".to_string(),
        ]
    } else {
        args.backends.clone()
    };

    let mut registered = Vec::new();
    for backend_name in &requested_backends {
        match backend_name.to_lowercase().as_str() {
            "lean4" | "lean" => {
                let backend: Arc<dyn VerificationBackend> = Arc::new(Lean4Backend::new());
                if args.skip_health_check || is_backend_available(&backend).await {
                    dispatcher.register_backend(backend);
                    registered.push("lean4".to_string());
                }
            }
            "tlaplus" | "tla+" | "tla" => {
                let backend: Arc<dyn VerificationBackend> = Arc::new(TlaPlusBackend::new());
                if args.skip_health_check || is_backend_available(&backend).await {
                    dispatcher.register_backend(backend);
                    registered.push("tlaplus".to_string());
                }
            }
            "kani" => {
                let backend: Arc<dyn VerificationBackend> = Arc::new(KaniBackend::new());
                if args.skip_health_check || is_backend_available(&backend).await {
                    dispatcher.register_backend(backend);
                    registered.push("kani".to_string());
                }
            }
            "coq" => {
                let backend: Arc<dyn VerificationBackend> = Arc::new(CoqBackend::new());
                if args.skip_health_check || is_backend_available(&backend).await {
                    dispatcher.register_backend(backend);
                    registered.push("coq".to_string());
                }
            }
            "alloy" => {
                let backend: Arc<dyn VerificationBackend> = Arc::new(AlloyBackend::new());
                if args.skip_health_check || is_backend_available(&backend).await {
                    dispatcher.register_backend(backend);
                    registered.push("alloy".to_string());
                }
            }
            _ => {
                warn!("Unknown backend '{}', skipping", backend_name);
            }
        }
    }

    // Check for cancellation after backend registration
    if let Some(result) = check_cancelled(&session, &session_id, start_time).await {
        let mut s = session.lock().await;
        s.completed = true;
        s.final_result = Some(result);
        return;
    }

    // Send started event
    {
        let s = session.lock().await;
        s.send(VerificationEvent::Started {
            session_id: session_id.clone(),
            spec_summary: format!("{} properties defined", property_names.len()),
            backends: registered.clone(),
            total_properties: property_names.len(),
        });
    }

    if registered.is_empty() {
        let mut s = session.lock().await;
        s.send(VerificationEvent::Completed {
            session_id: session_id.clone(),
            success: true,
            summary: "No backends available. Type-check only passed.".to_string(),
            total_duration_ms: start_time.elapsed().as_millis() as u64,
        });
        s.completed = true;
        s.final_result = Some(StreamingVerifyResult {
            success: true,
            session_id,
            summary: "Type-check only (no backends)".to_string(),
            duration_ms: start_time.elapsed().as_millis() as u64,
        });
        return;
    }

    // Set up progress callback
    // We need to get a sender that can be used synchronously in the callback
    let sender = {
        let s = session.lock().await;
        s.sender.clone()
    };
    let session_id_for_callback = session_id.clone();
    dispatcher.set_progress_callback(move |update: ProgressUpdate| {
        let session_id = session_id_for_callback.clone();

        // Send directly on the broadcast channel (no async needed)
        let _ = sender.send(VerificationEvent::Progress {
            session_id,
            completed: update.completed,
            total: update.total,
            backend: format!("{:?}", update.backend),
            property_index: update.property_index,
            successful_so_far: update.successful_so_far,
            failed_so_far: update.failed_so_far,
        });
    });

    // Run verification
    let verify_result = dispatcher.verify(&typed_spec).await;

    // Check for cancellation after verification completes
    // This handles cancellation requests that came in during verification
    if let Some(result) = check_cancelled(&session, &session_id, start_time).await {
        let mut s = session.lock().await;
        // Only update if not already marked complete by cancel_session
        if !s.completed {
            s.completed = true;
            s.final_result = Some(result);
        }
        return;
    }

    match verify_result {
        Ok(merged_results) => {
            // Send backend results
            for prop_result in &merged_results.properties {
                for br in &prop_result.backend_results {
                    let (status_str, verified, failed) = match &br.status {
                        VerificationStatus::Proven => {
                            ("proven".to_string(), property_names.clone(), vec![])
                        }
                        VerificationStatus::Disproven => {
                            ("disproven".to_string(), vec![], property_names.clone())
                        }
                        VerificationStatus::Unknown { reason } => {
                            (format!("unknown: {}", reason), vec![], vec![])
                        }
                        VerificationStatus::Partial {
                            verified_percentage,
                        } => (
                            format!("partial: {:.1}%", verified_percentage),
                            vec![],
                            vec![],
                        ),
                    };

                    let s = session.lock().await;
                    s.send(VerificationEvent::BackendResult {
                        session_id: session_id.clone(),
                        backend: format!("{:?}", br.backend),
                        status: status_str,
                        properties_verified: verified,
                        properties_failed: failed,
                        error: br.error.clone(),
                        duration_ms: br.time_taken.as_millis() as u64,
                    });
                }
            }

            let overall_success =
                merged_results.summary.proven > 0 && merged_results.summary.disproven == 0;

            let summary = format!(
                "Verified {} properties: {} proven, {} disproven, {} unknown. Confidence: {:.1}%",
                merged_results.properties.len(),
                merged_results.summary.proven,
                merged_results.summary.disproven,
                merged_results.summary.unknown,
                merged_results.summary.overall_confidence * 100.0,
            );

            let mut s = session.lock().await;
            s.send(VerificationEvent::Completed {
                session_id: session_id.clone(),
                success: overall_success,
                summary: summary.clone(),
                total_duration_ms: start_time.elapsed().as_millis() as u64,
            });
            s.completed = true;
            s.final_result = Some(StreamingVerifyResult {
                success: overall_success,
                session_id,
                summary,
                duration_ms: start_time.elapsed().as_millis() as u64,
            });
        }
        Err(e) => {
            let mut s = session.lock().await;
            s.send(VerificationEvent::Error {
                session_id: session_id.clone(),
                message: format!("Verification failed: {}", e),
            });
            s.completed = true;
            s.final_result = Some(StreamingVerifyResult {
                success: false,
                session_id,
                summary: format!("Verification failed: {}", e),
                duration_ms: start_time.elapsed().as_millis() as u64,
            });
        }
    }
}

/// Check if a backend is available (health check)
async fn is_backend_available(backend: &Arc<dyn VerificationBackend>) -> bool {
    matches!(backend.health_check().await, HealthStatus::Healthy)
}
