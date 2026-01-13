//! WebSocket handler functions

use super::messages::{
    ProgressEvent, VerificationPhase, WsMessage, WsSessionQuery, WsVerifyRequest,
};
use crate::routes::{backend_metric_label, AppState, BackendIdParam, CompilationResult};
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Query, State,
    },
    response::IntoResponse,
};
use dashprove_backends::BackendId;
use dashprove_usl::{parse, typecheck};
use futures_util::{SinkExt, StreamExt};
use std::{sync::Arc, time::Instant};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// WebSocket upgrade handler for /ws/verify
pub async fn ws_verify_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
    Query(query): Query<WsSessionQuery>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| {
        handle_socket(socket, state, query.session_id, query.correlation_id)
    })
}

/// Handle an individual WebSocket connection
async fn handle_socket(
    socket: WebSocket,
    state: Arc<AppState>,
    requested_session_id: Option<String>,
    correlation_id: Option<String>,
) {
    let (mut sender, mut receiver) = socket.split();
    let requested_session_id = requested_session_id.filter(|id| !id.is_empty());
    // Validate correlation_id: non-empty and reasonable length
    let correlation_id = correlation_id.filter(|id| !id.is_empty() && id.len() <= 128);

    // Resolve session for this connection (resume if provided, else create)
    let (session_id, resumed) = if let Some(provided_id) = requested_session_id {
        match state.session_manager.reconnect_session(&provided_id).await {
            Some(_) => (provided_id, true),
            None => {
                warn!(session_id = %provided_id, "WebSocket resume requested with unknown session_id");
                let error_msg = WsMessage::Error {
                    request_id: None,
                    error: "Unknown session_id".to_string(),
                    details: Some(
                        "Session not found or expired; start a new session without session_id"
                            .to_string(),
                    ),
                };
                let _ = sender
                    .send(Message::Text(serde_json::to_string(&error_msg).unwrap()))
                    .await;
                return;
            }
        }
    } else {
        (state.session_manager.create_session().await, false)
    };

    // Send connected message with session ID
    let connected_msg = WsMessage::Connected {
        session_id: session_id.clone(),
        resumed,
        correlation_id: correlation_id.clone(),
    };
    if let Err(e) = sender
        .send(Message::Text(
            serde_json::to_string(&connected_msg).unwrap(),
        ))
        .await
    {
        error!("Failed to send connected message: {}", e);
        state.session_manager.disconnect_session(&session_id).await;
        return;
    }

    info!(session_id = %session_id, "WebSocket client connected");

    while let Some(msg) = receiver.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                debug!("Received WebSocket message: {}", text);

                // Parse the request
                let request: WsVerifyRequest = match serde_json::from_str(&text) {
                    Ok(req) => req,
                    Err(e) => {
                        let error_msg = WsMessage::Error {
                            request_id: None,
                            error: "Invalid request format".to_string(),
                            details: Some(e.to_string()),
                        };
                        if let Err(e) = sender
                            .send(Message::Text(serde_json::to_string(&error_msg).unwrap()))
                            .await
                        {
                            error!("Failed to send error message: {}", e);
                        }
                        continue;
                    }
                };

                // Generate or use provided request ID
                let request_id = request
                    .request_id
                    .unwrap_or_else(|| Uuid::new_v4().to_string());

                // Increment request count for this session
                state
                    .session_manager
                    .increment_request_count(&session_id)
                    .await;

                // Send acceptance
                let accepted = WsMessage::Accepted {
                    request_id: request_id.clone(),
                };
                if let Err(e) = sender
                    .send(Message::Text(serde_json::to_string(&accepted).unwrap()))
                    .await
                {
                    error!("Failed to send accepted message: {}", e);
                    continue;
                }

                // Create progress channel
                let (progress_tx, mut progress_rx) = mpsc::channel::<ProgressEvent>(32);

                // Spawn verification task
                let spec = request.spec.clone();
                let backend = request.backend.clone();
                let req_id = request_id.clone();
                let state_clone = state.clone();

                tokio::spawn(async move {
                    run_verification(spec, backend, progress_tx, state_clone).await;
                });

                // Stream progress to client
                while let Some(event) = progress_rx.recv().await {
                    let msg = match event {
                        ProgressEvent::Phase(phase, message, percentage) => WsMessage::Progress {
                            request_id: req_id.clone(),
                            phase,
                            message,
                            percentage,
                        },
                        ProgressEvent::BackendStarted(backend) => WsMessage::BackendStarted {
                            request_id: req_id.clone(),
                            backend,
                        },
                        ProgressEvent::BackendCompleted(backend, result) => {
                            WsMessage::BackendCompleted {
                                request_id: req_id.clone(),
                                backend,
                                result,
                            }
                        }
                        ProgressEvent::Completed {
                            valid,
                            property_count,
                            compilations,
                            errors,
                        } => WsMessage::Completed {
                            request_id: req_id.clone(),
                            valid,
                            property_count,
                            compilations,
                            errors,
                        },
                        ProgressEvent::Error(error, details) => WsMessage::Error {
                            request_id: Some(req_id.clone()),
                            error,
                            details,
                        },
                    };

                    if let Err(e) = sender
                        .send(Message::Text(serde_json::to_string(&msg).unwrap()))
                        .await
                    {
                        error!("Failed to send progress message: {}", e);
                        break;
                    }
                }
            }
            Ok(Message::Close(_)) => {
                info!(session_id = %session_id, "WebSocket client disconnected");
                break;
            }
            Ok(Message::Ping(data)) => {
                if let Err(e) = sender.send(Message::Pong(data)).await {
                    warn!("Failed to send pong: {}", e);
                }
            }
            Ok(_) => {
                // Ignore binary and pong messages
            }
            Err(e) => {
                error!(session_id = %session_id, "WebSocket error: {}", e);
                break;
            }
        }
    }

    // Mark session as disconnected
    state.session_manager.disconnect_session(&session_id).await;
}

/// Run verification with progress reporting
async fn run_verification(
    spec: String,
    backend: Option<BackendIdParam>,
    progress_tx: mpsc::Sender<ProgressEvent>,
    state: Arc<AppState>,
) {
    let overall_start = Instant::now();

    // Phase 1: Parsing
    let _ = progress_tx
        .send(ProgressEvent::Phase(
            VerificationPhase::Parsing,
            "Parsing specification...".to_string(),
            Some(10),
        ))
        .await;

    let parsed = match parse(&spec) {
        Ok(p) => p,
        Err(e) => {
            state
                .metrics
                .record_verification_with_duration(false, overall_start.elapsed().as_secs_f64())
                .await;
            let _ = progress_tx
                .send(ProgressEvent::Error(
                    "Parse error".to_string(),
                    Some(e.to_string()),
                ))
                .await;
            return;
        }
    };

    // Phase 2: Type checking
    let _ = progress_tx
        .send(ProgressEvent::Phase(
            VerificationPhase::TypeChecking,
            "Type-checking specification...".to_string(),
            Some(20),
        ))
        .await;

    let typed_spec = match typecheck(parsed) {
        Ok(t) => t,
        Err(e) => {
            state
                .metrics
                .record_verification_with_duration(false, overall_start.elapsed().as_secs_f64())
                .await;
            let _ = progress_tx
                .send(ProgressEvent::Error(
                    "Type error".to_string(),
                    Some(e.to_string()),
                ))
                .await;
            return;
        }
    };

    // Phase 3: Compiling to backends
    let _ = progress_tx
        .send(ProgressEvent::Phase(
            VerificationPhase::Compiling,
            "Compiling to verification backends...".to_string(),
            Some(30),
        ))
        .await;

    let backends: Vec<BackendId> = if let Some(b) = backend {
        vec![b.into()]
    } else {
        vec![
            BackendId::Lean4,
            BackendId::TlaPlus,
            BackendId::Kani,
            BackendId::Alloy,
        ]
    };

    let mut compilations = Vec::new();
    let mut errors = Vec::new();
    let backend_count = backends.len();

    for (i, backend_id) in backends.into_iter().enumerate() {
        let backend_param: BackendIdParam = backend_id.into();
        let backend_label = backend_metric_label(backend_id);
        let backend_start = Instant::now();

        // Report backend started
        let _ = progress_tx
            .send(ProgressEvent::BackendStarted(backend_param.clone()))
            .await;

        let percentage = 30 + ((i * 60) / backend_count) as u8;
        let _ = progress_tx
            .send(ProgressEvent::Phase(
                VerificationPhase::Compiling,
                format!("Compiling to {:?}...", backend_param),
                Some(percentage),
            ))
            .await;

        let result = match backend_id {
            BackendId::Lean4 => {
                let output = dashprove_usl::compile_to_lean(&typed_spec);
                CompilationResult {
                    backend: BackendIdParam::Lean4,
                    code: output.code,
                }
            }
            BackendId::TlaPlus => {
                let output = dashprove_usl::compile_to_tlaplus(&typed_spec);
                CompilationResult {
                    backend: BackendIdParam::TlaPlus,
                    code: output.code,
                }
            }
            BackendId::Kani => {
                let output = dashprove_usl::compile_to_kani(&typed_spec);
                CompilationResult {
                    backend: BackendIdParam::Kani,
                    code: output.code,
                }
            }
            BackendId::Alloy => {
                let output = dashprove_usl::compile_to_alloy(&typed_spec);
                CompilationResult {
                    backend: BackendIdParam::Alloy,
                    code: output.code,
                }
            }
            other => {
                errors.push(format!("Backend {:?} not yet implemented", other));
                continue;
            }
        };

        // Report backend completed
        let _ = progress_tx
            .send(ProgressEvent::BackendCompleted(
                backend_param,
                result.clone(),
            ))
            .await;

        state
            .metrics
            .record_backend_duration(backend_label, true, backend_start.elapsed().as_secs_f64())
            .await;

        compilations.push(result);
    }

    // Phase 4: Complete
    let _ = progress_tx
        .send(ProgressEvent::Phase(
            VerificationPhase::Merging,
            "Merging results...".to_string(),
            Some(95),
        ))
        .await;

    let success = errors.is_empty();
    let _ = progress_tx
        .send(ProgressEvent::Completed {
            valid: true,
            property_count: typed_spec.spec.properties.len(),
            compilations,
            errors,
        })
        .await;

    state
        .metrics
        .record_verification_with_duration(success, overall_start.elapsed().as_secs_f64())
        .await;
}
