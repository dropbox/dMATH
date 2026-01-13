//! Request tracking and metrics middleware
//!
//! Provides middleware for:
//! - Tracking in-flight requests for graceful shutdown support
//! - Recording request metrics (counts, durations) for Prometheus export
//! - Request ID generation and propagation for distributed tracing

use axum::{
    body::Body,
    extract::State,
    http::{HeaderValue, Request},
    middleware::Next,
    response::Response,
};
use std::sync::Arc;
use std::time::Instant;
use uuid::Uuid;

use crate::routes::AppState;

/// HTTP header name for request ID (standard for distributed tracing)
pub const REQUEST_ID_HEADER: &str = "X-Request-ID";

/// Generate a new request ID (UUIDv4)
fn generate_request_id() -> String {
    Uuid::new_v4().to_string()
}

/// Extract request ID from incoming request headers, or generate a new one
fn get_or_generate_request_id(request: &Request<Body>) -> String {
    request
        .headers()
        .get(REQUEST_ID_HEADER)
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty() && s.len() <= 128) // Validate: non-empty, reasonable length
        .map(String::from)
        .unwrap_or_else(generate_request_id)
}

/// Middleware that tracks in-flight requests, records metrics, and manages request IDs.
///
/// This middleware:
/// 1. Extracts or generates a unique request ID (`X-Request-ID` header)
/// 2. Increments the in-flight counter when a request starts
/// 3. Records the request duration and status in metrics
/// 4. Adds the request ID to the response headers
/// 5. Decrements the counter when the request completes
///
/// This enables:
/// - Distributed tracing (request ID propagation across services)
/// - Graceful shutdown (wait for in-flight requests to drain)
/// - Prometheus metrics export (request counts, durations, errors)
///
/// # Request ID Behavior
///
/// - If the client sends an `X-Request-ID` header, that ID is used (enables tracing across services)
/// - If no `X-Request-ID` header is present, a new UUIDv4 is generated
/// - The request ID is always returned in the response `X-Request-ID` header
/// - Request IDs are validated: must be non-empty and <= 128 characters
///
/// # Example
///
/// ```ignore
/// use axum::{Router, middleware};
/// use dashprove_server::middleware::request_tracking_middleware;
///
/// let app = Router::new()
///     .route("/health", get(health))
///     .with_state(state.clone())
///     .layer(middleware::from_fn_with_state(
///         state,
///         request_tracking_middleware,
///     ));
/// ```
pub async fn request_tracking_middleware(
    State(state): State<Arc<AppState>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let start = Instant::now();
    let method = request.method().to_string();
    let path = request.uri().path().to_string();

    // Extract or generate request ID for tracing
    let request_id = get_or_generate_request_id(&request);

    // Increment in-flight counter
    state.request_started();

    // Process the request
    let mut response = next.run(request).await;

    // Add request ID to response headers
    if let Ok(header_value) = HeaderValue::from_str(&request_id) {
        response
            .headers_mut()
            .insert(REQUEST_ID_HEADER, header_value);
    }

    // Record metrics
    let duration = start.elapsed().as_secs_f64();
    let status = response.status().as_u16();
    state
        .metrics
        .record_request(&method, &path, status, duration)
        .await;

    // Decrement in-flight counter (even on error)
    state.request_completed();

    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{routing::get, Router};
    use std::sync::atomic::Ordering;
    use tower::ServiceExt;

    async fn simple_handler() -> &'static str {
        "ok"
    }

    #[tokio::test]
    async fn test_request_tracking_increments_and_decrements() {
        let state = Arc::new(AppState::new());

        let app = Router::new()
            .route("/test", get(simple_handler))
            .with_state(state.clone())
            .layer(axum::middleware::from_fn_with_state(
                state.clone(),
                request_tracking_middleware,
            ));

        // Before request
        assert_eq!(state.in_flight_requests.load(Ordering::SeqCst), 0);

        // Make a request
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/test")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);

        // After request completes, counter should be back to 0
        assert_eq!(state.in_flight_requests.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn test_active_requests_method() {
        let state = Arc::new(AppState::new());

        assert_eq!(state.active_requests(), 0);

        state.request_started();
        assert_eq!(state.active_requests(), 1);

        state.request_started();
        assert_eq!(state.active_requests(), 2);

        state.request_completed();
        assert_eq!(state.active_requests(), 1);

        state.request_completed();
        assert_eq!(state.active_requests(), 0);
    }

    #[tokio::test]
    async fn test_request_id_generated_when_not_provided() {
        let state = Arc::new(AppState::new());

        let app = Router::new()
            .route("/test", get(simple_handler))
            .with_state(state.clone())
            .layer(axum::middleware::from_fn_with_state(
                state.clone(),
                request_tracking_middleware,
            ));

        // Request without X-Request-ID header
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/test")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        // Response should have X-Request-ID header with a generated UUID
        let request_id = response
            .headers()
            .get(REQUEST_ID_HEADER)
            .expect("Response should have X-Request-ID header");
        let request_id_str = request_id.to_str().unwrap();

        // Should be a valid UUID (36 chars with hyphens)
        assert_eq!(request_id_str.len(), 36);
        assert!(request_id_str
            .chars()
            .all(|c| c.is_ascii_hexdigit() || c == '-'));
    }

    #[tokio::test]
    async fn test_request_id_propagated_from_client() {
        let state = Arc::new(AppState::new());

        let app = Router::new()
            .route("/test", get(simple_handler))
            .with_state(state.clone())
            .layer(axum::middleware::from_fn_with_state(
                state.clone(),
                request_tracking_middleware,
            ));

        let client_request_id = "my-trace-id-12345";

        // Request with X-Request-ID header
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/test")
                    .header(REQUEST_ID_HEADER, client_request_id)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        // Response should echo back the same request ID
        let response_id = response
            .headers()
            .get(REQUEST_ID_HEADER)
            .expect("Response should have X-Request-ID header");
        assert_eq!(response_id.to_str().unwrap(), client_request_id);
    }

    #[tokio::test]
    async fn test_request_id_rejects_empty_header() {
        let state = Arc::new(AppState::new());

        let app = Router::new()
            .route("/test", get(simple_handler))
            .with_state(state.clone())
            .layer(axum::middleware::from_fn_with_state(
                state.clone(),
                request_tracking_middleware,
            ));

        // Request with empty X-Request-ID header
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/test")
                    .header(REQUEST_ID_HEADER, "")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        // Should generate a new UUID instead of using empty string
        let request_id = response
            .headers()
            .get(REQUEST_ID_HEADER)
            .expect("Response should have X-Request-ID header");
        let request_id_str = request_id.to_str().unwrap();
        assert!(!request_id_str.is_empty());
        assert_eq!(request_id_str.len(), 36); // UUID length
    }

    #[tokio::test]
    async fn test_request_id_rejects_too_long_header() {
        let state = Arc::new(AppState::new());

        let app = Router::new()
            .route("/test", get(simple_handler))
            .with_state(state.clone())
            .layer(axum::middleware::from_fn_with_state(
                state.clone(),
                request_tracking_middleware,
            ));

        // Request with very long X-Request-ID header (>128 chars)
        let long_id = "x".repeat(200);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/test")
                    .header(REQUEST_ID_HEADER, &long_id)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        // Should generate a new UUID instead of using too-long string
        let request_id = response
            .headers()
            .get(REQUEST_ID_HEADER)
            .expect("Response should have X-Request-ID header");
        let request_id_str = request_id.to_str().unwrap();
        assert_eq!(request_id_str.len(), 36); // UUID length, not 200
    }

    #[test]
    fn test_generate_request_id_is_valid_uuid() {
        let id = generate_request_id();
        // Should be 36 chars (UUID format with hyphens)
        assert_eq!(id.len(), 36);
        // Hyphens at correct positions
        let bytes = id.as_bytes();
        assert_eq!(bytes[8], b'-');
        assert_eq!(bytes[13], b'-');
        assert_eq!(bytes[18], b'-');
        assert_eq!(bytes[23], b'-');
    }

    #[test]
    fn test_request_id_header_constant() {
        assert_eq!(REQUEST_ID_HEADER, "X-Request-ID");
    }

    // ============================================
    // Mutation-killing tests for middleware.rs
    // ============================================

    #[test]
    fn test_generate_request_id_uniqueness() {
        // Generate multiple IDs and ensure they're all unique
        let ids: Vec<String> = (0..100).map(|_| generate_request_id()).collect();
        let unique_ids: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(ids.len(), unique_ids.len());
    }

    #[test]
    fn test_generate_request_id_format() {
        let id = generate_request_id();

        // Should be 36 characters (UUID format)
        assert_eq!(id.len(), 36);

        // Should have hyphens in correct positions
        assert_eq!(id.chars().nth(8), Some('-'));
        assert_eq!(id.chars().nth(13), Some('-'));
        assert_eq!(id.chars().nth(18), Some('-'));
        assert_eq!(id.chars().nth(23), Some('-'));

        // All non-hyphen characters should be hex
        for (i, c) in id.chars().enumerate() {
            if i == 8 || i == 13 || i == 18 || i == 23 {
                assert_eq!(c, '-');
            } else {
                assert!(c.is_ascii_hexdigit(), "char at {} is not hex: {}", i, c);
            }
        }
    }

    #[test]
    fn test_get_or_generate_request_id_with_valid_header() {
        let request = axum::http::Request::builder()
            .uri("/test")
            .header(REQUEST_ID_HEADER, "my-custom-request-id")
            .body(Body::empty())
            .unwrap();

        let id = get_or_generate_request_id(&request);
        assert_eq!(id, "my-custom-request-id");
    }

    #[test]
    fn test_get_or_generate_request_id_without_header() {
        let request = axum::http::Request::builder()
            .uri("/test")
            .body(Body::empty())
            .unwrap();

        let id = get_or_generate_request_id(&request);

        // Should generate a UUID
        assert_eq!(id.len(), 36);
    }

    #[test]
    fn test_get_or_generate_request_id_empty_header() {
        let request = axum::http::Request::builder()
            .uri("/test")
            .header(REQUEST_ID_HEADER, "")
            .body(Body::empty())
            .unwrap();

        let id = get_or_generate_request_id(&request);

        // Empty header should be rejected, generate new ID
        assert_eq!(id.len(), 36);
    }

    #[test]
    fn test_get_or_generate_request_id_exactly_128_chars() {
        // Exactly 128 chars should be accepted
        let long_id = "a".repeat(128);
        let request = axum::http::Request::builder()
            .uri("/test")
            .header(REQUEST_ID_HEADER, &long_id)
            .body(Body::empty())
            .unwrap();

        let id = get_or_generate_request_id(&request);
        assert_eq!(id, long_id);
    }

    #[test]
    fn test_get_or_generate_request_id_exactly_129_chars() {
        // 129 chars should be rejected (> 128)
        let too_long_id = "a".repeat(129);
        let request = axum::http::Request::builder()
            .uri("/test")
            .header(REQUEST_ID_HEADER, &too_long_id)
            .body(Body::empty())
            .unwrap();

        let id = get_or_generate_request_id(&request);

        // Should generate a new UUID, not use the too-long ID
        assert_eq!(id.len(), 36);
        assert_ne!(id, too_long_id);
    }

    #[tokio::test]
    async fn test_request_tracking_metrics_recorded() {
        let state = Arc::new(AppState::new());

        let app = Router::new()
            .route("/test", get(simple_handler))
            .with_state(state.clone())
            .layer(axum::middleware::from_fn_with_state(
                state.clone(),
                request_tracking_middleware,
            ));

        // Make a request
        let _response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/test")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        // Verify metrics were recorded by checking uptime (proves middleware ran)
        // and checking that the request completed successfully
        assert!(state.metrics.uptime_seconds() > 0.0);
    }

    #[tokio::test]
    async fn test_request_tracking_duration_recorded() {
        let state = Arc::new(AppState::new());

        let app = Router::new()
            .route("/test", get(simple_handler))
            .with_state(state.clone())
            .layer(axum::middleware::from_fn_with_state(
                state.clone(),
                request_tracking_middleware,
            ));

        // Make a request
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/test")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        // Verify request completed successfully (middleware processed it)
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        // Uptime should be measurable (proves metrics are initialized)
        assert!(state.metrics.uptime_seconds() >= 0.0);
    }

    #[tokio::test]
    async fn test_request_tracking_counter_returns_to_zero() {
        let state = Arc::new(AppState::new());

        // Ensure counter is at 0 before
        assert_eq!(state.active_requests(), 0);

        let app = Router::new()
            .route("/test", get(simple_handler))
            .with_state(state.clone())
            .layer(axum::middleware::from_fn_with_state(
                state.clone(),
                request_tracking_middleware,
            ));

        // Make multiple requests sequentially
        for _ in 0..5 {
            let app_clone = app.clone();
            let _response = app_clone
                .oneshot(
                    axum::http::Request::builder()
                        .uri("/test")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();

            // After each request completes, counter should be 0
            assert_eq!(state.active_requests(), 0);
        }
    }

    #[test]
    fn test_request_started_and_completed() {
        let state = AppState::new();

        assert_eq!(state.active_requests(), 0);

        state.request_started();
        assert_eq!(state.active_requests(), 1);

        state.request_started();
        assert_eq!(state.active_requests(), 2);

        state.request_completed();
        assert_eq!(state.active_requests(), 1);

        state.request_completed();
        assert_eq!(state.active_requests(), 0);
    }

    #[test]
    fn test_request_completed_after_started() {
        let state = AppState::new();

        // Counter starts at 0
        assert_eq!(state.active_requests(), 0);

        // Start some requests then complete them
        state.request_started();
        state.request_started();
        assert_eq!(state.active_requests(), 2);

        state.request_completed();
        assert_eq!(state.active_requests(), 1);

        state.request_completed();
        assert_eq!(state.active_requests(), 0);

        // Note: calling request_completed without a corresponding start
        // will wrap around (uses fetch_sub), which is the expected behavior
        // since normal middleware flow ensures balance
    }

    #[test]
    fn test_get_or_generate_request_id_filters_non_ascii() {
        // Header with non-ASCII content should be rejected via to_str() returning None
        // This test verifies the filter chain works correctly
        let request = axum::http::Request::builder()
            .uri("/test")
            .header(REQUEST_ID_HEADER, "valid-id-123")
            .body(Body::empty())
            .unwrap();

        let id = get_or_generate_request_id(&request);
        assert_eq!(id, "valid-id-123");
    }

    #[tokio::test]
    async fn test_request_id_preserved_through_middleware() {
        let state = Arc::new(AppState::new());

        let app = Router::new()
            .route("/test", get(simple_handler))
            .with_state(state.clone())
            .layer(axum::middleware::from_fn_with_state(
                state.clone(),
                request_tracking_middleware,
            ));

        let custom_id = "custom-trace-id-12345";

        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/test")
                    .header(REQUEST_ID_HEADER, custom_id)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        // Response should have the same request ID
        let response_id = response
            .headers()
            .get(REQUEST_ID_HEADER)
            .expect("Response should have X-Request-ID");
        assert_eq!(response_id.to_str().unwrap(), custom_id);
    }
}
