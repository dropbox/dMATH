//! Authentication and admin middleware

use super::error::AuthError;
use super::state::{AuthState, AuthenticatedUser};
use axum::{
    extract::{Request, State},
    http::{header, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};

/// Authentication middleware
///
/// Extracts API key from:
/// 1. `X-API-Key` header
/// 2. `Authorization: Bearer <key>` header
///
/// Sets extensions on the request:
/// - `AuthenticatedUser` with user info and admin status
pub async fn auth_middleware(
    State(auth): State<AuthState>,
    mut request: Request,
    next: Next,
) -> Response {
    // Skip auth for health endpoint
    if request.uri().path() == "/health" {
        return next.run(request).await;
    }

    // Extract API key from headers
    let api_key = extract_api_key(&request);

    // Read config to validate key
    let config = auth.config.read().await;

    // Validate API key if present
    let (key_id, rate_limit, is_admin) = match &api_key {
        Some(key) => {
            if let Some(info) = config.api_keys.get(key) {
                // Valid API key
                (key.clone(), info.rate_limit, info.is_admin)
            } else if config.required {
                // Invalid key when auth is required
                return (
                    StatusCode::UNAUTHORIZED,
                    Json(AuthError {
                        error: "Invalid API key".to_string(),
                        code: "invalid_api_key".to_string(),
                        retry_after: None,
                    }),
                )
                    .into_response();
            } else {
                // Invalid key but auth not required - treat as anonymous
                ("anonymous".to_string(), config.anonymous_rate_limit, false)
            }
        }
        None => {
            if config.required {
                return (
                    StatusCode::UNAUTHORIZED,
                    Json(AuthError {
                        error: "API key required".to_string(),
                        code: "missing_api_key".to_string(),
                        retry_after: None,
                    }),
                )
                    .into_response();
            }
            // No key and auth not required - anonymous
            ("anonymous".to_string(), config.anonymous_rate_limit, false)
        }
    };

    // Drop the config lock before continuing
    drop(config);

    // Store authenticated user info in request extensions
    let authenticated_user = AuthenticatedUser {
        api_key: api_key.clone(),
        is_admin,
    };
    request.extensions_mut().insert(authenticated_user);

    // Apply rate limiting
    match auth.rate_limiter.check(&key_id, rate_limit).await {
        Ok(count) => {
            // Add rate limit headers
            let mut response = next.run(request).await;
            let headers = response.headers_mut();
            headers.insert("X-RateLimit-Limit", rate_limit.to_string().parse().unwrap());
            headers.insert(
                "X-RateLimit-Remaining",
                (rate_limit.saturating_sub(count))
                    .to_string()
                    .parse()
                    .unwrap(),
            );
            response
        }
        Err(retry_after) => (
            StatusCode::TOO_MANY_REQUESTS,
            [(header::RETRY_AFTER, retry_after.to_string())],
            Json(AuthError {
                error: "Rate limit exceeded".to_string(),
                code: "rate_limited".to_string(),
                retry_after: Some(retry_after),
            }),
        )
            .into_response(),
    }
}

/// Extract API key from request headers
fn extract_api_key(request: &Request) -> Option<String> {
    // Try X-API-Key header first
    if let Some(value) = request.headers().get("X-API-Key") {
        if let Ok(key) = value.to_str() {
            return Some(key.to_string());
        }
    }

    // Try Authorization: Bearer header
    if let Some(value) = request.headers().get(header::AUTHORIZATION) {
        if let Ok(auth_str) = value.to_str() {
            if let Some(token) = auth_str.strip_prefix("Bearer ") {
                return Some(token.to_string());
            }
        }
    }

    None
}

/// Admin authentication middleware
///
/// Checks that the authenticated user has admin privileges.
/// Must be placed AFTER auth_middleware in the middleware chain.
/// Returns 403 Forbidden if the user is not an admin.
pub async fn admin_middleware(request: Request, next: Next) -> Response {
    // Get authenticated user from extensions (set by auth_middleware)
    let user = request.extensions().get::<AuthenticatedUser>();

    match user {
        Some(user) if user.is_admin => {
            // User is admin, allow request
            next.run(request).await
        }
        Some(_) => {
            // User is authenticated but not admin
            (
                StatusCode::FORBIDDEN,
                Json(AuthError {
                    error: "Admin privileges required".to_string(),
                    code: "admin_required".to_string(),
                    retry_after: None,
                }),
            )
                .into_response()
        }
        None => {
            // No authenticated user - this shouldn't happen if auth_middleware ran first
            (
                StatusCode::UNAUTHORIZED,
                Json(AuthError {
                    error: "Authentication required".to_string(),
                    code: "auth_required".to_string(),
                    retry_after: None,
                }),
            )
                .into_response()
        }
    }
}
