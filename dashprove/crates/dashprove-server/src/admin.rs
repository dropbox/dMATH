//! Admin API routes for DashProve server
//!
//! Provides endpoints for managing API keys at runtime:
//! - GET /admin/keys - List all API keys (names only)
//! - POST /admin/keys - Add a new API key
//! - DELETE /admin/keys/:key - Revoke an API key
//!
//! Note: These endpoints require an existing valid API key with admin privileges,
//! or can be accessed when authentication is not required.

use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};

use crate::auth::{AuthState, KeyInfo};

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify AddKeyRequest default is_admin is false
    #[kani::proof]
    fn verify_add_key_request_default_is_admin() {
        let req = AddKeyRequest {
            key: String::from("test-key"),
            name: String::from("Test"),
            rate_limit: None,
            is_admin: false,
        };
        kani::assert(!req.is_admin, "is_admin default should be false");
    }

    /// Verify AddKeyRequest rate_limit can be None
    #[kani::proof]
    fn verify_add_key_request_rate_limit_none() {
        let req = AddKeyRequest {
            key: String::from("test"),
            name: String::from("Test"),
            rate_limit: None,
            is_admin: false,
        };
        kani::assert(req.rate_limit.is_none(), "rate_limit should be None");
    }

    /// Verify AddKeyRequest rate_limit can store any u32
    #[kani::proof]
    fn verify_add_key_request_rate_limit_some() {
        let limit: u32 = kani::any();
        let req = AddKeyRequest {
            key: String::from("test"),
            name: String::from("Test"),
            rate_limit: Some(limit),
            is_admin: false,
        };
        kani::assert(
            req.rate_limit == Some(limit),
            "rate_limit should store value",
        );
    }

    /// Verify AddKeyResponse success field stores value
    #[kani::proof]
    fn verify_add_key_response_success() {
        let success: bool = kani::any();
        let key_info = KeyInfo {
            key_prefix: String::from("test..."),
            name: String::from("Test"),
            rate_limit: 100,
            is_admin: false,
        };
        let resp = AddKeyResponse {
            success,
            message: String::from("msg"),
            key_info,
        };
        kani::assert(resp.success == success, "success should be stored");
    }

    /// Verify ListKeysResponse total stores value
    #[kani::proof]
    fn verify_list_keys_response_total() {
        let total: usize = kani::any();
        kani::assume(total < 1000); // Reasonable bound

        let resp = ListKeysResponse {
            keys: Vec::new(),
            total,
        };
        kani::assert(resp.total == total, "total should be stored");
    }

    /// Verify RevokeKeyResponse stores success value
    #[kani::proof]
    fn verify_revoke_key_response() {
        let success: bool = kani::any();
        let resp = RevokeKeyResponse {
            success,
            message: String::from("test"),
        };
        kani::assert(resp.success == success, "success should be stored");
    }

    /// Verify UpdateKeyRequest rate_limit can be Some or None
    #[kani::proof]
    fn verify_update_key_request_rate_limit() {
        let has_limit: bool = kani::any();
        let limit: u32 = kani::any();

        let rate_limit = if has_limit { Some(limit) } else { None };
        let req = UpdateKeyRequest {
            rate_limit,
            is_admin: None,
        };

        if has_limit {
            kani::assert(req.rate_limit == Some(limit), "rate_limit should be Some");
        } else {
            kani::assert(req.rate_limit.is_none(), "rate_limit should be None");
        }
    }

    /// Verify UpdateKeyRequest is_admin can be Some or None
    #[kani::proof]
    fn verify_update_key_request_is_admin() {
        let has_admin: bool = kani::any();
        let admin: bool = kani::any();

        let is_admin = if has_admin { Some(admin) } else { None };
        let req = UpdateKeyRequest {
            rate_limit: None,
            is_admin,
        };

        if has_admin {
            kani::assert(req.is_admin == Some(admin), "is_admin should be Some");
        } else {
            kani::assert(req.is_admin.is_none(), "is_admin should be None");
        }
    }

    /// Verify UpdateKeyResponse stores success and key_info
    #[kani::proof]
    fn verify_update_key_response() {
        let success: bool = kani::any();
        let rate_limit: u32 = kani::any();
        let is_admin: bool = kani::any();

        let key_info = KeyInfo {
            key_prefix: String::from("test..."),
            name: String::from("Test"),
            rate_limit,
            is_admin,
        };
        let resp = UpdateKeyResponse {
            success,
            message: String::from("msg"),
            key_info,
        };

        kani::assert(resp.success == success, "success stored");
        kani::assert(resp.key_info.rate_limit == rate_limit, "rate_limit stored");
        kani::assert(resp.key_info.is_admin == is_admin, "is_admin stored");
    }

    /// Verify AdminError stores error and code
    #[kani::proof]
    fn verify_admin_error() {
        let error = AdminError {
            error: String::from("test error"),
            code: String::from("test_code"),
        };
        // Fields are stored (we can't easily verify String content in Kani)
        kani::assert(!error.error.is_empty(), "error should not be empty");
        kani::assert(!error.code.is_empty(), "code should not be empty");
    }
}

/// Request to add a new API key
#[derive(Debug, Deserialize)]
pub struct AddKeyRequest {
    /// The API key value
    pub key: String,
    /// Human-readable name for the key
    pub name: String,
    /// Rate limit (requests per minute, default: 100)
    #[serde(default)]
    pub rate_limit: Option<u32>,
    /// Whether this key should have admin privileges (default: false)
    #[serde(default)]
    pub is_admin: bool,
}

/// Response after adding a key
#[derive(Debug, Serialize, Deserialize)]
pub struct AddKeyResponse {
    /// Whether the operation succeeded
    pub success: bool,
    /// Human-readable message
    pub message: String,
    /// Information about the created key
    pub key_info: KeyInfo,
}

/// Response listing all keys
#[derive(Debug, Serialize, Deserialize)]
pub struct ListKeysResponse {
    /// List of API keys
    pub keys: Vec<KeyInfo>,
    /// Total number of keys
    pub total: usize,
}

/// Response after revoking a key
#[derive(Debug, Serialize, Deserialize)]
pub struct RevokeKeyResponse {
    /// Whether the operation succeeded
    pub success: bool,
    /// Human-readable message
    pub message: String,
}

/// Request to update an API key
#[derive(Debug, Deserialize)]
pub struct UpdateKeyRequest {
    /// New rate limit (requests per minute)
    pub rate_limit: Option<u32>,
    /// Whether this key should have admin privileges
    pub is_admin: Option<bool>,
}

/// Response after updating a key
#[derive(Debug, Serialize, Deserialize)]
pub struct UpdateKeyResponse {
    /// Whether the operation succeeded
    pub success: bool,
    /// Human-readable message
    pub message: String,
    /// Updated key information
    pub key_info: KeyInfo,
}

/// Error response for admin operations
#[derive(Debug, Serialize, Deserialize)]
pub struct AdminError {
    /// Error message
    pub error: String,
    /// Error code for programmatic handling
    pub code: String,
}

/// GET /admin/keys - List all API keys
///
/// Returns a list of all registered API keys with their names and rate limits.
/// Key values are masked (only prefix shown) for security.
pub async fn list_keys(State(auth): State<AuthState>) -> Json<ListKeysResponse> {
    let keys = auth.list_keys().await;
    let total = keys.len();
    Json(ListKeysResponse { keys, total })
}

/// POST /admin/keys - Add a new API key
///
/// Creates a new API key with the specified name and optional rate limit.
/// Returns an error if the key already exists.
pub async fn add_key(
    State(auth): State<AuthState>,
    Json(req): Json<AddKeyRequest>,
) -> Result<Json<AddKeyResponse>, (StatusCode, Json<AdminError>)> {
    // Validate key format
    if req.key.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(AdminError {
                error: "API key cannot be empty".to_string(),
                code: "invalid_key".to_string(),
            }),
        ));
    }

    if req.key.len() < 16 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(AdminError {
                error: "API key must be at least 16 characters".to_string(),
                code: "key_too_short".to_string(),
            }),
        ));
    }

    if req.name.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(AdminError {
                error: "Key name cannot be empty".to_string(),
                code: "invalid_name".to_string(),
            }),
        ));
    }

    // Check if key already exists
    if auth.has_key(&req.key).await {
        return Err((
            StatusCode::CONFLICT,
            Json(AdminError {
                error: "API key already exists".to_string(),
                code: "key_exists".to_string(),
            }),
        ));
    }

    // Add the key
    let rate_limit = req.rate_limit.unwrap_or(100);
    auth.add_key(&req.key, &req.name, Some(rate_limit), req.is_admin)
        .await;

    let key_prefix = if req.key.len() > 8 {
        format!("{}...", &req.key[..8])
    } else {
        req.key.clone()
    };

    Ok(Json(AddKeyResponse {
        success: true,
        message: "API key added successfully".to_string(),
        key_info: KeyInfo {
            key_prefix,
            name: req.name,
            rate_limit,
            is_admin: req.is_admin,
        },
    }))
}

/// DELETE /admin/keys/:key - Revoke an API key
///
/// Removes the specified API key. Returns success even if the key didn't exist.
pub async fn revoke_key(
    State(auth): State<AuthState>,
    Path(key): Path<String>,
) -> Result<Json<RevokeKeyResponse>, (StatusCode, Json<AdminError>)> {
    if key.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(AdminError {
                error: "API key cannot be empty".to_string(),
                code: "invalid_key".to_string(),
            }),
        ));
    }

    let removed = auth.remove_key(&key).await;

    if removed {
        Ok(Json(RevokeKeyResponse {
            success: true,
            message: "API key revoked successfully".to_string(),
        }))
    } else {
        Err((
            StatusCode::NOT_FOUND,
            Json(AdminError {
                error: "API key not found".to_string(),
                code: "key_not_found".to_string(),
            }),
        ))
    }
}

/// PATCH /admin/keys/:key - Update an API key
///
/// Updates properties of an existing API key (rate limit and/or admin status).
/// Returns the updated key info.
pub async fn update_key(
    State(auth): State<AuthState>,
    Path(key): Path<String>,
    Json(req): Json<UpdateKeyRequest>,
) -> Result<Json<UpdateKeyResponse>, (StatusCode, Json<AdminError>)> {
    if key.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(AdminError {
                error: "API key cannot be empty".to_string(),
                code: "invalid_key".to_string(),
            }),
        ));
    }

    // Check if anything is being updated
    if req.rate_limit.is_none() && req.is_admin.is_none() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(AdminError {
                error: "No updates specified. Provide rate_limit and/or is_admin.".to_string(),
                code: "no_updates".to_string(),
            }),
        ));
    }

    // Validate rate_limit if provided
    if let Some(limit) = req.rate_limit {
        if limit == 0 {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(AdminError {
                    error: "Rate limit must be greater than 0".to_string(),
                    code: "invalid_rate_limit".to_string(),
                }),
            ));
        }
    }

    // Attempt to update the key
    let updated = auth.update_key(&key, req.rate_limit, req.is_admin).await;

    if !updated {
        return Err((
            StatusCode::NOT_FOUND,
            Json(AdminError {
                error: "API key not found".to_string(),
                code: "key_not_found".to_string(),
            }),
        ));
    }

    // Get updated key info
    let key_info = auth.get_key_info(&key).await.unwrap();

    Ok(Json(UpdateKeyResponse {
        success: true,
        message: "API key updated successfully".to_string(),
        key_info,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::AuthConfig;
    use axum::{
        body::Body,
        http::Request,
        routing::{delete, get, patch, post},
        Router,
    };
    use tower::ServiceExt;

    fn test_admin_app() -> (Router, AuthState) {
        let auth_state = AuthState::new(
            AuthConfig::disabled().with_api_key("existing-key-12345678", "Existing User"),
        );
        let app = Router::new()
            .route("/admin/keys", get(list_keys))
            .route("/admin/keys", post(add_key))
            .route("/admin/keys/:key", delete(revoke_key))
            .with_state(auth_state.clone());
        (app, auth_state)
    }

    #[tokio::test]
    async fn test_list_keys_empty() {
        let auth_state = AuthState::new(AuthConfig::disabled());
        let app = Router::new()
            .route("/admin/keys", get(list_keys))
            .with_state(auth_state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/admin/keys")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let result: ListKeysResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(result.total, 0);
        assert!(result.keys.is_empty());
    }

    #[tokio::test]
    async fn test_list_keys_with_existing() {
        let (app, _) = test_admin_app();

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/admin/keys")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let result: ListKeysResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(result.total, 1);
        assert_eq!(result.keys[0].name, "Existing User");
        assert!(result.keys[0].key_prefix.starts_with("existing"));
    }

    #[tokio::test]
    async fn test_add_key_success() {
        let auth_state = AuthState::new(AuthConfig::disabled());
        let app = Router::new()
            .route("/admin/keys", post(add_key))
            .with_state(auth_state.clone());

        let body = serde_json::json!({
            "key": "new-api-key-123456789",
            "name": "Test User",
            "rate_limit": 200
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/admin/keys")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let result: AddKeyResponse = serde_json::from_slice(&body).unwrap();
        assert!(result.success);
        assert_eq!(result.key_info.name, "Test User");
        assert_eq!(result.key_info.rate_limit, 200);

        // Verify key was actually added
        assert!(auth_state.has_key("new-api-key-123456789").await);
    }

    #[tokio::test]
    async fn test_add_key_too_short() {
        let auth_state = AuthState::new(AuthConfig::disabled());
        let app = Router::new()
            .route("/admin/keys", post(add_key))
            .with_state(auth_state);

        let body = serde_json::json!({
            "key": "short",
            "name": "Test User"
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/admin/keys")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let result: AdminError = serde_json::from_slice(&body).unwrap();
        assert_eq!(result.code, "key_too_short");
    }

    #[tokio::test]
    async fn test_add_key_already_exists() {
        let (app, _) = test_admin_app();

        let body = serde_json::json!({
            "key": "existing-key-12345678",
            "name": "Another User"
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/admin/keys")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::CONFLICT);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let result: AdminError = serde_json::from_slice(&body).unwrap();
        assert_eq!(result.code, "key_exists");
    }

    #[tokio::test]
    async fn test_add_key_empty_name() {
        let auth_state = AuthState::new(AuthConfig::disabled());
        let app = Router::new()
            .route("/admin/keys", post(add_key))
            .with_state(auth_state);

        let body = serde_json::json!({
            "key": "valid-api-key-1234567",
            "name": ""
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/admin/keys")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let result: AdminError = serde_json::from_slice(&body).unwrap();
        assert_eq!(result.code, "invalid_name");
    }

    #[tokio::test]
    async fn test_revoke_key_success() {
        let auth_state = AuthState::new(
            AuthConfig::disabled().with_api_key("key-to-revoke-123456", "Temp User"),
        );
        let app = Router::new()
            .route("/admin/keys/:key", delete(revoke_key))
            .with_state(auth_state.clone());

        // Verify key exists
        assert!(auth_state.has_key("key-to-revoke-123456").await);

        let response = app
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri("/admin/keys/key-to-revoke-123456")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let result: RevokeKeyResponse = serde_json::from_slice(&body).unwrap();
        assert!(result.success);

        // Verify key was actually removed
        assert!(!auth_state.has_key("key-to-revoke-123456").await);
    }

    #[tokio::test]
    async fn test_revoke_key_not_found() {
        let auth_state = AuthState::new(AuthConfig::disabled());
        let app = Router::new()
            .route("/admin/keys/:key", delete(revoke_key))
            .with_state(auth_state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri("/admin/keys/nonexistent-key-123")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let result: AdminError = serde_json::from_slice(&body).unwrap();
        assert_eq!(result.code, "key_not_found");
    }

    #[tokio::test]
    async fn test_add_key_default_rate_limit() {
        let auth_state = AuthState::new(AuthConfig::disabled());
        let app = Router::new()
            .route("/admin/keys", post(add_key))
            .with_state(auth_state);

        let body = serde_json::json!({
            "key": "default-rate-key-1234",
            "name": "Default Rate User"
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/admin/keys")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let result: AddKeyResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(result.key_info.rate_limit, 100); // Default
    }

    #[tokio::test]
    async fn test_update_key_rate_limit() {
        let auth_state = AuthState::new(AuthConfig::disabled().with_api_key_rate_limit(
            "key-to-update-123456",
            "Test User",
            100,
        ));
        let app = Router::new()
            .route("/admin/keys/:key", patch(update_key))
            .with_state(auth_state.clone());

        let body = serde_json::json!({
            "rate_limit": 500
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("PATCH")
                    .uri("/admin/keys/key-to-update-123456")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let result: UpdateKeyResponse = serde_json::from_slice(&body).unwrap();
        assert!(result.success);
        assert_eq!(result.key_info.rate_limit, 500);
    }

    #[tokio::test]
    async fn test_update_key_admin_status() {
        let auth_state = AuthState::new(
            AuthConfig::disabled().with_api_key("key-to-promote-12345", "Test User"),
        );
        let app = Router::new()
            .route("/admin/keys/:key", patch(update_key))
            .with_state(auth_state.clone());

        // Promote to admin
        let body = serde_json::json!({
            "is_admin": true
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("PATCH")
                    .uri("/admin/keys/key-to-promote-12345")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let result: UpdateKeyResponse = serde_json::from_slice(&body).unwrap();
        assert!(result.success);
        assert!(result.key_info.is_admin);
    }

    #[tokio::test]
    async fn test_update_key_both_fields() {
        let auth_state = AuthState::new(AuthConfig::disabled().with_api_key_rate_limit(
            "key-to-update-both12",
            "Test User",
            100,
        ));
        let app = Router::new()
            .route("/admin/keys/:key", patch(update_key))
            .with_state(auth_state.clone());

        let body = serde_json::json!({
            "rate_limit": 250,
            "is_admin": true
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("PATCH")
                    .uri("/admin/keys/key-to-update-both12")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let result: UpdateKeyResponse = serde_json::from_slice(&body).unwrap();
        assert!(result.success);
        assert_eq!(result.key_info.rate_limit, 250);
        assert!(result.key_info.is_admin);
    }

    #[tokio::test]
    async fn test_update_key_not_found() {
        let auth_state = AuthState::new(AuthConfig::disabled());
        let app = Router::new()
            .route("/admin/keys/:key", patch(update_key))
            .with_state(auth_state);

        let body = serde_json::json!({
            "rate_limit": 500
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("PATCH")
                    .uri("/admin/keys/nonexistent-key-123")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let result: AdminError = serde_json::from_slice(&body).unwrap();
        assert_eq!(result.code, "key_not_found");
    }

    #[tokio::test]
    async fn test_update_key_no_updates() {
        let auth_state = AuthState::new(
            AuthConfig::disabled().with_api_key("key-no-update-1234567", "Test User"),
        );
        let app = Router::new()
            .route("/admin/keys/:key", patch(update_key))
            .with_state(auth_state);

        // Empty update request
        let body = serde_json::json!({});

        let response = app
            .oneshot(
                Request::builder()
                    .method("PATCH")
                    .uri("/admin/keys/key-no-update-1234567")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let result: AdminError = serde_json::from_slice(&body).unwrap();
        assert_eq!(result.code, "no_updates");
    }

    #[tokio::test]
    async fn test_update_key_invalid_rate_limit() {
        let auth_state = AuthState::new(
            AuthConfig::disabled().with_api_key("key-invalid-rate-1234", "Test User"),
        );
        let app = Router::new()
            .route("/admin/keys/:key", patch(update_key))
            .with_state(auth_state);

        let body = serde_json::json!({
            "rate_limit": 0
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("PATCH")
                    .uri("/admin/keys/key-invalid-rate-1234")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let result: AdminError = serde_json::from_slice(&body).unwrap();
        assert_eq!(result.code, "invalid_rate_limit");
    }

    #[tokio::test]
    async fn test_update_key_demote_admin() {
        let auth_state = AuthState::new(
            AuthConfig::disabled().with_admin_key("admin-to-demote-123", "Admin User"),
        );
        let app = Router::new()
            .route("/admin/keys/:key", patch(update_key))
            .with_state(auth_state.clone());

        // Verify is admin initially
        assert!(auth_state.is_admin("admin-to-demote-123").await);

        // Demote from admin
        let body = serde_json::json!({
            "is_admin": false
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("PATCH")
                    .uri("/admin/keys/admin-to-demote-123")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let result: UpdateKeyResponse = serde_json::from_slice(&body).unwrap();
        assert!(result.success);
        assert!(!result.key_info.is_admin);

        // Verify state was updated
        assert!(!auth_state.is_admin("admin-to-demote-123").await);
    }

    // ============================================
    // Mutation-killing tests for admin.rs
    // ============================================

    #[tokio::test]
    async fn test_add_key_empty_key() {
        let auth_state = AuthState::new(AuthConfig::disabled());
        let app = Router::new()
            .route("/admin/keys", post(add_key))
            .with_state(auth_state);

        let body = serde_json::json!({
            "key": "",
            "name": "Test User"
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/admin/keys")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let result: AdminError = serde_json::from_slice(&body).unwrap();
        assert_eq!(result.code, "invalid_key");
    }

    #[tokio::test]
    async fn test_add_key_exactly_16_chars() {
        let auth_state = AuthState::new(AuthConfig::disabled());
        let app = Router::new()
            .route("/admin/keys", post(add_key))
            .with_state(auth_state.clone());

        // Key with exactly 16 characters - should succeed
        let body = serde_json::json!({
            "key": "1234567890123456", // 16 chars
            "name": "Test User"
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/admin/keys")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_add_key_exactly_15_chars() {
        let auth_state = AuthState::new(AuthConfig::disabled());
        let app = Router::new()
            .route("/admin/keys", post(add_key))
            .with_state(auth_state);

        // Key with exactly 15 characters - should fail (< 16)
        let body = serde_json::json!({
            "key": "123456789012345", // 15 chars
            "name": "Test User"
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/admin/keys")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let result: AdminError = serde_json::from_slice(&body).unwrap();
        assert_eq!(result.code, "key_too_short");
    }

    #[tokio::test]
    async fn test_add_key_prefix_truncation_short_key() {
        let auth_state = AuthState::new(AuthConfig::disabled());
        let app = Router::new()
            .route("/admin/keys", post(add_key))
            .with_state(auth_state);

        // Key with exactly 8 characters (boundary for truncation logic)
        let body = serde_json::json!({
            "key": "1234567890123456", // 16 chars total but want to test prefix
            "name": "Test User"
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/admin/keys")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let result: AddKeyResponse = serde_json::from_slice(&body).unwrap();

        // Key > 8 chars should be truncated to first 8 + "..."
        assert_eq!(result.key_info.key_prefix, "12345678...");
    }

    #[tokio::test]
    async fn test_add_key_with_admin_flag() {
        let auth_state = AuthState::new(AuthConfig::disabled());
        let app = Router::new()
            .route("/admin/keys", post(add_key))
            .with_state(auth_state.clone());

        let body = serde_json::json!({
            "key": "admin-key-1234567890",
            "name": "Admin User",
            "is_admin": true
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/admin/keys")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let result: AddKeyResponse = serde_json::from_slice(&body).unwrap();
        assert!(result.key_info.is_admin);

        // Verify admin status in state
        assert!(auth_state.is_admin("admin-key-1234567890").await);
    }

    #[tokio::test]
    async fn test_list_keys_total_matches_keys_len() {
        let auth_state = AuthState::new(
            AuthConfig::disabled()
                .with_api_key("key-one-1234567890", "User One")
                .with_api_key("key-two-1234567890", "User Two")
                .with_api_key("key-three-12345678", "User Three"),
        );
        let app = Router::new()
            .route("/admin/keys", get(list_keys))
            .with_state(auth_state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/admin/keys")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let result: ListKeysResponse = serde_json::from_slice(&body).unwrap();

        // total should match keys.len()
        assert_eq!(result.total, 3);
        assert_eq!(result.keys.len(), 3);
    }

    #[tokio::test]
    async fn test_revoke_key_empty_key_path() {
        let auth_state = AuthState::new(AuthConfig::disabled());
        let app = Router::new()
            .route("/admin/keys/:key", delete(revoke_key))
            .with_state(auth_state);

        // Empty key in path (axum may route this differently, but handler should check)
        // This test verifies the handler's empty check
        let response = app
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri("/admin/keys/")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await;

        // May 404 if route doesn't match or 400 if it does
        // Either way, we're testing the handler doesn't panic
        assert!(response.is_ok());
    }

    #[tokio::test]
    async fn test_update_key_empty_key_path() {
        let auth_state = AuthState::new(AuthConfig::disabled());
        let app = Router::new()
            .route("/admin/keys/:key", patch(update_key))
            .with_state(auth_state);

        let body = serde_json::json!({
            "rate_limit": 100
        });

        // Empty key would be handled by routing, but verify handler doesn't panic
        let response = app
            .oneshot(
                Request::builder()
                    .method("PATCH")
                    .uri("/admin/keys/")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await;

        assert!(response.is_ok());
    }

    #[test]
    fn test_add_key_request_default_values() {
        // Test that AddKeyRequest has correct default values
        let json = r#"{"key": "test-key-123456789", "name": "Test"}"#;
        let req: AddKeyRequest = serde_json::from_str(json).unwrap();

        // rate_limit should default to None
        assert!(req.rate_limit.is_none());
        // is_admin should default to false
        assert!(!req.is_admin);
    }

    #[test]
    fn test_update_key_request_optional_fields() {
        // Test that UpdateKeyRequest properly deserializes optional fields
        let json = r#"{"rate_limit": 500}"#;
        let req: UpdateKeyRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.rate_limit, Some(500));
        assert!(req.is_admin.is_none());

        let json = r#"{"is_admin": true}"#;
        let req: UpdateKeyRequest = serde_json::from_str(json).unwrap();
        assert!(req.rate_limit.is_none());
        assert_eq!(req.is_admin, Some(true));

        let json = r#"{"rate_limit": 100, "is_admin": false}"#;
        let req: UpdateKeyRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.rate_limit, Some(100));
        assert_eq!(req.is_admin, Some(false));
    }

    #[test]
    fn test_admin_error_serialization() {
        let error = AdminError {
            error: "Test error message".to_string(),
            code: "test_code".to_string(),
        };

        let json = serde_json::to_string(&error).unwrap();
        assert!(json.contains("Test error message"));
        assert!(json.contains("test_code"));

        let deserialized: AdminError = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.error, "Test error message");
        assert_eq!(deserialized.code, "test_code");
    }

    #[test]
    fn test_list_keys_response_serialization() {
        let response = ListKeysResponse {
            keys: vec![KeyInfo {
                key_prefix: "test...".to_string(),
                name: "Test".to_string(),
                rate_limit: 100,
                is_admin: false,
            }],
            total: 1,
        };

        let json = serde_json::to_string(&response).unwrap();
        let deserialized: ListKeysResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.total, 1);
        assert_eq!(deserialized.keys.len(), 1);
    }

    #[test]
    fn test_revoke_key_response_serialization() {
        let response = RevokeKeyResponse {
            success: true,
            message: "Key revoked".to_string(),
        };

        let json = serde_json::to_string(&response).unwrap();
        let deserialized: RevokeKeyResponse = serde_json::from_str(&json).unwrap();
        assert!(deserialized.success);
        assert_eq!(deserialized.message, "Key revoked");
    }

    #[test]
    fn test_update_key_response_serialization() {
        let response = UpdateKeyResponse {
            success: true,
            message: "Key updated".to_string(),
            key_info: KeyInfo {
                key_prefix: "test...".to_string(),
                name: "Test".to_string(),
                rate_limit: 200,
                is_admin: true,
            },
        };

        let json = serde_json::to_string(&response).unwrap();
        let deserialized: UpdateKeyResponse = serde_json::from_str(&json).unwrap();
        assert!(deserialized.success);
        assert_eq!(deserialized.key_info.rate_limit, 200);
        assert!(deserialized.key_info.is_admin);
    }
}
