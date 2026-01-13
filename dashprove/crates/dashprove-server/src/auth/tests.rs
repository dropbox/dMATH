//! Tests for authentication module

use super::*;

#[test]
fn test_auth_config_default() {
    let config = AuthConfig::default();
    assert!(!config.required);
    assert!(config.api_keys.is_empty());
    assert_eq!(config.anonymous_rate_limit, 10);
}

#[test]
fn test_auth_config_with_api_key() {
    let config = AuthConfig::default()
        .with_api_key("test-key-123", "Test User")
        .with_api_key_rate_limit("premium-key", "Premium User", 1000);

    assert_eq!(config.api_keys.len(), 2);
    assert_eq!(config.api_keys["test-key-123"].name, "Test User");
    assert_eq!(config.api_keys["test-key-123"].rate_limit, 100);
    assert_eq!(config.api_keys["premium-key"].rate_limit, 1000);
}

#[test]
fn test_auth_config_required() {
    let config = AuthConfig::required();
    assert!(config.required);
}

#[tokio::test]
async fn test_rate_limiter_allows_under_limit() {
    let limiter = RateLimiter::new();

    // First request should succeed
    let result = limiter.check("test-key", 10).await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 1);

    // Second request should also succeed
    let result = limiter.check("test-key", 10).await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 2);
}

#[tokio::test]
async fn test_rate_limiter_blocks_over_limit() {
    let limiter = RateLimiter::new();

    // Use up the limit
    for i in 1..=5 {
        let result = limiter.check("test-key", 5).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), i);
    }

    // Next request should be blocked
    let result = limiter.check("test-key", 5).await;
    assert!(result.is_err());
    let retry_after = result.unwrap_err();
    assert!(retry_after > 0);
    assert!(retry_after <= 60);
}

#[tokio::test]
async fn test_rate_limiter_separate_keys() {
    let limiter = RateLimiter::new();

    // Key 1 uses its limit
    for _ in 0..5 {
        limiter.check("key1", 5).await.unwrap();
    }

    // Key 1 should be blocked
    assert!(limiter.check("key1", 5).await.is_err());

    // Key 2 should still work
    let result = limiter.check("key2", 5).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_rate_limiter_current_count() {
    let limiter = RateLimiter::new();

    assert_eq!(limiter.current_count("test-key").await, 0);

    limiter.check("test-key", 10).await.unwrap();
    limiter.check("test-key", 10).await.unwrap();

    assert_eq!(limiter.current_count("test-key").await, 2);
}

#[tokio::test]
async fn test_rate_limiter_cleanup() {
    let limiter = RateLimiter::new();

    limiter.check("test-key", 10).await.unwrap();

    // Cleanup shouldn't remove active buckets
    limiter.cleanup().await;

    assert_eq!(limiter.current_count("test-key").await, 1);
}

#[tokio::test]
async fn test_auth_state_disabled() {
    let state = AuthState::disabled();
    let config = state.config.read().await;
    assert!(!config.required);
}

#[test]
fn test_auth_error_serialization() {
    let error = AuthError {
        error: "Rate limit exceeded".to_string(),
        code: "rate_limited".to_string(),
        retry_after: Some(30),
    };

    let json = serde_json::to_string(&error).unwrap();
    assert!(json.contains("rate_limited"));
    assert!(json.contains("30"));
}

#[test]
fn test_auth_error_serialization_no_retry() {
    let error = AuthError {
        error: "Invalid API key".to_string(),
        code: "invalid_api_key".to_string(),
        retry_after: None,
    };

    let json = serde_json::to_string(&error).unwrap();
    assert!(!json.contains("retry_after"));
}

mod integration_tests {
    use super::*;
    use axum::{
        body::Body,
        http::{Request as HttpRequest, StatusCode},
        middleware,
        routing::get,
        Router,
    };
    use tower::ServiceExt;

    async fn test_handler() -> &'static str {
        "OK"
    }

    fn test_app(auth_config: AuthConfig) -> Router {
        let auth_state = AuthState::new(auth_config);
        Router::new()
            .route("/health", get(test_handler))
            .route("/test", get(test_handler))
            .layer(middleware::from_fn_with_state(auth_state, auth_middleware))
    }

    #[tokio::test]
    async fn test_health_bypasses_auth() {
        let app = test_app(AuthConfig::required());
        let response = app
            .oneshot(
                HttpRequest::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_auth_not_required_allows_anonymous() {
        let app = test_app(AuthConfig::disabled().with_anonymous_rate_limit(100));
        let response = app
            .oneshot(
                HttpRequest::builder()
                    .uri("/test")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_auth_required_rejects_anonymous() {
        let app = test_app(AuthConfig::required());
        let response = app
            .oneshot(
                HttpRequest::builder()
                    .uri("/test")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let error: AuthError = serde_json::from_slice(&body).unwrap();
        assert_eq!(error.code, "missing_api_key");
    }

    #[tokio::test]
    async fn test_auth_required_accepts_valid_key() {
        let config = AuthConfig::required().with_api_key("valid-key", "Test User");
        let app = test_app(config);
        let response = app
            .oneshot(
                HttpRequest::builder()
                    .uri("/test")
                    .header("X-API-Key", "valid-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_auth_required_rejects_invalid_key() {
        let config = AuthConfig::required().with_api_key("valid-key", "Test User");
        let app = test_app(config);
        let response = app
            .oneshot(
                HttpRequest::builder()
                    .uri("/test")
                    .header("X-API-Key", "invalid-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let error: AuthError = serde_json::from_slice(&body).unwrap();
        assert_eq!(error.code, "invalid_api_key");
    }

    #[tokio::test]
    async fn test_auth_bearer_token() {
        let config = AuthConfig::required().with_api_key("bearer-token-123", "Test User");
        let app = test_app(config);
        let response = app
            .oneshot(
                HttpRequest::builder()
                    .uri("/test")
                    .header("Authorization", "Bearer bearer-token-123")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_rate_limit_headers_present() {
        let config = AuthConfig::disabled().with_anonymous_rate_limit(100);
        let app = test_app(config);
        let response = app
            .oneshot(
                HttpRequest::builder()
                    .uri("/test")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        // Check rate limit headers
        assert!(response.headers().contains_key("X-RateLimit-Limit"));
        assert!(response.headers().contains_key("X-RateLimit-Remaining"));
        assert_eq!(response.headers().get("X-RateLimit-Limit").unwrap(), "100");
        assert_eq!(
            response.headers().get("X-RateLimit-Remaining").unwrap(),
            "99"
        );
    }

    #[tokio::test]
    async fn test_rate_limit_exceeded() {
        let config = AuthConfig::disabled().with_anonymous_rate_limit(2);
        let auth_state = AuthState::new(config);

        // Make requests until rate limited
        // First request
        let app =
            Router::new()
                .route("/test", get(test_handler))
                .layer(middleware::from_fn_with_state(
                    auth_state.clone(),
                    auth_middleware,
                ));
        let response = app
            .oneshot(
                HttpRequest::builder()
                    .uri("/test")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        // Second request
        let app =
            Router::new()
                .route("/test", get(test_handler))
                .layer(middleware::from_fn_with_state(
                    auth_state.clone(),
                    auth_middleware,
                ));
        let response = app
            .oneshot(
                HttpRequest::builder()
                    .uri("/test")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        // Third request should be rate limited
        let app =
            Router::new()
                .route("/test", get(test_handler))
                .layer(middleware::from_fn_with_state(
                    auth_state.clone(),
                    auth_middleware,
                ));
        let response = app
            .oneshot(
                HttpRequest::builder()
                    .uri("/test")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);

        // Check Retry-After header
        assert!(response.headers().contains_key("retry-after"));

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let error: AuthError = serde_json::from_slice(&body).unwrap();
        assert_eq!(error.code, "rate_limited");
        assert!(error.retry_after.is_some());
    }

    #[tokio::test]
    async fn test_different_api_keys_have_separate_limits() {
        let config = AuthConfig::disabled()
            .with_api_key_rate_limit("key1", "User 1", 1)
            .with_api_key_rate_limit("key2", "User 2", 1);
        let auth_state = AuthState::new(config);

        // Key1 uses its limit
        let app =
            Router::new()
                .route("/test", get(test_handler))
                .layer(middleware::from_fn_with_state(
                    auth_state.clone(),
                    auth_middleware,
                ));
        let response = app
            .oneshot(
                HttpRequest::builder()
                    .uri("/test")
                    .header("X-API-Key", "key1")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        // Key1 is now rate limited
        let app =
            Router::new()
                .route("/test", get(test_handler))
                .layer(middleware::from_fn_with_state(
                    auth_state.clone(),
                    auth_middleware,
                ));
        let response = app
            .oneshot(
                HttpRequest::builder()
                    .uri("/test")
                    .header("X-API-Key", "key1")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);

        // But key2 still works
        let app =
            Router::new()
                .route("/test", get(test_handler))
                .layer(middleware::from_fn_with_state(
                    auth_state.clone(),
                    auth_middleware,
                ));
        let response = app
            .oneshot(
                HttpRequest::builder()
                    .uri("/test")
                    .header("X-API-Key", "key2")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[test]
    fn test_admin_key_config() {
        let config = AuthConfig::default()
            .with_admin_key("admin-key-123", "Admin User")
            .with_api_key("user-key-456", "Regular User");

        assert_eq!(config.api_keys.len(), 2);
        assert!(config.api_keys["admin-key-123"].is_admin);
        assert!(!config.api_keys["user-key-456"].is_admin);
    }

    #[test]
    fn test_admin_key_rate_limit_config() {
        let config =
            AuthConfig::default().with_admin_key_rate_limit("admin-key-123", "Admin User", 500);

        assert!(config.api_keys["admin-key-123"].is_admin);
        assert_eq!(config.api_keys["admin-key-123"].rate_limit, 500);
    }

    #[tokio::test]
    async fn test_auth_state_is_admin() {
        let state = AuthState::new(
            AuthConfig::default()
                .with_admin_key("admin-key-1234567890", "Admin User")
                .with_api_key("user-key-1234567890", "Regular User"),
        );

        assert!(state.is_admin("admin-key-1234567890").await);
        assert!(!state.is_admin("user-key-1234567890").await);
        assert!(!state.is_admin("nonexistent-key").await);
    }

    #[tokio::test]
    async fn test_admin_middleware_allows_admin() {
        // Create auth state with admin key
        let auth_state =
            AuthState::new(AuthConfig::required().with_admin_key("admin-key-1234567890", "Admin"));

        // Create app with both auth and admin middleware
        let app = Router::new()
            .route("/admin", get(test_handler))
            .layer(middleware::from_fn(admin_middleware))
            .layer(middleware::from_fn_with_state(auth_state, auth_middleware));

        let response = app
            .oneshot(
                HttpRequest::builder()
                    .uri("/admin")
                    .header("X-API-Key", "admin-key-1234567890")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_admin_middleware_rejects_non_admin() {
        // Create auth state with regular key
        let auth_state =
            AuthState::new(AuthConfig::required().with_api_key("user-key-1234567890", "User"));

        // Create app with both auth and admin middleware
        let app = Router::new()
            .route("/admin", get(test_handler))
            .layer(middleware::from_fn(admin_middleware))
            .layer(middleware::from_fn_with_state(auth_state, auth_middleware));

        let response = app
            .oneshot(
                HttpRequest::builder()
                    .uri("/admin")
                    .header("X-API-Key", "user-key-1234567890")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::FORBIDDEN);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let error: AuthError = serde_json::from_slice(&body).unwrap();
        assert_eq!(error.code, "admin_required");
    }

    #[tokio::test]
    async fn test_persistence_save_and_load() {
        let temp_dir = std::env::temp_dir();
        let keys_file = temp_dir.join(format!("dashprove_test_keys_{}.json", std::process::id()));

        // Create auth state with persistence
        let state = AuthState::load_from_file(
            AuthConfig::disabled().with_api_key("initial-key-1234567", "Initial"),
            &keys_file,
        );

        // Add a key (should persist)
        state
            .add_key("added-key-12345678", "Added User", Some(200), false)
            .await;

        // Verify key was added
        assert!(state.has_key("added-key-12345678").await);

        // Create new state from same file
        let state2 = AuthState::load_from_file(AuthConfig::disabled(), &keys_file);

        // Should have both keys (initial from config merged, added from persistence)
        assert!(state2.has_key("initial-key-1234567").await);
        assert!(state2.has_key("added-key-12345678").await);

        // Cleanup
        let _ = std::fs::remove_file(&keys_file);
    }

    #[tokio::test]
    async fn test_persistence_remove_key() {
        let temp_dir = std::env::temp_dir();
        let keys_file = temp_dir.join(format!(
            "dashprove_test_keys_remove_{}.json",
            std::process::id()
        ));

        // Create auth state with persistence and add keys
        let state = AuthState::load_from_file(
            AuthConfig::disabled()
                .with_api_key("key1-1234567890123", "User 1")
                .with_api_key("key2-1234567890123", "User 2"),
            &keys_file,
        );

        // Remove a key (should persist)
        let removed = state.remove_key("key1-1234567890123").await;
        assert!(removed);

        // Create new state from same file
        let state2 = AuthState::load_from_file(AuthConfig::disabled(), &keys_file);

        // Should only have key2
        assert!(!state2.has_key("key1-1234567890123").await);
        assert!(state2.has_key("key2-1234567890123").await);

        // Cleanup
        let _ = std::fs::remove_file(&keys_file);
    }

    #[tokio::test]
    async fn test_update_key_rate_limit() {
        let state = AuthState::new(AuthConfig::disabled().with_api_key_rate_limit(
            "test-key-1234567890",
            "Test User",
            100,
        ));

        // Update rate limit
        let updated = state
            .update_key("test-key-1234567890", Some(500), None)
            .await;
        assert!(updated);

        // Verify rate limit was updated
        let key_info = state.get_key_info("test-key-1234567890").await.unwrap();
        assert_eq!(key_info.rate_limit, 500);
        assert!(!key_info.is_admin); // Should remain unchanged
    }

    #[tokio::test]
    async fn test_update_key_admin_status() {
        let state =
            AuthState::new(AuthConfig::disabled().with_api_key("test-key-1234567890", "Test User"));

        // Promote to admin
        let updated = state
            .update_key("test-key-1234567890", None, Some(true))
            .await;
        assert!(updated);

        // Verify admin status was updated
        assert!(state.is_admin("test-key-1234567890").await);

        // Demote from admin
        let updated = state
            .update_key("test-key-1234567890", None, Some(false))
            .await;
        assert!(updated);

        // Verify admin status was updated
        assert!(!state.is_admin("test-key-1234567890").await);
    }

    #[tokio::test]
    async fn test_update_key_both_fields() {
        let state = AuthState::new(AuthConfig::disabled().with_api_key_rate_limit(
            "test-key-1234567890",
            "Test User",
            100,
        ));

        // Update both rate limit and admin status
        let updated = state
            .update_key("test-key-1234567890", Some(250), Some(true))
            .await;
        assert!(updated);

        // Verify both were updated
        let key_info = state.get_key_info("test-key-1234567890").await.unwrap();
        assert_eq!(key_info.rate_limit, 250);
        assert!(key_info.is_admin);
    }

    #[tokio::test]
    async fn test_update_key_not_found() {
        let state = AuthState::new(AuthConfig::disabled());

        // Try to update non-existent key
        let updated = state
            .update_key("nonexistent-key-123", Some(500), None)
            .await;
        assert!(!updated);
    }

    #[tokio::test]
    async fn test_get_key_info() {
        let state = AuthState::new(
            AuthConfig::disabled()
                .with_api_key_rate_limit("test-key-1234567890", "Test User", 200)
                .with_admin_key("admin-key-123456789", "Admin User"),
        );

        // Get regular key info
        let key_info = state.get_key_info("test-key-1234567890").await;
        assert!(key_info.is_some());
        let info = key_info.unwrap();
        assert_eq!(info.name, "Test User");
        assert_eq!(info.rate_limit, 200);
        assert!(!info.is_admin);
        assert!(info.key_prefix.starts_with("test-key"));

        // Get admin key info
        let admin_info = state.get_key_info("admin-key-123456789").await;
        assert!(admin_info.is_some());
        let info = admin_info.unwrap();
        assert!(info.is_admin);

        // Get non-existent key
        let no_info = state.get_key_info("nonexistent-key").await;
        assert!(no_info.is_none());
    }

    #[tokio::test]
    async fn test_update_key_persists() {
        let temp_dir = std::env::temp_dir();
        let keys_file = temp_dir.join(format!(
            "dashprove_test_keys_update_{}.json",
            std::process::id()
        ));

        // Create auth state with persistence
        let state = AuthState::load_from_file(
            AuthConfig::disabled().with_api_key_rate_limit("key-to-update-12345", "User", 100),
            &keys_file,
        );

        // Update the key
        let updated = state
            .update_key("key-to-update-12345", Some(300), Some(true))
            .await;
        assert!(updated);

        // Create new state from same file
        let state2 = AuthState::load_from_file(AuthConfig::disabled(), &keys_file);

        // Verify updates were persisted
        let key_info = state2.get_key_info("key-to-update-12345").await.unwrap();
        assert_eq!(key_info.rate_limit, 300);
        assert!(key_info.is_admin);

        // Cleanup
        let _ = std::fs::remove_file(&keys_file);
    }
}
