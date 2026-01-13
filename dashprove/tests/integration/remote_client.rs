//! Integration tests for the remote HTTP client
//!
//! These tests verify the remote client can communicate with a test server.
//! Note: Tests that require a running server are marked with #[ignore].
//!
//! The remote module is available because dashprove is in dev-dependencies
//! with features = ["remote"] in the workspace Cargo.toml.

use dashprove::remote::{
    BackendIdParam, Change, DashFlowClient, HealthResponse, RemoteClient, RemoteConfig,
    VerifyRequest, VerifyResponse,
};

/// Test client creation with various configs
#[test]
fn test_client_creation_variants() {
    let config1 = RemoteConfig::default();
    let client1 = RemoteClient::new(config1);
    assert_eq!(client1.base_url(), "http://localhost:3000");

    let config2 = RemoteConfig::local(8080)
        .with_api_key("test-key")
        .with_timeout(60);
    let client2 = RemoteClient::new(config2);
    assert_eq!(client2.base_url(), "http://localhost:8080");

    let _dashflow = DashFlowClient::local(3000);
}

/// Test request serialization for verify endpoint
#[test]
fn test_verify_request_formats() {
    let req = VerifyRequest {
        spec: "theorem test { true }".to_string(),
        backend: None,
        use_ml: false,
        ml_min_confidence: 0.5,
    };
    let json = serde_json::to_string(&req).unwrap();
    assert!(json.contains("theorem test"));
    assert!(!json.contains("backend")); // None should be skipped
    assert!(!json.contains("use_ml")); // false should be skipped
    assert!(!json.contains("ml_min_confidence")); // default should be skipped

    let req_with_backend = VerifyRequest {
        spec: "theorem test { true }".to_string(),
        backend: Some(BackendIdParam::Lean4),
        use_ml: false,
        ml_min_confidence: 0.5,
    };
    let json2 = serde_json::to_string(&req_with_backend).unwrap();
    assert!(json2.contains("lean4"));

    // Test with ML enabled
    let req_with_ml = VerifyRequest {
        spec: "theorem test { true }".to_string(),
        backend: None,
        use_ml: true,
        ml_min_confidence: 0.7,
    };
    let json3 = serde_json::to_string(&req_with_ml).unwrap();
    assert!(json3.contains("use_ml"));
    assert!(json3.contains("0.7"));
}

/// Test change serialization for incremental verify
#[test]
fn test_change_serialization() {
    let changes = vec![
        Change {
            change_type: "modified".to_string(),
            name: "test_theorem".to_string(),
        },
        Change {
            change_type: "added".to_string(),
            name: "new_invariant".to_string(),
        },
    ];

    let json = serde_json::to_string(&changes).unwrap();
    assert!(json.contains("modified"));
    assert!(json.contains("test_theorem"));
    assert!(json.contains("added"));
    assert!(json.contains("new_invariant"));
}

/// Test parsing health response
#[test]
fn test_parse_health_response() {
    let json = r#"{
        "status": "healthy",
        "ready": true,
        "shutdown_state": "running",
        "in_flight_requests": 0,
        "active_sessions": 0
    }"#;

    let resp: HealthResponse = serde_json::from_str(json).unwrap();
    assert_eq!(resp.status, "healthy");
    assert!(resp.ready);
    assert_eq!(resp.shutdown_state, "running");
}

/// Test parsing draining health response
#[test]
fn test_parse_draining_health_response() {
    let json = r#"{
        "status": "draining",
        "ready": false,
        "shutdown_state": "draining",
        "in_flight_requests": 5,
        "active_sessions": 2
    }"#;

    let resp: HealthResponse = serde_json::from_str(json).unwrap();
    assert_eq!(resp.status, "draining");
    assert!(!resp.ready);
    assert_eq!(resp.in_flight_requests, 5);
}

/// Test parsing verify response with compilations
#[test]
fn test_parse_verify_response_with_compilations() {
    let json = r#"{
        "valid": true,
        "property_count": 2,
        "compilations": [
            {
                "backend": "lean4",
                "code": "theorem test : True := trivial"
            },
            {
                "backend": "tlaplus",
                "code": "MODULE Test\nVARIABLES x"
            }
        ],
        "errors": []
    }"#;

    let resp: VerifyResponse = serde_json::from_str(json).unwrap();
    assert!(resp.valid);
    assert_eq!(resp.property_count, 2);
    assert_eq!(resp.compilations.len(), 2);
    assert!(resp.compilations[0].code.contains("theorem"));
    assert!(resp.compilations[1].code.contains("MODULE"));
}

/// Test parsing verify response with errors
#[test]
fn test_parse_verify_response_with_errors() {
    let json = r#"{
        "valid": false,
        "property_count": 0,
        "compilations": [],
        "errors": ["Parse error at line 1: unexpected token", "Type error: unknown type"]
    }"#;

    let resp: VerifyResponse = serde_json::from_str(json).unwrap();
    assert!(!resp.valid);
    assert_eq!(resp.errors.len(), 2);
    assert!(resp.errors[0].contains("Parse error"));
}

/// Test connection failure handling (no server needed)
#[tokio::test]
async fn test_connection_failure() {
    // Connect to a port where nothing is running
    let client = RemoteClient::local(59999);

    let result = client.health().await;

    // Should fail with HTTP error
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, dashprove::remote::RemoteError::Http(_)));
}

// ============ Live Server Tests ============
// These tests require a running DashProve server.
// Run: cargo run -p dashprove-server -- --port 3000
// Then: cargo test --test integration -- --ignored

/// Test against a live server (requires server to be running)
#[tokio::test]
#[ignore = "requires running server: cargo run -p dashprove-server -- --port 3000"]
async fn test_live_server_health() {
    let client = RemoteClient::local(3000);
    let health = client.health().await.expect("health check should succeed");
    assert_eq!(health.status, "healthy");
    assert!(health.ready);
}

/// Test verify endpoint against live server
#[tokio::test]
#[ignore = "requires running server"]
async fn test_live_server_verify() {
    let client = RemoteClient::local(3000);

    let result = client
        .verify("theorem test { true }")
        .await
        .expect("verify should succeed");

    assert!(result.valid);
    assert!(result.errors.is_empty());
}

/// Test verify with specific backend
#[tokio::test]
#[ignore = "requires running server"]
async fn test_live_server_verify_with_backend() {
    let client = RemoteClient::local(3000);

    let result = client
        .verify_with_backend("theorem test { true }", Some(BackendIdParam::Lean4))
        .await
        .expect("verify should succeed");

    assert!(result.valid);
}

/// Test backends endpoint
#[tokio::test]
#[ignore = "requires running server"]
async fn test_live_server_backends() {
    let client = RemoteClient::local(3000);

    let backends = client.backends().await.expect("backends should succeed");
    assert!(!backends.backends.is_empty());
}

/// Test corpus search
#[tokio::test]
#[ignore = "requires running server"]
async fn test_live_server_corpus_search() {
    let client = RemoteClient::local(3000);

    let result = client
        .corpus_search("forall x", 5)
        .await
        .expect("corpus search should succeed");

    // Result may be empty if corpus is empty, but call should not error
    // (total_corpus_size is usize, so always >= 0)
    let _ = result.total_corpus_size;
}

/// Test DashFlow client convenience methods
#[tokio::test]
#[ignore = "requires running server"]
async fn test_dashflow_client_ready() {
    let client = DashFlowClient::local(3000);

    let ready = client.is_ready().await;
    assert!(ready);
}

/// Test DashFlow verify_modification
#[tokio::test]
#[ignore = "requires running server"]
async fn test_dashflow_verify_modification() {
    let client = DashFlowClient::local(3000);

    let spec = r#"
        theorem graph_safety {
            forall g: Graph . acyclic(g) implies terminates(g)
        }
    "#;

    let result = client
        .verify_modification(spec)
        .await
        .expect("verify should succeed");

    assert!(result.safe);
}

/// Test error handling for invalid spec
#[tokio::test]
#[ignore = "requires running server"]
async fn test_live_server_invalid_spec() {
    let client = RemoteClient::local(3000);

    let result = client.verify("this is not valid USL!").await;

    // Should succeed but return invalid
    match result {
        Ok(resp) => {
            assert!(!resp.valid);
            assert!(!resp.errors.is_empty());
        }
        Err(e) => {
            // Server error is also acceptable for invalid input
            println!("Got error (expected): {:?}", e);
        }
    }
}
