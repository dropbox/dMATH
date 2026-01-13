//! Tests for Apalache backend

use super::*;

#[test]
fn test_apalache_backend_creation() {
    let backend = ApalacheBackend::new();
    assert_eq!(backend.id(), BackendId::Apalache);
}

#[test]
fn test_apalache_supports_temporal_and_invariant() {
    let backend = ApalacheBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::Temporal));
    assert!(supported.contains(&PropertyType::Invariant));
}

#[test]
fn test_apalache_config_defaults() {
    let config = ApalacheConfig::default();
    assert_eq!(config.mode, ApalacheMode::Check);
    assert_eq!(config.length, Some(10));
    assert_eq!(config.smt_solver, "z3");
    assert!(!config.debug);
}

#[test]
fn test_apalache_mode_command() {
    assert_eq!(ApalacheMode::Check.command(), "check");
    assert_eq!(ApalacheMode::Parse.command(), "parse");
    assert_eq!(ApalacheMode::Simulate.command(), "simulate");
}

#[tokio::test]
async fn test_apalache_health_check_without_installation() {
    // Create backend with custom path that doesn't exist
    let config = ApalacheConfig {
        apalache_path: Some("/nonexistent/apalache".into()),
        ..Default::default()
    };

    let backend = ApalacheBackend::with_config(config);
    let status = backend.health_check().await;

    // Should be unavailable since the path doesn't exist
    assert!(matches!(status, HealthStatus::Unavailable { .. }));
}

#[test]
fn test_apalache_detection_enum() {
    // Test that detection enum variants work
    let standalone = ApalacheDetection::Standalone("/usr/bin/apalache-mc".into());
    let jar = ApalacheDetection::Jar {
        java_path: "/usr/bin/java".into(),
        jar_path: "/opt/apalache/lib/apalache.jar".into(),
    };
    let not_found = ApalacheDetection::NotFound("not installed".to_string());

    // Just ensure the enum variants can be constructed
    assert!(matches!(standalone, ApalacheDetection::Standalone(_)));
    assert!(matches!(jar, ApalacheDetection::Jar { .. }));
    assert!(matches!(not_found, ApalacheDetection::NotFound(_)));
}
