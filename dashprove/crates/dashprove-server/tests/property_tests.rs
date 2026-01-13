//! Property-based tests for dashprove-server using proptest

use dashprove_backends::BackendId;
use dashprove_server::{
    auth::{ApiKeyInfo, AuthConfig, KeyInfo},
    cache::{CacheStats, CachedResult},
    routes::{
        BackendIdParam, ChangeKind, CompilationResult, CorpusType, HealthStatusResponse,
        OutputFormat, PropertyTypeResponse, ShutdownState, TimePeriodParam,
    },
    ws::{
        ListSessionsResponse, SessionInfoResponse, VerificationPhase, WsMessage, WsVerifyRequest,
    },
};
use proptest::prelude::*;
use std::time::Duration;

// ============================================================================
// Strategy generators
// ============================================================================

fn arbitrary_backend() -> impl Strategy<Value = BackendId> {
    prop::sample::select(dashprove::SUPPORTED_BACKENDS.to_vec())
}

fn arbitrary_backend_id_param() -> impl Strategy<Value = BackendIdParam> {
    let params: Vec<_> = dashprove::SUPPORTED_BACKENDS
        .iter()
        .copied()
        .map(BackendIdParam::from)
        .collect();
    prop::sample::select(params)
}

fn arbitrary_verification_phase() -> impl Strategy<Value = VerificationPhase> {
    prop_oneof![
        Just(VerificationPhase::Parsing),
        Just(VerificationPhase::TypeChecking),
        Just(VerificationPhase::Compiling),
        Just(VerificationPhase::Verifying),
        Just(VerificationPhase::Merging),
    ]
}

fn arbitrary_api_key_info() -> impl Strategy<Value = ApiKeyInfo> {
    ("[a-z]{3,20}", 1u32..10000, any::<bool>()).prop_map(|(name, rate_limit, is_admin)| {
        ApiKeyInfo {
            name,
            rate_limit,
            is_admin,
        }
    })
}

fn arbitrary_auth_config() -> impl Strategy<Value = AuthConfig> {
    (
        any::<bool>(),
        prop::collection::hash_map("[a-zA-Z0-9]{16,32}", arbitrary_api_key_info(), 0..5),
        1u32..1000,
    )
        .prop_map(|(required, api_keys, anonymous_rate_limit)| AuthConfig {
            required,
            api_keys,
            anonymous_rate_limit,
        })
}

fn arbitrary_session_info_response() -> impl Strategy<Value = SessionInfoResponse> {
    (
        "[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}",
        0u64..86400,
        0u32..10000,
        any::<bool>(),
    )
        .prop_map(|(session_id, uptime_seconds, request_count, connected)| {
            SessionInfoResponse {
                session_id,
                uptime_seconds,
                request_count,
                connected,
            }
        })
}

fn arbitrary_list_sessions_response() -> impl Strategy<Value = ListSessionsResponse> {
    prop::collection::vec(arbitrary_session_info_response(), 0..10).prop_map(|sessions| {
        let active_count = sessions.iter().filter(|s| s.connected).count();
        ListSessionsResponse {
            sessions,
            active_count,
        }
    })
}

fn arbitrary_ws_verify_request() -> impl Strategy<Value = WsVerifyRequest> {
    (
        "property [a-z]+ \\{ [a-z ]+ \\}",
        prop::option::of(arbitrary_backend_id_param()),
        prop::option::of("[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}"),
    )
        .prop_map(|(spec, backend, request_id)| WsVerifyRequest {
            spec,
            backend,
            request_id,
        })
}

fn arbitrary_cached_result() -> impl Strategy<Value = CachedResult> {
    (
        any::<u64>(),
        any::<bool>(),
        arbitrary_backend(),
        "[a-zA-Z_][a-zA-Z0-9_\\s]*",
        prop::option::of("[a-zA-Z ]+"),
        0u64..1_000_000_000,
        60u64..86400,
    )
        .prop_map(
            |(property_hash, valid, backend, backend_code, error, created_secs, ttl_secs)| {
                CachedResult {
                    property_hash,
                    valid,
                    backend,
                    backend_code,
                    error,
                    created_at: Duration::from_secs(created_secs),
                    ttl_secs,
                }
            },
        )
}

fn arbitrary_key_info() -> impl Strategy<Value = KeyInfo> {
    (
        "[a-zA-Z0-9]{4,8}\\.\\.\\.",
        "[a-z]{3,20}",
        1u32..10000,
        any::<bool>(),
    )
        .prop_map(|(key_prefix, name, rate_limit, is_admin)| KeyInfo {
            key_prefix,
            name,
            rate_limit,
            is_admin,
        })
}

fn arbitrary_compilation_result() -> impl Strategy<Value = CompilationResult> {
    (arbitrary_backend_id_param(), "[a-zA-Z_ ]+")
        .prop_map(|(backend, code)| CompilationResult { backend, code })
}

fn arbitrary_cache_stats() -> impl Strategy<Value = CacheStats> {
    (
        0usize..1000,
        0usize..1000,
        0usize..1000,
        100usize..10000,
        60u64..86400,
    )
        .prop_map(
            |(total, valid, expired, max_entries, default_ttl_secs)| CacheStats {
                total_entries: total,
                valid_entries: valid.min(total),
                expired_entries: expired.min(total),
                max_entries,
                default_ttl_secs,
            },
        )
}

fn arbitrary_output_format() -> impl Strategy<Value = OutputFormat> {
    prop_oneof![Just(OutputFormat::Json), Just(OutputFormat::Html),]
}

fn arbitrary_corpus_type() -> impl Strategy<Value = CorpusType> {
    prop_oneof![Just(CorpusType::Proofs), Just(CorpusType::Counterexamples),]
}

fn arbitrary_time_period_param() -> impl Strategy<Value = TimePeriodParam> {
    prop_oneof![
        Just(TimePeriodParam::Day),
        Just(TimePeriodParam::Week),
        Just(TimePeriodParam::Month),
    ]
}

fn arbitrary_shutdown_state() -> impl Strategy<Value = ShutdownState> {
    prop_oneof![
        Just(ShutdownState::Running),
        Just(ShutdownState::Draining),
        Just(ShutdownState::ShuttingDown),
    ]
}

fn arbitrary_health_status_response() -> impl Strategy<Value = HealthStatusResponse> {
    (
        "[a-zA-Z ]+",
        prop_oneof![
            Just(0u8), // Healthy
            Just(1u8), // Degraded
            Just(2u8), // Unavailable
        ],
    )
        .prop_map(|(reason, variant)| match variant {
            0 => HealthStatusResponse::Healthy,
            1 => HealthStatusResponse::Degraded { reason },
            _ => HealthStatusResponse::Unavailable { reason },
        })
}

fn arbitrary_change_kind() -> impl Strategy<Value = ChangeKind> {
    prop_oneof![
        Just(ChangeKind::FileModified),
        Just(ChangeKind::FileAdded),
        Just(ChangeKind::FileDeleted),
        Just(ChangeKind::FunctionModified),
        Just(ChangeKind::TypeAdded),
        Just(ChangeKind::TypeModified),
        Just(ChangeKind::DependencyChanged),
    ]
}

fn arbitrary_property_type_response() -> impl Strategy<Value = PropertyTypeResponse> {
    prop_oneof![
        Just(PropertyTypeResponse::Theorem),
        Just(PropertyTypeResponse::Temporal),
        Just(PropertyTypeResponse::Contract),
        Just(PropertyTypeResponse::Invariant),
        Just(PropertyTypeResponse::Refinement),
        Just(PropertyTypeResponse::NeuralRobustness),
        Just(PropertyTypeResponse::NeuralReachability),
        Just(PropertyTypeResponse::Probabilistic),
        Just(PropertyTypeResponse::SecurityProtocol),
    ]
}

// ============================================================================
// AuthConfig property tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn auth_config_default_is_not_required(seed in any::<u64>()) {
        let _ = seed;
        let config = AuthConfig::default();
        prop_assert!(!config.required);
        prop_assert!(config.api_keys.is_empty());
        prop_assert_eq!(config.anonymous_rate_limit, 10);
    }

    #[test]
    fn auth_config_disabled_is_not_required(seed in any::<u64>()) {
        let _ = seed;
        let config = AuthConfig::disabled();
        prop_assert!(!config.required);
    }

    #[test]
    fn auth_config_required_is_required(seed in any::<u64>()) {
        let _ = seed;
        let config = AuthConfig::required();
        prop_assert!(config.required);
    }

    #[test]
    fn auth_config_with_api_key_adds_key(key in "[a-zA-Z0-9]{16,32}", name in "[a-z]{3,20}") {
        let config = AuthConfig::default().with_api_key(&key, &name);
        prop_assert!(config.api_keys.contains_key(&key));
        prop_assert_eq!(&config.api_keys[&key].name, &name);
        prop_assert_eq!(config.api_keys[&key].rate_limit, 100);
        prop_assert!(!config.api_keys[&key].is_admin);
    }

    #[test]
    fn auth_config_with_api_key_rate_limit_sets_limit(
        key in "[a-zA-Z0-9]{16,32}",
        name in "[a-z]{3,20}",
        rate_limit in 1u32..10000
    ) {
        let config = AuthConfig::default().with_api_key_rate_limit(&key, &name, rate_limit);
        prop_assert!(config.api_keys.contains_key(&key));
        prop_assert_eq!(config.api_keys[&key].rate_limit, rate_limit);
    }

    #[test]
    fn auth_config_with_admin_key_sets_admin(key in "[a-zA-Z0-9]{16,32}", name in "[a-z]{3,20}") {
        let config = AuthConfig::default().with_admin_key(&key, &name);
        prop_assert!(config.api_keys.contains_key(&key));
        prop_assert!(config.api_keys[&key].is_admin);
    }

    #[test]
    fn auth_config_with_admin_key_rate_limit_sets_both(
        key in "[a-zA-Z0-9]{16,32}",
        name in "[a-z]{3,20}",
        rate_limit in 1u32..10000
    ) {
        let config = AuthConfig::default().with_admin_key_rate_limit(&key, &name, rate_limit);
        prop_assert!(config.api_keys.contains_key(&key));
        prop_assert!(config.api_keys[&key].is_admin);
        prop_assert_eq!(config.api_keys[&key].rate_limit, rate_limit);
    }

    #[test]
    fn auth_config_with_anonymous_rate_limit_sets_limit(limit in 1u32..1000) {
        let config = AuthConfig::default().with_anonymous_rate_limit(limit);
        prop_assert_eq!(config.anonymous_rate_limit, limit);
    }

    #[test]
    fn auth_config_json_roundtrip(config in arbitrary_auth_config()) {
        let json = serde_json::to_string(&config).expect("serialize");
        let parsed: AuthConfig = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(config.required, parsed.required);
        prop_assert_eq!(config.anonymous_rate_limit, parsed.anonymous_rate_limit);
        prop_assert_eq!(config.api_keys.len(), parsed.api_keys.len());
    }
}

// ============================================================================
// ApiKeyInfo property tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn api_key_info_json_roundtrip(info in arbitrary_api_key_info()) {
        let json = serde_json::to_string(&info).expect("serialize");
        let parsed: ApiKeyInfo = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(info.name, parsed.name);
        prop_assert_eq!(info.rate_limit, parsed.rate_limit);
        prop_assert_eq!(info.is_admin, parsed.is_admin);
    }

    #[test]
    fn api_key_info_rate_limit_always_positive(info in arbitrary_api_key_info()) {
        prop_assert!(info.rate_limit >= 1);
    }
}

// ============================================================================
// KeyInfo property tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn key_info_json_roundtrip(info in arbitrary_key_info()) {
        let json = serde_json::to_string(&info).expect("serialize");
        let parsed: KeyInfo = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(info.key_prefix, parsed.key_prefix);
        prop_assert_eq!(info.name, parsed.name);
        prop_assert_eq!(info.rate_limit, parsed.rate_limit);
        prop_assert_eq!(info.is_admin, parsed.is_admin);
    }

    #[test]
    fn key_info_prefix_ends_with_dots(info in arbitrary_key_info()) {
        prop_assert!(info.key_prefix.ends_with("..."));
    }
}

// ============================================================================
// CachedResult property tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn cached_result_json_roundtrip(result in arbitrary_cached_result()) {
        let json = serde_json::to_string(&result).expect("serialize");
        let parsed: CachedResult = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(result.property_hash, parsed.property_hash);
        prop_assert_eq!(result.valid, parsed.valid);
        prop_assert_eq!(result.backend, parsed.backend);
        prop_assert_eq!(result.backend_code, parsed.backend_code);
        prop_assert_eq!(result.error, parsed.error);
        // Duration serializes to seconds
        prop_assert_eq!(result.created_at.as_secs(), parsed.created_at.as_secs());
        prop_assert_eq!(result.ttl_secs, parsed.ttl_secs);
    }

    #[test]
    fn cached_result_ttl_always_at_least_60(result in arbitrary_cached_result()) {
        prop_assert!(result.ttl_secs >= 60);
    }

    #[test]
    fn cached_result_backend_is_valid(result in arbitrary_cached_result()) {
        // Just ensure the backend variant is accessible
        let _ = format!("{:?}", result.backend);
    }
}

// ============================================================================
// CacheStats property tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn cache_stats_json_roundtrip(stats in arbitrary_cache_stats()) {
        let json = serde_json::to_string(&stats).expect("serialize");
        let parsed: CacheStats = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(stats.total_entries, parsed.total_entries);
        prop_assert_eq!(stats.valid_entries, parsed.valid_entries);
        prop_assert_eq!(stats.expired_entries, parsed.expired_entries);
        prop_assert_eq!(stats.max_entries, parsed.max_entries);
        prop_assert_eq!(stats.default_ttl_secs, parsed.default_ttl_secs);
    }

    #[test]
    fn cache_stats_valid_less_than_total(stats in arbitrary_cache_stats()) {
        prop_assert!(stats.valid_entries <= stats.total_entries);
    }

    #[test]
    fn cache_stats_ttl_at_least_60(stats in arbitrary_cache_stats()) {
        prop_assert!(stats.default_ttl_secs >= 60);
    }
}

// ============================================================================
// VerificationPhase property tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn verification_phase_json_roundtrip(phase in arbitrary_verification_phase()) {
        let json = serde_json::to_string(&phase).expect("serialize");
        let parsed: VerificationPhase = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(format!("{:?}", phase), format!("{:?}", parsed));
    }

    #[test]
    fn verification_phase_debug_non_empty(phase in arbitrary_verification_phase()) {
        let debug = format!("{:?}", phase);
        prop_assert!(!debug.is_empty());
    }

    #[test]
    fn verification_phase_serializes_to_snake_case(phase in arbitrary_verification_phase()) {
        let json = serde_json::to_string(&phase).expect("serialize");
        // JSON should be a quoted string in snake_case
        prop_assert!(json.starts_with('"') && json.ends_with('"'));
        let inner = &json[1..json.len()-1];
        // Snake case: lowercase with underscores
        prop_assert!(inner.chars().all(|c| c.is_ascii_lowercase() || c == '_'),
            "Phase '{}' should be snake_case", inner);
    }
}

// ============================================================================
// SessionInfoResponse property tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn session_info_response_json_roundtrip(info in arbitrary_session_info_response()) {
        let json = serde_json::to_string(&info).expect("serialize");
        let parsed: SessionInfoResponse = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(info.session_id, parsed.session_id);
        prop_assert_eq!(info.uptime_seconds, parsed.uptime_seconds);
        prop_assert_eq!(info.request_count, parsed.request_count);
        prop_assert_eq!(info.connected, parsed.connected);
    }

    #[test]
    fn session_info_response_session_id_is_uuid_format(info in arbitrary_session_info_response()) {
        // UUID format: 8-4-4-4-12 hex digits
        let parts: Vec<&str> = info.session_id.split('-').collect();
        prop_assert_eq!(parts.len(), 5);
        prop_assert_eq!(parts[0].len(), 8);
        prop_assert_eq!(parts[1].len(), 4);
        prop_assert_eq!(parts[2].len(), 4);
        prop_assert_eq!(parts[3].len(), 4);
        prop_assert_eq!(parts[4].len(), 12);
    }
}

// ============================================================================
// ListSessionsResponse property tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn list_sessions_response_json_roundtrip(response in arbitrary_list_sessions_response()) {
        let json = serde_json::to_string(&response).expect("serialize");
        let parsed: ListSessionsResponse = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(response.sessions.len(), parsed.sessions.len());
        prop_assert_eq!(response.active_count, parsed.active_count);
    }

    #[test]
    fn list_sessions_response_active_count_matches_connected(response in arbitrary_list_sessions_response()) {
        let actual_connected = response.sessions.iter().filter(|s| s.connected).count();
        prop_assert_eq!(response.active_count, actual_connected);
    }
}

// ============================================================================
// WsVerifyRequest property tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn ws_verify_request_json_roundtrip(req in arbitrary_ws_verify_request()) {
        let json = serde_json::to_string(&req).expect("serialize");
        let parsed: WsVerifyRequest = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(req.spec, parsed.spec);
        prop_assert_eq!(req.request_id, parsed.request_id);
    }

    #[test]
    fn ws_verify_request_spec_non_empty(req in arbitrary_ws_verify_request()) {
        prop_assert!(!req.spec.is_empty());
    }
}

// ============================================================================
// WsMessage property tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn ws_message_connected_json_roundtrip(
        session_id in "[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}",
        resumed in any::<bool>(),
        correlation_id in prop::option::of("[a-zA-Z0-9]{8,32}")
    ) {
        let msg = WsMessage::Connected {
            session_id: session_id.clone(),
            resumed,
            correlation_id: correlation_id.clone(),
        };
        let json = serde_json::to_string(&msg).expect("serialize");
        let parsed: WsMessage = serde_json::from_str(&json).expect("deserialize");
        if let WsMessage::Connected { session_id: sid, resumed: r, correlation_id: cid } = parsed {
            prop_assert_eq!(session_id, sid);
            prop_assert_eq!(resumed, r);
            prop_assert_eq!(correlation_id, cid);
        } else {
            prop_assert!(false, "Expected Connected variant");
        }
    }

    #[test]
    fn ws_message_accepted_json_roundtrip(
        request_id in "[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}"
    ) {
        let msg = WsMessage::Accepted { request_id: request_id.clone() };
        let json = serde_json::to_string(&msg).expect("serialize");
        let parsed: WsMessage = serde_json::from_str(&json).expect("deserialize");
        if let WsMessage::Accepted { request_id: rid } = parsed {
            prop_assert_eq!(request_id, rid);
        } else {
            prop_assert!(false, "Expected Accepted variant");
        }
    }

    #[test]
    fn ws_message_progress_json_roundtrip(
        request_id in "[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}",
        phase in arbitrary_verification_phase(),
        message in "[a-zA-Z ]+",
        percentage in prop::option::of(0u8..=100)
    ) {
        let msg = WsMessage::Progress {
            request_id: request_id.clone(),
            phase: phase.clone(),
            message: message.clone(),
            percentage,
        };
        let json = serde_json::to_string(&msg).expect("serialize");
        let parsed: WsMessage = serde_json::from_str(&json).expect("deserialize");
        if let WsMessage::Progress { request_id: rid, message: m, percentage: p, .. } = parsed {
            prop_assert_eq!(request_id, rid);
            prop_assert_eq!(message, m);
            prop_assert_eq!(percentage, p);
        } else {
            prop_assert!(false, "Expected Progress variant");
        }
    }

    #[test]
    fn ws_message_error_json_roundtrip(
        request_id in prop::option::of("[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}"),
        error in "[a-zA-Z0-9 ]+",
        details in prop::option::of("[a-zA-Z0-9 ]+")
    ) {
        let msg = WsMessage::Error {
            request_id: request_id.clone(),
            error: error.clone(),
            details: details.clone(),
        };
        let json = serde_json::to_string(&msg).expect("serialize");
        let parsed: WsMessage = serde_json::from_str(&json).expect("deserialize");
        if let WsMessage::Error { request_id: rid, error: e, details: d } = parsed {
            prop_assert_eq!(request_id, rid);
            prop_assert_eq!(error, e);
            prop_assert_eq!(details, d);
        } else {
            prop_assert!(false, "Expected Error variant");
        }
    }
}

// ============================================================================
// BackendIdParam property tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn backend_id_param_json_roundtrip(backend in arbitrary_backend_id_param()) {
        let json = serde_json::to_string(&backend).expect("serialize");
        let parsed: BackendIdParam = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(format!("{:?}", backend), format!("{:?}", parsed));
    }

    #[test]
    fn backend_id_param_to_backend_id_conversion(backend in arbitrary_backend_id_param()) {
        let backend_id: BackendId = backend.into();
        // Just ensure conversion works
        let _ = format!("{:?}", backend_id);
    }
}

// ============================================================================
// CompilationResult property tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn compilation_result_json_roundtrip(result in arbitrary_compilation_result()) {
        let json = serde_json::to_string(&result).expect("serialize");
        let parsed: CompilationResult = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(result.code, parsed.code);
    }

    #[test]
    fn compilation_result_code_non_empty(result in arbitrary_compilation_result()) {
        prop_assert!(!result.code.is_empty());
    }
}

// ============================================================================
// OutputFormat property tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn output_format_json_roundtrip(format in arbitrary_output_format()) {
        let json = serde_json::to_string(&format).expect("serialize");
        let parsed: OutputFormat = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(format!("{:?}", format), format!("{:?}", parsed));
    }
}

// ============================================================================
// CorpusType property tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn corpus_type_json_roundtrip(corpus_type in arbitrary_corpus_type()) {
        let json = serde_json::to_string(&corpus_type).expect("serialize");
        let parsed: CorpusType = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(format!("{:?}", corpus_type), format!("{:?}", parsed));
    }
}

// ============================================================================
// TimePeriodParam property tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn time_period_param_json_roundtrip(period in arbitrary_time_period_param()) {
        let json = serde_json::to_string(&period).expect("serialize");
        let parsed: TimePeriodParam = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(format!("{:?}", period), format!("{:?}", parsed));
    }
}

// ============================================================================
// ShutdownState property tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn shutdown_state_debug_non_empty(state in arbitrary_shutdown_state()) {
        let debug = format!("{:?}", state);
        prop_assert!(!debug.is_empty());
    }
}

// ============================================================================
// HealthStatusResponse property tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn health_status_response_json_roundtrip(status in arbitrary_health_status_response()) {
        let json = serde_json::to_string(&status).expect("serialize");
        let parsed: HealthStatusResponse = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(format!("{:?}", status), format!("{:?}", parsed));
    }
}

// ============================================================================
// ChangeKind property tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn change_kind_json_roundtrip(kind in arbitrary_change_kind()) {
        let json = serde_json::to_string(&kind).expect("serialize");
        let parsed: ChangeKind = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(format!("{:?}", kind), format!("{:?}", parsed));
    }
}

// ============================================================================
// PropertyTypeResponse property tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn property_type_response_json_roundtrip(prop_type in arbitrary_property_type_response()) {
        let json = serde_json::to_string(&prop_type).expect("serialize");
        let parsed: PropertyTypeResponse = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(format!("{:?}", prop_type), format!("{:?}", parsed));
    }
}

// ============================================================================
// Builder pattern property tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn auth_config_builder_chaining_is_commutative(
        key1 in "[a-zA-Z0-9]{16}",
        key2 in "[a-zA-Z0-9]{17}",
        name1 in "[a-z]{5}",
        name2 in "[a-z]{6}",
        rate_limit in 1u32..1000
    ) {
        // Order of adding keys shouldn't matter for the final result
        let config1 = AuthConfig::default()
            .with_api_key(&key1, &name1)
            .with_admin_key(&key2, &name2)
            .with_anonymous_rate_limit(rate_limit);

        let config2 = AuthConfig::default()
            .with_anonymous_rate_limit(rate_limit)
            .with_admin_key(&key2, &name2)
            .with_api_key(&key1, &name1);

        prop_assert_eq!(config1.api_keys.len(), config2.api_keys.len());
        prop_assert_eq!(config1.anonymous_rate_limit, config2.anonymous_rate_limit);
        prop_assert!(config1.api_keys.contains_key(&key1));
        prop_assert!(config1.api_keys.contains_key(&key2));
        prop_assert!(config2.api_keys.contains_key(&key1));
        prop_assert!(config2.api_keys.contains_key(&key2));
    }

    #[test]
    fn auth_config_builder_overwrite_key(
        key in "[a-zA-Z0-9]{16}",
        name1 in "[a-z]{5}",
        name2 in "[a-z]{6}",
        rate1 in 1u32..500,
        rate2 in 501u32..1000
    ) {
        // Adding same key twice should overwrite
        let config = AuthConfig::default()
            .with_api_key_rate_limit(&key, &name1, rate1)
            .with_api_key_rate_limit(&key, &name2, rate2);

        prop_assert_eq!(config.api_keys.len(), 1);
        prop_assert_eq!(&config.api_keys[&key].name, &name2);
        prop_assert_eq!(config.api_keys[&key].rate_limit, rate2);
    }
}
