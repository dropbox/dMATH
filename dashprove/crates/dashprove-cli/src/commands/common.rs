//! Common utilities shared across CLI commands

use dashprove::backends::{BackendId, HealthStatus, VerificationBackend};
use dashprove::usl::{suggest_tactics_for_property, Property};
use std::path::PathBuf;

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify parse_backend returns Some for "lean"
    #[kani::proof]
    fn verify_parse_backend_lean() {
        let result = parse_backend("lean");
        kani::assert(result.is_some(), "lean should parse");
        kani::assert(result == Some(BackendId::Lean4), "lean should map to Lean4");
    }

    /// Verify parse_backend returns Some for "lean4"
    #[kani::proof]
    fn verify_parse_backend_lean4() {
        let result = parse_backend("lean4");
        kani::assert(result.is_some(), "lean4 should parse");
        kani::assert(
            result == Some(BackendId::Lean4),
            "lean4 should map to Lean4",
        );
    }

    /// Verify parse_backend returns Some for "tla+"
    #[kani::proof]
    fn verify_parse_backend_tlaplus() {
        let result = parse_backend("tla+");
        kani::assert(result.is_some(), "tla+ should parse");
        kani::assert(
            result == Some(BackendId::TlaPlus),
            "tla+ should map to TlaPlus",
        );
    }

    /// Verify parse_backend returns Some for "tlaplus"
    #[kani::proof]
    fn verify_parse_backend_tlaplus_variant() {
        let result = parse_backend("tlaplus");
        kani::assert(result.is_some(), "tlaplus should parse");
        kani::assert(
            result == Some(BackendId::TlaPlus),
            "tlaplus should map to TlaPlus",
        );
    }

    /// Verify parse_backend returns Some for "tla"
    #[kani::proof]
    fn verify_parse_backend_tla() {
        let result = parse_backend("tla");
        kani::assert(result.is_some(), "tla should parse");
        kani::assert(
            result == Some(BackendId::TlaPlus),
            "tla should map to TlaPlus",
        );
    }

    /// Verify parse_backend returns Some for "kani"
    #[kani::proof]
    fn verify_parse_backend_kani() {
        let result = parse_backend("kani");
        kani::assert(result.is_some(), "kani should parse");
        kani::assert(result == Some(BackendId::Kani), "kani should map to Kani");
    }

    /// Verify parse_backend returns Some for "alloy"
    #[kani::proof]
    fn verify_parse_backend_alloy() {
        let result = parse_backend("alloy");
        kani::assert(result.is_some(), "alloy should parse");
        kani::assert(
            result == Some(BackendId::Alloy),
            "alloy should map to Alloy",
        );
    }

    /// Verify parse_backend returns Some for "isabelle"
    #[kani::proof]
    fn verify_parse_backend_isabelle() {
        let result = parse_backend("isabelle");
        kani::assert(result.is_some(), "isabelle should parse");
        kani::assert(
            result == Some(BackendId::Isabelle),
            "isabelle should map to Isabelle",
        );
    }

    /// Verify parse_backend returns Some for "coq"
    #[kani::proof]
    fn verify_parse_backend_coq() {
        let result = parse_backend("coq");
        kani::assert(result.is_some(), "coq should parse");
        kani::assert(result == Some(BackendId::Coq), "coq should map to Coq");
    }

    /// Verify parse_backend returns Some for "dafny"
    #[kani::proof]
    fn verify_parse_backend_dafny() {
        let result = parse_backend("dafny");
        kani::assert(result.is_some(), "dafny should parse");
        kani::assert(
            result == Some(BackendId::Dafny),
            "dafny should map to Dafny",
        );
    }

    /// Verify parse_backend returns Some for "platform_api"
    #[kani::proof]
    fn verify_parse_backend_platform_api() {
        let result = parse_backend("platform_api");
        kani::assert(result.is_some(), "platform_api should parse");
        kani::assert(
            result == Some(BackendId::PlatformApi),
            "platform_api should map to PlatformApi",
        );
    }

    /// Verify parse_backend returns Some for "platform-api"
    #[kani::proof]
    fn verify_parse_backend_platform_api_hyphen() {
        let result = parse_backend("platform-api");
        kani::assert(result.is_some(), "platform-api should parse");
        kani::assert(
            result == Some(BackendId::PlatformApi),
            "platform-api should map to PlatformApi",
        );
    }

    /// Verify parse_backend returns Some for "platform"
    #[kani::proof]
    fn verify_parse_backend_platform() {
        let result = parse_backend("platform");
        kani::assert(result.is_some(), "platform should parse");
        kani::assert(
            result == Some(BackendId::PlatformApi),
            "platform should map to PlatformApi",
        );
    }

    /// Verify parse_backend returns None for unknown backend
    #[kani::proof]
    fn verify_parse_backend_unknown() {
        let result = parse_backend("unknown_backend");
        kani::assert(result.is_none(), "unknown backend should return None");
    }

    /// Verify parse_backend is case-insensitive for "LEAN"
    #[kani::proof]
    fn verify_parse_backend_case_insensitive_lean() {
        let result = parse_backend("LEAN");
        kani::assert(result.is_some(), "LEAN should parse");
        kani::assert(result == Some(BackendId::Lean4), "LEAN should map to Lean4");
    }

    /// Verify parse_backend is case-insensitive for "TLA+"
    #[kani::proof]
    fn verify_parse_backend_case_insensitive_tlaplus() {
        let result = parse_backend("TLA+");
        kani::assert(result.is_some(), "TLA+ should parse");
        kani::assert(
            result == Some(BackendId::TlaPlus),
            "TLA+ should map to TlaPlus",
        );
    }

    /// Verify parse_backend is case-insensitive for "Kani"
    #[kani::proof]
    fn verify_parse_backend_case_insensitive_kani() {
        let result = parse_backend("Kani");
        kani::assert(result.is_some(), "Kani should parse");
        kani::assert(result == Some(BackendId::Kani), "Kani should map to Kani");
    }

    /// Verify resolve_data_dir returns provided path when Some
    #[kani::proof]
    fn verify_resolve_data_dir_some() {
        let path = resolve_data_dir(Some("/custom/path"));
        kani::assert(
            path == PathBuf::from("/custom/path"),
            "should return provided path",
        );
    }

    /// Verify default_data_dir contains .dashprove
    #[kani::proof]
    fn verify_default_data_dir_contains_dashprove() {
        let path = default_data_dir();
        // The path should end with .dashprove
        let path_str = path.to_string_lossy();
        kani::assert(
            path_str.ends_with(".dashprove"),
            "default path should end with .dashprove",
        );
    }
}

/// Parse a backend name into BackendId
pub fn parse_backend(name: &str) -> Option<BackendId> {
    match name.to_lowercase().as_str() {
        "lean" | "lean4" => Some(BackendId::Lean4),
        "tla+" | "tlaplus" | "tla" => Some(BackendId::TlaPlus),
        "kani" => Some(BackendId::Kani),
        "alloy" => Some(BackendId::Alloy),
        "isabelle" => Some(BackendId::Isabelle),
        "coq" => Some(BackendId::Coq),
        "dafny" => Some(BackendId::Dafny),
        "platform_api" | "platform-api" | "platform" => Some(BackendId::PlatformApi),
        _ => None,
    }
}

/// Check if a backend is available
pub async fn is_backend_available(backend: &dyn VerificationBackend) -> bool {
    matches!(backend.health_check().await, HealthStatus::Healthy)
}

/// Get the default data directory for learning data
pub fn default_data_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".dashprove")
}

/// Resolve data directory from option or default
pub fn resolve_data_dir(data_dir: Option<&str>) -> PathBuf {
    data_dir.map(PathBuf::from).unwrap_or_else(default_data_dir)
}

/// Get compiler-suggested tactics for a property
pub fn get_compiler_tactics(property: &Property) -> Vec<String> {
    suggest_tactics_for_property(property)
}
