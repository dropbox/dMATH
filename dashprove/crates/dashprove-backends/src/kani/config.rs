//! Configuration types for Kani backend

use std::path::PathBuf;
use std::time::Duration;

// =============================================
// Kani Proofs for Configuration Types
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify KaniConfig::default produces valid defaults
    #[kani::proof]
    fn proof_kani_config_default_timeout_is_300_seconds() {
        let config = KaniConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "Default timeout should be 300 seconds",
        );
    }

    #[kani::proof]
    fn proof_kani_config_default_cargo_path_is_none() {
        let config = KaniConfig::default();
        kani::assert(
            config.cargo_path.is_none(),
            "Default cargo_path should be None",
        );
    }

    #[kani::proof]
    fn proof_kani_config_default_project_dir_is_none() {
        let config = KaniConfig::default();
        kani::assert(
            config.project_dir.is_none(),
            "Default project_dir should be None",
        );
    }

    #[kani::proof]
    fn proof_kani_config_default_concrete_playback_is_false() {
        let config = KaniConfig::default();
        kani::assert(
            !config.enable_concrete_playback,
            "Default enable_concrete_playback should be false",
        );
    }

    /// Verify KaniDetection::NotFound preserves reason
    #[kani::proof]
    fn proof_kani_detection_not_found_preserves_reason() {
        let reason = "test reason".to_string();
        let detection = KaniDetection::NotFound(reason.clone());
        if let KaniDetection::NotFound(stored_reason) = detection {
            kani::assert(
                stored_reason == reason,
                "NotFound should preserve the reason",
            );
        } else {
            kani::assert(false, "Should be NotFound variant");
        }
    }

    /// Verify KaniOutput struct can be constructed
    #[kani::proof]
    fn proof_kani_output_construction() {
        let output = KaniOutput {
            stdout: String::new(),
            stderr: String::new(),
            exit_code: Some(0),
            duration: Duration::from_secs(1),
        };
        kani::assert(output.exit_code == Some(0), "Exit code should be 0");
        kani::assert(
            output.duration == Duration::from_secs(1),
            "Duration should be 1s",
        );
    }

    #[kani::proof]
    fn proof_kani_output_exit_code_none() {
        let output = KaniOutput {
            stdout: "test".to_string(),
            stderr: "error".to_string(),
            exit_code: None,
            duration: Duration::from_millis(500),
        };
        kani::assert(output.exit_code.is_none(), "Exit code should be None");
    }

    /// Verify KaniDetection::Available stores path
    #[kani::proof]
    fn proof_kani_detection_available_stores_path() {
        let path = PathBuf::from("/usr/bin/cargo");
        let detection = KaniDetection::Available {
            cargo_path: path.clone(),
        };
        if let KaniDetection::Available { cargo_path } = detection {
            kani::assert(cargo_path == path, "Available should store cargo_path");
        } else {
            kani::assert(false, "Should be Available variant");
        }
    }
}

/// Configuration for Kani backend
#[derive(Debug, Clone)]
pub struct KaniConfig {
    /// Path to `cargo` binary (if not in PATH)
    pub cargo_path: Option<PathBuf>,
    /// Path to the Rust crate containing contract implementations
    pub project_dir: Option<PathBuf>,
    /// Timeout for verification
    pub timeout: Duration,
    /// Enable concrete playback to capture counterexamples
    pub enable_concrete_playback: bool,
}

impl Default for KaniConfig {
    fn default() -> Self {
        Self {
            cargo_path: None,
            project_dir: None,
            timeout: Duration::from_secs(300),
            enable_concrete_playback: false,
        }
    }
}

/// Result of Kani detection
#[derive(Debug, Clone)]
pub enum KaniDetection {
    /// Kani is available
    Available {
        /// Path to cargo binary
        cargo_path: PathBuf,
    },
    /// Kani is not found
    NotFound(String),
}

/// Captured output from cargo kani
pub struct KaniOutput {
    /// Standard output
    pub stdout: String,
    /// Standard error
    pub stderr: String,
    /// Exit code
    pub exit_code: Option<i32>,
    /// Time taken
    pub duration: Duration,
}
