//! Configuration types for Kani Fast backend

use std::path::PathBuf;
use std::time::Duration;

/// Verification mode for Kani Fast
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VerificationMode {
    /// Standard Kani bounded model checking
    #[default]
    Bounded,
    /// K-induction for unbounded verification
    KInduction,
    /// CHC solving via Z3 Spacer
    Chc,
    /// Portfolio: run multiple solvers in parallel
    Portfolio,
    /// Auto: automatically select best strategy
    Auto,
}

impl VerificationMode {
    /// Convert to CLI flag string
    #[must_use]
    pub fn as_flag(self) -> Option<&'static str> {
        match self {
            VerificationMode::Bounded => None, // default mode
            VerificationMode::KInduction => Some("--kinduction"),
            VerificationMode::Chc => Some("--chc"),
            VerificationMode::Portfolio => Some("--portfolio"),
            VerificationMode::Auto => Some("--auto"),
        }
    }
}

/// Configuration for Kani Fast backend
#[derive(Debug, Clone)]
pub struct KaniFastConfig {
    /// Path to `kani-fast` binary (if not in PATH)
    pub cli_path: Option<PathBuf>,
    /// Path to the Rust crate containing contract implementations
    pub project_dir: Option<PathBuf>,
    /// Timeout for verification
    pub timeout: Duration,
    /// Verification mode to use
    pub mode: VerificationMode,
    /// Enable AI-assisted invariant synthesis
    pub enable_ai: bool,
    /// Maximum AI synthesis attempts
    pub ai_max_attempts: u32,
    /// Enable counterexample explanations
    pub enable_explain: bool,
    /// Generate Lean5 proof certificates
    pub enable_lean5: bool,
    /// Specific harness to verify (None = all harnesses)
    pub harness: Option<String>,
    /// Number of parallel solvers for portfolio mode
    pub parallel_solvers: Option<usize>,
}

impl Default for KaniFastConfig {
    fn default() -> Self {
        Self {
            cli_path: None,
            project_dir: None,
            timeout: Duration::from_secs(300),
            mode: VerificationMode::default(),
            enable_ai: false,
            ai_max_attempts: 5,
            enable_explain: true,
            enable_lean5: false,
            harness: None,
            parallel_solvers: None,
        }
    }
}

/// Result of Kani Fast detection
#[derive(Debug, Clone)]
pub enum KaniFastDetection {
    /// Kani Fast CLI is available
    Available {
        /// Path to kani-fast binary
        cli_path: PathBuf,
    },
    /// Kani Fast is not found
    NotFound(String),
}

/// Captured output from kani-fast CLI
pub struct KaniFastOutput {
    /// Standard output
    pub stdout: String,
    /// Standard error
    pub stderr: String,
    /// Exit code
    pub exit_code: Option<i32>,
    /// Time taken
    pub duration: Duration,
}

// =============================================
// Kani Proofs for Configuration Types
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    #[kani::proof]
    fn proof_kanifast_config_default_timeout_is_300_seconds() {
        let config = KaniFastConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "Default timeout should be 300 seconds",
        );
    }

    #[kani::proof]
    fn proof_kanifast_config_default_mode_is_bounded() {
        let config = KaniFastConfig::default();
        kani::assert(
            matches!(config.mode, VerificationMode::Bounded),
            "Default mode should be Bounded",
        );
    }

    #[kani::proof]
    fn proof_kanifast_config_default_ai_disabled() {
        let config = KaniFastConfig::default();
        kani::assert(!config.enable_ai, "AI should be disabled by default");
    }

    #[kani::proof]
    fn proof_kanifast_config_default_explain_enabled() {
        let config = KaniFastConfig::default();
        kani::assert(
            config.enable_explain,
            "Explain should be enabled by default",
        );
    }

    #[kani::proof]
    fn proof_verification_mode_bounded_no_flag() {
        let mode = VerificationMode::Bounded;
        kani::assert(mode.as_flag().is_none(), "Bounded mode should have no flag");
    }

    #[kani::proof]
    fn proof_verification_mode_kinduction_flag() {
        let mode = VerificationMode::KInduction;
        kani::assert(
            mode.as_flag() == Some("--kinduction"),
            "KInduction mode should have --kinduction flag",
        );
    }

    #[kani::proof]
    fn proof_verification_mode_portfolio_flag() {
        let mode = VerificationMode::Portfolio;
        kani::assert(
            mode.as_flag() == Some("--portfolio"),
            "Portfolio mode should have --portfolio flag",
        );
    }

    #[kani::proof]
    fn proof_kanifast_detection_not_found_preserves_reason() {
        let reason = "test reason".to_string();
        let detection = KaniFastDetection::NotFound(reason.clone());
        if let KaniFastDetection::NotFound(stored_reason) = detection {
            kani::assert(
                stored_reason == reason,
                "NotFound should preserve the reason",
            );
        } else {
            kani::assert(false, "Should be NotFound variant");
        }
    }
}
