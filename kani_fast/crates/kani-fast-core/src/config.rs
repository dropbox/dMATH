//! Configuration types for Kani Fast

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

/// Configuration for the Kani wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KaniConfig {
    /// Timeout for verification (default: 5 minutes)
    pub timeout: Duration,

    /// Path to cargo binary (auto-detected if None)
    pub cargo_path: Option<PathBuf>,

    /// Enable concrete playback mode for counterexamples
    pub enable_concrete_playback: bool,

    /// Default loop unwinding bound
    pub default_unwind: Option<u32>,

    /// SAT solver to use (cadical, minisat, etc.)
    pub solver: Option<String>,

    /// Extra arguments to pass to cargo kani
    pub extra_args: Vec<String>,
}

impl Default for KaniConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(300),
            cargo_path: None,
            enable_concrete_playback: true,
            default_unwind: None,
            solver: None,
            extra_args: Vec::new(),
        }
    }
}

/// Raw output from a Kani invocation
#[derive(Debug, Clone)]
pub struct KaniOutput {
    /// Standard output
    pub stdout: String,

    /// Standard error
    pub stderr: String,

    /// Exit code (None if process was killed)
    pub exit_code: Option<i32>,

    /// Time taken for verification
    pub duration: Duration,
}

impl KaniOutput {
    /// Get combined stdout and stderr
    pub fn combined(&self) -> String {
        format!("{}\n{}", self.stdout, self.stderr)
    }
}

/// Verification mode for Kani Fast
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum VerificationMode {
    /// Automatic mode selection based on code analysis
    #[default]
    Auto,

    /// Standard bounded model checking (baseline Kani)
    Bounded,

    /// K-induction for unbounded verification
    KInduction,

    /// CHC-based verification via Spacer
    Chc,

    /// Portfolio mode: run multiple strategies in parallel
    Portfolio,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kani_config_default() {
        let config = KaniConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert!(config.cargo_path.is_none());
        assert!(config.enable_concrete_playback);
        assert!(config.default_unwind.is_none());
        assert!(config.solver.is_none());
        assert!(config.extra_args.is_empty());
    }

    #[test]
    fn test_kani_config_custom() {
        let config = KaniConfig {
            timeout: Duration::from_secs(60),
            cargo_path: Some(PathBuf::from("/usr/bin/cargo")),
            enable_concrete_playback: false,
            default_unwind: Some(10),
            solver: Some("cadical".to_string()),
            extra_args: vec!["--verbose".to_string()],
        };
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.cargo_path, Some(PathBuf::from("/usr/bin/cargo")));
        assert!(!config.enable_concrete_playback);
        assert_eq!(config.default_unwind, Some(10));
        assert_eq!(config.solver, Some("cadical".to_string()));
        assert_eq!(config.extra_args, vec!["--verbose"]);
    }

    #[test]
    fn test_kani_config_serialization() {
        let config = KaniConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: KaniConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.timeout, deserialized.timeout);
        assert_eq!(
            config.enable_concrete_playback,
            deserialized.enable_concrete_playback
        );
    }

    #[test]
    fn test_kani_output_combined() {
        let output = KaniOutput {
            stdout: "verification complete".to_string(),
            stderr: "warning: unused variable".to_string(),
            exit_code: Some(0),
            duration: Duration::from_secs(5),
        };
        let combined = output.combined();
        assert!(combined.contains("verification complete"));
        assert!(combined.contains("warning: unused variable"));
    }

    #[test]
    fn test_kani_output_no_exit_code() {
        let output = KaniOutput {
            stdout: String::new(),
            stderr: "killed".to_string(),
            exit_code: None,
            duration: Duration::from_secs(30),
        };
        assert!(output.exit_code.is_none());
    }

    #[test]
    fn test_verification_mode_default() {
        let mode = VerificationMode::default();
        assert_eq!(mode, VerificationMode::Auto);
    }

    #[test]
    fn test_verification_mode_equality() {
        assert_eq!(VerificationMode::Bounded, VerificationMode::Bounded);
        assert_ne!(VerificationMode::KInduction, VerificationMode::Chc);
    }

    #[test]
    fn test_verification_mode_serialization() {
        let modes = [
            VerificationMode::Auto,
            VerificationMode::Bounded,
            VerificationMode::KInduction,
            VerificationMode::Chc,
            VerificationMode::Portfolio,
        ];
        for mode in modes {
            let json = serde_json::to_string(&mode).unwrap();
            let deserialized: VerificationMode = serde_json::from_str(&json).unwrap();
            assert_eq!(mode, deserialized);
        }
    }

    #[test]
    fn test_kani_config_clone() {
        let config = KaniConfig {
            timeout: Duration::from_secs(60),
            cargo_path: Some(PathBuf::from("/usr/bin/cargo")),
            enable_concrete_playback: false,
            default_unwind: Some(10),
            solver: Some("cadical".to_string()),
            extra_args: vec!["--verbose".to_string()],
        };
        let cloned = config.clone();
        assert_eq!(config.timeout, cloned.timeout);
        assert_eq!(config.cargo_path, cloned.cargo_path);
        assert_eq!(
            config.enable_concrete_playback,
            cloned.enable_concrete_playback
        );
        assert_eq!(config.default_unwind, cloned.default_unwind);
        assert_eq!(config.solver, cloned.solver);
        assert_eq!(config.extra_args, cloned.extra_args);
    }

    #[test]
    fn test_kani_config_debug() {
        let config = KaniConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("KaniConfig"));
        assert!(debug_str.contains("timeout"));
        assert!(debug_str.contains("enable_concrete_playback"));
    }

    #[test]
    fn test_kani_config_serialization_roundtrip() {
        let config = KaniConfig {
            timeout: Duration::from_secs(120),
            cargo_path: Some(PathBuf::from("/custom/path")),
            enable_concrete_playback: false,
            default_unwind: Some(50),
            solver: Some("z3".to_string()),
            extra_args: vec!["--tests".to_string(), "--quiet".to_string()],
        };
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: KaniConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.timeout, deserialized.timeout);
        assert_eq!(config.cargo_path, deserialized.cargo_path);
        assert_eq!(
            config.enable_concrete_playback,
            deserialized.enable_concrete_playback
        );
        assert_eq!(config.default_unwind, deserialized.default_unwind);
        assert_eq!(config.solver, deserialized.solver);
        assert_eq!(config.extra_args, deserialized.extra_args);
    }

    #[test]
    fn test_kani_config_json_deserialization() {
        let json = r#"{
            "timeout": {"secs": 30, "nanos": 0},
            "cargo_path": "/path/to/cargo",
            "enable_concrete_playback": true,
            "default_unwind": 20,
            "solver": "minisat",
            "extra_args": []
        }"#;
        let config: KaniConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.timeout, Duration::from_secs(30));
        assert_eq!(config.cargo_path, Some(PathBuf::from("/path/to/cargo")));
        assert!(config.enable_concrete_playback);
        assert_eq!(config.default_unwind, Some(20));
        assert_eq!(config.solver, Some("minisat".to_string()));
        assert!(config.extra_args.is_empty());
    }

    #[test]
    fn test_kani_config_with_no_optional_fields() {
        let config = KaniConfig {
            timeout: Duration::from_secs(60),
            cargo_path: None,
            enable_concrete_playback: false,
            default_unwind: None,
            solver: None,
            extra_args: Vec::new(),
        };
        assert!(config.cargo_path.is_none());
        assert!(config.default_unwind.is_none());
        assert!(config.solver.is_none());
    }

    #[test]
    fn test_kani_config_with_multiple_extra_args() {
        let config = KaniConfig {
            extra_args: vec![
                "--tests".to_string(),
                "--quiet".to_string(),
                "--verbose".to_string(),
                "--unstable".to_string(),
            ],
            ..Default::default()
        };
        assert_eq!(config.extra_args.len(), 4);
        assert!(config.extra_args.contains(&"--tests".to_string()));
        assert!(config.extra_args.contains(&"--unstable".to_string()));
    }

    #[test]
    fn test_kani_output_debug() {
        let output = KaniOutput {
            stdout: "out".to_string(),
            stderr: "err".to_string(),
            exit_code: Some(0),
            duration: Duration::from_secs(1),
        };
        let debug_str = format!("{:?}", output);
        assert!(debug_str.contains("KaniOutput"));
        assert!(debug_str.contains("stdout"));
        assert!(debug_str.contains("stderr"));
    }

    #[test]
    fn test_kani_output_clone() {
        let output = KaniOutput {
            stdout: "verification complete".to_string(),
            stderr: "warning: test".to_string(),
            exit_code: Some(1),
            duration: Duration::from_millis(500),
        };
        let cloned = output.clone();
        assert_eq!(output.stdout, cloned.stdout);
        assert_eq!(output.stderr, cloned.stderr);
        assert_eq!(output.exit_code, cloned.exit_code);
        assert_eq!(output.duration, cloned.duration);
    }

    #[test]
    fn test_kani_output_combined_with_empty_strings() {
        let output = KaniOutput {
            stdout: String::new(),
            stderr: String::new(),
            exit_code: Some(0),
            duration: Duration::ZERO,
        };
        let combined = output.combined();
        assert_eq!(combined.trim(), "");
    }

    #[test]
    fn test_kani_output_combined_multiline() {
        let output = KaniOutput {
            stdout: "line1\nline2\nline3".to_string(),
            stderr: "error1\nerror2".to_string(),
            exit_code: None,
            duration: Duration::from_secs(10),
        };
        let combined = output.combined();
        assert!(combined.contains("line1"));
        assert!(combined.contains("line2"));
        assert!(combined.contains("line3"));
        assert!(combined.contains("error1"));
        assert!(combined.contains("error2"));
    }

    #[test]
    fn test_verification_mode_debug() {
        let modes = [
            VerificationMode::Auto,
            VerificationMode::Bounded,
            VerificationMode::KInduction,
            VerificationMode::Chc,
            VerificationMode::Portfolio,
        ];
        for mode in modes {
            let debug_str = format!("{:?}", mode);
            assert!(!debug_str.is_empty());
        }
    }

    #[test]
    fn test_verification_mode_clone() {
        let mode = VerificationMode::KInduction;
        #[allow(clippy::clone_on_copy)]
        let cloned = mode.clone();
        assert_eq!(mode, cloned);
    }

    #[test]
    fn test_verification_mode_copy() {
        let mode = VerificationMode::Chc;
        let copied: VerificationMode = mode;
        assert_eq!(mode, copied);
    }

    #[test]
    fn test_verification_mode_json_values() {
        assert_eq!(
            serde_json::to_string(&VerificationMode::Auto).unwrap(),
            "\"Auto\""
        );
        assert_eq!(
            serde_json::to_string(&VerificationMode::Bounded).unwrap(),
            "\"Bounded\""
        );
        assert_eq!(
            serde_json::to_string(&VerificationMode::KInduction).unwrap(),
            "\"KInduction\""
        );
        assert_eq!(
            serde_json::to_string(&VerificationMode::Chc).unwrap(),
            "\"Chc\""
        );
        assert_eq!(
            serde_json::to_string(&VerificationMode::Portfolio).unwrap(),
            "\"Portfolio\""
        );
    }

    #[test]
    fn test_verification_mode_from_json() {
        assert_eq!(
            serde_json::from_str::<VerificationMode>("\"Auto\"").unwrap(),
            VerificationMode::Auto
        );
        assert_eq!(
            serde_json::from_str::<VerificationMode>("\"Bounded\"").unwrap(),
            VerificationMode::Bounded
        );
        assert_eq!(
            serde_json::from_str::<VerificationMode>("\"KInduction\"").unwrap(),
            VerificationMode::KInduction
        );
        assert_eq!(
            serde_json::from_str::<VerificationMode>("\"Chc\"").unwrap(),
            VerificationMode::Chc
        );
        assert_eq!(
            serde_json::from_str::<VerificationMode>("\"Portfolio\"").unwrap(),
            VerificationMode::Portfolio
        );
    }

    #[test]
    fn test_kani_config_extreme_values() {
        let config = KaniConfig {
            timeout: Duration::from_secs(u64::MAX),
            cargo_path: Some(PathBuf::from("")),
            enable_concrete_playback: true,
            default_unwind: Some(u32::MAX),
            solver: Some(String::new()),
            extra_args: Vec::new(),
        };
        assert_eq!(config.timeout, Duration::from_secs(u64::MAX));
        assert_eq!(config.default_unwind, Some(u32::MAX));
    }

    #[test]
    fn test_kani_output_exit_codes() {
        let outputs = [
            KaniOutput {
                stdout: String::new(),
                stderr: String::new(),
                exit_code: Some(0),
                duration: Duration::ZERO,
            },
            KaniOutput {
                stdout: String::new(),
                stderr: String::new(),
                exit_code: Some(1),
                duration: Duration::ZERO,
            },
            KaniOutput {
                stdout: String::new(),
                stderr: String::new(),
                exit_code: Some(-1),
                duration: Duration::ZERO,
            },
            KaniOutput {
                stdout: String::new(),
                stderr: String::new(),
                exit_code: Some(127),
                duration: Duration::ZERO,
            },
        ];
        assert_eq!(outputs[0].exit_code, Some(0));
        assert_eq!(outputs[1].exit_code, Some(1));
        assert_eq!(outputs[2].exit_code, Some(-1));
        assert_eq!(outputs[3].exit_code, Some(127));
    }
}
