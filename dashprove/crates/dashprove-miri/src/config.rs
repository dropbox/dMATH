//! MIRI configuration options

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

/// Configuration options for MIRI execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiriConfig {
    /// Path to cargo executable (None = use PATH)
    pub cargo_path: Option<PathBuf>,

    /// Timeout for MIRI execution
    pub timeout: Duration,

    /// MIRI-specific flags
    pub flags: MiriFlags,

    /// Additional environment variables
    pub env_vars: Vec<(String, String)>,

    /// Number of parallel jobs (None = default)
    pub jobs: Option<usize>,
}

impl Default for MiriConfig {
    fn default() -> Self {
        Self {
            cargo_path: None,
            timeout: Duration::from_secs(300), // 5 minutes default
            flags: MiriFlags::default(),
            env_vars: Vec::new(),
            jobs: None,
        }
    }
}

impl MiriConfig {
    /// Create a new config with custom timeout
    pub fn with_timeout(timeout: Duration) -> Self {
        Self {
            timeout,
            ..Default::default()
        }
    }

    /// Create a strict configuration for thorough UB detection
    pub fn strict() -> Self {
        Self {
            flags: MiriFlags::strict(),
            ..Default::default()
        }
    }

    /// Create a permissive configuration for faster testing
    pub fn permissive() -> Self {
        Self {
            flags: MiriFlags::permissive(),
            ..Default::default()
        }
    }
}

/// MIRI-specific flags for controlling detection behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiriFlags {
    /// Disable isolation (allows file I/O, etc.)
    pub disable_isolation: bool,

    /// Check for stacked borrows violations
    pub check_stacked_borrows: bool,

    /// Check for data races (requires threads)
    pub check_data_races: bool,

    /// Symbolic alignment checks
    pub symbolic_alignment: bool,

    /// Track raw pointers
    pub track_raw_pointers: bool,

    /// Seed for deterministic execution
    pub seed: Option<u64>,

    /// Number of operations before yielding in concurrent tests
    pub preemption_rate: Option<f64>,

    /// Additional raw MIRI flags
    pub raw_flags: Vec<String>,
}

impl Default for MiriFlags {
    fn default() -> Self {
        Self {
            disable_isolation: false,
            check_stacked_borrows: true,
            check_data_races: true,
            symbolic_alignment: true,
            track_raw_pointers: false,
            seed: None,
            preemption_rate: None,
            raw_flags: Vec::new(),
        }
    }
}

impl MiriFlags {
    /// Create strict flags for thorough checking
    pub fn strict() -> Self {
        Self {
            disable_isolation: false,
            check_stacked_borrows: true,
            check_data_races: true,
            symbolic_alignment: true,
            track_raw_pointers: true,
            seed: Some(0),              // Deterministic
            preemption_rate: Some(0.1), // Frequent context switches
            raw_flags: Vec::new(),
        }
    }

    /// Create permissive flags for faster testing
    pub fn permissive() -> Self {
        Self {
            disable_isolation: true,
            check_stacked_borrows: true,
            check_data_races: false,
            symbolic_alignment: false,
            track_raw_pointers: false,
            seed: None,
            preemption_rate: None,
            raw_flags: Vec::new(),
        }
    }

    /// Convert flags to MIRIFLAGS environment variable value
    pub fn to_miriflags(&self) -> String {
        let mut flags = Vec::new();

        if self.disable_isolation {
            flags.push("-Zmiri-disable-isolation".to_string());
        }

        if !self.check_stacked_borrows {
            flags.push("-Zmiri-disable-stacked-borrows".to_string());
        }

        if !self.check_data_races {
            flags.push("-Zmiri-disable-data-race-detector".to_string());
        }

        if self.symbolic_alignment {
            flags.push("-Zmiri-symbolic-alignment-check".to_string());
        }

        if self.track_raw_pointers {
            flags.push("-Zmiri-track-raw-pointers".to_string());
        }

        if let Some(seed) = self.seed {
            flags.push(format!("-Zmiri-seed={}", seed));
        }

        if let Some(rate) = self.preemption_rate {
            flags.push(format!("-Zmiri-preemption-rate={}", rate));
        }

        flags.extend(self.raw_flags.clone());

        flags.join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = MiriConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert!(config.cargo_path.is_none());
    }

    #[test]
    fn test_strict_config() {
        let config = MiriConfig::strict();
        assert!(config.flags.check_stacked_borrows);
        assert!(config.flags.track_raw_pointers);
        assert_eq!(config.flags.seed, Some(0));
    }

    #[test]
    fn test_permissive_config() {
        let config = MiriConfig::permissive();
        assert!(config.flags.disable_isolation);
        assert!(!config.flags.check_data_races);
    }

    #[test]
    fn test_miriflags_generation() {
        let flags = MiriFlags {
            disable_isolation: true,
            check_stacked_borrows: false,
            check_data_races: true,
            symbolic_alignment: true,
            track_raw_pointers: true,
            seed: Some(42),
            preemption_rate: Some(0.5),
            raw_flags: vec!["-Zmiri-custom=value".to_string()],
        };

        let miriflags = flags.to_miriflags();
        assert!(miriflags.contains("-Zmiri-disable-isolation"));
        assert!(miriflags.contains("-Zmiri-disable-stacked-borrows"));
        assert!(miriflags.contains("-Zmiri-symbolic-alignment-check"));
        assert!(miriflags.contains("-Zmiri-track-raw-pointers"));
        assert!(miriflags.contains("-Zmiri-seed=42"));
        assert!(miriflags.contains("-Zmiri-preemption-rate=0.5"));
        assert!(miriflags.contains("-Zmiri-custom=value"));
    }

    #[test]
    fn test_default_flags() {
        let flags = MiriFlags::default();
        let miriflags = flags.to_miriflags();
        // Default should have symbolic alignment enabled
        assert!(miriflags.contains("-Zmiri-symbolic-alignment-check"));
        // But not disable isolation
        assert!(!miriflags.contains("-Zmiri-disable-isolation"));
    }

    #[test]
    fn test_miriflags_check_data_races_disabled() {
        // Test the !check_data_races branch (line 148)
        let flags = MiriFlags {
            check_data_races: false,
            ..Default::default()
        };
        let miriflags = flags.to_miriflags();
        assert!(
            miriflags.contains("-Zmiri-disable-data-race-detector"),
            "Should have disable data race flag when check_data_races is false"
        );

        // Test check_data_races enabled does NOT have the disable flag
        let flags2 = MiriFlags {
            check_data_races: true,
            ..Default::default()
        };
        let miriflags2 = flags2.to_miriflags();
        assert!(
            !miriflags2.contains("-Zmiri-disable-data-race-detector"),
            "Should NOT have disable data race flag when check_data_races is true"
        );
    }
}
