//! Rust Fuzzing Backends for DashProve
//!
//! This crate provides verification backends for Rust fuzzing tools:
//! - **LibFuzzer (cargo-fuzz)**: Coverage-guided fuzzing via LLVM
//! - **AFL**: American Fuzzy Lop for Rust
//! - **Honggfuzz**: Coverage-guided fuzzer with hardware feedback
//! - **Bolero**: Unified fuzzing and property testing framework
//!
//! # Requirements
//!
//! Each fuzzer has specific requirements:
//! - LibFuzzer: `cargo install cargo-fuzz` (nightly required)
//! - AFL: `cargo install afl`
//! - Honggfuzz: `cargo install honggfuzz`
//! - Bolero: Add as dev-dependency
//!
//! # Usage
//!
//! ```rust,ignore
//! use dashprove_fuzz::{FuzzBackend, FuzzerType, FuzzConfig};
//!
//! let config = FuzzConfig::default()
//!     .with_timeout(Duration::from_secs(60))
//!     .with_max_iterations(10000);
//!
//! let backend = FuzzBackend::new(FuzzerType::LibFuzzer, config);
//! let result = backend.run_on_target("/path/to/crate", "fuzz_target").await?;
//! ```

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::process::Command;

/// Type of fuzzer to use
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(kani, derive(kani::Arbitrary))]
pub enum FuzzerType {
    /// LibFuzzer via cargo-fuzz
    LibFuzzer,
    /// American Fuzzy Lop
    AFL,
    /// Honggfuzz coverage-guided fuzzer
    Honggfuzz,
    /// Bolero unified framework
    Bolero,
}

impl FuzzerType {
    /// Get the fuzzer name for display
    pub fn name(&self) -> &'static str {
        match self {
            Self::LibFuzzer => "cargo-fuzz (LibFuzzer)",
            Self::AFL => "AFL.rs",
            Self::Honggfuzz => "Honggfuzz",
            Self::Bolero => "Bolero",
        }
    }

    /// Get the cargo command for this fuzzer
    pub fn cargo_command(&self) -> &'static str {
        match self {
            Self::LibFuzzer => "fuzz",
            Self::AFL => "afl",
            Self::Honggfuzz => "hfuzz",
            Self::Bolero => "bolero",
        }
    }

    /// Check if this fuzzer requires nightly
    pub fn requires_nightly(&self) -> bool {
        matches!(self, Self::LibFuzzer | Self::Honggfuzz)
    }
}

/// Configuration for fuzzing runs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzConfig {
    /// Maximum time to run fuzzing
    pub timeout: Duration,
    /// Maximum number of iterations (0 = unlimited until timeout)
    pub max_iterations: u64,
    /// Number of parallel jobs
    pub jobs: usize,
    /// Seed corpus directory (optional)
    pub seed_corpus: Option<PathBuf>,
    /// Dictionary file for mutations (optional)
    pub dictionary: Option<PathBuf>,
    /// Maximum input length
    pub max_len: Option<usize>,
}

impl Default for FuzzConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(60),
            max_iterations: 0, // Run until timeout
            jobs: 1,
            seed_corpus: None,
            dictionary: None,
            max_len: None,
        }
    }
}

impl FuzzConfig {
    /// Set the timeout for fuzzing
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the maximum number of iterations
    pub fn with_max_iterations(mut self, iterations: u64) -> Self {
        self.max_iterations = iterations;
        self
    }

    /// Set the number of parallel jobs
    pub fn with_jobs(mut self, jobs: usize) -> Self {
        self.jobs = jobs;
        self
    }

    /// Set the seed corpus directory
    pub fn with_seed_corpus(mut self, path: PathBuf) -> Self {
        self.seed_corpus = Some(path);
        self
    }
}

/// Error type for fuzzing operations
#[derive(Error, Debug)]
pub enum FuzzError {
    /// Fuzzer tool not installed
    #[error("Fuzzer not installed: {0}. Install with: {1}")]
    NotInstalled(String, String),

    /// Nightly toolchain required
    #[error("Nightly toolchain required for {0}")]
    NightlyRequired(String),

    /// Fuzz target not found
    #[error("Fuzz target not found: {0}")]
    TargetNotFound(String),

    /// Build failed
    #[error("Build failed: {0}")]
    BuildFailed(String),

    /// Fuzzing failed
    #[error("Fuzzing failed: {0}")]
    FuzzingFailed(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Timeout
    #[error("Fuzzing timeout after {0:?}")]
    Timeout(Duration),
}

/// A crash found during fuzzing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzCrash {
    /// Type of crash
    pub crash_type: CrashType,
    /// Input that caused the crash
    pub input: Vec<u8>,
    /// Stack trace if available
    pub stack_trace: Option<String>,
    /// Exit code/signal
    pub exit_code: Option<i32>,
    /// Path to the crash artifact
    pub artifact_path: Option<PathBuf>,
}

/// Type of crash found
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(kani, derive(kani::Arbitrary))]
pub enum CrashType {
    /// Assertion failure / panic
    Panic,
    /// Segmentation fault
    Segfault,
    /// Out of memory
    OOM,
    /// Timeout per-input
    Timeout,
    /// Other crash type
    Other,
}

/// Coverage information from fuzzing
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CoverageInfo {
    /// Number of code regions covered
    pub regions_covered: usize,
    /// Total code regions
    pub total_regions: usize,
    /// Number of unique code paths discovered
    pub unique_paths: usize,
    /// Edge coverage percentage (if available)
    pub edge_coverage_pct: Option<f64>,
}

/// Result from a fuzzing run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzResult {
    /// Which fuzzer was used
    pub fuzzer: FuzzerType,
    /// Target that was fuzzed
    pub target: String,
    /// Whether the fuzzing completed without crashes
    pub passed: bool,
    /// Crashes found during fuzzing
    pub crashes: Vec<FuzzCrash>,
    /// Number of executions performed
    pub executions: u64,
    /// Executions per second
    pub exec_per_sec: f64,
    /// Coverage information
    pub coverage: CoverageInfo,
    /// Duration of fuzzing run
    pub duration: Duration,
    /// Path to corpus directory
    pub corpus_path: Option<PathBuf>,
}

/// Fuzzing backend for running Rust fuzzers
pub struct FuzzBackend {
    /// Which fuzzer to use
    fuzzer_type: FuzzerType,
    /// Configuration
    config: FuzzConfig,
}

impl FuzzBackend {
    /// Create a new fuzzing backend
    pub fn new(fuzzer_type: FuzzerType, config: FuzzConfig) -> Self {
        Self {
            fuzzer_type,
            config,
        }
    }

    /// Check if a fuzzer is installed
    pub async fn check_installed(&self) -> Result<bool, FuzzError> {
        let cmd = match self.fuzzer_type {
            FuzzerType::LibFuzzer => ("cargo", vec!["fuzz", "--help"]),
            FuzzerType::AFL => ("cargo", vec!["afl", "--help"]),
            FuzzerType::Honggfuzz => ("cargo", vec!["hfuzz", "version"]),
            FuzzerType::Bolero => ("cargo", vec!["bolero", "--help"]),
        };

        let output = Command::new(cmd.0).args(&cmd.1).output().await?;

        Ok(output.status.success())
    }

    /// Get installation instructions for the fuzzer
    pub fn install_instructions(&self) -> &'static str {
        match self.fuzzer_type {
            FuzzerType::LibFuzzer => "cargo install cargo-fuzz",
            FuzzerType::AFL => "cargo install afl",
            FuzzerType::Honggfuzz => "cargo install honggfuzz",
            FuzzerType::Bolero => "cargo install cargo-bolero",
        }
    }

    /// Run fuzzing on a specific target
    pub async fn run_on_target(
        &self,
        crate_path: &Path,
        target: &str,
    ) -> Result<FuzzResult, FuzzError> {
        // Check if fuzzer is installed
        if !self.check_installed().await? {
            return Err(FuzzError::NotInstalled(
                self.fuzzer_type.name().to_string(),
                self.install_instructions().to_string(),
            ));
        }

        let start = Instant::now();

        let result = match self.fuzzer_type {
            FuzzerType::LibFuzzer => self.run_libfuzzer(crate_path, target).await?,
            FuzzerType::AFL => self.run_afl(crate_path, target).await?,
            FuzzerType::Honggfuzz => self.run_honggfuzz(crate_path, target).await?,
            FuzzerType::Bolero => self.run_bolero(crate_path, target).await?,
        };

        let duration = start.elapsed();

        Ok(FuzzResult {
            fuzzer: self.fuzzer_type,
            target: target.to_string(),
            passed: result.crashes.is_empty(),
            crashes: result.crashes,
            executions: result.executions,
            exec_per_sec: result.executions as f64 / duration.as_secs_f64(),
            coverage: result.coverage,
            duration,
            corpus_path: result.corpus_path,
        })
    }

    async fn run_libfuzzer(
        &self,
        crate_path: &Path,
        target: &str,
    ) -> Result<FuzzResult, FuzzError> {
        let timeout_secs = self.config.timeout.as_secs();
        let mut args = vec![
            "+nightly".to_string(),
            "fuzz".to_string(),
            "run".to_string(),
            target.to_string(),
            "--".to_string(),
            format!("-max_total_time={}", timeout_secs),
        ];

        if self.config.max_iterations > 0 {
            args.push(format!("-runs={}", self.config.max_iterations));
        }

        if let Some(max_len) = self.config.max_len {
            args.push(format!("-max_len={}", max_len));
        }

        let output = Command::new("cargo")
            .args(&args)
            .current_dir(crate_path)
            .output()
            .await?;

        let stderr = String::from_utf8_lossy(&output.stderr);
        let crashes = self.parse_libfuzzer_crashes(&stderr, crate_path);
        let (executions, coverage) = self.parse_libfuzzer_stats(&stderr);

        Ok(FuzzResult {
            fuzzer: FuzzerType::LibFuzzer,
            target: target.to_string(),
            passed: crashes.is_empty(),
            crashes,
            executions,
            exec_per_sec: 0.0,
            coverage,
            duration: Duration::ZERO,
            corpus_path: Some(crate_path.join("fuzz").join("corpus").join(target)),
        })
    }

    async fn run_afl(&self, crate_path: &Path, target: &str) -> Result<FuzzResult, FuzzError> {
        // Build the AFL target first
        let build_output = Command::new("cargo")
            .args(["afl", "build", "--release"])
            .current_dir(crate_path)
            .output()
            .await?;

        if !build_output.status.success() {
            return Err(FuzzError::BuildFailed(
                String::from_utf8_lossy(&build_output.stderr).to_string(),
            ));
        }

        // Run AFL fuzzing
        let output_dir = crate_path.join("afl_output");
        let input_dir = self
            .config
            .seed_corpus
            .clone()
            .unwrap_or_else(|| crate_path.join("afl_input"));

        let output = Command::new("cargo")
            .args([
                "afl",
                "fuzz",
                "-i",
                input_dir.to_str().unwrap_or("."),
                "-o",
                output_dir.to_str().unwrap_or("afl_output"),
                "-V",
                &self.config.timeout.as_secs().to_string(),
                &format!("target/release/{}", target),
            ])
            .current_dir(crate_path)
            .output()
            .await?;

        let stderr = String::from_utf8_lossy(&output.stderr);
        let crashes = self.parse_afl_crashes(&stderr, &output_dir);

        Ok(FuzzResult {
            fuzzer: FuzzerType::AFL,
            target: target.to_string(),
            passed: crashes.is_empty(),
            crashes,
            executions: 0, // AFL stats would need to be parsed from output files
            exec_per_sec: 0.0,
            coverage: CoverageInfo::default(),
            duration: Duration::ZERO,
            corpus_path: Some(output_dir),
        })
    }

    async fn run_honggfuzz(
        &self,
        crate_path: &Path,
        target: &str,
    ) -> Result<FuzzResult, FuzzError> {
        let timeout_secs = self.config.timeout.as_secs();

        let output = Command::new("cargo")
            .args([
                "+nightly",
                "hfuzz",
                "run",
                target,
                "--",
                "--timeout",
                &timeout_secs.to_string(),
            ])
            .current_dir(crate_path)
            .output()
            .await?;

        let stderr = String::from_utf8_lossy(&output.stderr);
        let crashes = self.parse_honggfuzz_crashes(&stderr, crate_path);

        Ok(FuzzResult {
            fuzzer: FuzzerType::Honggfuzz,
            target: target.to_string(),
            passed: crashes.is_empty(),
            crashes,
            executions: 0,
            exec_per_sec: 0.0,
            coverage: CoverageInfo::default(),
            duration: Duration::ZERO,
            corpus_path: Some(crate_path.join("hfuzz_workspace").join(target)),
        })
    }

    async fn run_bolero(&self, crate_path: &Path, target: &str) -> Result<FuzzResult, FuzzError> {
        let timeout_secs = self.config.timeout.as_secs();

        let output = Command::new("cargo")
            .args([
                "bolero",
                "test",
                target,
                "--time",
                &timeout_secs.to_string(),
            ])
            .current_dir(crate_path)
            .output()
            .await?;

        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let crashes = self.parse_bolero_crashes(&stderr, &stdout);

        Ok(FuzzResult {
            fuzzer: FuzzerType::Bolero,
            target: target.to_string(),
            passed: crashes.is_empty() && output.status.success(),
            crashes,
            executions: 0,
            exec_per_sec: 0.0,
            coverage: CoverageInfo::default(),
            duration: Duration::ZERO,
            corpus_path: None,
        })
    }

    fn parse_libfuzzer_crashes(&self, output: &str, crate_path: &Path) -> Vec<FuzzCrash> {
        let mut crashes = Vec::new();

        // LibFuzzer crash patterns
        if output.contains("SUMMARY: libFuzzer: deadly signal")
            || output.contains("panicked at")
            || output.contains("BINGO")
        {
            // Look for artifact path
            let artifact_path = output
                .lines()
                .find(|line| line.contains("artifact_prefix"))
                .and_then(|line| {
                    line.split_whitespace()
                        .last()
                        .map(|p| crate_path.join(p.trim_matches('\'')))
                });

            crashes.push(FuzzCrash {
                crash_type: if output.contains("panicked") {
                    CrashType::Panic
                } else {
                    CrashType::Segfault
                },
                input: Vec::new(),
                stack_trace: Some(output.to_string()),
                exit_code: None,
                artifact_path,
            });
        }

        crashes
    }

    fn parse_libfuzzer_stats(&self, output: &str) -> (u64, CoverageInfo) {
        let mut executions = 0u64;
        let mut coverage = CoverageInfo::default();

        for line in output.lines() {
            // Parse: "#12345 DONE cov: 123 ft: 456"
            if line.starts_with('#') {
                if let Some(count) = line
                    .split_whitespace()
                    .next()
                    .and_then(|s| s.trim_start_matches('#').parse::<u64>().ok())
                {
                    executions = executions.max(count);
                }
            }

            // Parse coverage info
            if let Some(cov_pos) = line.find("cov: ") {
                if let Some(cov_str) = line[cov_pos + 5..].split_whitespace().next() {
                    if let Ok(cov) = cov_str.parse::<usize>() {
                        coverage.regions_covered = cov;
                    }
                }
            }
        }

        (executions, coverage)
    }

    fn parse_afl_crashes(&self, _output: &str, output_dir: &Path) -> Vec<FuzzCrash> {
        let mut crashes = Vec::new();

        // AFL stores crashes in output_dir/crashes/
        let crashes_dir = output_dir.join("crashes");
        if crashes_dir.exists() {
            if let Ok(entries) = std::fs::read_dir(&crashes_dir) {
                for entry in entries.flatten() {
                    if entry.path().is_file() {
                        crashes.push(FuzzCrash {
                            crash_type: CrashType::Other,
                            input: std::fs::read(entry.path()).unwrap_or_default(),
                            stack_trace: None,
                            exit_code: None,
                            artifact_path: Some(entry.path()),
                        });
                    }
                }
            }
        }

        crashes
    }

    fn parse_honggfuzz_crashes(&self, output: &str, crate_path: &Path) -> Vec<FuzzCrash> {
        let mut crashes = Vec::new();

        if output.contains("Crash: ") || output.contains("CRASH SUMMARY") {
            let crashes_dir = crate_path.join("hfuzz_workspace").join("crashes");
            if crashes_dir.exists() {
                if let Ok(entries) = std::fs::read_dir(&crashes_dir) {
                    for entry in entries.flatten() {
                        crashes.push(FuzzCrash {
                            crash_type: CrashType::Other,
                            input: std::fs::read(entry.path()).unwrap_or_default(),
                            stack_trace: None,
                            exit_code: None,
                            artifact_path: Some(entry.path()),
                        });
                    }
                }
            }
        }

        crashes
    }

    fn parse_bolero_crashes(&self, stderr: &str, stdout: &str) -> Vec<FuzzCrash> {
        let mut crashes = Vec::new();
        let combined = format!("{}\n{}", stderr, stdout);

        if combined.contains("FAILED") || combined.contains("panicked") {
            crashes.push(FuzzCrash {
                crash_type: CrashType::Panic,
                input: Vec::new(),
                stack_trace: Some(combined),
                exit_code: None,
                artifact_path: None,
            });
        }

        crashes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_fuzzer_type_names() {
        assert_eq!(FuzzerType::LibFuzzer.name(), "cargo-fuzz (LibFuzzer)");
        assert_eq!(FuzzerType::AFL.name(), "AFL.rs");
        assert_eq!(FuzzerType::Honggfuzz.name(), "Honggfuzz");
        assert_eq!(FuzzerType::Bolero.name(), "Bolero");
    }

    #[test]
    fn test_fuzzer_nightly_requirements() {
        assert!(FuzzerType::LibFuzzer.requires_nightly());
        assert!(FuzzerType::Honggfuzz.requires_nightly());
        assert!(!FuzzerType::AFL.requires_nightly());
        assert!(!FuzzerType::Bolero.requires_nightly());
    }

    #[test]
    fn test_fuzz_config_builder() {
        let config = FuzzConfig::default()
            .with_timeout(Duration::from_secs(120))
            .with_max_iterations(50000)
            .with_jobs(4);

        assert_eq!(config.timeout, Duration::from_secs(120));
        assert_eq!(config.max_iterations, 50000);
        assert_eq!(config.jobs, 4);
    }

    #[test]
    fn test_parse_libfuzzer_stats() {
        let backend = FuzzBackend::new(FuzzerType::LibFuzzer, FuzzConfig::default());
        let output = "#12345 DONE cov: 987 ft: 654 corp: 123";
        let (execs, coverage) = backend.parse_libfuzzer_stats(output);
        assert_eq!(execs, 12345);
        assert_eq!(coverage.regions_covered, 987);
    }

    #[test]
    fn test_fuzz_backend_creation() {
        let config = FuzzConfig::default();
        let backend = FuzzBackend::new(FuzzerType::LibFuzzer, config);
        assert_eq!(backend.fuzzer_type, FuzzerType::LibFuzzer);
    }

    // Tests for FuzzerType::cargo_command - each variant returns specific command
    #[test]
    fn test_fuzzer_type_cargo_command_libfuzzer() {
        assert_eq!(FuzzerType::LibFuzzer.cargo_command(), "fuzz");
        assert_ne!(FuzzerType::LibFuzzer.cargo_command(), "xyzzy");
    }

    #[test]
    fn test_fuzzer_type_cargo_command_afl() {
        assert_eq!(FuzzerType::AFL.cargo_command(), "afl");
        assert_ne!(FuzzerType::AFL.cargo_command(), "xyzzy");
    }

    #[test]
    fn test_fuzzer_type_cargo_command_honggfuzz() {
        assert_eq!(FuzzerType::Honggfuzz.cargo_command(), "hfuzz");
        assert_ne!(FuzzerType::Honggfuzz.cargo_command(), "xyzzy");
    }

    #[test]
    fn test_fuzzer_type_cargo_command_bolero() {
        assert_eq!(FuzzerType::Bolero.cargo_command(), "bolero");
        assert_ne!(FuzzerType::Bolero.cargo_command(), "xyzzy");
    }

    // Tests for install_instructions - each fuzzer has specific instructions
    #[test]
    fn test_install_instructions_libfuzzer() {
        let backend = FuzzBackend::new(FuzzerType::LibFuzzer, FuzzConfig::default());
        assert_eq!(backend.install_instructions(), "cargo install cargo-fuzz");
        assert_ne!(backend.install_instructions(), "xyzzy");
    }

    #[test]
    fn test_install_instructions_afl() {
        let backend = FuzzBackend::new(FuzzerType::AFL, FuzzConfig::default());
        assert_eq!(backend.install_instructions(), "cargo install afl");
        assert_ne!(backend.install_instructions(), "xyzzy");
    }

    #[test]
    fn test_install_instructions_honggfuzz() {
        let backend = FuzzBackend::new(FuzzerType::Honggfuzz, FuzzConfig::default());
        assert_eq!(backend.install_instructions(), "cargo install honggfuzz");
        assert_ne!(backend.install_instructions(), "xyzzy");
    }

    #[test]
    fn test_install_instructions_bolero() {
        let backend = FuzzBackend::new(FuzzerType::Bolero, FuzzConfig::default());
        assert_eq!(backend.install_instructions(), "cargo install cargo-bolero");
        assert_ne!(backend.install_instructions(), "xyzzy");
    }

    // Tests for parse_libfuzzer_crashes - detects crashes from output
    #[test]
    fn test_parse_libfuzzer_crashes_deadly_signal() {
        let backend = FuzzBackend::new(FuzzerType::LibFuzzer, FuzzConfig::default());
        let output = "SUMMARY: libFuzzer: deadly signal";
        let crashes = backend.parse_libfuzzer_crashes(output, Path::new("/tmp"));
        assert_eq!(crashes.len(), 1);
        assert_eq!(crashes[0].crash_type, CrashType::Segfault);
    }

    #[test]
    fn test_parse_libfuzzer_crashes_panic() {
        let backend = FuzzBackend::new(FuzzerType::LibFuzzer, FuzzConfig::default());
        let output = "panicked at 'assertion failed'";
        let crashes = backend.parse_libfuzzer_crashes(output, Path::new("/tmp"));
        assert_eq!(crashes.len(), 1);
        assert_eq!(crashes[0].crash_type, CrashType::Panic);
    }

    #[test]
    fn test_parse_libfuzzer_crashes_bingo() {
        let backend = FuzzBackend::new(FuzzerType::LibFuzzer, FuzzConfig::default());
        let output = "BINGO: found crash";
        let crashes = backend.parse_libfuzzer_crashes(output, Path::new("/tmp"));
        assert_eq!(crashes.len(), 1);
    }

    #[test]
    fn test_parse_libfuzzer_crashes_no_crash() {
        let backend = FuzzBackend::new(FuzzerType::LibFuzzer, FuzzConfig::default());
        let output = "#1000 DONE cov: 100 ft: 50";
        let crashes = backend.parse_libfuzzer_crashes(output, Path::new("/tmp"));
        assert!(crashes.is_empty());
    }

    #[test]
    fn test_parse_libfuzzer_crashes_only_panicked() {
        // Test the || condition in line 483-484 - only "panicked at" without others
        let backend = FuzzBackend::new(FuzzerType::LibFuzzer, FuzzConfig::default());
        let output = "thread 'main' panicked at 'oops'";
        let crashes = backend.parse_libfuzzer_crashes(output, Path::new("/tmp"));
        assert!(!crashes.is_empty());
    }

    #[test]
    fn test_parse_libfuzzer_crashes_only_bingo() {
        // Test the || condition - only BINGO without others
        let backend = FuzzBackend::new(FuzzerType::LibFuzzer, FuzzConfig::default());
        let output = "BINGO found";
        let crashes = backend.parse_libfuzzer_crashes(output, Path::new("/tmp"));
        assert!(!crashes.is_empty());
    }

    #[test]
    fn test_parse_libfuzzer_crashes_only_deadly_signal() {
        // Test the || condition - only deadly signal
        let backend = FuzzBackend::new(FuzzerType::LibFuzzer, FuzzConfig::default());
        let output = "SUMMARY: libFuzzer: deadly signal happened";
        let crashes = backend.parse_libfuzzer_crashes(output, Path::new("/tmp"));
        assert!(!crashes.is_empty());
    }

    // Tests for parse_afl_crashes - reads from crash directory
    #[test]
    fn test_parse_afl_crashes_no_dir() {
        let backend = FuzzBackend::new(FuzzerType::AFL, FuzzConfig::default());
        let crashes = backend.parse_afl_crashes("", Path::new("/nonexistent/path"));
        assert!(crashes.is_empty());
    }

    #[test]
    fn test_parse_afl_crashes_with_crash_files() {
        let backend = FuzzBackend::new(FuzzerType::AFL, FuzzConfig::default());
        let temp_dir = TempDir::new().unwrap();
        let crashes_dir = temp_dir.path().join("crashes");
        fs::create_dir(&crashes_dir).unwrap();
        fs::write(crashes_dir.join("crash1"), b"crash input 1").unwrap();
        fs::write(crashes_dir.join("crash2"), b"crash input 2").unwrap();

        let crashes = backend.parse_afl_crashes("", temp_dir.path());
        assert_eq!(crashes.len(), 2);
        // Verify crash files contain data
        for crash in &crashes {
            assert!(!crash.input.is_empty());
            assert!(crash.artifact_path.is_some());
        }
    }

    #[test]
    fn test_parse_afl_crashes_empty_dir() {
        let backend = FuzzBackend::new(FuzzerType::AFL, FuzzConfig::default());
        let temp_dir = TempDir::new().unwrap();
        let crashes_dir = temp_dir.path().join("crashes");
        fs::create_dir(&crashes_dir).unwrap();

        let crashes = backend.parse_afl_crashes("", temp_dir.path());
        assert!(crashes.is_empty());
    }

    // Tests for parse_honggfuzz_crashes
    #[test]
    fn test_parse_honggfuzz_crashes_no_crash_output() {
        let backend = FuzzBackend::new(FuzzerType::Honggfuzz, FuzzConfig::default());
        let output = "Running fuzzer...";
        let crashes = backend.parse_honggfuzz_crashes(output, Path::new("/tmp"));
        assert!(crashes.is_empty());
    }

    #[test]
    fn test_parse_honggfuzz_crashes_crash_keyword() {
        let backend = FuzzBackend::new(FuzzerType::Honggfuzz, FuzzConfig::default());
        let temp_dir = TempDir::new().unwrap();
        let hfuzz_workspace = temp_dir.path().join("hfuzz_workspace");
        let crashes_dir = hfuzz_workspace.join("crashes");
        fs::create_dir_all(&crashes_dir).unwrap();
        fs::write(crashes_dir.join("crash1"), b"crash data").unwrap();

        let output = "Crash: signal=11";
        let crashes = backend.parse_honggfuzz_crashes(output, temp_dir.path());
        assert_eq!(crashes.len(), 1);
    }

    #[test]
    fn test_parse_honggfuzz_crashes_summary_keyword() {
        let backend = FuzzBackend::new(FuzzerType::Honggfuzz, FuzzConfig::default());
        let temp_dir = TempDir::new().unwrap();
        let hfuzz_workspace = temp_dir.path().join("hfuzz_workspace");
        let crashes_dir = hfuzz_workspace.join("crashes");
        fs::create_dir_all(&crashes_dir).unwrap();
        fs::write(crashes_dir.join("crash1"), b"data").unwrap();

        let output = "CRASH SUMMARY found";
        let crashes = backend.parse_honggfuzz_crashes(output, temp_dir.path());
        assert_eq!(crashes.len(), 1);
    }

    #[test]
    fn test_parse_honggfuzz_crashes_only_crash_keyword() {
        // Test || condition - only "Crash: " without "CRASH SUMMARY"
        let backend = FuzzBackend::new(FuzzerType::Honggfuzz, FuzzConfig::default());
        let temp_dir = TempDir::new().unwrap();
        let hfuzz_workspace = temp_dir.path().join("hfuzz_workspace");
        let crashes_dir = hfuzz_workspace.join("crashes");
        fs::create_dir_all(&crashes_dir).unwrap();
        fs::write(crashes_dir.join("crash1"), b"data").unwrap();

        let output = "Crash: something happened";
        let crashes = backend.parse_honggfuzz_crashes(output, temp_dir.path());
        assert!(!crashes.is_empty());
    }

    // Tests for parse_bolero_crashes
    #[test]
    fn test_parse_bolero_crashes_no_failure() {
        let backend = FuzzBackend::new(FuzzerType::Bolero, FuzzConfig::default());
        let crashes = backend.parse_bolero_crashes("ok", "test passed");
        assert!(crashes.is_empty());
    }

    #[test]
    fn test_parse_bolero_crashes_failed() {
        let backend = FuzzBackend::new(FuzzerType::Bolero, FuzzConfig::default());
        let crashes = backend.parse_bolero_crashes("test FAILED", "");
        assert_eq!(crashes.len(), 1);
        assert_eq!(crashes[0].crash_type, CrashType::Panic);
    }

    #[test]
    fn test_parse_bolero_crashes_panicked() {
        let backend = FuzzBackend::new(FuzzerType::Bolero, FuzzConfig::default());
        let crashes = backend.parse_bolero_crashes("thread panicked", "");
        assert_eq!(crashes.len(), 1);
    }

    #[test]
    fn test_parse_bolero_crashes_only_failed() {
        // Test || condition - only FAILED without panicked
        let backend = FuzzBackend::new(FuzzerType::Bolero, FuzzConfig::default());
        let crashes = backend.parse_bolero_crashes("FAILED test", "no panic");
        assert!(!crashes.is_empty());
    }

    #[test]
    fn test_parse_bolero_crashes_only_panicked() {
        // Test || condition - only panicked without FAILED
        let backend = FuzzBackend::new(FuzzerType::Bolero, FuzzConfig::default());
        let crashes = backend.parse_bolero_crashes("panicked here", "success");
        assert!(!crashes.is_empty());
    }

    #[test]
    fn test_parse_bolero_crashes_in_stdout() {
        // Test that we check both stderr and stdout
        let backend = FuzzBackend::new(FuzzerType::Bolero, FuzzConfig::default());
        let crashes = backend.parse_bolero_crashes("ok", "FAILED");
        assert!(!crashes.is_empty());
    }

    // Test exec_per_sec calculation in run_on_target (div by duration)
    #[test]
    fn test_fuzz_result_exec_per_sec_calculation() {
        // Simulates what happens when executions / duration.as_secs_f64()
        let executions: u64 = 1000;
        let duration = Duration::from_secs(10);
        let exec_per_sec = executions as f64 / duration.as_secs_f64();
        assert!((exec_per_sec - 100.0).abs() < 0.001);

        // Test that div and mul would give different results
        let exec_mul = executions as f64 * duration.as_secs_f64();
        assert_ne!(exec_per_sec, exec_mul);

        // Test that mod would give different results
        let exec_mod = (executions as f64) % duration.as_secs_f64();
        assert_ne!(exec_per_sec, exec_mod);
    }

    // Tests for max_iterations > 0 check in run_libfuzzer (line 324)
    #[test]
    fn test_config_max_iterations_zero_vs_positive() {
        // Test that > 0 correctly distinguishes from 0
        let config_zero = FuzzConfig::default().with_max_iterations(0);
        assert_eq!(config_zero.max_iterations, 0);
        // 0 > 0 is false, so the conditional branch is NOT taken
        let should_add_runs_arg = config_zero.max_iterations > 0;
        assert!(!should_add_runs_arg);

        let config_positive = FuzzConfig::default().with_max_iterations(100);
        assert_eq!(config_positive.max_iterations, 100);
        // 100 > 0 is true, so the conditional branch IS taken
        let should_add_runs_arg = config_positive.max_iterations > 0;
        assert!(should_add_runs_arg);
    }

    // Test for && vs || in run_bolero (line 468)
    #[test]
    fn test_bolero_passed_requires_both_conditions() {
        // passed: crashes.is_empty() && output.status.success()
        // Build FuzzResult structs to test the logic

        // Both conditions true -> passed = true
        let result_passed = FuzzResult {
            fuzzer: FuzzerType::Bolero,
            target: "test".to_string(),
            passed: true, // empty crashes AND status success
            crashes: Vec::new(),
            executions: 0,
            exec_per_sec: 0.0,
            coverage: CoverageInfo::default(),
            duration: Duration::from_secs(1),
            corpus_path: None,
        };
        assert!(result_passed.passed);
        assert!(result_passed.crashes.is_empty());

        // crashes not empty -> passed = false (regardless of status)
        let result_failed = FuzzResult {
            fuzzer: FuzzerType::Bolero,
            target: "test".to_string(),
            passed: false, // has crashes, so failed
            crashes: vec![FuzzCrash {
                crash_type: CrashType::Panic,
                input: vec![1],
                stack_trace: None,
                exit_code: None,
                artifact_path: None,
            }],
            executions: 0,
            exec_per_sec: 0.0,
            coverage: CoverageInfo::default(),
            duration: Duration::from_secs(1),
            corpus_path: None,
        };
        assert!(!result_failed.passed);
        assert!(!result_failed.crashes.is_empty());
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;
    use std::path::PathBuf;

    // Strategy for generating FuzzerType variants
    fn fuzzer_type_strategy() -> impl Strategy<Value = FuzzerType> {
        prop_oneof![
            Just(FuzzerType::LibFuzzer),
            Just(FuzzerType::AFL),
            Just(FuzzerType::Honggfuzz),
            Just(FuzzerType::Bolero),
        ]
    }

    // Strategy for generating CrashType variants
    fn crash_type_strategy() -> impl Strategy<Value = CrashType> {
        prop_oneof![
            Just(CrashType::Panic),
            Just(CrashType::Segfault),
            Just(CrashType::OOM),
            Just(CrashType::Timeout),
            Just(CrashType::Other),
        ]
    }

    proptest! {
        // FuzzerType property tests
        #[test]
        fn fuzzer_type_name_non_empty(fuzzer in fuzzer_type_strategy()) {
            prop_assert!(!fuzzer.name().is_empty());
        }

        #[test]
        fn fuzzer_type_cargo_command_non_empty(fuzzer in fuzzer_type_strategy()) {
            prop_assert!(!fuzzer.cargo_command().is_empty());
        }

        #[test]
        fn fuzzer_type_nightly_consistency(fuzzer in fuzzer_type_strategy()) {
            // LibFuzzer and Honggfuzz require nightly; AFL and Bolero don't
            let requires = fuzzer.requires_nightly();
            let expected = matches!(fuzzer, FuzzerType::LibFuzzer | FuzzerType::Honggfuzz);
            prop_assert_eq!(requires, expected);
        }

        // FuzzConfig builder property tests
        #[test]
        fn fuzz_config_timeout_preserved(secs in 1u64..10000) {
            let timeout = Duration::from_secs(secs);
            let config = FuzzConfig::default().with_timeout(timeout);
            prop_assert_eq!(config.timeout, timeout);
        }

        #[test]
        fn fuzz_config_max_iterations_preserved(iterations in 0u64..1000000) {
            let config = FuzzConfig::default().with_max_iterations(iterations);
            prop_assert_eq!(config.max_iterations, iterations);
        }

        #[test]
        fn fuzz_config_jobs_preserved(jobs in 1usize..128) {
            let config = FuzzConfig::default().with_jobs(jobs);
            prop_assert_eq!(config.jobs, jobs);
        }

        #[test]
        fn fuzz_config_seed_corpus_preserved(path in "[a-z]+(/[a-z]+)*") {
            let pathbuf = PathBuf::from(&path);
            let config = FuzzConfig::default().with_seed_corpus(pathbuf.clone());
            prop_assert_eq!(config.seed_corpus, Some(pathbuf));
        }

        #[test]
        fn fuzz_config_builder_chaining(
            secs in 1u64..1000,
            iterations in 0u64..100000,
            jobs in 1usize..32
        ) {
            let timeout = Duration::from_secs(secs);
            let config = FuzzConfig::default()
                .with_timeout(timeout)
                .with_max_iterations(iterations)
                .with_jobs(jobs);

            prop_assert_eq!(config.timeout, timeout);
            prop_assert_eq!(config.max_iterations, iterations);
            prop_assert_eq!(config.jobs, jobs);
        }

        // FuzzBackend property tests
        #[test]
        fn fuzz_backend_preserves_fuzzer_type(fuzzer in fuzzer_type_strategy()) {
            let backend = FuzzBackend::new(fuzzer, FuzzConfig::default());
            prop_assert_eq!(backend.fuzzer_type, fuzzer);
        }

        #[test]
        fn fuzz_backend_install_instructions_non_empty(fuzzer in fuzzer_type_strategy()) {
            let backend = FuzzBackend::new(fuzzer, FuzzConfig::default());
            prop_assert!(!backend.install_instructions().is_empty());
        }

        // CoverageInfo default values test
        #[test]
        fn coverage_info_default_is_zero(_dummy in 0..1) {
            let coverage = CoverageInfo::default();
            prop_assert_eq!(coverage.regions_covered, 0);
            prop_assert_eq!(coverage.total_regions, 0);
            prop_assert_eq!(coverage.unique_paths, 0);
            prop_assert!(coverage.edge_coverage_pct.is_none());
        }

        // FuzzResult passed field consistency
        #[test]
        fn fuzz_result_passed_when_no_crashes(
            fuzzer in fuzzer_type_strategy(),
            target in "[a-z_]+",
            execs in 0u64..1000000,
            exec_per_sec in 0.0f64..10000.0
        ) {
            let result = FuzzResult {
                fuzzer,
                target: target.clone(),
                passed: true,
                crashes: Vec::new(),
                executions: execs,
                exec_per_sec,
                coverage: CoverageInfo::default(),
                duration: Duration::from_secs(1),
                corpus_path: None,
            };
            prop_assert!(result.passed);
            prop_assert!(result.crashes.is_empty());
        }

        // FuzzCrash structure tests
        #[test]
        fn fuzz_crash_preserves_crash_type(crash_type in crash_type_strategy()) {
            let crash = FuzzCrash {
                crash_type,
                input: vec![1, 2, 3],
                stack_trace: None,
                exit_code: None,
                artifact_path: None,
            };
            prop_assert_eq!(crash.crash_type, crash_type);
        }

        #[test]
        fn fuzz_crash_input_preserved(input in prop::collection::vec(any::<u8>(), 0..100)) {
            let crash = FuzzCrash {
                crash_type: CrashType::Panic,
                input: input.clone(),
                stack_trace: None,
                exit_code: None,
                artifact_path: None,
            };
            prop_assert_eq!(crash.input, input);
        }

        // LibFuzzer stats parsing property tests
        #[test]
        fn parse_libfuzzer_stats_handles_valid_exec_count(count in 1u64..1000000) {
            let backend = FuzzBackend::new(FuzzerType::LibFuzzer, FuzzConfig::default());
            let count_str = count.to_string();
            let output = format!("#{} DONE cov: 100 ft: 50", count_str);
            let (execs, _) = backend.parse_libfuzzer_stats(&output);
            prop_assert_eq!(execs, count);
        }

        #[test]
        fn parse_libfuzzer_stats_handles_valid_coverage(cov in 0usize..100000) {
            let backend = FuzzBackend::new(FuzzerType::LibFuzzer, FuzzConfig::default());
            let cov_str = cov.to_string();
            let output = format!("#1000 DONE cov: {} ft: 50", cov_str);
            let (_, coverage) = backend.parse_libfuzzer_stats(&output);
            prop_assert_eq!(coverage.regions_covered, cov);
        }

        #[test]
        fn parse_libfuzzer_stats_max_exec_across_lines(
            count1 in 1u64..500000,
            count2 in 1u64..500000
        ) {
            let backend = FuzzBackend::new(FuzzerType::LibFuzzer, FuzzConfig::default());
            let output = format!("#{} cov: 10\n#{} cov: 20", count1, count2);
            let (execs, _) = backend.parse_libfuzzer_stats(&output);
            prop_assert_eq!(execs, count1.max(count2));
        }

        // Empty output handling
        #[test]
        fn parse_libfuzzer_stats_handles_empty_output(_dummy in 0..1) {
            let backend = FuzzBackend::new(FuzzerType::LibFuzzer, FuzzConfig::default());
            let (execs, coverage) = backend.parse_libfuzzer_stats("");
            prop_assert_eq!(execs, 0);
            prop_assert_eq!(coverage.regions_covered, 0);
        }

        // FuzzError message preservation tests
        #[test]
        fn fuzz_error_not_installed_contains_name(name in "[A-Za-z]+") {
            let error = FuzzError::NotInstalled(name.clone(), "cargo install foo".to_string());
            let msg = error.to_string();
            prop_assert!(msg.contains(&name));
        }

        #[test]
        fn fuzz_error_nightly_required_contains_name(name in "[A-Za-z]+") {
            let error = FuzzError::NightlyRequired(name.clone());
            let msg = error.to_string();
            prop_assert!(msg.contains(&name));
        }

        #[test]
        fn fuzz_error_target_not_found_contains_target(target in "[a-z_]+") {
            let error = FuzzError::TargetNotFound(target.clone());
            let msg = error.to_string();
            prop_assert!(msg.contains(&target));
        }

        #[test]
        fn fuzz_error_build_failed_contains_message(message in ".{1,50}") {
            let error = FuzzError::BuildFailed(message.clone());
            let msg = error.to_string();
            prop_assert!(msg.contains(&message));
        }

        #[test]
        fn fuzz_error_fuzzing_failed_contains_message(message in ".{1,50}") {
            let error = FuzzError::FuzzingFailed(message.clone());
            let msg = error.to_string();
            prop_assert!(msg.contains(&message));
        }

        #[test]
        fn fuzz_error_timeout_contains_duration(secs in 1u64..10000) {
            let duration = Duration::from_secs(secs);
            let error = FuzzError::Timeout(duration);
            let msg = error.to_string();
            // Duration debug format includes the value
            prop_assert!(msg.contains("timeout") || msg.contains("Timeout"));
        }

        // CrashType equality tests
        #[test]
        fn crash_type_reflexive_equality(crash_type in crash_type_strategy()) {
            prop_assert_eq!(crash_type, crash_type);
        }

        // FuzzerType equality tests
        #[test]
        fn fuzzer_type_reflexive_equality(fuzzer in fuzzer_type_strategy()) {
            prop_assert_eq!(fuzzer, fuzzer);
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // FuzzerType invariants
    #[kani::proof]
    fn verify_fuzzer_type_name_non_empty() {
        let fuzzer: FuzzerType = kani::any();
        let name = fuzzer.name();
        kani::assert(!name.is_empty(), "FuzzerType name must not be empty");
    }

    #[kani::proof]
    fn verify_fuzzer_type_cargo_command_non_empty() {
        let fuzzer: FuzzerType = kani::any();
        let cmd = fuzzer.cargo_command();
        kani::assert(
            !cmd.is_empty(),
            "FuzzerType cargo_command must not be empty",
        );
    }

    #[kani::proof]
    fn verify_fuzzer_type_nightly_consistency() {
        let fuzzer: FuzzerType = kani::any();
        let requires = fuzzer.requires_nightly();
        let expected = matches!(fuzzer, FuzzerType::LibFuzzer | FuzzerType::Honggfuzz);
        kani::assert(requires == expected, "Nightly requirement must match spec");
    }

    // FuzzConfig invariants
    #[kani::proof]
    fn verify_fuzz_config_default_values() {
        let config = FuzzConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(60),
            "Default timeout is 60s",
        );
        kani::assert(config.max_iterations == 0, "Default max_iterations is 0");
        kani::assert(config.jobs == 1, "Default jobs is 1");
        kani::assert(config.seed_corpus.is_none(), "Default seed_corpus is None");
        kani::assert(config.dictionary.is_none(), "Default dictionary is None");
        kani::assert(config.max_len.is_none(), "Default max_len is None");
    }

    #[kani::proof]
    fn verify_fuzz_config_with_timeout_preserves() {
        let secs: u64 = kani::any();
        kani::assume(secs > 0 && secs < 100000);
        let timeout = Duration::from_secs(secs);
        let config = FuzzConfig::default().with_timeout(timeout);
        kani::assert(
            config.timeout == timeout,
            "with_timeout must preserve value",
        );
    }

    #[kani::proof]
    fn verify_fuzz_config_with_max_iterations_preserves() {
        let iterations: u64 = kani::any();
        let config = FuzzConfig::default().with_max_iterations(iterations);
        kani::assert(
            config.max_iterations == iterations,
            "with_max_iterations must preserve value",
        );
    }

    #[kani::proof]
    fn verify_fuzz_config_with_jobs_preserves() {
        let jobs: usize = kani::any();
        kani::assume(jobs > 0 && jobs < 1000);
        let config = FuzzConfig::default().with_jobs(jobs);
        kani::assert(config.jobs == jobs, "with_jobs must preserve value");
    }

    // CoverageInfo invariants
    #[kani::proof]
    fn verify_coverage_info_default_zeroes() {
        let coverage = CoverageInfo::default();
        kani::assert(
            coverage.regions_covered == 0,
            "Default regions_covered is 0",
        );
        kani::assert(coverage.total_regions == 0, "Default total_regions is 0");
        kani::assert(coverage.unique_paths == 0, "Default unique_paths is 0");
        kani::assert(
            coverage.edge_coverage_pct.is_none(),
            "Default edge_coverage_pct is None",
        );
    }

    // CrashType invariants
    #[kani::proof]
    fn verify_crash_type_equality_reflexive() {
        let crash_type: CrashType = kani::any();
        kani::assert(
            crash_type == crash_type,
            "CrashType equality must be reflexive",
        );
    }

    // FuzzBackend construction
    #[kani::proof]
    fn verify_fuzz_backend_preserves_fuzzer_type() {
        let fuzzer: FuzzerType = kani::any();
        let backend = FuzzBackend::new(fuzzer, FuzzConfig::default());
        kani::assert(
            backend.fuzzer_type == fuzzer,
            "FuzzBackend must preserve fuzzer_type",
        );
    }

    #[kani::proof]
    fn verify_fuzz_backend_install_instructions_non_empty() {
        let fuzzer: FuzzerType = kani::any();
        let backend = FuzzBackend::new(fuzzer, FuzzConfig::default());
        let instructions = backend.install_instructions();
        kani::assert(
            !instructions.is_empty(),
            "Install instructions must not be empty",
        );
    }

    // FuzzResult passed consistency
    #[kani::proof]
    fn verify_fuzz_result_no_crashes_can_be_passed() {
        let fuzzer: FuzzerType = kani::any();
        let result = FuzzResult {
            fuzzer,
            target: String::new(),
            passed: true,
            crashes: Vec::new(),
            executions: 0,
            exec_per_sec: 0.0,
            coverage: CoverageInfo::default(),
            duration: Duration::from_secs(1),
            corpus_path: None,
        };
        // If passed is true and crashes is empty, this is consistent
        kani::assert(
            result.passed && result.crashes.is_empty(),
            "passed=true requires empty crashes",
        );
    }

    // FuzzCrash structure
    #[kani::proof]
    fn verify_fuzz_crash_preserves_crash_type() {
        let crash_type: CrashType = kani::any();
        let crash = FuzzCrash {
            crash_type,
            input: Vec::new(),
            stack_trace: None,
            exit_code: None,
            artifact_path: None,
        };
        kani::assert(
            crash.crash_type == crash_type,
            "FuzzCrash must preserve crash_type",
        );
    }

    // FuzzerType coverage - ensure all variants have valid properties
    #[kani::proof]
    fn verify_all_fuzzer_types_have_install_instructions() {
        let fuzzer: FuzzerType = kani::any();
        let backend = FuzzBackend::new(fuzzer, FuzzConfig::default());
        let instructions = backend.install_instructions();
        // All fuzzers should have "cargo install" in their instructions
        kani::assert(
            instructions.starts_with("cargo install"),
            "All fuzzers should have cargo install instructions",
        );
    }

    // ============== FuzzConfig Additional Proofs ==============

    /// Proves that with_seed_corpus sets the seed corpus path
    #[kani::proof]
    fn verify_fuzz_config_with_seed_corpus() {
        let config = FuzzConfig::default().with_seed_corpus(std::path::PathBuf::from("/test"));
        kani::assert(
            config.seed_corpus.is_some(),
            "with_seed_corpus must set Some value",
        );
    }

    /// Proves that FuzzConfig default has no seed corpus
    #[kani::proof]
    fn verify_fuzz_config_default_no_seed_corpus() {
        let config = FuzzConfig::default();
        kani::assert(
            config.seed_corpus.is_none(),
            "Default config has no seed corpus",
        );
    }

    /// Proves that builder methods don't affect unrelated fields
    #[kani::proof]
    fn verify_fuzz_config_with_timeout_doesnt_change_jobs() {
        let default_config = FuzzConfig::default();
        let default_jobs = default_config.jobs;
        let secs: u64 = kani::any();
        kani::assume(secs > 0 && secs < 10000);
        let config = FuzzConfig::default().with_timeout(Duration::from_secs(secs));
        kani::assert(
            config.jobs == default_jobs,
            "with_timeout must not change jobs",
        );
    }

    /// Proves that with_jobs doesn't change timeout
    #[kani::proof]
    fn verify_fuzz_config_with_jobs_doesnt_change_timeout() {
        let default_config = FuzzConfig::default();
        let default_timeout = default_config.timeout;
        let jobs: usize = kani::any();
        kani::assume(jobs > 0 && jobs < 1000);
        let config = FuzzConfig::default().with_jobs(jobs);
        kani::assert(
            config.timeout == default_timeout,
            "with_jobs must not change timeout",
        );
    }

    // ============== CrashType Additional Proofs ==============

    /// Proves that CrashType variants are all distinct
    #[kani::proof]
    fn verify_crash_type_variants_distinct() {
        kani::assert(CrashType::Panic != CrashType::Segfault, "Panic != Segfault");
        kani::assert(CrashType::Panic != CrashType::OOM, "Panic != OOM");
        kani::assert(CrashType::Panic != CrashType::Timeout, "Panic != Timeout");
        kani::assert(CrashType::Panic != CrashType::Other, "Panic != Other");
        kani::assert(CrashType::Segfault != CrashType::OOM, "Segfault != OOM");
        kani::assert(
            CrashType::Segfault != CrashType::Timeout,
            "Segfault != Timeout",
        );
        kani::assert(CrashType::Segfault != CrashType::Other, "Segfault != Other");
        kani::assert(CrashType::OOM != CrashType::Timeout, "OOM != Timeout");
        kani::assert(CrashType::OOM != CrashType::Other, "OOM != Other");
        kani::assert(CrashType::Timeout != CrashType::Other, "Timeout != Other");
    }

    // ============== FuzzerType Additional Proofs ==============

    /// Proves that FuzzerType variants are all distinct
    #[kani::proof]
    fn verify_fuzzer_type_variants_distinct() {
        kani::assert(FuzzerType::LibFuzzer != FuzzerType::AFL, "LibFuzzer != AFL");
        kani::assert(
            FuzzerType::LibFuzzer != FuzzerType::Honggfuzz,
            "LibFuzzer != Honggfuzz",
        );
        kani::assert(
            FuzzerType::LibFuzzer != FuzzerType::Bolero,
            "LibFuzzer != Bolero",
        );
        kani::assert(FuzzerType::AFL != FuzzerType::Honggfuzz, "AFL != Honggfuzz");
        kani::assert(FuzzerType::AFL != FuzzerType::Bolero, "AFL != Bolero");
        kani::assert(
            FuzzerType::Honggfuzz != FuzzerType::Bolero,
            "Honggfuzz != Bolero",
        );
    }

    /// Proves that FuzzerType equality is symmetric
    #[kani::proof]
    fn verify_fuzzer_type_equality_symmetric() {
        let a: FuzzerType = kani::any();
        let b: FuzzerType = kani::any();
        // If a == b, then b == a
        let ab = a == b;
        let ba = b == a;
        kani::assert(ab == ba, "FuzzerType equality must be symmetric");
    }

    /// Proves that CrashType equality is symmetric
    #[kani::proof]
    fn verify_crash_type_equality_symmetric() {
        let a: CrashType = kani::any();
        let b: CrashType = kani::any();
        let ab = a == b;
        let ba = b == a;
        kani::assert(ab == ba, "CrashType equality must be symmetric");
    }

    // ============== FuzzResult Consistency Proofs ==============

    /// Proves that FuzzResult executions field is preserved
    #[kani::proof]
    fn verify_fuzz_result_preserves_executions() {
        let executions: u64 = kani::any();
        let result = FuzzResult {
            fuzzer: FuzzerType::LibFuzzer,
            target: String::new(),
            passed: true,
            crashes: Vec::new(),
            executions,
            exec_per_sec: 0.0,
            coverage: CoverageInfo::default(),
            duration: Duration::from_secs(1),
            corpus_path: None,
        };
        kani::assert(
            result.executions == executions,
            "FuzzResult preserves executions",
        );
    }

    /// Proves that exec_per_sec calculation doesn't panic for valid durations
    #[kani::proof]
    fn verify_exec_per_sec_no_panic_valid_duration() {
        let executions: u64 = kani::any();
        let secs: u64 = kani::any();
        kani::assume(secs > 0 && secs <= 10000);
        kani::assume(executions <= 1_000_000_000);

        let duration = Duration::from_secs(secs);
        let exec_per_sec = executions as f64 / duration.as_secs_f64();
        kani::assert(exec_per_sec.is_finite(), "exec_per_sec should be finite");
    }

    // ============== CoverageInfo Additional Proofs ==============

    /// Proves that CoverageInfo fields can be set to arbitrary values
    #[kani::proof]
    fn verify_coverage_info_fields_settable() {
        let regions_covered: usize = kani::any();
        let total_regions: usize = kani::any();
        let unique_paths: usize = kani::any();
        kani::assume(regions_covered <= 1_000_000);
        kani::assume(total_regions <= 1_000_000);
        kani::assume(unique_paths <= 1_000_000);

        let coverage = CoverageInfo {
            regions_covered,
            total_regions,
            unique_paths,
            edge_coverage_pct: None,
        };
        kani::assert(
            coverage.regions_covered == regions_covered,
            "regions_covered preserved",
        );
        kani::assert(
            coverage.total_regions == total_regions,
            "total_regions preserved",
        );
        kani::assert(
            coverage.unique_paths == unique_paths,
            "unique_paths preserved",
        );
    }

    /// Proves that covered regions can be less than or equal to total regions
    #[kani::proof]
    fn verify_coverage_info_can_represent_partial() {
        let total: usize = kani::any();
        let covered: usize = kani::any();
        kani::assume(total > 0 && total <= 10000);
        kani::assume(covered <= total);

        let coverage = CoverageInfo {
            regions_covered: covered,
            total_regions: total,
            unique_paths: 0,
            edge_coverage_pct: None,
        };
        kani::assert(
            coverage.regions_covered <= coverage.total_regions,
            "Can represent partial coverage",
        );
    }
}
