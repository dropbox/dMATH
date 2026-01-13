//! CDSChecker backend for C/C++11 memory model checking
//!
//! This backend runs CDSChecker to verify concurrent code against the C/C++11
//! memory model. CDSChecker systematically explores all possible executions
//! under the relaxed memory model semantics.

// =============================================
// Kani Proofs for CDSChecker Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- ExplorationStrategy Default Tests ----

    /// Verify ExplorationStrategy::default is DFS
    #[kani::proof]
    fn proof_exploration_strategy_default_is_dfs() {
        let strategy = ExplorationStrategy::default();
        kani::assert(
            strategy == ExplorationStrategy::DFS,
            "Default strategy should be DFS",
        );
    }

    // ---- MemoryModelMode Default Tests ----

    /// Verify MemoryModelMode::default is Full
    #[kani::proof]
    fn proof_memory_model_mode_default_is_full() {
        let mode = MemoryModelMode::default();
        kani::assert(mode == MemoryModelMode::Full, "Default mode should be Full");
    }

    // ---- CDSCheckerConfig Default Tests ----

    /// Verify CDSCheckerConfig::default timeout is 300 seconds
    #[kani::proof]
    fn proof_cdschecker_config_default_timeout() {
        let config = CDSCheckerConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "Default timeout should be 300 seconds",
        );
    }

    /// Verify CDSCheckerConfig::default binary_path is None
    #[kani::proof]
    fn proof_cdschecker_config_default_binary_path_none() {
        let config = CDSCheckerConfig::default();
        kani::assert(
            config.binary_path.is_none(),
            "Default binary_path should be None",
        );
    }

    /// Verify CDSCheckerConfig::default cdschecker_path is None
    #[kani::proof]
    fn proof_cdschecker_config_default_cdschecker_path_none() {
        let config = CDSCheckerConfig::default();
        kani::assert(
            config.cdschecker_path.is_none(),
            "Default cdschecker_path should be None",
        );
    }

    /// Verify CDSCheckerConfig::default strategy is DFS
    #[kani::proof]
    fn proof_cdschecker_config_default_strategy() {
        let config = CDSCheckerConfig::default();
        kani::assert(
            config.strategy == ExplorationStrategy::DFS,
            "Default strategy should be DFS",
        );
    }

    /// Verify CDSCheckerConfig::default memory_model is Full
    #[kani::proof]
    fn proof_cdschecker_config_default_memory_model() {
        let config = CDSCheckerConfig::default();
        kani::assert(
            config.memory_model == MemoryModelMode::Full,
            "Default memory_model should be Full",
        );
    }

    /// Verify CDSCheckerConfig::default max_executions is 10000
    #[kani::proof]
    fn proof_cdschecker_config_default_max_executions() {
        let config = CDSCheckerConfig::default();
        kani::assert(
            config.max_executions == 10000,
            "Default max_executions should be 10000",
        );
    }

    /// Verify CDSCheckerConfig::default verbose is false
    #[kani::proof]
    fn proof_cdschecker_config_default_verbose_false() {
        let config = CDSCheckerConfig::default();
        kani::assert(!config.verbose, "Default verbose should be false");
    }

    /// Verify CDSCheckerConfig::default detect_cycles is true
    #[kani::proof]
    fn proof_cdschecker_config_default_detect_cycles_true() {
        let config = CDSCheckerConfig::default();
        kani::assert(config.detect_cycles, "Default detect_cycles should be true");
    }

    /// Verify CDSCheckerConfig::default thread_count is 4
    #[kani::proof]
    fn proof_cdschecker_config_default_thread_count() {
        let config = CDSCheckerConfig::default();
        kani::assert(config.thread_count == 4, "Default thread_count should be 4");
    }

    // ---- CDSCheckerConfig Builder Tests ----

    /// Verify with_timeout preserves timeout value
    #[kani::proof]
    fn proof_cdschecker_config_with_timeout() {
        let config = CDSCheckerConfig::default().with_timeout(Duration::from_secs(120));
        kani::assert(
            config.timeout == Duration::from_secs(120),
            "with_timeout should set timeout",
        );
    }

    /// Verify with_strategy preserves strategy value
    #[kani::proof]
    fn proof_cdschecker_config_with_strategy_random() {
        let config = CDSCheckerConfig::default().with_strategy(ExplorationStrategy::Random);
        kani::assert(
            config.strategy == ExplorationStrategy::Random,
            "with_strategy should set strategy",
        );
    }

    /// Verify with_strategy preserves BFS
    #[kani::proof]
    fn proof_cdschecker_config_with_strategy_bfs() {
        let config = CDSCheckerConfig::default().with_strategy(ExplorationStrategy::BFS);
        kani::assert(
            config.strategy == ExplorationStrategy::BFS,
            "with_strategy should set BFS",
        );
    }

    /// Verify with_memory_model preserves TSO
    #[kani::proof]
    fn proof_cdschecker_config_with_memory_model_tso() {
        let config = CDSCheckerConfig::default().with_memory_model(MemoryModelMode::TSO);
        kani::assert(
            config.memory_model == MemoryModelMode::TSO,
            "with_memory_model should set TSO",
        );
    }

    /// Verify with_memory_model preserves SequentialConsistency
    #[kani::proof]
    fn proof_cdschecker_config_with_memory_model_sc() {
        let config =
            CDSCheckerConfig::default().with_memory_model(MemoryModelMode::SequentialConsistency);
        kani::assert(
            config.memory_model == MemoryModelMode::SequentialConsistency,
            "with_memory_model should set SC",
        );
    }

    /// Verify with_max_executions preserves value
    #[kani::proof]
    fn proof_cdschecker_config_with_max_executions() {
        let config = CDSCheckerConfig::default().with_max_executions(5000);
        kani::assert(
            config.max_executions == 5000,
            "with_max_executions should set value",
        );
    }

    /// Verify with_verbose preserves true
    #[kani::proof]
    fn proof_cdschecker_config_with_verbose_true() {
        let config = CDSCheckerConfig::default().with_verbose(true);
        kani::assert(config.verbose, "with_verbose(true) should set verbose");
    }

    /// Verify with_detect_cycles preserves false
    #[kani::proof]
    fn proof_cdschecker_config_with_detect_cycles_false() {
        let config = CDSCheckerConfig::default().with_detect_cycles(false);
        kani::assert(
            !config.detect_cycles,
            "with_detect_cycles(false) should disable",
        );
    }

    /// Verify with_thread_count preserves value
    #[kani::proof]
    fn proof_cdschecker_config_with_thread_count() {
        let config = CDSCheckerConfig::default().with_thread_count(8);
        kani::assert(
            config.thread_count == 8,
            "with_thread_count should set value",
        );
    }

    // ---- CDSCheckerBackend Construction Tests ----

    /// Verify CDSCheckerBackend::new uses default config timeout
    #[kani::proof]
    fn proof_cdschecker_backend_new_default_timeout() {
        let backend = CDSCheckerBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(300),
            "New backend should use default timeout",
        );
    }

    /// Verify CDSCheckerBackend::default equals CDSCheckerBackend::new
    #[kani::proof]
    fn proof_cdschecker_backend_default_equals_new() {
        let default_backend = CDSCheckerBackend::default();
        let new_backend = CDSCheckerBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "Default and new should have same timeout",
        );
    }

    /// Verify CDSCheckerBackend::with_config preserves custom timeout
    #[kani::proof]
    fn proof_cdschecker_backend_with_config_timeout() {
        let config = CDSCheckerConfig {
            binary_path: None,
            cdschecker_path: None,
            timeout: Duration::from_secs(600),
            strategy: ExplorationStrategy::DFS,
            memory_model: MemoryModelMode::Full,
            max_executions: 10000,
            verbose: false,
            detect_cycles: true,
            thread_count: 4,
        };
        let backend = CDSCheckerBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "with_config should preserve timeout",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify CDSCheckerBackend::id returns CDSChecker
    #[kani::proof]
    fn proof_cdschecker_backend_id() {
        let backend = CDSCheckerBackend::new();
        kani::assert(
            backend.id() == BackendId::CDSChecker,
            "Backend id should be CDSChecker",
        );
    }

    /// Verify CDSCheckerBackend::supports includes DataRace
    #[kani::proof]
    fn proof_cdschecker_backend_supports_data_race() {
        let backend = CDSCheckerBackend::new();
        let supported = backend.supports();
        let has_data_race = supported.iter().any(|p| *p == PropertyType::DataRace);
        kani::assert(has_data_race, "Should support DataRace property");
    }

    /// Verify CDSCheckerBackend::supports includes MemorySafety
    #[kani::proof]
    fn proof_cdschecker_backend_supports_memory_safety() {
        let backend = CDSCheckerBackend::new();
        let supported = backend.supports();
        let has_memory_safety = supported.iter().any(|p| *p == PropertyType::MemorySafety);
        kani::assert(has_memory_safety, "Should support MemorySafety property");
    }

    /// Verify CDSCheckerBackend::supports returns exactly 2 properties
    #[kani::proof]
    fn proof_cdschecker_backend_supports_length() {
        let backend = CDSCheckerBackend::new();
        let supported = backend.supports();
        kani::assert(supported.len() == 2, "Should support exactly 2 properties");
    }

    // ---- extract_number Tests ----

    /// Verify extract_number returns None for no numbers
    #[kani::proof]
    fn proof_extract_number_no_numbers() {
        let result = extract_number("no numbers here");
        kani::assert(result.is_none(), "Should return None for no numbers");
    }
}

use crate::counterexample::StructuredCounterexample;
use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
    VerificationStatus,
};
use async_trait::async_trait;
use dashprove_usl::typecheck::TypedSpec;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokio::process::Command;

/// CDSChecker exploration strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ExplorationStrategy {
    /// Depth-first search (default)
    #[default]
    DFS,
    /// Breadth-first search
    BFS,
    /// Random exploration
    Random,
}

/// Memory model strictness
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum MemoryModelMode {
    /// Full C/C++11 memory model (relaxed, acquire-release, sequential consistency)
    #[default]
    Full,
    /// Sequential consistency only (easier to verify, but less precise)
    SequentialConsistency,
    /// TSO (Total Store Order) - x86-like memory model
    TSO,
}

/// Configuration for CDSChecker backend
#[derive(Debug, Clone)]
pub struct CDSCheckerConfig {
    /// Path to the binary to analyze
    pub binary_path: Option<PathBuf>,
    /// Path to CDSChecker installation
    pub cdschecker_path: Option<PathBuf>,
    /// Timeout for analysis
    pub timeout: Duration,
    /// Exploration strategy
    pub strategy: ExplorationStrategy,
    /// Memory model mode
    pub memory_model: MemoryModelMode,
    /// Maximum number of executions to explore
    pub max_executions: u64,
    /// Enable verbose output
    pub verbose: bool,
    /// Enable cycle detection
    pub detect_cycles: bool,
    /// Number of threads in the program (hint)
    pub thread_count: u32,
}

impl Default for CDSCheckerConfig {
    fn default() -> Self {
        Self {
            binary_path: None,
            cdschecker_path: None,
            timeout: Duration::from_secs(300),
            strategy: ExplorationStrategy::default(),
            memory_model: MemoryModelMode::default(),
            max_executions: 10000,
            verbose: false,
            detect_cycles: true,
            thread_count: 4,
        }
    }
}

impl CDSCheckerConfig {
    /// Set the binary path
    pub fn with_binary_path(mut self, path: PathBuf) -> Self {
        self.binary_path = Some(path);
        self
    }

    /// Set CDSChecker installation path
    pub fn with_cdschecker_path(mut self, path: PathBuf) -> Self {
        self.cdschecker_path = Some(path);
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set exploration strategy
    pub fn with_strategy(mut self, strategy: ExplorationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set memory model mode
    pub fn with_memory_model(mut self, mode: MemoryModelMode) -> Self {
        self.memory_model = mode;
        self
    }

    /// Set maximum executions
    pub fn with_max_executions(mut self, max: u64) -> Self {
        self.max_executions = max;
        self
    }

    /// Enable verbose output
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Enable/disable cycle detection
    pub fn with_detect_cycles(mut self, detect: bool) -> Self {
        self.detect_cycles = detect;
        self
    }

    /// Set thread count hint
    pub fn with_thread_count(mut self, count: u32) -> Self {
        self.thread_count = count;
        self
    }
}

/// CDSChecker verification backend for C/C++11 memory model
///
/// CDSChecker explores all possible executions of concurrent programs under
/// the C/C++11 relaxed memory model. It can find:
/// - Data races under relaxed memory ordering
/// - Memory order violations
/// - Assertion failures in concurrent code
/// - Cycle violations (release sequence, modification order, etc.)
///
/// # Requirements
///
/// Build CDSChecker from source:
/// ```bash
/// git clone https://github.com/computersforpeace/cdschecker
/// cd cdschecker
/// make
/// ```
///
/// # Usage with Rust
///
/// CDSChecker operates on C/C++ binaries linked against its runtime library.
/// For Rust code, you would need to:
/// 1. Create a C wrapper for your concurrent Rust code via FFI
/// 2. Link against CDSChecker's runtime library
/// 3. Compile with CDSChecker's model-checking instrumentation
///
/// This backend primarily targets C/C++ concurrent code verification.
pub struct CDSCheckerBackend {
    config: CDSCheckerConfig,
}

impl CDSCheckerBackend {
    /// Create a new CDSChecker backend with default configuration
    pub fn new() -> Self {
        Self {
            config: CDSCheckerConfig::default(),
        }
    }

    /// Create a new CDSChecker backend with custom configuration
    pub fn with_config(config: CDSCheckerConfig) -> Self {
        Self { config }
    }

    /// Run CDSChecker verification on a binary
    pub async fn analyze_binary(&self, binary_path: &Path) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        // Determine CDSChecker command
        let cdschecker_cmd = self
            .config
            .cdschecker_path
            .clone()
            .unwrap_or_else(|| PathBuf::from("cdschecker"));

        // Build command arguments
        let mut args = Vec::new();

        // Add exploration strategy
        match self.config.strategy {
            ExplorationStrategy::DFS => args.push("-d".to_string()),
            ExplorationStrategy::BFS => args.push("-b".to_string()),
            ExplorationStrategy::Random => args.push("-r".to_string()),
        }

        // Add max executions
        args.push("-m".to_string());
        args.push(self.config.max_executions.to_string());

        // Add verbose flag
        if self.config.verbose {
            args.push("-v".to_string());
        }

        // Add cycle detection
        if self.config.detect_cycles {
            args.push("-c".to_string());
        }

        // Add the binary to analyze
        args.push(binary_path.to_string_lossy().to_string());

        // Set environment for memory model
        let mut cmd = Command::new(&cdschecker_cmd);
        cmd.args(&args);

        match self.config.memory_model {
            MemoryModelMode::Full => {
                cmd.env("CDSCHECKER_MM", "c11");
            }
            MemoryModelMode::SequentialConsistency => {
                cmd.env("CDSCHECKER_MM", "sc");
            }
            MemoryModelMode::TSO => {
                cmd.env("CDSCHECKER_MM", "tso");
            }
        }

        let output = tokio::time::timeout(self.config.timeout, cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| {
                BackendError::VerificationFailed(format!("Failed to run CDSChecker: {}", e))
            })?;

        let duration = start.elapsed();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined = format!("{}\n{}", stdout, stderr);

        // Parse CDSChecker output
        let (status, findings, stats) =
            self.parse_cdschecker_output(&combined, output.status.success());

        // Build diagnostics
        let mut diagnostics = Vec::new();

        // Summary
        let summary = match &status {
            VerificationStatus::Proven => format!(
                "CDSChecker: All {} executions verified under {} memory model",
                stats.executions_explored,
                match self.config.memory_model {
                    MemoryModelMode::Full => "C/C++11",
                    MemoryModelMode::SequentialConsistency => "Sequential Consistency",
                    MemoryModelMode::TSO => "TSO",
                }
            ),
            VerificationStatus::Disproven => {
                format!(
                    "CDSChecker: {} bug(s) found in {} executions",
                    findings.len(),
                    stats.executions_explored
                )
            }
            VerificationStatus::Partial {
                verified_percentage,
            } => {
                format!(
                    "CDSChecker: {:.1}% of executions clean ({} explored)",
                    verified_percentage, stats.executions_explored
                )
            }
            VerificationStatus::Unknown { reason } => format!("CDSChecker: {}", reason),
        };
        diagnostics.push(summary);

        // Add stats
        if stats.executions_explored > 0 {
            diagnostics.push(format!(
                "Executions: {} explored, {} data races, {} assertion failures",
                stats.executions_explored, stats.data_races, stats.assertion_failures
            ));
        }

        diagnostics.extend(findings.clone());

        // Build counterexample if bugs found
        let counterexample = if !findings.is_empty() {
            Some(StructuredCounterexample::from_raw(combined.clone()))
        } else {
            None
        };

        Ok(BackendResult {
            backend: BackendId::CDSChecker,
            status,
            proof: None,
            counterexample,
            diagnostics,
            time_taken: duration,
        })
    }

    fn parse_cdschecker_output(
        &self,
        output: &str,
        success: bool,
    ) -> (VerificationStatus, Vec<String>, CDSCheckerStats) {
        let mut findings = Vec::new();
        let mut stats = CDSCheckerStats::default();

        // Parse CDSChecker output
        for line in output.lines() {
            // Statistics patterns
            if line.contains("Total executions:") {
                if let Some(num) = extract_number(line) {
                    stats.executions_explored = num;
                }
            }
            if line.contains("Buggy executions:") {
                if let Some(num) = extract_number(line) {
                    stats.buggy_executions = num;
                }
            }

            // Bug patterns
            if line.contains("Data race")
                || line.contains("data race")
                || line.contains("DATA RACE")
            {
                findings.push(line.trim().to_string());
                stats.data_races += 1;
            }
            if line.contains("Assertion failure") || line.contains("ASSERTION FAILED") {
                findings.push(line.trim().to_string());
                stats.assertion_failures += 1;
            }
            if line.contains("Memory order violation") || line.contains("MO VIOLATION") {
                findings.push(line.trim().to_string());
                stats.memory_order_violations += 1;
            }
            if line.contains("Cycle detected") || line.contains("CYCLE") {
                findings.push(line.trim().to_string());
                stats.cycles_detected += 1;
            }
        }

        let status = if findings.is_empty() && success {
            VerificationStatus::Proven
        } else if !findings.is_empty() {
            VerificationStatus::Disproven
        } else if !success {
            // Check for installation issues
            if output.contains("not found") || output.contains("command not found") {
                return (
                    VerificationStatus::Unknown {
                        reason: "CDSChecker not installed. Build from: https://github.com/computersforpeace/cdschecker".to_string(),
                    },
                    Vec::new(),
                    stats,
                );
            }
            if output.contains("not a valid executable") {
                return (
                    VerificationStatus::Unknown {
                        reason: "Binary must be compiled with CDSChecker instrumentation"
                            .to_string(),
                    },
                    Vec::new(),
                    stats,
                );
            }
            VerificationStatus::Unknown {
                reason: "Verification failed with unknown error".to_string(),
            }
        } else if stats.executions_explored > 0 && stats.buggy_executions == 0 {
            VerificationStatus::Proven
        } else if stats.buggy_executions > 0 {
            // Calculate percentage
            let clean = stats
                .executions_explored
                .saturating_sub(stats.buggy_executions);
            let percentage = if stats.executions_explored > 0 {
                (clean as f64 / stats.executions_explored as f64) * 100.0
            } else {
                0.0
            };
            VerificationStatus::Partial {
                verified_percentage: percentage,
            }
        } else {
            VerificationStatus::Unknown {
                reason: "No executions explored".to_string(),
            }
        };

        (status, findings, stats)
    }

    /// Check if CDSChecker is installed
    pub async fn check_installed(&self) -> Result<bool, BackendError> {
        let cdschecker_cmd = self
            .config
            .cdschecker_path
            .clone()
            .unwrap_or_else(|| PathBuf::from("cdschecker"));

        let output = Command::new(&cdschecker_cmd)
            .arg("--version")
            .output()
            .await;

        match output {
            Ok(out) if out.status.success() => Ok(true),
            Ok(out) => {
                // Some versions just print help on --version
                let stderr = String::from_utf8_lossy(&out.stderr);
                let stdout = String::from_utf8_lossy(&out.stdout);
                Ok(stderr.contains("CDSChecker")
                    || stdout.contains("CDSChecker")
                    || stderr.contains("C/C++11")
                    || stdout.contains("C/C++11"))
            }
            Err(_) => Ok(false),
        }
    }
}

/// Helper to extract a number from a line
fn extract_number(line: &str) -> Option<u64> {
    line.split_whitespace()
        .filter_map(|w| w.trim_matches(|c: char| !c.is_ascii_digit()).parse().ok())
        .next()
}

/// Statistics from CDSChecker run
#[derive(Debug, Clone, Default)]
struct CDSCheckerStats {
    executions_explored: u64,
    buggy_executions: u64,
    data_races: u64,
    assertion_failures: u64,
    memory_order_violations: u64,
    cycles_detected: u64,
}

impl Default for CDSCheckerBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VerificationBackend for CDSCheckerBackend {
    fn id(&self) -> BackendId {
        BackendId::CDSChecker
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::DataRace, PropertyType::MemorySafety]
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let binary_path = self.config.binary_path.clone().ok_or_else(|| {
            BackendError::Unavailable(
                "CDSChecker backend requires binary_path pointing to a CDSChecker-instrumented executable"
                    .to_string(),
            )
        })?;

        self.analyze_binary(&binary_path).await
    }

    async fn health_check(&self) -> HealthStatus {
        match self.check_installed().await {
            Ok(true) => HealthStatus::Healthy,
            Ok(false) => HealthStatus::Unavailable {
                reason: "CDSChecker not installed. Build from: https://github.com/computersforpeace/cdschecker"
                    .to_string(),
            },
            Err(e) => HealthStatus::Unavailable {
                reason: format!("Failed to check CDSChecker installation: {}", e),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cdschecker_config_default() {
        let config = CDSCheckerConfig::default();
        assert!(config.binary_path.is_none());
        assert!(config.cdschecker_path.is_none());
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert_eq!(config.strategy, ExplorationStrategy::DFS);
        assert_eq!(config.memory_model, MemoryModelMode::Full);
        assert_eq!(config.max_executions, 10000);
        assert!(!config.verbose);
        assert!(config.detect_cycles);
        assert_eq!(config.thread_count, 4);
    }

    #[test]
    fn test_cdschecker_config_builder() {
        let config = CDSCheckerConfig::default()
            .with_binary_path(PathBuf::from("/test/binary"))
            .with_cdschecker_path(PathBuf::from("/opt/cdschecker"))
            .with_timeout(Duration::from_secs(120))
            .with_strategy(ExplorationStrategy::Random)
            .with_memory_model(MemoryModelMode::TSO)
            .with_max_executions(5000)
            .with_verbose(true)
            .with_detect_cycles(false)
            .with_thread_count(8);

        assert_eq!(config.binary_path, Some(PathBuf::from("/test/binary")));
        assert_eq!(
            config.cdschecker_path,
            Some(PathBuf::from("/opt/cdschecker"))
        );
        assert_eq!(config.timeout, Duration::from_secs(120));
        assert_eq!(config.strategy, ExplorationStrategy::Random);
        assert_eq!(config.memory_model, MemoryModelMode::TSO);
        assert_eq!(config.max_executions, 5000);
        assert!(config.verbose);
        assert!(!config.detect_cycles);
        assert_eq!(config.thread_count, 8);
    }

    #[test]
    fn test_cdschecker_backend_id() {
        let backend = CDSCheckerBackend::new();
        assert_eq!(backend.id(), BackendId::CDSChecker);
    }

    #[test]
    fn test_cdschecker_supports_data_race() {
        let backend = CDSCheckerBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::DataRace));
        assert!(supported.contains(&PropertyType::MemorySafety));
    }

    #[test]
    fn test_cdschecker_parse_output_clean() {
        let backend = CDSCheckerBackend::new();
        let output =
            "CDSChecker model checking...\nTotal executions: 1000\nBuggy executions: 0\nComplete.";
        let (status, findings, stats) = backend.parse_cdschecker_output(output, true);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(findings.is_empty());
        assert_eq!(stats.executions_explored, 1000);
        assert_eq!(stats.buggy_executions, 0);
    }

    #[test]
    fn test_cdschecker_parse_output_with_data_race() {
        let backend = CDSCheckerBackend::new();
        let output =
            "Total executions: 500\nData race detected at location 0x1234\nBuggy executions: 3";
        let (status, findings, stats) = backend.parse_cdschecker_output(output, false);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(!findings.is_empty());
        assert_eq!(stats.data_races, 1);
    }

    #[test]
    fn test_cdschecker_parse_output_not_installed() {
        let backend = CDSCheckerBackend::new();
        let output = "cdschecker: command not found";
        let (status, _, _) = backend.parse_cdschecker_output(output, false);
        match status {
            VerificationStatus::Unknown { reason } => {
                assert!(reason.contains("not installed"));
            }
            _ => panic!("Expected Unknown status"),
        }
    }

    #[test]
    fn test_extract_number() {
        assert_eq!(extract_number("Total executions: 1000"), Some(1000));
        assert_eq!(extract_number("Buggy executions: 5"), Some(5));
        assert_eq!(extract_number("No numbers here"), None);
    }

    #[tokio::test]
    async fn test_cdschecker_health_check() {
        let backend = CDSCheckerBackend::new();
        let health = backend.health_check().await;
        // CDSChecker may or may not be installed
        match health {
            HealthStatus::Healthy => {}
            HealthStatus::Unavailable { reason } => {
                assert!(reason.contains("CDSChecker") || reason.contains("cdschecker"));
            }
            HealthStatus::Degraded { .. } => {}
        }
    }

    #[tokio::test]
    async fn test_cdschecker_verify_requires_binary_path() {
        use dashprove_usl::{parse, typecheck};

        let backend = CDSCheckerBackend::new();
        let spec = parse("theorem test { true }").expect("parse");
        let typed_spec = typecheck(spec).expect("typecheck");

        let result = backend.verify(&typed_spec).await;
        assert!(result.is_err());
        if let Err(BackendError::Unavailable(msg)) = result {
            assert!(msg.contains("binary_path"));
        } else {
            panic!("Expected Unavailable error");
        }
    }

    #[test]
    fn test_exploration_strategy_default() {
        assert_eq!(ExplorationStrategy::default(), ExplorationStrategy::DFS);
    }

    #[test]
    fn test_memory_model_mode_default() {
        assert_eq!(MemoryModelMode::default(), MemoryModelMode::Full);
    }
}
