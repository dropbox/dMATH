//! Valgrind backend for memory debugging and profiling
//!
//! This backend runs Valgrind's memory checking tools to detect memory errors
//! in compiled programs. Supports memcheck, helgrind (race detection), and
//! other Valgrind tools.

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

/// Valgrind tool selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ValgrindTool {
    /// Memcheck - Memory error detector (default)
    #[default]
    Memcheck,
    /// Helgrind - Thread error detector (data races)
    Helgrind,
    /// DRD - Another thread error detector
    DRD,
    /// Massif - Heap profiler
    Massif,
    /// Cachegrind - Cache profiler
    Cachegrind,
    /// Callgrind - Call graph profiler
    Callgrind,
}

/// Memcheck leak check level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum LeakCheckLevel {
    /// No leak checking
    No,
    /// Summary only
    Summary,
    /// Full leak details (default)
    #[default]
    Full,
    /// Full with reachable blocks
    Reachable,
}

/// Configuration for Valgrind backend
#[derive(Debug, Clone)]
pub struct ValgrindConfig {
    /// Path to the binary to analyze
    pub binary_path: Option<PathBuf>,
    /// Arguments to pass to the binary
    pub binary_args: Vec<String>,
    /// Path to Valgrind installation
    pub valgrind_path: Option<PathBuf>,
    /// Timeout for analysis
    pub timeout: Duration,
    /// Valgrind tool to use
    pub tool: ValgrindTool,
    /// Leak check level (for memcheck)
    pub leak_check: LeakCheckLevel,
    /// Track origins of uninitialized values
    pub track_origins: bool,
    /// Show reachable blocks in leak output
    pub show_reachable: bool,
    /// Generate XML output
    pub xml_output: bool,
    /// Suppressions file path
    pub suppressions: Option<PathBuf>,
    /// Error exit code (exit with error on Valgrind errors)
    pub error_exitcode: i32,
    /// Enable verbose output
    pub verbose: bool,
    /// Additional Valgrind options
    pub extra_options: Vec<String>,
}

impl Default for ValgrindConfig {
    fn default() -> Self {
        Self {
            binary_path: None,
            binary_args: Vec::new(),
            valgrind_path: None,
            timeout: Duration::from_secs(300),
            tool: ValgrindTool::default(),
            leak_check: LeakCheckLevel::default(),
            track_origins: true,
            show_reachable: false,
            xml_output: false,
            suppressions: None,
            error_exitcode: 1,
            verbose: false,
            extra_options: Vec::new(),
        }
    }
}

impl ValgrindConfig {
    /// Set the binary path
    pub fn with_binary_path(mut self, path: PathBuf) -> Self {
        self.binary_path = Some(path);
        self
    }

    /// Set binary arguments
    pub fn with_binary_args(mut self, args: Vec<String>) -> Self {
        self.binary_args = args;
        self
    }

    /// Set Valgrind installation path
    pub fn with_valgrind_path(mut self, path: PathBuf) -> Self {
        self.valgrind_path = Some(path);
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set Valgrind tool
    pub fn with_tool(mut self, tool: ValgrindTool) -> Self {
        self.tool = tool;
        self
    }

    /// Set leak check level
    pub fn with_leak_check(mut self, level: LeakCheckLevel) -> Self {
        self.leak_check = level;
        self
    }

    /// Enable/disable origin tracking
    pub fn with_track_origins(mut self, enable: bool) -> Self {
        self.track_origins = enable;
        self
    }

    /// Enable/disable showing reachable blocks
    pub fn with_show_reachable(mut self, enable: bool) -> Self {
        self.show_reachable = enable;
        self
    }

    /// Enable XML output
    pub fn with_xml_output(mut self, enable: bool) -> Self {
        self.xml_output = enable;
        self
    }

    /// Set suppressions file
    pub fn with_suppressions(mut self, path: PathBuf) -> Self {
        self.suppressions = Some(path);
        self
    }

    /// Set error exit code
    pub fn with_error_exitcode(mut self, code: i32) -> Self {
        self.error_exitcode = code;
        self
    }

    /// Enable verbose output
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Add extra Valgrind options
    pub fn with_extra_options(mut self, options: Vec<String>) -> Self {
        self.extra_options = options;
        self
    }
}

/// Valgrind verification backend for memory debugging
///
/// Valgrind is a comprehensive memory debugging and profiling toolkit.
/// This backend supports:
///
/// - **Memcheck**: Detect memory leaks, use of uninitialized memory,
///   invalid memory access (reads/writes), double-frees, etc.
/// - **Helgrind**: Detect data races and threading errors
/// - **DRD**: Another race condition detector
/// - **Massif/Cachegrind/Callgrind**: Profiling tools
///
/// # Requirements
///
/// Install Valgrind:
/// ```bash
/// # Linux (Debian/Ubuntu)
/// apt install valgrind
///
/// # Linux (Fedora/RHEL)
/// dnf install valgrind
///
/// # macOS (limited support, may require building from source)
/// # Note: macOS support is limited and may not work on ARM
/// ```
///
/// # Usage with Rust
///
/// Build your Rust binary with debug symbols:
/// ```bash
/// cargo build  # Debug build (default has symbols)
/// # Or for release with symbols:
/// RUSTFLAGS="-C debuginfo=2" cargo build --release
/// ```
///
/// Then run with Valgrind:
/// ```bash
/// valgrind --tool=memcheck --leak-check=full ./target/debug/my_program
/// ```
pub struct ValgrindBackend {
    config: ValgrindConfig,
}

impl ValgrindBackend {
    /// Create a new Valgrind backend with default configuration
    pub fn new() -> Self {
        Self {
            config: ValgrindConfig::default(),
        }
    }

    /// Create a new Valgrind backend with custom configuration
    pub fn with_config(config: ValgrindConfig) -> Self {
        Self { config }
    }

    /// Run Valgrind analysis on a binary
    pub async fn analyze_binary(&self, binary_path: &Path) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        // Determine Valgrind command
        let valgrind_cmd = self
            .config
            .valgrind_path
            .clone()
            .unwrap_or_else(|| PathBuf::from("valgrind"));

        // Build command arguments
        let mut args = Vec::new();

        // Add tool selection
        let tool_arg = match self.config.tool {
            ValgrindTool::Memcheck => "--tool=memcheck",
            ValgrindTool::Helgrind => "--tool=helgrind",
            ValgrindTool::DRD => "--tool=drd",
            ValgrindTool::Massif => "--tool=massif",
            ValgrindTool::Cachegrind => "--tool=cachegrind",
            ValgrindTool::Callgrind => "--tool=callgrind",
        };
        args.push(tool_arg.to_string());

        // Memcheck-specific options
        if matches!(self.config.tool, ValgrindTool::Memcheck) {
            let leak_check_arg = match self.config.leak_check {
                LeakCheckLevel::No => "--leak-check=no",
                LeakCheckLevel::Summary => "--leak-check=summary",
                LeakCheckLevel::Full => "--leak-check=full",
                LeakCheckLevel::Reachable => "--leak-check=full",
            };
            args.push(leak_check_arg.to_string());

            if self.config.track_origins {
                args.push("--track-origins=yes".to_string());
            }

            if self.config.show_reachable
                || matches!(self.config.leak_check, LeakCheckLevel::Reachable)
            {
                args.push("--show-reachable=yes".to_string());
            }
        }

        // Error exit code
        args.push(format!("--error-exitcode={}", self.config.error_exitcode));

        // XML output
        if self.config.xml_output {
            args.push("--xml=yes".to_string());
        }

        // Suppressions file
        if let Some(ref supp) = self.config.suppressions {
            args.push(format!("--suppressions={}", supp.display()));
        }

        // Verbose
        if self.config.verbose {
            args.push("--verbose".to_string());
        }

        // Extra options
        for opt in &self.config.extra_options {
            args.push(opt.clone());
        }

        // Add the binary and its arguments
        args.push(binary_path.to_string_lossy().to_string());
        args.extend(self.config.binary_args.clone());

        // Run Valgrind
        let mut cmd = Command::new(&valgrind_cmd);
        cmd.args(&args);

        let output = tokio::time::timeout(self.config.timeout, cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| {
                BackendError::VerificationFailed(format!("Failed to run Valgrind: {}", e))
            })?;

        let duration = start.elapsed();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined = format!("{}\n{}", stdout, stderr);

        // Parse Valgrind output
        let (status, findings, stats) = self.parse_valgrind_output(
            &combined,
            output.status.code().unwrap_or(-1),
            self.config.error_exitcode,
        );

        // Build diagnostics
        let mut diagnostics = Vec::new();

        // Summary
        let tool_name = match self.config.tool {
            ValgrindTool::Memcheck => "Memcheck",
            ValgrindTool::Helgrind => "Helgrind",
            ValgrindTool::DRD => "DRD",
            ValgrindTool::Massif => "Massif",
            ValgrindTool::Cachegrind => "Cachegrind",
            ValgrindTool::Callgrind => "Callgrind",
        };

        let summary = match &status {
            VerificationStatus::Proven => format!("Valgrind {}: No errors detected", tool_name),
            VerificationStatus::Disproven => {
                format!(
                    "Valgrind {}: {} error(s) detected",
                    tool_name, stats.total_errors
                )
            }
            VerificationStatus::Partial {
                verified_percentage: _,
            } => {
                format!(
                    "Valgrind {}: {} warning(s), {} error(s)",
                    tool_name, stats.warnings, stats.total_errors
                )
            }
            VerificationStatus::Unknown { reason } => format!("Valgrind {}: {}", tool_name, reason),
        };
        diagnostics.push(summary);

        // Add stats for memcheck
        if matches!(self.config.tool, ValgrindTool::Memcheck) && stats.total_errors > 0 {
            diagnostics.push(format!(
                "Memory errors: {} invalid reads, {} invalid writes, {} leaks ({} bytes)",
                stats.invalid_reads, stats.invalid_writes, stats.leaks, stats.bytes_leaked
            ));
        }

        // Add stats for helgrind/drd
        if matches!(self.config.tool, ValgrindTool::Helgrind | ValgrindTool::DRD)
            && stats.total_errors > 0
        {
            diagnostics.push(format!(
                "Threading errors: {} data races, {} lock order violations",
                stats.data_races, stats.lock_violations
            ));
        }

        diagnostics.extend(findings.clone());

        // Build counterexample if errors found
        let counterexample = if !findings.is_empty() {
            Some(StructuredCounterexample::from_raw(combined.clone()))
        } else {
            None
        };

        Ok(BackendResult {
            backend: BackendId::Valgrind,
            status,
            proof: None,
            counterexample,
            diagnostics,
            time_taken: duration,
        })
    }

    fn parse_valgrind_output(
        &self,
        output: &str,
        exit_code: i32,
        error_exitcode: i32,
    ) -> (VerificationStatus, Vec<String>, ValgrindStats) {
        let mut findings = Vec::new();
        let mut stats = ValgrindStats::default();

        // Parse Valgrind output
        for line in output.lines() {
            // Error summary patterns
            if line.contains("ERROR SUMMARY:") {
                if let Some(num) = extract_number(line) {
                    stats.total_errors = num;
                }
            }

            // Memory leak patterns
            if line.contains("definitely lost:") {
                if let Some(bytes) = extract_bytes(line) {
                    stats.bytes_leaked += bytes;
                    stats.leaks += 1;
                }
            }
            if line.contains("indirectly lost:") {
                if let Some(bytes) = extract_bytes(line) {
                    stats.bytes_leaked += bytes;
                }
            }

            // Invalid memory access
            if line.contains("Invalid read") {
                stats.invalid_reads += 1;
                findings.push(line.trim().to_string());
            }
            if line.contains("Invalid write") {
                stats.invalid_writes += 1;
                findings.push(line.trim().to_string());
            }
            if line.contains("Invalid free") || line.contains("double free") {
                stats.invalid_frees += 1;
                findings.push(line.trim().to_string());
            }

            // Uninitialized values
            if line.contains("uninitialised")
                || line.contains("uninitialized")
                || line.contains("Conditional jump or move depends on uninitialised")
            {
                stats.uninitialized_uses += 1;
                findings.push(line.trim().to_string());
            }

            // Data races (helgrind/drd)
            if line.contains("data race")
                || line.contains("Possible data race")
                || line.contains("Conflicting")
            {
                stats.data_races += 1;
                findings.push(line.trim().to_string());
            }

            // Lock violations
            if line.contains("lock order")
                || line.contains("Lock order violation")
                || line.contains("deadlock")
            {
                stats.lock_violations += 1;
                findings.push(line.trim().to_string());
            }

            // Warnings
            if line.contains("Warning:") || line.contains("WARNING:") {
                stats.warnings += 1;
            }
        }

        let status = if exit_code == error_exitcode {
            // Valgrind found errors
            VerificationStatus::Disproven
        } else if stats.total_errors == 0 && findings.is_empty() {
            VerificationStatus::Proven
        } else if stats.total_errors > 0 || !findings.is_empty() {
            VerificationStatus::Disproven
        } else if output.contains("not found") || output.contains("command not found") {
            return (
                VerificationStatus::Unknown {
                    reason: "Valgrind not installed".to_string(),
                },
                Vec::new(),
                stats,
            );
        } else if output.contains("not supported") {
            return (
                VerificationStatus::Unknown {
                    reason: "Valgrind not supported on this platform (e.g., macOS ARM)".to_string(),
                },
                Vec::new(),
                stats,
            );
        } else {
            VerificationStatus::Proven
        };

        (status, findings, stats)
    }

    /// Check if Valgrind is installed
    pub async fn check_installed(&self) -> Result<bool, BackendError> {
        let valgrind_cmd = self
            .config
            .valgrind_path
            .clone()
            .unwrap_or_else(|| PathBuf::from("valgrind"));

        let output = Command::new(&valgrind_cmd).arg("--version").output().await;

        match output {
            Ok(out) if out.status.success() => {
                let stdout = String::from_utf8_lossy(&out.stdout);
                Ok(stdout.contains("valgrind"))
            }
            Ok(_) => Ok(false),
            Err(_) => Ok(false),
        }
    }
}

/// Helper to strip Valgrind PID prefix (==12345==) from a line
fn strip_valgrind_prefix(line: &str) -> &str {
    // Valgrind format: ==PID== message
    // Strip the ==PID== prefix if present
    if let Some(after_first) = line.strip_prefix("==") {
        if let Some(idx) = after_first.find("==") {
            let rest = &after_first[idx + 2..];
            return rest.trim_start();
        }
    }
    line
}

/// Helper to extract a number from a line (after stripping Valgrind prefix)
fn extract_number(line: &str) -> Option<u64> {
    // Strip Valgrind PID prefix first
    let clean_line = strip_valgrind_prefix(line);
    clean_line
        .split_whitespace()
        .filter_map(|w| {
            w.trim_matches(|c: char| !c.is_ascii_digit())
                .replace(',', "")
                .parse()
                .ok()
        })
        .next()
}

/// Helper to extract bytes from a leak line
fn extract_bytes(line: &str) -> Option<u64> {
    // Strip Valgrind PID prefix first
    let clean_line = strip_valgrind_prefix(line);
    // Pattern: "definitely lost: 123 bytes in 4 blocks"
    for word in clean_line.split_whitespace() {
        if let Ok(bytes) = word.trim_matches(|c: char| !c.is_ascii_digit()).parse() {
            return Some(bytes);
        }
    }
    None
}

/// Statistics from Valgrind run
#[derive(Debug, Clone, Default)]
struct ValgrindStats {
    total_errors: u64,
    warnings: u64,
    invalid_reads: u64,
    invalid_writes: u64,
    invalid_frees: u64,
    uninitialized_uses: u64,
    leaks: u64,
    bytes_leaked: u64,
    data_races: u64,
    lock_violations: u64,
}

impl Default for ValgrindBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VerificationBackend for ValgrindBackend {
    fn id(&self) -> BackendId {
        BackendId::Valgrind
    }

    fn supports(&self) -> Vec<PropertyType> {
        match self.config.tool {
            ValgrindTool::Memcheck => vec![PropertyType::MemorySafety, PropertyType::MemoryLeak],
            ValgrindTool::Helgrind | ValgrindTool::DRD => vec![PropertyType::DataRace],
            _ => vec![], // Profiling tools don't verify properties
        }
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let binary_path = self.config.binary_path.clone().ok_or_else(|| {
            BackendError::Unavailable(
                "Valgrind backend requires binary_path pointing to an executable".to_string(),
            )
        })?;

        self.analyze_binary(&binary_path).await
    }

    async fn health_check(&self) -> HealthStatus {
        match self.check_installed().await {
            Ok(true) => HealthStatus::Healthy,
            Ok(false) => HealthStatus::Unavailable {
                reason: "Valgrind not installed. Install via package manager (apt/dnf/brew)"
                    .to_string(),
            },
            Err(e) => HealthStatus::Unavailable {
                reason: format!("Failed to check Valgrind installation: {}", e),
            },
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ===== ValgrindTool defaults =====

    #[kani::proof]
    fn verify_tool_default_memcheck() {
        let tool = ValgrindTool::default();
        assert!(matches!(tool, ValgrindTool::Memcheck));
    }

    // ===== LeakCheckLevel defaults =====

    #[kani::proof]
    fn verify_leak_check_default_full() {
        let level = LeakCheckLevel::default();
        assert!(matches!(level, LeakCheckLevel::Full));
    }

    // ===== ValgrindConfig defaults =====

    #[kani::proof]
    fn verify_config_defaults_timeout() {
        let config = ValgrindConfig::default();
        assert!(config.timeout == Duration::from_secs(300));
    }

    #[kani::proof]
    fn verify_config_defaults_tool() {
        let config = ValgrindConfig::default();
        assert!(matches!(config.tool, ValgrindTool::Memcheck));
        assert!(matches!(config.leak_check, LeakCheckLevel::Full));
    }

    #[kani::proof]
    fn verify_config_defaults_options() {
        let config = ValgrindConfig::default();
        assert!(config.binary_path.is_none());
        assert!(config.binary_args.is_empty());
        assert!(config.valgrind_path.is_none());
        assert!(config.track_origins);
        assert!(!config.show_reachable);
        assert!(!config.xml_output);
        assert!(config.suppressions.is_none());
        assert!(config.error_exitcode == 1);
        assert!(!config.verbose);
    }

    // ===== Config builders =====

    #[kani::proof]
    fn verify_config_with_binary_path() {
        let config = ValgrindConfig::default().with_binary_path(PathBuf::from("/test/binary"));
        assert!(config.binary_path == Some(PathBuf::from("/test/binary")));
    }

    #[kani::proof]
    fn verify_config_with_tool() {
        let config = ValgrindConfig::default().with_tool(ValgrindTool::Helgrind);
        assert!(matches!(config.tool, ValgrindTool::Helgrind));
    }

    #[kani::proof]
    fn verify_config_with_leak_check() {
        let config = ValgrindConfig::default().with_leak_check(LeakCheckLevel::Summary);
        assert!(matches!(config.leak_check, LeakCheckLevel::Summary));
    }

    #[kani::proof]
    fn verify_config_with_track_origins() {
        let config = ValgrindConfig::default().with_track_origins(false);
        assert!(!config.track_origins);
    }

    #[kani::proof]
    fn verify_config_with_timeout() {
        let config = ValgrindConfig::default().with_timeout(Duration::from_secs(60));
        assert!(config.timeout == Duration::from_secs(60));
    }

    #[kani::proof]
    fn verify_config_with_error_exitcode() {
        let config = ValgrindConfig::default().with_error_exitcode(42);
        assert!(config.error_exitcode == 42);
    }

    // ===== Backend construction =====

    #[kani::proof]
    fn verify_backend_new_uses_defaults() {
        let backend = ValgrindBackend::new();
        assert!(backend.config.timeout == Duration::from_secs(300));
        assert!(matches!(backend.config.tool, ValgrindTool::Memcheck));
    }

    #[kani::proof]
    fn verify_backend_default_matches_new() {
        let b1 = ValgrindBackend::new();
        let b2 = ValgrindBackend::default();
        assert!(b1.config.timeout == b2.config.timeout);
        assert!(b1.config.track_origins == b2.config.track_origins);
    }

    // ===== ID and supports =====

    #[kani::proof]
    fn verify_backend_id() {
        let backend = ValgrindBackend::new();
        assert!(matches!(backend.id(), BackendId::Valgrind));
    }

    #[kani::proof]
    fn verify_supports_memcheck() {
        let backend = ValgrindBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::MemorySafety));
        assert!(supported.contains(&PropertyType::MemoryLeak));
    }

    #[kani::proof]
    fn verify_supports_helgrind() {
        let config = ValgrindConfig::default().with_tool(ValgrindTool::Helgrind);
        let backend = ValgrindBackend::with_config(config);
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::DataRace));
    }

    // ===== Output parsing =====

    #[kani::proof]
    fn verify_parse_output_clean() {
        let backend = ValgrindBackend::new();
        let (status, _, stats) = backend.parse_valgrind_output("ERROR SUMMARY: 0 errors", 0, 1);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(stats.total_errors == 0);
    }

    #[kani::proof]
    fn verify_parse_output_invalid_read() {
        let backend = ValgrindBackend::new();
        let (status, findings, stats) =
            backend.parse_valgrind_output("Invalid read of size 4\nERROR SUMMARY: 1 errors", 1, 1);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(stats.invalid_reads == 1);
        assert!(!findings.is_empty());
    }

    #[kani::proof]
    fn verify_parse_output_data_race() {
        let backend = ValgrindBackend::new();
        let (_, _, stats) = backend.parse_valgrind_output("Possible data race during read", 1, 1);
        assert!(stats.data_races == 1);
    }

    // ===== Helper functions =====

    #[kani::proof]
    fn verify_strip_valgrind_prefix_with_prefix() {
        let result = strip_valgrind_prefix("==12345== ERROR SUMMARY: 5 errors");
        assert!(result == "ERROR SUMMARY: 5 errors");
    }

    #[kani::proof]
    fn verify_strip_valgrind_prefix_without_prefix() {
        let result = strip_valgrind_prefix("No prefix here");
        assert!(result == "No prefix here");
    }

    #[kani::proof]
    fn verify_extract_number_simple() {
        let result = extract_number("ERROR SUMMARY: 5 errors");
        assert!(result == Some(5));
    }

    #[kani::proof]
    fn verify_extract_number_with_prefix() {
        let result = extract_number("==12345== ERROR SUMMARY: 5 errors");
        assert!(result == Some(5));
    }

    #[kani::proof]
    fn verify_extract_bytes_simple() {
        let result = extract_bytes("definitely lost: 128 bytes in 2 blocks");
        assert!(result == Some(128));
    }

    #[kani::proof]
    fn verify_extract_bytes_with_prefix() {
        let result = extract_bytes("==12345== definitely lost: 128 bytes in 2 blocks");
        assert!(result == Some(128));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valgrind_config_default() {
        let config = ValgrindConfig::default();
        assert!(config.binary_path.is_none());
        assert!(config.binary_args.is_empty());
        assert!(config.valgrind_path.is_none());
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert_eq!(config.tool, ValgrindTool::Memcheck);
        assert_eq!(config.leak_check, LeakCheckLevel::Full);
        assert!(config.track_origins);
        assert!(!config.show_reachable);
        assert!(!config.xml_output);
        assert!(config.suppressions.is_none());
        assert_eq!(config.error_exitcode, 1);
        assert!(!config.verbose);
        assert!(config.extra_options.is_empty());
    }

    #[test]
    fn test_valgrind_config_builder() {
        let config = ValgrindConfig::default()
            .with_binary_path(PathBuf::from("/test/binary"))
            .with_binary_args(vec!["--arg1".to_string(), "--arg2".to_string()])
            .with_valgrind_path(PathBuf::from("/usr/local/bin/valgrind"))
            .with_timeout(Duration::from_secs(120))
            .with_tool(ValgrindTool::Helgrind)
            .with_leak_check(LeakCheckLevel::Summary)
            .with_track_origins(false)
            .with_show_reachable(true)
            .with_xml_output(true)
            .with_suppressions(PathBuf::from("/test/supp.txt"))
            .with_error_exitcode(42)
            .with_verbose(true)
            .with_extra_options(vec!["--gen-suppressions=all".to_string()]);

        assert_eq!(config.binary_path, Some(PathBuf::from("/test/binary")));
        assert_eq!(config.binary_args, vec!["--arg1", "--arg2"]);
        assert_eq!(
            config.valgrind_path,
            Some(PathBuf::from("/usr/local/bin/valgrind"))
        );
        assert_eq!(config.timeout, Duration::from_secs(120));
        assert_eq!(config.tool, ValgrindTool::Helgrind);
        assert_eq!(config.leak_check, LeakCheckLevel::Summary);
        assert!(!config.track_origins);
        assert!(config.show_reachable);
        assert!(config.xml_output);
        assert_eq!(config.suppressions, Some(PathBuf::from("/test/supp.txt")));
        assert_eq!(config.error_exitcode, 42);
        assert!(config.verbose);
        assert_eq!(config.extra_options, vec!["--gen-suppressions=all"]);
    }

    #[test]
    fn test_valgrind_backend_id() {
        let backend = ValgrindBackend::new();
        assert_eq!(backend.id(), BackendId::Valgrind);
    }

    #[test]
    fn test_valgrind_supports_memcheck() {
        let backend = ValgrindBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::MemorySafety));
        assert!(supported.contains(&PropertyType::MemoryLeak));
    }

    #[test]
    fn test_valgrind_supports_helgrind() {
        let backend = ValgrindBackend::with_config(
            ValgrindConfig::default().with_tool(ValgrindTool::Helgrind),
        );
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::DataRace));
    }

    #[test]
    fn test_valgrind_parse_output_clean() {
        let backend = ValgrindBackend::new();
        let output = "==12345== Memcheck, a memory error detector\n==12345== ERROR SUMMARY: 0 errors from 0 contexts";
        let (status, findings, stats) = backend.parse_valgrind_output(output, 0, 1);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(findings.is_empty());
        assert_eq!(stats.total_errors, 0);
    }

    #[test]
    fn test_valgrind_parse_output_with_errors() {
        let backend = ValgrindBackend::new();
        let output = r#"==12345== Invalid read of size 4
==12345==    at 0x1234: foo (test.c:10)
==12345== ERROR SUMMARY: 3 errors from 2 contexts"#;
        let (status, findings, stats) = backend.parse_valgrind_output(output, 1, 1);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(!findings.is_empty());
        assert_eq!(stats.total_errors, 3);
        assert_eq!(stats.invalid_reads, 1);
    }

    #[test]
    fn test_valgrind_parse_output_with_leaks() {
        let backend = ValgrindBackend::new();
        let output = r#"==12345== LEAK SUMMARY:
==12345==    definitely lost: 128 bytes in 2 blocks
==12345== ERROR SUMMARY: 2 errors from 1 contexts"#;
        let (status, _, stats) = backend.parse_valgrind_output(output, 1, 1);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert_eq!(stats.bytes_leaked, 128);
        assert_eq!(stats.leaks, 1);
    }

    #[test]
    fn test_valgrind_parse_output_data_race() {
        let backend = ValgrindBackend::with_config(
            ValgrindConfig::default().with_tool(ValgrindTool::Helgrind),
        );
        let output =
            "==12345== Possible data race during read of size 8\n==12345== ERROR SUMMARY: 1 errors";
        let (status, findings, stats) = backend.parse_valgrind_output(output, 1, 1);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(!findings.is_empty());
        assert_eq!(stats.data_races, 1);
    }

    #[test]
    fn test_strip_valgrind_prefix() {
        assert_eq!(
            strip_valgrind_prefix("==12345== ERROR SUMMARY: 5 errors"),
            "ERROR SUMMARY: 5 errors"
        );
        assert_eq!(strip_valgrind_prefix("No prefix here"), "No prefix here");
    }

    #[test]
    fn test_extract_number() {
        assert_eq!(extract_number("ERROR SUMMARY: 5 errors"), Some(5));
        assert_eq!(extract_number("ERROR SUMMARY: 1,234 errors"), Some(1234));
        assert_eq!(extract_number("No numbers here"), None);
        // With Valgrind prefix
        assert_eq!(extract_number("==12345== ERROR SUMMARY: 5 errors"), Some(5));
    }

    #[test]
    fn test_extract_bytes() {
        assert_eq!(
            extract_bytes("definitely lost: 128 bytes in 2 blocks"),
            Some(128)
        );
        assert_eq!(extract_bytes("No bytes info"), None);
        // With Valgrind prefix
        assert_eq!(
            extract_bytes("==12345== definitely lost: 128 bytes in 2 blocks"),
            Some(128)
        );
    }

    #[tokio::test]
    async fn test_valgrind_health_check() {
        let backend = ValgrindBackend::new();
        let health = backend.health_check().await;
        // Valgrind may or may not be installed
        match health {
            HealthStatus::Healthy => {}
            HealthStatus::Unavailable { reason } => {
                assert!(
                    reason.contains("Valgrind")
                        || reason.contains("valgrind")
                        || reason.contains("not installed")
                );
            }
            HealthStatus::Degraded { .. } => {}
        }
    }

    #[tokio::test]
    async fn test_valgrind_verify_requires_binary_path() {
        use dashprove_usl::{parse, typecheck};

        let backend = ValgrindBackend::new();
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
    fn test_valgrind_tool_default() {
        assert_eq!(ValgrindTool::default(), ValgrindTool::Memcheck);
    }

    #[test]
    fn test_leak_check_level_default() {
        assert_eq!(LeakCheckLevel::default(), LeakCheckLevel::Full);
    }
}
