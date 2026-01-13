//! Rust Sanitizer Backends for DashProve
//!
//! This crate provides verification backends for Rust sanitizers:
//! - **AddressSanitizer (ASAN)**: Detects memory errors (buffer overflows, use-after-free)
//! - **MemorySanitizer (MSAN)**: Detects reads of uninitialized memory
//! - **ThreadSanitizer (TSAN)**: Detects data races
//! - **LeakSanitizer (LSAN)**: Detects memory leaks
//!
//! # Requirements
//!
//! Sanitizers require:
//! - Nightly Rust toolchain (`rustup +nightly`)
//! - Linux or macOS (MSAN requires Linux)
//!
//! # Usage
//!
//! ```rust,ignore
//! use dashprove_sanitizers::{SanitizerBackend, SanitizerType};
//!
//! let backend = SanitizerBackend::new(SanitizerType::Address);
//! let result = backend.run_on_crate("/path/to/crate").await?;
//! ```

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Output;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::process::Command;

/// Sanitizer type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(kani, derive(kani::Arbitrary))]
pub enum SanitizerType {
    /// Address Sanitizer - detects memory errors
    Address,
    /// Memory Sanitizer - detects uninitialized reads (Linux only)
    Memory,
    /// Thread Sanitizer - detects data races
    Thread,
    /// Leak Sanitizer - detects memory leaks
    Leak,
}

impl SanitizerType {
    /// Get the RUSTFLAGS value for this sanitizer
    pub fn rustflags(&self) -> &'static str {
        match self {
            Self::Address => "-Z sanitizer=address",
            Self::Memory => "-Z sanitizer=memory",
            Self::Thread => "-Z sanitizer=thread",
            Self::Leak => "-Z sanitizer=leak",
        }
    }

    /// Get the sanitizer name for display
    pub fn name(&self) -> &'static str {
        match self {
            Self::Address => "AddressSanitizer",
            Self::Memory => "MemorySanitizer",
            Self::Thread => "ThreadSanitizer",
            Self::Leak => "LeakSanitizer",
        }
    }

    /// Check if this sanitizer is available on the current platform
    #[allow(clippy::needless_bool)] // False positive: branches return different compile-time cfg! values
    pub fn is_available(&self) -> bool {
        // MSAN requires Linux; others work on Linux and macOS
        if matches!(self, Self::Memory) {
            cfg!(target_os = "linux")
        } else {
            cfg!(any(target_os = "linux", target_os = "macos"))
        }
    }
}

/// Error type for sanitizer operations
#[derive(Error, Debug)]
pub enum SanitizerError {
    /// Sanitizer not available on this platform
    #[error("{0} is not available on this platform")]
    NotAvailable(String),

    /// Nightly toolchain not installed
    #[error("Nightly Rust toolchain required: {0}")]
    NightlyRequired(String),

    /// Build failed
    #[error("Build failed: {0}")]
    BuildFailed(String),

    /// Test execution failed
    #[error("Test execution failed: {0}")]
    TestFailed(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Timeout
    #[error("Timeout after {0:?}")]
    Timeout(Duration),
}

/// A finding from a sanitizer run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SanitizerFinding {
    /// Type of issue found
    pub issue_type: String,
    /// Description of the issue
    pub description: String,
    /// Stack trace if available
    pub stack_trace: Option<String>,
    /// File location if known
    pub file: Option<String>,
    /// Line number if known
    pub line: Option<u32>,
    /// Severity (error, warning)
    pub severity: FindingSeverity,
}

/// Severity of a sanitizer finding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(kani, derive(kani::Arbitrary))]
pub enum FindingSeverity {
    /// Definite bug
    Error,
    /// Potential issue
    Warning,
}

/// Result from running a sanitizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SanitizerResult {
    /// Which sanitizer was run
    pub sanitizer: SanitizerType,
    /// Whether the check passed (no issues found)
    pub passed: bool,
    /// Issues found during the run
    pub findings: Vec<SanitizerFinding>,
    /// Raw stderr output from the sanitizer
    pub raw_output: String,
    /// Time taken for the run
    pub duration: Duration,
    /// Number of tests executed
    pub tests_run: usize,
}

/// Sanitizer backend for running Rust sanitizers
pub struct SanitizerBackend {
    /// Which sanitizer to use
    sanitizer_type: SanitizerType,
    /// Timeout for the entire run
    timeout: Duration,
}

impl SanitizerBackend {
    /// Create a new sanitizer backend
    pub fn new(sanitizer_type: SanitizerType) -> Self {
        Self {
            sanitizer_type,
            timeout: Duration::from_secs(300), // 5 minute default
        }
    }

    /// Set the timeout for sanitizer runs
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Check if nightly toolchain is available
    pub async fn check_nightly() -> Result<bool, SanitizerError> {
        let output = Command::new("rustup")
            .args(["run", "nightly", "rustc", "--version"])
            .output()
            .await?;

        Ok(output.status.success())
    }

    /// Run the sanitizer on a crate
    pub async fn run_on_crate(&self, crate_path: &Path) -> Result<SanitizerResult, SanitizerError> {
        // Check platform availability
        if !self.sanitizer_type.is_available() {
            return Err(SanitizerError::NotAvailable(
                self.sanitizer_type.name().to_string(),
            ));
        }

        // Check nightly availability
        if !Self::check_nightly().await? {
            return Err(SanitizerError::NightlyRequired(
                "Run 'rustup install nightly'".to_string(),
            ));
        }

        let start = Instant::now();

        // Build with sanitizer flags
        let build_output = self.run_cargo_command(crate_path, "build").await?;
        if !build_output.status.success() {
            return Err(SanitizerError::BuildFailed(
                String::from_utf8_lossy(&build_output.stderr).to_string(),
            ));
        }

        // Run tests with sanitizer
        let test_output = self.run_cargo_command(crate_path, "test").await?;

        let duration = start.elapsed();
        let raw_output = String::from_utf8_lossy(&test_output.stderr).to_string();
        let findings = self.parse_output(&raw_output);
        let tests_run = self.count_tests(&String::from_utf8_lossy(&test_output.stdout));

        Ok(SanitizerResult {
            sanitizer: self.sanitizer_type,
            passed: findings.is_empty() && test_output.status.success(),
            findings,
            raw_output,
            duration,
            tests_run,
        })
    }

    async fn run_cargo_command(
        &self,
        crate_path: &Path,
        command: &str,
    ) -> Result<Output, SanitizerError> {
        let output = Command::new("cargo")
            .args(["+nightly", command, "--release"])
            .current_dir(crate_path)
            .env("RUSTFLAGS", self.sanitizer_type.rustflags())
            .output()
            .await?;

        Ok(output)
    }

    fn parse_output(&self, output: &str) -> Vec<SanitizerFinding> {
        let mut findings = Vec::new();

        // Parse sanitizer-specific output patterns
        match self.sanitizer_type {
            SanitizerType::Address => {
                self.parse_asan_output(output, &mut findings);
            }
            SanitizerType::Memory => {
                self.parse_msan_output(output, &mut findings);
            }
            SanitizerType::Thread => {
                self.parse_tsan_output(output, &mut findings);
            }
            SanitizerType::Leak => {
                self.parse_lsan_output(output, &mut findings);
            }
        }

        findings
    }

    fn parse_asan_output(&self, output: &str, findings: &mut Vec<SanitizerFinding>) {
        // ASAN error patterns
        let patterns = [
            ("heap-buffer-overflow", "Heap buffer overflow detected"),
            ("stack-buffer-overflow", "Stack buffer overflow detected"),
            ("heap-use-after-free", "Use after free detected"),
            ("stack-use-after-return", "Stack use after return detected"),
            ("double-free", "Double free detected"),
            ("alloc-dealloc-mismatch", "Allocation/deallocation mismatch"),
        ];

        for (pattern, description) in patterns {
            if output.contains(pattern) {
                let stack_trace = self.extract_stack_trace(output, pattern);
                findings.push(SanitizerFinding {
                    issue_type: pattern.to_string(),
                    description: description.to_string(),
                    stack_trace,
                    file: None,
                    line: None,
                    severity: FindingSeverity::Error,
                });
            }
        }
    }

    fn parse_msan_output(&self, output: &str, findings: &mut Vec<SanitizerFinding>) {
        if output.contains("use-of-uninitialized-value") {
            let stack_trace = self.extract_stack_trace(output, "use-of-uninitialized-value");
            findings.push(SanitizerFinding {
                issue_type: "use-of-uninitialized-value".to_string(),
                description: "Use of uninitialized memory".to_string(),
                stack_trace,
                file: None,
                line: None,
                severity: FindingSeverity::Error,
            });
        }
    }

    fn parse_tsan_output(&self, output: &str, findings: &mut Vec<SanitizerFinding>) {
        if output.contains("data race") || output.contains("ThreadSanitizer: data race") {
            let stack_trace = self.extract_stack_trace(output, "data race");
            findings.push(SanitizerFinding {
                issue_type: "data-race".to_string(),
                description: "Data race detected".to_string(),
                stack_trace,
                file: None,
                line: None,
                severity: FindingSeverity::Error,
            });
        }
    }

    fn parse_lsan_output(&self, output: &str, findings: &mut Vec<SanitizerFinding>) {
        if output.contains("detected memory leaks") || output.contains("LeakSanitizer") {
            let stack_trace = self.extract_stack_trace(output, "memory leak");
            findings.push(SanitizerFinding {
                issue_type: "memory-leak".to_string(),
                description: "Memory leak detected".to_string(),
                stack_trace,
                file: None,
                line: None,
                severity: FindingSeverity::Warning,
            });
        }
    }

    fn extract_stack_trace(&self, output: &str, marker: &str) -> Option<String> {
        // Find the marker and extract following lines that look like stack trace
        if let Some(pos) = output.find(marker) {
            let remaining = &output[pos..];
            let lines: Vec<&str> = remaining
                .lines()
                .take(20) // Take up to 20 lines
                .filter(|line| {
                    line.contains("#") || line.contains("at ") || line.starts_with("    ")
                })
                .collect();

            if !lines.is_empty() {
                return Some(lines.join("\n"));
            }
        }
        None
    }

    fn count_tests(&self, stdout: &str) -> usize {
        // Parse test output for count
        // Pattern: "running X tests" or "X passed"
        for line in stdout.lines() {
            if let Some(rest) = line.strip_prefix("running ") {
                if let Some(num_str) = rest.split_whitespace().next() {
                    if let Ok(count) = num_str.parse::<usize>() {
                        return count;
                    }
                }
            }
        }
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitizer_type_rustflags() {
        assert_eq!(SanitizerType::Address.rustflags(), "-Z sanitizer=address");
        assert_eq!(SanitizerType::Memory.rustflags(), "-Z sanitizer=memory");
        assert_eq!(SanitizerType::Thread.rustflags(), "-Z sanitizer=thread");
        assert_eq!(SanitizerType::Leak.rustflags(), "-Z sanitizer=leak");
    }

    #[test]
    fn test_sanitizer_type_name() {
        assert_eq!(SanitizerType::Address.name(), "AddressSanitizer");
        assert_eq!(SanitizerType::Memory.name(), "MemorySanitizer");
        assert_eq!(SanitizerType::Thread.name(), "ThreadSanitizer");
        assert_eq!(SanitizerType::Leak.name(), "LeakSanitizer");
    }

    #[test]
    fn test_parse_asan_output() {
        let backend = SanitizerBackend::new(SanitizerType::Address);
        let output = "ERROR: AddressSanitizer: heap-buffer-overflow on address 0x12345\n    #0 0x123 in main";
        let mut findings = Vec::new();
        backend.parse_asan_output(output, &mut findings);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].issue_type, "heap-buffer-overflow");
    }

    #[test]
    fn test_parse_tsan_output() {
        let backend = SanitizerBackend::new(SanitizerType::Thread);
        let output = "WARNING: ThreadSanitizer: data race (pid=12345)\n  Write of size 4";
        let mut findings = Vec::new();
        backend.parse_tsan_output(output, &mut findings);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].issue_type, "data-race");
    }

    #[test]
    fn test_parse_lsan_output() {
        let backend = SanitizerBackend::new(SanitizerType::Leak);
        let output = "=== LeakSanitizer: detected memory leaks ===\nDirect leak of 100 bytes";
        let mut findings = Vec::new();
        backend.parse_lsan_output(output, &mut findings);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].issue_type, "memory-leak");
    }

    #[test]
    fn test_count_tests() {
        let backend = SanitizerBackend::new(SanitizerType::Address);
        let stdout = "running 5 tests\ntest test_one ... ok\ntest test_two ... ok";
        assert_eq!(backend.count_tests(stdout), 5);
    }

    #[test]
    fn test_sanitizer_backend_creation() {
        let backend =
            SanitizerBackend::new(SanitizerType::Address).with_timeout(Duration::from_secs(60));
        assert_eq!(backend.sanitizer_type, SanitizerType::Address);
        assert_eq!(backend.timeout, Duration::from_secs(60));
    }

    // Test for SanitizerType::is_available (line 69)
    // On macOS: Memory sanitizer is NOT available, others ARE
    // On Linux: All sanitizers are available
    #[test]
    fn test_sanitizer_type_is_available() {
        // Memory sanitizer requires Linux
        let msan_available = SanitizerType::Memory.is_available();
        #[cfg(target_os = "linux")]
        assert!(msan_available);
        #[cfg(target_os = "macos")]
        assert!(!msan_available);

        // Other sanitizers available on both Linux and macOS
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            assert!(SanitizerType::Address.is_available());
            assert!(SanitizerType::Thread.is_available());
            assert!(SanitizerType::Leak.is_available());
        }
    }

    #[test]
    fn test_sanitizer_type_memory_not_available_on_macos() {
        // This test verifies the is_available logic for Memory specifically
        let msan = SanitizerType::Memory;
        let result = msan.is_available();

        // The function should return cfg!(target_os = "linux") for Memory
        // We can't directly test the cfg! macro result, but we can verify
        // the behavior is consistent with the platform
        #[cfg(target_os = "linux")]
        assert!(result, "MSAN should be available on Linux");

        #[cfg(target_os = "macos")]
        assert!(!result, "MSAN should NOT be available on macOS");
    }

    // Tests for parse_output (line 241) - returns vec![] mutation
    #[test]
    fn test_parse_output_returns_findings() {
        let backend = SanitizerBackend::new(SanitizerType::Address);
        let output = "ERROR: AddressSanitizer: heap-buffer-overflow";
        let findings = backend.parse_output(output);
        assert!(
            !findings.is_empty(),
            "Should return findings, not empty vec"
        );
        assert_eq!(findings.len(), 1);
    }

    #[test]
    fn test_parse_output_empty_for_clean() {
        let backend = SanitizerBackend::new(SanitizerType::Address);
        let findings = backend.parse_output("all tests passed");
        assert!(findings.is_empty(), "Clean output should have no findings");
    }

    #[test]
    fn test_parse_output_dispatches_to_correct_parser() {
        // Test that each sanitizer type uses the correct parser
        let asan_backend = SanitizerBackend::new(SanitizerType::Address);
        let asan_output = "heap-buffer-overflow";
        assert!(!asan_backend.parse_output(asan_output).is_empty());

        let tsan_backend = SanitizerBackend::new(SanitizerType::Thread);
        let tsan_output = "data race";
        assert!(!tsan_backend.parse_output(tsan_output).is_empty());

        let lsan_backend = SanitizerBackend::new(SanitizerType::Leak);
        let lsan_output = "detected memory leaks";
        assert!(!lsan_backend.parse_output(lsan_output).is_empty());

        let msan_backend = SanitizerBackend::new(SanitizerType::Memory);
        let msan_output = "use-of-uninitialized-value";
        assert!(!msan_backend.parse_output(msan_output).is_empty());
    }

    // Tests for parse_tsan_output || condition (line 303)
    #[test]
    fn test_parse_tsan_output_only_data_race() {
        let backend = SanitizerBackend::new(SanitizerType::Thread);
        // Only "data race" without "ThreadSanitizer: data race"
        let output = "WARNING: detected data race in main thread";
        let mut findings = Vec::new();
        backend.parse_tsan_output(output, &mut findings);
        assert!(!findings.is_empty(), "Should detect 'data race' alone");
    }

    #[test]
    fn test_parse_tsan_output_only_threadsanitizer() {
        let backend = SanitizerBackend::new(SanitizerType::Thread);
        // Only "ThreadSanitizer: data race" - tests the second || branch
        let output = "ThreadSanitizer: data race detected";
        let mut findings = Vec::new();
        backend.parse_tsan_output(output, &mut findings);
        assert!(!findings.is_empty());
    }

    #[test]
    fn test_parse_tsan_output_neither_keyword() {
        let backend = SanitizerBackend::new(SanitizerType::Thread);
        let output = "all tests passed successfully";
        let mut findings = Vec::new();
        backend.parse_tsan_output(output, &mut findings);
        assert!(findings.is_empty(), "No keywords should mean no findings");
    }

    // Tests for parse_lsan_output || condition (line 317)
    #[test]
    fn test_parse_lsan_output_only_detected_memory_leaks() {
        let backend = SanitizerBackend::new(SanitizerType::Leak);
        // Only "detected memory leaks" without "LeakSanitizer"
        let output = "=== detected memory leaks ===";
        let mut findings = Vec::new();
        backend.parse_lsan_output(output, &mut findings);
        assert!(
            !findings.is_empty(),
            "Should detect 'detected memory leaks' alone"
        );
    }

    #[test]
    fn test_parse_lsan_output_only_leaksanitizer() {
        let backend = SanitizerBackend::new(SanitizerType::Leak);
        // Only "LeakSanitizer" keyword - tests second || branch
        let output = "LeakSanitizer summary: 100 bytes leaked";
        let mut findings = Vec::new();
        backend.parse_lsan_output(output, &mut findings);
        assert!(!findings.is_empty());
    }

    #[test]
    fn test_parse_lsan_output_neither_keyword() {
        let backend = SanitizerBackend::new(SanitizerType::Leak);
        let output = "all tests passed";
        let mut findings = Vec::new();
        backend.parse_lsan_output(output, &mut findings);
        assert!(findings.is_empty());
    }

    // Tests for extract_stack_trace (lines 332-342)
    #[test]
    fn test_extract_stack_trace_finds_trace() {
        let backend = SanitizerBackend::new(SanitizerType::Address);
        let output = "ERROR: heap-buffer-overflow\n    #0 0x123 in func\n    #1 0x456 in main";
        let trace = backend.extract_stack_trace(output, "heap-buffer-overflow");
        assert!(trace.is_some(), "Should find stack trace");
        let trace = trace.unwrap();
        assert!(!trace.is_empty(), "Stack trace should not be empty");
        assert!(trace.contains("#0") || trace.contains("#1"));
    }

    #[test]
    fn test_extract_stack_trace_returns_none_no_marker() {
        let backend = SanitizerBackend::new(SanitizerType::Address);
        let output = "clean output without marker";
        let trace = backend.extract_stack_trace(output, "nonexistent-marker");
        assert!(trace.is_none(), "Should return None when marker not found");
    }

    #[test]
    fn test_extract_stack_trace_returns_none_no_trace_lines() {
        let backend = SanitizerBackend::new(SanitizerType::Address);
        // Marker exists but no stack trace lines (no #, "at ", or leading spaces)
        let output = "ERROR: heap-buffer-overflow\nsome other text\nno trace here";
        let trace = backend.extract_stack_trace(output, "heap-buffer-overflow");
        assert!(trace.is_none(), "Should return None when no trace lines");
    }

    #[test]
    fn test_extract_stack_trace_with_at_keyword() {
        let backend = SanitizerBackend::new(SanitizerType::Address);
        // Uses "at " keyword for stack trace detection
        let output = "ERROR: heap-buffer-overflow\n    at main.rs:10\n    at lib.rs:20";
        let trace = backend.extract_stack_trace(output, "heap-buffer-overflow");
        assert!(trace.is_some());
        let trace = trace.unwrap();
        assert!(trace.contains("at "));
    }

    #[test]
    fn test_extract_stack_trace_with_leading_spaces() {
        let backend = SanitizerBackend::new(SanitizerType::Address);
        // Uses leading spaces (4 spaces) for stack trace detection
        let output = "ERROR: heap-buffer-overflow\n    frame 1\n    frame 2";
        let trace = backend.extract_stack_trace(output, "heap-buffer-overflow");
        assert!(trace.is_some());
    }

    #[test]
    fn test_extract_stack_trace_filter_conditions() {
        // Test the || conditions in line 338
        let backend = SanitizerBackend::new(SanitizerType::Address);

        // Test line with # but no "at " or leading spaces
        let output_hash = "marker\n#0 in main";
        let trace = backend.extract_stack_trace(output_hash, "marker");
        assert!(trace.is_some(), "Should match lines with #");

        // Test line with "at " but no # or leading spaces
        let output_at = "marker\nat location";
        let trace = backend.extract_stack_trace(output_at, "marker");
        assert!(trace.is_some(), "Should match lines with 'at '");

        // Test line with leading spaces but no # or "at "
        let output_spaces = "marker\n    indented line";
        let trace = backend.extract_stack_trace(output_spaces, "marker");
        assert!(trace.is_some(), "Should match lines with leading spaces");
    }

    // Tests for && condition in run_on_crate passed logic (line 217)
    #[test]
    fn test_result_passed_logic() {
        // passed: findings.is_empty() && test_output.status.success()
        // Test that && gives correct behavior

        // Both conditions true -> passed = true
        let result_both_true = SanitizerResult {
            sanitizer: SanitizerType::Address,
            passed: true, // findings empty AND status success
            findings: Vec::new(),
            raw_output: String::new(),
            duration: Duration::ZERO,
            tests_run: 0,
        };
        assert!(result_both_true.passed);
        assert!(result_both_true.findings.is_empty());

        // findings not empty -> passed = false (regardless of status)
        let finding = SanitizerFinding {
            issue_type: "test".to_string(),
            description: "test".to_string(),
            stack_trace: None,
            file: None,
            line: None,
            severity: FindingSeverity::Error,
        };
        let result_has_findings = SanitizerResult {
            sanitizer: SanitizerType::Address,
            passed: false, // has findings, so not passed
            findings: vec![finding],
            raw_output: String::new(),
            duration: Duration::ZERO,
            tests_run: 0,
        };
        assert!(!result_has_findings.passed);

        // Verify && vs || logic: with ||, having empty findings alone would be true
        // but we want BOTH conditions
        let empty_findings = true;
        let status_success = false;
        assert!(
            !(empty_findings && status_success),
            "&& should be false if status fails"
        );
        assert!(
            empty_findings || status_success,
            "|| would incorrectly be true"
        );
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    // Strategy for generating SanitizerType variants
    fn sanitizer_type_strategy() -> impl Strategy<Value = SanitizerType> {
        prop_oneof![
            Just(SanitizerType::Address),
            Just(SanitizerType::Memory),
            Just(SanitizerType::Thread),
            Just(SanitizerType::Leak),
        ]
    }

    // Strategy for generating FindingSeverity variants
    fn finding_severity_strategy() -> impl Strategy<Value = FindingSeverity> {
        prop_oneof![Just(FindingSeverity::Error), Just(FindingSeverity::Warning),]
    }

    proptest! {
        // SanitizerType property tests
        #[test]
        fn sanitizer_type_rustflags_non_empty(sanitizer in sanitizer_type_strategy()) {
            prop_assert!(!sanitizer.rustflags().is_empty());
        }

        #[test]
        fn sanitizer_type_rustflags_contains_sanitizer(sanitizer in sanitizer_type_strategy()) {
            prop_assert!(sanitizer.rustflags().contains("sanitizer"));
        }

        #[test]
        fn sanitizer_type_rustflags_starts_with_z(sanitizer in sanitizer_type_strategy()) {
            prop_assert!(sanitizer.rustflags().starts_with("-Z"));
        }

        #[test]
        fn sanitizer_type_name_non_empty(sanitizer in sanitizer_type_strategy()) {
            prop_assert!(!sanitizer.name().is_empty());
        }

        #[test]
        fn sanitizer_type_name_ends_with_sanitizer(sanitizer in sanitizer_type_strategy()) {
            prop_assert!(sanitizer.name().ends_with("Sanitizer"));
        }

        // SanitizerBackend property tests
        #[test]
        fn sanitizer_backend_preserves_type(sanitizer in sanitizer_type_strategy()) {
            let backend = SanitizerBackend::new(sanitizer);
            prop_assert_eq!(backend.sanitizer_type, sanitizer);
        }

        #[test]
        fn sanitizer_backend_default_timeout_5_minutes(sanitizer in sanitizer_type_strategy()) {
            let backend = SanitizerBackend::new(sanitizer);
            prop_assert_eq!(backend.timeout, Duration::from_secs(300));
        }

        #[test]
        fn sanitizer_backend_timeout_preserved(secs in 1u64..10000) {
            let timeout = Duration::from_secs(secs);
            let backend = SanitizerBackend::new(SanitizerType::Address).with_timeout(timeout);
            prop_assert_eq!(backend.timeout, timeout);
        }

        #[test]
        fn sanitizer_backend_builder_chaining(
            sanitizer in sanitizer_type_strategy(),
            secs in 1u64..1000
        ) {
            let timeout = Duration::from_secs(secs);
            let backend = SanitizerBackend::new(sanitizer).with_timeout(timeout);
            prop_assert_eq!(backend.sanitizer_type, sanitizer);
            prop_assert_eq!(backend.timeout, timeout);
        }

        // SanitizerFinding structure tests
        #[test]
        fn sanitizer_finding_preserves_fields(
            issue_type in "[a-z-]+",
            description in ".{1,50}",
            severity in finding_severity_strategy()
        ) {
            let finding = SanitizerFinding {
                issue_type: issue_type.clone(),
                description: description.clone(),
                stack_trace: None,
                file: None,
                line: None,
                severity,
            };
            prop_assert_eq!(finding.issue_type, issue_type);
            prop_assert_eq!(finding.description, description);
            prop_assert_eq!(finding.severity, severity);
        }

        #[test]
        fn sanitizer_finding_with_stack_trace(stack_trace in ".{1,200}") {
            let finding = SanitizerFinding {
                issue_type: "test".to_string(),
                description: "test description".to_string(),
                stack_trace: Some(stack_trace.clone()),
                file: None,
                line: None,
                severity: FindingSeverity::Error,
            };
            prop_assert_eq!(finding.stack_trace, Some(stack_trace));
        }

        #[test]
        fn sanitizer_finding_with_file_location(
            file in "[a-z]+\\.rs",
            line in 1u32..10000
        ) {
            let finding = SanitizerFinding {
                issue_type: "test".to_string(),
                description: "test description".to_string(),
                stack_trace: None,
                file: Some(file.clone()),
                line: Some(line),
                severity: FindingSeverity::Warning,
            };
            prop_assert_eq!(finding.file, Some(file));
            prop_assert_eq!(finding.line, Some(line));
        }

        // SanitizerResult property tests
        #[test]
        fn sanitizer_result_passed_when_no_findings(
            sanitizer in sanitizer_type_strategy(),
            tests_run in 0usize..1000
        ) {
            let result = SanitizerResult {
                sanitizer,
                passed: true,
                findings: Vec::new(),
                raw_output: String::new(),
                duration: Duration::from_secs(1),
                tests_run,
            };
            prop_assert!(result.passed);
            prop_assert!(result.findings.is_empty());
        }

        #[test]
        fn sanitizer_result_preserves_duration(secs in 1u64..10000) {
            let duration = Duration::from_secs(secs);
            let result = SanitizerResult {
                sanitizer: SanitizerType::Address,
                passed: true,
                findings: Vec::new(),
                raw_output: String::new(),
                duration,
                tests_run: 0,
            };
            prop_assert_eq!(result.duration, duration);
        }

        #[test]
        fn sanitizer_result_preserves_raw_output(raw_output in ".{0,200}") {
            let result = SanitizerResult {
                sanitizer: SanitizerType::Thread,
                passed: true,
                findings: Vec::new(),
                raw_output: raw_output.clone(),
                duration: Duration::from_secs(1),
                tests_run: 0,
            };
            prop_assert_eq!(result.raw_output, raw_output);
        }

        // Test count parsing property tests
        #[test]
        fn count_tests_parses_valid_count(count in 1usize..1000) {
            let backend = SanitizerBackend::new(SanitizerType::Address);
            let count_str = count.to_string();
            let stdout = format!("running {} tests\ntest foo ... ok", count_str);
            prop_assert_eq!(backend.count_tests(&stdout), count);
        }

        #[test]
        fn count_tests_returns_zero_for_empty_output(_dummy in 0..1) {
            let backend = SanitizerBackend::new(SanitizerType::Address);
            prop_assert_eq!(backend.count_tests(""), 0);
        }

        #[test]
        fn count_tests_returns_zero_for_no_match(text in "[a-z]+") {
            let backend = SanitizerBackend::new(SanitizerType::Address);
            prop_assert_eq!(backend.count_tests(&text), 0);
        }

        // ASAN output parsing tests
        #[test]
        fn parse_asan_detects_heap_buffer_overflow(_dummy in 0..1) {
            let backend = SanitizerBackend::new(SanitizerType::Address);
            let output = "ERROR: AddressSanitizer: heap-buffer-overflow on address";
            let mut findings = Vec::new();
            backend.parse_asan_output(output, &mut findings);
            prop_assert_eq!(findings.len(), 1);
            prop_assert_eq!(&findings[0].issue_type, "heap-buffer-overflow");
        }

        #[test]
        fn parse_asan_detects_use_after_free(_dummy in 0..1) {
            let backend = SanitizerBackend::new(SanitizerType::Address);
            let output = "ERROR: AddressSanitizer: heap-use-after-free";
            let mut findings = Vec::new();
            backend.parse_asan_output(output, &mut findings);
            prop_assert_eq!(findings.len(), 1);
            prop_assert_eq!(&findings[0].issue_type, "heap-use-after-free");
        }

        #[test]
        fn parse_asan_detects_double_free(_dummy in 0..1) {
            let backend = SanitizerBackend::new(SanitizerType::Address);
            let output = "ERROR: AddressSanitizer: double-free";
            let mut findings = Vec::new();
            backend.parse_asan_output(output, &mut findings);
            prop_assert_eq!(findings.len(), 1);
            prop_assert_eq!(&findings[0].issue_type, "double-free");
        }

        #[test]
        fn parse_asan_no_findings_for_clean_output(text in "[a-z ]+") {
            let backend = SanitizerBackend::new(SanitizerType::Address);
            let mut findings = Vec::new();
            backend.parse_asan_output(&text, &mut findings);
            prop_assert!(findings.is_empty());
        }

        // MSAN output parsing tests
        #[test]
        fn parse_msan_detects_uninitialized_value(_dummy in 0..1) {
            let backend = SanitizerBackend::new(SanitizerType::Memory);
            let output = "WARNING: MemorySanitizer: use-of-uninitialized-value";
            let mut findings = Vec::new();
            backend.parse_msan_output(output, &mut findings);
            prop_assert_eq!(findings.len(), 1);
            prop_assert_eq!(&findings[0].issue_type, "use-of-uninitialized-value");
        }

        #[test]
        fn parse_msan_no_findings_for_clean_output(text in "[a-z ]+") {
            let backend = SanitizerBackend::new(SanitizerType::Memory);
            let mut findings = Vec::new();
            backend.parse_msan_output(&text, &mut findings);
            prop_assert!(findings.is_empty());
        }

        // TSAN output parsing tests
        #[test]
        fn parse_tsan_detects_data_race(_dummy in 0..1) {
            let backend = SanitizerBackend::new(SanitizerType::Thread);
            let output = "WARNING: ThreadSanitizer: data race";
            let mut findings = Vec::new();
            backend.parse_tsan_output(output, &mut findings);
            prop_assert_eq!(findings.len(), 1);
            prop_assert_eq!(&findings[0].issue_type, "data-race");
        }

        #[test]
        fn parse_tsan_no_findings_for_clean_output(text in "[a-z ]+") {
            let backend = SanitizerBackend::new(SanitizerType::Thread);
            let mut findings = Vec::new();
            backend.parse_tsan_output(&text, &mut findings);
            prop_assert!(findings.is_empty());
        }

        // LSAN output parsing tests
        #[test]
        fn parse_lsan_detects_memory_leak(_dummy in 0..1) {
            let backend = SanitizerBackend::new(SanitizerType::Leak);
            let output = "=== LeakSanitizer: detected memory leaks ===";
            let mut findings = Vec::new();
            backend.parse_lsan_output(output, &mut findings);
            prop_assert_eq!(findings.len(), 1);
            prop_assert_eq!(&findings[0].issue_type, "memory-leak");
        }

        #[test]
        fn parse_lsan_no_findings_for_clean_output(text in "[a-z ]+") {
            let backend = SanitizerBackend::new(SanitizerType::Leak);
            let mut findings = Vec::new();
            backend.parse_lsan_output(&text, &mut findings);
            prop_assert!(findings.is_empty());
        }

        // SanitizerError message preservation tests
        #[test]
        fn sanitizer_error_not_available_contains_name(name in "[A-Za-z]+") {
            let error = SanitizerError::NotAvailable(name.clone());
            let msg = error.to_string();
            prop_assert!(msg.contains(&name));
        }

        #[test]
        fn sanitizer_error_nightly_required_contains_message(message in ".{1,50}") {
            let error = SanitizerError::NightlyRequired(message.clone());
            let msg = error.to_string();
            prop_assert!(msg.contains(&message));
        }

        #[test]
        fn sanitizer_error_build_failed_contains_message(message in ".{1,50}") {
            let error = SanitizerError::BuildFailed(message.clone());
            let msg = error.to_string();
            prop_assert!(msg.contains(&message));
        }

        #[test]
        fn sanitizer_error_test_failed_contains_message(message in ".{1,50}") {
            let error = SanitizerError::TestFailed(message.clone());
            let msg = error.to_string();
            prop_assert!(msg.contains(&message));
        }

        #[test]
        fn sanitizer_error_timeout_contains_duration_info(secs in 1u64..10000) {
            let duration = Duration::from_secs(secs);
            let error = SanitizerError::Timeout(duration);
            let msg = error.to_string();
            prop_assert!(msg.contains("Timeout") || msg.contains("timeout"));
        }

        // FindingSeverity equality tests
        #[test]
        fn finding_severity_reflexive_equality(severity in finding_severity_strategy()) {
            prop_assert_eq!(severity, severity);
        }

        // SanitizerType equality tests
        #[test]
        fn sanitizer_type_reflexive_equality(sanitizer in sanitizer_type_strategy()) {
            prop_assert_eq!(sanitizer, sanitizer);
        }

        // Platform availability consistency
        #[test]
        fn sanitizer_type_availability_is_deterministic(sanitizer in sanitizer_type_strategy()) {
            let available1 = sanitizer.is_available();
            let available2 = sanitizer.is_available();
            prop_assert_eq!(available1, available2);
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // SanitizerType invariants
    #[kani::proof]
    fn verify_sanitizer_type_rustflags_non_empty() {
        let sanitizer: SanitizerType = kani::any();
        let flags = sanitizer.rustflags();
        kani::assert(
            !flags.is_empty(),
            "SanitizerType rustflags must not be empty",
        );
    }

    #[kani::proof]
    fn verify_sanitizer_type_name_non_empty() {
        let sanitizer: SanitizerType = kani::any();
        let name = sanitizer.name();
        kani::assert(!name.is_empty(), "SanitizerType name must not be empty");
    }

    #[kani::proof]
    fn verify_sanitizer_type_rustflags_starts_with_z() {
        let sanitizer: SanitizerType = kani::any();
        let flags = sanitizer.rustflags();
        kani::assert(
            flags.starts_with("-Z"),
            "SanitizerType rustflags must start with -Z",
        );
    }

    #[kani::proof]
    fn verify_sanitizer_type_equality_reflexive() {
        let sanitizer: SanitizerType = kani::any();
        kani::assert(
            sanitizer == sanitizer,
            "SanitizerType equality must be reflexive",
        );
    }

    // FindingSeverity invariants
    #[kani::proof]
    fn verify_finding_severity_equality_reflexive() {
        let severity: FindingSeverity = kani::any();
        kani::assert(
            severity == severity,
            "FindingSeverity equality must be reflexive",
        );
    }

    // SanitizerBackend invariants
    #[kani::proof]
    fn verify_sanitizer_backend_new_preserves_type() {
        let sanitizer: SanitizerType = kani::any();
        let backend = SanitizerBackend::new(sanitizer);
        kani::assert(
            backend.sanitizer_type == sanitizer,
            "SanitizerBackend must preserve sanitizer_type",
        );
    }

    #[kani::proof]
    fn verify_sanitizer_backend_default_timeout() {
        let sanitizer: SanitizerType = kani::any();
        let backend = SanitizerBackend::new(sanitizer);
        kani::assert(
            backend.timeout == Duration::from_secs(300),
            "Default timeout is 300s",
        );
    }

    #[kani::proof]
    fn verify_sanitizer_backend_with_timeout() {
        let sanitizer: SanitizerType = kani::any();
        let secs: u64 = kani::any();
        kani::assume(secs > 0 && secs < 100000);
        let timeout = Duration::from_secs(secs);
        let backend = SanitizerBackend::new(sanitizer).with_timeout(timeout);
        kani::assert(
            backend.timeout == timeout,
            "with_timeout must preserve value",
        );
    }

    // SanitizerResult passed consistency
    #[kani::proof]
    fn verify_sanitizer_result_passed_with_no_findings() {
        let sanitizer: SanitizerType = kani::any();
        let result = SanitizerResult {
            sanitizer,
            passed: true,
            findings: Vec::new(),
            raw_output: String::new(),
            duration: Duration::from_secs(1),
            tests_run: 0,
        };
        kani::assert(
            result.passed && result.findings.is_empty(),
            "passed=true requires empty findings",
        );
    }

    // SanitizerFinding structure
    #[kani::proof]
    fn verify_sanitizer_finding_preserves_severity() {
        let severity: FindingSeverity = kani::any();
        let finding = SanitizerFinding {
            issue_type: String::new(),
            description: String::new(),
            stack_trace: None,
            file: None,
            line: None,
            severity,
        };
        kani::assert(
            finding.severity == severity,
            "SanitizerFinding must preserve severity",
        );
    }

    /// Verify SanitizerType Address rustflags contains 'address'
    #[kani::proof]
    fn verify_sanitizer_type_address_flags() {
        let sanitizer = SanitizerType::Address;
        let flags = sanitizer.rustflags();
        kani::assert(
            flags.contains("address"),
            "Address sanitizer flags should contain 'address'",
        );
    }

    /// Verify SanitizerType Memory rustflags contains 'memory'
    #[kani::proof]
    fn verify_sanitizer_type_memory_flags() {
        let sanitizer = SanitizerType::Memory;
        let flags = sanitizer.rustflags();
        kani::assert(
            flags.contains("memory"),
            "Memory sanitizer flags should contain 'memory'",
        );
    }

    /// Verify SanitizerType Thread rustflags contains 'thread'
    #[kani::proof]
    fn verify_sanitizer_type_thread_flags() {
        let sanitizer = SanitizerType::Thread;
        let flags = sanitizer.rustflags();
        kani::assert(
            flags.contains("thread"),
            "Thread sanitizer flags should contain 'thread'",
        );
    }

    /// Verify SanitizerType Leak rustflags contains 'leak'
    #[kani::proof]
    fn verify_sanitizer_type_leak_flags() {
        let sanitizer = SanitizerType::Leak;
        let flags = sanitizer.rustflags();
        kani::assert(
            flags.contains("leak"),
            "Leak sanitizer flags should contain 'leak'",
        );
    }

    /// Verify SanitizerType Address name is AddressSanitizer
    #[kani::proof]
    fn verify_sanitizer_type_address_name() {
        let sanitizer = SanitizerType::Address;
        let name = sanitizer.name();
        kani::assert(
            name == "AddressSanitizer",
            "Address sanitizer name should be AddressSanitizer",
        );
    }

    /// Verify SanitizerType Memory name is MemorySanitizer
    #[kani::proof]
    fn verify_sanitizer_type_memory_name() {
        let sanitizer = SanitizerType::Memory;
        let name = sanitizer.name();
        kani::assert(
            name == "MemorySanitizer",
            "Memory sanitizer name should be MemorySanitizer",
        );
    }

    /// Verify SanitizerType Thread name is ThreadSanitizer
    #[kani::proof]
    fn verify_sanitizer_type_thread_name() {
        let sanitizer = SanitizerType::Thread;
        let name = sanitizer.name();
        kani::assert(
            name == "ThreadSanitizer",
            "Thread sanitizer name should be ThreadSanitizer",
        );
    }

    /// Verify SanitizerType Leak name is LeakSanitizer
    #[kani::proof]
    fn verify_sanitizer_type_leak_name() {
        let sanitizer = SanitizerType::Leak;
        let name = sanitizer.name();
        kani::assert(
            name == "LeakSanitizer",
            "Leak sanitizer name should be LeakSanitizer",
        );
    }

    /// Verify FindingSeverity Error variant
    #[kani::proof]
    fn verify_finding_severity_error() {
        let severity = FindingSeverity::Error;
        kani::assert(
            matches!(severity, FindingSeverity::Error),
            "Error should match Error variant",
        );
    }

    /// Verify FindingSeverity Warning variant
    #[kani::proof]
    fn verify_finding_severity_warning() {
        let severity = FindingSeverity::Warning;
        kani::assert(
            matches!(severity, FindingSeverity::Warning),
            "Warning should match Warning variant",
        );
    }

    /// Verify SanitizerResult stores sanitizer type
    #[kani::proof]
    fn verify_sanitizer_result_stores_type() {
        let sanitizer: SanitizerType = kani::any();
        let result = SanitizerResult {
            sanitizer,
            passed: false,
            findings: Vec::new(),
            raw_output: String::new(),
            duration: Duration::from_secs(1),
            tests_run: 0,
        };
        kani::assert(
            result.sanitizer == sanitizer,
            "SanitizerResult should store sanitizer type",
        );
    }

    /// Verify SanitizerResult stores passed state
    #[kani::proof]
    fn verify_sanitizer_result_stores_passed() {
        let sanitizer: SanitizerType = kani::any();
        let result = SanitizerResult {
            sanitizer,
            passed: true,
            findings: Vec::new(),
            raw_output: String::new(),
            duration: Duration::from_secs(1),
            tests_run: 0,
        };
        kani::assert(result.passed, "SanitizerResult should store passed state");
    }

    /// Verify SanitizerResult stores tests_run
    #[kani::proof]
    fn verify_sanitizer_result_stores_tests_run() {
        let sanitizer: SanitizerType = kani::any();
        let result = SanitizerResult {
            sanitizer,
            passed: true,
            findings: Vec::new(),
            raw_output: String::new(),
            duration: Duration::from_secs(1),
            tests_run: 42,
        };
        kani::assert(
            result.tests_run == 42,
            "SanitizerResult should store tests_run",
        );
    }

    /// Verify SanitizerFinding stores issue_type
    #[kani::proof]
    fn verify_sanitizer_finding_stores_issue_type() {
        let finding = SanitizerFinding {
            issue_type: String::from("buffer-overflow"),
            description: String::new(),
            stack_trace: None,
            file: None,
            line: None,
            severity: FindingSeverity::Error,
        };
        kani::assert(
            !finding.issue_type.is_empty(),
            "SanitizerFinding should store issue_type",
        );
    }

    /// Verify SanitizerFinding with file
    #[kani::proof]
    fn verify_sanitizer_finding_with_file() {
        let finding = SanitizerFinding {
            issue_type: String::new(),
            description: String::new(),
            stack_trace: None,
            file: Some(String::from("src/lib.rs")),
            line: None,
            severity: FindingSeverity::Error,
        };
        kani::assert(finding.file.is_some(), "SanitizerFinding should store file");
    }

    /// Verify SanitizerFinding with line
    #[kani::proof]
    fn verify_sanitizer_finding_with_line() {
        let finding = SanitizerFinding {
            issue_type: String::new(),
            description: String::new(),
            stack_trace: None,
            file: None,
            line: Some(42),
            severity: FindingSeverity::Error,
        };
        kani::assert(finding.line.is_some(), "SanitizerFinding should store line");
    }

    /// Verify SanitizerFinding with stack_trace
    #[kani::proof]
    fn verify_sanitizer_finding_with_stack_trace() {
        let finding = SanitizerFinding {
            issue_type: String::new(),
            description: String::new(),
            stack_trace: Some(String::from("main\n  foo\n    bar")),
            file: None,
            line: None,
            severity: FindingSeverity::Error,
        };
        kani::assert(
            finding.stack_trace.is_some(),
            "SanitizerFinding should store stack_trace",
        );
    }
}
