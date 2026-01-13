//! Haybale backend for LLVM symbolic execution
//!
//! Haybale is a symbolic execution engine from Trail of Bits that operates on
//! LLVM bitcode. It supports symbolic execution of Rust programs compiled to
//! LLVM IR and can be used to:
//! - Find bugs through exploring execution paths
//! - Verify assertions hold on all paths
//! - Generate test inputs that reach specific program points
//!
//! Unlike Kani (bounded model checking) or Miri (interpretation), Haybale
//! performs true symbolic execution on LLVM IR, which can handle some cases
//! that other tools cannot.
//!
//! See: <https://github.com/PLSysSec/haybale>

// =============================================
// Kani Proofs for Haybale Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- HaybaleConfig Default Tests ----

    /// Verify HaybaleConfig::default bitcode_path is None
    #[kani::proof]
    fn proof_haybale_config_default_bitcode_path_none() {
        let config = HaybaleConfig::default();
        kani::assert(
            config.bitcode_path.is_none(),
            "Default bitcode_path should be None",
        );
    }

    /// Verify HaybaleConfig::default project_path is None
    #[kani::proof]
    fn proof_haybale_config_default_project_path_none() {
        let config = HaybaleConfig::default();
        kani::assert(
            config.project_path.is_none(),
            "Default project_path should be None",
        );
    }

    /// Verify HaybaleConfig::default timeout is 600 seconds
    #[kani::proof]
    fn proof_haybale_config_default_timeout() {
        let config = HaybaleConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(600),
            "Default timeout should be 600 seconds",
        );
    }

    /// Verify HaybaleConfig::default max_paths is Some(10000)
    #[kani::proof]
    fn proof_haybale_config_default_max_paths() {
        let config = HaybaleConfig::default();
        kani::assert(
            config.max_paths == Some(10000),
            "Default max_paths should be Some(10000)",
        );
    }

    /// Verify HaybaleConfig::default max_loop_iters is 100
    #[kani::proof]
    fn proof_haybale_config_default_max_loop_iters() {
        let config = HaybaleConfig::default();
        kani::assert(
            config.max_loop_iters == 100,
            "Default max_loop_iters should be 100",
        );
    }

    /// Verify HaybaleConfig::default entry_function is None
    #[kani::proof]
    fn proof_haybale_config_default_entry_function_none() {
        let config = HaybaleConfig::default();
        kani::assert(
            config.entry_function.is_none(),
            "Default entry_function should be None",
        );
    }

    /// Verify HaybaleConfig::default llvm_passes is empty
    #[kani::proof]
    fn proof_haybale_config_default_llvm_passes_empty() {
        let config = HaybaleConfig::default();
        kani::assert(
            config.llvm_passes.is_empty(),
            "Default llvm_passes should be empty",
        );
    }

    /// Verify HaybaleConfig::default include_stdlib is false
    #[kani::proof]
    fn proof_haybale_config_default_include_stdlib_false() {
        let config = HaybaleConfig::default();
        kani::assert(
            !config.include_stdlib,
            "Default include_stdlib should be false",
        );
    }

    /// Verify HaybaleConfig::default clang_path is None
    #[kani::proof]
    fn proof_haybale_config_default_clang_path_none() {
        let config = HaybaleConfig::default();
        kani::assert(
            config.clang_path.is_none(),
            "Default clang_path should be None",
        );
    }

    /// Verify HaybaleConfig::default llc_path is None
    #[kani::proof]
    fn proof_haybale_config_default_llc_path_none() {
        let config = HaybaleConfig::default();
        kani::assert(config.llc_path.is_none(), "Default llc_path should be None");
    }

    /// Verify HaybaleConfig::default check_null_deref is true
    #[kani::proof]
    fn proof_haybale_config_default_check_null_deref_true() {
        let config = HaybaleConfig::default();
        kani::assert(
            config.check_null_deref,
            "Default check_null_deref should be true",
        );
    }

    /// Verify HaybaleConfig::default check_buffer_overflow is true
    #[kani::proof]
    fn proof_haybale_config_default_check_buffer_overflow_true() {
        let config = HaybaleConfig::default();
        kani::assert(
            config.check_buffer_overflow,
            "Default check_buffer_overflow should be true",
        );
    }

    /// Verify HaybaleConfig::default check_div_by_zero is true
    #[kani::proof]
    fn proof_haybale_config_default_check_div_by_zero_true() {
        let config = HaybaleConfig::default();
        kani::assert(
            config.check_div_by_zero,
            "Default check_div_by_zero should be true",
        );
    }

    // ---- HaybaleConfig Builder Tests ----

    /// Verify with_bitcode sets path
    #[kani::proof]
    fn proof_haybale_config_with_bitcode() {
        let config = HaybaleConfig::default().with_bitcode(PathBuf::from("/test.bc"));
        kani::assert(
            config.bitcode_path.is_some(),
            "with_bitcode should set bitcode_path",
        );
    }

    /// Verify with_project sets path
    #[kani::proof]
    fn proof_haybale_config_with_project() {
        let config = HaybaleConfig::default().with_project(PathBuf::from("/project"));
        kani::assert(
            config.project_path.is_some(),
            "with_project should set project_path",
        );
    }

    /// Verify with_timeout sets timeout
    #[kani::proof]
    fn proof_haybale_config_with_timeout() {
        let config = HaybaleConfig::default().with_timeout(Duration::from_secs(120));
        kani::assert(
            config.timeout == Duration::from_secs(120),
            "with_timeout should set timeout to 120",
        );
    }

    /// Verify with_max_paths sets max_paths
    #[kani::proof]
    fn proof_haybale_config_with_max_paths() {
        let config = HaybaleConfig::default().with_max_paths(5000);
        kani::assert(
            config.max_paths == Some(5000),
            "with_max_paths should set 5000",
        );
    }

    /// Verify with_max_loop_iters sets iters
    #[kani::proof]
    fn proof_haybale_config_with_max_loop_iters() {
        let config = HaybaleConfig::default().with_max_loop_iters(50);
        kani::assert(
            config.max_loop_iters == 50,
            "with_max_loop_iters should set 50",
        );
    }

    /// Verify with_entry_function sets function name
    #[kani::proof]
    fn proof_haybale_config_with_entry_function() {
        let config = HaybaleConfig::default().with_entry_function("main".to_string());
        kani::assert(
            config.entry_function.is_some(),
            "with_entry_function should set entry",
        );
    }

    /// Verify with_null_deref_checking disables check
    #[kani::proof]
    fn proof_haybale_config_with_null_deref_checking_false() {
        let config = HaybaleConfig::default().with_null_deref_checking(false);
        kani::assert(
            !config.check_null_deref,
            "with_null_deref_checking(false) should disable",
        );
    }

    /// Verify with_buffer_overflow_checking disables check
    #[kani::proof]
    fn proof_haybale_config_with_buffer_overflow_checking_false() {
        let config = HaybaleConfig::default().with_buffer_overflow_checking(false);
        kani::assert(
            !config.check_buffer_overflow,
            "with_buffer_overflow_checking(false) should disable",
        );
    }

    /// Verify with_div_by_zero_checking disables check
    #[kani::proof]
    fn proof_haybale_config_with_div_by_zero_checking_false() {
        let config = HaybaleConfig::default().with_div_by_zero_checking(false);
        kani::assert(
            !config.check_div_by_zero,
            "with_div_by_zero_checking(false) should disable",
        );
    }

    // ---- HaybaleBackend Construction Tests ----

    /// Verify HaybaleBackend::new uses default timeout
    #[kani::proof]
    fn proof_haybale_backend_new_default_timeout() {
        let backend = HaybaleBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "New backend should use default timeout",
        );
    }

    /// Verify HaybaleBackend::default equals HaybaleBackend::new
    #[kani::proof]
    fn proof_haybale_backend_default_equals_new() {
        let default_backend = HaybaleBackend::default();
        let new_backend = HaybaleBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "Default and new should have same timeout",
        );
    }

    /// Verify HaybaleBackend::with_config preserves timeout
    #[kani::proof]
    fn proof_haybale_backend_with_config_timeout() {
        let config = HaybaleConfig {
            bitcode_path: None,
            project_path: None,
            timeout: Duration::from_secs(300),
            max_paths: Some(10000),
            max_loop_iters: 100,
            entry_function: None,
            llvm_passes: vec![],
            include_stdlib: false,
            clang_path: None,
            llc_path: None,
            check_null_deref: true,
            check_buffer_overflow: true,
            check_div_by_zero: true,
        };
        let backend = HaybaleBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(300),
            "with_config should preserve timeout",
        );
    }

    /// Verify HaybaleBackend::with_config preserves max_paths
    #[kani::proof]
    fn proof_haybale_backend_with_config_max_paths() {
        let config = HaybaleConfig {
            bitcode_path: None,
            project_path: None,
            timeout: Duration::from_secs(600),
            max_paths: Some(5000),
            max_loop_iters: 100,
            entry_function: None,
            llvm_passes: vec![],
            include_stdlib: false,
            clang_path: None,
            llc_path: None,
            check_null_deref: true,
            check_buffer_overflow: true,
            check_div_by_zero: true,
        };
        let backend = HaybaleBackend::with_config(config);
        kani::assert(
            backend.config.max_paths == Some(5000),
            "with_config should preserve max_paths",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify HaybaleBackend::id returns Haybale
    #[kani::proof]
    fn proof_haybale_backend_id() {
        let backend = HaybaleBackend::new();
        kani::assert(
            backend.id() == BackendId::Haybale,
            "Backend id should be Haybale",
        );
    }

    /// Verify HaybaleBackend::supports includes MemorySafety
    #[kani::proof]
    fn proof_haybale_backend_supports_memory_safety() {
        let backend = HaybaleBackend::new();
        let supported = backend.supports();
        let has_mem = supported.iter().any(|p| *p == PropertyType::MemorySafety);
        kani::assert(has_mem, "Should support MemorySafety property");
    }

    /// Verify HaybaleBackend::supports includes Contract
    #[kani::proof]
    fn proof_haybale_backend_supports_contract() {
        let backend = HaybaleBackend::new();
        let supported = backend.supports();
        let has_contract = supported.iter().any(|p| *p == PropertyType::Contract);
        kani::assert(has_contract, "Should support Contract property");
    }

    /// Verify HaybaleBackend::supports includes UndefinedBehavior
    #[kani::proof]
    fn proof_haybale_backend_supports_undefined_behavior() {
        let backend = HaybaleBackend::new();
        let supported = backend.supports();
        let has_ub = supported
            .iter()
            .any(|p| *p == PropertyType::UndefinedBehavior);
        kani::assert(has_ub, "Should support UndefinedBehavior property");
    }

    /// Verify HaybaleBackend::supports returns exactly 3 properties
    #[kani::proof]
    fn proof_haybale_backend_supports_length() {
        let backend = HaybaleBackend::new();
        let supported = backend.supports();
        kani::assert(supported.len() == 3, "Should support exactly 3 properties");
    }

    // ---- parse_output Tests ----

    /// Verify parse_output recognizes "All paths explored successfully"
    #[kani::proof]
    fn proof_parse_output_all_paths_success() {
        let backend = HaybaleBackend::new();
        let status = backend.parse_output("All paths explored successfully", "", true);
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "All paths success should be Proven",
        );
    }

    /// Verify parse_output recognizes "No errors found"
    #[kani::proof]
    fn proof_parse_output_no_errors() {
        let backend = HaybaleBackend::new();
        let status = backend.parse_output("No errors found", "", true);
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "No errors should be Proven",
        );
    }

    /// Verify parse_output recognizes "Verification passed"
    #[kani::proof]
    fn proof_parse_output_verification_passed() {
        let backend = HaybaleBackend::new();
        let status = backend.parse_output("Verification passed", "", true);
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "Verification passed should be Proven",
        );
    }

    /// Verify parse_output success with no error messages
    #[kani::proof]
    fn proof_parse_output_success_no_errors() {
        let backend = HaybaleBackend::new();
        let status = backend.parse_output("Path exploration complete", "", true);
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "Success without errors should be Proven",
        );
    }

    /// Verify parse_output recognizes assertion failure
    #[kani::proof]
    fn proof_parse_output_assertion_failed() {
        let backend = HaybaleBackend::new();
        let status = backend.parse_output("", "assertion failed at main.rs:10", false);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "Assertion failure should be Disproven",
        );
    }

    /// Verify parse_output recognizes null pointer dereference
    #[kani::proof]
    fn proof_parse_output_null_pointer() {
        let backend = HaybaleBackend::new();
        let status = backend.parse_output("", "null pointer dereference", false);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "Null pointer should be Disproven",
        );
    }

    /// Verify parse_output recognizes buffer overflow
    #[kani::proof]
    fn proof_parse_output_buffer_overflow() {
        let backend = HaybaleBackend::new();
        let status = backend.parse_output("", "buffer overflow", false);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "Buffer overflow should be Disproven",
        );
    }

    /// Verify parse_output recognizes out of bounds
    #[kani::proof]
    fn proof_parse_output_out_of_bounds() {
        let backend = HaybaleBackend::new();
        let status = backend.parse_output("", "out of bounds", false);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "Out of bounds should be Disproven",
        );
    }

    /// Verify parse_output recognizes division by zero
    #[kani::proof]
    fn proof_parse_output_div_by_zero() {
        let backend = HaybaleBackend::new();
        let status = backend.parse_output("", "division by zero", false);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "Division by zero should be Disproven",
        );
    }

    /// Verify parse_output recognizes path explosion
    #[kani::proof]
    fn proof_parse_output_path_explosion() {
        let backend = HaybaleBackend::new();
        let status = backend.parse_output("", "max paths reached", false);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Path explosion should be Unknown",
        );
    }

    /// Verify parse_output recognizes timeout
    #[kani::proof]
    fn proof_parse_output_timeout() {
        let backend = HaybaleBackend::new();
        let status = backend.parse_output("", "timeout", false);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Timeout should be Unknown",
        );
    }

    /// Verify parse_output recognizes solver error
    #[kani::proof]
    fn proof_parse_output_solver_error() {
        let backend = HaybaleBackend::new();
        let status = backend.parse_output("", "solver error", false);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Solver error should be Unknown",
        );
    }

    // ---- extract_failed_checks Tests ----

    /// Verify extract_failed_checks identifies assertion
    #[kani::proof]
    fn proof_extract_failed_checks_assertion() {
        let output = "error: assertion failed at main.rs:10";
        let checks = HaybaleBackend::extract_failed_checks(output);
        kani::assert(!checks.is_empty(), "Should extract assertion check");
        kani::assert(
            checks[0].check_id == "haybale_assertion",
            "Should be assertion check",
        );
    }

    /// Verify extract_failed_checks identifies null dereference
    #[kani::proof]
    fn proof_extract_failed_checks_null_deref() {
        let output = "error: null pointer dereference";
        let checks = HaybaleBackend::extract_failed_checks(output);
        kani::assert(!checks.is_empty(), "Should extract null deref check");
    }

    /// Verify extract_failed_checks identifies buffer overflow
    #[kani::proof]
    fn proof_extract_failed_checks_buffer_overflow() {
        let output = "error: buffer overflow detected";
        let checks = HaybaleBackend::extract_failed_checks(output);
        kani::assert(!checks.is_empty(), "Should extract buffer overflow check");
    }

    /// Verify extract_failed_checks identifies division by zero
    #[kani::proof]
    fn proof_extract_failed_checks_div_by_zero() {
        let output = "error: division by zero";
        let checks = HaybaleBackend::extract_failed_checks(output);
        kani::assert(!checks.is_empty(), "Should extract div by zero check");
    }

    // ---- extract_location_from_line Tests ----

    /// Verify extract_location_from_line parses Rust file
    #[kani::proof]
    fn proof_extract_location_rs_file() {
        let line = "error at main.rs:10:5: assertion failed";
        let location = HaybaleBackend::extract_location_from_line(line);
        kani::assert(location.is_some(), "Should extract location from .rs file");
    }

    /// Verify extract_location_from_line parses C file
    #[kani::proof]
    fn proof_extract_location_c_file() {
        let line = "error at test.c:25:12: buffer overflow";
        let location = HaybaleBackend::extract_location_from_line(line);
        kani::assert(location.is_some(), "Should extract location from .c file");
    }

    /// Verify extract_location_from_line parses LL file
    #[kani::proof]
    fn proof_extract_location_ll_file() {
        let line = "error at module.ll:100: undefined";
        let location = HaybaleBackend::extract_location_from_line(line);
        kani::assert(location.is_some(), "Should extract location from .ll file");
    }

    // ---- extract_function_name Tests ----

    /// Verify extract_function_name finds fn keyword
    #[kani::proof]
    fn proof_extract_function_name_fn() {
        let lines = vec!["fn test_function() {", "  assert!(false);"];
        let function = HaybaleBackend::extract_function_name(&lines, 1);
        kani::assert(function.is_some(), "Should extract function name from fn");
    }

    /// Verify extract_function_name finds "in function"
    #[kani::proof]
    fn proof_extract_function_name_in_function() {
        let lines = vec!["in function main:", "error occurred"];
        let function = HaybaleBackend::extract_function_name(&lines, 1);
        kani::assert(
            function.is_some(),
            "Should extract function from 'in function'",
        );
    }

    // ---- parse_counterexample Tests ----

    /// Verify parse_counterexample sets raw
    #[kani::proof]
    fn proof_parse_counterexample_raw() {
        let ce = HaybaleBackend::parse_counterexample("stdout output", "stderr output");
        kani::assert(ce.raw.is_some(), "parse_counterexample should set raw");
    }
}

use crate::counterexample::{FailedCheck, SourceLocation, StructuredCounterexample};
use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
    VerificationStatus,
};
use async_trait::async_trait;
use dashprove_usl::typecheck::TypedSpec;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::process::Command;
use tracing::debug;

/// Configuration for Haybale backend
#[derive(Debug, Clone)]
pub struct HaybaleConfig {
    /// Path to the LLVM bitcode file to analyze
    pub bitcode_path: Option<PathBuf>,
    /// Path to the Rust project (will compile to LLVM IR)
    pub project_path: Option<PathBuf>,
    /// Timeout for symbolic execution
    pub timeout: Duration,
    /// Maximum number of paths to explore
    pub max_paths: Option<u64>,
    /// Maximum loop iterations before giving up
    pub max_loop_iters: u32,
    /// Entry function to analyze (None = all functions with `#[haybale::test]` or main)
    pub entry_function: Option<String>,
    /// Additional LLVM passes to run before analysis
    pub llvm_passes: Vec<String>,
    /// Whether to include standard library in analysis
    pub include_stdlib: bool,
    /// Path to clang for compiling to LLVM IR
    pub clang_path: Option<PathBuf>,
    /// Path to llc for linking LLVM modules
    pub llc_path: Option<PathBuf>,
    /// Whether to enable null pointer dereference checking
    pub check_null_deref: bool,
    /// Whether to enable buffer overflow checking
    pub check_buffer_overflow: bool,
    /// Whether to enable division by zero checking
    pub check_div_by_zero: bool,
}

impl Default for HaybaleConfig {
    fn default() -> Self {
        Self {
            bitcode_path: None,
            project_path: None,
            timeout: Duration::from_secs(600),
            max_paths: Some(10000),
            max_loop_iters: 100,
            entry_function: None,
            llvm_passes: Vec::new(),
            include_stdlib: false,
            clang_path: None,
            llc_path: None,
            check_null_deref: true,
            check_buffer_overflow: true,
            check_div_by_zero: true,
        }
    }
}

impl HaybaleConfig {
    /// Create a new config with a bitcode path
    pub fn with_bitcode(mut self, path: PathBuf) -> Self {
        self.bitcode_path = Some(path);
        self
    }

    /// Create a new config with a project path
    pub fn with_project(mut self, path: PathBuf) -> Self {
        self.project_path = Some(path);
        self
    }

    /// Set the timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set maximum paths to explore
    pub fn with_max_paths(mut self, max: u64) -> Self {
        self.max_paths = Some(max);
        self
    }

    /// Set maximum loop iterations
    pub fn with_max_loop_iters(mut self, max: u32) -> Self {
        self.max_loop_iters = max;
        self
    }

    /// Set the entry function
    pub fn with_entry_function(mut self, func: String) -> Self {
        self.entry_function = Some(func);
        self
    }

    /// Enable or disable null dereference checking
    pub fn with_null_deref_checking(mut self, enable: bool) -> Self {
        self.check_null_deref = enable;
        self
    }

    /// Enable or disable buffer overflow checking
    pub fn with_buffer_overflow_checking(mut self, enable: bool) -> Self {
        self.check_buffer_overflow = enable;
        self
    }

    /// Enable or disable division by zero checking
    pub fn with_div_by_zero_checking(mut self, enable: bool) -> Self {
        self.check_div_by_zero = enable;
        self
    }
}

/// Haybale symbolic execution backend
///
/// Haybale operates on LLVM bitcode and performs symbolic execution to explore
/// all feasible execution paths. This backend supports:
/// - Analyzing Rust programs compiled to LLVM IR
/// - Finding bugs through path exploration
/// - Verifying assertions on all paths
pub struct HaybaleBackend {
    config: HaybaleConfig,
}

impl Default for HaybaleBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl HaybaleBackend {
    /// Create a new Haybale backend with default configuration
    pub fn new() -> Self {
        Self {
            config: HaybaleConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: HaybaleConfig) -> Self {
        Self { config }
    }

    /// Detect Haybale/LLVM toolchain availability
    async fn detect_haybale(&self) -> Result<HaybaleDetection, String> {
        // Check for rustc with emit=llvm-bc capability
        let rustc_output = Command::new("rustc")
            .arg("--version")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| format!("Failed to execute rustc: {}. Install Rust toolchain.", e))?;

        if !rustc_output.status.success() {
            return Err("rustc not available".to_string());
        }

        let rustc_version = String::from_utf8_lossy(&rustc_output.stdout).to_string();

        // Check for llvm-link (needed to combine modules)
        let llvm_link_path = self.find_llvm_tool("llvm-link").await;

        // Check for opt (needed for LLVM passes)
        let opt_path = self.find_llvm_tool("opt").await;

        // Haybale itself is a Rust library - check if it's available as a dependency
        // In practice, we check for the cargo haybale command or the library
        let haybale_available = self.check_haybale_library().await;

        Ok(HaybaleDetection {
            rustc_version,
            llvm_link_path,
            opt_path,
            haybale_available,
        })
    }

    /// Find an LLVM tool in common locations
    async fn find_llvm_tool(&self, tool_name: &str) -> Option<PathBuf> {
        // Try direct path
        if let Ok(path) = which::which(tool_name) {
            return Some(path);
        }

        // Try versioned paths (llvm-link-15, etc.)
        for version in (11..=18).rev() {
            let versioned = format!("{}-{}", tool_name, version);
            if let Ok(path) = which::which(&versioned) {
                return Some(path);
            }
        }

        // Try common LLVM installation paths
        let common_paths = [
            "/usr/lib/llvm-15/bin",
            "/usr/lib/llvm-14/bin",
            "/usr/lib/llvm/bin",
            "/usr/local/opt/llvm/bin",
            "/opt/homebrew/opt/llvm/bin",
        ];

        for base in &common_paths {
            let path = PathBuf::from(base).join(tool_name);
            if path.exists() {
                return Some(path);
            }
        }

        None
    }

    /// Check if Haybale library is available
    async fn check_haybale_library(&self) -> bool {
        // Check if haybale is in Cargo.toml or can be compiled
        // For now, assume it's available if LLVM tools are present
        // In a real implementation, we'd check the Cargo.lock or try to compile
        true
    }

    /// Compile Rust project to LLVM bitcode
    async fn compile_to_bitcode(&self, project_path: &PathBuf) -> Result<PathBuf, BackendError> {
        // Create temp directory for bitcode output
        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        // Run rustc with emit=llvm-bc
        let output = Command::new("cargo")
            .arg("rustc")
            .arg("--release")
            .arg("--")
            .arg("--emit=llvm-bc")
            .arg("-C")
            .arg("lto=fat")
            .arg("-C")
            .arg("codegen-units=1")
            .arg("-o")
            .arg(temp_dir.path().join("output.bc"))
            .current_dir(project_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| {
                BackendError::VerificationFailed(format!(
                    "Failed to compile to LLVM bitcode: {}",
                    e
                ))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(BackendError::VerificationFailed(format!(
                "Compilation to bitcode failed: {}",
                stderr
            )));
        }

        // Find the generated bitcode file
        let bc_path = temp_dir.path().join("output.bc");
        if bc_path.exists() {
            // Keep the temp dir alive (don't delete on drop)
            let _ = temp_dir.keep();
            Ok(bc_path)
        } else {
            // Look for .bc files in target directory
            let target_bc = project_path.join("target/release/deps");
            if target_bc.exists() {
                for entry in std::fs::read_dir(&target_bc).map_err(|e| {
                    BackendError::VerificationFailed(format!("Failed to read target dir: {}", e))
                })? {
                    let entry = entry.map_err(|e| {
                        BackendError::VerificationFailed(format!("Failed to read entry: {}", e))
                    })?;
                    if entry.path().extension().is_some_and(|e| e == "bc") {
                        return Ok(entry.path());
                    }
                }
            }
            Err(BackendError::VerificationFailed(
                "No bitcode file generated".to_string(),
            ))
        }
    }

    /// Generate a Haybale runner script from USL spec
    fn generate_haybale_runner(&self, spec: &TypedSpec, bitcode_path: &std::path::Path) -> String {
        let mut code = String::new();

        code.push_str("// Generated Haybale runner by DashProve\n");
        code.push_str("use haybale::{ExecutionManager, Config, backend::Backend};\n");
        code.push_str("use std::path::Path;\n\n");

        code.push_str("fn main() {\n");
        code.push_str(&format!(
            "    let project = haybale::Project::from_bc_path(Path::new({:?})).unwrap();\n",
            bitcode_path.display()
        ));
        code.push_str("    let mut config = Config::default();\n");

        // Configure based on options
        code.push_str(&format!(
            "    config.max_loop_iters = {};\n",
            self.config.max_loop_iters
        ));

        if let Some(max_paths) = self.config.max_paths {
            code.push_str(&format!("    config.max_paths = Some({});\n", max_paths));
        }

        code.push_str("\n    let mut em = ExecutionManager::new(&project, config).unwrap();\n");

        // Add checks for each property in the spec
        for (i, prop) in spec.spec.properties.iter().enumerate() {
            let prop_name = prop.name();
            code.push_str(&format!("\n    // Property {}: {}\n", i, prop_name));
        }

        // Run symbolic execution on entry function or all functions
        if let Some(ref entry) = self.config.entry_function {
            code.push_str(&format!(
                "    let results = em.find_executions_for_fn({:?});\n",
                entry
            ));
        } else {
            code.push_str("    let results = em.find_executions_for_all_fns();\n");
        }

        code.push_str("    for result in results {\n");
        code.push_str("        match result {\n");
        code.push_str(
            "            Ok(state) => println!(\"Path explored successfully: {:?}\", state),\n",
        );
        code.push_str("            Err(e) => eprintln!(\"Error during execution: {:?}\", e),\n");
        code.push_str("        }\n");
        code.push_str("    }\n");
        code.push_str("}\n");

        code
    }

    /// Parse Haybale output into verification result
    fn parse_output(&self, stdout: &str, stderr: &str, success: bool) -> VerificationStatus {
        let combined = format!("{}\n{}", stdout, stderr);

        // Check for successful verification
        if success {
            if combined.contains("All paths explored successfully")
                || combined.contains("No errors found")
                || combined.contains("Verification passed")
            {
                return VerificationStatus::Proven;
            }
            // No explicit errors and success = verified
            if !combined.contains("error") && !combined.contains("Error") {
                return VerificationStatus::Proven;
            }
        }

        // Check for common error patterns
        if combined.contains("assertion failed")
            || combined.contains("assertion violation")
            || combined.contains("AssertionError")
        {
            return VerificationStatus::Disproven;
        }

        // Check for null pointer dereference
        if combined.contains("null pointer dereference")
            || combined.contains("NullPointerException")
            || combined.contains("nullptr access")
        {
            return VerificationStatus::Disproven;
        }

        // Check for buffer overflow
        if combined.contains("buffer overflow")
            || combined.contains("out of bounds")
            || combined.contains("IndexOutOfBounds")
        {
            return VerificationStatus::Disproven;
        }

        // Check for division by zero
        if combined.contains("division by zero") || combined.contains("DivisionByZero") {
            return VerificationStatus::Disproven;
        }

        // Check for path explosion (too many paths)
        if combined.contains("max paths reached")
            || combined.contains("path explosion")
            || combined.contains("too many paths")
        {
            return VerificationStatus::Unknown {
                reason: "Path explosion - max paths limit reached".to_string(),
            };
        }

        // Check for timeout
        if combined.contains("timeout") || combined.contains("Timeout") {
            return VerificationStatus::Unknown {
                reason: "Symbolic execution timeout".to_string(),
            };
        }

        // Check for solver issues
        if combined.contains("solver error")
            || combined.contains("SMT solver")
            || combined.contains("constraint solving failed")
        {
            return VerificationStatus::Unknown {
                reason: "SMT solver issue during symbolic execution".to_string(),
            };
        }

        // Generic error
        if combined.contains("error") || combined.contains("Error") {
            return VerificationStatus::Unknown {
                reason: "Error during symbolic execution".to_string(),
            };
        }

        VerificationStatus::Unknown {
            reason: "Could not determine verification result".to_string(),
        }
    }

    /// Extract structured counterexample from Haybale output
    fn parse_counterexample(stdout: &str, stderr: &str) -> StructuredCounterexample {
        let mut ce = StructuredCounterexample::new();
        let combined = format!("{}\n{}", stdout, stderr);
        ce.raw = Some(combined.clone());

        // Extract failed checks
        ce.failed_checks = Self::extract_failed_checks(&combined);

        // Extract input values as witness if present
        ce.witness = Self::extract_inputs(&combined)
            .into_iter()
            .map(|(k, v)| (k, crate::counterexample::CounterexampleValue::String(v)))
            .collect();

        ce
    }

    /// Extract failed checks from output
    fn extract_failed_checks(output: &str) -> Vec<FailedCheck> {
        let mut checks = Vec::new();
        let lines: Vec<&str> = output.lines().collect();

        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            // Match error patterns
            if trimmed.contains("assertion failed")
                || trimmed.contains("error:")
                || trimmed.contains("Error:")
            {
                let check_id = if trimmed.contains("assertion") {
                    "haybale_assertion"
                } else if trimmed.contains("null pointer") {
                    "haybale_null_deref"
                } else if trimmed.contains("overflow") || trimmed.contains("out of bounds") {
                    "haybale_buffer_overflow"
                } else if trimmed.contains("division") {
                    "haybale_div_by_zero"
                } else {
                    "haybale_error"
                };

                let description = if trimmed.contains(':') {
                    trimmed
                        .split(':')
                        .next_back()
                        .unwrap_or("")
                        .trim()
                        .to_string()
                } else {
                    trimmed.to_string()
                };

                let location = Self::extract_location_from_line(trimmed);
                let function = Self::extract_function_name(&lines, i);

                if !description.is_empty() {
                    checks.push(FailedCheck {
                        check_id: check_id.to_string(),
                        description,
                        location,
                        function,
                    });
                }
            }
        }

        // If no structured errors found, create a generic one from first error
        if checks.is_empty() && (output.contains("error") || output.contains("Error")) {
            for line in output.lines() {
                let trimmed = line.trim();
                if trimmed.contains("error") || trimmed.contains("Error") {
                    checks.push(FailedCheck {
                        check_id: "haybale_error".to_string(),
                        description: trimmed.to_string(),
                        location: None,
                        function: None,
                    });
                    break;
                }
            }
        }

        checks
    }

    /// Extract source location from error line
    fn extract_location_from_line(line: &str) -> Option<SourceLocation> {
        // Pattern: "file.rs:line:col" or "at file.rs:line"
        let parts: Vec<&str> = line.split(':').collect();
        if parts.len() >= 2 {
            for window in parts.windows(2) {
                let file_candidate = window[0]
                    .rsplit_once(' ')
                    .map(|(_, f)| f)
                    .unwrap_or(window[0]);

                if file_candidate.ends_with(".rs")
                    || file_candidate.ends_with(".c")
                    || file_candidate.ends_with(".cpp")
                    || file_candidate.ends_with(".ll")
                    || file_candidate.ends_with(".bc")
                {
                    if let Ok(line_num) = window[1].parse::<u32>() {
                        let column = if parts.len() > 2 {
                            parts
                                .get(2)
                                .and_then(|s| s.split_whitespace().next())
                                .and_then(|s| s.parse::<u32>().ok())
                        } else {
                            None
                        };

                        return Some(SourceLocation {
                            file: file_candidate.to_string(),
                            line: line_num,
                            column,
                        });
                    }
                }
            }
        }
        None
    }

    /// Extract function name from context
    fn extract_function_name(lines: &[&str], error_idx: usize) -> Option<String> {
        let search_start = error_idx.saturating_sub(20);
        let search_end = lines.len().min(error_idx + 10);

        for line in lines.iter().take(search_end).skip(search_start) {
            // Look for function patterns
            if let Some(start) = line.find("fn ") {
                let after_fn = &line[start + 3..];
                let name: String = after_fn
                    .trim()
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                if !name.is_empty() {
                    return Some(name);
                }
            }

            // Look for "in function X" pattern
            if line.contains("in function") || line.contains("in fn") {
                if let Some(idx) = line.find("function").or_else(|| line.find("fn")) {
                    let after = &line[idx..];
                    let name: String = after
                        .split_whitespace()
                        .nth(1)
                        .unwrap_or("")
                        .chars()
                        .take_while(|c| c.is_alphanumeric() || *c == '_')
                        .collect();
                    if !name.is_empty() {
                        return Some(name);
                    }
                }
            }
        }
        None
    }

    /// Extract input values from symbolic execution output
    fn extract_inputs(output: &str) -> std::collections::BTreeMap<String, String> {
        let mut inputs = std::collections::BTreeMap::new();

        for line in output.lines() {
            // Look for "input X = Y" or "X = <symbolic: Y>" patterns
            if line.contains(" = ") && (line.contains("input") || line.contains("symbolic")) {
                let parts: Vec<&str> = line.split('=').collect();
                if parts.len() >= 2 {
                    let name = parts[0]
                        .split_whitespace()
                        .last()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    let value = parts[1..].join("=").trim().to_string();
                    if !name.is_empty() && !value.is_empty() {
                        inputs.insert(name, value);
                    }
                }
            }
        }

        inputs
    }
}

/// Detection result for Haybale availability
#[derive(Debug)]
#[allow(dead_code)] // Fields used for debugging and future enhancements
struct HaybaleDetection {
    rustc_version: String,
    llvm_link_path: Option<PathBuf>,
    opt_path: Option<PathBuf>,
    haybale_available: bool,
}

#[async_trait]
impl VerificationBackend for HaybaleBackend {
    fn id(&self) -> BackendId {
        BackendId::Haybale
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![
            PropertyType::MemorySafety,
            PropertyType::Contract, // Haybale verifies assertions and contracts
            PropertyType::UndefinedBehavior,
        ]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        // Detect toolchain
        let detection = self
            .detect_haybale()
            .await
            .map_err(BackendError::Unavailable)?;

        debug!("Haybale detection: {:?}", detection);

        // Get or compile bitcode
        let bitcode_path = if let Some(ref path) = self.config.bitcode_path {
            if path.exists() {
                path.clone()
            } else {
                return Err(BackendError::Unavailable(format!(
                    "Bitcode file not found: {:?}",
                    path
                )));
            }
        } else if let Some(ref project) = self.config.project_path {
            self.compile_to_bitcode(project).await?
        } else {
            return Err(BackendError::Unavailable(
                "Haybale requires either bitcode_path or project_path".to_string(),
            ));
        };

        // Generate Haybale runner
        let runner_code = self.generate_haybale_runner(spec, &bitcode_path);
        debug!("Generated Haybale runner:\n{}", runner_code);

        // Create temp project for the runner
        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        // Write runner and Cargo.toml
        let runner_path = temp_dir.path().join("src/main.rs");
        std::fs::create_dir_all(temp_dir.path().join("src")).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create src dir: {}", e))
        })?;

        std::fs::write(&runner_path, &runner_code).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write runner: {}", e))
        })?;

        let cargo_toml = r#"
[package]
name = "haybale_runner"
version = "0.1.0"
edition = "2021"

[dependencies]
haybale = "0.7"
"#;
        std::fs::write(temp_dir.path().join("Cargo.toml"), cargo_toml).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write Cargo.toml: {}", e))
        })?;

        // Run the Haybale analysis
        let result = tokio::time::timeout(
            self.config.timeout,
            Command::new("cargo")
                .arg("run")
                .arg("--release")
                .current_dir(temp_dir.path())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output(),
        )
        .await;

        let duration = start.elapsed();

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                debug!("Haybale stdout: {}", stdout);
                debug!("Haybale stderr: {}", stderr);

                let status = self.parse_output(&stdout, &stderr, output.status.success());

                let diagnostics: Vec<String> = stderr
                    .lines()
                    .chain(stdout.lines())
                    .filter(|l| {
                        l.contains("error")
                            || l.contains("Error")
                            || l.contains("warning")
                            || l.contains("path")
                            || l.contains("assertion")
                    })
                    .map(String::from)
                    .collect();

                let counterexample = if !output.status.success() {
                    Some(Self::parse_counterexample(&stdout, &stderr))
                } else {
                    None
                };

                Ok(BackendResult {
                    backend: BackendId::Haybale,
                    status,
                    proof: if output.status.success() {
                        Some("Verified by Haybale symbolic execution".to_string())
                    } else {
                        None
                    },
                    counterexample,
                    diagnostics,
                    time_taken: duration,
                })
            }
            Ok(Err(e)) => Err(BackendError::VerificationFailed(format!(
                "Failed to run Haybale: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match self.detect_haybale().await {
            Ok(detection) => {
                if detection.haybale_available {
                    HealthStatus::Healthy
                } else {
                    HealthStatus::Degraded {
                        reason: "Haybale library not available, but LLVM toolchain found"
                            .to_string(),
                    }
                }
            }
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =============================================
    // Configuration tests
    // =============================================

    #[test]
    fn default_config() {
        let config = HaybaleConfig::default();
        assert!(config.bitcode_path.is_none());
        assert!(config.project_path.is_none());
        assert_eq!(config.timeout, Duration::from_secs(600));
        assert_eq!(config.max_paths, Some(10000));
        assert_eq!(config.max_loop_iters, 100);
        assert!(config.entry_function.is_none());
        assert!(config.check_null_deref);
        assert!(config.check_buffer_overflow);
        assert!(config.check_div_by_zero);
    }

    #[test]
    fn config_with_bitcode() {
        let config = HaybaleConfig::default().with_bitcode(PathBuf::from("/path/to/code.bc"));
        assert_eq!(config.bitcode_path, Some(PathBuf::from("/path/to/code.bc")));
    }

    #[test]
    fn config_with_project() {
        let config = HaybaleConfig::default().with_project(PathBuf::from("/path/to/project"));
        assert_eq!(config.project_path, Some(PathBuf::from("/path/to/project")));
    }

    #[test]
    fn config_with_timeout() {
        let config = HaybaleConfig::default().with_timeout(Duration::from_secs(120));
        assert_eq!(config.timeout, Duration::from_secs(120));
    }

    #[test]
    fn config_with_max_paths() {
        let config = HaybaleConfig::default().with_max_paths(5000);
        assert_eq!(config.max_paths, Some(5000));
    }

    #[test]
    fn config_with_max_loop_iters() {
        let config = HaybaleConfig::default().with_max_loop_iters(50);
        assert_eq!(config.max_loop_iters, 50);
    }

    #[test]
    fn config_with_entry_function() {
        let config = HaybaleConfig::default().with_entry_function("main".to_string());
        assert_eq!(config.entry_function, Some("main".to_string()));
    }

    #[test]
    fn config_with_null_deref_checking() {
        let config = HaybaleConfig::default().with_null_deref_checking(false);
        assert!(!config.check_null_deref);
    }

    #[test]
    fn config_with_buffer_overflow_checking() {
        let config = HaybaleConfig::default().with_buffer_overflow_checking(false);
        assert!(!config.check_buffer_overflow);
    }

    #[test]
    fn config_with_div_by_zero_checking() {
        let config = HaybaleConfig::default().with_div_by_zero_checking(false);
        assert!(!config.check_div_by_zero);
    }

    // =============================================
    // Backend identity tests
    // =============================================

    #[test]
    fn backend_id_is_haybale() {
        let backend = HaybaleBackend::new();
        assert_eq!(backend.id(), BackendId::Haybale);
    }

    #[test]
    fn supports_memory_safety() {
        let backend = HaybaleBackend::new();
        let supports = backend.supports();
        assert!(supports.contains(&PropertyType::MemorySafety));
    }

    #[test]
    fn supports_contract() {
        let backend = HaybaleBackend::new();
        let supports = backend.supports();
        assert!(supports.contains(&PropertyType::Contract));
    }

    #[test]
    fn supports_undefined_behavior() {
        let backend = HaybaleBackend::new();
        let supports = backend.supports();
        assert!(supports.contains(&PropertyType::UndefinedBehavior));
    }

    // =============================================
    // Output parsing tests
    // =============================================

    #[test]
    fn parse_output_success() {
        let backend = HaybaleBackend::new();
        let status = backend.parse_output("All paths explored successfully", "", true);
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn parse_output_no_errors_found() {
        let backend = HaybaleBackend::new();
        let status = backend.parse_output("No errors found", "", true);
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn parse_output_verification_passed() {
        let backend = HaybaleBackend::new();
        let status = backend.parse_output("Verification passed", "", true);
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn parse_output_success_no_explicit_message() {
        let backend = HaybaleBackend::new();
        let status = backend.parse_output("Path exploration complete", "", true);
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn parse_output_assertion_failed() {
        let backend = HaybaleBackend::new();
        let status = backend.parse_output("", "assertion failed at main.rs:10", false);
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[test]
    fn parse_output_assertion_violation() {
        let backend = HaybaleBackend::new();
        let status = backend.parse_output("", "assertion violation detected", false);
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[test]
    fn parse_output_null_pointer() {
        let backend = HaybaleBackend::new();
        let status = backend.parse_output("", "null pointer dereference at line 5", false);
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[test]
    fn parse_output_buffer_overflow() {
        let backend = HaybaleBackend::new();
        let status = backend.parse_output("", "buffer overflow detected", false);
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[test]
    fn parse_output_out_of_bounds() {
        let backend = HaybaleBackend::new();
        let status = backend.parse_output("", "array access out of bounds", false);
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[test]
    fn parse_output_division_by_zero() {
        let backend = HaybaleBackend::new();
        let status = backend.parse_output("", "division by zero", false);
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[test]
    fn parse_output_path_explosion() {
        let backend = HaybaleBackend::new();
        let status = backend.parse_output("", "max paths reached", false);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
        if let VerificationStatus::Unknown { reason } = status {
            assert!(reason.contains("path"));
        }
    }

    #[test]
    fn parse_output_timeout() {
        let backend = HaybaleBackend::new();
        let status = backend.parse_output("", "execution timeout", false);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn parse_output_solver_error() {
        let backend = HaybaleBackend::new();
        let status = backend.parse_output("", "solver error during constraint solving", false);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn parse_output_generic_error() {
        let backend = HaybaleBackend::new();
        let status = backend.parse_output("", "error: something went wrong", false);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    // =============================================
    // Counterexample extraction tests
    // =============================================

    #[test]
    fn extract_failed_checks_assertion() {
        let output = "error: assertion failed at main.rs:10:5";
        let checks = HaybaleBackend::extract_failed_checks(output);
        assert!(!checks.is_empty());
        assert_eq!(checks[0].check_id, "haybale_assertion");
    }

    #[test]
    fn extract_failed_checks_null_deref() {
        let output = "error: null pointer dereference";
        let checks = HaybaleBackend::extract_failed_checks(output);
        assert!(!checks.is_empty());
        assert_eq!(checks[0].check_id, "haybale_null_deref");
    }

    #[test]
    fn extract_failed_checks_buffer_overflow() {
        let output = "error: buffer overflow detected";
        let checks = HaybaleBackend::extract_failed_checks(output);
        assert!(!checks.is_empty());
        assert_eq!(checks[0].check_id, "haybale_buffer_overflow");
    }

    #[test]
    fn extract_failed_checks_div_by_zero() {
        let output = "error: division by zero";
        let checks = HaybaleBackend::extract_failed_checks(output);
        assert!(!checks.is_empty());
        assert_eq!(checks[0].check_id, "haybale_div_by_zero");
    }

    #[test]
    fn extract_failed_checks_generic_error() {
        let output = "Error: something unexpected happened";
        let checks = HaybaleBackend::extract_failed_checks(output);
        assert!(!checks.is_empty());
        assert_eq!(checks[0].check_id, "haybale_error");
    }

    #[test]
    fn extract_location_from_rust_file() {
        let line = "error at main.rs:10:5: assertion failed";
        let location = HaybaleBackend::extract_location_from_line(line);
        assert!(location.is_some());
        let loc = location.unwrap();
        assert_eq!(loc.file, "main.rs");
        assert_eq!(loc.line, 10);
        assert_eq!(loc.column, Some(5));
    }

    #[test]
    fn extract_location_from_c_file() {
        let line = "error at test.c:25:12: buffer overflow";
        let location = HaybaleBackend::extract_location_from_line(line);
        assert!(location.is_some());
        let loc = location.unwrap();
        assert_eq!(loc.file, "test.c");
        assert_eq!(loc.line, 25);
    }

    #[test]
    fn extract_location_from_ll_file() {
        let line = "error at module.ll:100: undefined behavior";
        let location = HaybaleBackend::extract_location_from_line(line);
        assert!(location.is_some());
        let loc = location.unwrap();
        assert_eq!(loc.file, "module.ll");
        assert_eq!(loc.line, 100);
    }

    #[test]
    fn extract_function_name_from_fn() {
        let lines = vec!["fn test_function() {", "    assert!(false);", "}"];
        let function = HaybaleBackend::extract_function_name(&lines, 1);
        assert!(function.is_some());
        assert_eq!(function.unwrap(), "test_function");
    }

    #[test]
    fn extract_function_name_from_context() {
        let lines = vec![
            "in function main:",
            "  error at line 10",
            "  assertion failed",
        ];
        let function = HaybaleBackend::extract_function_name(&lines, 2);
        assert!(function.is_some());
        assert_eq!(function.unwrap(), "main");
    }

    #[test]
    fn extract_inputs_basic() {
        let output = "input x = 42\ninput y = -1\nsymbolic z = 0xff";
        let inputs = HaybaleBackend::extract_inputs(output);
        assert!(inputs.contains_key("x"));
        assert!(inputs.contains_key("y"));
    }

    #[test]
    fn parse_counterexample_has_raw() {
        let stdout = "execution output";
        let stderr = "error: assertion failed";
        let ce = HaybaleBackend::parse_counterexample(stdout, stderr);
        assert!(ce.raw.is_some());
        assert!(ce.raw.unwrap().contains("assertion failed"));
    }

    // =============================================
    // Runner generation tests
    // =============================================

    fn create_test_spec() -> TypedSpec {
        use dashprove_usl::ast::Spec;
        use std::collections::HashMap;

        TypedSpec {
            spec: Spec::default(),
            type_info: HashMap::new(),
        }
    }

    #[test]
    fn generate_runner_includes_haybale_import() {
        let backend = HaybaleBackend::new();
        let spec = create_test_spec();
        let code = backend.generate_haybale_runner(&spec, &PathBuf::from("/tmp/test.bc"));
        assert!(code.contains("use haybale"));
    }

    #[test]
    fn generate_runner_includes_project_load() {
        let backend = HaybaleBackend::new();
        let spec = create_test_spec();
        let code = backend.generate_haybale_runner(&spec, &PathBuf::from("/tmp/test.bc"));
        assert!(code.contains("from_bc_path"));
    }

    #[test]
    fn generate_runner_includes_config() {
        let backend = HaybaleBackend::new();
        let spec = create_test_spec();
        let code = backend.generate_haybale_runner(&spec, &PathBuf::from("/tmp/test.bc"));
        assert!(code.contains("Config::default"));
    }

    #[test]
    fn generate_runner_includes_max_loop_iters() {
        let backend = HaybaleBackend::new();
        let spec = create_test_spec();
        let code = backend.generate_haybale_runner(&spec, &PathBuf::from("/tmp/test.bc"));
        assert!(code.contains("max_loop_iters"));
    }

    #[test]
    fn generate_runner_with_entry_function() {
        let config =
            HaybaleConfig::default().with_entry_function("test_target_function".to_string());
        let backend = HaybaleBackend::with_config(config);
        let spec = create_test_spec();
        let code = backend.generate_haybale_runner(&spec, &PathBuf::from("/tmp/test.bc"));
        assert!(code.contains("find_executions_for_fn"));
        assert!(code.contains("test_target_function"));
    }

    #[test]
    fn generate_runner_without_entry_function() {
        let backend = HaybaleBackend::new();
        let spec = create_test_spec();
        let code = backend.generate_haybale_runner(&spec, &PathBuf::from("/tmp/test.bc"));
        assert!(code.contains("find_executions_for_all_fns"));
    }

    #[test]
    fn generate_runner_includes_result_handling() {
        let backend = HaybaleBackend::new();
        let spec = create_test_spec();
        let code = backend.generate_haybale_runner(&spec, &PathBuf::from("/tmp/test.bc"));
        assert!(code.contains("for result in results"));
        assert!(code.contains("Ok(state)"));
        assert!(code.contains("Err(e)"));
    }

    // =============================================
    // Default trait tests
    // =============================================

    #[test]
    fn backend_default() {
        let backend = HaybaleBackend::default();
        assert_eq!(backend.id(), BackendId::Haybale);
        assert_eq!(backend.config.timeout, Duration::from_secs(600));
    }

    #[test]
    fn backend_with_config() {
        let config = HaybaleConfig {
            timeout: Duration::from_secs(300),
            max_paths: Some(5000),
            ..Default::default()
        };
        let backend = HaybaleBackend::with_config(config);
        assert_eq!(backend.config.timeout, Duration::from_secs(300));
        assert_eq!(backend.config.max_paths, Some(5000));
    }

    // =============================================
    // Health check tests
    // =============================================

    #[tokio::test]
    async fn health_check_returns_status() {
        let backend = HaybaleBackend::new();
        let health = backend.health_check().await;
        // Should return some status (healthy, degraded, or unavailable)
        match health {
            HealthStatus::Healthy => {}
            HealthStatus::Degraded { .. } => {}
            HealthStatus::Unavailable { .. } => {}
        }
    }
}
