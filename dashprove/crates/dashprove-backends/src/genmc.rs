//! GenMC backend for stateless model checking of concurrent programs
//!
//! This backend runs GenMC to verify concurrent code under various memory models.
//! GenMC uses dynamic partial order reduction (DPOR) algorithms to efficiently
//! explore concurrent program executions.

// =============================================
// Kani Proofs for GenMC Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- GenMCMemoryModel Default Tests ----

    /// Verify GenMCMemoryModel::default is RA (Release-Acquire)
    #[kani::proof]
    fn proof_genmc_memory_model_default_is_ra() {
        let model = GenMCMemoryModel::default();
        kani::assert(
            model == GenMCMemoryModel::RA,
            "Default memory model should be RA",
        );
    }

    // ---- DPORAlgorithm Default Tests ----

    /// Verify DPORAlgorithm::default is Source
    #[kani::proof]
    fn proof_dpor_algorithm_default_is_source() {
        let algorithm = DPORAlgorithm::default();
        kani::assert(
            algorithm == DPORAlgorithm::Source,
            "Default DPOR algorithm should be Source",
        );
    }

    // ---- GenMCConfig Default Tests ----

    /// Verify GenMCConfig::default source_path is None
    #[kani::proof]
    fn proof_genmc_config_default_source_path_none() {
        let config = GenMCConfig::default();
        kani::assert(
            config.source_path.is_none(),
            "Default source_path should be None",
        );
    }

    /// Verify GenMCConfig::default genmc_path is None
    #[kani::proof]
    fn proof_genmc_config_default_genmc_path_none() {
        let config = GenMCConfig::default();
        kani::assert(
            config.genmc_path.is_none(),
            "Default genmc_path should be None",
        );
    }

    /// Verify GenMCConfig::default timeout is 300 seconds
    #[kani::proof]
    fn proof_genmc_config_default_timeout() {
        let config = GenMCConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "Default timeout should be 300 seconds",
        );
    }

    /// Verify GenMCConfig::default memory_model is RA
    #[kani::proof]
    fn proof_genmc_config_default_memory_model() {
        let config = GenMCConfig::default();
        kani::assert(
            config.memory_model == GenMCMemoryModel::RA,
            "Default memory model should be RA",
        );
    }

    /// Verify GenMCConfig::default dpor_algorithm is Source
    #[kani::proof]
    fn proof_genmc_config_default_dpor_algorithm() {
        let config = GenMCConfig::default();
        kani::assert(
            config.dpor_algorithm == DPORAlgorithm::Source,
            "Default DPOR algorithm should be Source",
        );
    }

    /// Verify GenMCConfig::default max_executions is 0 (unlimited)
    #[kani::proof]
    fn proof_genmc_config_default_max_executions() {
        let config = GenMCConfig::default();
        kani::assert(
            config.max_executions == 0,
            "Default max_executions should be 0",
        );
    }

    /// Verify GenMCConfig::default verbose is false
    #[kani::proof]
    fn proof_genmc_config_default_verbose_false() {
        let config = GenMCConfig::default();
        kani::assert(!config.verbose, "Default verbose should be false");
    }

    /// Verify GenMCConfig::default symmetry_reduction is true
    #[kani::proof]
    fn proof_genmc_config_default_symmetry_reduction_true() {
        let config = GenMCConfig::default();
        kani::assert(
            config.symmetry_reduction,
            "Default symmetry_reduction should be true",
        );
    }

    /// Verify GenMCConfig::default loop_bound is 10
    #[kani::proof]
    fn proof_genmc_config_default_loop_bound() {
        let config = GenMCConfig::default();
        kani::assert(config.loop_bound == 10, "Default loop_bound should be 10");
    }

    /// Verify GenMCConfig::default lock_aware is true
    #[kani::proof]
    fn proof_genmc_config_default_lock_aware_true() {
        let config = GenMCConfig::default();
        kani::assert(config.lock_aware, "Default lock_aware should be true");
    }

    /// Verify GenMCConfig::default compiler_flags is empty
    #[kani::proof]
    fn proof_genmc_config_default_compiler_flags_empty() {
        let config = GenMCConfig::default();
        kani::assert(
            config.compiler_flags.is_empty(),
            "Default compiler_flags should be empty",
        );
    }

    // ---- GenMCConfig Builder Tests ----

    /// Verify with_source_path sets the path
    #[kani::proof]
    fn proof_genmc_config_with_source_path() {
        let config = GenMCConfig::default().with_source_path(PathBuf::from("/test"));
        kani::assert(
            config.source_path.is_some(),
            "with_source_path should set source_path",
        );
    }

    /// Verify with_genmc_path sets the path
    #[kani::proof]
    fn proof_genmc_config_with_genmc_path() {
        let config = GenMCConfig::default().with_genmc_path(PathBuf::from("/opt/genmc"));
        kani::assert(
            config.genmc_path.is_some(),
            "with_genmc_path should set genmc_path",
        );
    }

    /// Verify with_timeout sets the timeout
    #[kani::proof]
    fn proof_genmc_config_with_timeout() {
        let config = GenMCConfig::default().with_timeout(Duration::from_secs(120));
        kani::assert(
            config.timeout == Duration::from_secs(120),
            "with_timeout should set timeout to 120 seconds",
        );
    }

    /// Verify with_memory_model sets TSO
    #[kani::proof]
    fn proof_genmc_config_with_memory_model_tso() {
        let config = GenMCConfig::default().with_memory_model(GenMCMemoryModel::TSO);
        kani::assert(
            config.memory_model == GenMCMemoryModel::TSO,
            "Should set TSO memory model",
        );
    }

    /// Verify with_memory_model sets SC
    #[kani::proof]
    fn proof_genmc_config_with_memory_model_sc() {
        let config = GenMCConfig::default().with_memory_model(GenMCMemoryModel::SC);
        kani::assert(
            config.memory_model == GenMCMemoryModel::SC,
            "Should set SC memory model",
        );
    }

    /// Verify with_dpor_algorithm sets Optimal
    #[kani::proof]
    fn proof_genmc_config_with_dpor_optimal() {
        let config = GenMCConfig::default().with_dpor_algorithm(DPORAlgorithm::Optimal);
        kani::assert(
            config.dpor_algorithm == DPORAlgorithm::Optimal,
            "Should set Optimal DPOR",
        );
    }

    /// Verify with_max_executions sets the value
    #[kani::proof]
    fn proof_genmc_config_with_max_executions() {
        let config = GenMCConfig::default().with_max_executions(5000);
        kani::assert(
            config.max_executions == 5000,
            "Should set max_executions to 5000",
        );
    }

    /// Verify with_verbose sets verbose to true
    #[kani::proof]
    fn proof_genmc_config_with_verbose_true() {
        let config = GenMCConfig::default().with_verbose(true);
        kani::assert(config.verbose, "with_verbose(true) should set verbose");
    }

    /// Verify with_verbose sets verbose to false
    #[kani::proof]
    fn proof_genmc_config_with_verbose_false() {
        let config = GenMCConfig::default().with_verbose(false);
        kani::assert(!config.verbose, "with_verbose(false) should unset verbose");
    }

    /// Verify with_symmetry_reduction disables
    #[kani::proof]
    fn proof_genmc_config_with_symmetry_reduction_false() {
        let config = GenMCConfig::default().with_symmetry_reduction(false);
        kani::assert(
            !config.symmetry_reduction,
            "Should disable symmetry reduction",
        );
    }

    /// Verify with_loop_bound sets the bound
    #[kani::proof]
    fn proof_genmc_config_with_loop_bound() {
        let config = GenMCConfig::default().with_loop_bound(20);
        kani::assert(config.loop_bound == 20, "Should set loop_bound to 20");
    }

    /// Verify with_lock_aware disables
    #[kani::proof]
    fn proof_genmc_config_with_lock_aware_false() {
        let config = GenMCConfig::default().with_lock_aware(false);
        kani::assert(!config.lock_aware, "Should disable lock_aware");
    }

    // ---- GenMCBackend Construction Tests ----

    /// Verify GenMCBackend::new uses default timeout
    #[kani::proof]
    fn proof_genmc_backend_new_default_timeout() {
        let backend = GenMCBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(300),
            "New backend should use default timeout",
        );
    }

    /// Verify GenMCBackend::default equals GenMCBackend::new
    #[kani::proof]
    fn proof_genmc_backend_default_equals_new() {
        let default_backend = GenMCBackend::default();
        let new_backend = GenMCBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "Default and new should have same timeout",
        );
    }

    /// Verify GenMCBackend::with_config preserves timeout
    #[kani::proof]
    fn proof_genmc_backend_with_config_timeout() {
        let mut config = GenMCConfig::default();
        config.timeout = Duration::from_secs(600);
        let backend = GenMCBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "with_config should preserve timeout",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify GenMCBackend::id returns GenMC
    #[kani::proof]
    fn proof_genmc_backend_id() {
        let backend = GenMCBackend::new();
        kani::assert(
            backend.id() == BackendId::GenMC,
            "Backend id should be GenMC",
        );
    }

    /// Verify GenMCBackend::supports includes DataRace
    #[kani::proof]
    fn proof_genmc_backend_supports_data_race() {
        let backend = GenMCBackend::new();
        let supported = backend.supports();
        let has_race = supported.iter().any(|p| *p == PropertyType::DataRace);
        kani::assert(has_race, "Should support DataRace property");
    }

    /// Verify GenMCBackend::supports includes MemorySafety
    #[kani::proof]
    fn proof_genmc_backend_supports_memory_safety() {
        let backend = GenMCBackend::new();
        let supported = backend.supports();
        let has_mem = supported.iter().any(|p| *p == PropertyType::MemorySafety);
        kani::assert(has_mem, "Should support MemorySafety property");
    }

    /// Verify GenMCBackend::supports includes UndefinedBehavior
    #[kani::proof]
    fn proof_genmc_backend_supports_undefined_behavior() {
        let backend = GenMCBackend::new();
        let supported = backend.supports();
        let has_ub = supported
            .iter()
            .any(|p| *p == PropertyType::UndefinedBehavior);
        kani::assert(has_ub, "Should support UndefinedBehavior property");
    }

    /// Verify GenMCBackend::supports returns exactly 3 properties
    #[kani::proof]
    fn proof_genmc_backend_supports_length() {
        let backend = GenMCBackend::new();
        let supported = backend.supports();
        kani::assert(supported.len() == 3, "Should support exactly 3 properties");
    }

    // ---- extract_number Tests ----

    /// Verify extract_number parses "1234" from line
    #[kani::proof]
    fn proof_extract_number_finds_number() {
        let result = extract_number("Explored 1234 executions");
        kani::assert(result == Some(1234), "Should extract 1234");
    }

    /// Verify extract_number returns None for no numbers
    #[kani::proof]
    fn proof_extract_number_no_numbers() {
        let result = extract_number("No numbers here");
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

/// GenMC memory model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum GenMCMemoryModel {
    /// Sequential Consistency (SC)
    SC,
    /// Total Store Order (TSO) - x86-like
    TSO,
    /// Partial Store Order (PSO)
    PSO,
    /// Release-Acquire (RA) - C/C++11 acquire-release
    #[default]
    RA,
    /// Relaxed Memory Order (RMO)
    RMO,
    /// RC11 - Repaired C11 memory model
    RC11,
    /// IMM - Intermediate Memory Model
    IMM,
}

/// DPOR algorithm variant
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum DPORAlgorithm {
    /// Source-sets DPOR (default, optimal)
    #[default]
    Source,
    /// Reads-from DPOR
    ReadsFrom,
    /// Optimal DPOR
    Optimal,
    /// Persistent sets DPOR
    Persistent,
}

/// Configuration for GenMC backend
#[derive(Debug, Clone)]
pub struct GenMCConfig {
    /// Path to the source file or binary to analyze
    pub source_path: Option<PathBuf>,
    /// Path to GenMC installation
    pub genmc_path: Option<PathBuf>,
    /// Timeout for analysis
    pub timeout: Duration,
    /// Memory model to use
    pub memory_model: GenMCMemoryModel,
    /// DPOR algorithm
    pub dpor_algorithm: DPORAlgorithm,
    /// Maximum number of executions to explore
    pub max_executions: u64,
    /// Enable verbose output
    pub verbose: bool,
    /// Enable symmetry reduction
    pub symmetry_reduction: bool,
    /// Bound on loop iterations
    pub loop_bound: u32,
    /// Enable lock-aware exploration
    pub lock_aware: bool,
    /// Additional compiler flags
    pub compiler_flags: Vec<String>,
}

impl Default for GenMCConfig {
    fn default() -> Self {
        Self {
            source_path: None,
            genmc_path: None,
            timeout: Duration::from_secs(300),
            memory_model: GenMCMemoryModel::default(),
            dpor_algorithm: DPORAlgorithm::default(),
            max_executions: 0, // 0 = unlimited
            verbose: false,
            symmetry_reduction: true,
            loop_bound: 10,
            lock_aware: true,
            compiler_flags: Vec::new(),
        }
    }
}

impl GenMCConfig {
    /// Set the source path
    pub fn with_source_path(mut self, path: PathBuf) -> Self {
        self.source_path = Some(path);
        self
    }

    /// Set GenMC installation path
    pub fn with_genmc_path(mut self, path: PathBuf) -> Self {
        self.genmc_path = Some(path);
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set memory model
    pub fn with_memory_model(mut self, model: GenMCMemoryModel) -> Self {
        self.memory_model = model;
        self
    }

    /// Set DPOR algorithm
    pub fn with_dpor_algorithm(mut self, algorithm: DPORAlgorithm) -> Self {
        self.dpor_algorithm = algorithm;
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

    /// Enable/disable symmetry reduction
    pub fn with_symmetry_reduction(mut self, enable: bool) -> Self {
        self.symmetry_reduction = enable;
        self
    }

    /// Set loop bound
    pub fn with_loop_bound(mut self, bound: u32) -> Self {
        self.loop_bound = bound;
        self
    }

    /// Enable/disable lock-aware exploration
    pub fn with_lock_aware(mut self, enable: bool) -> Self {
        self.lock_aware = enable;
        self
    }

    /// Add compiler flags
    pub fn with_compiler_flags(mut self, flags: Vec<String>) -> Self {
        self.compiler_flags = flags;
        self
    }
}

/// GenMC verification backend for stateless model checking
///
/// GenMC is a stateless model checker for concurrent programs that uses
/// dynamic partial order reduction (DPOR) to efficiently explore all
/// possible thread interleavings. It supports multiple memory models:
///
/// - Sequential Consistency (SC)
/// - Total Store Order (TSO)
/// - Release-Acquire (RA)
/// - RC11 (Repaired C11)
/// - IMM (Intermediate Memory Model)
///
/// # Requirements
///
/// Install GenMC:
/// ```bash
/// git clone https://github.com/MPI-SWS/genmc
/// cd genmc
/// mkdir build && cd build
/// cmake ..
/// make
/// make install
/// ```
///
/// # Supported Languages
///
/// GenMC primarily works with C/C++ programs using the LLVM framework.
/// It instruments the code via Clang and systematically explores executions.
///
/// For Rust concurrent code, you would need to either:
/// 1. Export C-compatible interfaces and test those
/// 2. Use GenMC's LLVM bitcode interface with Rust-compiled LLVM IR
pub struct GenMCBackend {
    config: GenMCConfig,
}

impl GenMCBackend {
    /// Create a new GenMC backend with default configuration
    pub fn new() -> Self {
        Self {
            config: GenMCConfig::default(),
        }
    }

    /// Create a new GenMC backend with custom configuration
    pub fn with_config(config: GenMCConfig) -> Self {
        Self { config }
    }

    /// Run GenMC verification on a source file
    pub async fn analyze_source(&self, source_path: &Path) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        // Determine GenMC command
        let genmc_cmd = self
            .config
            .genmc_path
            .clone()
            .unwrap_or_else(|| PathBuf::from("genmc"));

        // Build command arguments
        let mut args = Vec::new();

        // Add memory model
        let mm_flag = match self.config.memory_model {
            GenMCMemoryModel::SC => "--sc",
            GenMCMemoryModel::TSO => "--tso",
            GenMCMemoryModel::PSO => "--pso",
            GenMCMemoryModel::RA => "--ra",
            GenMCMemoryModel::RMO => "--rmo",
            GenMCMemoryModel::RC11 => "--rc11",
            GenMCMemoryModel::IMM => "--imm",
        };
        args.push(mm_flag.to_string());

        // Add DPOR algorithm
        let dpor_flag = match self.config.dpor_algorithm {
            DPORAlgorithm::Source => "--dpor=source",
            DPORAlgorithm::ReadsFrom => "--dpor=rf",
            DPORAlgorithm::Optimal => "--dpor=optimal",
            DPORAlgorithm::Persistent => "--dpor=persistent",
        };
        args.push(dpor_flag.to_string());

        // Add max executions
        if self.config.max_executions > 0 {
            args.push(format!("--bound={}", self.config.max_executions));
        }

        // Add loop bound
        args.push(format!("--unroll={}", self.config.loop_bound));

        // Add symmetry reduction
        if self.config.symmetry_reduction {
            args.push("--symmetry".to_string());
        }

        // Add lock-aware exploration
        if self.config.lock_aware {
            args.push("--lock-aware".to_string());
        }

        // Add verbose flag
        if self.config.verbose {
            args.push("-v".to_string());
        }

        // Add compiler flags
        for flag in &self.config.compiler_flags {
            args.push(format!("--cflags={}", flag));
        }

        // Add the source file
        args.push("--".to_string());
        args.push(source_path.to_string_lossy().to_string());

        // Run GenMC
        let mut cmd = Command::new(&genmc_cmd);
        cmd.args(&args);

        let output = tokio::time::timeout(self.config.timeout, cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| BackendError::VerificationFailed(format!("Failed to run GenMC: {}", e)))?;

        let duration = start.elapsed();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined = format!("{}\n{}", stdout, stderr);

        // Parse GenMC output
        let (status, findings, stats) = self.parse_genmc_output(&combined, output.status.success());

        // Build diagnostics
        let mut diagnostics = Vec::new();

        // Summary
        let mm_name = match self.config.memory_model {
            GenMCMemoryModel::SC => "Sequential Consistency",
            GenMCMemoryModel::TSO => "TSO",
            GenMCMemoryModel::PSO => "PSO",
            GenMCMemoryModel::RA => "Release-Acquire",
            GenMCMemoryModel::RMO => "RMO",
            GenMCMemoryModel::RC11 => "RC11",
            GenMCMemoryModel::IMM => "IMM",
        };

        let summary = match &status {
            VerificationStatus::Proven => format!(
                "GenMC: All {} executions verified under {} memory model",
                stats.executions_explored, mm_name
            ),
            VerificationStatus::Disproven => {
                format!(
                    "GenMC: {} bug(s) found in {} executions",
                    findings.len(),
                    stats.executions_explored
                )
            }
            VerificationStatus::Partial {
                verified_percentage,
            } => {
                format!(
                    "GenMC: {:.1}% of executions clean ({} explored)",
                    verified_percentage, stats.executions_explored
                )
            }
            VerificationStatus::Unknown { reason } => format!("GenMC: {}", reason),
        };
        diagnostics.push(summary);

        // Add stats
        if stats.executions_explored > 0 {
            diagnostics.push(format!(
                "Executions: {}, Data races: {}, Assertion failures: {}, Memory errors: {}",
                stats.executions_explored,
                stats.data_races,
                stats.assertion_failures,
                stats.memory_errors
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
            backend: BackendId::GenMC,
            status,
            proof: None,
            counterexample,
            diagnostics,
            time_taken: duration,
        })
    }

    fn parse_genmc_output(
        &self,
        output: &str,
        success: bool,
    ) -> (VerificationStatus, Vec<String>, GenMCStats) {
        let mut findings = Vec::new();
        let mut stats = GenMCStats::default();

        // Parse GenMC output
        for line in output.lines() {
            // Statistics patterns
            if line.contains("Explored") && line.contains("execution") {
                if let Some(num) = extract_number(line) {
                    stats.executions_explored = num;
                }
            }
            if line.contains("Total blocked:") || line.contains("blocked executions") {
                if let Some(num) = extract_number(line) {
                    stats.blocked_executions = num;
                }
            }

            // Bug patterns
            if line.contains("Data race") || line.contains("data race") {
                findings.push(line.trim().to_string());
                stats.data_races += 1;
            }
            if line.contains("Assertion") && line.contains("fail") {
                findings.push(line.trim().to_string());
                stats.assertion_failures += 1;
            }
            if line.contains("Memory error")
                || line.contains("invalid memory access")
                || line.contains("use-after-free")
                || line.contains("double-free")
            {
                findings.push(line.trim().to_string());
                stats.memory_errors += 1;
            }
            if line.contains("Deadlock") || line.contains("deadlock") {
                findings.push(line.trim().to_string());
                stats.deadlocks += 1;
            }
            if line.contains("Livelock") || line.contains("livelock") {
                findings.push(line.trim().to_string());
                stats.livelocks += 1;
            }

            // Error pattern from GenMC
            if line.starts_with("Error:") || line.starts_with("ERROR:") {
                findings.push(line.trim().to_string());
            }
        }

        let status = if findings.is_empty() && success {
            if stats.executions_explored > 0 {
                VerificationStatus::Proven
            } else {
                VerificationStatus::Unknown {
                    reason: "No executions explored".to_string(),
                }
            }
        } else if !findings.is_empty() {
            VerificationStatus::Disproven
        } else if !success {
            // Check for installation issues
            if output.contains("not found") || output.contains("command not found") {
                return (
                    VerificationStatus::Unknown {
                        reason: "GenMC not installed. Build from: https://github.com/MPI-SWS/genmc"
                            .to_string(),
                    },
                    Vec::new(),
                    stats,
                );
            }
            if output.contains("compilation failed") || output.contains("clang error") {
                return (
                    VerificationStatus::Unknown {
                        reason: "Source file compilation failed".to_string(),
                    },
                    Vec::new(),
                    stats,
                );
            }
            VerificationStatus::Unknown {
                reason: "Verification failed with unknown error".to_string(),
            }
        } else {
            VerificationStatus::Proven
        };

        (status, findings, stats)
    }

    /// Check if GenMC is installed
    pub async fn check_installed(&self) -> Result<bool, BackendError> {
        let genmc_cmd = self
            .config
            .genmc_path
            .clone()
            .unwrap_or_else(|| PathBuf::from("genmc"));

        let output = Command::new(&genmc_cmd).arg("--version").output().await;

        match output {
            Ok(out) if out.status.success() => Ok(true),
            Ok(out) => {
                // Some versions print to stderr
                let stderr = String::from_utf8_lossy(&out.stderr);
                let stdout = String::from_utf8_lossy(&out.stdout);
                Ok(stderr.contains("GenMC")
                    || stdout.contains("GenMC")
                    || stderr.contains("genmc")
                    || stdout.contains("genmc"))
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

/// Statistics from GenMC run
#[derive(Debug, Clone, Default)]
struct GenMCStats {
    executions_explored: u64,
    blocked_executions: u64,
    data_races: u64,
    assertion_failures: u64,
    memory_errors: u64,
    deadlocks: u64,
    livelocks: u64,
}

impl Default for GenMCBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VerificationBackend for GenMCBackend {
    fn id(&self) -> BackendId {
        BackendId::GenMC
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![
            PropertyType::DataRace,
            PropertyType::MemorySafety,
            PropertyType::UndefinedBehavior,
        ]
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let source_path = self.config.source_path.clone().ok_or_else(|| {
            BackendError::Unavailable(
                "GenMC backend requires source_path pointing to a C/C++ source file".to_string(),
            )
        })?;

        self.analyze_source(&source_path).await
    }

    async fn health_check(&self) -> HealthStatus {
        match self.check_installed().await {
            Ok(true) => HealthStatus::Healthy,
            Ok(false) => HealthStatus::Unavailable {
                reason: "GenMC not installed. Build from: https://github.com/MPI-SWS/genmc"
                    .to_string(),
            },
            Err(e) => HealthStatus::Unavailable {
                reason: format!("Failed to check GenMC installation: {}", e),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genmc_config_default() {
        let config = GenMCConfig::default();
        assert!(config.source_path.is_none());
        assert!(config.genmc_path.is_none());
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert_eq!(config.memory_model, GenMCMemoryModel::RA);
        assert_eq!(config.dpor_algorithm, DPORAlgorithm::Source);
        assert_eq!(config.max_executions, 0);
        assert!(!config.verbose);
        assert!(config.symmetry_reduction);
        assert_eq!(config.loop_bound, 10);
        assert!(config.lock_aware);
        assert!(config.compiler_flags.is_empty());
    }

    #[test]
    fn test_genmc_config_builder() {
        let config = GenMCConfig::default()
            .with_source_path(PathBuf::from("/test/source.c"))
            .with_genmc_path(PathBuf::from("/opt/genmc/bin/genmc"))
            .with_timeout(Duration::from_secs(120))
            .with_memory_model(GenMCMemoryModel::TSO)
            .with_dpor_algorithm(DPORAlgorithm::Optimal)
            .with_max_executions(5000)
            .with_verbose(true)
            .with_symmetry_reduction(false)
            .with_loop_bound(20)
            .with_lock_aware(false)
            .with_compiler_flags(vec!["-O2".to_string(), "-DNDEBUG".to_string()]);

        assert_eq!(config.source_path, Some(PathBuf::from("/test/source.c")));
        assert_eq!(
            config.genmc_path,
            Some(PathBuf::from("/opt/genmc/bin/genmc"))
        );
        assert_eq!(config.timeout, Duration::from_secs(120));
        assert_eq!(config.memory_model, GenMCMemoryModel::TSO);
        assert_eq!(config.dpor_algorithm, DPORAlgorithm::Optimal);
        assert_eq!(config.max_executions, 5000);
        assert!(config.verbose);
        assert!(!config.symmetry_reduction);
        assert_eq!(config.loop_bound, 20);
        assert!(!config.lock_aware);
        assert_eq!(config.compiler_flags, vec!["-O2", "-DNDEBUG"]);
    }

    #[test]
    fn test_genmc_backend_id() {
        let backend = GenMCBackend::new();
        assert_eq!(backend.id(), BackendId::GenMC);
    }

    #[test]
    fn test_genmc_supports_properties() {
        let backend = GenMCBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::DataRace));
        assert!(supported.contains(&PropertyType::MemorySafety));
        assert!(supported.contains(&PropertyType::UndefinedBehavior));
    }

    #[test]
    fn test_genmc_parse_output_clean() {
        let backend = GenMCBackend::new();
        let output =
            "GenMC model checking...\nExplored 1234 executions\nNo errors found\nVerification complete.";
        let (status, findings, stats) = backend.parse_genmc_output(output, true);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(findings.is_empty());
        assert_eq!(stats.executions_explored, 1234);
    }

    #[test]
    fn test_genmc_parse_output_with_data_race() {
        let backend = GenMCBackend::new();
        let output = "Explored 500 executions\nData race detected between threads T1 and T2";
        let (status, findings, stats) = backend.parse_genmc_output(output, false);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(!findings.is_empty());
        assert_eq!(stats.data_races, 1);
    }

    #[test]
    fn test_genmc_parse_output_with_deadlock() {
        let backend = GenMCBackend::new();
        let output = "Explored 100 executions\nDeadlock detected in execution 42";
        let (status, _findings, stats) = backend.parse_genmc_output(output, false);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert_eq!(stats.deadlocks, 1);
    }

    #[test]
    fn test_genmc_parse_output_not_installed() {
        let backend = GenMCBackend::new();
        let output = "genmc: command not found";
        let (status, _, _) = backend.parse_genmc_output(output, false);
        match status {
            VerificationStatus::Unknown { reason } => {
                assert!(reason.contains("not installed"));
            }
            _ => panic!("Expected Unknown status"),
        }
    }

    #[test]
    fn test_extract_number() {
        assert_eq!(extract_number("Explored 1234 executions"), Some(1234));
        assert_eq!(extract_number("Total blocked: 5"), Some(5));
        assert_eq!(extract_number("No numbers here"), None);
    }

    #[tokio::test]
    async fn test_genmc_health_check() {
        let backend = GenMCBackend::new();
        let health = backend.health_check().await;
        // GenMC may or may not be installed
        match health {
            HealthStatus::Healthy => {}
            HealthStatus::Unavailable { reason } => {
                assert!(reason.contains("GenMC") || reason.contains("genmc"));
            }
            HealthStatus::Degraded { .. } => {}
        }
    }

    #[tokio::test]
    async fn test_genmc_verify_requires_source_path() {
        use dashprove_usl::{parse, typecheck};

        let backend = GenMCBackend::new();
        let spec = parse("theorem test { true }").expect("parse");
        let typed_spec = typecheck(spec).expect("typecheck");

        let result = backend.verify(&typed_spec).await;
        assert!(result.is_err());
        if let Err(BackendError::Unavailable(msg)) = result {
            assert!(msg.contains("source_path"));
        } else {
            panic!("Expected Unavailable error");
        }
    }

    #[test]
    fn test_memory_model_default() {
        assert_eq!(GenMCMemoryModel::default(), GenMCMemoryModel::RA);
    }

    #[test]
    fn test_dpor_algorithm_default() {
        assert_eq!(DPORAlgorithm::default(), DPORAlgorithm::Source);
    }
}
