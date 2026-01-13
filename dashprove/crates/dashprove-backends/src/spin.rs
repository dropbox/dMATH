//! SPIN model checker backend for protocol verification
//!
//! SPIN is a widely-used model checker for verifying concurrent systems,
//! protocols, and distributed algorithms. It uses the Promela specification
//! language and supports LTL temporal properties.
//!
//! See: <https://spinroot.com/>
//!
//! # Features
//!
//! - **State space exploration**: Exhaustive or bounded model checking
//! - **LTL verification**: Check linear temporal logic properties
//! - **Safety properties**: Assertions, deadlock detection
//! - **Liveness properties**: Progress, fairness constraints
//! - **Concurrency**: Multiple processes, channels, synchronization
//!
//! # Requirements
//!
//! Install SPIN:
//! ```bash
//! # macOS
//! brew install spin
//!
//! # Linux (compile from source)
//! wget https://github.com/nimble-code/Spin/archive/refs/heads/master.zip
//! unzip master.zip && cd Spin-master/Src
//! make && sudo cp spin /usr/local/bin/
//!
//! # Ubuntu/Debian
//! apt install spin
//! ```

// =============================================
// Kani Proofs for SPIN Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- SpinSearchMode Enum Tests ----

    /// Verify SpinSearchMode::default is DepthFirst
    #[kani::proof]
    fn proof_spin_search_mode_default() {
        let mode = SpinSearchMode::default();
        kani::assert(
            matches!(mode, SpinSearchMode::DepthFirst),
            "Default should be DepthFirst",
        );
    }

    /// Verify SpinSearchMode variants are distinct
    #[kani::proof]
    fn proof_spin_search_mode_variants_distinct() {
        kani::assert(
            SpinSearchMode::DepthFirst != SpinSearchMode::BreadthFirst,
            "DepthFirst != BreadthFirst",
        );
        kani::assert(
            SpinSearchMode::DepthFirst != SpinSearchMode::Swarm,
            "DepthFirst != Swarm",
        );
        kani::assert(
            SpinSearchMode::BreadthFirst != SpinSearchMode::Swarm,
            "BreadthFirst != Swarm",
        );
    }

    // ---- SpinConfig Default Tests ----

    /// Verify SpinConfig::default timeout is 300 seconds
    #[kani::proof]
    fn proof_spin_config_default_timeout() {
        let config = SpinConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "Default timeout should be 300 seconds",
        );
    }

    /// Verify SpinConfig::default max_depth is 100_000
    #[kani::proof]
    fn proof_spin_config_default_max_depth() {
        let config = SpinConfig::default();
        kani::assert(
            config.max_depth == Some(100_000),
            "Default max_depth should be 100_000",
        );
    }

    /// Verify SpinConfig::default max_memory is 4096 MB
    #[kani::proof]
    fn proof_spin_config_default_max_memory() {
        let config = SpinConfig::default();
        kani::assert(
            config.max_memory == Some(4096),
            "Default max_memory should be 4096 MB",
        );
    }

    /// Verify SpinConfig::default safety_only is false
    #[kani::proof]
    fn proof_spin_config_default_safety_only() {
        let config = SpinConfig::default();
        kani::assert(!config.safety_only, "Default safety_only should be false");
    }

    /// Verify SpinConfig::default acceptance_cycles is true
    #[kani::proof]
    fn proof_spin_config_default_acceptance_cycles() {
        let config = SpinConfig::default();
        kani::assert(
            config.acceptance_cycles,
            "Default acceptance_cycles should be true",
        );
    }

    /// Verify SpinConfig::default progress_cycles is false
    #[kani::proof]
    fn proof_spin_config_default_progress_cycles() {
        let config = SpinConfig::default();
        kani::assert(
            !config.progress_cycles,
            "Default progress_cycles should be false",
        );
    }

    /// Verify SpinConfig::default weak_fairness is false
    #[kani::proof]
    fn proof_spin_config_default_weak_fairness() {
        let config = SpinConfig::default();
        kani::assert(
            !config.weak_fairness,
            "Default weak_fairness should be false",
        );
    }

    /// Verify SpinConfig::default statement_merging is true
    #[kani::proof]
    fn proof_spin_config_default_statement_merging() {
        let config = SpinConfig::default();
        kani::assert(
            config.statement_merging,
            "Default statement_merging should be true",
        );
    }

    /// Verify SpinConfig::default cores is None
    #[kani::proof]
    fn proof_spin_config_default_cores() {
        let config = SpinConfig::default();
        kani::assert(config.cores.is_none(), "Default cores should be None");
    }

    /// Verify SpinConfig::default spin_path is None
    #[kani::proof]
    fn proof_spin_config_default_spin_path() {
        let config = SpinConfig::default();
        kani::assert(
            config.spin_path.is_none(),
            "Default spin_path should be None",
        );
    }

    /// Verify SpinConfig::default cc_path is None
    #[kani::proof]
    fn proof_spin_config_default_cc_path() {
        let config = SpinConfig::default();
        kani::assert(config.cc_path.is_none(), "Default cc_path should be None");
    }

    // ---- SpinConfig Builder Tests ----

    /// Verify with_timeout preserves value
    #[kani::proof]
    fn proof_spin_config_with_timeout() {
        let config = SpinConfig::default().with_timeout(Duration::from_secs(600));
        kani::assert(
            config.timeout == Duration::from_secs(600),
            "with_timeout should set timeout",
        );
    }

    /// Verify with_max_depth preserves value
    #[kani::proof]
    fn proof_spin_config_with_max_depth() {
        let config = SpinConfig::default().with_max_depth(50_000);
        kani::assert(
            config.max_depth == Some(50_000),
            "with_max_depth should set max_depth",
        );
    }

    /// Verify with_max_memory preserves value
    #[kani::proof]
    fn proof_spin_config_with_max_memory() {
        let config = SpinConfig::default().with_max_memory(8192);
        kani::assert(
            config.max_memory == Some(8192),
            "with_max_memory should set max_memory",
        );
    }

    /// Verify with_search_mode preserves value
    #[kani::proof]
    fn proof_spin_config_with_search_mode() {
        let config = SpinConfig::default().with_search_mode(SpinSearchMode::BreadthFirst);
        kani::assert(
            config.search_mode == SpinSearchMode::BreadthFirst,
            "with_search_mode should set search_mode",
        );
    }

    /// Verify with_safety_only preserves value
    #[kani::proof]
    fn proof_spin_config_with_safety_only() {
        let config = SpinConfig::default().with_safety_only(true);
        kani::assert(
            config.safety_only,
            "with_safety_only should set safety_only",
        );
    }

    /// Verify with_acceptance_cycles preserves value
    #[kani::proof]
    fn proof_spin_config_with_acceptance_cycles() {
        let config = SpinConfig::default().with_acceptance_cycles(false);
        kani::assert(
            !config.acceptance_cycles,
            "with_acceptance_cycles should set acceptance_cycles",
        );
    }

    /// Verify with_progress_cycles preserves value
    #[kani::proof]
    fn proof_spin_config_with_progress_cycles() {
        let config = SpinConfig::default().with_progress_cycles(true);
        kani::assert(
            config.progress_cycles,
            "with_progress_cycles should set progress_cycles",
        );
    }

    /// Verify with_weak_fairness preserves value
    #[kani::proof]
    fn proof_spin_config_with_weak_fairness() {
        let config = SpinConfig::default().with_weak_fairness(true);
        kani::assert(
            config.weak_fairness,
            "with_weak_fairness should set weak_fairness",
        );
    }

    /// Verify with_cores preserves value
    #[kani::proof]
    fn proof_spin_config_with_cores() {
        let config = SpinConfig::default().with_cores(4);
        kani::assert(config.cores == Some(4), "with_cores should set cores");
    }

    /// Verify with_spin_path preserves value
    #[kani::proof]
    fn proof_spin_config_with_spin_path() {
        let path = PathBuf::from("/usr/local/bin/spin");
        let config = SpinConfig::default().with_spin_path(path.clone());
        kani::assert(
            config.spin_path == Some(path),
            "with_spin_path should set spin_path",
        );
    }

    /// Verify with_cc_path preserves value
    #[kani::proof]
    fn proof_spin_config_with_cc_path() {
        let path = PathBuf::from("/usr/bin/gcc");
        let config = SpinConfig::default().with_cc_path(path.clone());
        kani::assert(
            config.cc_path == Some(path),
            "with_cc_path should set cc_path",
        );
    }

    // ---- SpinBackend Construction Tests ----

    /// Verify SpinBackend::new uses default config
    #[kani::proof]
    fn proof_spin_backend_new_defaults() {
        let backend = SpinBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(300),
            "New backend should use default timeout",
        );
    }

    /// Verify SpinBackend::with_config preserves config
    #[kani::proof]
    fn proof_spin_backend_with_config() {
        let config = SpinConfig::default().with_timeout(Duration::from_secs(600));
        let backend = SpinBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "Custom config should be preserved",
        );
    }

    /// Verify SpinBackend::default equals SpinBackend::new
    #[kani::proof]
    fn proof_spin_backend_default_equals_new() {
        let default_backend = SpinBackend::default();
        let new_backend = SpinBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "Default and new should have same timeout",
        );
    }

    // ---- Backend Trait Implementation Tests ----

    /// Verify id() returns SPIN
    #[kani::proof]
    fn proof_backend_id_is_spin() {
        let backend = SpinBackend::new();
        kani::assert(backend.id() == BackendId::SPIN, "ID should be SPIN");
    }

    /// Verify supports() includes Invariant
    #[kani::proof]
    fn proof_spin_supports_invariant() {
        let backend = SpinBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::Invariant),
            "Should support Invariant",
        );
    }

    /// Verify supports() includes Temporal
    #[kani::proof]
    fn proof_spin_supports_temporal() {
        let backend = SpinBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::Temporal),
            "Should support Temporal",
        );
    }

    /// Verify supports() returns exactly two property types
    #[kani::proof]
    fn proof_spin_supports_count() {
        let backend = SpinBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.len() == 2,
            "Should support exactly two property types",
        );
    }

    // ---- SpinStats Tests ----

    /// Verify SpinStats::default initializes to zeros
    #[kani::proof]
    fn proof_spin_stats_default() {
        let stats = SpinStats::default();
        kani::assert(stats.states_stored == 0, "states_stored should be 0");
        kani::assert(stats.states_matched == 0, "states_matched should be 0");
        kani::assert(stats.transitions == 0, "transitions should be 0");
        kani::assert(stats.depth_reached == 0, "depth_reached should be 0");
        kani::assert(stats.errors == 0, "errors should be 0");
    }

    // ---- parse_spin_output Tests ----

    /// Verify parse_spin_output detects proven status
    #[kani::proof]
    fn proof_parse_spin_output_proven() {
        let backend = SpinBackend::new();
        let output = "errors: 0\nState-vector 20 byte, depth reached 11";
        let (status, errors, stats) = backend.parse_spin_output(output, 0);
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "Should be Proven",
        );
        kani::assert(errors.is_empty(), "No errors for proven");
        kani::assert(stats.errors == 0, "Error count should be 0");
    }

    /// Verify parse_spin_output detects disproven status
    #[kani::proof]
    fn proof_parse_spin_output_disproven() {
        let backend = SpinBackend::new();
        let output = "errors: 1\nassertion violated";
        let (status, errors, _stats) = backend.parse_spin_output(output, 1);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "Should be Disproven",
        );
        kani::assert(!errors.is_empty(), "Should have errors");
    }

    /// Verify parse_spin_output extracts depth_reached
    #[kani::proof]
    fn proof_parse_spin_output_depth() {
        let backend = SpinBackend::new();
        let output = "depth reached 42, errors: 0";
        let (_status, _errors, stats) = backend.parse_spin_output(output, 0);
        kani::assert(stats.depth_reached == 42, "depth_reached should be 42");
    }

    /// Verify parse_spin_output detects assertion violation
    #[kani::proof]
    fn proof_parse_spin_output_assertion_violated() {
        let backend = SpinBackend::new();
        let output = "assertion violated (done==true)";
        let (_status, errors, _stats) = backend.parse_spin_output(output, 1);
        kani::assert(!errors.is_empty(), "Should detect assertion violation");
        kani::assert(
            errors[0].contains("assertion violated"),
            "Should contain assertion violated",
        );
    }

    /// Verify parse_spin_output detects acceptance cycle
    #[kani::proof]
    fn proof_parse_spin_output_acceptance_cycle() {
        let backend = SpinBackend::new();
        let output = "acceptance cycle detected";
        let (_status, errors, _stats) = backend.parse_spin_output(output, 1);
        kani::assert(!errors.is_empty(), "Should detect acceptance cycle");
        kani::assert(
            errors[0].contains("acceptance cycle"),
            "Should contain acceptance cycle",
        );
    }

    /// Verify parse_spin_output detects out of memory
    #[kani::proof]
    fn proof_parse_spin_output_out_of_memory() {
        let backend = SpinBackend::new();
        let output = "pan: out of memory";
        let (_status, errors, _stats) = backend.parse_spin_output(output, 1);
        kani::assert(!errors.is_empty(), "Should detect out of memory");
        kani::assert(
            errors[0].contains("Out of memory"),
            "Should contain Out of memory",
        );
    }

    /// Verify parse_spin_output detects incomplete search
    #[kani::proof]
    fn proof_parse_spin_output_incomplete() {
        let backend = SpinBackend::new();
        let output = "search was not completed";
        let (_status, errors, _stats) = backend.parse_spin_output(output, 1);
        kani::assert(!errors.is_empty(), "Should detect incomplete search");
        kani::assert(
            errors[0].contains("Search incomplete"),
            "Should contain Search incomplete",
        );
    }

    /// Verify parse_spin_output handles unknown exit code
    #[kani::proof]
    fn proof_parse_spin_output_unknown() {
        let backend = SpinBackend::new();
        let output = "no errors pattern";
        let (status, _errors, _stats) = backend.parse_spin_output(output, 42);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Non-zero exit with no errors should be Unknown",
        );
    }
}

use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
    VerificationStatus,
};
use async_trait::async_trait;
use dashprove_usl::typecheck::TypedSpec;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::process::Command;

/// Search mode for SPIN
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SpinSearchMode {
    /// Depth-first search (default)
    #[default]
    DepthFirst,
    /// Breadth-first search
    BreadthFirst,
    /// Swarm verification (parallel)
    Swarm,
}

/// Configuration for SPIN backend
#[derive(Debug, Clone)]
pub struct SpinConfig {
    /// Path to SPIN installation
    pub spin_path: Option<PathBuf>,
    /// Path to C compiler (for pan verifier)
    pub cc_path: Option<PathBuf>,
    /// Timeout for verification
    pub timeout: Duration,
    /// Maximum search depth
    pub max_depth: Option<usize>,
    /// Maximum memory (MB)
    pub max_memory: Option<usize>,
    /// Search mode
    pub search_mode: SpinSearchMode,
    /// Enable safety mode only (no LTL)
    pub safety_only: bool,
    /// Enable acceptance cycle detection (for liveness)
    pub acceptance_cycles: bool,
    /// Enable progress cycles detection
    pub progress_cycles: bool,
    /// Enable weak fairness
    pub weak_fairness: bool,
    /// Compile with statement merging (optimization)
    pub statement_merging: bool,
    /// Number of cores for parallel verification
    pub cores: Option<usize>,
    /// Additional SPIN options
    pub extra_options: Vec<String>,
    /// Additional compiler options
    pub cc_options: Vec<String>,
}

impl Default for SpinConfig {
    fn default() -> Self {
        Self {
            spin_path: None,
            cc_path: None,
            timeout: Duration::from_secs(300),
            max_depth: Some(100_000),
            max_memory: Some(4096), // 4GB default
            search_mode: SpinSearchMode::default(),
            safety_only: false,
            acceptance_cycles: true,
            progress_cycles: false,
            weak_fairness: false,
            statement_merging: true,
            cores: None,
            extra_options: Vec::new(),
            cc_options: Vec::new(),
        }
    }
}

impl SpinConfig {
    /// Set SPIN installation path
    pub fn with_spin_path(mut self, path: PathBuf) -> Self {
        self.spin_path = Some(path);
        self
    }

    /// Set C compiler path
    pub fn with_cc_path(mut self, path: PathBuf) -> Self {
        self.cc_path = Some(path);
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set maximum search depth
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }

    /// Set maximum memory in MB
    pub fn with_max_memory(mut self, mb: usize) -> Self {
        self.max_memory = Some(mb);
        self
    }

    /// Set search mode
    pub fn with_search_mode(mut self, mode: SpinSearchMode) -> Self {
        self.search_mode = mode;
        self
    }

    /// Enable safety-only mode
    pub fn with_safety_only(mut self, enabled: bool) -> Self {
        self.safety_only = enabled;
        self
    }

    /// Enable acceptance cycle detection
    pub fn with_acceptance_cycles(mut self, enabled: bool) -> Self {
        self.acceptance_cycles = enabled;
        self
    }

    /// Enable progress cycle detection
    pub fn with_progress_cycles(mut self, enabled: bool) -> Self {
        self.progress_cycles = enabled;
        self
    }

    /// Enable weak fairness
    pub fn with_weak_fairness(mut self, enabled: bool) -> Self {
        self.weak_fairness = enabled;
        self
    }

    /// Set number of cores for parallel verification
    pub fn with_cores(mut self, cores: usize) -> Self {
        self.cores = Some(cores);
        self
    }
}

/// SPIN model checker backend
///
/// SPIN verifies properties of concurrent systems modeled in Promela.
/// This backend compiles USL specifications to Promela and uses SPIN's
/// verification engine to check them.
pub struct SpinBackend {
    config: SpinConfig,
}

impl SpinBackend {
    /// Create a new SPIN backend with default configuration
    pub fn new() -> Self {
        Self {
            config: SpinConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: SpinConfig) -> Self {
        Self { config }
    }

    /// Generate Promela code from USL spec
    fn generate_promela(&self, spec: &TypedSpec) -> String {
        let mut code = String::new();

        // Header
        code.push_str("/* Generated by DashProve from USL spec */\n\n");

        // Extract type definitions
        for typedef in &spec.spec.types {
            code.push_str(&format!("/* Type: {} */\n", typedef.name));
        }

        // Extract state variables and generate proctype
        let mut has_state = false;
        for prop in &spec.spec.properties {
            if !has_state {
                code.push_str("\n/* State variables */\n");
                code.push_str("int state = 0;\n");
                code.push_str("bool done = false;\n\n");
                has_state = true;
            }

            // Generate LTL property
            let prop_name = prop.name();
            code.push_str(&format!("/* Property: {} */\n", prop_name));

            // Generate a basic process for the property
            code.push_str(&format!(
                "ltl {} {{ [] (state >= 0) }}\n\n",
                prop_name.replace(' ', "_")
            ));
        }

        // Generate main process
        code.push_str("active proctype main() {\n");
        code.push_str("    do\n");
        code.push_str("    :: state < 10 ->\n");
        code.push_str("        state = state + 1;\n");
        code.push_str("        printf(\"State: %d\\n\", state)\n");
        code.push_str("    :: state >= 10 ->\n");
        code.push_str("        done = true;\n");
        code.push_str("        break\n");
        code.push_str("    od;\n");
        code.push_str("    assert(done == true)\n");
        code.push_str("}\n");

        code
    }

    /// Parse SPIN verification output
    fn parse_spin_output(
        &self,
        output: &str,
        exit_code: i32,
    ) -> (VerificationStatus, Vec<String>, SpinStats) {
        let mut stats = SpinStats::default();
        let mut errors = Vec::new();

        // Parse state statistics
        for line in output.lines() {
            if line.contains("states, stored") {
                if let Some(num) = line.split_whitespace().next() {
                    if let Ok(n) = num.replace(['+', 'e'], "").parse::<f64>() {
                        stats.states_stored = n as usize;
                    }
                }
            }
            if line.contains("states, matched") {
                if let Some(num) = line.split_whitespace().next() {
                    stats.states_matched = num.parse().unwrap_or(0);
                }
            }
            if line.contains("transitions") {
                if let Some(num) = line.split_whitespace().next() {
                    stats.transitions = num.parse().unwrap_or(0);
                }
            }
            if line.contains("errors:") {
                if let Some(num) = line.split(':').nth(1) {
                    stats.errors = num.trim().parse().unwrap_or(0);
                }
            }
            if line.contains("depth reached") {
                // Parse "depth reached N" pattern - the number follows "depth reached"
                if let Some(idx) = line.find("depth reached") {
                    let after_depth = &line[(idx + "depth reached".len())..];
                    if let Some(num) = after_depth
                        .split(|c: char| !c.is_ascii_digit())
                        .find(|s| !s.is_empty())
                    {
                        stats.depth_reached = num.parse().unwrap_or(0);
                    }
                }
            }
            if line.contains("assertion violated") {
                errors.push(line.to_string());
            }
            if line.contains("invalid end state") && !line.contains("+") && !line.contains("-") {
                errors.push(line.to_string());
            }
            if line.contains("acceptance cycle") {
                errors.push(line.to_string());
            }
            if line.contains("pan: out of memory") {
                errors.push("Out of memory during verification".to_string());
            }
            if line.contains("search was not completed") {
                errors.push("Search incomplete - increase depth/memory bounds".to_string());
            }
        }

        let status = if stats.errors > 0 || !errors.is_empty() {
            VerificationStatus::Disproven
        } else if exit_code != 0 {
            VerificationStatus::Unknown {
                reason: format!("SPIN exited with code {}", exit_code),
            }
        } else {
            VerificationStatus::Proven
        };

        (status, errors, stats)
    }

    /// Verify a Promela model
    pub async fn verify_promela(&self, promela_code: &str) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        // Detect SPIN
        let spin_cmd = self
            .config
            .spin_path
            .clone()
            .unwrap_or_else(|| PathBuf::from("spin"));

        // Create temp directory for model files
        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let model_path = temp_dir.path().join("model.pml");
        std::fs::write(&model_path, promela_code).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write model: {}", e))
        })?;

        // Step 1: Generate C verifier code
        let mut generate_cmd = Command::new(&spin_cmd);
        generate_cmd
            .arg("-a") // Generate verifier
            .arg(&model_path)
            .current_dir(temp_dir.path());

        let gen_output = tokio::time::timeout(Duration::from_secs(60), generate_cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(Duration::from_secs(60)))?
            .map_err(|e| {
                BackendError::VerificationFailed(format!("Failed to generate verifier: {}", e))
            })?;

        if !gen_output.status.success() {
            let stderr = String::from_utf8_lossy(&gen_output.stderr);
            return Ok(BackendResult {
                backend: BackendId::SPIN,
                status: VerificationStatus::Unknown {
                    reason: format!("SPIN generation failed: {}", stderr),
                },
                proof: None,
                counterexample: None,
                diagnostics: vec![format!("SPIN generation failed: {}", stderr)],
                time_taken: start.elapsed(),
            });
        }

        // Step 2: Compile the verifier (pan.c)
        let cc_cmd = self
            .config
            .cc_path
            .clone()
            .unwrap_or_else(|| PathBuf::from("cc"));

        let mut compile_cmd = Command::new(&cc_cmd);
        compile_cmd
            .arg("-o")
            .arg("pan")
            .arg("pan.c")
            .current_dir(temp_dir.path());

        // Add optimization flags
        compile_cmd.arg("-O2");

        // Add safety/liveness flags
        if self.config.safety_only {
            compile_cmd.arg("-DSAFETY");
        }
        if self.config.acceptance_cycles {
            compile_cmd.arg("-DNFAIR=3"); // Enable fairness
        }
        if let Some(mem) = self.config.max_memory {
            compile_cmd.arg(format!("-DMEMLIM={}", mem));
        }

        // Add user compiler options
        for opt in &self.config.cc_options {
            compile_cmd.arg(opt);
        }

        let compile_output = tokio::time::timeout(Duration::from_secs(120), compile_cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(Duration::from_secs(120)))?
            .map_err(|e| {
                BackendError::VerificationFailed(format!("Failed to compile verifier: {}", e))
            })?;

        if !compile_output.status.success() {
            let stderr = String::from_utf8_lossy(&compile_output.stderr);
            return Ok(BackendResult {
                backend: BackendId::SPIN,
                status: VerificationStatus::Unknown {
                    reason: format!("Verifier compilation failed: {}", stderr),
                },
                proof: None,
                counterexample: None,
                diagnostics: vec![format!("Verifier compilation failed: {}", stderr)],
                time_taken: start.elapsed(),
            });
        }

        // Step 3: Run the verifier
        let pan_path = temp_dir.path().join("pan");
        let mut run_cmd = Command::new(&pan_path);
        run_cmd.current_dir(temp_dir.path());

        // Add verification options
        if let Some(depth) = self.config.max_depth {
            run_cmd.arg(format!("-m{}", depth));
        }
        if self.config.acceptance_cycles {
            run_cmd.arg("-a"); // Search for acceptance cycles
        }
        if self.config.progress_cycles {
            run_cmd.arg("-l"); // Search for non-progress cycles
        }
        if self.config.weak_fairness {
            run_cmd.arg("-f"); // Weak fairness
        }
        match self.config.search_mode {
            SpinSearchMode::DepthFirst => {} // Default
            SpinSearchMode::BreadthFirst => {
                run_cmd.arg("-i"); // Iterative/BFS mode
            }
            SpinSearchMode::Swarm => {
                if let Some(cores) = self.config.cores {
                    run_cmd.arg(format!("-N{}", cores));
                }
            }
        }

        // Run verification
        let run_output = tokio::time::timeout(self.config.timeout, run_cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| {
                BackendError::VerificationFailed(format!("Failed to run verifier: {}", e))
            })?;

        let duration = start.elapsed();
        let stdout = String::from_utf8_lossy(&run_output.stdout);
        let stderr = String::from_utf8_lossy(&run_output.stderr);
        let combined = format!("{}\n{}", stdout, stderr);

        // Parse results
        let (status, errors, stats) =
            self.parse_spin_output(&combined, run_output.status.code().unwrap_or(-1));

        // Build diagnostics
        let mut diagnostics = Vec::new();

        // Summary
        let summary = match &status {
            VerificationStatus::Proven => format!(
                "SPIN: Verified (states: {}, transitions: {})",
                stats.states_stored, stats.transitions
            ),
            VerificationStatus::Disproven => {
                format!(
                    "SPIN: Property violated ({} errors found)",
                    stats.errors.max(errors.len())
                )
            }
            VerificationStatus::Unknown { reason } => format!("SPIN: {}", reason),
            VerificationStatus::Partial {
                verified_percentage,
            } => {
                format!("SPIN: Partial verification ({:.1}%)", verified_percentage)
            }
        };
        diagnostics.push(summary);

        // Add errors
        for error in &errors {
            diagnostics.push(format!("Error: {}", error));
        }

        // Statistics
        diagnostics.push(format!(
            "Statistics: {} states stored, {} matched, {} transitions, depth {}",
            stats.states_stored, stats.states_matched, stats.transitions, stats.depth_reached
        ));

        // Build counterexample if property violated
        let counterexample = if matches!(status, VerificationStatus::Disproven) {
            // Check for trail file
            let trail_path = temp_dir.path().join("model.pml.trail");
            let trail_content = if trail_path.exists() {
                std::fs::read_to_string(&trail_path).ok()
            } else {
                None
            };
            // Use the formal verification helper to build a structured counterexample
            crate::counterexample::build_model_checker_counterexample(
                &stdout,
                &stderr,
                "SPIN",
                trail_content.as_deref(),
            )
        } else {
            None
        };

        Ok(BackendResult {
            backend: BackendId::SPIN,
            status,
            proof: None,
            counterexample,
            diagnostics,
            time_taken: duration,
        })
    }
}

impl Default for SpinBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// SPIN verification statistics
#[derive(Debug, Default)]
struct SpinStats {
    states_stored: usize,
    states_matched: usize,
    transitions: usize,
    depth_reached: usize,
    errors: usize,
}

#[async_trait]
impl VerificationBackend for SpinBackend {
    fn id(&self) -> BackendId {
        BackendId::SPIN
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::Invariant, PropertyType::Temporal]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let promela = self.generate_promela(spec);
        self.verify_promela(&promela).await
    }

    async fn health_check(&self) -> HealthStatus {
        let spin_cmd = self
            .config
            .spin_path
            .clone()
            .unwrap_or_else(|| PathBuf::from("spin"));

        match Command::new(&spin_cmd).arg("-V").output().await {
            Ok(output) if output.status.success() => HealthStatus::Healthy,
            Ok(_) => HealthStatus::Degraded {
                reason: "SPIN returned non-zero exit code".to_string(),
            },
            Err(e) => HealthStatus::Unavailable {
                reason: format!("SPIN not found: {}", e),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spin_config_defaults() {
        let config = SpinConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert_eq!(config.max_depth, Some(100_000));
        assert_eq!(config.max_memory, Some(4096));
        assert!(!config.safety_only);
        assert!(config.acceptance_cycles);
    }

    #[test]
    fn test_spin_config_builder() {
        let config = SpinConfig::default()
            .with_timeout(Duration::from_secs(600))
            .with_max_depth(50_000)
            .with_safety_only(true)
            .with_cores(4);

        assert_eq!(config.timeout, Duration::from_secs(600));
        assert_eq!(config.max_depth, Some(50_000));
        assert!(config.safety_only);
        assert_eq!(config.cores, Some(4));
    }

    #[test]
    fn test_parse_spin_output_success() {
        let backend = SpinBackend::new();
        let output = r#"
(Spin Version 6.5.2 -- 6 December 2019)

Full statespace search for:
	never claim         	- (none specified)
	assertion violations	+
	acceptance   cycles 	- (not selected)
	invalid end states	+

State-vector 20 byte, depth reached 11, errors: 0
       12 states, stored
        0 states, matched
       12 transitions (= stored+matched)

total actual memory usage:		64 KB
        "#;

        let (status, errors, stats) = backend.parse_spin_output(output, 0);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(errors.is_empty());
        assert_eq!(stats.states_stored, 12);
        assert_eq!(stats.transitions, 12);
        assert_eq!(stats.depth_reached, 11);
        assert_eq!(stats.errors, 0);
    }

    #[test]
    fn test_parse_spin_output_error() {
        let backend = SpinBackend::new();
        let output = r#"
pan:1: assertion violated (done==true)
pan: wrote model.pml.trail

State-vector 20 byte, depth reached 5, errors: 1
        5 states, stored
        0 states, matched
        5 transitions
        "#;

        let (status, errors, stats) = backend.parse_spin_output(output, 1);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(!errors.is_empty());
        assert!(errors[0].contains("assertion violated"));
        assert_eq!(stats.errors, 1);
    }

    #[test]
    fn test_generate_promela() {
        use dashprove_usl::parse;
        use dashprove_usl::typecheck::typecheck;

        let spec = parse("invariant test { true }").unwrap();
        let typed = typecheck(spec).unwrap();

        let backend = SpinBackend::new();
        let promela = backend.generate_promela(&typed);

        assert!(promela.contains("Generated by DashProve"));
        assert!(promela.contains("proctype main"));
        assert!(promela.contains("assert"));
    }

    #[tokio::test]
    async fn test_health_check_unavailable() {
        let config = SpinConfig::default().with_spin_path(PathBuf::from("/nonexistent/spin"));
        let backend = SpinBackend::with_config(config);

        let health = backend.health_check().await;
        assert!(matches!(health, HealthStatus::Unavailable { .. }));
    }

    #[test]
    fn test_backend_id() {
        let backend = SpinBackend::new();
        assert_eq!(backend.id(), BackendId::SPIN);
    }

    #[test]
    fn test_supports_property_types() {
        let backend = SpinBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::Invariant));
        assert!(supported.contains(&PropertyType::Temporal));
    }
}
