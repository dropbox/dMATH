//! Flux backend for refinement type checking in Rust
//!
//! This backend runs Flux to verify refinement type annotations in Rust code.
//! Flux extends Rust's type system with refinement types, allowing specification
//! of rich invariants on values (e.g., "this integer is positive").

// =============================================
// Kani Proofs for Flux Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- FluxMode Enum Tests ----

    /// Verify FluxMode::default returns Full
    #[kani::proof]
    fn proof_flux_mode_default() {
        let mode = FluxMode::default();
        kani::assert(
            matches!(mode, FluxMode::Full),
            "default mode should be Full",
        );
    }

    /// Verify FluxMode variants are distinct
    #[kani::proof]
    fn proof_flux_mode_variants() {
        kani::assert(
            FluxMode::Full != FluxMode::TypeCheckOnly,
            "Full != TypeCheckOnly",
        );
        kani::assert(
            FluxMode::Full != FluxMode::FunctionOnly,
            "Full != FunctionOnly",
        );
        kani::assert(
            FluxMode::TypeCheckOnly != FluxMode::FunctionOnly,
            "TypeCheckOnly != FunctionOnly",
        );
    }

    // ---- FluxSolver Enum Tests ----

    /// Verify FluxSolver::default returns Z3
    #[kani::proof]
    fn proof_flux_solver_default() {
        let solver = FluxSolver::default();
        kani::assert(
            matches!(solver, FluxSolver::Z3),
            "default solver should be Z3",
        );
    }

    /// Verify FluxSolver variants are distinct
    #[kani::proof]
    fn proof_flux_solver_variants() {
        kani::assert(FluxSolver::Z3 != FluxSolver::CVC5, "Z3 != CVC5");
    }

    // ---- FluxConfig Default Tests ----

    /// Verify FluxConfig::default sets baseline values
    #[kani::proof]
    fn proof_flux_config_defaults() {
        let config = FluxConfig::default();
        kani::assert(
            config.crate_path.is_none(),
            "crate_path should default to None",
        );
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "timeout should default to 300 seconds",
        );
        kani::assert(
            config.mode == FluxMode::Full,
            "mode should default to FluxMode::Full",
        );
        kani::assert(
            config.solver == FluxSolver::Z3,
            "solver should default to FluxSolver::Z3",
        );
        kani::assert(
            config.target_functions.is_empty(),
            "targets should default empty",
        );
        kani::assert(!config.verbose, "verbose should default to false");
        kani::assert(
            config.smt_timeout_ms == 30_000,
            "smt_timeout_ms should default to 30_000",
        );
        kani::assert(
            config.flux_path.is_none(),
            "flux_path should default to None",
        );
    }

    // ---- FluxConfig Builder Tests ----

    /// Verify builder helpers update FluxConfig fields
    #[kani::proof]
    fn proof_flux_config_builder_updates() {
        let config = FluxConfig::default()
            .with_crate_path(PathBuf::from("/crate"))
            .with_timeout(Duration::from_secs(120))
            .with_mode(FluxMode::TypeCheckOnly)
            .with_solver(FluxSolver::CVC5)
            .with_target_functions(vec!["foo".into(), "bar".into()])
            .with_verbose(true)
            .with_smt_timeout(5_000)
            .with_flux_path(PathBuf::from("/opt/flux"));

        kani::assert(
            config.crate_path == Some(PathBuf::from("/crate")),
            "with_crate_path should set crate_path",
        );
        kani::assert(
            config.timeout == Duration::from_secs(120),
            "with_timeout should set timeout",
        );
        kani::assert(
            config.mode == FluxMode::TypeCheckOnly,
            "with_mode should set verification mode",
        );
        kani::assert(
            config.solver == FluxSolver::CVC5,
            "with_solver should set solver",
        );
        kani::assert(
            config.target_functions == vec!["foo".to_string(), "bar".to_string()],
            "with_target_functions should set targets",
        );
        kani::assert(config.verbose, "with_verbose should enable verbose");
        kani::assert(
            config.smt_timeout_ms == 5_000,
            "with_smt_timeout should set SMT timeout",
        );
        kani::assert(
            config.flux_path == Some(PathBuf::from("/opt/flux")),
            "with_flux_path should set flux_path",
        );
    }

    // ---- FluxBackend Construction Tests ----

    /// Verify FluxBackend::new uses default configuration
    #[kani::proof]
    fn proof_flux_backend_new_defaults() {
        let backend = FluxBackend::new();
        kani::assert(
            backend.config.mode == FluxMode::Full,
            "new backend should default to FluxMode::Full",
        );
        kani::assert(
            backend.config.solver == FluxSolver::Z3,
            "new backend should default to FluxSolver::Z3",
        );
        kani::assert(
            backend.config.timeout == Duration::from_secs(300),
            "new backend should default timeout to 300 seconds",
        );
    }

    /// Verify FluxBackend::default matches FluxBackend::new
    #[kani::proof]
    fn proof_flux_backend_default_equals_new() {
        let default_backend = FluxBackend::default();
        let new_backend = FluxBackend::new();
        kani::assert(
            default_backend.config.mode == new_backend.config.mode,
            "default and new should share mode",
        );
        kani::assert(
            default_backend.config.solver == new_backend.config.solver,
            "default and new should share solver",
        );
    }

    /// Verify FluxBackend::with_config preserves custom configuration
    #[kani::proof]
    fn proof_flux_backend_with_config() {
        let config = FluxConfig {
            crate_path: Some(PathBuf::from("/workdir")),
            timeout: Duration::from_secs(30),
            mode: FluxMode::FunctionOnly,
            solver: FluxSolver::CVC5,
            target_functions: vec!["target_fn".into()],
            verbose: true,
            smt_timeout_ms: 1000,
            flux_path: Some(PathBuf::from("/bin/flux")),
        };
        let backend = FluxBackend::with_config(config);
        kani::assert(
            backend.config.crate_path == Some(PathBuf::from("/workdir")),
            "with_config should preserve crate_path",
        );
        kani::assert(
            backend.config.mode == FluxMode::FunctionOnly,
            "with_config should preserve mode",
        );
        kani::assert(
            backend.config.solver == FluxSolver::CVC5,
            "with_config should preserve solver",
        );
        kani::assert(
            backend.config.target_functions == vec!["target_fn".to_string()],
            "with_config should preserve target functions",
        );
        kani::assert(
            backend.config.smt_timeout_ms == 1000,
            "with_config should preserve SMT timeout",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify id() returns BackendId::Flux
    #[kani::proof]
    fn proof_flux_backend_id() {
        let backend = FluxBackend::new();
        kani::assert(
            backend.id() == BackendId::Flux,
            "FluxBackend id should be BackendId::Flux",
        );
    }

    /// Verify supports() includes Contract and Invariant
    #[kani::proof]
    fn proof_flux_backend_supports() {
        let backend = FluxBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::Contract),
            "supports should include Contract",
        );
        kani::assert(
            supported.contains(&PropertyType::Invariant),
            "supports should include Invariant",
        );
    }

    // ---- Flux Output Parsing Tests ----

    /// Verify parse_flux_output marks success when verified lines found
    #[kani::proof]
    fn proof_parse_flux_output_success() {
        let backend = FluxBackend::new();
        let output = "Checking crate...\nfunction `abs_positive` verified\nVerified 3 functions";
        let (status, findings) = backend.parse_flux_output(output, true);
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "parse_flux_output should mark Proven for successful output",
        );
        kani::assert(findings.is_empty(), "findings should be empty on success");
    }

    /// Verify parse_flux_output detects refinement errors
    #[kani::proof]
    fn proof_parse_flux_output_error() {
        let backend = FluxBackend::new();
        let output = "error[FLUX]: refinement type error in `foo`\n  --> src/lib.rs:10:5";
        let (status, findings) = backend.parse_flux_output(output, false);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "refinement errors should mark status as Disproven",
        );
        kani::assert(
            !findings.is_empty(),
            "refinement errors should produce findings entries",
        );
    }

    /// Verify parse_flux_output reports missing installation as Unknown
    #[kani::proof]
    fn proof_parse_flux_output_not_installed() {
        let backend = FluxBackend::new();
        let output = "error: no such command: `flux`";
        let (status, findings) = backend.parse_flux_output(output, false);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "missing flux should produce Unknown status",
        );
        kani::assert(
            findings.is_empty(),
            "installation errors should have no findings",
        );
    }

    /// Verify parse_flux_output handles no annotations as Unknown
    #[kani::proof]
    fn proof_parse_flux_output_no_annotations() {
        let backend = FluxBackend::new();
        let (status, _) =
            backend.parse_flux_output("checking crate with no flux annotations", true);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "lack of annotations should produce Unknown status",
        );
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

/// Flux verification mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum FluxMode {
    /// Full verification with all checks
    #[default]
    Full,
    /// Type checking only (no SMT verification)
    TypeCheckOnly,
    /// Check specific function(s)
    FunctionOnly,
}

/// SMT solver backend for Flux
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum FluxSolver {
    /// Z3 SMT solver (default)
    #[default]
    Z3,
    /// CVC5 SMT solver
    CVC5,
}

/// Configuration for Flux backend
#[derive(Debug, Clone)]
pub struct FluxConfig {
    /// Path to the crate to analyze
    pub crate_path: Option<PathBuf>,
    /// Timeout for analysis
    pub timeout: Duration,
    /// Verification mode
    pub mode: FluxMode,
    /// SMT solver backend
    pub solver: FluxSolver,
    /// Specific functions to check (if mode is FunctionOnly)
    pub target_functions: Vec<String>,
    /// Enable verbose output
    pub verbose: bool,
    /// Maximum SMT timeout per query (in milliseconds)
    pub smt_timeout_ms: u32,
    /// Path to flux binary (if not using rustup component)
    pub flux_path: Option<PathBuf>,
}

impl Default for FluxConfig {
    fn default() -> Self {
        Self {
            crate_path: None,
            timeout: Duration::from_secs(300),
            mode: FluxMode::default(),
            solver: FluxSolver::default(),
            target_functions: Vec::new(),
            verbose: false,
            smt_timeout_ms: 30000,
            flux_path: None,
        }
    }
}

impl FluxConfig {
    /// Set the crate path
    pub fn with_crate_path(mut self, path: PathBuf) -> Self {
        self.crate_path = Some(path);
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set verification mode
    pub fn with_mode(mut self, mode: FluxMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set SMT solver
    pub fn with_solver(mut self, solver: FluxSolver) -> Self {
        self.solver = solver;
        self
    }

    /// Set target functions (for FunctionOnly mode)
    pub fn with_target_functions(mut self, functions: Vec<String>) -> Self {
        self.target_functions = functions;
        self
    }

    /// Enable verbose output
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Set SMT timeout per query
    pub fn with_smt_timeout(mut self, timeout_ms: u32) -> Self {
        self.smt_timeout_ms = timeout_ms;
        self
    }

    /// Set custom flux binary path
    pub fn with_flux_path(mut self, path: PathBuf) -> Self {
        self.flux_path = Some(path);
        self
    }
}

/// Flux verification backend for refinement types
///
/// Flux brings refinement types to Rust, allowing you to:
/// - Express rich invariants on function inputs/outputs
/// - Verify array bounds at compile time
/// - Ensure numeric invariants (positive, non-zero, etc.)
/// - Express ownership and aliasing constraints
///
/// # Requirements
///
/// Install Flux via rustup (experimental component):
/// ```bash
/// rustup +nightly component add flux
/// ```
///
/// Or build from source:
/// ```bash
/// git clone https://github.com/flux-rs/flux
/// cd flux
/// cargo build --release
/// ```
///
/// # Example Annotations
///
/// ```rust,ignore
/// #[flux::sig(fn(x: i32{v: v > 0}) -> i32{v: v > 0})]
/// fn abs_positive(x: i32) -> i32 {
///     x
/// }
/// ```
pub struct FluxBackend {
    config: FluxConfig,
}

impl FluxBackend {
    /// Create a new Flux backend with default configuration
    pub fn new() -> Self {
        Self {
            config: FluxConfig::default(),
        }
    }

    /// Create a new Flux backend with custom configuration
    pub fn with_config(config: FluxConfig) -> Self {
        Self { config }
    }

    /// Run Flux verification on a crate
    pub async fn analyze_crate(&self, crate_path: &Path) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        // Determine flux command
        let flux_cmd = self
            .config
            .flux_path
            .clone()
            .unwrap_or_else(|| PathBuf::from("cargo"));

        // Build command arguments
        let mut args = Vec::new();

        // If using cargo, invoke cargo-flux
        if flux_cmd.to_string_lossy().contains("cargo") {
            args.push("flux".to_string());
        }

        args.push("check".to_string());

        // Add verbose flag
        if self.config.verbose {
            args.push("--verbose".to_string());
        }

        // Add solver configuration via environment
        let solver_str = match self.config.solver {
            FluxSolver::Z3 => "z3",
            FluxSolver::CVC5 => "cvc5",
        };

        // Build the command
        let mut cmd = Command::new(&flux_cmd);
        cmd.args(&args)
            .current_dir(crate_path)
            .env("FLUX_SOLVER", solver_str)
            .env("FLUX_SMT_TIMEOUT", self.config.smt_timeout_ms.to_string());

        // Add mode-specific environment
        match self.config.mode {
            FluxMode::TypeCheckOnly => {
                cmd.env("FLUX_NO_SMT", "1");
            }
            FluxMode::FunctionOnly if !self.config.target_functions.is_empty() => {
                cmd.env("FLUX_FUNCTIONS", self.config.target_functions.join(","));
            }
            _ => {}
        }

        let output = tokio::time::timeout(self.config.timeout, cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| BackendError::VerificationFailed(format!("Failed to run Flux: {}", e)))?;

        let duration = start.elapsed();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined = format!("{}\n{}", stdout, stderr);

        // Parse Flux output
        let (status, findings) = self.parse_flux_output(&combined, output.status.success());

        // Build diagnostics
        let mut diagnostics = Vec::new();

        // Summary
        let summary = match &status {
            VerificationStatus::Proven => {
                format!("Flux: All refinement types verified with {}", solver_str)
            }
            VerificationStatus::Disproven => {
                format!("Flux: {} refinement type error(s) found", findings.len())
            }
            VerificationStatus::Partial {
                verified_percentage,
            } => {
                format!(
                    "Flux: {:.1}% of refinement types verified",
                    verified_percentage
                )
            }
            VerificationStatus::Unknown { reason } => format!("Flux: {}", reason),
        };
        diagnostics.push(summary);
        diagnostics.extend(findings.clone());

        // Build counterexample if errors found
        let counterexample = if !findings.is_empty() {
            Some(StructuredCounterexample::from_raw(combined.clone()))
        } else {
            None
        };

        Ok(BackendResult {
            backend: BackendId::Flux,
            status,
            proof: None,
            counterexample,
            diagnostics,
            time_taken: duration,
        })
    }

    fn parse_flux_output(&self, output: &str, success: bool) -> (VerificationStatus, Vec<String>) {
        let mut findings = Vec::new();
        let mut error_count = 0;
        let mut verified_count = 0;

        // Parse Flux output for errors and verification results
        for line in output.lines() {
            // Flux error patterns:
            // - "error[FLUX]: refinement type error"
            // - "error: unsatisfiable refinement"
            // - "error: cannot verify"
            // - "refinement type mismatch"
            if line.contains("error[FLUX]")
                || line.contains("refinement type error")
                || line.contains("unsatisfiable refinement")
                || line.contains("cannot verify")
                || line.contains("refinement type mismatch")
            {
                findings.push(line.trim().to_string());
                error_count += 1;
            }

            // Success patterns:
            // - "function `foo` verified"
            // - "Verified N functions"
            if line.contains("verified") && !line.contains("cannot") {
                verified_count += 1;
            }
        }

        let status = if error_count == 0 && success {
            VerificationStatus::Proven
        } else if error_count > 0 {
            VerificationStatus::Disproven
        } else if !success {
            // Check for installation issues
            if output.contains("no such command") || output.contains("not found") {
                return (
                    VerificationStatus::Unknown {
                        reason:
                            "Flux is not installed. Install via: rustup +nightly component add flux"
                                .to_string(),
                    },
                    Vec::new(),
                );
            }
            if output.contains("unsupported") {
                return (
                    VerificationStatus::Unknown {
                        reason: "Flux requires nightly Rust with flux component".to_string(),
                    },
                    Vec::new(),
                );
            }
            VerificationStatus::Unknown {
                reason: "Verification failed with unknown error".to_string(),
            }
        } else if verified_count > 0 && error_count == 0 {
            VerificationStatus::Proven
        } else {
            VerificationStatus::Unknown {
                reason: "No refinement type annotations found".to_string(),
            }
        };

        (status, findings)
    }

    /// Check if Flux is installed
    pub async fn check_installed(&self) -> Result<bool, BackendError> {
        // Try cargo flux
        let output = Command::new("cargo")
            .args(["flux", "--version"])
            .output()
            .await;

        match output {
            Ok(out) if out.status.success() => Ok(true),
            _ => {
                // Try rustup component check
                let rustup_output = Command::new("rustup")
                    .args(["+nightly", "component", "list"])
                    .output()
                    .await;

                if let Ok(out) = rustup_output {
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    Ok(stdout.contains("flux") && stdout.contains("installed"))
                } else {
                    Ok(false)
                }
            }
        }
    }
}

impl Default for FluxBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VerificationBackend for FluxBackend {
    fn id(&self) -> BackendId {
        BackendId::Flux
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::Contract, PropertyType::Invariant]
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let crate_path = self.config.crate_path.clone().ok_or_else(|| {
            BackendError::Unavailable(
                "Flux backend requires crate_path pointing to a Rust crate with flux annotations"
                    .to_string(),
            )
        })?;

        self.analyze_crate(&crate_path).await
    }

    async fn health_check(&self) -> HealthStatus {
        match self.check_installed().await {
            Ok(true) => HealthStatus::Healthy,
            Ok(false) => HealthStatus::Unavailable {
                reason: "Flux not installed. Install via: rustup +nightly component add flux"
                    .to_string(),
            },
            Err(e) => HealthStatus::Unavailable {
                reason: format!("Failed to check Flux installation: {}", e),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flux_config_default() {
        let config = FluxConfig::default();
        assert!(config.crate_path.is_none());
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert_eq!(config.mode, FluxMode::Full);
        assert_eq!(config.solver, FluxSolver::Z3);
        assert!(config.target_functions.is_empty());
        assert!(!config.verbose);
        assert_eq!(config.smt_timeout_ms, 30000);
        assert!(config.flux_path.is_none());
    }

    #[test]
    fn test_flux_config_builder() {
        let config = FluxConfig::default()
            .with_crate_path(PathBuf::from("/test/path"))
            .with_timeout(Duration::from_secs(120))
            .with_mode(FluxMode::TypeCheckOnly)
            .with_solver(FluxSolver::CVC5)
            .with_target_functions(vec!["foo".to_string(), "bar".to_string()])
            .with_verbose(true)
            .with_smt_timeout(5000)
            .with_flux_path(PathBuf::from("/custom/flux"));

        assert_eq!(config.crate_path, Some(PathBuf::from("/test/path")));
        assert_eq!(config.timeout, Duration::from_secs(120));
        assert_eq!(config.mode, FluxMode::TypeCheckOnly);
        assert_eq!(config.solver, FluxSolver::CVC5);
        assert_eq!(config.target_functions, vec!["foo", "bar"]);
        assert!(config.verbose);
        assert_eq!(config.smt_timeout_ms, 5000);
        assert_eq!(config.flux_path, Some(PathBuf::from("/custom/flux")));
    }

    #[test]
    fn test_flux_backend_id() {
        let backend = FluxBackend::new();
        assert_eq!(backend.id(), BackendId::Flux);
    }

    #[test]
    fn test_flux_supports_contracts() {
        let backend = FluxBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::Contract));
        assert!(supported.contains(&PropertyType::Invariant));
    }

    #[test]
    fn test_flux_parse_output_clean() {
        let backend = FluxBackend::new();
        let output = "Checking crate...\nfunction `abs_positive` verified\nVerified 3 functions";
        let (status, findings) = backend.parse_flux_output(output, true);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(findings.is_empty());
    }

    #[test]
    fn test_flux_parse_output_with_error() {
        let backend = FluxBackend::new();
        let output = "error[FLUX]: refinement type error in `foo`\n  --> src/lib.rs:10:5";
        let (status, findings) = backend.parse_flux_output(output, false);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(!findings.is_empty());
    }

    #[test]
    fn test_flux_parse_output_not_installed() {
        let backend = FluxBackend::new();
        let output = "error: no such command: `flux`\nDid you mean `fix`?";
        let (status, _) = backend.parse_flux_output(output, false);
        match status {
            VerificationStatus::Unknown { reason } => {
                assert!(reason.contains("not installed"));
            }
            _ => panic!("Expected Unknown status"),
        }
    }

    #[tokio::test]
    async fn test_flux_health_check() {
        let backend = FluxBackend::new();
        let health = backend.health_check().await;
        // Flux may or may not be installed; just check we get a valid response
        match health {
            HealthStatus::Healthy => {}
            HealthStatus::Unavailable { reason } => {
                assert!(reason.contains("Flux") || reason.contains("flux"));
            }
            HealthStatus::Degraded { .. } => {}
        }
    }

    #[tokio::test]
    async fn test_flux_verify_requires_crate_path() {
        use dashprove_usl::{parse, typecheck};

        let backend = FluxBackend::new();
        let spec = parse("theorem test { true }").expect("parse");
        let typed_spec = typecheck(spec).expect("typecheck");

        let result = backend.verify(&typed_spec).await;
        assert!(result.is_err());
        if let Err(BackendError::Unavailable(msg)) = result {
            assert!(msg.contains("crate_path"));
        } else {
            panic!("Expected Unavailable error");
        }
    }

    #[test]
    fn test_flux_mode_default() {
        assert_eq!(FluxMode::default(), FluxMode::Full);
    }

    #[test]
    fn test_flux_solver_default() {
        assert_eq!(FluxSolver::default(), FluxSolver::Z3);
    }
}
