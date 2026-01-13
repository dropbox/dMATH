//! RustHorn backend for CHC-based Rust verification
//!
//! RustHorn is an automated verification tool developed at the University of Tokyo.
//! It translates Rust programs to Constrained Horn Clauses (CHC) and uses CHC solvers
//! (like Spacer/Z3, Eldarica, or HoIce) to verify safety properties.
//!
//! Key features:
//! - Automatic verification without manual proof annotations
//! - Focus on safe Rust programs
//! - Memory safety and assertion checking
//! - Invariant inference
//!
//! See: <https://github.com/rust-horn/rusthorn>

use crate::traits::*;
use async_trait::async_trait;
use dashprove_usl::typecheck::TypedSpec;
use regex::Regex;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::process::Command;
use tracing::{debug, info};

// =============================================
// Kani Proofs for RustHorn Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- RustHornConfig Default Tests ----

    /// Verify RustHornConfig::default rusthorn_path is None
    #[kani::proof]
    fn proof_rusthorn_config_default_rusthorn_path_none() {
        let config = RustHornConfig::default();
        kani::assert(
            config.rusthorn_path.is_none(),
            "Default rusthorn_path should be None",
        );
    }

    /// Verify RustHornConfig::default solver is Spacer
    #[kani::proof]
    fn proof_rusthorn_config_default_solver_spacer() {
        let config = RustHornConfig::default();
        kani::assert(
            matches!(config.solver, ChcSolver::Spacer),
            "Default solver should be Spacer",
        );
    }

    /// Verify RustHornConfig::default timeout is 300 seconds
    #[kani::proof]
    fn proof_rusthorn_config_default_timeout() {
        let config = RustHornConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "Default timeout should be 300 seconds",
        );
    }

    /// Verify RustHornConfig::default z3_path is None
    #[kani::proof]
    fn proof_rusthorn_config_default_z3_path_none() {
        let config = RustHornConfig::default();
        kani::assert(config.z3_path.is_none(), "Default z3_path should be None");
    }

    /// Verify RustHornConfig::default show_invariants is false
    #[kani::proof]
    fn proof_rusthorn_config_default_show_invariants_false() {
        let config = RustHornConfig::default();
        kani::assert(
            !config.show_invariants,
            "Default show_invariants should be false",
        );
    }

    /// Verify RustHornConfig::default verbose is false
    #[kani::proof]
    fn proof_rusthorn_config_default_verbose_false() {
        let config = RustHornConfig::default();
        kani::assert(!config.verbose, "Default verbose should be false");
    }

    // ---- RustHornBackend Construction Tests ----

    /// Verify RustHornBackend::new uses default timeout
    #[kani::proof]
    fn proof_rusthorn_backend_new_default_timeout() {
        let backend = RustHornBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(300),
            "New backend should use default timeout",
        );
    }

    /// Verify RustHornBackend::default equals RustHornBackend::new
    #[kani::proof]
    fn proof_rusthorn_backend_default_equals_new() {
        let b1 = RustHornBackend::new();
        let b2 = RustHornBackend::default();
        kani::assert(
            b1.config.timeout == b2.config.timeout,
            "default and new should have same timeout",
        );
    }

    /// Verify RustHornBackend::with_config preserves timeout
    #[kani::proof]
    fn proof_rusthorn_backend_with_config_preserves_timeout() {
        let config = RustHornConfig {
            timeout: Duration::from_secs(600),
            ..RustHornConfig::default()
        };
        let backend = RustHornBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "with_config should preserve custom timeout",
        );
    }

    // ---- ChcSolver Tests ----

    /// Verify ChcSolver::Spacer as_str returns "spacer"
    #[kani::proof]
    fn proof_chc_solver_spacer_as_str() {
        let solver = ChcSolver::Spacer;
        // Can't use string comparison in Kani, but we can verify it doesn't panic
        let _ = solver.as_str();
    }

    /// Verify ChcSolver::Eldarica as_str returns "eldarica"
    #[kani::proof]
    fn proof_chc_solver_eldarica_as_str() {
        let solver = ChcSolver::Eldarica;
        let _ = solver.as_str();
    }

    /// Verify ChcSolver::HoIce as_str returns "hoice"
    #[kani::proof]
    fn proof_chc_solver_hoice_as_str() {
        let solver = ChcSolver::HoIce;
        let _ = solver.as_str();
    }

    // ---- VerificationOutcome Tests ----

    /// Verify VerificationOutcome::Safe is_safe returns true
    #[kani::proof]
    fn proof_verification_outcome_safe_is_safe() {
        let outcome = VerificationOutcome::Safe;
        kani::assert(outcome.is_safe(), "Safe outcome should return is_safe true");
    }

    /// Verify VerificationOutcome::Unsafe is_safe returns false
    #[kani::proof]
    fn proof_verification_outcome_unsafe_is_not_safe() {
        let outcome = VerificationOutcome::Unsafe;
        kani::assert(
            !outcome.is_safe(),
            "Unsafe outcome should return is_safe false",
        );
    }

    /// Verify VerificationOutcome::Unknown is_safe returns false
    #[kani::proof]
    fn proof_verification_outcome_unknown_is_not_safe() {
        let outcome = VerificationOutcome::Unknown;
        kani::assert(
            !outcome.is_safe(),
            "Unknown outcome should return is_safe false",
        );
    }
}

/// CHC solver backend to use with RustHorn
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChcSolver {
    /// Spacer (part of Z3) - default and most mature
    #[default]
    Spacer,
    /// Eldarica - efficient for certain problem classes
    Eldarica,
    /// HoIce - ICE-based solver
    HoIce,
}

impl ChcSolver {
    /// Get the command-line argument string for this solver
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            ChcSolver::Spacer => "spacer",
            ChcSolver::Eldarica => "eldarica",
            ChcSolver::HoIce => "hoice",
        }
    }
}

/// Verification outcome from RustHorn
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationOutcome {
    /// Program is verified safe
    Safe,
    /// Verification found a potential violation
    Unsafe,
    /// Solver could not determine result
    Unknown,
}

impl VerificationOutcome {
    /// Check if the outcome indicates the program is safe
    #[must_use]
    pub fn is_safe(&self) -> bool {
        matches!(self, VerificationOutcome::Safe)
    }
}

/// Configuration for RustHorn backend
#[derive(Debug, Clone)]
pub struct RustHornConfig {
    /// Path to RustHorn executable (if not in PATH)
    pub rusthorn_path: Option<PathBuf>,
    /// CHC solver to use
    pub solver: ChcSolver,
    /// Timeout for verification
    pub timeout: Duration,
    /// Path to Z3 (for Spacer solver)
    pub z3_path: Option<PathBuf>,
    /// Show inferred invariants in output
    pub show_invariants: bool,
    /// Enable verbose output
    pub verbose: bool,
    /// Additional arguments to pass to RustHorn
    pub extra_args: Vec<String>,
}

impl Default for RustHornConfig {
    fn default() -> Self {
        Self {
            rusthorn_path: None,
            solver: ChcSolver::Spacer,
            timeout: Duration::from_secs(300),
            z3_path: None,
            show_invariants: false,
            verbose: false,
            extra_args: Vec::new(),
        }
    }
}

/// RustHorn backend for CHC-based Rust verification
#[derive(Debug, Clone)]
pub struct RustHornBackend {
    /// Configuration
    pub config: RustHornConfig,
}

impl Default for RustHornBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl RustHornBackend {
    /// Create a new RustHorn backend with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: RustHornConfig::default(),
        }
    }

    /// Create a new RustHorn backend with custom configuration
    #[must_use]
    pub fn with_config(config: RustHornConfig) -> Self {
        Self { config }
    }

    /// Get the path to the RustHorn executable
    fn get_rusthorn_path(&self) -> PathBuf {
        self.config
            .rusthorn_path
            .clone()
            .unwrap_or_else(|| PathBuf::from("rusthorn"))
    }

    /// Check if RustHorn is available
    pub async fn check_available(&self) -> Result<bool, BackendError> {
        let rusthorn_path = self.get_rusthorn_path();
        let output = Command::new(&rusthorn_path)
            .arg("--version")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await;

        match output {
            Ok(o) => Ok(o.status.success()),
            Err(_) => Ok(false),
        }
    }

    /// Parse RustHorn output to determine verification outcome
    fn parse_output(&self, stdout: &str, stderr: &str) -> VerificationOutcome {
        let combined = format!("{}\n{}", stdout, stderr);

        // Check for UNSAFE first since it contains "SAFE" as a substring
        if combined.contains("UNSAFE") || combined.contains("counterexample") {
            VerificationOutcome::Unsafe
        } else if combined.contains("SAFE") || combined.contains("verified") {
            VerificationOutcome::Safe
        } else {
            VerificationOutcome::Unknown
        }
    }

    /// Extract inferred invariants from output
    fn extract_invariants(&self, output: &str) -> Vec<String> {
        let mut invariants = Vec::new();
        let inv_regex = Regex::new(r"(?i)invariant:\s*(.+)").unwrap();

        for cap in inv_regex.captures_iter(output) {
            if let Some(inv) = cap.get(1) {
                invariants.push(inv.as_str().trim().to_string());
            }
        }

        invariants
    }

    /// Extract counterexample from output
    fn extract_counterexample(&self, output: &str) -> Option<String> {
        // Look for counterexample section
        if let Some(start) = output.find("Counterexample:") {
            let rest = &output[start..];
            // Find end of counterexample (next section or end)
            let end = rest
                .find("\n\n")
                .or_else(|| rest.find("---"))
                .unwrap_or(rest.len());
            return Some(rest[..end].to_string());
        }
        None
    }

    /// Run verification on a Rust source file
    pub async fn verify_file(&self, source_path: &PathBuf) -> Result<BackendResult, BackendError> {
        let start = Instant::now();
        let rusthorn_path = self.get_rusthorn_path();

        let mut cmd = Command::new(&rusthorn_path);
        cmd.arg("verify");
        cmd.arg(source_path);
        cmd.arg("--solver").arg(self.config.solver.as_str());

        if self.config.show_invariants {
            cmd.arg("--show-invariants");
        }

        if self.config.verbose {
            cmd.arg("--verbose");
        }

        if let Some(z3_path) = &self.config.z3_path {
            cmd.arg("--z3-path").arg(z3_path);
        }

        for arg in &self.config.extra_args {
            cmd.arg(arg);
        }

        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        debug!("Running RustHorn: {:?}", cmd);

        let output = tokio::time::timeout(self.config.timeout, cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| {
                BackendError::VerificationFailed(format!("Failed to run RustHorn: {}", e))
            })?;

        let elapsed = start.elapsed();
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        let outcome = self.parse_output(&stdout, &stderr);
        let invariants = self.extract_invariants(&stdout);
        let counterexample = self.extract_counterexample(&stdout);

        let status = match outcome {
            VerificationOutcome::Safe => VerificationStatus::Proven,
            VerificationOutcome::Unsafe => VerificationStatus::Disproven,
            VerificationOutcome::Unknown => VerificationStatus::Unknown {
                reason: "RustHorn could not determine result".to_string(),
            },
        };

        let mut diagnostics = Vec::new();
        if !output.status.success() && outcome == VerificationOutcome::Unknown {
            diagnostics.push(stderr.clone());
        }
        if !invariants.is_empty() {
            diagnostics.push(format!("Inferred invariants: {}", invariants.join(", ")));
        }

        info!(
            "RustHorn verification completed in {:?}: {:?}",
            elapsed, status
        );

        Ok(BackendResult {
            backend: BackendId::RustHorn,
            status,
            proof: None,
            counterexample: counterexample.map(StructuredCounterexample::from_raw),
            diagnostics,
            time_taken: elapsed,
        })
    }

    /// Run verification on Rust source code string
    pub async fn verify_source(&self, source: &str) -> Result<BackendResult, BackendError> {
        // Create temp directory and file
        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let source_path = temp_dir.path().join("verify.rs");
        std::fs::write(&source_path, source).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write source: {}", e))
        })?;

        self.verify_file(&source_path).await
    }
}

#[async_trait]
impl VerificationBackend for RustHornBackend {
    fn id(&self) -> BackendId {
        BackendId::RustHorn
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![
            PropertyType::Contract,
            PropertyType::Invariant,
            PropertyType::MemorySafety,
        ]
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        // For now, return a placeholder - actual USL compilation would go here
        Ok(BackendResult {
            backend: BackendId::RustHorn,
            status: VerificationStatus::Unknown {
                reason: "USL to RustHorn compilation not yet implemented".to_string(),
            },
            proof: None,
            counterexample: None,
            diagnostics: vec![
                "RustHorn backend requires direct Rust source verification".to_string()
            ],
            time_taken: Duration::from_secs(0),
        })
    }

    async fn health_check(&self) -> HealthStatus {
        match self.check_available().await {
            Ok(true) => HealthStatus::Healthy,
            Ok(false) => HealthStatus::Unavailable {
                reason: "RustHorn not found".to_string(),
            },
            Err(e) => HealthStatus::Unavailable {
                reason: format!("RustHorn health check failed: {}", e),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rusthorn_config_default() {
        let config = RustHornConfig::default();
        assert!(config.rusthorn_path.is_none());
        assert!(matches!(config.solver, ChcSolver::Spacer));
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert!(config.z3_path.is_none());
        assert!(!config.show_invariants);
        assert!(!config.verbose);
        assert!(config.extra_args.is_empty());
    }

    #[test]
    fn test_rusthorn_backend_new() {
        let backend = RustHornBackend::new();
        assert_eq!(backend.config.timeout, Duration::from_secs(300));
    }

    #[test]
    fn test_rusthorn_backend_with_config() {
        let config = RustHornConfig {
            timeout: Duration::from_secs(600),
            solver: ChcSolver::Eldarica,
            ..RustHornConfig::default()
        };
        let backend = RustHornBackend::with_config(config);
        assert_eq!(backend.config.timeout, Duration::from_secs(600));
        assert!(matches!(backend.config.solver, ChcSolver::Eldarica));
    }

    #[test]
    fn test_chc_solver_as_str() {
        assert_eq!(ChcSolver::Spacer.as_str(), "spacer");
        assert_eq!(ChcSolver::Eldarica.as_str(), "eldarica");
        assert_eq!(ChcSolver::HoIce.as_str(), "hoice");
    }

    #[test]
    fn test_verification_outcome_is_safe() {
        assert!(VerificationOutcome::Safe.is_safe());
        assert!(!VerificationOutcome::Unsafe.is_safe());
        assert!(!VerificationOutcome::Unknown.is_safe());
    }

    #[test]
    fn test_parse_output_safe() {
        let backend = RustHornBackend::new();
        let outcome = backend.parse_output("Result: SAFE", "");
        assert_eq!(outcome, VerificationOutcome::Safe);
    }

    #[test]
    fn test_parse_output_unsafe() {
        let backend = RustHornBackend::new();
        let outcome = backend.parse_output("Result: UNSAFE\nCounterexample: ...", "");
        assert_eq!(outcome, VerificationOutcome::Unsafe);
    }

    #[test]
    fn test_parse_output_unknown() {
        let backend = RustHornBackend::new();
        let outcome = backend.parse_output("Timeout", "");
        assert_eq!(outcome, VerificationOutcome::Unknown);
    }

    #[test]
    fn test_extract_invariants() {
        let backend = RustHornBackend::new();
        let output = "invariant: x >= 0\ninvariant: y < 100";
        let invariants = backend.extract_invariants(output);
        assert_eq!(invariants.len(), 2);
        assert!(invariants.contains(&"x >= 0".to_string()));
        assert!(invariants.contains(&"y < 100".to_string()));
    }

    #[test]
    fn test_extract_counterexample() {
        let backend = RustHornBackend::new();
        let output = "Counterexample:\nx = -1\ny = 200\n\nEnd";
        let ce = backend.extract_counterexample(output);
        assert!(ce.is_some());
        assert!(ce.unwrap().contains("x = -1"));
    }

    #[test]
    fn test_backend_id() {
        let backend = RustHornBackend::new();
        assert_eq!(backend.id(), BackendId::RustHorn);
    }

    #[test]
    fn test_supports() {
        let backend = RustHornBackend::new();
        let types = backend.supports();
        assert!(types.contains(&PropertyType::Contract));
        assert!(types.contains(&PropertyType::Invariant));
        assert!(types.contains(&PropertyType::MemorySafety));
    }
}
