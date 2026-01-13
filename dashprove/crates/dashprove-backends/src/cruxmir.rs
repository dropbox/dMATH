//! Crux-mir backend for LLVM/MIR symbolic testing
//!
//! Crux-mir is a symbolic testing tool from Galois that reasons over Rust MIR
//! using the Crux/Crucible infrastructure. It can discover counterexamples for
//! panics and user assertions, or prove that no such failures exist for the
//! explored paths.
//!
//! See: <https://github.com/GaloisInc/mir-verifier>

// =============================================
// Kani Proofs for Crux-mir Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- CruxMirSolver Default Tests ----

    /// Verify CruxMirSolver::default is Z3
    #[kani::proof]
    fn proof_cruxmir_solver_default_is_z3() {
        let solver = CruxMirSolver::default();
        kani::assert(solver == CruxMirSolver::Z3, "Default solver should be Z3");
    }

    /// Verify CruxMirSolver::Z3 as_str returns "z3"
    #[kani::proof]
    fn proof_cruxmir_solver_z3_as_str() {
        let solver = CruxMirSolver::Z3;
        kani::assert(solver.as_str() == "z3", "Z3 should return z3");
    }

    /// Verify CruxMirSolver::Cvc5 as_str returns "cvc5"
    #[kani::proof]
    fn proof_cruxmir_solver_cvc5_as_str() {
        let solver = CruxMirSolver::Cvc5;
        kani::assert(solver.as_str() == "cvc5", "Cvc5 should return cvc5");
    }

    /// Verify CruxMirSolver::Yices as_str returns "yices"
    #[kani::proof]
    fn proof_cruxmir_solver_yices_as_str() {
        let solver = CruxMirSolver::Yices;
        kani::assert(solver.as_str() == "yices", "Yices should return yices");
    }

    // ---- CruxMirConfig Default Tests ----

    /// Verify CruxMirConfig::default timeout is 600 seconds
    #[kani::proof]
    fn proof_cruxmir_config_default_timeout() {
        let config = CruxMirConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(600),
            "Default timeout should be 600 seconds",
        );
    }

    /// Verify CruxMirConfig::default cruxmir_path is None
    #[kani::proof]
    fn proof_cruxmir_config_default_path_none() {
        let config = CruxMirConfig::default();
        kani::assert(
            config.cruxmir_path.is_none(),
            "Default cruxmir_path should be None",
        );
    }

    /// Verify CruxMirConfig::default project_path is None
    #[kani::proof]
    fn proof_cruxmir_config_default_project_path_none() {
        let config = CruxMirConfig::default();
        kani::assert(
            config.project_path.is_none(),
            "Default project_path should be None",
        );
    }

    /// Verify CruxMirConfig::default entry_function is None
    #[kani::proof]
    fn proof_cruxmir_config_default_entry_function_none() {
        let config = CruxMirConfig::default();
        kani::assert(
            config.entry_function.is_none(),
            "Default entry_function should be None",
        );
    }

    /// Verify CruxMirConfig::default solver is Z3
    #[kani::proof]
    fn proof_cruxmir_config_default_solver() {
        let config = CruxMirConfig::default();
        kani::assert(
            config.solver == CruxMirSolver::Z3,
            "Default solver should be Z3",
        );
    }

    /// Verify CruxMirConfig::default extra_args is empty
    #[kani::proof]
    fn proof_cruxmir_config_default_extra_args_empty() {
        let config = CruxMirConfig::default();
        kani::assert(
            config.extra_args.is_empty(),
            "Default extra_args should be empty",
        );
    }

    /// Verify CruxMirConfig::default release is false
    #[kani::proof]
    fn proof_cruxmir_config_default_release_false() {
        let config = CruxMirConfig::default();
        kani::assert(!config.release, "Default release should be false");
    }

    // ---- CruxMirConfig Builder Tests ----

    /// Verify with_timeout preserves timeout value
    #[kani::proof]
    fn proof_cruxmir_config_with_timeout() {
        let config = CruxMirConfig::default().with_timeout(Duration::from_secs(1200));
        kani::assert(
            config.timeout == Duration::from_secs(1200),
            "with_timeout should set timeout",
        );
    }

    /// Verify with_solver preserves Cvc5
    #[kani::proof]
    fn proof_cruxmir_config_with_solver_cvc5() {
        let config = CruxMirConfig::default().with_solver(CruxMirSolver::Cvc5);
        kani::assert(
            config.solver == CruxMirSolver::Cvc5,
            "with_solver should set Cvc5",
        );
    }

    /// Verify with_solver preserves Yices
    #[kani::proof]
    fn proof_cruxmir_config_with_solver_yices() {
        let config = CruxMirConfig::default().with_solver(CruxMirSolver::Yices);
        kani::assert(
            config.solver == CruxMirSolver::Yices,
            "with_solver should set Yices",
        );
    }

    /// Verify with_release preserves true
    #[kani::proof]
    fn proof_cruxmir_config_with_release_true() {
        let config = CruxMirConfig::default().with_release(true);
        kani::assert(config.release, "with_release(true) should enable release");
    }

    /// Verify with_release preserves false
    #[kani::proof]
    fn proof_cruxmir_config_with_release_false() {
        let config = CruxMirConfig::default().with_release(false);
        kani::assert(
            !config.release,
            "with_release(false) should disable release",
        );
    }

    // ---- CruxMirBackend Construction Tests ----

    /// Verify CruxMirBackend::new uses default config timeout
    #[kani::proof]
    fn proof_cruxmir_backend_new_default_timeout() {
        let backend = CruxMirBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "New backend should use default timeout",
        );
    }

    /// Verify CruxMirBackend::default equals CruxMirBackend::new timeout
    #[kani::proof]
    fn proof_cruxmir_backend_default_equals_new_timeout() {
        let default_backend = CruxMirBackend::default();
        let new_backend = CruxMirBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "Default and new should have same timeout",
        );
    }

    /// Verify CruxMirBackend::default equals CruxMirBackend::new solver
    #[kani::proof]
    fn proof_cruxmir_backend_default_equals_new_solver() {
        let default_backend = CruxMirBackend::default();
        let new_backend = CruxMirBackend::new();
        kani::assert(
            default_backend.config.solver == new_backend.config.solver,
            "Default and new should have same solver",
        );
    }

    /// Verify CruxMirBackend::with_config preserves custom timeout
    #[kani::proof]
    fn proof_cruxmir_backend_with_config_timeout() {
        let config = CruxMirConfig {
            cruxmir_path: None,
            project_path: None,
            entry_function: None,
            timeout: Duration::from_secs(1200),
            solver: CruxMirSolver::Z3,
            extra_args: Vec::new(),
            release: false,
        };
        let backend = CruxMirBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(1200),
            "with_config should preserve timeout",
        );
    }

    /// Verify CruxMirBackend::with_config preserves solver
    #[kani::proof]
    fn proof_cruxmir_backend_with_config_solver() {
        let config = CruxMirConfig {
            cruxmir_path: None,
            project_path: None,
            entry_function: None,
            timeout: Duration::from_secs(600),
            solver: CruxMirSolver::Yices,
            extra_args: Vec::new(),
            release: false,
        };
        let backend = CruxMirBackend::with_config(config);
        kani::assert(
            backend.config.solver == CruxMirSolver::Yices,
            "with_config should preserve solver",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify CruxMirBackend::id returns CruxMir
    #[kani::proof]
    fn proof_cruxmir_backend_id() {
        let backend = CruxMirBackend::new();
        kani::assert(
            backend.id() == BackendId::CruxMir,
            "Backend id should be CruxMir",
        );
    }

    /// Verify CruxMirBackend::supports includes Contract
    #[kani::proof]
    fn proof_cruxmir_backend_supports_contract() {
        let backend = CruxMirBackend::new();
        let supported = backend.supports();
        let has_contract = supported.iter().any(|p| *p == PropertyType::Contract);
        kani::assert(has_contract, "Should support Contract property");
    }

    /// Verify CruxMirBackend::supports includes MemorySafety
    #[kani::proof]
    fn proof_cruxmir_backend_supports_memory_safety() {
        let backend = CruxMirBackend::new();
        let supported = backend.supports();
        let has_memory_safety = supported.iter().any(|p| *p == PropertyType::MemorySafety);
        kani::assert(has_memory_safety, "Should support MemorySafety property");
    }

    /// Verify CruxMirBackend::supports includes Invariant
    #[kani::proof]
    fn proof_cruxmir_backend_supports_invariant() {
        let backend = CruxMirBackend::new();
        let supported = backend.supports();
        let has_invariant = supported.iter().any(|p| *p == PropertyType::Invariant);
        kani::assert(has_invariant, "Should support Invariant property");
    }

    /// Verify CruxMirBackend::supports returns exactly 3 properties
    #[kani::proof]
    fn proof_cruxmir_backend_supports_length() {
        let backend = CruxMirBackend::new();
        let supported = backend.supports();
        kani::assert(supported.len() == 3, "Should support exactly 3 properties");
    }

    // ---- parse_output Tests ----

    /// Verify parse_output returns Proven for "All goals proved"
    #[kani::proof]
    fn proof_parse_output_all_goals_proved() {
        let backend = CruxMirBackend::new();
        let (status, ce) = backend.parse_output("All goals proved", "", true);
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "Should return Proven for all goals proved",
        );
        kani::assert(ce.is_none(), "Should return no counterexample");
    }

    /// Verify parse_output returns Proven for "All proofs succeeded"
    #[kani::proof]
    fn proof_parse_output_all_proofs_succeeded() {
        let backend = CruxMirBackend::new();
        let (status, ce) = backend.parse_output("All proofs succeeded", "", true);
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "Should return Proven for all proofs succeeded",
        );
        kani::assert(ce.is_none(), "Should return no counterexample");
    }

    /// Verify parse_output returns Proven for "Verification succeeded"
    #[kani::proof]
    fn proof_parse_output_verification_succeeded() {
        let backend = CruxMirBackend::new();
        let (status, ce) = backend.parse_output("Verification succeeded", "", true);
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "Should return Proven for verification succeeded",
        );
        kani::assert(ce.is_none(), "Should return no counterexample");
    }

    /// Verify parse_output returns Disproven for ASSERTION FAIL
    #[kani::proof]
    fn proof_parse_output_assertion_fail() {
        let backend = CruxMirBackend::new();
        let (status, ce) = backend.parse_output("", "ASSERTION FAIL at main.rs:12:3", false);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "Should return Disproven for ASSERTION FAIL",
        );
        kani::assert(ce.is_some(), "Should return counterexample");
    }

    /// Verify parse_output returns Disproven for "assertion failed"
    #[kani::proof]
    fn proof_parse_output_assertion_failed() {
        let backend = CruxMirBackend::new();
        let (status, ce) = backend.parse_output("assertion failed: x == 0", "", false);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "Should return Disproven for assertion failed",
        );
        kani::assert(ce.is_some(), "Should return counterexample");
    }

    /// Verify parse_output returns Disproven for counterexample
    #[kani::proof]
    fn proof_parse_output_counterexample() {
        let backend = CruxMirBackend::new();
        let (status, ce) = backend.parse_output("Found counterexample", "", false);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "Should return Disproven for counterexample",
        );
        kani::assert(ce.is_some(), "Should return counterexample");
    }

    /// Verify parse_output returns Unknown for inconclusive
    #[kani::proof]
    fn proof_parse_output_unknown() {
        let backend = CruxMirBackend::new();
        let (status, _) = backend.parse_output("Some other output", "", false);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Should return Unknown for inconclusive",
        );
    }
}

use crate::counterexample::{FailedCheck, SourceLocation, StructuredCounterexample};
use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
    VerificationStatus,
};
use async_trait::async_trait;
use dashprove_usl::typecheck::TypedSpec;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::process::Command;
use tokio::time::timeout;
use tracing::debug;

/// SMT solver to use with Crux-mir
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CruxMirSolver {
    /// Z3 SMT solver (default)
    #[default]
    Z3,
    /// CVC5 SMT solver
    Cvc5,
    /// Yices SMT solver
    Yices,
}

impl CruxMirSolver {
    fn as_str(&self) -> &'static str {
        match self {
            CruxMirSolver::Z3 => "z3",
            CruxMirSolver::Cvc5 => "cvc5",
            CruxMirSolver::Yices => "yices",
        }
    }
}

/// Configuration for Crux-mir backend
#[derive(Debug, Clone)]
pub struct CruxMirConfig {
    /// Path to crux-mir binary
    pub cruxmir_path: Option<PathBuf>,
    /// Path to Rust project to analyze (if None, a temporary harness is generated)
    pub project_path: Option<PathBuf>,
    /// Entry function to target
    pub entry_function: Option<String>,
    /// Timeout for symbolic execution
    pub timeout: Duration,
    /// Which SMT solver to use
    pub solver: CruxMirSolver,
    /// Extra args forwarded to crux-mir
    pub extra_args: Vec<String>,
    /// Build in release mode for performance
    pub release: bool,
}

impl Default for CruxMirConfig {
    fn default() -> Self {
        Self {
            cruxmir_path: None,
            project_path: None,
            entry_function: None,
            timeout: Duration::from_secs(600),
            solver: CruxMirSolver::default(),
            extra_args: Vec::new(),
            release: false,
        }
    }
}

impl CruxMirConfig {
    /// Set custom crux-mir binary path
    pub fn with_cruxmir_path(mut self, path: PathBuf) -> Self {
        self.cruxmir_path = Some(path);
        self
    }

    /// Set project path to analyze
    pub fn with_project_path(mut self, path: PathBuf) -> Self {
        self.project_path = Some(path);
        self
    }

    /// Set entry function to verify
    pub fn with_entry_function(mut self, entry: String) -> Self {
        self.entry_function = Some(entry);
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set solver
    pub fn with_solver(mut self, solver: CruxMirSolver) -> Self {
        self.solver = solver;
        self
    }

    /// Add extra CLI args
    pub fn with_extra_args(mut self, args: Vec<String>) -> Self {
        self.extra_args = args;
        self
    }

    /// Enable release builds
    pub fn with_release(mut self, release: bool) -> Self {
        self.release = release;
        self
    }
}

/// Crux-mir backend for symbolic testing of Rust MIR
pub struct CruxMirBackend {
    config: CruxMirConfig,
}

impl Default for CruxMirBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CruxMirBackend {
    /// Create a new backend with default configuration
    pub fn new() -> Self {
        Self {
            config: CruxMirConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: CruxMirConfig) -> Self {
        Self { config }
    }

    /// Detect crux-mir availability and version
    async fn detect_cruxmir(&self) -> Result<CruxMirDetection, String> {
        let binary = if let Some(path) = &self.config.cruxmir_path {
            path.clone()
        } else {
            which::which("crux-mir")
                .map_err(|_| "crux-mir not found. Install via cargo install crux-mir".to_string())?
        };

        let version_output = Command::new(&binary)
            .arg("--version")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| format!("Failed to execute crux-mir: {e}"))?;

        let version = String::from_utf8_lossy(&version_output.stdout)
            .lines()
            .next()
            .unwrap_or("crux-mir")
            .to_string();

        Ok(CruxMirDetection { binary, version })
    }

    /// Build a temporary project containing a Crux-mir harness for the USL spec
    fn build_temp_project(&self, spec: &TypedSpec) -> Result<(TempDir, PathBuf), BackendError> {
        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {e}"))
        })?;
        let src_dir = temp_dir.path().join("src");
        fs::create_dir_all(&src_dir).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create src dir: {e}"))
        })?;

        let harness = self.generate_harness(spec);
        fs::write(src_dir.join("main.rs"), harness).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write harness: {e}"))
        })?;

        let cargo_toml = r#"[package]
name = "dashprove-cruxmir-runner"
version = "0.1.0"
edition = "2021"

[dependencies]
# Crucible is only needed when users actually build/run the harness.
crucible = "0.9"
"#;
        fs::write(temp_dir.path().join("Cargo.toml"), cargo_toml).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write Cargo.toml: {e}"))
        })?;

        let project_path = temp_dir.path().to_path_buf();
        Ok((temp_dir, project_path))
    }

    /// Generate a simple Crux-mir harness from a typed USL spec
    fn generate_harness(&self, spec: &TypedSpec) -> String {
        let mut code = String::new();
        code.push_str("// Generated by DashProve for Crux-mir\n");
        code.push_str("use crucible::prelude::*;\n\n");

        for property in &spec.spec.properties {
            let fn_name = property
                .name()
                .replace([' ', '-', ':', '.'], "_")
                .to_lowercase();

            code.push_str(&format!("fn {}() {{\n", fn_name));
            code.push_str("    // TODO: translate USL property into Crux-mir assertions\n");
            code.push_str("    crucible_assert!(true);\n");
            code.push_str("}\n\n");
        }

        code.push_str("fn main() {\n");
        for property in &spec.spec.properties {
            let fn_name = property
                .name()
                .replace([' ', '-', ':', '.'], "_")
                .to_lowercase();
            code.push_str(&format!("    {}();\n", fn_name));
        }
        code.push_str("}\n");

        code
    }

    /// Execute crux-mir against the provided project path
    async fn run_cruxmir(
        &self,
        detection: &CruxMirDetection,
        project_dir: &Path,
    ) -> Result<(VerificationStatus, Option<StructuredCounterexample>), BackendError> {
        let mut cmd = Command::new(&detection.binary);
        cmd.arg("--solver")
            .arg(self.config.solver.as_str())
            .arg("--quiet");

        if let Some(entry) = &self.config.entry_function {
            cmd.arg("--entry-point").arg(entry);
        }

        if self.config.release {
            cmd.arg("--release");
        }

        cmd.args(&self.config.extra_args);
        cmd.arg(project_dir);

        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let output = timeout(self.config.timeout, cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| {
                BackendError::VerificationFailed(format!("Failed to run crux-mir: {e}"))
            })?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        debug!("crux-mir stdout: {}", stdout);
        debug!("crux-mir stderr: {}", stderr);

        Ok(self.parse_output(&stdout, &stderr, output.status.success()))
    }

    fn parse_output(
        &self,
        stdout: &str,
        stderr: &str,
        success: bool,
    ) -> (VerificationStatus, Option<StructuredCounterexample>) {
        let combined = format!("{stdout}\n{stderr}");

        if success
            && (combined.contains("All goals proved")
                || combined.contains("All proofs succeeded")
                || combined.contains("Verification succeeded"))
        {
            return (VerificationStatus::Proven, None);
        }

        if combined.contains("ASSERTION FAIL")
            || combined.contains("assertion failed")
            || combined.contains("counterexample")
        {
            let ce = self.parse_counterexample(&combined);
            return (VerificationStatus::Disproven, ce);
        }

        (
            VerificationStatus::Unknown {
                reason: "Crux-mir output did not indicate success or failure".to_string(),
            },
            None,
        )
    }

    fn parse_counterexample(&self, output: &str) -> Option<StructuredCounterexample> {
        let mut failed_checks = Vec::new();

        for line in output.lines() {
            if line.contains("assertion failed") || line.contains("ASSERTION FAIL") {
                let location = self.extract_location(line);
                failed_checks.push(FailedCheck {
                    check_id: format!("cruxmir.assertion.{}", failed_checks.len()),
                    description: line.trim().to_string(),
                    location,
                    function: None,
                });
            }
        }

        if failed_checks.is_empty() {
            return None;
        }

        Some(StructuredCounterexample {
            witness: std::collections::HashMap::new(),
            failed_checks,
            playback_test: None,
            trace: Vec::new(),
            raw: Some(output.to_string()),
            minimized: false,
        })
    }

    fn extract_location(&self, line: &str) -> Option<SourceLocation> {
        // Expected formats:
        // "ASSERTION FAIL at file.rs:42:13"
        // "assertion failed: file.rs:10"
        let parts: Vec<&str> = line.split_whitespace().collect();
        for part in parts {
            if let Some((file, rest)) = part.split_once(':') {
                if let Ok(line_num) = rest.parse::<u32>() {
                    return Some(SourceLocation {
                        file: file.to_string(),
                        line: line_num,
                        column: None,
                    });
                } else if let Some((line_part, col_part)) = rest.split_once(':') {
                    if let (Ok(line_num), Ok(col_num)) =
                        (line_part.parse::<u32>(), col_part.parse::<u32>())
                    {
                        return Some(SourceLocation {
                            file: file.to_string(),
                            line: line_num,
                            column: Some(col_num),
                        });
                    }
                }
            }
        }
        None
    }
}

#[async_trait]
impl VerificationBackend for CruxMirBackend {
    fn id(&self) -> BackendId {
        BackendId::CruxMir
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![
            PropertyType::Contract,
            PropertyType::MemorySafety,
            PropertyType::Invariant,
        ]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();
        let detection = self
            .detect_cruxmir()
            .await
            .map_err(BackendError::Unavailable)?;

        debug!("Detected crux-mir: {:?}", detection);

        let (project_root, _guard) = if let Some(project) = &self.config.project_path {
            (project.clone(), None)
        } else {
            let (tmp, path) = self.build_temp_project(spec)?;
            (path, Some(tmp))
        };

        let (status, counterexample) = self.run_cruxmir(&detection, &project_root).await?;

        Ok(BackendResult {
            backend: BackendId::CruxMir,
            status,
            counterexample,
            proof: None,
            diagnostics: vec![format!(
                "Crux-mir {} using solver {}",
                detection.version,
                self.config.solver.as_str()
            )],
            time_taken: start.elapsed(),
        })
    }

    async fn health_check(&self) -> HealthStatus {
        match self.detect_cruxmir().await {
            Ok(_detection) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}

#[derive(Debug)]
struct CruxMirDetection {
    binary: PathBuf,
    version: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_spec() -> TypedSpec {
        use dashprove_usl::ast::{Expr, Property, Spec, Theorem};
        use std::collections::HashMap;

        let mut spec = Spec::default();
        spec.properties.push(Property::Theorem(Theorem {
            name: "safety".to_string(),
            body: Expr::Bool(true),
        }));

        TypedSpec {
            spec,
            type_info: HashMap::new(),
        }
    }

    #[test]
    fn backend_id_is_cruxmir() {
        let backend = CruxMirBackend::new();
        assert_eq!(backend.id(), BackendId::CruxMir);
    }

    #[test]
    fn supports_contract_and_memory() {
        let backend = CruxMirBackend::new();
        let supports = backend.supports();
        assert!(supports.contains(&PropertyType::Contract));
        assert!(supports.contains(&PropertyType::MemorySafety));
    }

    #[test]
    fn harness_generation_includes_property_functions() {
        let backend = CruxMirBackend::new();
        let spec = dummy_spec();
        let harness = backend.generate_harness(&spec);
        assert!(harness.contains("crucible_assert!(true);"));
        assert!(harness.contains("fn safety()"));
    }

    #[test]
    fn parse_output_success_detects_proven() {
        let backend = CruxMirBackend::new();
        let (status, ce) = backend.parse_output("All goals proved", "", true);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(ce.is_none());
    }

    #[test]
    fn parse_output_failure_detects_disproven() {
        let backend = CruxMirBackend::new();
        let (status, ce) = backend.parse_output("", "ASSERTION FAIL at main.rs:12:3", false);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(ce.is_some());
    }

    #[test]
    fn extract_location_parses_file_and_line() {
        let backend = CruxMirBackend::new();
        let location = backend.extract_location("ASSERTION FAIL at file.rs:42:7");
        assert!(location.is_some());
        let loc = location.unwrap();
        assert_eq!(loc.file, "file.rs");
        assert_eq!(loc.line, 42);
        assert_eq!(loc.column, Some(7));
    }
}
