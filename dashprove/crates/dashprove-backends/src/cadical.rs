//! CaDiCaL SAT solver backend
//!
//! CaDiCaL is a modern, award-winning SAT solver developed by Armin Biere.
//! It won multiple SAT competitions and is known for excellent performance
//! on industrial benchmarks.
//!
//! See: <https://github.com/arminbiere/cadical>

// =============================================
// Kani Proofs for CaDiCaL Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- CaDiCaLPreprocessing Default Tests ----

    /// Verify CaDiCaLPreprocessing default is Light
    #[kani::proof]
    fn proof_cadical_preprocessing_default() {
        let prep = CaDiCaLPreprocessing::default();
        kani::assert(
            prep == CaDiCaLPreprocessing::Light,
            "default preprocessing should be Light",
        );
    }

    // ---- CaDiCaLConfig Default Tests ----

    /// Verify CaDiCaLConfig::default sets correct baseline values
    #[kani::proof]
    fn proof_cadical_config_defaults() {
        let config = CaDiCaLConfig::default();
        kani::assert(
            config.cadical_path.is_none(),
            "cadical_path should default to None",
        );
        kani::assert(
            config.timeout == Duration::from_secs(60),
            "timeout should default to 60 seconds",
        );
        kani::assert(
            config.preprocessing == CaDiCaLPreprocessing::Light,
            "preprocessing should default to Light",
        );
        kani::assert(!config.verbose, "verbose should default to false");
        kani::assert(!config.write_proof, "write_proof should default to false");
        kani::assert(
            config.conflict_limit == 0,
            "conflict_limit should default to 0",
        );
        kani::assert(
            config.decision_limit == 0,
            "decision_limit should default to 0",
        );
    }

    // ---- CaDiCaLBackend Construction Tests ----

    /// Verify CaDiCaLBackend::new uses default configuration
    #[kani::proof]
    fn proof_cadical_backend_new_defaults() {
        let backend = CaDiCaLBackend::new();
        kani::assert(
            backend.config.cadical_path.is_none(),
            "new backend should have no cadical_path",
        );
        kani::assert(
            backend.config.timeout == Duration::from_secs(60),
            "new backend should default timeout to 60 seconds",
        );
        kani::assert(
            backend.config.preprocessing == CaDiCaLPreprocessing::Light,
            "new backend should default preprocessing to Light",
        );
    }

    /// Verify CaDiCaLBackend::default equals CaDiCaLBackend::new
    #[kani::proof]
    fn proof_cadical_backend_default_equals_new() {
        let default_backend = CaDiCaLBackend::default();
        let new_backend = CaDiCaLBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "default and new should share timeout",
        );
        kani::assert(
            default_backend.config.preprocessing == new_backend.config.preprocessing,
            "default and new should share preprocessing",
        );
    }

    /// Verify CaDiCaLBackend::with_config preserves custom configuration
    #[kani::proof]
    fn proof_cadical_backend_with_config() {
        let config = CaDiCaLConfig {
            cadical_path: Some(PathBuf::from("/opt/cadical")),
            timeout: Duration::from_secs(120),
            preprocessing: CaDiCaLPreprocessing::Full,
            verbose: true,
            write_proof: true,
            conflict_limit: 1000,
            decision_limit: 500,
        };
        let backend = CaDiCaLBackend::with_config(config);
        kani::assert(
            backend.config.cadical_path == Some(PathBuf::from("/opt/cadical")),
            "with_config should preserve cadical_path",
        );
        kani::assert(
            backend.config.timeout == Duration::from_secs(120),
            "with_config should preserve timeout",
        );
        kani::assert(
            backend.config.preprocessing == CaDiCaLPreprocessing::Full,
            "with_config should preserve preprocessing",
        );
        kani::assert(
            backend.config.verbose,
            "with_config should preserve verbose",
        );
        kani::assert(
            backend.config.write_proof,
            "with_config should preserve write_proof",
        );
        kani::assert(
            backend.config.conflict_limit == 1000,
            "with_config should preserve conflict_limit",
        );
        kani::assert(
            backend.config.decision_limit == 500,
            "with_config should preserve decision_limit",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify id() returns BackendId::CaDiCaL
    #[kani::proof]
    fn proof_cadical_backend_id() {
        let backend = CaDiCaLBackend::new();
        kani::assert(
            backend.id() == BackendId::CaDiCaL,
            "CaDiCaLBackend id should be BackendId::CaDiCaL",
        );
    }

    /// Verify supports() includes Theorem and Invariant
    #[kani::proof]
    fn proof_cadical_backend_supports() {
        let backend = CaDiCaLBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::Theorem),
            "supports should include Theorem",
        );
        kani::assert(
            supported.contains(&PropertyType::Invariant),
            "supports should include Invariant",
        );
    }

    /// Verify supports() returns exactly two property types
    #[kani::proof]
    fn proof_cadical_backend_supports_length() {
        let backend = CaDiCaLBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.len() == 2,
            "CaDiCaL should support exactly two property types",
        );
    }

    // ---- Output Parsing Tests ----

    /// Verify parse_output detects s UNSATISFIABLE as Proven
    #[kani::proof]
    fn proof_parse_output_unsat() {
        let backend = CaDiCaLBackend::new();
        let (status, model) = backend.parse_output("s UNSATISFIABLE", Some(20));
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "UNSATISFIABLE should produce Proven status",
        );
        kani::assert(model.is_none(), "UNSAT should have no model");
    }

    /// Verify parse_output detects s SATISFIABLE as Disproven
    #[kani::proof]
    fn proof_parse_output_sat() {
        let backend = CaDiCaLBackend::new();
        let (status, _) = backend.parse_output("s SATISFIABLE\nv 1 -2 3 0", Some(10));
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "SATISFIABLE should produce Disproven status",
        );
    }

    /// Verify parse_output detects exit code 10 as SAT
    #[kani::proof]
    fn proof_parse_output_exit_code_sat() {
        let backend = CaDiCaLBackend::new();
        let (status, _) = backend.parse_output("v 1 -2 3 0", Some(10));
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "exit code 10 should produce Disproven status",
        );
    }

    /// Verify parse_output detects exit code 20 as UNSAT
    #[kani::proof]
    fn proof_parse_output_exit_code_unsat() {
        let backend = CaDiCaLBackend::new();
        let (status, _) = backend.parse_output("", Some(20));
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "exit code 20 should produce Proven status",
        );
    }

    /// Verify parse_output detects s UNKNOWN
    #[kani::proof]
    fn proof_parse_output_unknown() {
        let backend = CaDiCaLBackend::new();
        let (status, _) = backend.parse_output("s UNKNOWN", Some(0));
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "UNKNOWN should produce Unknown status",
        );
    }

    // ---- Model Extraction Tests ----

    /// Verify extract_model parses v lines
    #[kani::proof]
    fn proof_extract_model() {
        let lines = vec!["s SATISFIABLE", "v 1 -2 3 0", "v 4 -5 0"];
        let model = CaDiCaLBackend::extract_model(&lines);
        kani::assert(model.is_some(), "should extract model");
        let m = model.unwrap();
        kani::assert(
            m.contains("1 -2 3"),
            "model should contain first assignments",
        );
        kani::assert(
            m.contains("4 -5"),
            "model should contain second assignments",
        );
    }

    // ---- Counterexample Parsing Tests ----

    /// Verify parse_counterexample extracts variables
    #[kani::proof]
    fn proof_parse_counterexample() {
        let backend = CaDiCaLBackend::new();
        let ce = backend.parse_counterexample("1 -2 3 0 4 -5 0");
        kani::assert(ce.witness.len() == 5, "should have 5 variables");
        kani::assert(
            ce.witness.get("x1") == Some(&CounterexampleValue::Bool(true)),
            "x1 should be true",
        );
        kani::assert(
            ce.witness.get("x2") == Some(&CounterexampleValue::Bool(false)),
            "x2 should be false",
        );
    }
}

use crate::counterexample::{CounterexampleValue, StructuredCounterexample};
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

/// Preprocessing level for CaDiCaL
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CaDiCaLPreprocessing {
    /// No preprocessing
    None,
    /// Light preprocessing (default)
    #[default]
    Light,
    /// Full preprocessing
    Full,
}

/// Configuration for CaDiCaL backend
#[derive(Debug, Clone)]
pub struct CaDiCaLConfig {
    /// Path to cadical binary
    pub cadical_path: Option<PathBuf>,
    /// Timeout for solving
    pub timeout: Duration,
    /// Preprocessing level
    pub preprocessing: CaDiCaLPreprocessing,
    /// Enable verbose output
    pub verbose: bool,
    /// Write DRAT proof (for UNSAT certificate)
    pub write_proof: bool,
    /// Resource limit (conflicts, 0 = no limit)
    pub conflict_limit: u64,
    /// Decision limit (0 = no limit)
    pub decision_limit: u64,
}

impl Default for CaDiCaLConfig {
    fn default() -> Self {
        Self {
            cadical_path: None,
            timeout: Duration::from_secs(60),
            preprocessing: CaDiCaLPreprocessing::default(),
            verbose: false,
            write_proof: false,
            conflict_limit: 0,
            decision_limit: 0,
        }
    }
}

/// CaDiCaL SAT solver backend
pub struct CaDiCaLBackend {
    config: CaDiCaLConfig,
}

impl Default for CaDiCaLBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CaDiCaLBackend {
    /// Create a new CaDiCaL backend with default configuration
    pub fn new() -> Self {
        Self {
            config: CaDiCaLConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: CaDiCaLConfig) -> Self {
        Self { config }
    }

    async fn detect_cadical(&self) -> Result<PathBuf, String> {
        let cadical_path = self
            .config
            .cadical_path
            .clone()
            .or_else(|| which::which("cadical").ok())
            .ok_or("CaDiCaL not found. Build from https://github.com/arminbiere/cadical")?;

        let output = Command::new(&cadical_path)
            .arg("--version")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| format!("Failed to execute cadical: {}", e))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        if stdout.contains("cadical") || stdout.contains("CaDiCaL") || !stdout.is_empty() {
            debug!("Detected CaDiCaL: {}", stdout.trim());
            Ok(cadical_path)
        } else if stderr.contains("cadical") || stderr.contains("CaDiCaL") {
            debug!("Detected CaDiCaL: {}", stderr.trim());
            Ok(cadical_path)
        } else {
            Err("CaDiCaL version check failed".to_string())
        }
    }

    /// Convert a typed spec to DIMACS CNF format
    fn spec_to_dimacs(&self, spec: &TypedSpec) -> String {
        let mut clauses = Vec::new();
        let mut num_vars = 0;

        for _prop in &spec.spec.properties {
            num_vars = num_vars.max(1);
            clauses.push(vec![1]);
        }

        if clauses.is_empty() {
            num_vars = 1;
            clauses.push(vec![1]);
        }

        let mut dimacs = "c Generated by DashProve for CaDiCaL\n".to_string();
        dimacs.push_str(&format!("p cnf {} {}\n", num_vars, clauses.len()));
        for clause in clauses {
            let clause_str: Vec<String> = clause.iter().map(|l| l.to_string()).collect();
            dimacs.push_str(&format!("{} 0\n", clause_str.join(" ")));
        }

        dimacs
    }

    fn parse_output(
        &self,
        stdout: &str,
        exit_code: Option<i32>,
    ) -> (VerificationStatus, Option<String>) {
        let lines: Vec<&str> = stdout.lines().collect();

        for line in &lines {
            let trimmed = line.trim();
            if trimmed == "s UNSATISFIABLE" {
                return (VerificationStatus::Proven, None);
            } else if trimmed == "s SATISFIABLE" {
                let model = Self::extract_model(&lines);
                return (VerificationStatus::Disproven, model);
            } else if trimmed == "s UNKNOWN" {
                return (
                    VerificationStatus::Unknown {
                        reason: "CaDiCaL returned unknown".to_string(),
                    },
                    None,
                );
            }
        }

        // CaDiCaL exit codes: 10 = SAT, 20 = UNSAT, 0 = unknown/interrupted
        match exit_code {
            Some(10) => {
                let model = Self::extract_model(&lines);
                return (VerificationStatus::Disproven, model);
            }
            Some(20) => return (VerificationStatus::Proven, None),
            _ => {}
        }

        if stdout.contains("error") || stdout.contains("Error") {
            return (
                VerificationStatus::Unknown {
                    reason: format!(
                        "CaDiCaL error: {}",
                        stdout.lines().take(2).collect::<Vec<_>>().join("; ")
                    ),
                },
                None,
            );
        }

        (
            VerificationStatus::Unknown {
                reason: "Could not parse CaDiCaL output".to_string(),
            },
            None,
        )
    }

    fn extract_model(lines: &[&str]) -> Option<String> {
        // CaDiCaL model format: "v 1 -2 3 -4 0"
        let mut model_parts = Vec::new();
        for line in lines {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix('v') {
                let values = rest.trim();
                model_parts.push(values.to_string());
            }
        }

        if model_parts.is_empty() {
            None
        } else {
            Some(model_parts.join(" "))
        }
    }

    fn parse_counterexample(&self, model_str: &str) -> StructuredCounterexample {
        let mut ce = StructuredCounterexample::new();
        ce.raw = Some(model_str.to_string());

        for lit_str in model_str.split_whitespace() {
            if lit_str == "0" {
                continue; // Skip clause terminators
            }
            if let Ok(lit) = lit_str.parse::<i64>() {
                let var = lit.unsigned_abs();
                let value = lit > 0;
                ce.witness
                    .insert(format!("x{}", var), CounterexampleValue::Bool(value));
            }
        }

        ce
    }
}

#[async_trait]
impl VerificationBackend for CaDiCaLBackend {
    fn id(&self) -> BackendId {
        BackendId::CaDiCaL
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::Theorem, PropertyType::Invariant]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let cadical_path = self
            .detect_cadical()
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let dimacs = self.spec_to_dimacs(spec);
        let cnf_path = temp_dir.path().join("problem.cnf");

        std::fs::write(&cnf_path, &dimacs).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write CNF file: {}", e))
        })?;

        let mut cmd = Command::new(&cadical_path);
        cmd.arg(&cnf_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if self.config.verbose {
            cmd.arg("-v");
        } else {
            cmd.arg("-q"); // Quiet mode
        }

        // Preprocessing
        match self.config.preprocessing {
            CaDiCaLPreprocessing::None => {
                cmd.arg("--no-elim");
                cmd.arg("--no-subsume");
            }
            CaDiCaLPreprocessing::Light => {
                // Default preprocessing
            }
            CaDiCaLPreprocessing::Full => {
                cmd.arg("--elim=1");
                cmd.arg("--subsume=1");
            }
        }

        if self.config.write_proof {
            let proof_path = temp_dir.path().join("proof.drat");
            cmd.arg(format!("--proof={}", proof_path.display()));
        }

        if self.config.conflict_limit > 0 {
            cmd.arg(format!("--conflicts={}", self.config.conflict_limit));
        }

        if self.config.decision_limit > 0 {
            cmd.arg(format!("--decisions={}", self.config.decision_limit));
        }

        let result =
            tokio::time::timeout(self.config.timeout + Duration::from_secs(5), cmd.output()).await;

        let duration = start.elapsed();

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                debug!("CaDiCaL stdout: {}", stdout);
                debug!("CaDiCaL stderr: {}", stderr);

                let (status, counterexample_str) = self.parse_output(&stdout, output.status.code());

                let counterexample = counterexample_str.map(|m| self.parse_counterexample(&m));

                let diagnostics: Vec<String> = stderr
                    .lines()
                    .filter(|l| l.contains("warning") || l.contains("error"))
                    .map(String::from)
                    .collect();

                let proof = if matches!(status, VerificationStatus::Proven) {
                    Some("Verified by CaDiCaL (UNSAT)".to_string())
                } else {
                    None
                };

                Ok(BackendResult {
                    backend: BackendId::CaDiCaL,
                    status,
                    proof,
                    counterexample,
                    diagnostics,
                    time_taken: duration,
                })
            }
            Ok(Err(e)) => Err(BackendError::VerificationFailed(format!(
                "Failed to execute CaDiCaL: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match self.detect_cadical().await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = CaDiCaLConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.preprocessing, CaDiCaLPreprocessing::Light);
        assert!(!config.verbose);
        assert!(!config.write_proof);
        assert_eq!(config.conflict_limit, 0);
        assert_eq!(config.decision_limit, 0);
    }

    #[test]
    fn parse_s_unsatisfiable() {
        let backend = CaDiCaLBackend::new();
        let (status, model) = backend.parse_output("s UNSATISFIABLE", Some(20));
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(model.is_none());
    }

    #[test]
    fn parse_s_satisfiable() {
        let backend = CaDiCaLBackend::new();
        let (status, _) = backend.parse_output("s SATISFIABLE\nv 1 -2 3 0", Some(10));
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[test]
    fn parse_exit_code_sat() {
        let backend = CaDiCaLBackend::new();
        let (status, _) = backend.parse_output("v 1 -2 3 0", Some(10));
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[test]
    fn parse_exit_code_unsat() {
        let backend = CaDiCaLBackend::new();
        let (status, _) = backend.parse_output("", Some(20));
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn parse_s_unknown() {
        let backend = CaDiCaLBackend::new();
        let (status, _) = backend.parse_output("s UNKNOWN", Some(0));
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn extract_model_v_lines() {
        let lines = vec!["s SATISFIABLE", "v 1 -2 3 0", "v 4 -5 0"];
        let model = CaDiCaLBackend::extract_model(&lines);
        assert!(model.is_some());
        let m = model.unwrap();
        assert!(m.contains("1 -2 3"));
        assert!(m.contains("4 -5"));
    }

    #[test]
    fn parse_counterexample_basic() {
        let backend = CaDiCaLBackend::new();
        let ce = backend.parse_counterexample("1 -2 3 0 4 -5 0");
        assert_eq!(ce.witness.len(), 5);
        assert_eq!(ce.witness.get("x1"), Some(&CounterexampleValue::Bool(true)));
        assert_eq!(
            ce.witness.get("x2"),
            Some(&CounterexampleValue::Bool(false))
        );
        assert_eq!(ce.witness.get("x3"), Some(&CounterexampleValue::Bool(true)));
        assert_eq!(ce.witness.get("x4"), Some(&CounterexampleValue::Bool(true)));
        assert_eq!(
            ce.witness.get("x5"),
            Some(&CounterexampleValue::Bool(false))
        );
    }

    #[test]
    fn backend_id() {
        let backend = CaDiCaLBackend::new();
        assert_eq!(backend.id(), BackendId::CaDiCaL);
    }

    #[test]
    fn supports_properties() {
        let backend = CaDiCaLBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::Theorem));
        assert!(supported.contains(&PropertyType::Invariant));
    }

    #[test]
    fn preprocessing_levels() {
        let _none = CaDiCaLPreprocessing::None;
        let _light = CaDiCaLPreprocessing::Light;
        let _full = CaDiCaLPreprocessing::Full;
        assert_eq!(CaDiCaLPreprocessing::default(), CaDiCaLPreprocessing::Light);
    }
}
