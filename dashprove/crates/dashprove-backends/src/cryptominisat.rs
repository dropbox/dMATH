//! CryptoMiniSat SAT solver backend
//!
//! CryptoMiniSat is a SAT solver with advanced XOR clause reasoning,
//! particularly suited for cryptographic problems.
//!
//! Input format: DIMACS CNF (with optional XOR clause extensions)
//! Output format:
//! - s SATISFIABLE / s UNSATISFIABLE
//! - v 1 -2 3 0  (satisfying assignment)
//!
//! See: <https://github.com/msoos/cryptominisat>

// =============================================
// Kani Proofs for CryptoMiniSat Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- CryptoMiniSatConfig Default Tests ----

    /// Verify CryptoMiniSatConfig::default timeout is 60 seconds
    #[kani::proof]
    fn proof_cryptominisat_config_default_timeout() {
        let config = CryptoMiniSatConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(60),
            "Default timeout should be 60 seconds",
        );
    }

    /// Verify CryptoMiniSatConfig::default cryptominisat_path is None
    #[kani::proof]
    fn proof_cryptominisat_config_default_path_none() {
        let config = CryptoMiniSatConfig::default();
        kani::assert(
            config.cryptominisat_path.is_none(),
            "Default cryptominisat_path should be None",
        );
    }

    /// Verify CryptoMiniSatConfig::default threads is 1
    #[kani::proof]
    fn proof_cryptominisat_config_default_threads() {
        let config = CryptoMiniSatConfig::default();
        kani::assert(config.threads == 1, "Default threads should be 1");
    }

    /// Verify CryptoMiniSatConfig::default xor_reasoning is true
    #[kani::proof]
    fn proof_cryptominisat_config_default_xor_reasoning_true() {
        let config = CryptoMiniSatConfig::default();
        kani::assert(config.xor_reasoning, "Default xor_reasoning should be true");
    }

    /// Verify CryptoMiniSatConfig::default max_conflicts is 0
    #[kani::proof]
    fn proof_cryptominisat_config_default_max_conflicts() {
        let config = CryptoMiniSatConfig::default();
        kani::assert(
            config.max_conflicts == 0,
            "Default max_conflicts should be 0",
        );
    }

    /// Verify CryptoMiniSatConfig::default verbosity is 0
    #[kani::proof]
    fn proof_cryptominisat_config_default_verbosity() {
        let config = CryptoMiniSatConfig::default();
        kani::assert(config.verbosity == 0, "Default verbosity should be 0");
    }

    // ---- CryptoMiniSatBackend Construction Tests ----

    /// Verify CryptoMiniSatBackend::new uses default config timeout
    #[kani::proof]
    fn proof_cryptominisat_backend_new_default_timeout() {
        let backend = CryptoMiniSatBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(60),
            "New backend should use default timeout",
        );
    }

    /// Verify CryptoMiniSatBackend::default equals CryptoMiniSatBackend::new
    #[kani::proof]
    fn proof_cryptominisat_backend_default_equals_new() {
        let default_backend = CryptoMiniSatBackend::default();
        let new_backend = CryptoMiniSatBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "Default and new should have same timeout",
        );
    }

    /// Verify CryptoMiniSatBackend::with_config preserves custom timeout
    #[kani::proof]
    fn proof_cryptominisat_backend_with_config_timeout() {
        let config = CryptoMiniSatConfig {
            cryptominisat_path: None,
            timeout: Duration::from_secs(120),
            threads: 1,
            xor_reasoning: true,
            max_conflicts: 0,
            verbosity: 0,
        };
        let backend = CryptoMiniSatBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(120),
            "with_config should preserve timeout",
        );
    }

    /// Verify CryptoMiniSatBackend::with_config preserves threads
    #[kani::proof]
    fn proof_cryptominisat_backend_with_config_threads() {
        let config = CryptoMiniSatConfig {
            cryptominisat_path: None,
            timeout: Duration::from_secs(60),
            threads: 8,
            xor_reasoning: true,
            max_conflicts: 0,
            verbosity: 0,
        };
        let backend = CryptoMiniSatBackend::with_config(config);
        kani::assert(
            backend.config.threads == 8,
            "with_config should preserve threads",
        );
    }

    /// Verify CryptoMiniSatBackend::with_config preserves xor_reasoning
    #[kani::proof]
    fn proof_cryptominisat_backend_with_config_xor_reasoning() {
        let config = CryptoMiniSatConfig {
            cryptominisat_path: None,
            timeout: Duration::from_secs(60),
            threads: 1,
            xor_reasoning: false,
            max_conflicts: 0,
            verbosity: 0,
        };
        let backend = CryptoMiniSatBackend::with_config(config);
        kani::assert(
            !backend.config.xor_reasoning,
            "with_config should preserve xor_reasoning",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify CryptoMiniSatBackend::id returns CryptoMiniSat
    #[kani::proof]
    fn proof_cryptominisat_backend_id() {
        let backend = CryptoMiniSatBackend::new();
        kani::assert(
            backend.id() == BackendId::CryptoMiniSat,
            "Backend id should be CryptoMiniSat",
        );
    }

    /// Verify CryptoMiniSatBackend::supports includes Theorem
    #[kani::proof]
    fn proof_cryptominisat_backend_supports_theorem() {
        let backend = CryptoMiniSatBackend::new();
        let supported = backend.supports();
        let has_theorem = supported.iter().any(|p| *p == PropertyType::Theorem);
        kani::assert(has_theorem, "Should support Theorem property");
    }

    /// Verify CryptoMiniSatBackend::supports includes Invariant
    #[kani::proof]
    fn proof_cryptominisat_backend_supports_invariant() {
        let backend = CryptoMiniSatBackend::new();
        let supported = backend.supports();
        let has_invariant = supported.iter().any(|p| *p == PropertyType::Invariant);
        kani::assert(has_invariant, "Should support Invariant property");
    }

    /// Verify CryptoMiniSatBackend::supports returns exactly 2 properties
    #[kani::proof]
    fn proof_cryptominisat_backend_supports_length() {
        let backend = CryptoMiniSatBackend::new();
        let supported = backend.supports();
        kani::assert(supported.len() == 2, "Should support exactly 2 properties");
    }

    // ---- parse_output Tests ----

    /// Verify parse_output returns Proven for UNSATISFIABLE
    #[kani::proof]
    fn proof_parse_output_unsat() {
        let backend = CryptoMiniSatBackend::new();
        let (status, model) = backend.parse_output("s UNSATISFIABLE\n", "");
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "Should return Proven for UNSATISFIABLE",
        );
        kani::assert(model.is_none(), "Should return no model for UNSAT");
    }

    /// Verify parse_output returns Disproven for SATISFIABLE
    #[kani::proof]
    fn proof_parse_output_sat() {
        let backend = CryptoMiniSatBackend::new();
        let (status, model) = backend.parse_output("s SATISFIABLE\nv 1 -2 3 0\n", "");
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "Should return Disproven for SATISFIABLE",
        );
        kani::assert(model.is_some(), "Should return model for SAT");
    }

    /// Verify parse_output returns Unknown for UNKNOWN
    #[kani::proof]
    fn proof_parse_output_unknown() {
        let backend = CryptoMiniSatBackend::new();
        let (status, model) = backend.parse_output("s UNKNOWN\n", "");
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Should return Unknown for s UNKNOWN",
        );
        kani::assert(model.is_none(), "Should return no model for UNKNOWN");
    }

    /// Verify parse_output returns Unknown for INDETERMINATE
    #[kani::proof]
    fn proof_parse_output_indeterminate() {
        let backend = CryptoMiniSatBackend::new();
        let (status, _) = backend.parse_output("s INDETERMINATE\n", "");
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Should return Unknown for INDETERMINATE",
        );
    }

    /// Verify parse_output returns Unknown for TIMEOUT
    #[kani::proof]
    fn proof_parse_output_timeout() {
        let backend = CryptoMiniSatBackend::new();
        let (status, _) = backend.parse_output("TIMEOUT\n", "");
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Should return Unknown for TIMEOUT",
        );
    }

    /// Verify parse_output returns Unknown for empty output
    #[kani::proof]
    fn proof_parse_output_empty() {
        let backend = CryptoMiniSatBackend::new();
        let (status, model) = backend.parse_output("", "");
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Should return Unknown for empty output",
        );
        kani::assert(model.is_none(), "Should return no model for empty output");
    }
}

use crate::counterexample::{CounterexampleValue, StructuredCounterexample};
use crate::kissat::DimacsCnf;
use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
    VerificationStatus,
};
use async_trait::async_trait;
use dashprove_usl::typecheck::TypedSpec;
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::process::Command;
use tracing::debug;

/// Configuration for CryptoMiniSat backend
#[derive(Debug, Clone)]
pub struct CryptoMiniSatConfig {
    /// Path to CryptoMiniSat binary
    pub cryptominisat_path: Option<PathBuf>,
    /// Timeout for solving
    pub timeout: Duration,
    /// Number of threads
    pub threads: u32,
    /// Enable XOR reasoning (Gauss-Jordan elimination)
    pub xor_reasoning: bool,
    /// Maximum number of conflicts (0 = no limit)
    pub max_conflicts: u64,
    /// Verbosity level (0-5)
    pub verbosity: u32,
}

impl Default for CryptoMiniSatConfig {
    fn default() -> Self {
        Self {
            cryptominisat_path: None,
            timeout: Duration::from_secs(60),
            threads: 1,
            xor_reasoning: true,
            max_conflicts: 0,
            verbosity: 0,
        }
    }
}

/// CryptoMiniSat solver backend
pub struct CryptoMiniSatBackend {
    config: CryptoMiniSatConfig,
}

impl Default for CryptoMiniSatBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CryptoMiniSatBackend {
    pub fn new() -> Self {
        Self {
            config: CryptoMiniSatConfig::default(),
        }
    }

    pub fn with_config(config: CryptoMiniSatConfig) -> Self {
        Self { config }
    }

    async fn detect_cryptominisat(&self) -> Result<PathBuf, String> {
        let cms_path = self
            .config
            .cryptominisat_path
            .clone()
            .or_else(|| which::which("cryptominisat5").ok())
            .or_else(|| which::which("cryptominisat").ok())
            .ok_or(
                "CryptoMiniSat not found. Install via pip install pycryptosat or build from source",
            )?;

        let output = Command::new(&cms_path)
            .arg("--version")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| format!("Failed to execute CryptoMiniSat: {}", e))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined = format!("{}{}", stdout, stderr);

        if combined.contains("CryptoMiniSat")
            || combined.contains("cryptominisat")
            || output.status.success()
        {
            debug!("Detected CryptoMiniSat");
            Ok(cms_path)
        } else {
            Err("CryptoMiniSat detection failed".to_string())
        }
    }

    /// Parse CryptoMiniSat output to extract verification result and model
    fn parse_output(&self, stdout: &str, _stderr: &str) -> (VerificationStatus, Option<Vec<i32>>) {
        // Check for UNSAT (property holds)
        if stdout.contains("s UNSATISFIABLE") {
            return (VerificationStatus::Proven, None);
        }

        // Check for SAT (counterexample exists)
        if stdout.contains("s SATISFIABLE") {
            let mut model = Vec::new();
            for line in stdout.lines() {
                let trimmed = line.trim();
                if let Some(rest) = trimmed.strip_prefix('v') {
                    // Parse literals from the line
                    let parts: Vec<&str> = rest.split_whitespace().collect();
                    for part in parts {
                        if let Ok(lit) = part.parse::<i32>() {
                            if lit != 0 {
                                model.push(lit);
                            }
                        }
                    }
                }
            }
            return (VerificationStatus::Disproven, Some(model));
        }

        // Check for UNKNOWN
        if stdout.contains("s UNKNOWN")
            || stdout.contains("s INDETERMINATE")
            || stdout.contains("TIMEOUT")
        {
            return (
                VerificationStatus::Unknown {
                    reason: "CryptoMiniSat returned unknown (timeout or resource limit)"
                        .to_string(),
                },
                None,
            );
        }

        // Default
        (
            VerificationStatus::Unknown {
                reason: "Could not parse CryptoMiniSat output".to_string(),
            },
            None,
        )
    }

    /// Parse model into structured counterexample
    fn parse_counterexample(
        &self,
        model: &[i32],
        var_names: &HashMap<u32, String>,
    ) -> StructuredCounterexample {
        let mut ce = StructuredCounterexample::new();

        // Build raw string
        let model_str: String = model.iter().map(|l| format!("{} ", l)).collect();
        ce.raw = Some(format!("SAT model: {}", model_str.trim()));

        // Convert model to named variables
        for &lit in model {
            let var_num = lit.unsigned_abs();
            let value = lit > 0;

            if let Some(name) = var_names.get(&var_num) {
                ce.witness
                    .insert(name.clone(), CounterexampleValue::Bool(value));
            } else {
                ce.witness
                    .insert(format!("x{}", var_num), CounterexampleValue::Bool(value));
            }
        }

        ce
    }

    /// Compile a USL spec to DIMACS CNF
    ///
    /// Uses the same CNF generation as Kissat.
    fn compile_to_dimacs(&self, spec: &TypedSpec) -> Result<DimacsCnf, String> {
        use dashprove_usl::ast::Property;

        let mut cnf = DimacsCnf::new();

        if spec.spec.properties.is_empty() {
            return Err("No properties to verify".to_string());
        }

        // Add variables for each property
        for (i, property) in spec.spec.properties.iter().enumerate() {
            let name = property.name();
            let kind = match property {
                Property::Theorem(_) => "thm",
                Property::Invariant(_) => "inv",
                Property::Contract(_) => "ctr",
                Property::Temporal(_) => "tmp",
                Property::Refinement(_) => "ref",
                Property::Probabilistic(_) => "prob",
                Property::Security(_) => "sec",
                Property::Semantic(_) => "sem",
                Property::PlatformApi(_) => "api",
                Property::Bisimulation(_) => "bisim",
                Property::Version(_) => "ver",
                Property::Capability(_) => "cap",
                Property::DistributedInvariant(_) => "dinv",
                Property::DistributedTemporal(_) => "dtemp",
                Property::Composed(_) => "comp",
                Property::ImprovementProposal(_) => "imp",
                Property::VerificationGate(_) => "gate",
                Property::Rollback(_) => "rollback",
            };

            let var = cnf.add_variable(&format!("{}_{}", kind, name));
            cnf.add_clause(vec![var as i32]);

            let aux = cnf.add_variable(&format!("aux_{}_{}", kind, i));
            cnf.add_clause(vec![-(var as i32), aux as i32]);
        }

        if cnf.clauses.is_empty() {
            let var = cnf.add_variable("trivial");
            cnf.add_clause(vec![var as i32]);
            cnf.add_clause(vec![-(var as i32)]);
        }

        Ok(cnf)
    }
}

#[async_trait]
impl VerificationBackend for CryptoMiniSatBackend {
    fn id(&self) -> BackendId {
        BackendId::CryptoMiniSat
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::Theorem, PropertyType::Invariant]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let cms_path = self
            .detect_cryptominisat()
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        // Compile to DIMACS CNF
        let cnf = self.compile_to_dimacs(spec).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to compile to DIMACS: {}", e))
        })?;

        let cnf_path = temp_dir.path().join("spec.cnf");
        std::fs::write(&cnf_path, cnf.to_dimacs()).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write CNF file: {}", e))
        })?;

        // Build CryptoMiniSat command
        let mut cmd = Command::new(&cms_path);
        cmd.arg(&cnf_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Add configuration options
        if self.config.threads > 1 {
            cmd.arg(format!("--threads={}", self.config.threads));
        }
        if !self.config.xor_reasoning {
            cmd.arg("--gaussuntil=0"); // Disable Gauss-Jordan
        }
        if self.config.max_conflicts > 0 {
            cmd.arg(format!("--maxconfl={}", self.config.max_conflicts));
        }
        if self.config.verbosity > 0 {
            cmd.arg(format!("--verb={}", self.config.verbosity));
        }

        // Execute with timeout
        let result =
            tokio::time::timeout(self.config.timeout + Duration::from_secs(5), cmd.output()).await;

        let duration = start.elapsed();

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                debug!("CryptoMiniSat stdout: {}", stdout);
                debug!("CryptoMiniSat stderr: {}", stderr);

                let (status, model) = self.parse_output(&stdout, &stderr);

                let counterexample = model.map(|m| self.parse_counterexample(&m, &cnf.var_names));

                let diagnostics: Vec<String> = stderr
                    .lines()
                    .filter(|l| l.contains("warning") || l.contains("error") || l.contains("ERROR"))
                    .map(String::from)
                    .collect();

                let proof = if matches!(status, VerificationStatus::Proven) {
                    Some("Verified by CryptoMiniSat (UNSAT)".to_string())
                } else {
                    None
                };

                Ok(BackendResult {
                    backend: BackendId::CryptoMiniSat,
                    status,
                    proof,
                    counterexample,
                    diagnostics,
                    time_taken: duration,
                })
            }
            Ok(Err(e)) => Err(BackendError::VerificationFailed(format!(
                "Failed to execute CryptoMiniSat: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match self.detect_cryptominisat().await {
            Ok(_) => HealthStatus::Healthy,
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
        let config = CryptoMiniSatConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.threads, 1);
        assert!(config.xor_reasoning);
        assert_eq!(config.max_conflicts, 0);
        assert_eq!(config.verbosity, 0);
    }

    #[test]
    fn custom_config() {
        let config = CryptoMiniSatConfig {
            cryptominisat_path: Some(PathBuf::from("/usr/bin/cryptominisat5")),
            timeout: Duration::from_secs(120),
            threads: 4,
            xor_reasoning: false,
            max_conflicts: 100000,
            verbosity: 2,
        };
        assert_eq!(config.timeout, Duration::from_secs(120));
        assert_eq!(config.threads, 4);
        assert!(!config.xor_reasoning);
    }

    #[test]
    fn backend_id() {
        let backend = CryptoMiniSatBackend::new();
        assert_eq!(backend.id(), BackendId::CryptoMiniSat);
    }

    #[test]
    fn backend_supports() {
        let backend = CryptoMiniSatBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::Theorem));
        assert!(supported.contains(&PropertyType::Invariant));
    }

    // =============================================
    // Output parsing tests
    // =============================================

    #[test]
    fn parse_unsat() {
        let backend = CryptoMiniSatBackend::new();
        let (status, model) = backend.parse_output("s UNSATISFIABLE\n", "");
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(model.is_none());
    }

    #[test]
    fn parse_sat_simple() {
        let backend = CryptoMiniSatBackend::new();
        let output = "s SATISFIABLE\nv 1 -2 3 0\n";
        let (status, model) = backend.parse_output(output, "");
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(model.is_some());
        let m = model.unwrap();
        assert_eq!(m, vec![1, -2, 3]);
    }

    #[test]
    fn parse_sat_multiline() {
        let backend = CryptoMiniSatBackend::new();
        let output = "s SATISFIABLE\nv 1 -2 3\nv 4 -5 0\n";
        let (status, model) = backend.parse_output(output, "");
        assert!(matches!(status, VerificationStatus::Disproven));
        let m = model.unwrap();
        assert_eq!(m, vec![1, -2, 3, 4, -5]);
    }

    #[test]
    fn parse_sat_with_header() {
        let backend = CryptoMiniSatBackend::new();
        let output = "c CryptoMiniSat version 5.11.11\nc Using DIMACS\ns SATISFIABLE\nv 1 2 -3 0\n";
        let (status, model) = backend.parse_output(output, "");
        assert!(matches!(status, VerificationStatus::Disproven));
        let m = model.unwrap();
        assert_eq!(m, vec![1, 2, -3]);
    }

    #[test]
    fn parse_unknown() {
        let backend = CryptoMiniSatBackend::new();
        let (status, model) = backend.parse_output("s UNKNOWN\n", "");
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
        assert!(model.is_none());
    }

    #[test]
    fn parse_indeterminate() {
        let backend = CryptoMiniSatBackend::new();
        let (status, _) = backend.parse_output("s INDETERMINATE\n", "");
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn parse_timeout() {
        let backend = CryptoMiniSatBackend::new();
        let (status, _) = backend.parse_output("TIMEOUT\n", "");
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn parse_empty_output() {
        let backend = CryptoMiniSatBackend::new();
        let (status, model) = backend.parse_output("", "");
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
        assert!(model.is_none());
    }

    // =============================================
    // Counterexample parsing tests
    // =============================================

    #[test]
    fn parse_counterexample_simple() {
        let backend = CryptoMiniSatBackend::new();
        let model = vec![1, -2, 3];
        let mut var_names = HashMap::new();
        var_names.insert(1, "x".to_string());
        var_names.insert(2, "y".to_string());
        var_names.insert(3, "z".to_string());

        let ce = backend.parse_counterexample(&model, &var_names);

        assert!(ce.witness.contains_key("x"));
        assert!(ce.witness.contains_key("y"));
        assert!(ce.witness.contains_key("z"));

        assert_eq!(ce.witness["x"], CounterexampleValue::Bool(true));
        assert_eq!(ce.witness["y"], CounterexampleValue::Bool(false));
        assert_eq!(ce.witness["z"], CounterexampleValue::Bool(true));
    }

    #[test]
    fn parse_counterexample_unnamed_vars() {
        let backend = CryptoMiniSatBackend::new();
        let model = vec![1, -2];
        let var_names = HashMap::new();

        let ce = backend.parse_counterexample(&model, &var_names);

        assert!(ce.witness.contains_key("x1"));
        assert!(ce.witness.contains_key("x2"));
    }

    #[test]
    fn parse_counterexample_has_raw() {
        let backend = CryptoMiniSatBackend::new();
        let model = vec![1, -2];
        let var_names = HashMap::new();

        let ce = backend.parse_counterexample(&model, &var_names);

        assert!(ce.raw.is_some());
        assert!(ce.raw.unwrap().contains("SAT model"));
    }

    #[test]
    fn parse_counterexample_summary() {
        let backend = CryptoMiniSatBackend::new();
        let model = vec![1, -2];
        let mut var_names = HashMap::new();
        var_names.insert(1, "p".to_string());
        var_names.insert(2, "q".to_string());

        let ce = backend.parse_counterexample(&model, &var_names);
        let summary = ce.summary();

        assert!(summary.contains("Witness:"));
    }

    // =============================================
    // Integration-style tests
    // =============================================

    #[test]
    fn full_output_unsat_verification() {
        let backend = CryptoMiniSatBackend::new();

        let output = r#"c CryptoMiniSat version 5.11.11
c Reading DIMACS file 'spec.cnf'
c -- simplification round 1 --
c Parsing finished.
c -- occur-based preprocessing
c Preprocessing finished
s UNSATISFIABLE
"#;

        let (status, _) = backend.parse_output(output, "");
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn full_output_sat_verification() {
        let backend = CryptoMiniSatBackend::new();

        let output = r#"c CryptoMiniSat version 5.11.11
c Reading DIMACS file 'spec.cnf'
s SATISFIABLE
v 1 -2 3 -4 5 0
"#;

        let (status, model) = backend.parse_output(output, "");
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(model.is_some());
        assert_eq!(model.unwrap(), vec![1, -2, 3, -4, 5]);
    }

    #[test]
    fn full_output_sat_long_model() {
        let backend = CryptoMiniSatBackend::new();

        let output = r#"s SATISFIABLE
v 1 2 3 4 5 6 7 8 9 10
v 11 12 -13 14 -15 -16 17 18 -19 20
v 21 0
"#;

        let (status, model) = backend.parse_output(output, "");
        assert!(matches!(status, VerificationStatus::Disproven));
        let m = model.unwrap();
        assert_eq!(m.len(), 21);
        assert_eq!(m[12], -13);
        assert_eq!(m[14], -15);
        assert_eq!(m[15], -16);
    }

    // =============================================
    // Edge cases and configuration
    // =============================================

    #[test]
    fn backend_with_custom_config() {
        let config = CryptoMiniSatConfig {
            timeout: Duration::from_secs(300),
            threads: 8,
            xor_reasoning: true,
            max_conflicts: 500000,
            verbosity: 1,
            ..Default::default()
        };
        let backend = CryptoMiniSatBackend::with_config(config);
        assert_eq!(backend.config.timeout, Duration::from_secs(300));
        assert_eq!(backend.config.threads, 8);
    }

    #[test]
    fn parse_output_with_warnings() {
        let backend = CryptoMiniSatBackend::new();
        let (status, _) = backend.parse_output("s SATISFIABLE\nv 1 0\n", "c WARNING: Large CNF\n");
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[test]
    fn parse_output_xor_extension() {
        // CryptoMiniSat supports XOR clauses in extended DIMACS
        let backend = CryptoMiniSatBackend::new();

        let output = r#"c Using XOR-clause extension
c Gaussian elimination performed
s UNSATISFIABLE
"#;

        let (status, _) = backend.parse_output(output, "");
        assert!(matches!(status, VerificationStatus::Proven));
    }
}
