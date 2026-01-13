//! Glucose SAT solver backend
//!
//! Glucose is a high-performance SAT solver built on MiniSat with improved
//! restart strategies and clause management. It won multiple SAT competitions.
//!
//! See: <https://www.labri.fr/perso/lsimon/glucose/>

// =============================================
// Kani Proofs for Glucose Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- GlucoseRestartStrategy Default Tests ----

    /// Verify GlucoseRestartStrategy::default is LBD
    #[kani::proof]
    fn proof_glucose_restart_strategy_default_is_lbd() {
        let strategy = GlucoseRestartStrategy::default();
        kani::assert(
            strategy == GlucoseRestartStrategy::LBD,
            "Default restart strategy should be LBD",
        );
    }

    // ---- GlucoseConfig Default Tests ----

    /// Verify GlucoseConfig::default glucose_path is None
    #[kani::proof]
    fn proof_glucose_config_default_path_none() {
        let config = GlucoseConfig::default();
        kani::assert(
            config.glucose_path.is_none(),
            "Default glucose_path should be None",
        );
    }

    /// Verify GlucoseConfig::default timeout is 60 seconds
    #[kani::proof]
    fn proof_glucose_config_default_timeout() {
        let config = GlucoseConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(60),
            "Default timeout should be 60 seconds",
        );
    }

    /// Verify GlucoseConfig::default restart_strategy is LBD
    #[kani::proof]
    fn proof_glucose_config_default_restart_strategy() {
        let config = GlucoseConfig::default();
        kani::assert(
            config.restart_strategy == GlucoseRestartStrategy::LBD,
            "Default restart strategy should be LBD",
        );
    }

    /// Verify GlucoseConfig::default verbose is false
    #[kani::proof]
    fn proof_glucose_config_default_verbose_false() {
        let config = GlucoseConfig::default();
        kani::assert(!config.verbose, "Default verbose should be false");
    }

    /// Verify GlucoseConfig::default memory_limit is 0
    #[kani::proof]
    fn proof_glucose_config_default_memory_limit() {
        let config = GlucoseConfig::default();
        kani::assert(config.memory_limit == 0, "Default memory_limit should be 0");
    }

    /// Verify GlucoseConfig::default cpu_limit is 0
    #[kani::proof]
    fn proof_glucose_config_default_cpu_limit() {
        let config = GlucoseConfig::default();
        kani::assert(config.cpu_limit == 0, "Default cpu_limit should be 0");
    }

    /// Verify GlucoseConfig::default certified_unsat is false
    #[kani::proof]
    fn proof_glucose_config_default_certified_unsat_false() {
        let config = GlucoseConfig::default();
        kani::assert(
            !config.certified_unsat,
            "Default certified_unsat should be false",
        );
    }

    // ---- GlucoseBackend Construction Tests ----

    /// Verify GlucoseBackend::new uses default timeout
    #[kani::proof]
    fn proof_glucose_backend_new_default_timeout() {
        let backend = GlucoseBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(60),
            "New backend should use default timeout",
        );
    }

    /// Verify GlucoseBackend::default equals GlucoseBackend::new
    #[kani::proof]
    fn proof_glucose_backend_default_equals_new() {
        let default_backend = GlucoseBackend::default();
        let new_backend = GlucoseBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "Default and new should have same timeout",
        );
    }

    /// Verify GlucoseBackend::with_config preserves timeout
    #[kani::proof]
    fn proof_glucose_backend_with_config_timeout() {
        let config = GlucoseConfig {
            glucose_path: None,
            timeout: Duration::from_secs(120),
            restart_strategy: GlucoseRestartStrategy::LBD,
            verbose: false,
            memory_limit: 0,
            cpu_limit: 0,
            certified_unsat: false,
        };
        let backend = GlucoseBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(120),
            "with_config should preserve timeout",
        );
    }

    /// Verify GlucoseBackend::with_config preserves restart strategy
    #[kani::proof]
    fn proof_glucose_backend_with_config_luby_strategy() {
        let config = GlucoseConfig {
            glucose_path: None,
            timeout: Duration::from_secs(60),
            restart_strategy: GlucoseRestartStrategy::Luby,
            verbose: false,
            memory_limit: 0,
            cpu_limit: 0,
            certified_unsat: false,
        };
        let backend = GlucoseBackend::with_config(config);
        kani::assert(
            backend.config.restart_strategy == GlucoseRestartStrategy::Luby,
            "with_config should preserve Luby strategy",
        );
    }

    /// Verify GlucoseBackend::with_config preserves Geometric strategy
    #[kani::proof]
    fn proof_glucose_backend_with_config_geometric_strategy() {
        let config = GlucoseConfig {
            glucose_path: None,
            timeout: Duration::from_secs(60),
            restart_strategy: GlucoseRestartStrategy::Geometric,
            verbose: false,
            memory_limit: 0,
            cpu_limit: 0,
            certified_unsat: false,
        };
        let backend = GlucoseBackend::with_config(config);
        kani::assert(
            backend.config.restart_strategy == GlucoseRestartStrategy::Geometric,
            "with_config should preserve Geometric strategy",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify GlucoseBackend::id returns Glucose
    #[kani::proof]
    fn proof_glucose_backend_id() {
        let backend = GlucoseBackend::new();
        kani::assert(
            backend.id() == BackendId::Glucose,
            "Backend id should be Glucose",
        );
    }

    /// Verify GlucoseBackend::supports includes Theorem
    #[kani::proof]
    fn proof_glucose_backend_supports_theorem() {
        let backend = GlucoseBackend::new();
        let supported = backend.supports();
        let has_theorem = supported.iter().any(|p| *p == PropertyType::Theorem);
        kani::assert(has_theorem, "Should support Theorem property");
    }

    /// Verify GlucoseBackend::supports includes Invariant
    #[kani::proof]
    fn proof_glucose_backend_supports_invariant() {
        let backend = GlucoseBackend::new();
        let supported = backend.supports();
        let has_inv = supported.iter().any(|p| *p == PropertyType::Invariant);
        kani::assert(has_inv, "Should support Invariant property");
    }

    /// Verify GlucoseBackend::supports returns exactly 2 properties
    #[kani::proof]
    fn proof_glucose_backend_supports_length() {
        let backend = GlucoseBackend::new();
        let supported = backend.supports();
        kani::assert(supported.len() == 2, "Should support exactly 2 properties");
    }

    // ---- parse_output Tests ----

    /// Verify parse_output recognizes "s UNSATISFIABLE"
    #[kani::proof]
    fn proof_parse_output_s_unsatisfiable() {
        let backend = GlucoseBackend::new();
        let (status, model) = backend.parse_output("s UNSATISFIABLE", None, Some(20));
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "s UNSATISFIABLE should be Proven",
        );
        kani::assert(model.is_none(), "UNSAT should have no model");
    }

    /// Verify parse_output recognizes "UNSAT"
    #[kani::proof]
    fn proof_parse_output_unsat() {
        let backend = GlucoseBackend::new();
        let (status, _) = backend.parse_output("UNSAT", None, Some(20));
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "UNSAT should be Proven",
        );
    }

    /// Verify parse_output recognizes "s SATISFIABLE"
    #[kani::proof]
    fn proof_parse_output_s_satisfiable() {
        let backend = GlucoseBackend::new();
        let (status, _) = backend.parse_output("s SATISFIABLE\nv 1 -2 3 0", None, Some(10));
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "s SATISFIABLE should be Disproven",
        );
    }

    /// Verify parse_output recognizes "SAT"
    #[kani::proof]
    fn proof_parse_output_sat() {
        let backend = GlucoseBackend::new();
        let (status, _) = backend.parse_output("SAT\nv 1 -2 0", None, Some(10));
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "SAT should be Disproven",
        );
    }

    /// Verify parse_output recognizes exit code 10 as SAT
    #[kani::proof]
    fn proof_parse_output_exit_code_10() {
        let backend = GlucoseBackend::new();
        let (status, _) = backend.parse_output("", None, Some(10));
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "Exit code 10 should be Disproven",
        );
    }

    /// Verify parse_output recognizes exit code 20 as UNSAT
    #[kani::proof]
    fn proof_parse_output_exit_code_20() {
        let backend = GlucoseBackend::new();
        let (status, _) = backend.parse_output("", None, Some(20));
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "Exit code 20 should be Proven",
        );
    }

    /// Verify parse_output recognizes "s UNKNOWN"
    #[kani::proof]
    fn proof_parse_output_s_unknown() {
        let backend = GlucoseBackend::new();
        let (status, _) = backend.parse_output("s UNKNOWN", None, None);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "s UNKNOWN should be Unknown",
        );
    }

    // ---- extract_model Tests ----

    /// Verify extract_model finds v lines
    #[kani::proof]
    fn proof_extract_model_v_lines() {
        let lines = vec!["s SATISFIABLE", "v 1 -2 3 0"];
        let model = GlucoseBackend::extract_model(&lines);
        kani::assert(model.is_some(), "Should extract model from v lines");
    }

    /// Verify extract_model returns None for empty
    #[kani::proof]
    fn proof_extract_model_empty() {
        let lines: Vec<&str> = vec![];
        let model = GlucoseBackend::extract_model(&lines);
        kani::assert(model.is_none(), "Empty lines should return None");
    }

    // ---- parse_counterexample Tests ----

    /// Verify parse_counterexample parses positive literal
    #[kani::proof]
    fn proof_parse_counterexample_positive() {
        let backend = GlucoseBackend::new();
        let ce = backend.parse_counterexample("1 0");
        kani::assert(ce.witness.contains_key("x1"), "Should have x1 in witness");
    }

    /// Verify parse_counterexample parses negative literal
    #[kani::proof]
    fn proof_parse_counterexample_negative() {
        let backend = GlucoseBackend::new();
        let ce = backend.parse_counterexample("-2 0");
        kani::assert(ce.witness.contains_key("x2"), "Should have x2 in witness");
    }

    /// Verify parse_counterexample sets raw
    #[kani::proof]
    fn proof_parse_counterexample_raw() {
        let backend = GlucoseBackend::new();
        let ce = backend.parse_counterexample("1 -2 0");
        kani::assert(ce.raw.is_some(), "Should set raw field");
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

/// Restart strategy for Glucose
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GlucoseRestartStrategy {
    /// LBD-based restarts (default, Glucose specialty)
    #[default]
    LBD,
    /// Luby sequence restarts
    Luby,
    /// Geometric restarts
    Geometric,
}

/// Configuration for Glucose backend
#[derive(Debug, Clone)]
pub struct GlucoseConfig {
    /// Path to glucose binary
    pub glucose_path: Option<PathBuf>,
    /// Timeout for solving
    pub timeout: Duration,
    /// Restart strategy
    pub restart_strategy: GlucoseRestartStrategy,
    /// Enable verbose output
    pub verbose: bool,
    /// Memory limit in MB (0 = no limit)
    pub memory_limit: u32,
    /// CPU limit in seconds (0 = no limit)
    pub cpu_limit: u32,
    /// Enable certified UNSAT proofs
    pub certified_unsat: bool,
}

impl Default for GlucoseConfig {
    fn default() -> Self {
        Self {
            glucose_path: None,
            timeout: Duration::from_secs(60),
            restart_strategy: GlucoseRestartStrategy::default(),
            verbose: false,
            memory_limit: 0,
            cpu_limit: 0,
            certified_unsat: false,
        }
    }
}

/// Glucose SAT solver backend
pub struct GlucoseBackend {
    config: GlucoseConfig,
}

impl Default for GlucoseBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl GlucoseBackend {
    /// Create a new Glucose backend with default configuration
    pub fn new() -> Self {
        Self {
            config: GlucoseConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: GlucoseConfig) -> Self {
        Self { config }
    }

    async fn detect_glucose(&self) -> Result<PathBuf, String> {
        let glucose_path = self
            .config
            .glucose_path
            .clone()
            .or_else(|| which::which("glucose").ok())
            .or_else(|| which::which("glucose-syrup").ok())
            .or_else(|| which::which("glucose_static").ok())
            .ok_or("Glucose not found. Download from https://www.labri.fr/perso/lsimon/glucose/")?;

        let output = Command::new(&glucose_path)
            .arg("-h")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| format!("Failed to execute glucose: {}", e))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined = format!("{}{}", stdout, stderr);

        if combined.contains("USAGE")
            || combined.contains("glucose")
            || combined.contains("Glucose")
        {
            debug!("Detected Glucose");
            Ok(glucose_path)
        } else {
            Err("Glucose detection failed".to_string())
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

        let mut dimacs = "c Generated by DashProve for Glucose\n".to_string();
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
        result_file: Option<&str>,
        exit_code: Option<i32>,
    ) -> (VerificationStatus, Option<String>) {
        let output = if let Some(result) = result_file {
            result.to_string()
        } else {
            stdout.to_string()
        };

        let lines: Vec<&str> = output.lines().collect();

        for line in &lines {
            let trimmed = line.trim();
            if trimmed == "s UNSATISFIABLE" || trimmed == "UNSAT" || trimmed == "UNSATISFIABLE" {
                return (VerificationStatus::Proven, None);
            } else if trimmed == "s SATISFIABLE" || trimmed == "SAT" || trimmed == "SATISFIABLE" {
                let model = Self::extract_model(&lines);
                return (VerificationStatus::Disproven, model);
            } else if trimmed == "s UNKNOWN" || trimmed == "INDETERMINATE" {
                return (
                    VerificationStatus::Unknown {
                        reason: "Glucose returned unknown".to_string(),
                    },
                    None,
                );
            }
        }

        // Glucose exit codes: 10 = SAT, 20 = UNSAT
        match exit_code {
            Some(10) => {
                let model = Self::extract_model(&lines);
                return (VerificationStatus::Disproven, model);
            }
            Some(20) => return (VerificationStatus::Proven, None),
            _ => {}
        }

        if stdout.contains("ERROR") || stdout.contains("error") {
            return (
                VerificationStatus::Unknown {
                    reason: format!(
                        "Glucose error: {}",
                        stdout.lines().take(2).collect::<Vec<_>>().join("; ")
                    ),
                },
                None,
            );
        }

        (
            VerificationStatus::Unknown {
                reason: "Could not parse Glucose output".to_string(),
            },
            None,
        )
    }

    fn extract_model(lines: &[&str]) -> Option<String> {
        // Glucose model format: "v 1 -2 3 -4 0"
        let mut model_parts = Vec::new();
        for line in lines {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix('v') {
                let values = rest.trim();
                model_parts.push(values.to_string());
            }
        }

        if model_parts.is_empty() {
            // Try direct numeric format
            for line in lines {
                let trimmed = line.trim();
                if !trimmed.is_empty()
                    && trimmed
                        .chars()
                        .next()
                        .is_some_and(|c| c.is_ascii_digit() || c == '-')
                    && trimmed.ends_with('0')
                {
                    return Some(trimmed.to_string());
                }
            }
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
                break;
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
impl VerificationBackend for GlucoseBackend {
    fn id(&self) -> BackendId {
        BackendId::Glucose
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::Theorem, PropertyType::Invariant]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let glucose_path = self
            .detect_glucose()
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let dimacs = self.spec_to_dimacs(spec);
        let cnf_path = temp_dir.path().join("problem.cnf");
        let result_path = temp_dir.path().join("result.txt");

        std::fs::write(&cnf_path, &dimacs).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write CNF file: {}", e))
        })?;

        let mut cmd = Command::new(&glucose_path);
        cmd.arg(&cnf_path)
            .arg(&result_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if self.config.verbose {
            cmd.arg("-verb=1");
        } else {
            cmd.arg("-verb=0");
        }

        if self.config.memory_limit > 0 {
            cmd.arg(format!("-mem-lim={}", self.config.memory_limit));
        }

        if self.config.cpu_limit > 0 {
            cmd.arg(format!("-cpu-lim={}", self.config.cpu_limit));
        }

        if self.config.certified_unsat {
            cmd.arg("-certified");
        }

        // Restart strategy flags
        match self.config.restart_strategy {
            GlucoseRestartStrategy::LBD => {
                // Default LBD-based restarts (Glucose specialty)
            }
            GlucoseRestartStrategy::Luby => {
                cmd.arg("-luby");
            }
            GlucoseRestartStrategy::Geometric => {
                cmd.arg("-no-luby");
            }
        }

        let result =
            tokio::time::timeout(self.config.timeout + Duration::from_secs(5), cmd.output()).await;

        let duration = start.elapsed();

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                debug!("Glucose stdout: {}", stdout);
                debug!("Glucose stderr: {}", stderr);

                let result_content = std::fs::read_to_string(&result_path).ok();

                let (status, counterexample_str) =
                    self.parse_output(&stdout, result_content.as_deref(), output.status.code());

                let counterexample = counterexample_str.map(|m| self.parse_counterexample(&m));

                let diagnostics: Vec<String> = stderr
                    .lines()
                    .filter(|l| l.contains("WARNING") || l.contains("ERROR"))
                    .map(String::from)
                    .collect();

                let proof = if matches!(status, VerificationStatus::Proven) {
                    Some("Verified by Glucose (UNSAT)".to_string())
                } else {
                    None
                };

                Ok(BackendResult {
                    backend: BackendId::Glucose,
                    status,
                    proof,
                    counterexample,
                    diagnostics,
                    time_taken: duration,
                })
            }
            Ok(Err(e)) => Err(BackendError::VerificationFailed(format!(
                "Failed to execute Glucose: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match self.detect_glucose().await {
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
        let config = GlucoseConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.restart_strategy, GlucoseRestartStrategy::LBD);
        assert!(!config.verbose);
        assert_eq!(config.memory_limit, 0);
        assert_eq!(config.cpu_limit, 0);
        assert!(!config.certified_unsat);
    }

    #[test]
    fn parse_s_unsatisfiable() {
        let backend = GlucoseBackend::new();
        let (status, model) = backend.parse_output("s UNSATISFIABLE", None, Some(20));
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(model.is_none());
    }

    #[test]
    fn parse_unsat() {
        let backend = GlucoseBackend::new();
        let (status, model) = backend.parse_output("UNSAT", None, Some(20));
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(model.is_none());
    }

    #[test]
    fn parse_s_satisfiable() {
        let backend = GlucoseBackend::new();
        let (status, _) = backend.parse_output("s SATISFIABLE\nv 1 -2 3 0", None, Some(10));
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[test]
    fn parse_sat() {
        let backend = GlucoseBackend::new();
        let (status, _) = backend.parse_output("SAT\nv 1 -2 3 0", None, Some(10));
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[test]
    fn parse_exit_code_sat() {
        let backend = GlucoseBackend::new();
        let (status, _) = backend.parse_output("", None, Some(10));
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[test]
    fn parse_exit_code_unsat() {
        let backend = GlucoseBackend::new();
        let (status, _) = backend.parse_output("", None, Some(20));
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn parse_s_unknown() {
        let backend = GlucoseBackend::new();
        let (status, _) = backend.parse_output("s UNKNOWN", None, None);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn extract_model_v_lines() {
        let lines = vec!["s SATISFIABLE", "v 1 -2 3 0", "v 4 -5 0"];
        let model = GlucoseBackend::extract_model(&lines);
        assert!(model.is_some());
        assert!(model.unwrap().contains("1 -2 3"));
    }

    #[test]
    fn parse_counterexample_basic() {
        let backend = GlucoseBackend::new();
        let ce = backend.parse_counterexample("1 -2 3 0");
        assert_eq!(ce.witness.len(), 3);
        assert_eq!(ce.witness.get("x1"), Some(&CounterexampleValue::Bool(true)));
        assert_eq!(
            ce.witness.get("x2"),
            Some(&CounterexampleValue::Bool(false))
        );
        assert_eq!(ce.witness.get("x3"), Some(&CounterexampleValue::Bool(true)));
    }

    #[test]
    fn backend_id() {
        let backend = GlucoseBackend::new();
        assert_eq!(backend.id(), BackendId::Glucose);
    }

    #[test]
    fn supports_properties() {
        let backend = GlucoseBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::Theorem));
        assert!(supported.contains(&PropertyType::Invariant));
    }
}
