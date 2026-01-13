//! MathSAT SMT solver backend
//!
//! MathSAT is an efficient SMT solver from Fondazione Bruno Kessler (FBK).
//! It supports a rich set of theories including linear arithmetic, arrays,
//! bit-vectors, and floating-point arithmetic.
//!
//! See: <https://mathsat.fbk.eu/>

use crate::counterexample::{CounterexampleValue, StructuredCounterexample};
use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
    VerificationStatus,
};
use async_trait::async_trait;
use dashprove_usl::compile_to_smtlib2_with_logic;
use dashprove_usl::typecheck::TypedSpec;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::process::Command;
use tracing::debug;

/// Theory configuration for MathSAT
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MathSatTheory {
    /// Automatic theory selection (default)
    #[default]
    Auto,
    /// Quantifier-free linear integer arithmetic
    QFLIA,
    /// Quantifier-free linear real arithmetic
    QFLRA,
    /// Quantifier-free linear integer and real arithmetic
    QFLIRA,
    /// Quantifier-free bit-vectors
    QFBV,
    /// Quantifier-free arrays with bit-vectors
    QFABV,
    /// Quantifier-free floating-point arithmetic
    QFFP,
}

impl MathSatTheory {
    fn as_logic(&self) -> &'static str {
        match self {
            MathSatTheory::Auto => "ALL",
            MathSatTheory::QFLIA => "QF_LIA",
            MathSatTheory::QFLRA => "QF_LRA",
            MathSatTheory::QFLIRA => "QF_LIRA",
            MathSatTheory::QFBV => "QF_BV",
            MathSatTheory::QFABV => "QF_ABV",
            MathSatTheory::QFFP => "QF_FP",
        }
    }
}

/// Configuration for MathSAT backend
#[derive(Debug, Clone)]
pub struct MathSatConfig {
    /// Path to mathsat binary
    pub mathsat_path: Option<PathBuf>,
    /// Timeout for verification
    pub timeout: Duration,
    /// Theory to use
    pub theory: MathSatTheory,
    /// Enable model generation
    pub produce_models: bool,
    /// Enable proof generation
    pub produce_proofs: bool,
    /// Enable interpolant generation
    pub produce_interpolants: bool,
    /// Random seed for solver
    pub random_seed: Option<u64>,
}

impl Default for MathSatConfig {
    fn default() -> Self {
        Self {
            mathsat_path: None,
            timeout: Duration::from_secs(60),
            theory: MathSatTheory::default(),
            produce_models: true,
            produce_proofs: false,
            produce_interpolants: false,
            random_seed: None,
        }
    }
}

/// MathSAT SMT solver backend
pub struct MathSatBackend {
    config: MathSatConfig,
}

impl Default for MathSatBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl MathSatBackend {
    /// Create a new MathSAT backend with default configuration
    pub fn new() -> Self {
        Self {
            config: MathSatConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: MathSatConfig) -> Self {
        Self { config }
    }

    async fn detect_mathsat(&self) -> Result<PathBuf, String> {
        let mathsat_path = self
            .config
            .mathsat_path
            .clone()
            .or_else(|| which::which("mathsat").ok())
            .ok_or("MathSAT not found. Download from https://mathsat.fbk.eu/download.html")?;

        let output = Command::new(&mathsat_path)
            .arg("-version")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| format!("Failed to execute mathsat: {}", e))?;

        if output.status.success() {
            let version = String::from_utf8_lossy(&output.stdout);
            debug!("Detected MathSAT version: {}", version.trim());
            Ok(mathsat_path)
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            if stderr.contains("MathSAT") {
                debug!("Detected MathSAT: {}", stderr.trim());
                Ok(mathsat_path)
            } else {
                Err("MathSAT version check failed".to_string())
            }
        }
    }

    fn parse_output(
        &self,
        stdout: &str,
        stderr: &str,
        success: bool,
    ) -> (VerificationStatus, Option<String>) {
        let combined = format!("{}\n{}", stdout, stderr);
        let lines: Vec<&str> = combined.lines().collect();

        for line in &lines {
            let trimmed = line.trim();
            if trimmed == "unsat" {
                return (VerificationStatus::Proven, None);
            } else if trimmed == "sat" {
                let model = Self::extract_model(&lines);
                return (VerificationStatus::Disproven, model);
            } else if trimmed == "unknown" {
                return (
                    VerificationStatus::Unknown {
                        reason: "MathSAT returned unknown".to_string(),
                    },
                    None,
                );
            }
        }

        if !success || combined.contains("error") || combined.contains("Error") {
            return (
                VerificationStatus::Unknown {
                    reason: format!(
                        "MathSAT error: {}",
                        combined.lines().take(3).collect::<Vec<_>>().join("; ")
                    ),
                },
                None,
            );
        }

        (
            VerificationStatus::Unknown {
                reason: "Could not parse MathSAT output".to_string(),
            },
            None,
        )
    }

    fn extract_model(lines: &[&str]) -> Option<String> {
        let mut model_lines = Vec::new();
        let mut in_model = false;
        let mut depth = 0;

        for line in lines {
            let trimmed = line.trim();
            if trimmed.starts_with("(model") || trimmed == "(" {
                in_model = true;
            }
            if in_model {
                model_lines.push(*line);
                depth += line.matches('(').count();
                depth = depth.saturating_sub(line.matches(')').count());
                if depth == 0 && !model_lines.is_empty() && model_lines.len() > 1 {
                    break;
                }
            }
        }

        if model_lines.is_empty() {
            None
        } else {
            Some(model_lines.join("\n"))
        }
    }

    fn parse_counterexample(&self, model_str: &str) -> StructuredCounterexample {
        let mut ce = StructuredCounterexample::new();
        ce.raw = Some(model_str.to_string());

        // Parse SMT-LIB2 model format
        let definitions = Self::extract_definitions(model_str);
        for (name, _sort, value) in definitions {
            let parsed_value = Self::parse_value(&value);
            ce.witness.insert(name, parsed_value);
        }

        ce
    }

    fn extract_definitions(model_str: &str) -> Vec<(String, String, String)> {
        let mut definitions = Vec::new();

        for line in model_str.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("(define-fun") {
                // Parse "(define-fun name () type value)"
                let parts: Vec<&str> = trimmed
                    .trim_start_matches("(define-fun")
                    .trim_end_matches(')')
                    .split_whitespace()
                    .collect();

                if parts.len() >= 4 {
                    let name = parts[0].to_string();
                    // Skip "()" - parts[1] and parts[2] would be "(" and ")"
                    let sort = parts.get(3).unwrap_or(&"Unknown").to_string();
                    let value = parts.last().unwrap_or(&"").to_string();
                    definitions.push((name, sort, value));
                }
            }
        }

        definitions
    }

    fn parse_value(value: &str) -> CounterexampleValue {
        let trimmed = value.trim().trim_matches(|c| c == '(' || c == ')');

        if trimmed == "true" {
            return CounterexampleValue::Bool(true);
        }
        if trimmed == "false" {
            return CounterexampleValue::Bool(false);
        }

        // Integer
        if let Ok(n) = trimmed.parse::<i128>() {
            return CounterexampleValue::Int {
                value: n,
                type_hint: Some("Int".to_string()),
            };
        }

        // Negative integer (- n)
        if trimmed.starts_with("-") {
            if let Ok(n) = trimmed.parse::<i128>() {
                return CounterexampleValue::Int {
                    value: n,
                    type_hint: Some("Int".to_string()),
                };
            }
        }

        // Float/Real
        if let Ok(f) = trimmed.parse::<f64>() {
            return CounterexampleValue::Float { value: f };
        }

        // BitVector binary: #b1010
        if let Some(binary_str) = trimmed.strip_prefix("#b") {
            if let Ok(n) = i128::from_str_radix(binary_str, 2) {
                return CounterexampleValue::Int {
                    value: n,
                    type_hint: Some("BitVec".to_string()),
                };
            }
        }

        // BitVector hex: #x0A
        if let Some(hex_str) = trimmed.strip_prefix("#x") {
            if let Ok(n) = i128::from_str_radix(hex_str, 16) {
                return CounterexampleValue::Int {
                    value: n,
                    type_hint: Some("BitVec".to_string()),
                };
            }
        }

        // Floating-point: (fp ...)
        if trimmed.starts_with("fp") || trimmed.contains("fp.") {
            return CounterexampleValue::Unknown(format!("FloatingPoint: {}", trimmed));
        }

        CounterexampleValue::Unknown(trimmed.to_string())
    }
}

#[async_trait]
impl VerificationBackend for MathSatBackend {
    fn id(&self) -> BackendId {
        BackendId::MathSAT
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![
            PropertyType::Theorem,
            PropertyType::Invariant,
            PropertyType::Contract,
        ]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let mathsat_path = self
            .detect_mathsat()
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        // Compile to SMT-LIB2
        let logic = self.config.theory.as_logic();
        let compiled = compile_to_smtlib2_with_logic(spec, logic);
        let smt_path = temp_dir.path().join("spec.smt2");
        std::fs::write(&smt_path, &compiled.code).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write SMT-LIB2 file: {}", e))
        })?;

        let mut cmd = Command::new(&mathsat_path);
        cmd.arg("-input=smt2")
            .arg(&smt_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if self.config.produce_models {
            cmd.arg("-model");
        }

        if self.config.produce_proofs {
            cmd.arg("-proof");
        }

        if let Some(seed) = self.config.random_seed {
            cmd.arg(format!("-random_seed={}", seed));
        }

        let timeout_ms = self.config.timeout.as_millis();
        cmd.arg(format!("-timeout={}", timeout_ms));

        let result =
            tokio::time::timeout(self.config.timeout + Duration::from_secs(5), cmd.output()).await;

        let duration = start.elapsed();

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                debug!("MathSAT stdout: {}", stdout);
                debug!("MathSAT stderr: {}", stderr);

                let (status, counterexample_str) =
                    self.parse_output(&stdout, &stderr, output.status.success());

                let counterexample = counterexample_str.map(|m| self.parse_counterexample(&m));

                let diagnostics: Vec<String> = stderr
                    .lines()
                    .filter(|l| l.contains("warning") || l.contains("error"))
                    .map(String::from)
                    .collect();

                let proof = if matches!(status, VerificationStatus::Proven) {
                    Some("Verified by MathSAT (unsat)".to_string())
                } else {
                    None
                };

                Ok(BackendResult {
                    backend: BackendId::MathSAT,
                    status,
                    proof,
                    counterexample,
                    diagnostics,
                    time_taken: duration,
                })
            }
            Ok(Err(e)) => Err(BackendError::VerificationFailed(format!(
                "Failed to execute MathSAT: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match self.detect_mathsat().await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ===== MathSatTheory as_logic proofs =====

    #[kani::proof]
    fn verify_theory_auto_logic() {
        assert!(MathSatTheory::Auto.as_logic() == "ALL");
    }

    #[kani::proof]
    fn verify_theory_qflia_logic() {
        assert!(MathSatTheory::QFLIA.as_logic() == "QF_LIA");
    }

    #[kani::proof]
    fn verify_theory_qflra_logic() {
        assert!(MathSatTheory::QFLRA.as_logic() == "QF_LRA");
    }

    #[kani::proof]
    fn verify_theory_qflira_logic() {
        assert!(MathSatTheory::QFLIRA.as_logic() == "QF_LIRA");
    }

    #[kani::proof]
    fn verify_theory_qfbv_logic() {
        assert!(MathSatTheory::QFBV.as_logic() == "QF_BV");
    }

    #[kani::proof]
    fn verify_theory_qfabv_logic() {
        assert!(MathSatTheory::QFABV.as_logic() == "QF_ABV");
    }

    #[kani::proof]
    fn verify_theory_qffp_logic() {
        assert!(MathSatTheory::QFFP.as_logic() == "QF_FP");
    }

    #[kani::proof]
    fn verify_theory_default_is_auto() {
        let theory = MathSatTheory::default();
        assert!(matches!(theory, MathSatTheory::Auto));
    }

    // ===== MathSatConfig default proofs =====

    #[kani::proof]
    fn verify_config_default_timeout() {
        let config = MathSatConfig::default();
        assert!(config.timeout.as_secs() == 60);
    }

    #[kani::proof]
    fn verify_config_default_theory() {
        let config = MathSatConfig::default();
        assert!(matches!(config.theory, MathSatTheory::Auto));
    }

    #[kani::proof]
    fn verify_config_default_produce_models() {
        let config = MathSatConfig::default();
        assert!(config.produce_models);
    }

    #[kani::proof]
    fn verify_config_default_produce_proofs_false() {
        let config = MathSatConfig::default();
        assert!(!config.produce_proofs);
    }

    #[kani::proof]
    fn verify_config_default_produce_interpolants_false() {
        let config = MathSatConfig::default();
        assert!(!config.produce_interpolants);
    }

    #[kani::proof]
    fn verify_config_default_random_seed_none() {
        let config = MathSatConfig::default();
        assert!(config.random_seed.is_none());
    }

    #[kani::proof]
    fn verify_config_default_mathsat_path_none() {
        let config = MathSatConfig::default();
        assert!(config.mathsat_path.is_none());
    }

    // ===== MathSatBackend construction proofs =====

    #[kani::proof]
    fn verify_backend_new_has_default_config() {
        let backend = MathSatBackend::new();
        assert!(backend.config.timeout.as_secs() == 60);
        assert!(backend.config.produce_models);
    }

    #[kani::proof]
    fn verify_backend_default_equals_new() {
        let b1 = MathSatBackend::new();
        let b2 = MathSatBackend::default();
        assert!(b1.config.timeout == b2.config.timeout);
        assert!(b1.config.produce_models == b2.config.produce_models);
    }

    #[kani::proof]
    fn verify_backend_with_config_preserves_timeout() {
        let config = MathSatConfig {
            mathsat_path: None,
            timeout: Duration::from_secs(120),
            theory: MathSatTheory::QFLIA,
            produce_models: false,
            produce_proofs: true,
            produce_interpolants: false,
            random_seed: Some(42),
        };
        let backend = MathSatBackend::with_config(config);
        assert!(backend.config.timeout.as_secs() == 120);
    }

    #[kani::proof]
    fn verify_backend_with_config_preserves_theory() {
        let config = MathSatConfig {
            mathsat_path: None,
            timeout: Duration::from_secs(60),
            theory: MathSatTheory::QFBV,
            produce_models: true,
            produce_proofs: false,
            produce_interpolants: false,
            random_seed: None,
        };
        let backend = MathSatBackend::with_config(config);
        assert!(matches!(backend.config.theory, MathSatTheory::QFBV));
    }

    // ===== Backend ID proof =====

    #[kani::proof]
    fn verify_backend_id() {
        let backend = MathSatBackend::new();
        assert!(matches!(backend.id(), BackendId::MathSAT));
    }

    // ===== Supports proofs =====

    #[kani::proof]
    fn verify_supports_theorem() {
        let backend = MathSatBackend::new();
        let supported = backend.supports();
        let has_theorem = supported.iter().any(|p| matches!(p, PropertyType::Theorem));
        assert!(has_theorem);
    }

    #[kani::proof]
    fn verify_supports_invariant() {
        let backend = MathSatBackend::new();
        let supported = backend.supports();
        let has_invariant = supported
            .iter()
            .any(|p| matches!(p, PropertyType::Invariant));
        assert!(has_invariant);
    }

    #[kani::proof]
    fn verify_supports_contract() {
        let backend = MathSatBackend::new();
        let supported = backend.supports();
        let has_contract = supported
            .iter()
            .any(|p| matches!(p, PropertyType::Contract));
        assert!(has_contract);
    }

    #[kani::proof]
    fn verify_supports_count() {
        let backend = MathSatBackend::new();
        let supported = backend.supports();
        assert!(supported.len() == 3);
    }

    // ===== parse_output proofs =====

    #[kani::proof]
    fn verify_parse_output_unsat_proven() {
        let backend = MathSatBackend::new();
        let (status, model) = backend.parse_output("unsat", "", true);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(model.is_none());
    }

    #[kani::proof]
    fn verify_parse_output_sat_disproven() {
        let backend = MathSatBackend::new();
        let (status, _) = backend.parse_output("sat", "", true);
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[kani::proof]
    fn verify_parse_output_unknown_returns_unknown() {
        let backend = MathSatBackend::new();
        let (status, _) = backend.parse_output("unknown", "", true);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[kani::proof]
    fn verify_parse_output_error_returns_unknown() {
        let backend = MathSatBackend::new();
        let (status, _) = backend.parse_output("", "error: syntax", false);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[kani::proof]
    fn verify_parse_output_empty_not_success_unknown() {
        let backend = MathSatBackend::new();
        let (status, _) = backend.parse_output("", "", false);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[kani::proof]
    fn verify_parse_output_unsat_in_stderr() {
        let backend = MathSatBackend::new();
        let (status, _) = backend.parse_output("", "unsat", true);
        assert!(matches!(status, VerificationStatus::Proven));
    }

    // ===== extract_model proofs =====

    #[kani::proof]
    fn verify_extract_model_empty_returns_none() {
        let lines: Vec<&str> = vec![];
        let model = MathSatBackend::extract_model(&lines);
        assert!(model.is_none());
    }

    #[kani::proof]
    fn verify_extract_model_no_model_marker() {
        let lines = vec!["sat", "some output"];
        let model = MathSatBackend::extract_model(&lines);
        assert!(model.is_none());
    }

    // ===== parse_value proofs =====

    #[kani::proof]
    fn verify_parse_value_true() {
        match MathSatBackend::parse_value("true") {
            CounterexampleValue::Bool(b) => assert!(b),
            _ => panic!("Expected Bool"),
        }
    }

    #[kani::proof]
    fn verify_parse_value_false() {
        match MathSatBackend::parse_value("false") {
            CounterexampleValue::Bool(b) => assert!(!b),
            _ => panic!("Expected Bool"),
        }
    }

    #[kani::proof]
    fn verify_parse_value_positive_int() {
        match MathSatBackend::parse_value("42") {
            CounterexampleValue::Int { value, .. } => assert!(value == 42),
            _ => panic!("Expected Int"),
        }
    }

    #[kani::proof]
    fn verify_parse_value_negative_int() {
        match MathSatBackend::parse_value("-42") {
            CounterexampleValue::Int { value, .. } => assert!(value == -42),
            _ => panic!("Expected Int"),
        }
    }

    #[kani::proof]
    fn verify_parse_value_zero() {
        match MathSatBackend::parse_value("0") {
            CounterexampleValue::Int { value, .. } => assert!(value == 0),
            _ => panic!("Expected Int"),
        }
    }

    #[kani::proof]
    fn verify_parse_value_binary_bitvec() {
        match MathSatBackend::parse_value("#b1010") {
            CounterexampleValue::Int { value, type_hint } => {
                assert!(value == 10);
                assert!(type_hint == Some("BitVec".to_string()));
            }
            _ => panic!("Expected Int"),
        }
    }

    #[kani::proof]
    fn verify_parse_value_hex_bitvec() {
        match MathSatBackend::parse_value("#xFF") {
            CounterexampleValue::Int { value, type_hint } => {
                assert!(value == 255);
                assert!(type_hint == Some("BitVec".to_string()));
            }
            _ => panic!("Expected Int"),
        }
    }

    #[kani::proof]
    fn verify_parse_value_floating_point_unknown() {
        match MathSatBackend::parse_value("(fp #b0 #x7F #x000000)") {
            CounterexampleValue::Unknown(s) => assert!(s.contains("FloatingPoint")),
            _ => panic!("Expected Unknown"),
        }
    }

    #[kani::proof]
    fn verify_parse_value_unknown_string() {
        match MathSatBackend::parse_value("some_identifier") {
            CounterexampleValue::Unknown(s) => assert!(s == "some_identifier"),
            _ => panic!("Expected Unknown"),
        }
    }

    // ===== extract_definitions proofs =====

    #[kani::proof]
    fn verify_extract_definitions_empty() {
        let defs = MathSatBackend::extract_definitions("");
        assert!(defs.is_empty());
    }

    #[kani::proof]
    fn verify_extract_definitions_no_define_fun() {
        let defs = MathSatBackend::extract_definitions("sat\n(model)");
        assert!(defs.is_empty());
    }

    // ===== parse_counterexample proofs =====

    #[kani::proof]
    fn verify_parse_counterexample_empty_model() {
        let backend = MathSatBackend::new();
        let cex = backend.parse_counterexample("");
        assert!(cex.witness.is_empty());
    }

    #[kani::proof]
    fn verify_parse_counterexample_preserves_raw() {
        let backend = MathSatBackend::new();
        let cex = backend.parse_counterexample("raw_model");
        assert!(cex.raw == Some("raw_model".to_string()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = MathSatConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.theory, MathSatTheory::Auto);
        assert!(config.produce_models);
        assert!(!config.produce_proofs);
        assert!(!config.produce_interpolants);
        assert!(config.random_seed.is_none());
    }

    #[test]
    fn theory_as_logic() {
        assert_eq!(MathSatTheory::Auto.as_logic(), "ALL");
        assert_eq!(MathSatTheory::QFLIA.as_logic(), "QF_LIA");
        assert_eq!(MathSatTheory::QFLRA.as_logic(), "QF_LRA");
        assert_eq!(MathSatTheory::QFLIRA.as_logic(), "QF_LIRA");
        assert_eq!(MathSatTheory::QFBV.as_logic(), "QF_BV");
        assert_eq!(MathSatTheory::QFABV.as_logic(), "QF_ABV");
        assert_eq!(MathSatTheory::QFFP.as_logic(), "QF_FP");
    }

    #[test]
    fn parse_unsat() {
        let backend = MathSatBackend::new();
        let (status, model) = backend.parse_output("unsat", "", true);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(model.is_none());
    }

    #[test]
    fn parse_sat() {
        let backend = MathSatBackend::new();
        let (status, _) = backend.parse_output("sat", "", true);
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[test]
    fn parse_unknown() {
        let backend = MathSatBackend::new();
        let (status, _) = backend.parse_output("unknown", "", true);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn parse_error() {
        let backend = MathSatBackend::new();
        let (status, _) = backend.parse_output("", "error: syntax", false);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn parse_value_bool() {
        assert_eq!(
            MathSatBackend::parse_value("true"),
            CounterexampleValue::Bool(true)
        );
        assert_eq!(
            MathSatBackend::parse_value("false"),
            CounterexampleValue::Bool(false)
        );
    }

    #[test]
    fn parse_value_int() {
        match MathSatBackend::parse_value("42") {
            CounterexampleValue::Int { value, .. } => assert_eq!(value, 42),
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn parse_value_negative_int() {
        match MathSatBackend::parse_value("-42") {
            CounterexampleValue::Int { value, .. } => assert_eq!(value, -42),
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn parse_value_float() {
        match MathSatBackend::parse_value("1.234") {
            CounterexampleValue::Float { value } => assert!((value - 1.234).abs() < 0.001),
            _ => panic!("Expected Float"),
        }
    }

    #[test]
    fn parse_value_bitvec_binary() {
        match MathSatBackend::parse_value("#b1010") {
            CounterexampleValue::Int { value, type_hint } => {
                assert_eq!(value, 10);
                assert_eq!(type_hint, Some("BitVec".to_string()));
            }
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn parse_value_bitvec_hex() {
        match MathSatBackend::parse_value("#xFF") {
            CounterexampleValue::Int { value, type_hint } => {
                assert_eq!(value, 255);
                assert_eq!(type_hint, Some("BitVec".to_string()));
            }
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn parse_value_floating_point() {
        match MathSatBackend::parse_value("(fp #b0 #x7F #x000000)") {
            CounterexampleValue::Unknown(s) => assert!(s.contains("FloatingPoint")),
            _ => panic!("Expected Unknown"),
        }
    }

    #[test]
    fn extract_model_basic() {
        let lines = vec!["sat", "(model", "  (define-fun x () Int 5)", ")"];
        let model = MathSatBackend::extract_model(&lines);
        assert!(model.is_some());
    }

    #[test]
    fn backend_id() {
        let backend = MathSatBackend::new();
        assert_eq!(backend.id(), BackendId::MathSAT);
    }

    #[test]
    fn supports_properties() {
        let backend = MathSatBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::Theorem));
        assert!(supported.contains(&PropertyType::Invariant));
        assert!(supported.contains(&PropertyType::Contract));
    }
}
