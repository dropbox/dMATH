//! Boolector SMT solver backend
//!
//! Boolector is an efficient SMT solver specialized for bit-vectors and arrays.
//! It excels at hardware verification and bounded model checking problems.
//!
//! See: <https://boolector.github.io/>

// =============================================
// Kani Proofs for Boolector Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- BoolectorEngine Tests ----

    /// Verify BoolectorEngine::as_str returns correct strings
    #[kani::proof]
    fn proof_boolector_engine_as_str() {
        kani::assert(
            BoolectorEngine::CaDiCaL.as_str() == "cadical",
            "CaDiCaL should return 'cadical'",
        );
        kani::assert(
            BoolectorEngine::Lingeling.as_str() == "lingeling",
            "Lingeling should return 'lingeling'",
        );
        kani::assert(
            BoolectorEngine::MiniSat.as_str() == "minisat",
            "MiniSat should return 'minisat'",
        );
        kani::assert(
            BoolectorEngine::PicoSat.as_str() == "picosat",
            "PicoSat should return 'picosat'",
        );
    }

    /// Verify BoolectorEngine default is CaDiCaL
    #[kani::proof]
    fn proof_boolector_engine_default() {
        let engine = BoolectorEngine::default();
        kani::assert(
            engine == BoolectorEngine::CaDiCaL,
            "default engine should be CaDiCaL",
        );
    }

    // ---- BoolectorConfig Default Tests ----

    /// Verify BoolectorConfig::default sets correct baseline values
    #[kani::proof]
    fn proof_boolector_config_defaults() {
        let config = BoolectorConfig::default();
        kani::assert(
            config.boolector_path.is_none(),
            "boolector_path should default to None",
        );
        kani::assert(
            config.timeout == Duration::from_secs(60),
            "timeout should default to 60 seconds",
        );
        kani::assert(config.logic == "QF_BV", "logic should default to QF_BV");
        kani::assert(
            config.engine == BoolectorEngine::CaDiCaL,
            "engine should default to CaDiCaL",
        );
        kani::assert(
            config.produce_models,
            "produce_models should default to true",
        );
        kani::assert(
            config.simplify_level == 3,
            "simplify_level should default to 3",
        );
    }

    // ---- BoolectorBackend Construction Tests ----

    /// Verify BoolectorBackend::new uses default configuration
    #[kani::proof]
    fn proof_boolector_backend_new_defaults() {
        let backend = BoolectorBackend::new();
        kani::assert(
            backend.config.boolector_path.is_none(),
            "new backend should have no boolector_path",
        );
        kani::assert(
            backend.config.timeout == Duration::from_secs(60),
            "new backend should default timeout to 60 seconds",
        );
        kani::assert(
            backend.config.logic == "QF_BV",
            "new backend should default logic to QF_BV",
        );
    }

    /// Verify BoolectorBackend::default equals BoolectorBackend::new
    #[kani::proof]
    fn proof_boolector_backend_default_equals_new() {
        let default_backend = BoolectorBackend::default();
        let new_backend = BoolectorBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "default and new should share timeout",
        );
        kani::assert(
            default_backend.config.logic == new_backend.config.logic,
            "default and new should share logic",
        );
    }

    /// Verify BoolectorBackend::with_config preserves custom configuration
    #[kani::proof]
    fn proof_boolector_backend_with_config() {
        let config = BoolectorConfig {
            boolector_path: Some(PathBuf::from("/opt/boolector")),
            timeout: Duration::from_secs(120),
            logic: "QF_ABV".to_string(),
            engine: BoolectorEngine::Lingeling,
            produce_models: false,
            simplify_level: 4,
        };
        let backend = BoolectorBackend::with_config(config);
        kani::assert(
            backend.config.boolector_path == Some(PathBuf::from("/opt/boolector")),
            "with_config should preserve boolector_path",
        );
        kani::assert(
            backend.config.timeout == Duration::from_secs(120),
            "with_config should preserve timeout",
        );
        kani::assert(
            backend.config.logic == "QF_ABV",
            "with_config should preserve logic",
        );
        kani::assert(
            backend.config.engine == BoolectorEngine::Lingeling,
            "with_config should preserve engine",
        );
        kani::assert(
            !backend.config.produce_models,
            "with_config should preserve produce_models",
        );
        kani::assert(
            backend.config.simplify_level == 4,
            "with_config should preserve simplify_level",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify id() returns BackendId::Boolector
    #[kani::proof]
    fn proof_boolector_backend_id() {
        let backend = BoolectorBackend::new();
        kani::assert(
            backend.id() == BackendId::Boolector,
            "BoolectorBackend id should be BackendId::Boolector",
        );
    }

    /// Verify supports() includes Theorem, Invariant, and Contract
    #[kani::proof]
    fn proof_boolector_backend_supports() {
        let backend = BoolectorBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::Theorem),
            "supports should include Theorem",
        );
        kani::assert(
            supported.contains(&PropertyType::Invariant),
            "supports should include Invariant",
        );
        kani::assert(
            supported.contains(&PropertyType::Contract),
            "supports should include Contract",
        );
    }

    /// Verify supports() returns exactly three property types
    #[kani::proof]
    fn proof_boolector_backend_supports_length() {
        let backend = BoolectorBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.len() == 3,
            "Boolector should support exactly three property types",
        );
    }

    // ---- Output Parsing Tests ----

    /// Verify parse_output detects unsat as Proven
    #[kani::proof]
    fn proof_parse_output_unsat() {
        let backend = BoolectorBackend::new();
        let (status, model) = backend.parse_output("unsat", "", true);
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "unsat should produce Proven status",
        );
        kani::assert(model.is_none(), "unsat should have no model");
    }

    /// Verify parse_output detects sat as Disproven
    #[kani::proof]
    fn proof_parse_output_sat() {
        let backend = BoolectorBackend::new();
        let (status, _) = backend.parse_output("sat", "", true);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "sat should produce Disproven status",
        );
    }

    /// Verify parse_output detects unknown
    #[kani::proof]
    fn proof_parse_output_unknown() {
        let backend = BoolectorBackend::new();
        let (status, _) = backend.parse_output("unknown", "", true);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "unknown should produce Unknown status",
        );
    }

    /// Verify parse_output detects errors
    #[kani::proof]
    fn proof_parse_output_error() {
        let backend = BoolectorBackend::new();
        let (status, _) = backend.parse_output("", "error: parse error", false);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "error should produce Unknown status",
        );
    }

    // ---- Value Parsing Tests ----

    /// Verify parse_value parses booleans
    #[kani::proof]
    fn proof_parse_value_bool() {
        kani::assert(
            BoolectorBackend::parse_value("true") == CounterexampleValue::Bool(true),
            "should parse true",
        );
        kani::assert(
            BoolectorBackend::parse_value("false") == CounterexampleValue::Bool(false),
            "should parse false",
        );
    }

    /// Verify parse_value parses binary bitvectors
    #[kani::proof]
    fn proof_parse_value_bitvec_binary() {
        let value = BoolectorBackend::parse_value("#b1010");
        match value {
            CounterexampleValue::Int { value, type_hint } => {
                kani::assert(value == 10, "binary 1010 should be 10");
                kani::assert(
                    type_hint.is_some() && type_hint.unwrap().contains("BitVec"),
                    "should have BitVec type hint",
                );
            }
            _ => kani::assert(false, "should be Int variant"),
        }
    }

    /// Verify parse_value parses hex bitvectors
    #[kani::proof]
    fn proof_parse_value_bitvec_hex() {
        let value = BoolectorBackend::parse_value("#xFF");
        match value {
            CounterexampleValue::Int { value, .. } => {
                kani::assert(value == 255, "hex FF should be 255");
            }
            _ => kani::assert(false, "should be Int variant"),
        }
    }

    /// Verify parse_value parses plain integers
    #[kani::proof]
    fn proof_parse_value_int() {
        let value = BoolectorBackend::parse_value("42");
        match value {
            CounterexampleValue::Int { value, .. } => {
                kani::assert(value == 42, "should parse 42");
            }
            _ => kani::assert(false, "should be Int variant"),
        }
    }
}

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

/// SAT solving engine for Boolector
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BoolectorEngine {
    /// CaDiCaL solver (default)
    #[default]
    CaDiCaL,
    /// Lingeling solver
    Lingeling,
    /// MiniSat solver
    MiniSat,
    /// PicoSat solver
    PicoSat,
}

impl BoolectorEngine {
    fn as_str(&self) -> &'static str {
        match self {
            BoolectorEngine::CaDiCaL => "cadical",
            BoolectorEngine::Lingeling => "lingeling",
            BoolectorEngine::MiniSat => "minisat",
            BoolectorEngine::PicoSat => "picosat",
        }
    }
}

/// Configuration for Boolector backend
#[derive(Debug, Clone)]
pub struct BoolectorConfig {
    /// Path to boolector binary
    pub boolector_path: Option<PathBuf>,
    /// Timeout for verification
    pub timeout: Duration,
    /// SMT-LIB2 logic (QF_BV, QF_ABV, QF_AUFBV)
    pub logic: String,
    /// SAT solver engine to use
    pub engine: BoolectorEngine,
    /// Enable model generation
    pub produce_models: bool,
    /// Simplification level (0-4)
    pub simplify_level: u8,
}

impl Default for BoolectorConfig {
    fn default() -> Self {
        Self {
            boolector_path: None,
            timeout: Duration::from_secs(60),
            logic: "QF_BV".to_string(),
            engine: BoolectorEngine::default(),
            produce_models: true,
            simplify_level: 3,
        }
    }
}

/// Boolector SMT solver backend
pub struct BoolectorBackend {
    config: BoolectorConfig,
}

impl Default for BoolectorBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl BoolectorBackend {
    /// Create a new Boolector backend with default configuration
    pub fn new() -> Self {
        Self {
            config: BoolectorConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: BoolectorConfig) -> Self {
        Self { config }
    }

    async fn detect_boolector(&self) -> Result<PathBuf, String> {
        let boolector_path = self
            .config
            .boolector_path
            .clone()
            .or_else(|| which::which("boolector").ok())
            .or_else(|| which::which("btor2tools-btorsim").ok())
            .ok_or("Boolector not found. Build from https://github.com/Boolector/boolector")?;

        let output = Command::new(&boolector_path)
            .arg("--version")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| format!("Failed to execute boolector: {}", e))?;

        if output.status.success() {
            let version = String::from_utf8_lossy(&output.stdout);
            debug!("Detected Boolector version: {}", version.trim());
            Ok(boolector_path)
        } else {
            // Boolector may output version to stderr
            let stderr = String::from_utf8_lossy(&output.stderr);
            if stderr.contains("Boolector") || stderr.contains("boolector") {
                debug!("Detected Boolector: {}", stderr.trim());
                Ok(boolector_path)
            } else {
                Err("Boolector version check failed".to_string())
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
                        reason: "Boolector returned unknown".to_string(),
                    },
                    None,
                );
            }
        }

        if !success || combined.contains("error") || combined.contains("Error") {
            return (
                VerificationStatus::Unknown {
                    reason: format!(
                        "Boolector error: {}",
                        combined.lines().take(3).collect::<Vec<_>>().join("; ")
                    ),
                },
                None,
            );
        }

        (
            VerificationStatus::Unknown {
                reason: "Could not parse Boolector output".to_string(),
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
            if trimmed.starts_with("(model") {
                in_model = true;
            }
            if in_model {
                model_lines.push(*line);
                depth += line.matches('(').count();
                depth = depth.saturating_sub(line.matches(')').count());
                if depth == 0 && !model_lines.is_empty() {
                    break;
                }
            }
        }

        if model_lines.is_empty() {
            // Boolector may output model as simple assignments
            let assignment_lines: Vec<&str> = lines
                .iter()
                .copied()
                .filter(|l| l.contains('=') || l.starts_with("  "))
                .collect();
            if !assignment_lines.is_empty() {
                return Some(assignment_lines.join("\n"));
            }
            None
        } else {
            Some(model_lines.join("\n"))
        }
    }

    fn parse_counterexample(&self, model_str: &str) -> StructuredCounterexample {
        let mut ce = StructuredCounterexample::new();
        ce.raw = Some(model_str.to_string());

        // Parse Boolector model format
        for line in model_str.lines() {
            let trimmed = line.trim();

            // Handle "(define-fun name () type value)" format
            if trimmed.starts_with("(define-fun") {
                if let Some((name, value)) = Self::parse_define_fun(trimmed) {
                    ce.witness.insert(name, value);
                }
            }
            // Handle "name = value" format
            else if let Some(eq_pos) = trimmed.find('=') {
                let name = trimmed[..eq_pos].trim().to_string();
                let value_str = trimmed[eq_pos + 1..].trim();
                let value = Self::parse_value(value_str);
                if !name.is_empty() {
                    ce.witness.insert(name, value);
                }
            }
        }

        ce
    }

    fn parse_define_fun(line: &str) -> Option<(String, CounterexampleValue)> {
        // Simple parse for "(define-fun name () type value)"
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 5 {
            let name = parts[1].to_string();
            // Find the last non-) token as the value
            let value_str = parts.last()?.trim_end_matches(')');
            let value = Self::parse_value(value_str);
            Some((name, value))
        } else {
            None
        }
    }

    fn parse_value(value: &str) -> CounterexampleValue {
        let trimmed = value.trim().trim_matches(|c| c == '(' || c == ')');

        if trimmed == "true" {
            return CounterexampleValue::Bool(true);
        }
        if trimmed == "false" {
            return CounterexampleValue::Bool(false);
        }

        // BitVector binary: #b1010
        if let Some(binary_str) = trimmed.strip_prefix("#b") {
            if let Ok(n) = i128::from_str_radix(binary_str, 2) {
                return CounterexampleValue::Int {
                    value: n,
                    type_hint: Some(format!("BitVec[{}]", binary_str.len())),
                };
            }
        }

        // BitVector hex: #x0A
        if let Some(hex_str) = trimmed.strip_prefix("#x") {
            if let Ok(n) = i128::from_str_radix(hex_str, 16) {
                return CounterexampleValue::Int {
                    value: n,
                    type_hint: Some(format!("BitVec[{}]", hex_str.len() * 4)),
                };
            }
        }

        // Plain integer
        if let Ok(n) = trimmed.parse::<i128>() {
            return CounterexampleValue::Int {
                value: n,
                type_hint: None,
            };
        }

        CounterexampleValue::Unknown(trimmed.to_string())
    }
}

#[async_trait]
impl VerificationBackend for BoolectorBackend {
    fn id(&self) -> BackendId {
        BackendId::Boolector
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

        let boolector_path = self
            .detect_boolector()
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        // Compile to SMT-LIB2 with bit-vector logic
        let compiled = compile_to_smtlib2_with_logic(spec, &self.config.logic);
        let smt_path = temp_dir.path().join("spec.smt2");
        std::fs::write(&smt_path, &compiled.code).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write SMT-LIB2 file: {}", e))
        })?;

        let mut cmd = Command::new(&boolector_path);
        cmd.arg(&smt_path)
            .arg("--smt2")
            .arg(format!("--sat-engine={}", self.config.engine.as_str()))
            .arg(format!("--simplify={}", self.config.simplify_level))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if self.config.produce_models {
            cmd.arg("--model");
        }

        let timeout_secs = self.config.timeout.as_secs();
        cmd.arg(format!("--time={}", timeout_secs));

        let result =
            tokio::time::timeout(self.config.timeout + Duration::from_secs(5), cmd.output()).await;

        let duration = start.elapsed();

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                debug!("Boolector stdout: {}", stdout);
                debug!("Boolector stderr: {}", stderr);

                let (status, counterexample_str) =
                    self.parse_output(&stdout, &stderr, output.status.success());

                let counterexample = counterexample_str.map(|m| self.parse_counterexample(&m));

                let diagnostics: Vec<String> = stderr
                    .lines()
                    .filter(|l| l.contains("warning") || l.contains("error"))
                    .map(String::from)
                    .collect();

                let proof = if matches!(status, VerificationStatus::Proven) {
                    Some("Verified by Boolector (unsat)".to_string())
                } else {
                    None
                };

                Ok(BackendResult {
                    backend: BackendId::Boolector,
                    status,
                    proof,
                    counterexample,
                    diagnostics,
                    time_taken: duration,
                })
            }
            Ok(Err(e)) => Err(BackendError::VerificationFailed(format!(
                "Failed to execute Boolector: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match self.detect_boolector().await {
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
        let config = BoolectorConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.logic, "QF_BV");
        assert!(config.produce_models);
        assert_eq!(config.simplify_level, 3);
        assert_eq!(config.engine, BoolectorEngine::CaDiCaL);
    }

    #[test]
    fn engine_as_str() {
        assert_eq!(BoolectorEngine::CaDiCaL.as_str(), "cadical");
        assert_eq!(BoolectorEngine::Lingeling.as_str(), "lingeling");
        assert_eq!(BoolectorEngine::MiniSat.as_str(), "minisat");
        assert_eq!(BoolectorEngine::PicoSat.as_str(), "picosat");
    }

    #[test]
    fn parse_unsat() {
        let backend = BoolectorBackend::new();
        let (status, model) = backend.parse_output("unsat", "", true);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(model.is_none());
    }

    #[test]
    fn parse_sat() {
        let backend = BoolectorBackend::new();
        let (status, _) = backend.parse_output("sat", "", true);
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[test]
    fn parse_unknown() {
        let backend = BoolectorBackend::new();
        let (status, _) = backend.parse_output("unknown", "", true);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn parse_error() {
        let backend = BoolectorBackend::new();
        let (status, _) = backend.parse_output("", "error: parse error", false);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn parse_value_bool() {
        assert_eq!(
            BoolectorBackend::parse_value("true"),
            CounterexampleValue::Bool(true)
        );
        assert_eq!(
            BoolectorBackend::parse_value("false"),
            CounterexampleValue::Bool(false)
        );
    }

    #[test]
    fn parse_value_bitvec_binary() {
        match BoolectorBackend::parse_value("#b1010") {
            CounterexampleValue::Int { value, type_hint } => {
                assert_eq!(value, 10);
                assert!(type_hint.unwrap().contains("BitVec"));
            }
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn parse_value_bitvec_hex() {
        match BoolectorBackend::parse_value("#xFF") {
            CounterexampleValue::Int { value, type_hint } => {
                assert_eq!(value, 255);
                assert!(type_hint.unwrap().contains("BitVec"));
            }
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn parse_value_int() {
        match BoolectorBackend::parse_value("42") {
            CounterexampleValue::Int { value, .. } => assert_eq!(value, 42),
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn parse_counterexample_assignment() {
        let backend = BoolectorBackend::new();
        let model = "x = #b1010\ny = #xFF";
        let ce = backend.parse_counterexample(model);
        assert!(ce.witness.contains_key("x"));
        assert!(ce.witness.contains_key("y"));
    }

    #[test]
    fn extract_model_smt2() {
        let lines = vec![
            "sat",
            "(model",
            "  (define-fun x () (_ BitVec 8) #xFF)",
            ")",
        ];
        let model = BoolectorBackend::extract_model(&lines);
        assert!(model.is_some());
    }

    #[test]
    fn extract_model_simple() {
        let lines = vec!["sat", "  x = #b1010"];
        let model = BoolectorBackend::extract_model(&lines);
        assert!(model.is_some());
    }

    #[test]
    fn backend_id() {
        let backend = BoolectorBackend::new();
        assert_eq!(backend.id(), BackendId::Boolector);
    }

    #[test]
    fn supports_properties() {
        let backend = BoolectorBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::Theorem));
        assert!(supported.contains(&PropertyType::Invariant));
        assert!(supported.contains(&PropertyType::Contract));
    }
}
