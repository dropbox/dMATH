//! Yices SMT solver backend
//!
//! Yices is a high-performance SMT solver developed at SRI International.
//! It supports SMT-LIB2 format and excels at bit-vector and linear arithmetic problems.
//!
//! See: <https://yices.csl.sri.com/>

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

/// Configuration for Yices backend
#[derive(Debug, Clone)]
pub struct YicesConfig {
    /// Path to yices-smt2 binary
    pub yices_path: Option<PathBuf>,
    /// Timeout for verification
    pub timeout: Duration,
    /// SMT-LIB2 logic to use
    pub logic: String,
    /// Enable model generation
    pub produce_models: bool,
    /// Enable incremental mode
    pub incremental: bool,
}

impl Default for YicesConfig {
    fn default() -> Self {
        Self {
            yices_path: None,
            timeout: Duration::from_secs(60),
            logic: "ALL".to_string(),
            produce_models: true,
            incremental: false,
        }
    }
}

/// Yices SMT solver backend
pub struct YicesBackend {
    config: YicesConfig,
}

impl Default for YicesBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl YicesBackend {
    /// Create a new Yices backend with default configuration
    pub fn new() -> Self {
        Self {
            config: YicesConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: YicesConfig) -> Self {
        Self { config }
    }

    async fn detect_yices(&self) -> Result<PathBuf, String> {
        let yices_path = self
            .config
            .yices_path
            .clone()
            .or_else(|| which::which("yices-smt2").ok())
            .or_else(|| which::which("yices").ok())
            .ok_or("Yices not found. Install from https://yices.csl.sri.com/downloads.html")?;

        let output = Command::new(&yices_path)
            .arg("--version")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| format!("Failed to execute yices: {}", e))?;

        if output.status.success() {
            let version = String::from_utf8_lossy(&output.stdout);
            debug!("Detected Yices version: {}", version.trim());
            Ok(yices_path)
        } else {
            Err("Yices version check failed".to_string())
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
                        reason: "Yices returned unknown".to_string(),
                    },
                    None,
                );
            }
        }

        if !success || combined.contains("error") || combined.contains("Error") {
            return (
                VerificationStatus::Unknown {
                    reason: format!(
                        "Yices error: {}",
                        combined.lines().take(3).collect::<Vec<_>>().join("; ")
                    ),
                },
                None,
            );
        }

        (
            VerificationStatus::Unknown {
                reason: "Could not parse Yices output".to_string(),
            },
            None,
        )
    }

    fn extract_model(lines: &[&str]) -> Option<String> {
        let mut in_model = false;
        let mut model_lines = Vec::new();
        let mut depth = 0;

        for line in lines {
            let trimmed = line.trim();
            if trimmed.starts_with("(model") || (trimmed == "(" && depth == 0) {
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
            None
        } else {
            Some(model_lines.join("\n"))
        }
    }

    fn parse_counterexample(&self, model_str: &str) -> StructuredCounterexample {
        let mut ce = StructuredCounterexample::new();
        ce.raw = Some(model_str.to_string());

        // Yices uses similar format to Z3 for models
        for line in model_str.lines() {
            let trimmed = line.trim();
            // Parse "(= name value)" or "(define-fun name () type value)"
            if trimmed.starts_with("(=") && trimmed.ends_with(')') {
                let inner = &trimmed[2..trimmed.len() - 1].trim();
                let parts: Vec<&str> = inner.splitn(2, ' ').collect();
                if parts.len() == 2 {
                    let name = parts[0].trim().to_string();
                    let value = Self::parse_value(parts[1].trim());
                    ce.witness.insert(name, value);
                }
            }
        }

        ce
    }

    fn parse_value(value: &str) -> CounterexampleValue {
        let trimmed = value.trim();

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
        if trimmed.starts_with("(-") && trimmed.ends_with(')') {
            let inner = trimmed[2..trimmed.len() - 1].trim();
            if let Ok(n) = inner.parse::<i128>() {
                return CounterexampleValue::Int {
                    value: -n,
                    type_hint: Some("Int".to_string()),
                };
            }
        }

        // Float
        if let Ok(f) = trimmed.parse::<f64>() {
            return CounterexampleValue::Float { value: f };
        }

        // BitVector binary
        if let Some(binary_str) = trimmed.strip_prefix("0b") {
            if let Ok(n) = i128::from_str_radix(binary_str, 2) {
                return CounterexampleValue::Int {
                    value: n,
                    type_hint: Some("BitVec".to_string()),
                };
            }
        }

        // BitVector hex
        if let Some(hex_str) = trimmed.strip_prefix("0x") {
            if let Ok(n) = i128::from_str_radix(hex_str, 16) {
                return CounterexampleValue::Int {
                    value: n,
                    type_hint: Some("BitVec".to_string()),
                };
            }
        }

        CounterexampleValue::Unknown(trimmed.to_string())
    }
}

#[async_trait]
impl VerificationBackend for YicesBackend {
    fn id(&self) -> BackendId {
        BackendId::Yices
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

        let yices_path = self
            .detect_yices()
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        // Compile to SMT-LIB2
        let compiled = compile_to_smtlib2_with_logic(spec, &self.config.logic);
        let smt_path = temp_dir.path().join("spec.smt2");
        std::fs::write(&smt_path, &compiled.code).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write SMT-LIB2 file: {}", e))
        })?;

        let mut cmd = Command::new(&yices_path);
        cmd.arg(&smt_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if self.config.incremental {
            cmd.arg("--incremental");
        }

        let timeout_secs = self.config.timeout.as_secs();
        cmd.arg(format!("--timeout={}", timeout_secs));

        let result =
            tokio::time::timeout(self.config.timeout + Duration::from_secs(5), cmd.output()).await;

        let duration = start.elapsed();

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                debug!("Yices stdout: {}", stdout);
                debug!("Yices stderr: {}", stderr);

                let (status, counterexample_str) =
                    self.parse_output(&stdout, &stderr, output.status.success());

                let counterexample = counterexample_str.map(|m| self.parse_counterexample(&m));

                let diagnostics: Vec<String> = stderr
                    .lines()
                    .filter(|l| l.contains("warning") || l.contains("error"))
                    .map(String::from)
                    .collect();

                let proof = if matches!(status, VerificationStatus::Proven) {
                    Some("Verified by Yices (unsat)".to_string())
                } else {
                    None
                };

                Ok(BackendResult {
                    backend: BackendId::Yices,
                    status,
                    proof,
                    counterexample,
                    diagnostics,
                    time_taken: duration,
                })
            }
            Ok(Err(e)) => Err(BackendError::VerificationFailed(format!(
                "Failed to execute Yices: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match self.detect_yices().await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ===== YicesConfig defaults =====

    #[kani::proof]
    fn verify_config_defaults_timeout() {
        let config = YicesConfig::default();
        assert!(config.timeout == Duration::from_secs(60));
    }

    #[kani::proof]
    fn verify_config_defaults_logic() {
        let config = YicesConfig::default();
        assert!(config.logic == "ALL");
    }

    #[kani::proof]
    fn verify_config_defaults_produce_models() {
        let config = YicesConfig::default();
        assert!(config.produce_models);
    }

    #[kani::proof]
    fn verify_config_defaults_incremental() {
        let config = YicesConfig::default();
        assert!(!config.incremental);
    }

    #[kani::proof]
    fn verify_config_defaults_yices_path() {
        let config = YicesConfig::default();
        assert!(config.yices_path.is_none());
    }

    // ===== YicesBackend construction =====

    #[kani::proof]
    fn verify_backend_new_uses_defaults() {
        let backend = YicesBackend::new();
        assert!(backend.config.timeout == Duration::from_secs(60));
        assert!(backend.config.logic == "ALL");
        assert!(backend.config.produce_models);
    }

    #[kani::proof]
    fn verify_backend_default_matches_new() {
        let b1 = YicesBackend::new();
        let b2 = YicesBackend::default();
        assert!(b1.config.timeout == b2.config.timeout);
        assert!(b1.config.logic == b2.config.logic);
        assert!(b1.config.produce_models == b2.config.produce_models);
    }

    #[kani::proof]
    fn verify_backend_with_config_preserves_values() {
        let config = YicesConfig {
            yices_path: Some(PathBuf::from("/usr/bin/yices")),
            timeout: Duration::from_secs(120),
            logic: "QF_LIA".to_string(),
            produce_models: false,
            incremental: true,
        };
        let backend = YicesBackend::with_config(config);
        assert!(backend.config.timeout == Duration::from_secs(120));
        assert!(backend.config.logic == "QF_LIA");
        assert!(!backend.config.produce_models);
        assert!(backend.config.incremental);
    }

    // ===== ID and supports =====

    #[kani::proof]
    fn verify_backend_id() {
        let backend = YicesBackend::new();
        assert!(matches!(backend.id(), BackendId::Yices));
    }

    #[kani::proof]
    fn verify_supports_theorem_invariant_contract() {
        let backend = YicesBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::Theorem));
        assert!(supported.contains(&PropertyType::Invariant));
        assert!(supported.contains(&PropertyType::Contract));
        assert!(supported.len() == 3);
    }

    // ===== Output parsing =====

    #[kani::proof]
    fn verify_parse_output_unsat() {
        let backend = YicesBackend::new();
        let (status, model) = backend.parse_output("unsat", "", true);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(model.is_none());
    }

    #[kani::proof]
    fn verify_parse_output_sat() {
        let backend = YicesBackend::new();
        let (status, _) = backend.parse_output("sat", "", true);
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[kani::proof]
    fn verify_parse_output_unknown() {
        let backend = YicesBackend::new();
        let (status, _) = backend.parse_output("unknown", "", true);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[kani::proof]
    fn verify_parse_output_error() {
        let backend = YicesBackend::new();
        let (status, _) = backend.parse_output("", "error: syntax", false);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    // ===== Value parsing =====

    #[kani::proof]
    fn verify_parse_value_true() {
        let value = YicesBackend::parse_value("true");
        assert!(matches!(value, CounterexampleValue::Bool(true)));
    }

    #[kani::proof]
    fn verify_parse_value_false() {
        let value = YicesBackend::parse_value("false");
        assert!(matches!(value, CounterexampleValue::Bool(false)));
    }

    #[kani::proof]
    fn verify_parse_value_positive_int() {
        let value = YicesBackend::parse_value("42");
        match value {
            CounterexampleValue::Int { value: v, .. } => assert!(v == 42),
            _ => panic!("Expected Int"),
        }
    }

    #[kani::proof]
    fn verify_parse_value_negative_int() {
        let value = YicesBackend::parse_value("(- 42)");
        match value {
            CounterexampleValue::Int { value: v, .. } => assert!(v == -42),
            _ => panic!("Expected Int"),
        }
    }

    #[kani::proof]
    fn verify_parse_value_bitvec_binary() {
        let value = YicesBackend::parse_value("0b1010");
        match value {
            CounterexampleValue::Int { value: v, .. } => assert!(v == 10),
            _ => panic!("Expected Int"),
        }
    }

    #[kani::proof]
    fn verify_parse_value_bitvec_hex() {
        let value = YicesBackend::parse_value("0xFF");
        match value {
            CounterexampleValue::Int { value: v, .. } => assert!(v == 255),
            _ => panic!("Expected Int"),
        }
    }

    #[kani::proof]
    fn verify_parse_value_unknown() {
        let value = YicesBackend::parse_value("(some complex expr)");
        assert!(matches!(value, CounterexampleValue::Unknown(_)));
    }

    // ===== Model extraction =====

    #[kani::proof]
    fn verify_extract_model_with_model_keyword() {
        let lines = vec!["sat", "(model", "(= x 5)", ")"];
        let model = YicesBackend::extract_model(&lines);
        assert!(model.is_some());
    }

    #[kani::proof]
    fn verify_extract_model_empty() {
        let lines: Vec<&str> = vec!["unsat"];
        let model = YicesBackend::extract_model(&lines);
        assert!(model.is_none());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = YicesConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.logic, "ALL");
        assert!(config.produce_models);
        assert!(!config.incremental);
    }

    #[test]
    fn parse_unsat() {
        let backend = YicesBackend::new();
        let (status, model) = backend.parse_output("unsat", "", true);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(model.is_none());
    }

    #[test]
    fn parse_sat() {
        let backend = YicesBackend::new();
        let (status, _) = backend.parse_output("sat", "", true);
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[test]
    fn parse_unknown() {
        let backend = YicesBackend::new();
        let (status, _) = backend.parse_output("unknown", "", true);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn parse_error() {
        let backend = YicesBackend::new();
        let (status, _) = backend.parse_output("", "error: syntax error", false);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn parse_value_bool() {
        assert_eq!(
            YicesBackend::parse_value("true"),
            CounterexampleValue::Bool(true)
        );
        assert_eq!(
            YicesBackend::parse_value("false"),
            CounterexampleValue::Bool(false)
        );
    }

    #[test]
    fn parse_value_int() {
        match YicesBackend::parse_value("42") {
            CounterexampleValue::Int { value, .. } => assert_eq!(value, 42),
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn parse_value_negative_int() {
        match YicesBackend::parse_value("(- 42)") {
            CounterexampleValue::Int { value, .. } => assert_eq!(value, -42),
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn parse_value_float() {
        match YicesBackend::parse_value("1.234") {
            CounterexampleValue::Float { value } => assert!((value - 1.234).abs() < 0.001),
            _ => panic!("Expected Float"),
        }
    }

    #[test]
    fn parse_value_bitvec_binary() {
        match YicesBackend::parse_value("0b1010") {
            CounterexampleValue::Int { value, .. } => assert_eq!(value, 10),
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn parse_value_bitvec_hex() {
        match YicesBackend::parse_value("0xFF") {
            CounterexampleValue::Int { value, .. } => assert_eq!(value, 255),
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn parse_counterexample_basic() {
        let backend = YicesBackend::new();
        let model = "(model\n(= x 42)\n)";
        let ce = backend.parse_counterexample(model);
        assert!(ce.raw.is_some());
    }

    #[test]
    fn extract_model_basic() {
        let lines = vec!["sat", "(model", "(= x 5)", ")"];
        let model = YicesBackend::extract_model(&lines);
        assert!(model.is_some());
    }

    #[test]
    fn backend_id() {
        let backend = YicesBackend::new();
        assert_eq!(backend.id(), BackendId::Yices);
    }

    #[test]
    fn supports_properties() {
        let backend = YicesBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::Theorem));
        assert!(supported.contains(&PropertyType::Invariant));
        assert!(supported.contains(&PropertyType::Contract));
    }
}
