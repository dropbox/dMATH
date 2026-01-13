//! veriT SMT solver backend
//!
//! veriT is an SMT solver developed at LORIA that produces detailed proofs of unsatisfiability.
//! It's particularly suited for integration with proof assistants like Isabelle/HOL and Coq.
//!
//! veriT features:
//! - Detailed proof output in a checkable format
//! - Support for quantifiers and uninterpreted functions
//! - Strong performance on verification conditions
//! - Integration with Why3 and Frama-C
//!
//! See: <https://verit.loria.fr/>

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

/// Configuration for veriT backend
#[derive(Debug, Clone)]
pub struct VeriTConfig {
    /// Path to veriT binary
    pub verit_path: Option<PathBuf>,
    /// Timeout for solving
    pub timeout: Duration,
    /// Enable proof production (--proof=file)
    pub produce_proofs: bool,
    /// SMT-LIB2 logic to use
    pub logic: String,
    /// Enable model generation
    pub produce_models: bool,
    /// Output format for proof ("lfsc", "alethe", etc.)
    pub proof_format: String,
}

impl Default for VeriTConfig {
    fn default() -> Self {
        Self {
            verit_path: None,
            timeout: Duration::from_secs(60),
            produce_proofs: false,
            logic: "QF_LIA".to_string(),
            produce_models: true,
            proof_format: "alethe".to_string(),
        }
    }
}

/// veriT SMT solver backend
pub struct VeriTBackend {
    config: VeriTConfig,
}

impl Default for VeriTBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl VeriTBackend {
    /// Create a new veriT backend with default configuration
    pub fn new() -> Self {
        Self {
            config: VeriTConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: VeriTConfig) -> Self {
        Self { config }
    }

    async fn detect_verit(&self) -> Result<PathBuf, String> {
        let verit_path = self
            .config
            .verit_path
            .clone()
            .or_else(|| which::which("veriT").ok())
            .or_else(|| which::which("verit").ok())
            .ok_or("veriT not found. Install from https://verit.loria.fr/")?;

        // Verify the binary works
        let output = Command::new(&verit_path)
            .arg("--help")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| format!("Failed to execute veriT: {}", e))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stdout.contains("veriT") || stderr.contains("veriT") || stdout.contains("Usage") {
            debug!("Detected veriT");
            Ok(verit_path)
        } else {
            Err("veriT detection failed".to_string())
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

        // Find the sat/unsat result
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
                        reason: "veriT returned unknown (timeout or incomplete)".to_string(),
                    },
                    None,
                );
            }
        }

        // Check for errors
        if !success || combined.contains("error") || combined.contains("Error") {
            return (
                VerificationStatus::Unknown {
                    reason: format!(
                        "veriT error: {}",
                        combined.lines().take(3).collect::<Vec<_>>().join("; ")
                    ),
                },
                None,
            );
        }

        (
            VerificationStatus::Unknown {
                reason: "Could not parse veriT output".to_string(),
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

    /// Parse a veriT model into a structured counterexample
    /// veriT uses standard SMT-LIB2 model format
    fn parse_counterexample(&self, model_str: &str) -> StructuredCounterexample {
        let mut ce = StructuredCounterexample::new();
        ce.raw = Some(model_str.to_string());

        let definitions = Self::extract_definitions(model_str);
        for (name, sort, value) in definitions {
            let parsed_value = Self::parse_smt_value(&value, &sort);
            ce.witness.insert(name, parsed_value);
        }

        ce
    }

    fn extract_definitions(model_str: &str) -> Vec<(String, String, String)> {
        let mut definitions = Vec::new();
        let chars: Vec<char> = model_str.chars().collect();
        let mut pos = 0;

        while pos < chars.len() {
            if let Some(start) = Self::find_define_fun(&chars, pos) {
                if let Some((name, sort, value, end)) = Self::parse_define_fun(&chars, start) {
                    definitions.push((name, sort, value));
                    pos = end;
                } else {
                    pos = start + 1;
                }
            } else {
                break;
            }
        }

        definitions
    }

    fn find_define_fun(chars: &[char], start: usize) -> Option<usize> {
        let pattern: Vec<char> = "(define-fun".chars().collect();
        let mut i = start;
        while i + pattern.len() <= chars.len() {
            if chars[i..i + pattern.len()] == pattern[..] {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    fn parse_define_fun(chars: &[char], start: usize) -> Option<(String, String, String, usize)> {
        let mut pos = start + "(define-fun ".len();

        // Skip whitespace
        while pos < chars.len() && chars[pos].is_whitespace() {
            pos += 1;
        }

        // Read name
        let name_start = pos;
        while pos < chars.len() && !chars[pos].is_whitespace() && chars[pos] != '(' {
            pos += 1;
        }
        let name: String = chars[name_start..pos].iter().collect();

        // Skip whitespace
        while pos < chars.len() && chars[pos].is_whitespace() {
            pos += 1;
        }

        // Skip argument list "()"
        if pos < chars.len() && chars[pos] == '(' {
            let mut depth = 1;
            pos += 1;
            while pos < chars.len() && depth > 0 {
                if chars[pos] == '(' {
                    depth += 1;
                } else if chars[pos] == ')' {
                    depth -= 1;
                }
                pos += 1;
            }
        }

        // Skip whitespace
        while pos < chars.len() && chars[pos].is_whitespace() {
            pos += 1;
        }

        // Read sort
        let sort = if pos < chars.len() && chars[pos] == '(' {
            let sort_start = pos;
            let mut depth = 1;
            pos += 1;
            while pos < chars.len() && depth > 0 {
                if chars[pos] == '(' {
                    depth += 1;
                } else if chars[pos] == ')' {
                    depth -= 1;
                }
                pos += 1;
            }
            chars[sort_start..pos].iter().collect()
        } else {
            let sort_start = pos;
            while pos < chars.len() && !chars[pos].is_whitespace() && chars[pos] != ')' {
                pos += 1;
            }
            chars[sort_start..pos].iter().collect()
        };

        // Skip whitespace
        while pos < chars.len() && chars[pos].is_whitespace() {
            pos += 1;
        }

        // Read value
        let value = if pos < chars.len() && chars[pos] == '(' {
            let value_start = pos;
            let mut depth = 1;
            pos += 1;
            while pos < chars.len() && depth > 0 {
                if chars[pos] == '(' {
                    depth += 1;
                } else if chars[pos] == ')' {
                    depth -= 1;
                }
                pos += 1;
            }
            chars[value_start..pos].iter().collect()
        } else {
            let value_start = pos;
            while pos < chars.len() && !chars[pos].is_whitespace() && chars[pos] != ')' {
                pos += 1;
            }
            chars[value_start..pos].iter().collect()
        };

        // Find closing ')'
        while pos < chars.len() && chars[pos] != ')' {
            pos += 1;
        }
        if pos < chars.len() {
            pos += 1;
        }

        if name.is_empty() {
            None
        } else {
            Some((name, sort, value, pos))
        }
    }

    fn parse_smt_value(value: &str, sort: &str) -> CounterexampleValue {
        let trimmed = value.trim();

        if trimmed == "true" {
            return CounterexampleValue::Bool(true);
        }
        if trimmed == "false" {
            return CounterexampleValue::Bool(false);
        }

        if let Some(int_val) = Self::parse_int_value(trimmed) {
            return CounterexampleValue::Int {
                value: int_val,
                type_hint: Some(sort.to_string()),
            };
        }

        if sort == "Real" || sort.contains("Real") {
            if let Some(float_val) = Self::parse_real_value(trimmed) {
                return CounterexampleValue::Float { value: float_val };
            }
        }

        if trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() >= 2 {
            return CounterexampleValue::String(trimmed[1..trimmed.len() - 1].to_string());
        }

        CounterexampleValue::Unknown(trimmed.to_string())
    }

    fn parse_int_value(value: &str) -> Option<i128> {
        let trimmed = value.trim();

        if let Ok(n) = trimmed.parse::<i128>() {
            return Some(n);
        }

        if trimmed.starts_with("(-") && trimmed.ends_with(')') {
            let inner = trimmed[2..trimmed.len() - 1].trim();
            if let Ok(n) = inner.parse::<i128>() {
                return Some(-n);
            }
        }

        if trimmed.starts_with("(- ") && trimmed.ends_with(')') {
            let inner = trimmed[3..trimmed.len() - 1].trim();
            if let Ok(n) = inner.parse::<i128>() {
                return Some(-n);
            }
        }

        None
    }

    fn parse_real_value(value: &str) -> Option<f64> {
        let trimmed = value.trim();

        if let Ok(f) = trimmed.parse::<f64>() {
            return Some(f);
        }

        if trimmed.starts_with("(/") && trimmed.ends_with(')') {
            let inner = trimmed[2..trimmed.len() - 1].trim();
            let parts: Vec<&str> = inner.split_whitespace().collect();
            if parts.len() == 2 {
                if let (Ok(num), Ok(denom)) = (parts[0].parse::<f64>(), parts[1].parse::<f64>()) {
                    if denom != 0.0 {
                        return Some(num / denom);
                    }
                }
            }
        }

        if trimmed.starts_with("(-") && trimmed.ends_with(')') {
            let inner = trimmed[2..trimmed.len() - 1].trim();
            if let Some(f) = Self::parse_real_value(inner) {
                return Some(-f);
            }
        }

        None
    }
}

#[async_trait]
impl VerificationBackend for VeriTBackend {
    fn id(&self) -> BackendId {
        BackendId::VeriT
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

        let verit_path = self
            .detect_verit()
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

        // Build command - veriT uses --input=file syntax
        let mut cmd = Command::new(&verit_path);
        cmd.arg(format!("--input={}", smt_path.display()))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Add timeout option
        cmd.arg(format!("--max-time={}", self.config.timeout.as_secs()));

        // Add proof production if enabled
        if self.config.produce_proofs {
            let proof_path = temp_dir.path().join("proof.alethe");
            cmd.arg(format!("--proof={}", proof_path.display()));
            cmd.arg(format!(
                "--proof-format-and-target={}",
                self.config.proof_format
            ));
        }

        let result =
            tokio::time::timeout(self.config.timeout + Duration::from_secs(5), cmd.output()).await;

        let duration = start.elapsed();

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                debug!("veriT stdout: {}", stdout);
                debug!("veriT stderr: {}", stderr);

                let (status, counterexample_str) =
                    self.parse_output(&stdout, &stderr, output.status.success());

                let counterexample = counterexample_str.map(|m| self.parse_counterexample(&m));

                let diagnostics: Vec<String> = stderr
                    .lines()
                    .filter(|l| l.contains("warning") || l.contains("error"))
                    .map(String::from)
                    .collect();

                let proof = if matches!(status, VerificationStatus::Proven) {
                    Some("Verified by veriT (unsat)".to_string())
                } else {
                    None
                };

                Ok(BackendResult {
                    backend: BackendId::VeriT,
                    status,
                    proof,
                    counterexample,
                    diagnostics,
                    time_taken: duration,
                })
            }
            Ok(Err(e)) => Err(BackendError::VerificationFailed(format!(
                "Failed to execute veriT: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match self.detect_verit().await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ===== VeriTConfig defaults =====

    #[kani::proof]
    fn verify_config_defaults_timeout() {
        let config = VeriTConfig::default();
        assert!(config.timeout == Duration::from_secs(60));
    }

    #[kani::proof]
    fn verify_config_defaults_options() {
        let config = VeriTConfig::default();
        assert!(config.verit_path.is_none());
        assert!(config.logic == "QF_LIA");
        assert!(!config.produce_proofs);
        assert!(config.produce_models);
        assert!(config.proof_format == "alethe");
    }

    // ===== Backend construction =====

    #[kani::proof]
    fn verify_backend_new_uses_defaults() {
        let backend = VeriTBackend::new();
        assert!(backend.config.timeout == Duration::from_secs(60));
        assert!(backend.config.produce_models);
    }

    #[kani::proof]
    fn verify_backend_default_matches_new() {
        let b1 = VeriTBackend::new();
        let b2 = VeriTBackend::default();
        assert!(b1.config.timeout == b2.config.timeout);
        assert!(b1.config.logic == b2.config.logic);
    }

    #[kani::proof]
    fn verify_backend_with_config_preserves_values() {
        let config = VeriTConfig {
            verit_path: Some(PathBuf::from("/usr/bin/veriT")),
            timeout: Duration::from_secs(30),
            produce_proofs: true,
            logic: "QF_UF".to_string(),
            produce_models: false,
            proof_format: "lfsc".to_string(),
        };
        let backend = VeriTBackend::with_config(config);
        assert!(backend.config.timeout == Duration::from_secs(30));
        assert!(backend.config.produce_proofs);
        assert!(backend.config.logic == "QF_UF");
    }

    // ===== ID and supports =====

    #[kani::proof]
    fn verify_backend_id() {
        let backend = VeriTBackend::new();
        assert!(matches!(backend.id(), BackendId::VeriT));
    }

    #[kani::proof]
    fn verify_supports_theorem_invariant_contract() {
        let backend = VeriTBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::Theorem));
        assert!(supported.contains(&PropertyType::Invariant));
        assert!(supported.contains(&PropertyType::Contract));
        assert!(supported.len() == 3);
    }

    // ===== Output parsing =====

    #[kani::proof]
    fn verify_parse_output_unsat() {
        let backend = VeriTBackend::new();
        let (status, model) = backend.parse_output("unsat", "", true);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(model.is_none());
    }

    #[kani::proof]
    fn verify_parse_output_sat() {
        let backend = VeriTBackend::new();
        let (status, _) = backend.parse_output("sat", "", true);
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[kani::proof]
    fn verify_parse_output_unknown() {
        let backend = VeriTBackend::new();
        let (status, _) = backend.parse_output("unknown", "", true);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    // ===== SMT value parsing =====

    #[kani::proof]
    fn verify_parse_smt_value_true() {
        let result = VeriTBackend::parse_smt_value("true", "Bool");
        assert!(matches!(result, CounterexampleValue::Bool(true)));
    }

    #[kani::proof]
    fn verify_parse_smt_value_false() {
        let result = VeriTBackend::parse_smt_value("false", "Bool");
        assert!(matches!(result, CounterexampleValue::Bool(false)));
    }

    #[kani::proof]
    fn verify_parse_int_value_positive() {
        let result = VeriTBackend::parse_int_value("42");
        assert!(result == Some(42));
    }

    #[kani::proof]
    fn verify_parse_int_value_negative() {
        let result = VeriTBackend::parse_int_value("(- 42)");
        assert!(result == Some(-42));
    }

    #[kani::proof]
    fn verify_parse_real_value_decimal() {
        let result = VeriTBackend::parse_real_value("2.75");
        assert!(result.is_some());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = VeriTConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.logic, "QF_LIA");
        assert!(config.produce_models);
        assert_eq!(config.proof_format, "alethe");
    }

    #[test]
    fn backend_id() {
        let backend = VeriTBackend::new();
        assert_eq!(backend.id(), BackendId::VeriT);
    }

    #[test]
    fn parse_unsat() {
        let backend = VeriTBackend::new();
        let (status, model) = backend.parse_output("unsat", "", true);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(model.is_none());
    }

    #[test]
    fn parse_sat() {
        let backend = VeriTBackend::new();
        let (status, _) = backend.parse_output("sat", "", true);
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[test]
    fn parse_unknown() {
        let backend = VeriTBackend::new();
        let (status, _) = backend.parse_output("unknown", "", true);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn parse_error() {
        let backend = VeriTBackend::new();
        let (status, _) = backend.parse_output("", "error: syntax error", false);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn extract_model_basic() {
        let lines = vec!["sat", "(model", "  (define-fun x () Int 5)", ")"];
        let model = VeriTBackend::extract_model(&lines);
        assert!(model.is_some());
        let m = model.unwrap();
        assert!(m.contains("model"));
        assert!(m.contains("define-fun"));
    }

    // Counterexample parsing tests
    #[test]
    fn parse_counterexample_int() {
        let backend = VeriTBackend::new();
        let model = "(model\n  (define-fun x () Int 42)\n)";
        let ce = backend.parse_counterexample(model);
        assert!(ce.witness.contains_key("x"));
        match &ce.witness["x"] {
            CounterexampleValue::Int { value, .. } => assert_eq!(*value, 42),
            _ => panic!("Expected Int value"),
        }
    }

    #[test]
    fn parse_counterexample_negative_int() {
        let backend = VeriTBackend::new();
        let model = "(model\n  (define-fun x () Int (- 42))\n)";
        let ce = backend.parse_counterexample(model);
        assert!(ce.witness.contains_key("x"));
        match &ce.witness["x"] {
            CounterexampleValue::Int { value, .. } => assert_eq!(*value, -42),
            _ => panic!("Expected Int value"),
        }
    }

    #[test]
    fn parse_counterexample_bool() {
        let backend = VeriTBackend::new();
        let model = "(model\n  (define-fun b () Bool true)\n)";
        let ce = backend.parse_counterexample(model);
        assert!(ce.witness.contains_key("b"));
        assert_eq!(ce.witness["b"], CounterexampleValue::Bool(true));
    }

    #[test]
    fn parse_counterexample_real() {
        let backend = VeriTBackend::new();
        let model = "(model\n  (define-fun r () Real 2.75)\n)";
        let ce = backend.parse_counterexample(model);
        assert!(ce.witness.contains_key("r"));
        match &ce.witness["r"] {
            CounterexampleValue::Float { value } => assert!((value - 2.75).abs() < 0.001),
            _ => panic!("Expected Float value"),
        }
    }

    #[test]
    fn parse_counterexample_multiple_vars() {
        let backend = VeriTBackend::new();
        let model = "(model
  (define-fun x () Int 10)
  (define-fun y () Int 20)
  (define-fun z () Bool true)
)";
        let ce = backend.parse_counterexample(model);
        assert_eq!(ce.witness.len(), 3);
        assert!(ce.witness.contains_key("x"));
        assert!(ce.witness.contains_key("y"));
        assert!(ce.witness.contains_key("z"));
    }

    #[test]
    fn parse_counterexample_preserves_raw() {
        let backend = VeriTBackend::new();
        let model = "(model\n  (define-fun x () Int 5)\n)";
        let ce = backend.parse_counterexample(model);
        assert!(ce.raw.is_some());
        assert!(ce.raw.unwrap().contains("define-fun x"));
    }

    #[test]
    fn parse_int_value_positive() {
        assert_eq!(VeriTBackend::parse_int_value("42"), Some(42));
    }

    #[test]
    fn parse_int_value_negative_parens() {
        assert_eq!(VeriTBackend::parse_int_value("(- 42)"), Some(-42));
        assert_eq!(VeriTBackend::parse_int_value("(-42)"), Some(-42));
    }

    #[test]
    fn parse_int_value_zero() {
        assert_eq!(VeriTBackend::parse_int_value("0"), Some(0));
    }

    #[test]
    fn parse_int_value_invalid() {
        assert_eq!(VeriTBackend::parse_int_value("abc"), None);
    }

    #[test]
    fn parse_real_value_decimal() {
        let val = VeriTBackend::parse_real_value("2.75").unwrap();
        assert!((val - 2.75).abs() < 0.00001);
    }

    #[test]
    fn parse_real_value_rational() {
        let val = VeriTBackend::parse_real_value("(/ 1.0 3.0)").unwrap();
        assert!((val - 0.33333).abs() < 0.001);
    }

    #[test]
    fn has_structured_data() {
        let backend = VeriTBackend::new();
        let model = "(model\n  (define-fun x () Int 5)\n)";
        let ce = backend.parse_counterexample(model);
        assert!(ce.has_structured_data());
    }
}
