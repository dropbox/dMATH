//! Z3 SMT solver backend
//!
//! Z3 is a high-performance theorem prover from Microsoft Research.
//! It supports SMT-LIB2 format and can verify a wide range of properties
//! including linear arithmetic, arrays, bit-vectors, and uninterpreted functions.
//!
//! See: <https://github.com/Z3Prover/z3>

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

// =============================================
// Kani Proofs for Z3 Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify Z3Config::default timeout is 60 seconds
    #[kani::proof]
    fn proof_z3_config_default_timeout() {
        let config = Z3Config::default();
        kani::assert(
            config.timeout == Duration::from_secs(60),
            "Default timeout should be 60 seconds",
        );
    }

    /// Verify Z3Config::default logic is "ALL"
    #[kani::proof]
    fn proof_z3_config_default_logic() {
        let config = Z3Config::default();
        kani::assert(config.logic == "ALL", "Default logic should be ALL");
    }

    /// Verify Z3Config::default produces_models is true
    #[kani::proof]
    fn proof_z3_config_default_produce_models() {
        let config = Z3Config::default();
        kani::assert(config.produce_models, "Default should produce models");
    }

    /// Verify Z3Config::default memory_limit is 0
    #[kani::proof]
    fn proof_z3_config_default_memory_limit() {
        let config = Z3Config::default();
        kani::assert(config.memory_limit == 0, "Default memory_limit should be 0");
    }

    /// Verify Z3Config::default z3_path is None
    #[kani::proof]
    fn proof_z3_config_default_path_none() {
        let config = Z3Config::default();
        kani::assert(config.z3_path.is_none(), "Default z3_path should be None");
    }

    /// Verify Z3Backend::new uses default config
    #[kani::proof]
    fn proof_z3_backend_new_defaults() {
        let backend = Z3Backend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(60),
            "New backend should use default timeout",
        );
    }

    /// Verify Z3Backend::with_config preserves config
    #[kani::proof]
    fn proof_z3_backend_with_config() {
        let config = Z3Config {
            z3_path: Some(PathBuf::from("/custom/z3")),
            timeout: Duration::from_secs(120),
            logic: "QF_LIA".to_string(),
            produce_models: false,
            memory_limit: 1024,
        };
        let backend = Z3Backend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(120),
            "Custom timeout should be preserved",
        );
        kani::assert(
            backend.config.logic == "QF_LIA",
            "Custom logic should be preserved",
        );
    }

    /// Verify id() returns Z3
    #[kani::proof]
    fn proof_backend_id_is_z3() {
        let backend = Z3Backend::new();
        kani::assert(backend.id() == BackendId::Z3, "ID should be Z3");
    }

    /// Verify supports() includes Theorem
    #[kani::proof]
    fn proof_z3_supports_theorem() {
        let backend = Z3Backend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::Theorem),
            "Should support Theorem",
        );
    }

    /// Verify supports() includes Contract
    #[kani::proof]
    fn proof_z3_supports_contract() {
        let backend = Z3Backend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::Contract),
            "Should support Contract",
        );
    }

    /// Verify parse_int_value parses positive integers
    #[kani::proof]
    fn proof_parse_int_value_positive() {
        let result = Z3Backend::parse_int_value("42");
        kani::assert(result == Some(42), "Should parse 42");
    }

    /// Verify parse_int_value parses zero
    #[kani::proof]
    fn proof_parse_int_value_zero() {
        let result = Z3Backend::parse_int_value("0");
        kani::assert(result == Some(0), "Should parse 0");
    }

    /// Verify parse_int_value parses negative with parens
    #[kani::proof]
    fn proof_parse_int_value_negative_parens() {
        let result = Z3Backend::parse_int_value("(- 42)");
        kani::assert(result == Some(-42), "Should parse (- 42) as -42");
    }

    /// Verify parse_int_value returns None for non-numeric
    #[kani::proof]
    fn proof_parse_int_value_invalid() {
        let result = Z3Backend::parse_int_value("abc");
        kani::assert(result.is_none(), "Should return None for invalid input");
    }

    /// Verify parse_real_value parses decimal
    #[kani::proof]
    fn proof_parse_real_value_decimal() {
        let result = Z3Backend::parse_real_value("2.5");
        if let Some(v) = result {
            kani::assert(v > 2.4 && v < 2.6, "Should parse 2.5");
        } else {
            kani::assert(false, "Should parse decimal");
        }
    }

    /// Verify parse_real_value parses rational
    #[kani::proof]
    fn proof_parse_real_value_rational() {
        let result = Z3Backend::parse_real_value("(/ 1.0 2.0)");
        if let Some(v) = result {
            kani::assert(v > 0.4 && v < 0.6, "Should parse as 0.5");
        } else {
            kani::assert(false, "Should parse rational");
        }
    }

    /// Verify parse_smt_value parses bool true
    #[kani::proof]
    fn proof_parse_smt_value_bool_true() {
        let result = Z3Backend::parse_smt_value("true", "Bool");
        kani::assert(
            result == CounterexampleValue::Bool(true),
            "Should parse true",
        );
    }

    /// Verify parse_smt_value parses bool false
    #[kani::proof]
    fn proof_parse_smt_value_bool_false() {
        let result = Z3Backend::parse_smt_value("false", "Bool");
        kani::assert(
            result == CounterexampleValue::Bool(false),
            "Should parse false",
        );
    }

    /// Verify parse_smt_value parses integer
    #[kani::proof]
    fn proof_parse_smt_value_int() {
        let result = Z3Backend::parse_smt_value("123", "Int");
        if let CounterexampleValue::Int { value, .. } = result {
            kani::assert(value == 123, "Should parse 123");
        } else {
            kani::assert(false, "Should be Int variant");
        }
    }

    /// Verify parse_smt_value parses binary bitvector
    #[kani::proof]
    fn proof_parse_smt_value_bitvector_binary() {
        let result = Z3Backend::parse_smt_value("#b1010", "(_ BitVec 4)");
        if let CounterexampleValue::Int { value, .. } = result {
            kani::assert(value == 10, "#b1010 should be 10");
        } else {
            kani::assert(false, "Should be Int variant");
        }
    }

    /// Verify parse_smt_value parses hex bitvector
    #[kani::proof]
    fn proof_parse_smt_value_bitvector_hex() {
        let result = Z3Backend::parse_smt_value("#xFF", "(_ BitVec 8)");
        if let CounterexampleValue::Int { value, .. } = result {
            kani::assert(value == 255, "#xFF should be 255");
        } else {
            kani::assert(false, "Should be Int variant");
        }
    }

    /// Verify parse_smt_value parses string
    #[kani::proof]
    fn proof_parse_smt_value_string() {
        let result = Z3Backend::parse_smt_value("\"hello\"", "String");
        kani::assert(
            result == CounterexampleValue::String("hello".to_string()),
            "Should parse string",
        );
    }

    /// Verify extract_definitions parses single definition
    #[kani::proof]
    fn proof_extract_definitions_single() {
        let model = "(model\n  (define-fun x () Int 5)\n)";
        let defs = Z3Backend::extract_definitions(model);
        kani::assert(defs.len() == 1, "Should extract one definition");
        kani::assert(defs[0].0 == "x", "Name should be x");
        kani::assert(defs[0].1 == "Int", "Type should be Int");
    }

    /// Verify extract_model returns None for no model
    #[kani::proof]
    fn proof_extract_model_none() {
        let lines = vec!["unsat"];
        let model = Z3Backend::extract_model(&lines);
        kani::assert(model.is_none(), "Should return None when no model");
    }

    /// Verify extract_model extracts model
    #[kani::proof]
    fn proof_extract_model_present() {
        let lines = vec!["sat", "(model", "  (define-fun x () Int 5)", ")"];
        let model = Z3Backend::extract_model(&lines);
        kani::assert(model.is_some(), "Should extract model");
        kani::assert(
            model.unwrap().contains("define-fun"),
            "Model should contain define-fun",
        );
    }
}

/// Configuration for Z3 backend
#[derive(Debug, Clone)]
pub struct Z3Config {
    /// Path to z3 binary
    pub z3_path: Option<PathBuf>,
    /// Timeout for verification
    pub timeout: Duration,
    /// SMT-LIB2 logic to use (e.g., "ALL", "QF_LIA", "QF_LRA")
    pub logic: String,
    /// Enable model generation
    pub produce_models: bool,
    /// Memory limit in MB (0 = no limit)
    pub memory_limit: u32,
}

impl Default for Z3Config {
    fn default() -> Self {
        Self {
            z3_path: None,
            timeout: Duration::from_secs(60),
            logic: "ALL".to_string(),
            produce_models: true,
            memory_limit: 0,
        }
    }
}

/// Z3 SMT solver backend
pub struct Z3Backend {
    config: Z3Config,
}

impl Default for Z3Backend {
    fn default() -> Self {
        Self::new()
    }
}

impl Z3Backend {
    /// Create a new Z3 backend with default configuration
    pub fn new() -> Self {
        Self {
            config: Z3Config::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: Z3Config) -> Self {
        Self { config }
    }

    async fn detect_z3(&self) -> Result<PathBuf, String> {
        let z3_path = self
            .config
            .z3_path
            .clone()
            .or_else(|| which::which("z3").ok())
            .ok_or("Z3 not found. Install from https://github.com/Z3Prover/z3/releases")?;

        let output = Command::new(&z3_path)
            .arg("--version")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| format!("Failed to execute z3: {}", e))?;

        if output.status.success() {
            let version = String::from_utf8_lossy(&output.stdout);
            debug!("Detected Z3 version: {}", version.trim());
            Ok(z3_path)
        } else {
            Err("Z3 version check failed".to_string())
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
                // unsat means the negation of property is unsatisfiable
                // therefore the property holds
                return (VerificationStatus::Proven, None);
            } else if trimmed == "sat" {
                // sat means a counterexample exists
                // Extract model if present
                let model = Self::extract_model(&lines);
                return (VerificationStatus::Disproven, model);
            } else if trimmed == "unknown" {
                return (
                    VerificationStatus::Unknown {
                        reason: "Z3 returned unknown (timeout or resource limit)".to_string(),
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
                        "Z3 error: {}",
                        combined.lines().take(3).collect::<Vec<_>>().join("; ")
                    ),
                },
                None,
            );
        }

        (
            VerificationStatus::Unknown {
                reason: "Could not parse Z3 output".to_string(),
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

    /// Parse a raw Z3 model into a structured counterexample
    ///
    /// Z3 model format (SMT-LIB2):
    /// ```text
    /// (model
    ///   (define-fun x () Int 5)
    ///   (define-fun y () Bool true)
    ///   (define-fun z () Real (/ 3.0 2.0))
    /// )
    /// ```
    fn parse_counterexample(&self, model_str: &str) -> StructuredCounterexample {
        let mut ce = StructuredCounterexample::new();
        ce.raw = Some(model_str.to_string());

        // Parse each define-fun in the model
        let definitions = Self::extract_definitions(model_str);
        for (name, sort, value) in definitions {
            let parsed_value = Self::parse_smt_value(&value, &sort);
            ce.witness.insert(name, parsed_value);
        }

        ce
    }

    /// Extract (name, sort, value) tuples from SMT-LIB2 model
    fn extract_definitions(model_str: &str) -> Vec<(String, String, String)> {
        let mut definitions = Vec::new();

        // Simple regex-like parsing for define-fun
        // Format: (define-fun name () sort value)
        // or: (define-fun name ((arg type)) result_sort value)
        let chars: Vec<char> = model_str.chars().collect();
        let mut pos = 0;

        while pos < chars.len() {
            // Look for "(define-fun"
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

    /// Find the next "(define-fun" starting from pos
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

    /// Parse a define-fun s-expression starting at pos
    /// Returns (name, sort, value, end_pos)
    fn parse_define_fun(chars: &[char], start: usize) -> Option<(String, String, String, usize)> {
        // Skip "(define-fun "
        let mut pos = start + "(define-fun ".len();

        // Skip whitespace
        while pos < chars.len() && chars[pos].is_whitespace() {
            pos += 1;
        }

        // Read name (until whitespace or '(')
        let name_start = pos;
        while pos < chars.len() && !chars[pos].is_whitespace() && chars[pos] != '(' {
            pos += 1;
        }
        let name: String = chars[name_start..pos].iter().collect();

        // Skip whitespace
        while pos < chars.len() && chars[pos].is_whitespace() {
            pos += 1;
        }

        // Skip the argument list "()" or "((x Int)...)"
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

        // Read sort (either simple like "Int" or compound like "(Array Int Int)")
        let sort = if pos < chars.len() && chars[pos] == '(' {
            // Compound sort
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
            // Simple sort
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

        // Read value (may be s-expression or simple value)
        let value = if pos < chars.len() && chars[pos] == '(' {
            // S-expression value
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
            // Simple value
            let value_start = pos;
            while pos < chars.len() && !chars[pos].is_whitespace() && chars[pos] != ')' {
                pos += 1;
            }
            chars[value_start..pos].iter().collect()
        };

        // Find the closing ')' for the define-fun
        while pos < chars.len() && chars[pos] != ')' {
            pos += 1;
        }
        if pos < chars.len() {
            pos += 1; // Skip the closing ')'
        }

        if name.is_empty() {
            None
        } else {
            Some((name, sort, value, pos))
        }
    }

    /// Parse an SMT-LIB2 value into a CounterexampleValue
    fn parse_smt_value(value: &str, sort: &str) -> CounterexampleValue {
        let trimmed = value.trim();

        // Boolean values
        if trimmed == "true" {
            return CounterexampleValue::Bool(true);
        }
        if trimmed == "false" {
            return CounterexampleValue::Bool(false);
        }

        // Integer values (including negative: (- 5))
        if let Some(int_val) = Self::parse_int_value(trimmed) {
            return CounterexampleValue::Int {
                value: int_val,
                type_hint: Some(sort.to_string()),
            };
        }

        // Real/rational values (e.g., (/ 3.0 2.0) or 1.5)
        if sort == "Real" || sort.contains("Real") {
            if let Some(float_val) = Self::parse_real_value(trimmed) {
                return CounterexampleValue::Float { value: float_val };
            }
        }

        // BitVector values (e.g., #b1010 or #x0a)
        if let Some(binary_str) = trimmed.strip_prefix("#b") {
            if let Ok(int_val) = i128::from_str_radix(binary_str, 2) {
                return CounterexampleValue::Int {
                    value: int_val,
                    type_hint: Some(sort.to_string()),
                };
            }
        }
        if let Some(hex_str) = trimmed.strip_prefix("#x") {
            if let Ok(int_val) = i128::from_str_radix(hex_str, 16) {
                return CounterexampleValue::Int {
                    value: int_val,
                    type_hint: Some(sort.to_string()),
                };
            }
        }

        // String values (quoted)
        if trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() >= 2 {
            return CounterexampleValue::String(trimmed[1..trimmed.len() - 1].to_string());
        }

        // Array values: ((as const (Array Int Int)) 0) or (store ...)
        if sort.contains("Array") {
            return CounterexampleValue::Unknown(format!("Array: {}", trimmed));
        }

        // Default: unknown value
        CounterexampleValue::Unknown(trimmed.to_string())
    }

    /// Parse an SMT integer value, handling (- n) for negatives
    fn parse_int_value(value: &str) -> Option<i128> {
        let trimmed = value.trim();

        // Direct integer
        if let Ok(n) = trimmed.parse::<i128>() {
            return Some(n);
        }

        // Negative: (- n)
        if trimmed.starts_with("(-") && trimmed.ends_with(')') {
            let inner = trimmed[2..trimmed.len() - 1].trim();
            if let Ok(n) = inner.parse::<i128>() {
                return Some(-n);
            }
        }

        // Alternative negative: (- n) with space
        if trimmed.starts_with("(- ") && trimmed.ends_with(')') {
            let inner = trimmed[3..trimmed.len() - 1].trim();
            if let Ok(n) = inner.parse::<i128>() {
                return Some(-n);
            }
        }

        None
    }

    /// Parse an SMT real value, handling (/ num denom) for rationals
    fn parse_real_value(value: &str) -> Option<f64> {
        let trimmed = value.trim();

        // Direct float
        if let Ok(f) = trimmed.parse::<f64>() {
            return Some(f);
        }

        // Rational: (/ num denom)
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

        // Negative real: (- value)
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
impl VerificationBackend for Z3Backend {
    fn id(&self) -> BackendId {
        BackendId::Z3
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

        let z3_path = self.detect_z3().await.map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        // Compile to SMT-LIB2
        let compiled = compile_to_smtlib2_with_logic(spec, &self.config.logic);
        let smt_path = temp_dir.path().join("spec.smt2");
        std::fs::write(&smt_path, &compiled.code).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write SMT-LIB2 file: {}", e))
        })?;

        let mut cmd = Command::new(&z3_path);
        cmd.arg(&smt_path)
            .arg(format!("-t:{}", self.config.timeout.as_millis()))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if self.config.memory_limit > 0 {
            cmd.arg(format!("-memory:{}", self.config.memory_limit));
        }

        let result = tokio::time::timeout(
            self.config.timeout + Duration::from_secs(5), // Grace period
            cmd.output(),
        )
        .await;

        let duration = start.elapsed();

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                debug!("Z3 stdout: {}", stdout);
                debug!("Z3 stderr: {}", stderr);

                let (status, counterexample_str) =
                    self.parse_output(&stdout, &stderr, output.status.success());

                let counterexample = counterexample_str.map(|m| self.parse_counterexample(&m));

                let diagnostics: Vec<String> = stderr
                    .lines()
                    .filter(|l| l.contains("warning") || l.contains("error"))
                    .map(String::from)
                    .collect();

                let proof = if matches!(status, VerificationStatus::Proven) {
                    Some("Verified by Z3 (unsat)".to_string())
                } else {
                    None
                };

                Ok(BackendResult {
                    backend: BackendId::Z3,
                    status,
                    proof,
                    counterexample,
                    diagnostics,
                    time_taken: duration,
                })
            }
            Ok(Err(e)) => Err(BackendError::VerificationFailed(format!(
                "Failed to execute Z3: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match self.detect_z3().await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_unsat() {
        let backend = Z3Backend::new();
        let (status, model) = backend.parse_output("unsat", "", true);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(model.is_none());
    }

    #[test]
    fn parse_sat_with_model() {
        let backend = Z3Backend::new();
        let output = "sat\n(model\n  (define-fun x () Int 5)\n)";
        let (status, model) = backend.parse_output(output, "", true);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(model.is_some());
        assert!(model.unwrap().contains("define-fun x"));
    }

    #[test]
    fn parse_unknown() {
        let backend = Z3Backend::new();
        let (status, _) = backend.parse_output("unknown", "", true);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn parse_error() {
        let backend = Z3Backend::new();
        let (status, _) = backend.parse_output("", "error: invalid syntax", false);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn default_config() {
        let config = Z3Config::default();
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.logic, "ALL");
        assert!(config.produce_models);
        assert_eq!(config.memory_limit, 0);
    }

    #[test]
    fn extract_model_basic() {
        let lines = vec!["sat", "(model", "  (define-fun x () Int 5)", ")"];
        let model = Z3Backend::extract_model(&lines);
        assert!(model.is_some());
        let m = model.unwrap();
        assert!(m.contains("model"));
        assert!(m.contains("define-fun"));
    }

    // =============================================
    // Structured counterexample parsing tests
    // =============================================

    #[test]
    fn parse_counterexample_int() {
        let backend = Z3Backend::new();
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
        let backend = Z3Backend::new();
        let model = "(model\n  (define-fun x () Int (- 42))\n)";
        let ce = backend.parse_counterexample(model);
        assert!(ce.witness.contains_key("x"));
        match &ce.witness["x"] {
            CounterexampleValue::Int { value, .. } => assert_eq!(*value, -42),
            _ => panic!("Expected Int value"),
        }
    }

    #[test]
    fn parse_counterexample_bool_true() {
        let backend = Z3Backend::new();
        let model = "(model\n  (define-fun b () Bool true)\n)";
        let ce = backend.parse_counterexample(model);
        assert!(ce.witness.contains_key("b"));
        assert_eq!(ce.witness["b"], CounterexampleValue::Bool(true));
    }

    #[test]
    fn parse_counterexample_bool_false() {
        let backend = Z3Backend::new();
        let model = "(model\n  (define-fun b () Bool false)\n)";
        let ce = backend.parse_counterexample(model);
        assert!(ce.witness.contains_key("b"));
        assert_eq!(ce.witness["b"], CounterexampleValue::Bool(false));
    }

    #[test]
    fn parse_counterexample_real() {
        let backend = Z3Backend::new();
        let model = "(model\n  (define-fun r () Real 2.75)\n)";
        let ce = backend.parse_counterexample(model);
        assert!(ce.witness.contains_key("r"));
        match &ce.witness["r"] {
            CounterexampleValue::Float { value } => assert!((value - 2.75).abs() < 0.001),
            _ => panic!("Expected Float value"),
        }
    }

    #[test]
    fn parse_counterexample_rational() {
        let backend = Z3Backend::new();
        let model = "(model\n  (define-fun r () Real (/ 3.0 2.0))\n)";
        let ce = backend.parse_counterexample(model);
        assert!(ce.witness.contains_key("r"));
        match &ce.witness["r"] {
            CounterexampleValue::Float { value } => assert!((value - 1.5).abs() < 0.001),
            _ => panic!("Expected Float value"),
        }
    }

    #[test]
    fn parse_counterexample_bitvector_binary() {
        let backend = Z3Backend::new();
        let model = "(model\n  (define-fun bv () (_ BitVec 8) #b10101010)\n)";
        let ce = backend.parse_counterexample(model);
        assert!(ce.witness.contains_key("bv"));
        match &ce.witness["bv"] {
            CounterexampleValue::Int { value, .. } => assert_eq!(*value, 0b10101010),
            _ => panic!("Expected Int value"),
        }
    }

    #[test]
    fn parse_counterexample_bitvector_hex() {
        let backend = Z3Backend::new();
        let model = "(model\n  (define-fun bv () (_ BitVec 16) #xDEAD)\n)";
        let ce = backend.parse_counterexample(model);
        assert!(ce.witness.contains_key("bv"));
        match &ce.witness["bv"] {
            CounterexampleValue::Int { value, .. } => assert_eq!(*value, 0xDEAD),
            _ => panic!("Expected Int value"),
        }
    }

    #[test]
    fn parse_counterexample_string() {
        let backend = Z3Backend::new();
        let model = "(model\n  (define-fun s () String \"hello\")\n)";
        let ce = backend.parse_counterexample(model);
        assert!(ce.witness.contains_key("s"));
        assert_eq!(
            ce.witness["s"],
            CounterexampleValue::String("hello".to_string())
        );
    }

    #[test]
    fn parse_counterexample_multiple_vars() {
        let backend = Z3Backend::new();
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
        match &ce.witness["x"] {
            CounterexampleValue::Int { value, .. } => assert_eq!(*value, 10),
            _ => panic!("Expected Int"),
        }
        match &ce.witness["y"] {
            CounterexampleValue::Int { value, .. } => assert_eq!(*value, 20),
            _ => panic!("Expected Int"),
        }
        assert_eq!(ce.witness["z"], CounterexampleValue::Bool(true));
    }

    #[test]
    fn parse_counterexample_array() {
        let backend = Z3Backend::new();
        let model =
            "(model\n  (define-fun arr () (Array Int Int) ((as const (Array Int Int)) 0))\n)";
        let ce = backend.parse_counterexample(model);
        assert!(ce.witness.contains_key("arr"));
        // Arrays are parsed as Unknown with "Array:" prefix
        match &ce.witness["arr"] {
            CounterexampleValue::Unknown(s) => assert!(s.starts_with("Array:")),
            _ => panic!("Expected Unknown(Array:...)"),
        }
    }

    #[test]
    fn parse_counterexample_preserves_raw() {
        let backend = Z3Backend::new();
        let model = "(model\n  (define-fun x () Int 5)\n)";
        let ce = backend.parse_counterexample(model);
        assert!(ce.raw.is_some());
        assert!(ce.raw.unwrap().contains("define-fun x"));
    }

    #[test]
    fn parse_counterexample_empty_model() {
        let backend = Z3Backend::new();
        let model = "(model\n)";
        let ce = backend.parse_counterexample(model);
        assert!(ce.witness.is_empty());
    }

    #[test]
    fn parse_counterexample_negative_int_with_space() {
        let backend = Z3Backend::new();
        let model = "(model\n  (define-fun x () Int (- 100))\n)";
        let ce = backend.parse_counterexample(model);
        match &ce.witness["x"] {
            CounterexampleValue::Int { value, .. } => assert_eq!(*value, -100),
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn parse_int_value_positive() {
        assert_eq!(Z3Backend::parse_int_value("42"), Some(42));
    }

    #[test]
    fn parse_int_value_negative_parens() {
        assert_eq!(Z3Backend::parse_int_value("(- 42)"), Some(-42));
        assert_eq!(Z3Backend::parse_int_value("(-42)"), Some(-42));
    }

    #[test]
    fn parse_int_value_zero() {
        assert_eq!(Z3Backend::parse_int_value("0"), Some(0));
    }

    #[test]
    fn parse_int_value_invalid() {
        assert_eq!(Z3Backend::parse_int_value("abc"), None);
        assert_eq!(Z3Backend::parse_int_value(""), None);
    }

    #[test]
    fn parse_real_value_decimal() {
        let val = Z3Backend::parse_real_value("2.75").unwrap();
        assert!((val - 2.75).abs() < 0.00001);
    }

    #[test]
    fn parse_real_value_rational() {
        let val = Z3Backend::parse_real_value("(/ 1.0 3.0)").unwrap();
        assert!((val - 0.33333).abs() < 0.001);
    }

    #[test]
    fn parse_real_value_negative() {
        let val = Z3Backend::parse_real_value("(- 2.5)").unwrap();
        assert!((val - (-2.5)).abs() < 0.001);
    }

    #[test]
    fn extract_definitions_single() {
        let model = "(model\n  (define-fun x () Int 5)\n)";
        let defs = Z3Backend::extract_definitions(model);
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].0, "x");
        assert_eq!(defs[0].1, "Int");
        assert_eq!(defs[0].2, "5");
    }

    #[test]
    fn extract_definitions_multiple() {
        let model = "(model\n  (define-fun a () Int 1)\n  (define-fun b () Bool true)\n)";
        let defs = Z3Backend::extract_definitions(model);
        assert_eq!(defs.len(), 2);
    }

    #[test]
    fn extract_definitions_compound_sort() {
        let model = "(model\n  (define-fun arr () (Array Int Int) 0)\n)";
        let defs = Z3Backend::extract_definitions(model);
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].0, "arr");
        assert!(defs[0].1.contains("Array"));
    }

    #[test]
    fn parse_smt_value_bool() {
        assert_eq!(
            Z3Backend::parse_smt_value("true", "Bool"),
            CounterexampleValue::Bool(true)
        );
        assert_eq!(
            Z3Backend::parse_smt_value("false", "Bool"),
            CounterexampleValue::Bool(false)
        );
    }

    #[test]
    fn parse_smt_value_int() {
        match Z3Backend::parse_smt_value("123", "Int") {
            CounterexampleValue::Int { value, .. } => assert_eq!(value, 123),
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn has_structured_data() {
        let backend = Z3Backend::new();
        let model = "(model\n  (define-fun x () Int 5)\n)";
        let ce = backend.parse_counterexample(model);
        assert!(ce.has_structured_data());
    }

    #[test]
    fn structured_counterexample_summary() {
        let backend = Z3Backend::new();
        let model = "(model\n  (define-fun x () Int 5)\n)";
        let ce = backend.parse_counterexample(model);
        let summary = ce.summary();
        assert!(summary.contains("x = "));
    }
}
