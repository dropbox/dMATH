//! OpenSMT solver backend
//!
//! OpenSMT is an open-source SMT solver developed at USI Lugano.
//! It supports QF_LRA, QF_LIA, QF_UF, and other theories.
//!
//! OpenSMT is unique among SMT solvers for its:
//! - Strong interpolation support (Craig interpolants)
//! - Incremental solving with proof production
//! - Clean modular design
//!
//! See: <https://verify.inf.usi.ch/opensmt>

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

/// Configuration for OpenSMT backend
#[derive(Debug, Clone)]
pub struct OpenSmtConfig {
    /// Path to OpenSMT binary
    pub opensmt_path: Option<PathBuf>,
    /// Timeout for solving
    pub timeout: Duration,
    /// Enable verbose output
    pub verbose: bool,
    /// Enable proof production
    pub produce_proofs: bool,
    /// SMT-LIB2 logic to use (e.g., "QF_LRA", "QF_LIA", "QF_UF")
    pub logic: String,
    /// Enable model generation for sat results
    pub produce_models: bool,
}

impl Default for OpenSmtConfig {
    fn default() -> Self {
        Self {
            opensmt_path: None,
            timeout: Duration::from_secs(60),
            verbose: false,
            produce_proofs: false,
            logic: "QF_LIA".to_string(), // OpenSMT works best with quantifier-free logics
            produce_models: true,
        }
    }
}

/// OpenSMT solver backend
pub struct OpenSmtBackend {
    config: OpenSmtConfig,
}

impl Default for OpenSmtBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenSmtBackend {
    /// Create a new OpenSMT backend with default configuration
    pub fn new() -> Self {
        Self {
            config: OpenSmtConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: OpenSmtConfig) -> Self {
        Self { config }
    }

    async fn detect_opensmt(&self) -> Result<PathBuf, String> {
        let opensmt_path = self
            .config
            .opensmt_path
            .clone()
            .or_else(|| which::which("opensmt").ok())
            .ok_or("OpenSMT not found. Build from https://verify.inf.usi.ch/opensmt")?;

        // Verify the binary works
        let output = Command::new(&opensmt_path)
            .arg("--help")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| format!("Failed to execute OpenSMT: {}", e))?;

        // OpenSMT may return non-zero for --help, so just check we got output
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stdout.contains("opensmt") || stderr.contains("opensmt") || stdout.contains("Usage") {
            debug!("Detected OpenSMT");
            Ok(opensmt_path)
        } else {
            Err("OpenSMT detection failed".to_string())
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
                        reason: "OpenSMT returned unknown (timeout or incomplete)".to_string(),
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
                        "OpenSMT error: {}",
                        combined.lines().take(3).collect::<Vec<_>>().join("; ")
                    ),
                },
                None,
            );
        }

        (
            VerificationStatus::Unknown {
                reason: "Could not parse OpenSMT output".to_string(),
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

    /// Parse an OpenSMT model into a structured counterexample
    /// OpenSMT uses standard SMT-LIB2 model format
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
        // Skip "(define-fun "
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

        // Integer values
        if let Some(int_val) = Self::parse_int_value(trimmed) {
            return CounterexampleValue::Int {
                value: int_val,
                type_hint: Some(sort.to_string()),
            };
        }

        // Real/rational values
        if sort == "Real" || sort.contains("Real") {
            if let Some(float_val) = Self::parse_real_value(trimmed) {
                return CounterexampleValue::Float { value: float_val };
            }
        }

        // String values
        if trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() >= 2 {
            return CounterexampleValue::String(trimmed[1..trimmed.len() - 1].to_string());
        }

        // Default: unknown value
        CounterexampleValue::Unknown(trimmed.to_string())
    }

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

        // Negative with space: (- n)
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
impl VerificationBackend for OpenSmtBackend {
    fn id(&self) -> BackendId {
        BackendId::OpenSMT
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

        let opensmt_path = self
            .detect_opensmt()
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

        // Build command - OpenSMT takes the file as an argument
        let mut cmd = Command::new(&opensmt_path);
        cmd.arg(&smt_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let result =
            tokio::time::timeout(self.config.timeout + Duration::from_secs(5), cmd.output()).await;

        let duration = start.elapsed();

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                debug!("OpenSMT stdout: {}", stdout);
                debug!("OpenSMT stderr: {}", stderr);

                let (status, counterexample_str) =
                    self.parse_output(&stdout, &stderr, output.status.success());

                let counterexample = counterexample_str.map(|m| self.parse_counterexample(&m));

                let diagnostics: Vec<String> = stderr
                    .lines()
                    .filter(|l| l.contains("warning") || l.contains("error"))
                    .map(String::from)
                    .collect();

                let proof = if matches!(status, VerificationStatus::Proven) {
                    Some("Verified by OpenSMT (unsat)".to_string())
                } else {
                    None
                };

                Ok(BackendResult {
                    backend: BackendId::OpenSMT,
                    status,
                    proof,
                    counterexample,
                    diagnostics,
                    time_taken: duration,
                })
            }
            Ok(Err(e)) => Err(BackendError::VerificationFailed(format!(
                "Failed to execute OpenSMT: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match self.detect_opensmt().await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ===== OpenSmtConfig defaults =====

    #[kani::proof]
    fn verify_config_defaults() {
        let config = OpenSmtConfig::default();
        assert!(config.opensmt_path.is_none());
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert!(!config.verbose);
        assert!(!config.produce_proofs);
        assert_eq!(config.logic, "QF_LIA");
        assert!(config.produce_models);
    }

    // ===== Backend construction =====

    #[kani::proof]
    fn verify_backend_new_uses_defaults() {
        let backend = OpenSmtBackend::new();
        assert_eq!(backend.config.timeout, Duration::from_secs(60));
        assert_eq!(backend.config.logic, "QF_LIA");
    }

    #[kani::proof]
    fn verify_backend_default_equals_new() {
        let a = OpenSmtBackend::new();
        let b = OpenSmtBackend::default();
        assert_eq!(a.config.timeout, b.config.timeout);
        assert_eq!(a.config.logic, b.config.logic);
    }

    #[kani::proof]
    fn verify_backend_with_config_preserves_logic_and_models() {
        let cfg = OpenSmtConfig {
            opensmt_path: Some(PathBuf::from("/opt/opensmt")),
            timeout: Duration::from_secs(10),
            verbose: true,
            produce_proofs: true,
            logic: "QF_UF".to_string(),
            produce_models: false,
        };
        let backend = OpenSmtBackend::with_config(cfg);
        assert_eq!(backend.config.timeout, Duration::from_secs(10));
        assert!(backend.config.verbose);
        assert!(backend.config.produce_proofs);
        assert_eq!(backend.config.logic, "QF_UF");
        assert!(!backend.config.produce_models);
    }

    // ===== ID and supports =====

    #[kani::proof]
    fn verify_backend_id() {
        let backend = OpenSmtBackend::new();
        assert!(matches!(backend.id(), BackendId::OpenSMT));
    }

    #[kani::proof]
    fn verify_supports_three_property_types() {
        let backend = OpenSmtBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::Theorem));
        assert!(supported.contains(&PropertyType::Invariant));
        assert!(supported.contains(&PropertyType::Contract));
        assert_eq!(supported.len(), 3);
    }

    // ===== Output parsing =====

    #[kani::proof]
    fn verify_parse_output_unsat_is_proven() {
        let backend = OpenSmtBackend::new();
        let (status, model) = backend.parse_output("unsat", "", true);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(model.is_none());
    }

    #[kani::proof]
    fn verify_parse_output_sat_is_disproven() {
        let backend = OpenSmtBackend::new();
        let (status, model) =
            backend.parse_output("sat\n(model (define-fun x () Int 1))", "", true);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(model.is_some());
    }

    #[kani::proof]
    fn verify_parse_output_error_is_unknown() {
        let backend = OpenSmtBackend::new();
        let (status, model) = backend.parse_output("", "error: bad input", false);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
        assert!(model.is_none());
    }

    // ===== Counterexample parsing =====

    #[kani::proof]
    fn verify_parse_counterexample_handles_multiple_sorts() {
        let backend = OpenSmtBackend::new();
        let model = "(model
  (define-fun x () Int -5)
  (define-fun y () Real 2.5)
  (define-fun b () Bool true)
)";
        let ce = backend.parse_counterexample(model);
        assert!(matches!(
            ce.witness.get("b"),
            Some(CounterexampleValue::Bool(true))
        ));
        match ce.witness.get("x") {
            Some(CounterexampleValue::Int { value, .. }) => assert_eq!(*value, -5),
            _ => panic!("expected int value"),
        }
        match ce.witness.get("y") {
            Some(CounterexampleValue::Float { value }) => assert!((*value - 2.5).abs() < 0.001),
            _ => panic!("expected float value"),
        }
        assert!(ce.raw.as_ref().unwrap().contains("define-fun"));
    }

    #[kani::proof]
    fn verify_parse_int_value_supports_negative_forms() {
        assert_eq!(OpenSmtBackend::parse_int_value("(- 3)"), Some(-3));
        assert_eq!(OpenSmtBackend::parse_int_value("(-3)"), Some(-3));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = OpenSmtConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.logic, "QF_LIA");
        assert!(config.produce_models);
    }

    #[test]
    fn backend_id() {
        let backend = OpenSmtBackend::new();
        assert_eq!(backend.id(), BackendId::OpenSMT);
    }

    #[test]
    fn parse_unsat() {
        let backend = OpenSmtBackend::new();
        let (status, model) = backend.parse_output("unsat", "", true);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(model.is_none());
    }

    #[test]
    fn parse_sat() {
        let backend = OpenSmtBackend::new();
        let (status, _) = backend.parse_output("sat", "", true);
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[test]
    fn parse_unknown() {
        let backend = OpenSmtBackend::new();
        let (status, _) = backend.parse_output("unknown", "", true);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn parse_error() {
        let backend = OpenSmtBackend::new();
        let (status, _) = backend.parse_output("", "error: invalid syntax", false);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn extract_model_basic() {
        let lines = vec!["sat", "(model", "  (define-fun x () Int 5)", ")"];
        let model = OpenSmtBackend::extract_model(&lines);
        assert!(model.is_some());
        let m = model.unwrap();
        assert!(m.contains("model"));
        assert!(m.contains("define-fun"));
    }

    // Counterexample parsing tests
    #[test]
    fn parse_counterexample_int() {
        let backend = OpenSmtBackend::new();
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
        let backend = OpenSmtBackend::new();
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
        let backend = OpenSmtBackend::new();
        let model = "(model\n  (define-fun b () Bool true)\n)";
        let ce = backend.parse_counterexample(model);
        assert!(ce.witness.contains_key("b"));
        assert_eq!(ce.witness["b"], CounterexampleValue::Bool(true));
    }

    #[test]
    fn parse_counterexample_real() {
        let backend = OpenSmtBackend::new();
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
        let backend = OpenSmtBackend::new();
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
        let backend = OpenSmtBackend::new();
        let model = "(model\n  (define-fun x () Int 5)\n)";
        let ce = backend.parse_counterexample(model);
        assert!(ce.raw.is_some());
        assert!(ce.raw.unwrap().contains("define-fun x"));
    }

    #[test]
    fn parse_int_value_positive() {
        assert_eq!(OpenSmtBackend::parse_int_value("42"), Some(42));
    }

    #[test]
    fn parse_int_value_negative_parens() {
        assert_eq!(OpenSmtBackend::parse_int_value("(- 42)"), Some(-42));
        assert_eq!(OpenSmtBackend::parse_int_value("(-42)"), Some(-42));
    }

    #[test]
    fn parse_int_value_zero() {
        assert_eq!(OpenSmtBackend::parse_int_value("0"), Some(0));
    }

    #[test]
    fn parse_int_value_invalid() {
        assert_eq!(OpenSmtBackend::parse_int_value("abc"), None);
    }

    #[test]
    fn parse_real_value_decimal() {
        let val = OpenSmtBackend::parse_real_value("2.75").unwrap();
        assert!((val - 2.75).abs() < 0.00001);
    }

    #[test]
    fn parse_real_value_rational() {
        let val = OpenSmtBackend::parse_real_value("(/ 1.0 3.0)").unwrap();
        assert!((val - 0.33333).abs() < 0.001);
    }

    #[test]
    fn has_structured_data() {
        let backend = OpenSmtBackend::new();
        let model = "(model\n  (define-fun x () Int 5)\n)";
        let ce = backend.parse_counterexample(model);
        assert!(ce.has_structured_data());
    }
}
