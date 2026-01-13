//! CVC5 SMT solver backend
//!
//! CVC5 is an open-source SMT solver developed by Stanford University and
//! the University of Iowa. It's the successor to CVC4 and supports SMT-LIB2.
//!
//! CVC5 excels at:
//! - Strings and regular expressions
//! - Finite sets and relations
//! - Datatypes and recursive functions
//! - Bit-vectors and floating-point arithmetic
//!
//! See: <https://cvc5.github.io/>

// =============================================
// Kani Proofs for CVC5 Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- Cvc5Config Default Tests ----

    /// Verify Cvc5Config::default timeout is 60 seconds
    #[kani::proof]
    fn proof_cvc5_config_default_timeout() {
        let config = Cvc5Config::default();
        kani::assert(
            config.timeout == Duration::from_secs(60),
            "Default timeout should be 60 seconds",
        );
    }

    /// Verify Cvc5Config::default cvc5_path is None
    #[kani::proof]
    fn proof_cvc5_config_default_path_none() {
        let config = Cvc5Config::default();
        kani::assert(
            config.cvc5_path.is_none(),
            "Default cvc5_path should be None",
        );
    }

    /// Verify Cvc5Config::default logic is "ALL"
    #[kani::proof]
    fn proof_cvc5_config_default_logic() {
        let config = Cvc5Config::default();
        kani::assert(config.logic == "ALL", "Default logic should be ALL");
    }

    /// Verify Cvc5Config::default produce_models is true
    #[kani::proof]
    fn proof_cvc5_config_default_produce_models() {
        let config = Cvc5Config::default();
        kani::assert(
            config.produce_models,
            "Default produce_models should be true",
        );
    }

    /// Verify Cvc5Config::default incremental is false
    #[kani::proof]
    fn proof_cvc5_config_default_incremental() {
        let config = Cvc5Config::default();
        kani::assert(!config.incremental, "Default incremental should be false");
    }

    /// Verify Cvc5Config::default finite_model_find is false
    #[kani::proof]
    fn proof_cvc5_config_default_finite_model_find() {
        let config = Cvc5Config::default();
        kani::assert(
            !config.finite_model_find,
            "Default finite_model_find should be false",
        );
    }

    // ---- Cvc5Backend Construction Tests ----

    /// Verify Cvc5Backend::new uses default config timeout
    #[kani::proof]
    fn proof_cvc5_backend_new_defaults_timeout() {
        let backend = Cvc5Backend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(60),
            "New backend should use default timeout",
        );
    }

    /// Verify Cvc5Backend::new uses default config logic
    #[kani::proof]
    fn proof_cvc5_backend_new_defaults_logic() {
        let backend = Cvc5Backend::new();
        kani::assert(
            backend.config.logic == "ALL",
            "New backend should use default logic",
        );
    }

    /// Verify Cvc5Backend::default equals Cvc5Backend::new timeout
    #[kani::proof]
    fn proof_cvc5_backend_default_equals_new_timeout() {
        let default_backend = Cvc5Backend::default();
        let new_backend = Cvc5Backend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "Default and new should have same timeout",
        );
    }

    /// Verify Cvc5Backend::default equals Cvc5Backend::new logic
    #[kani::proof]
    fn proof_cvc5_backend_default_equals_new_logic() {
        let default_backend = Cvc5Backend::default();
        let new_backend = Cvc5Backend::new();
        kani::assert(
            default_backend.config.logic == new_backend.config.logic,
            "Default and new should have same logic",
        );
    }

    /// Verify Cvc5Backend::with_config preserves custom timeout
    #[kani::proof]
    fn proof_cvc5_backend_with_config_timeout() {
        let config = Cvc5Config {
            cvc5_path: None,
            timeout: Duration::from_secs(300),
            logic: "ALL".to_string(),
            produce_models: true,
            incremental: false,
            finite_model_find: false,
        };
        let backend = Cvc5Backend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(300),
            "with_config should preserve custom timeout",
        );
    }

    /// Verify Cvc5Backend::with_config preserves produce_models
    #[kani::proof]
    fn proof_cvc5_backend_with_config_produce_models() {
        let config = Cvc5Config {
            cvc5_path: None,
            timeout: Duration::from_secs(60),
            logic: "ALL".to_string(),
            produce_models: false,
            incremental: false,
            finite_model_find: false,
        };
        let backend = Cvc5Backend::with_config(config);
        kani::assert(
            !backend.config.produce_models,
            "with_config should preserve produce_models",
        );
    }

    /// Verify Cvc5Backend::with_config preserves incremental
    #[kani::proof]
    fn proof_cvc5_backend_with_config_incremental() {
        let config = Cvc5Config {
            cvc5_path: None,
            timeout: Duration::from_secs(60),
            logic: "ALL".to_string(),
            produce_models: true,
            incremental: true,
            finite_model_find: false,
        };
        let backend = Cvc5Backend::with_config(config);
        kani::assert(
            backend.config.incremental,
            "with_config should preserve incremental",
        );
    }

    /// Verify Cvc5Backend::with_config preserves finite_model_find
    #[kani::proof]
    fn proof_cvc5_backend_with_config_finite_model_find() {
        let config = Cvc5Config {
            cvc5_path: None,
            timeout: Duration::from_secs(60),
            logic: "ALL".to_string(),
            produce_models: true,
            incremental: false,
            finite_model_find: true,
        };
        let backend = Cvc5Backend::with_config(config);
        kani::assert(
            backend.config.finite_model_find,
            "with_config should preserve finite_model_find",
        );
    }

    // ---- Backend Trait Implementation Tests ----

    /// Verify Cvc5Backend::id returns BackendId::Cvc5
    #[kani::proof]
    fn proof_cvc5_backend_id() {
        let backend = Cvc5Backend::new();
        kani::assert(backend.id() == BackendId::Cvc5, "Backend id should be Cvc5");
    }

    /// Verify Cvc5Backend::supports includes Theorem
    #[kani::proof]
    fn proof_cvc5_backend_supports_theorem() {
        let backend = Cvc5Backend::new();
        let supported = backend.supports();
        let has_theorem = supported.iter().any(|p| *p == PropertyType::Theorem);
        kani::assert(has_theorem, "Backend should support Theorem");
    }

    /// Verify Cvc5Backend::supports includes Invariant
    #[kani::proof]
    fn proof_cvc5_backend_supports_invariant() {
        let backend = Cvc5Backend::new();
        let supported = backend.supports();
        let has_invariant = supported.iter().any(|p| *p == PropertyType::Invariant);
        kani::assert(has_invariant, "Backend should support Invariant");
    }

    /// Verify Cvc5Backend::supports includes Contract
    #[kani::proof]
    fn proof_cvc5_backend_supports_contract() {
        let backend = Cvc5Backend::new();
        let supported = backend.supports();
        let has_contract = supported.iter().any(|p| *p == PropertyType::Contract);
        kani::assert(has_contract, "Backend should support Contract");
    }

    /// Verify Cvc5Backend::supports returns exactly 3 property types
    #[kani::proof]
    fn proof_cvc5_backend_supports_count() {
        let backend = Cvc5Backend::new();
        let supported = backend.supports();
        kani::assert(
            supported.len() == 3,
            "Backend should support exactly 3 property types",
        );
    }

    // ---- parse_output Tests ----

    /// Verify parse_output returns Proven for "unsat"
    #[kani::proof]
    fn proof_cvc5_parse_output_unsat() {
        let backend = Cvc5Backend::new();
        let (status, model) = backend.parse_output("unsat", "", true);
        let is_proven = matches!(status, VerificationStatus::Proven);
        kani::assert(is_proven, "unsat should result in Proven");
        kani::assert(model.is_none(), "unsat should have no model");
    }

    /// Verify parse_output returns Disproven for "sat"
    #[kani::proof]
    fn proof_cvc5_parse_output_sat() {
        let backend = Cvc5Backend::new();
        let (status, _model) = backend.parse_output("sat", "", true);
        let is_disproven = matches!(status, VerificationStatus::Disproven);
        kani::assert(is_disproven, "sat should result in Disproven");
    }

    /// Verify parse_output returns Unknown for "unknown"
    #[kani::proof]
    fn proof_cvc5_parse_output_unknown() {
        let backend = Cvc5Backend::new();
        let (status, _model) = backend.parse_output("unknown", "", true);
        let is_unknown = matches!(status, VerificationStatus::Unknown { .. });
        kani::assert(is_unknown, "unknown should result in Unknown");
    }

    /// Verify parse_output returns Unknown for errors
    #[kani::proof]
    fn proof_cvc5_parse_output_error() {
        let backend = Cvc5Backend::new();
        let (status, _model) = backend.parse_output("error: parse error", "", false);
        let is_unknown = matches!(status, VerificationStatus::Unknown { .. });
        kani::assert(is_unknown, "error should result in Unknown");
    }

    /// Verify parse_output returns Unknown for Error in stderr
    #[kani::proof]
    fn proof_cvc5_parse_output_error_stderr() {
        let backend = Cvc5Backend::new();
        let (status, _model) = backend.parse_output("", "Error: something", false);
        let is_unknown = matches!(status, VerificationStatus::Unknown { .. });
        kani::assert(is_unknown, "Error in stderr should result in Unknown");
    }

    /// Verify parse_output returns Unknown for empty output
    #[kani::proof]
    fn proof_cvc5_parse_output_empty() {
        let backend = Cvc5Backend::new();
        let (status, _model) = backend.parse_output("", "", true);
        let is_unknown = matches!(status, VerificationStatus::Unknown { .. });
        kani::assert(is_unknown, "Empty output should result in Unknown");
    }

    // ---- parse_int_value Tests ----

    /// Verify parse_int_value parses positive integers
    #[kani::proof]
    fn proof_cvc5_parse_int_value_positive() {
        let result = Cvc5Backend::parse_int_value("42");
        kani::assert(result == Some(42), "Should parse 42");
    }

    /// Verify parse_int_value parses zero
    #[kani::proof]
    fn proof_cvc5_parse_int_value_zero() {
        let result = Cvc5Backend::parse_int_value("0");
        kani::assert(result == Some(0), "Should parse 0");
    }

    /// Verify parse_int_value parses negative with space: (- n)
    #[kani::proof]
    fn proof_cvc5_parse_int_value_negative_space() {
        let result = Cvc5Backend::parse_int_value("(- 42)");
        kani::assert(result == Some(-42), "Should parse (- 42)");
    }

    /// Verify parse_int_value parses negative without space: (-n)
    #[kani::proof]
    fn proof_cvc5_parse_int_value_negative_nospace() {
        let result = Cvc5Backend::parse_int_value("(-42)");
        kani::assert(result == Some(-42), "Should parse (-42)");
    }

    /// Verify parse_int_value returns None for invalid input
    #[kani::proof]
    fn proof_cvc5_parse_int_value_invalid() {
        let result = Cvc5Backend::parse_int_value("abc");
        kani::assert(result.is_none(), "Should return None for invalid");
    }

    /// Verify parse_int_value returns None for empty input
    #[kani::proof]
    fn proof_cvc5_parse_int_value_empty() {
        let result = Cvc5Backend::parse_int_value("");
        kani::assert(result.is_none(), "Should return None for empty");
    }

    // ---- parse_real_value Tests ----

    /// Verify parse_real_value parses decimal
    #[kani::proof]
    fn proof_cvc5_parse_real_value_decimal() {
        let result = Cvc5Backend::parse_real_value("2.5");
        let is_valid = result.is_some_and(|v| (v - 2.5).abs() < 0.0001);
        kani::assert(is_valid, "Should parse 2.5");
    }

    /// Verify parse_real_value parses integer as float
    #[kani::proof]
    fn proof_cvc5_parse_real_value_integer() {
        let result = Cvc5Backend::parse_real_value("42");
        let is_valid = result.is_some_and(|v| (v - 42.0).abs() < 0.0001);
        kani::assert(is_valid, "Should parse 42 as float");
    }

    /// Verify parse_real_value returns None for invalid
    #[kani::proof]
    fn proof_cvc5_parse_real_value_invalid() {
        let result = Cvc5Backend::parse_real_value("abc");
        kani::assert(result.is_none(), "Should return None for invalid");
    }

    // ---- parse_smt_value Tests ----

    /// Verify parse_smt_value returns Bool(true) for "true"
    #[kani::proof]
    fn proof_cvc5_parse_smt_value_true() {
        let result = Cvc5Backend::parse_smt_value("true", "Bool");
        let is_true = matches!(result, CounterexampleValue::Bool(true));
        kani::assert(is_true, "Should parse true as Bool(true)");
    }

    /// Verify parse_smt_value returns Bool(false) for "false"
    #[kani::proof]
    fn proof_cvc5_parse_smt_value_false() {
        let result = Cvc5Backend::parse_smt_value("false", "Bool");
        let is_false = matches!(result, CounterexampleValue::Bool(false));
        kani::assert(is_false, "Should parse false as Bool(false)");
    }

    /// Verify parse_smt_value returns Int for integer string
    #[kani::proof]
    fn proof_cvc5_parse_smt_value_int() {
        let result = Cvc5Backend::parse_smt_value("123", "Int");
        let is_int = matches!(result, CounterexampleValue::Int { value: 123, .. });
        kani::assert(is_int, "Should parse 123 as Int");
    }

    /// Verify parse_smt_value returns Int for bitvector binary
    #[kani::proof]
    fn proof_cvc5_parse_smt_value_bv_binary() {
        let result = Cvc5Backend::parse_smt_value("#b1010", "(_ BitVec 4)");
        let is_int = matches!(result, CounterexampleValue::Int { value: 10, .. });
        kani::assert(is_int, "Should parse #b1010 as Int 10");
    }

    /// Verify parse_smt_value returns Int for bitvector hex
    #[kani::proof]
    fn proof_cvc5_parse_smt_value_bv_hex() {
        let result = Cvc5Backend::parse_smt_value("#xFF", "(_ BitVec 8)");
        let is_int = matches!(result, CounterexampleValue::Int { value: 255, .. });
        kani::assert(is_int, "Should parse #xFF as Int 255");
    }

    /// Verify parse_smt_value returns String for quoted string
    #[kani::proof]
    fn proof_cvc5_parse_smt_value_string() {
        let result = Cvc5Backend::parse_smt_value("\"hello\"", "String");
        let is_string = matches!(result, CounterexampleValue::String(s) if s == "hello");
        kani::assert(is_string, "Should parse quoted string");
    }

    /// Verify parse_smt_value returns Unknown for Set type
    #[kani::proof]
    fn proof_cvc5_parse_smt_value_set() {
        let result = Cvc5Backend::parse_smt_value("(singleton 1)", "(Set Int)");
        let is_unknown = matches!(result, CounterexampleValue::Unknown(s) if s.starts_with("Set:"));
        kani::assert(is_unknown, "Should return Unknown for Set");
    }

    /// Verify parse_smt_value returns Unknown for Seq type
    #[kani::proof]
    fn proof_cvc5_parse_smt_value_seq() {
        let result = Cvc5Backend::parse_smt_value("(seq.unit 1)", "(Seq Int)");
        let is_unknown = matches!(result, CounterexampleValue::Unknown(s) if s.starts_with("Seq:"));
        kani::assert(is_unknown, "Should return Unknown for Seq");
    }

    /// Verify parse_smt_value returns Unknown for Array type
    #[kani::proof]
    fn proof_cvc5_parse_smt_value_array() {
        let result = Cvc5Backend::parse_smt_value("const", "(Array Int Int)");
        let is_unknown =
            matches!(result, CounterexampleValue::Unknown(s) if s.starts_with("Array:"));
        kani::assert(is_unknown, "Should return Unknown for Array");
    }

    // ---- extract_model Tests ----

    /// Verify extract_model returns None for empty lines
    #[kani::proof]
    fn proof_cvc5_extract_model_empty() {
        let lines: Vec<&str> = vec![];
        let result = Cvc5Backend::extract_model(&lines);
        kani::assert(result.is_none(), "Empty lines should return None");
    }

    /// Verify extract_model returns None for no model
    #[kani::proof]
    fn proof_cvc5_extract_model_no_model() {
        let lines = vec!["unsat"];
        let result = Cvc5Backend::extract_model(&lines);
        kani::assert(result.is_none(), "No model should return None");
    }

    // ---- find_define_fun Tests ----

    /// Verify find_define_fun finds pattern at start
    #[kani::proof]
    fn proof_cvc5_find_define_fun_at_start() {
        let chars: Vec<char> = "(define-fun x () Int 5)".chars().collect();
        let result = Cvc5Backend::find_define_fun(&chars, 0);
        kani::assert(result == Some(0), "Should find at position 0");
    }

    /// Verify find_define_fun finds pattern with offset
    #[kani::proof]
    fn proof_cvc5_find_define_fun_offset() {
        let chars: Vec<char> = "  (define-fun x () Int 5)".chars().collect();
        let result = Cvc5Backend::find_define_fun(&chars, 0);
        kani::assert(result == Some(2), "Should find at position 2");
    }

    /// Verify find_define_fun returns None when not found
    #[kani::proof]
    fn proof_cvc5_find_define_fun_not_found() {
        let chars: Vec<char> = "no match here".chars().collect();
        let result = Cvc5Backend::find_define_fun(&chars, 0);
        kani::assert(result.is_none(), "Should return None when not found");
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

/// Configuration for CVC5 backend
#[derive(Debug, Clone)]
pub struct Cvc5Config {
    /// Path to cvc5 binary
    pub cvc5_path: Option<PathBuf>,
    /// Timeout for verification (in seconds)
    pub timeout: Duration,
    /// SMT-LIB2 logic to use
    pub logic: String,
    /// Enable model generation
    pub produce_models: bool,
    /// Enable incremental mode
    pub incremental: bool,
    /// Enable finite model finding
    pub finite_model_find: bool,
}

impl Default for Cvc5Config {
    fn default() -> Self {
        Self {
            cvc5_path: None,
            timeout: Duration::from_secs(60),
            logic: "ALL".to_string(),
            produce_models: true,
            incremental: false,
            finite_model_find: false,
        }
    }
}

/// CVC5 SMT solver backend
pub struct Cvc5Backend {
    config: Cvc5Config,
}

impl Default for Cvc5Backend {
    fn default() -> Self {
        Self::new()
    }
}

impl Cvc5Backend {
    /// Create a new CVC5 backend with default configuration
    pub fn new() -> Self {
        Self {
            config: Cvc5Config::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: Cvc5Config) -> Self {
        Self { config }
    }

    async fn detect_cvc5(&self) -> Result<PathBuf, String> {
        let cvc5_path = self
            .config
            .cvc5_path
            .clone()
            .or_else(|| which::which("cvc5").ok())
            .ok_or("CVC5 not found. Install from https://cvc5.github.io/")?;

        let output = Command::new(&cvc5_path)
            .arg("--version")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| format!("Failed to execute cvc5: {}", e))?;

        if output.status.success() {
            let version = String::from_utf8_lossy(&output.stdout);
            debug!("Detected CVC5 version: {}", version.trim());
            Ok(cvc5_path)
        } else {
            Err("CVC5 version check failed".to_string())
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
                // CVC5 might provide a reason
                let reason = lines
                    .iter()
                    .find(|l| l.contains("reason-unknown"))
                    .map(|l| l.to_string())
                    .unwrap_or_else(|| "CVC5 returned unknown".to_string());
                return (VerificationStatus::Unknown { reason }, None);
            }
        }

        // Check for errors
        if !success || combined.contains("error") || combined.contains("Error") {
            let error_msg = lines
                .iter()
                .filter(|l| l.contains("error") || l.contains("Error"))
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join("; ");
            return (
                VerificationStatus::Unknown {
                    reason: format!("CVC5 error: {}", error_msg),
                },
                None,
            );
        }

        (
            VerificationStatus::Unknown {
                reason: "Could not parse CVC5 output".to_string(),
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
            if trimmed.starts_with("(model") || (trimmed == "(" && depth == 0 && !in_model) {
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

    /// Parse a raw CVC5 model into a structured counterexample
    ///
    /// CVC5 model format (SMT-LIB2):
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

        // Skip argument list
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

        // BitVector values
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

        // String values
        if trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() >= 2 {
            return CounterexampleValue::String(trimmed[1..trimmed.len() - 1].to_string());
        }

        // Set values (CVC5 specific)
        if sort.contains("Set") {
            return CounterexampleValue::Unknown(format!("Set: {}", trimmed));
        }

        // Sequence values (CVC5 specific)
        if sort.contains("Seq") {
            return CounterexampleValue::Unknown(format!("Seq: {}", trimmed));
        }

        // Array values
        if sort.contains("Array") {
            return CounterexampleValue::Unknown(format!("Array: {}", trimmed));
        }

        // Default: unknown value
        CounterexampleValue::Unknown(trimmed.to_string())
    }

    /// Parse an SMT integer value
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

    /// Parse an SMT real value
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
impl VerificationBackend for Cvc5Backend {
    fn id(&self) -> BackendId {
        BackendId::Cvc5
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

        let cvc5_path = self
            .detect_cvc5()
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

        let mut cmd = Command::new(&cvc5_path);
        cmd.arg(&smt_path)
            .arg(format!("--tlimit={}", self.config.timeout.as_millis()))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if self.config.produce_models {
            cmd.arg("--produce-models");
        }

        if self.config.incremental {
            cmd.arg("--incremental");
        }

        if self.config.finite_model_find {
            cmd.arg("--finite-model-find");
        }

        let result =
            tokio::time::timeout(self.config.timeout + Duration::from_secs(5), cmd.output()).await;

        let duration = start.elapsed();

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                debug!("CVC5 stdout: {}", stdout);
                debug!("CVC5 stderr: {}", stderr);

                let (status, counterexample_str) =
                    self.parse_output(&stdout, &stderr, output.status.success());

                let counterexample = counterexample_str.map(|m| self.parse_counterexample(&m));

                let diagnostics: Vec<String> = stderr
                    .lines()
                    .filter(|l| l.contains("warning") || l.contains("error"))
                    .map(String::from)
                    .collect();

                let proof = if matches!(status, VerificationStatus::Proven) {
                    Some("Verified by CVC5 (unsat)".to_string())
                } else {
                    None
                };

                Ok(BackendResult {
                    backend: BackendId::Cvc5,
                    status,
                    proof,
                    counterexample,
                    diagnostics,
                    time_taken: duration,
                })
            }
            Ok(Err(e)) => Err(BackendError::VerificationFailed(format!(
                "Failed to execute CVC5: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match self.detect_cvc5().await {
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
        let backend = Cvc5Backend::new();
        let (status, model) = backend.parse_output("unsat", "", true);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(model.is_none());
    }

    #[test]
    fn parse_sat_with_model() {
        let backend = Cvc5Backend::new();
        let output = "sat\n(model\n  (define-fun x () Int 42)\n)";
        let (status, model) = backend.parse_output(output, "", true);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(model.is_some());
        assert!(model.unwrap().contains("42"));
    }

    #[test]
    fn parse_unknown() {
        let backend = Cvc5Backend::new();
        let (status, _) = backend.parse_output("unknown", "", true);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn parse_error() {
        let backend = Cvc5Backend::new();
        let (status, _) = backend.parse_output("", "error: parse error", false);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn default_config() {
        let config = Cvc5Config::default();
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.logic, "ALL");
        assert!(config.produce_models);
        assert!(!config.incremental);
        assert!(!config.finite_model_find);
    }

    #[test]
    fn extract_model_basic() {
        let lines = vec!["sat", "(model", "  (define-fun x () Int 10)", ")"];
        let model = Cvc5Backend::extract_model(&lines);
        assert!(model.is_some());
        let m = model.unwrap();
        assert!(m.contains("define-fun"));
    }

    // =============================================
    // Structured counterexample parsing tests
    // =============================================

    #[test]
    fn parse_counterexample_int() {
        let backend = Cvc5Backend::new();
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
        let backend = Cvc5Backend::new();
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
        let backend = Cvc5Backend::new();
        let model = "(model\n  (define-fun b () Bool true)\n)";
        let ce = backend.parse_counterexample(model);
        assert!(ce.witness.contains_key("b"));
        assert_eq!(ce.witness["b"], CounterexampleValue::Bool(true));
    }

    #[test]
    fn parse_counterexample_bool_false() {
        let backend = Cvc5Backend::new();
        let model = "(model\n  (define-fun b () Bool false)\n)";
        let ce = backend.parse_counterexample(model);
        assert!(ce.witness.contains_key("b"));
        assert_eq!(ce.witness["b"], CounterexampleValue::Bool(false));
    }

    #[test]
    fn parse_counterexample_real() {
        let backend = Cvc5Backend::new();
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
        let backend = Cvc5Backend::new();
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
        let backend = Cvc5Backend::new();
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
        let backend = Cvc5Backend::new();
        let model = "(model\n  (define-fun bv () (_ BitVec 16) #xBEEF)\n)";
        let ce = backend.parse_counterexample(model);
        assert!(ce.witness.contains_key("bv"));
        match &ce.witness["bv"] {
            CounterexampleValue::Int { value, .. } => assert_eq!(*value, 0xBEEF),
            _ => panic!("Expected Int value"),
        }
    }

    #[test]
    fn parse_counterexample_string() {
        let backend = Cvc5Backend::new();
        let model = "(model\n  (define-fun s () String \"world\")\n)";
        let ce = backend.parse_counterexample(model);
        assert!(ce.witness.contains_key("s"));
        assert_eq!(
            ce.witness["s"],
            CounterexampleValue::String("world".to_string())
        );
    }

    #[test]
    fn parse_counterexample_multiple_vars() {
        let backend = Cvc5Backend::new();
        let model = "(model
  (define-fun a () Int 100)
  (define-fun b () Int 200)
  (define-fun c () Bool false)
)";
        let ce = backend.parse_counterexample(model);
        assert_eq!(ce.witness.len(), 3);
        assert!(ce.witness.contains_key("a"));
        assert!(ce.witness.contains_key("b"));
        assert!(ce.witness.contains_key("c"));
        match &ce.witness["a"] {
            CounterexampleValue::Int { value, .. } => assert_eq!(*value, 100),
            _ => panic!("Expected Int"),
        }
        match &ce.witness["b"] {
            CounterexampleValue::Int { value, .. } => assert_eq!(*value, 200),
            _ => panic!("Expected Int"),
        }
        assert_eq!(ce.witness["c"], CounterexampleValue::Bool(false));
    }

    #[test]
    fn parse_counterexample_set() {
        let backend = Cvc5Backend::new();
        let model = "(model\n  (define-fun s () (Set Int) (singleton 1))\n)";
        let ce = backend.parse_counterexample(model);
        assert!(ce.witness.contains_key("s"));
        match &ce.witness["s"] {
            CounterexampleValue::Unknown(s) => assert!(s.starts_with("Set:")),
            _ => panic!("Expected Unknown(Set:...)"),
        }
    }

    #[test]
    fn parse_counterexample_sequence() {
        let backend = Cvc5Backend::new();
        let model = "(model\n  (define-fun seq () (Seq Int) (seq.unit 5))\n)";
        let ce = backend.parse_counterexample(model);
        assert!(ce.witness.contains_key("seq"));
        match &ce.witness["seq"] {
            CounterexampleValue::Unknown(s) => assert!(s.starts_with("Seq:")),
            _ => panic!("Expected Unknown(Seq:...)"),
        }
    }

    #[test]
    fn parse_counterexample_array() {
        let backend = Cvc5Backend::new();
        let model =
            "(model\n  (define-fun arr () (Array Int Int) ((as const (Array Int Int)) 0))\n)";
        let ce = backend.parse_counterexample(model);
        assert!(ce.witness.contains_key("arr"));
        match &ce.witness["arr"] {
            CounterexampleValue::Unknown(s) => assert!(s.starts_with("Array:")),
            _ => panic!("Expected Unknown(Array:...)"),
        }
    }

    #[test]
    fn parse_counterexample_preserves_raw() {
        let backend = Cvc5Backend::new();
        let model = "(model\n  (define-fun x () Int 5)\n)";
        let ce = backend.parse_counterexample(model);
        assert!(ce.raw.is_some());
        assert!(ce.raw.unwrap().contains("define-fun x"));
    }

    #[test]
    fn parse_counterexample_empty_model() {
        let backend = Cvc5Backend::new();
        let model = "(model\n)";
        let ce = backend.parse_counterexample(model);
        assert!(ce.witness.is_empty());
    }

    #[test]
    fn parse_int_value_positive() {
        assert_eq!(Cvc5Backend::parse_int_value("42"), Some(42));
    }

    #[test]
    fn parse_int_value_negative_parens() {
        assert_eq!(Cvc5Backend::parse_int_value("(- 42)"), Some(-42));
        assert_eq!(Cvc5Backend::parse_int_value("(-42)"), Some(-42));
    }

    #[test]
    fn parse_int_value_zero() {
        assert_eq!(Cvc5Backend::parse_int_value("0"), Some(0));
    }

    #[test]
    fn parse_int_value_invalid() {
        assert_eq!(Cvc5Backend::parse_int_value("abc"), None);
        assert_eq!(Cvc5Backend::parse_int_value(""), None);
    }

    #[test]
    fn parse_real_value_decimal() {
        let val = Cvc5Backend::parse_real_value("2.75").unwrap();
        assert!((val - 2.75).abs() < 0.00001);
    }

    #[test]
    fn parse_real_value_rational() {
        let val = Cvc5Backend::parse_real_value("(/ 1.0 4.0)").unwrap();
        assert!((val - 0.25).abs() < 0.001);
    }

    #[test]
    fn parse_real_value_negative() {
        let val = Cvc5Backend::parse_real_value("(- 3.5)").unwrap();
        assert!((val - (-3.5)).abs() < 0.001);
    }

    #[test]
    fn extract_definitions_single() {
        let model = "(model\n  (define-fun x () Int 5)\n)";
        let defs = Cvc5Backend::extract_definitions(model);
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].0, "x");
        assert_eq!(defs[0].1, "Int");
        assert_eq!(defs[0].2, "5");
    }

    #[test]
    fn extract_definitions_multiple() {
        let model = "(model\n  (define-fun a () Int 1)\n  (define-fun b () Bool true)\n)";
        let defs = Cvc5Backend::extract_definitions(model);
        assert_eq!(defs.len(), 2);
    }

    #[test]
    fn extract_definitions_compound_sort() {
        let model = "(model\n  (define-fun arr () (Array Int Int) 0)\n)";
        let defs = Cvc5Backend::extract_definitions(model);
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].0, "arr");
        assert!(defs[0].1.contains("Array"));
    }

    #[test]
    fn parse_smt_value_bool() {
        assert_eq!(
            Cvc5Backend::parse_smt_value("true", "Bool"),
            CounterexampleValue::Bool(true)
        );
        assert_eq!(
            Cvc5Backend::parse_smt_value("false", "Bool"),
            CounterexampleValue::Bool(false)
        );
    }

    #[test]
    fn parse_smt_value_int() {
        match Cvc5Backend::parse_smt_value("999", "Int") {
            CounterexampleValue::Int { value, .. } => assert_eq!(value, 999),
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn has_structured_data() {
        let backend = Cvc5Backend::new();
        let model = "(model\n  (define-fun x () Int 5)\n)";
        let ce = backend.parse_counterexample(model);
        assert!(ce.has_structured_data());
    }

    #[test]
    fn structured_counterexample_summary() {
        let backend = Cvc5Backend::new();
        let model = "(model\n  (define-fun y () Int 10)\n)";
        let ce = backend.parse_counterexample(model);
        let summary = ce.summary();
        assert!(summary.contains("y = "));
    }
}
