//! Alt-Ergo SMT solver backend
//!
//! Alt-Ergo is an SMT solver dedicated to program verification.
//! It's the main back-end solver of Why3 and Frama-C.
//!
//! Alt-Ergo features:
//! - Native support for polymorphism and records
//! - Arithmetic reasoning with linear and non-linear solvers
//! - Support for bit-vectors, arrays, and algebraic datatypes
//! - Tight integration with Why3 for proof obligations
//!
//! See: <https://alt-ergo.ocamlpro.com/>

// =============================================
// Kani Proofs for Alt-Ergo Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- AltErgoConfig Default Tests ----

    /// Verify AltErgoConfig::default sets correct baseline values
    #[kani::proof]
    fn proof_altergo_config_defaults() {
        let config = AltErgoConfig::default();
        kani::assert(
            config.altergo_path.is_none(),
            "altergo_path should default to None",
        );
        kani::assert(
            config.timeout == Duration::from_secs(60),
            "timeout should default to 60 seconds",
        );
        kani::assert(!config.verbose, "verbose should default to false");
        kani::assert(
            config.input_format == "smtlib2",
            "input_format should default to smtlib2",
        );
        kani::assert(
            config.produce_models,
            "produce_models should default to true",
        );
        kani::assert(config.logic == "ALL", "logic should default to ALL");
    }

    // ---- AltErgoBackend Construction Tests ----

    /// Verify AltErgoBackend::new uses default configuration
    #[kani::proof]
    fn proof_altergo_backend_new_defaults() {
        let backend = AltErgoBackend::new();
        kani::assert(
            backend.config.altergo_path.is_none(),
            "new backend should have no altergo_path",
        );
        kani::assert(
            backend.config.timeout == Duration::from_secs(60),
            "new backend should default timeout to 60 seconds",
        );
        kani::assert(
            backend.config.input_format == "smtlib2",
            "new backend should default to smtlib2 format",
        );
    }

    /// Verify AltErgoBackend::default equals AltErgoBackend::new
    #[kani::proof]
    fn proof_altergo_backend_default_equals_new() {
        let default_backend = AltErgoBackend::default();
        let new_backend = AltErgoBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "default and new should share timeout",
        );
        kani::assert(
            default_backend.config.logic == new_backend.config.logic,
            "default and new should share logic",
        );
    }

    /// Verify AltErgoBackend::with_config preserves custom configuration
    #[kani::proof]
    fn proof_altergo_backend_with_config() {
        let config = AltErgoConfig {
            altergo_path: Some(PathBuf::from("/opt/alt-ergo")),
            timeout: Duration::from_secs(120),
            verbose: true,
            input_format: "native".to_string(),
            produce_models: false,
            logic: "LIA".to_string(),
        };
        let backend = AltErgoBackend::with_config(config);
        kani::assert(
            backend.config.altergo_path == Some(PathBuf::from("/opt/alt-ergo")),
            "with_config should preserve altergo_path",
        );
        kani::assert(
            backend.config.timeout == Duration::from_secs(120),
            "with_config should preserve timeout",
        );
        kani::assert(
            backend.config.verbose,
            "with_config should preserve verbose",
        );
        kani::assert(
            backend.config.input_format == "native",
            "with_config should preserve input_format",
        );
        kani::assert(
            !backend.config.produce_models,
            "with_config should preserve produce_models",
        );
        kani::assert(
            backend.config.logic == "LIA",
            "with_config should preserve logic",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify id() returns BackendId::AltErgo
    #[kani::proof]
    fn proof_altergo_backend_id() {
        let backend = AltErgoBackend::new();
        kani::assert(
            backend.id() == BackendId::AltErgo,
            "AltErgoBackend id should be BackendId::AltErgo",
        );
    }

    /// Verify supports() includes Theorem, Contract, and Invariant
    #[kani::proof]
    fn proof_altergo_backend_supports() {
        let backend = AltErgoBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::Theorem),
            "supports should include Theorem",
        );
        kani::assert(
            supported.contains(&PropertyType::Contract),
            "supports should include Contract",
        );
        kani::assert(
            supported.contains(&PropertyType::Invariant),
            "supports should include Invariant",
        );
    }

    /// Verify supports() returns exactly three property types
    #[kani::proof]
    fn proof_altergo_backend_supports_length() {
        let backend = AltErgoBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.len() == 3,
            "AltErgo should support exactly three property types",
        );
    }

    // ---- Output Parsing Tests ----

    /// Verify parse_output detects unsat as Proven
    #[kani::proof]
    fn proof_parse_output_unsat() {
        let backend = AltErgoBackend::new();
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
        let backend = AltErgoBackend::new();
        let (status, _) = backend.parse_output("sat", "", true);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "sat should produce Disproven status",
        );
    }

    /// Verify parse_output detects unknown
    #[kani::proof]
    fn proof_parse_output_unknown() {
        let backend = AltErgoBackend::new();
        let (status, _) = backend.parse_output("unknown", "", true);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "unknown should produce Unknown status",
        );
    }

    /// Verify parse_output detects Valid in native mode
    #[kani::proof]
    fn proof_parse_output_valid_native() {
        let backend = AltErgoBackend::new();
        let (status, _) = backend.parse_output("Valid (0.01s)", "", true);
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "Valid should produce Proven status",
        );
    }

    /// Verify parse_output detects Invalid in native mode
    #[kani::proof]
    fn proof_parse_output_invalid_native() {
        let backend = AltErgoBackend::new();
        let (status, _) = backend.parse_output("Invalid", "", true);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "Invalid should produce Disproven status",
        );
    }

    /// Verify parse_output detects "I don't know" in native mode
    #[kani::proof]
    fn proof_parse_output_i_dont_know() {
        let backend = AltErgoBackend::new();
        let (status, _) = backend.parse_output("I don't know", "", true);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "I don't know should produce Unknown status",
        );
    }

    /// Verify parse_output detects errors
    #[kani::proof]
    fn proof_parse_output_error() {
        let backend = AltErgoBackend::new();
        let (status, _) = backend.parse_output("", "Error: parse error", false);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Error output should produce Unknown status",
        );
    }

    // ---- Int Value Parsing Tests ----

    /// Verify parse_int_value parses positive integers
    #[kani::proof]
    fn proof_parse_int_positive() {
        let result = AltErgoBackend::parse_int_value("42");
        kani::assert(result == Some(42), "should parse positive integer");
    }

    /// Verify parse_int_value parses zero
    #[kani::proof]
    fn proof_parse_int_zero() {
        let result = AltErgoBackend::parse_int_value("0");
        kani::assert(result == Some(0), "should parse zero");
    }

    /// Verify parse_int_value parses negative with parentheses
    #[kani::proof]
    fn proof_parse_int_negative_parens() {
        let result = AltErgoBackend::parse_int_value("(- 42)");
        kani::assert(result == Some(-42), "should parse negative with space");
    }

    /// Verify parse_int_value parses negative without space
    #[kani::proof]
    fn proof_parse_int_negative_nospace() {
        let result = AltErgoBackend::parse_int_value("(-42)");
        kani::assert(result == Some(-42), "should parse negative without space");
    }

    /// Verify parse_int_value returns None for invalid input
    #[kani::proof]
    fn proof_parse_int_invalid() {
        let result = AltErgoBackend::parse_int_value("abc");
        kani::assert(result.is_none(), "should return None for invalid input");
    }

    // ---- Real Value Parsing Tests ----

    /// Verify parse_real_value parses decimal numbers
    #[kani::proof]
    fn proof_parse_real_decimal() {
        let result = AltErgoBackend::parse_real_value("2.5");
        kani::assert(result.is_some(), "should parse decimal");
        if let Some(val) = result {
            kani::assert((val - 2.5).abs() < 0.0001, "value should be 2.5");
        }
    }

    /// Verify parse_real_value parses rational form
    #[kani::proof]
    fn proof_parse_real_rational() {
        let result = AltErgoBackend::parse_real_value("(/ 1.0 2.0)");
        kani::assert(result.is_some(), "should parse rational");
        if let Some(val) = result {
            kani::assert((val - 0.5).abs() < 0.0001, "value should be 0.5");
        }
    }

    // ---- Native Value Parsing Tests ----

    /// Verify parse_native_value parses booleans
    #[kani::proof]
    fn proof_parse_native_bool_true() {
        let result = AltErgoBackend::parse_native_value("true");
        kani::assert(
            result == CounterexampleValue::Bool(true),
            "should parse true",
        );
    }

    /// Verify parse_native_value parses false
    #[kani::proof]
    fn proof_parse_native_bool_false() {
        let result = AltErgoBackend::parse_native_value("false");
        kani::assert(
            result == CounterexampleValue::Bool(false),
            "should parse false",
        );
    }

    /// Verify parse_native_value parses integers
    #[kani::proof]
    fn proof_parse_native_int() {
        let result = AltErgoBackend::parse_native_value("42");
        match result {
            CounterexampleValue::Int { value, .. } => {
                kani::assert(value == 42, "should parse integer value");
            }
            _ => kani::assert(false, "should be Int variant"),
        }
    }

    /// Verify parse_native_value parses floats
    #[kani::proof]
    fn proof_parse_native_float() {
        let result = AltErgoBackend::parse_native_value("3.14");
        match result {
            CounterexampleValue::Float { value } => {
                kani::assert((value - 3.14).abs() < 0.001, "should parse float value");
            }
            _ => kani::assert(false, "should be Float variant"),
        }
    }

    /// Verify parse_native_value parses quoted strings
    #[kani::proof]
    fn proof_parse_native_string() {
        let result = AltErgoBackend::parse_native_value("\"hello\"");
        kani::assert(
            result == CounterexampleValue::String("hello".to_string()),
            "should parse quoted string",
        );
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

/// Configuration for Alt-Ergo backend
#[derive(Debug, Clone)]
pub struct AltErgoConfig {
    /// Path to Alt-Ergo binary
    pub altergo_path: Option<PathBuf>,
    /// Timeout for solving
    pub timeout: Duration,
    /// Enable verbose output
    pub verbose: bool,
    /// Input format: "native", "smtlib2", "why3"
    pub input_format: String,
    /// Enable model generation
    pub produce_models: bool,
    /// SMT-LIB2 logic (only used when input_format is "smtlib2")
    pub logic: String,
}

impl Default for AltErgoConfig {
    fn default() -> Self {
        Self {
            altergo_path: None,
            timeout: Duration::from_secs(60),
            verbose: false,
            input_format: "smtlib2".to_string(),
            produce_models: true,
            logic: "ALL".to_string(),
        }
    }
}

/// Alt-Ergo SMT solver backend
pub struct AltErgoBackend {
    config: AltErgoConfig,
}

impl Default for AltErgoBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl AltErgoBackend {
    /// Create a new Alt-Ergo backend with default configuration
    pub fn new() -> Self {
        Self {
            config: AltErgoConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: AltErgoConfig) -> Self {
        Self { config }
    }

    async fn detect_altergo(&self) -> Result<PathBuf, String> {
        let altergo_path = self
            .config
            .altergo_path
            .clone()
            .or_else(|| which::which("alt-ergo").ok())
            .ok_or("Alt-Ergo not found. Install via opam install alt-ergo")?;

        let output = Command::new(&altergo_path)
            .arg("--version")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| format!("Failed to execute Alt-Ergo: {}", e))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        if stdout.contains("Alt-Ergo") || output.status.success() {
            debug!("Detected Alt-Ergo: {}", stdout.trim());
            Ok(altergo_path)
        } else {
            Err("Alt-Ergo detection failed".to_string())
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

        // Alt-Ergo output formats:
        // Native: "Valid (X.XXs)" or "Invalid" or "I don't know"
        // SMT-LIB2: "unsat" or "sat" + model or "unknown"
        for line in &lines {
            let trimmed = line.trim();

            // SMT-LIB2 mode results
            if trimmed == "unsat" {
                return (VerificationStatus::Proven, None);
            }
            if trimmed == "sat" {
                let model = Self::extract_model(&lines);
                return (VerificationStatus::Disproven, model);
            }
            if trimmed == "unknown" {
                return (
                    VerificationStatus::Unknown {
                        reason: "Alt-Ergo returned unknown".to_string(),
                    },
                    None,
                );
            }

            // Native mode results
            if trimmed.starts_with("Valid") {
                return (VerificationStatus::Proven, None);
            }
            if trimmed == "Invalid" || trimmed.starts_with("Invalid") {
                // Try to extract counterexample from native format
                let model = Self::extract_native_model(&lines);
                return (VerificationStatus::Disproven, model);
            }
            if trimmed.contains("I don't know") || trimmed.starts_with("Timeout") {
                return (
                    VerificationStatus::Unknown {
                        reason: "Alt-Ergo could not determine validity".to_string(),
                    },
                    None,
                );
            }
        }

        // Check for errors
        if !success || combined.contains("Error") || combined.contains("error:") {
            return (
                VerificationStatus::Unknown {
                    reason: format!(
                        "Alt-Ergo error: {}",
                        combined.lines().take(3).collect::<Vec<_>>().join("; ")
                    ),
                },
                None,
            );
        }

        (
            VerificationStatus::Unknown {
                reason: "Could not parse Alt-Ergo output".to_string(),
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

    /// Extract counterexample from Alt-Ergo's native output format
    fn extract_native_model(lines: &[&str]) -> Option<String> {
        let mut model_lines = Vec::new();
        let mut in_model = false;

        for line in lines {
            let trimmed = line.trim();
            // Alt-Ergo native format shows counterexample after "Invalid"
            if trimmed == "Invalid" || trimmed.starts_with("Invalid") {
                in_model = true;
                continue;
            }
            if in_model && !trimmed.is_empty() {
                // Native format: "x = value"
                if trimmed.contains('=') {
                    model_lines.push(trimmed);
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

        // Check if it's SMT-LIB2 format (has define-fun)
        if model_str.contains("define-fun") {
            let definitions = Self::extract_definitions(model_str);
            for (name, sort, value) in definitions {
                let parsed_value = Self::parse_smt_value(&value, &sort);
                ce.witness.insert(name, parsed_value);
            }
        } else {
            // Native format: "x = value" on each line
            for line in model_str.lines() {
                let trimmed = line.trim();
                if let Some((name, value)) = trimmed.split_once('=') {
                    let name = name.trim().to_string();
                    let value = value.trim();
                    let parsed_value = Self::parse_native_value(value);
                    ce.witness.insert(name, parsed_value);
                }
            }
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

        while pos < chars.len() && chars[pos].is_whitespace() {
            pos += 1;
        }

        let name_start = pos;
        while pos < chars.len() && !chars[pos].is_whitespace() && chars[pos] != '(' {
            pos += 1;
        }
        let name: String = chars[name_start..pos].iter().collect();

        while pos < chars.len() && chars[pos].is_whitespace() {
            pos += 1;
        }

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

        while pos < chars.len() && chars[pos].is_whitespace() {
            pos += 1;
        }

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

        while pos < chars.len() && chars[pos].is_whitespace() {
            pos += 1;
        }

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

    /// Parse a value from Alt-Ergo's native format
    fn parse_native_value(value: &str) -> CounterexampleValue {
        let trimmed = value.trim();

        if trimmed == "true" {
            return CounterexampleValue::Bool(true);
        }
        if trimmed == "false" {
            return CounterexampleValue::Bool(false);
        }

        // Try integer
        if let Ok(n) = trimmed.parse::<i128>() {
            return CounterexampleValue::Int {
                value: n,
                type_hint: None,
            };
        }

        // Try float
        if let Ok(f) = trimmed.parse::<f64>() {
            return CounterexampleValue::Float { value: f };
        }

        // Quoted string
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
impl VerificationBackend for AltErgoBackend {
    fn id(&self) -> BackendId {
        BackendId::AltErgo
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![
            PropertyType::Theorem,
            PropertyType::Contract,
            PropertyType::Invariant,
        ]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let altergo_path = self
            .detect_altergo()
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

        // Build command
        let mut cmd = Command::new(&altergo_path);

        // Set input format
        cmd.arg(format!("--input={}", self.config.input_format));

        // Set timeout
        cmd.arg(format!("--timelimit={}", self.config.timeout.as_secs()));

        // Enable verbose output if requested
        if self.config.verbose {
            cmd.arg("--verbose");
        }

        // Enable model generation
        if self.config.produce_models {
            cmd.arg("--produce-models=true");
        }

        // Add the input file
        cmd.arg(&smt_path);

        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

        let result =
            tokio::time::timeout(self.config.timeout + Duration::from_secs(5), cmd.output()).await;

        let duration = start.elapsed();

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                debug!("Alt-Ergo stdout: {}", stdout);
                debug!("Alt-Ergo stderr: {}", stderr);

                let (status, counterexample_str) =
                    self.parse_output(&stdout, &stderr, output.status.success());

                let counterexample = counterexample_str.map(|m| self.parse_counterexample(&m));

                let diagnostics: Vec<String> = stderr
                    .lines()
                    .filter(|l| {
                        l.contains("Warning") || l.contains("Error") || l.contains("warning")
                    })
                    .map(String::from)
                    .collect();

                let proof = if matches!(status, VerificationStatus::Proven) {
                    Some("Verified by Alt-Ergo".to_string())
                } else {
                    None
                };

                Ok(BackendResult {
                    backend: BackendId::AltErgo,
                    status,
                    proof,
                    counterexample,
                    diagnostics,
                    time_taken: duration,
                })
            }
            Ok(Err(e)) => Err(BackendError::VerificationFailed(format!(
                "Failed to execute Alt-Ergo: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match self.detect_altergo().await {
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
        let config = AltErgoConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.input_format, "smtlib2");
        assert!(config.produce_models);
        assert_eq!(config.logic, "ALL");
    }

    #[test]
    fn backend_id() {
        let backend = AltErgoBackend::new();
        assert_eq!(backend.id(), BackendId::AltErgo);
    }

    #[test]
    fn parse_valid_native() {
        let backend = AltErgoBackend::new();
        let (status, _) = backend.parse_output("Valid (0.01s)", "", true);
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn parse_invalid_native() {
        let backend = AltErgoBackend::new();
        let (status, _) = backend.parse_output("Invalid", "", true);
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[test]
    fn parse_unsat_smtlib() {
        let backend = AltErgoBackend::new();
        let (status, model) = backend.parse_output("unsat", "", true);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(model.is_none());
    }

    #[test]
    fn parse_sat_smtlib() {
        let backend = AltErgoBackend::new();
        let (status, _) = backend.parse_output("sat", "", true);
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[test]
    fn parse_unknown_smtlib() {
        let backend = AltErgoBackend::new();
        let (status, _) = backend.parse_output("unknown", "", true);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn parse_i_dont_know_native() {
        let backend = AltErgoBackend::new();
        let (status, _) = backend.parse_output("I don't know", "", true);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn parse_error() {
        let backend = AltErgoBackend::new();
        let (status, _) = backend.parse_output("", "Error: syntax error", false);
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn extract_model_basic() {
        let lines = vec!["sat", "(model", "  (define-fun x () Int 5)", ")"];
        let model = AltErgoBackend::extract_model(&lines);
        assert!(model.is_some());
        let m = model.unwrap();
        assert!(m.contains("model"));
        assert!(m.contains("define-fun"));
    }

    // Counterexample parsing tests
    #[test]
    fn parse_counterexample_int_smtlib() {
        let backend = AltErgoBackend::new();
        let model = "(model\n  (define-fun x () Int 42)\n)";
        let ce = backend.parse_counterexample(model);
        assert!(ce.witness.contains_key("x"));
        match &ce.witness["x"] {
            CounterexampleValue::Int { value, .. } => assert_eq!(*value, 42),
            _ => panic!("Expected Int value"),
        }
    }

    #[test]
    fn parse_counterexample_native_format() {
        let backend = AltErgoBackend::new();
        let model = "x = 42\ny = true\nz = 2.75";
        let ce = backend.parse_counterexample(model);
        assert_eq!(ce.witness.len(), 3);
        assert!(ce.witness.contains_key("x"));
        assert!(ce.witness.contains_key("y"));
        assert!(ce.witness.contains_key("z"));

        match &ce.witness["x"] {
            CounterexampleValue::Int { value, .. } => assert_eq!(*value, 42),
            _ => panic!("Expected Int value"),
        }
        assert_eq!(ce.witness["y"], CounterexampleValue::Bool(true));
        match &ce.witness["z"] {
            CounterexampleValue::Float { value } => assert!((value - 2.75).abs() < 0.001),
            _ => panic!("Expected Float value"),
        }
    }

    #[test]
    fn parse_counterexample_negative_int() {
        let backend = AltErgoBackend::new();
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
        let backend = AltErgoBackend::new();
        let model = "(model\n  (define-fun b () Bool true)\n)";
        let ce = backend.parse_counterexample(model);
        assert!(ce.witness.contains_key("b"));
        assert_eq!(ce.witness["b"], CounterexampleValue::Bool(true));
    }

    #[test]
    fn parse_counterexample_real() {
        let backend = AltErgoBackend::new();
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
        let backend = AltErgoBackend::new();
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
        let backend = AltErgoBackend::new();
        let model = "(model\n  (define-fun x () Int 5)\n)";
        let ce = backend.parse_counterexample(model);
        assert!(ce.raw.is_some());
        assert!(ce.raw.unwrap().contains("define-fun x"));
    }

    #[test]
    fn parse_int_value_positive() {
        assert_eq!(AltErgoBackend::parse_int_value("42"), Some(42));
    }

    #[test]
    fn parse_int_value_negative_parens() {
        assert_eq!(AltErgoBackend::parse_int_value("(- 42)"), Some(-42));
        assert_eq!(AltErgoBackend::parse_int_value("(-42)"), Some(-42));
    }

    #[test]
    fn parse_int_value_zero() {
        assert_eq!(AltErgoBackend::parse_int_value("0"), Some(0));
    }

    #[test]
    fn parse_int_value_invalid() {
        assert_eq!(AltErgoBackend::parse_int_value("abc"), None);
    }

    #[test]
    fn parse_real_value_decimal() {
        let val = AltErgoBackend::parse_real_value("2.75").unwrap();
        assert!((val - 2.75).abs() < 0.00001);
    }

    #[test]
    fn parse_real_value_rational() {
        let val = AltErgoBackend::parse_real_value("(/ 1.0 3.0)").unwrap();
        assert!((val - 0.33333).abs() < 0.001);
    }

    #[test]
    fn parse_native_value_int() {
        match AltErgoBackend::parse_native_value("42") {
            CounterexampleValue::Int { value, .. } => assert_eq!(value, 42),
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn parse_native_value_bool() {
        assert_eq!(
            AltErgoBackend::parse_native_value("true"),
            CounterexampleValue::Bool(true)
        );
    }

    #[test]
    fn parse_native_value_float() {
        match AltErgoBackend::parse_native_value("2.75") {
            CounterexampleValue::Float { value } => assert!((value - 2.75).abs() < 0.001),
            _ => panic!("Expected Float"),
        }
    }

    #[test]
    fn has_structured_data() {
        let backend = AltErgoBackend::new();
        let model = "(model\n  (define-fun x () Int 5)\n)";
        let ce = backend.parse_counterexample(model);
        assert!(ce.has_structured_data());
    }
}
