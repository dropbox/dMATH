//! CryptoVerif protocol verification backend
//!
//! CryptoVerif is a symbolic analyzer for cryptographic protocols.
//! It proves computational security properties in the computational model
//! (as opposed to the Dolev-Yao model).
//!
//! See: <https://cryptoverif.inria.fr/>
//!
//! # Features
//!
//! - **Computational model**: Security against probabilistic polynomial-time attackers
//! - **Automatic proofs**: Game-based transformations applied automatically
//! - **Interactive mode**: Step-by-step proof guidance
//! - **Secrecy and authentication**: Prove key secrecy, correspondence assertions
//!
//! # Requirements
//!
//! Download CryptoVerif from the official website:
//! ```bash
//! # Download from https://cryptoverif.inria.fr/
//! # Extract and add to PATH
//! ```

// =============================================
// Kani Proofs for CryptoVerif Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- CryptoVerifMode Default Tests ----

    /// Verify CryptoVerifMode::default is Auto
    #[kani::proof]
    fn proof_cryptoverif_mode_default_is_auto() {
        let mode = CryptoVerifMode::default();
        kani::assert(mode == CryptoVerifMode::Auto, "Default mode should be Auto");
    }

    // ---- CryptoVerifConfig Default Tests ----

    /// Verify CryptoVerifConfig::default timeout is 300 seconds
    #[kani::proof]
    fn proof_cryptoverif_config_default_timeout() {
        let config = CryptoVerifConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "Default timeout should be 300 seconds",
        );
    }

    /// Verify CryptoVerifConfig::default cryptoverif_path is None
    #[kani::proof]
    fn proof_cryptoverif_config_default_path_none() {
        let config = CryptoVerifConfig::default();
        kani::assert(
            config.cryptoverif_path.is_none(),
            "Default cryptoverif_path should be None",
        );
    }

    /// Verify CryptoVerifConfig::default mode is Auto
    #[kani::proof]
    fn proof_cryptoverif_config_default_mode() {
        let config = CryptoVerifConfig::default();
        kani::assert(
            config.mode == CryptoVerifMode::Auto,
            "Default mode should be Auto",
        );
    }

    /// Verify CryptoVerifConfig::default lib_dir is None
    #[kani::proof]
    fn proof_cryptoverif_config_default_lib_dir_none() {
        let config = CryptoVerifConfig::default();
        kani::assert(config.lib_dir.is_none(), "Default lib_dir should be None");
    }

    /// Verify CryptoVerifConfig::default verbose is false
    #[kani::proof]
    fn proof_cryptoverif_config_default_verbose_false() {
        let config = CryptoVerifConfig::default();
        kani::assert(!config.verbose, "Default verbose should be false");
    }

    /// Verify CryptoVerifConfig::default extra_args is empty
    #[kani::proof]
    fn proof_cryptoverif_config_default_extra_args_empty() {
        let config = CryptoVerifConfig::default();
        kani::assert(
            config.extra_args.is_empty(),
            "Default extra_args should be empty",
        );
    }

    // ---- CryptoVerifBackend Construction Tests ----

    /// Verify CryptoVerifBackend::new uses default config timeout
    #[kani::proof]
    fn proof_cryptoverif_backend_new_default_timeout() {
        let backend = CryptoVerifBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(300),
            "New backend should use default timeout",
        );
    }

    /// Verify CryptoVerifBackend::default equals CryptoVerifBackend::new
    #[kani::proof]
    fn proof_cryptoverif_backend_default_equals_new() {
        let default_backend = CryptoVerifBackend::default();
        let new_backend = CryptoVerifBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "Default and new should have same timeout",
        );
    }

    /// Verify CryptoVerifBackend::with_config preserves custom timeout
    #[kani::proof]
    fn proof_cryptoverif_backend_with_config_timeout() {
        let config = CryptoVerifConfig {
            cryptoverif_path: None,
            timeout: Duration::from_secs(600),
            mode: CryptoVerifMode::Auto,
            lib_dir: None,
            verbose: false,
            extra_args: vec![],
        };
        let backend = CryptoVerifBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "with_config should preserve timeout",
        );
    }

    /// Verify CryptoVerifBackend::with_config preserves Interactive mode
    #[kani::proof]
    fn proof_cryptoverif_backend_with_config_interactive() {
        let config = CryptoVerifConfig {
            cryptoverif_path: None,
            timeout: Duration::from_secs(300),
            mode: CryptoVerifMode::Interactive,
            lib_dir: None,
            verbose: false,
            extra_args: vec![],
        };
        let backend = CryptoVerifBackend::with_config(config);
        kani::assert(
            backend.config.mode == CryptoVerifMode::Interactive,
            "with_config should preserve Interactive mode",
        );
    }

    /// Verify CryptoVerifBackend::with_config preserves TypeCheck mode
    #[kani::proof]
    fn proof_cryptoverif_backend_with_config_typecheck() {
        let config = CryptoVerifConfig {
            cryptoverif_path: None,
            timeout: Duration::from_secs(300),
            mode: CryptoVerifMode::TypeCheck,
            lib_dir: None,
            verbose: false,
            extra_args: vec![],
        };
        let backend = CryptoVerifBackend::with_config(config);
        kani::assert(
            backend.config.mode == CryptoVerifMode::TypeCheck,
            "with_config should preserve TypeCheck mode",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify CryptoVerifBackend::id returns CryptoVerif
    #[kani::proof]
    fn proof_cryptoverif_backend_id() {
        let backend = CryptoVerifBackend::new();
        kani::assert(
            backend.id() == BackendId::CryptoVerif,
            "Backend id should be CryptoVerif",
        );
    }

    /// Verify CryptoVerifBackend::supports includes SecurityProtocol
    #[kani::proof]
    fn proof_cryptoverif_backend_supports_security_protocol() {
        let backend = CryptoVerifBackend::new();
        let supported = backend.supports();
        let has_security = supported
            .iter()
            .any(|p| *p == PropertyType::SecurityProtocol);
        kani::assert(has_security, "Should support SecurityProtocol property");
    }

    /// Verify CryptoVerifBackend::supports returns exactly 1 property
    #[kani::proof]
    fn proof_cryptoverif_backend_supports_length() {
        let backend = CryptoVerifBackend::new();
        let supported = backend.supports();
        kani::assert(supported.len() == 1, "Should support exactly 1 property");
    }

    // ---- sanitize_name Tests ----

    /// Verify sanitize_name replaces hyphens with underscores
    #[kani::proof]
    fn proof_sanitize_name_hyphen() {
        let result = CryptoVerifBackend::sanitize_name("my-func");
        kani::assert(result == "my_func", "Should replace hyphen with underscore");
    }

    /// Verify sanitize_name converts to lowercase
    #[kani::proof]
    fn proof_sanitize_name_lowercase() {
        let result = CryptoVerifBackend::sanitize_name("MyFunc");
        kani::assert(result == "myfunc", "Should convert to lowercase");
    }

    /// Verify sanitize_name replaces spaces
    #[kani::proof]
    fn proof_sanitize_name_spaces() {
        let result = CryptoVerifBackend::sanitize_name("my func");
        kani::assert(result == "my_func", "Should replace space with underscore");
    }

    /// Verify sanitize_name handles empty string
    #[kani::proof]
    fn proof_sanitize_name_empty() {
        let result = CryptoVerifBackend::sanitize_name("");
        kani::assert(result.is_empty(), "Empty string should remain empty");
    }

    /// Verify sanitize_name replaces colons
    #[kani::proof]
    fn proof_sanitize_name_colons() {
        let result = CryptoVerifBackend::sanitize_name("foo:bar");
        kani::assert(result == "foo_bar", "Should replace colon with underscore");
    }

    // ---- parse_cv_value Tests ----

    /// Verify parse_cv_value returns Bool(true) for "true"
    #[kani::proof]
    fn proof_parse_cv_value_true() {
        let value = CryptoVerifBackend::parse_cv_value("true");
        kani::assert(
            matches!(value, CounterexampleValue::Bool(true)),
            "Should parse true as Bool(true)",
        );
    }

    /// Verify parse_cv_value returns Bool(false) for "false"
    #[kani::proof]
    fn proof_parse_cv_value_false() {
        let value = CryptoVerifBackend::parse_cv_value("false");
        kani::assert(
            matches!(value, CounterexampleValue::Bool(false)),
            "Should parse false as Bool(false)",
        );
    }

    /// Verify parse_cv_value returns Int for "42"
    #[kani::proof]
    fn proof_parse_cv_value_int() {
        let value = CryptoVerifBackend::parse_cv_value("42");
        kani::assert(
            matches!(value, CounterexampleValue::Int { value: 42, .. }),
            "Should parse 42 as Int",
        );
    }

    /// Verify parse_cv_value returns String for non-bool non-int
    #[kani::proof]
    fn proof_parse_cv_value_string() {
        let value = CryptoVerifBackend::parse_cv_value("hello");
        kani::assert(
            matches!(value, CounterexampleValue::String(_)),
            "Should parse hello as String",
        );
    }

    // ---- parse_output Tests ----

    /// Verify parse_output returns Proven for "RESULT secret...is true"
    #[kani::proof]
    fn proof_parse_output_proven() {
        let backend = CryptoVerifBackend::new();
        let (status, _) = backend.parse_output("RESULT secret secret_key is true.", "", true);
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "Should return Proven for is true",
        );
    }

    /// Verify parse_output returns Disproven for "CANNOT prove"
    #[kani::proof]
    fn proof_parse_output_disproven() {
        let backend = CryptoVerifBackend::new();
        let (status, _) = backend.parse_output("CANNOT prove secret secret_key", "", false);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "Should return Disproven for CANNOT prove",
        );
    }

    /// Verify parse_output returns Unknown for timeout
    #[kani::proof]
    fn proof_parse_output_timeout() {
        let backend = CryptoVerifBackend::new();
        let (status, _) = backend.parse_output("timeout reached", "", false);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Should return Unknown for timeout",
        );
    }

    /// Verify parse_output returns Unknown for syntax error
    #[kani::proof]
    fn proof_parse_output_syntax_error() {
        let backend = CryptoVerifBackend::new();
        let (status, _) = backend.parse_output("Syntax error at line 5", "", false);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Should return Unknown for syntax error",
        );
    }
}

use crate::counterexample::{
    CounterexampleValue, FailedCheck, SourceLocation, StructuredCounterexample,
};
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

/// Proof mode for CryptoVerif
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CryptoVerifMode {
    /// Automatic proof search
    #[default]
    Auto,
    /// Interactive mode for manual guidance
    Interactive,
    /// Just type-check without proving
    TypeCheck,
}

/// Configuration for CryptoVerif backend
#[derive(Debug, Clone)]
pub struct CryptoVerifConfig {
    /// Path to cryptoverif binary
    pub cryptoverif_path: Option<PathBuf>,
    /// Timeout for verification
    pub timeout: Duration,
    /// Proof mode
    pub mode: CryptoVerifMode,
    /// Library directory (contains standard.cvl, etc.)
    pub lib_dir: Option<PathBuf>,
    /// Enable proof details in output
    pub verbose: bool,
    /// Additional CryptoVerif options
    pub extra_args: Vec<String>,
}

impl Default for CryptoVerifConfig {
    fn default() -> Self {
        Self {
            cryptoverif_path: None,
            timeout: Duration::from_secs(300),
            mode: CryptoVerifMode::default(),
            lib_dir: None,
            verbose: false,
            extra_args: vec![],
        }
    }
}

/// CryptoVerif protocol verification backend
pub struct CryptoVerifBackend {
    config: CryptoVerifConfig,
}

impl Default for CryptoVerifBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CryptoVerifBackend {
    /// Create a new CryptoVerif backend with default configuration
    pub fn new() -> Self {
        Self {
            config: CryptoVerifConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: CryptoVerifConfig) -> Self {
        Self { config }
    }

    async fn detect(&self) -> Result<PathBuf, String> {
        // Check configured path first
        if let Some(ref path) = self.config.cryptoverif_path {
            if path.exists() {
                return Ok(path.clone());
            }
        }

        // Try common binary names
        for name in ["cryptoverif", "cv", "cryptoverif.opt"] {
            if let Ok(path) = which::which(name) {
                // Verify it works
                let output = Command::new(&path)
                    .arg("-help")
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .output()
                    .await;

                if let Ok(out) = output {
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    if stdout.contains("CryptoVerif")
                        || stderr.contains("CryptoVerif")
                        || stdout.contains("cryptoverif")
                        || stderr.contains("cryptoverif")
                    {
                        debug!("Detected CryptoVerif at: {:?}", path);
                        return Ok(path);
                    }
                }
            }
        }

        // Check CRYPTOVERIF_HOME environment variable
        if let Ok(cv_home) = std::env::var("CRYPTOVERIF_HOME") {
            let cv_bin = PathBuf::from(&cv_home).join("cryptoverif");
            if cv_bin.exists() {
                return Ok(cv_bin);
            }
            let cv_bin = PathBuf::from(&cv_home).join("bin").join("cryptoverif");
            if cv_bin.exists() {
                return Ok(cv_bin);
            }
        }

        Err("CryptoVerif not found. Download from: https://cryptoverif.inria.fr/".to_string())
    }

    /// Generate CryptoVerif code from USL spec
    fn generate_cv_code(&self, spec: &TypedSpec) -> String {
        let mut code = String::new();
        code.push_str("(* Generated by DashProve *)\n\n");

        // Type declarations
        code.push_str("(* Type declarations *)\n");
        code.push_str("type key [fixed].\n");
        code.push_str("type nonce [fixed].\n");

        // Custom types
        for type_def in &spec.spec.types {
            let safe_name = Self::sanitize_name(&type_def.name);
            code.push_str(&format!("type {} [fixed].\n", safe_name));
        }
        code.push('\n');

        // Function declarations
        code.push_str("(* Function declarations *)\n");
        code.push_str("fun enc(bitstring, key): bitstring.\n");
        code.push_str("fun dec(bitstring, key): bitstring.\n\n");

        // Equations
        code.push_str("(* Equations *)\n");
        code.push_str("equation forall m: bitstring, k: key; dec(enc(m, k), k) = m.\n\n");

        // Secrecy assumptions
        code.push_str("(* Secrecy assumptions *)\n");
        code.push_str("free c: channel.\n");
        code.push_str("free secret_key: key [private].\n\n");

        // Queries for properties
        code.push_str("(* Queries *)\n");
        for (i, prop) in spec.spec.properties.iter().enumerate() {
            let prop_name = prop.name();
            let safe_name = Self::sanitize_name(&prop_name);
            let query_name = if safe_name.is_empty() {
                format!("query_{}", i)
            } else {
                format!("query_{}", safe_name)
            };

            code.push_str(&format!("(* Property: {} *)\n", prop_name));
            code.push_str(&format!(
                "query secret secret_key. (* {} *)\n\n",
                query_name
            ));
        }

        // If no properties, add a trivial query
        if spec.spec.properties.is_empty() {
            code.push_str("query secret secret_key.\n\n");
        }

        // Main process
        code.push_str("(* Protocol *)\n");
        code.push_str("let initiator =\n");
        code.push_str("  new n: nonce;\n");
        code.push_str("  out(c, enc(n, secret_key)).\n\n");

        code.push_str("process\n");
        code.push_str("  initiator\n");

        code
    }

    /// Sanitize a name for use in CryptoVerif
    fn sanitize_name(name: &str) -> String {
        name.replace([' ', '-', ':', '/', '\\', '.', '(', ')', '[', ']'], "_")
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '_')
            .collect::<String>()
            .to_lowercase()
    }

    fn parse_output(
        &self,
        stdout: &str,
        stderr: &str,
        success: bool,
    ) -> (VerificationStatus, Vec<String>) {
        let combined = format!("{}\n{}", stdout, stderr);
        let mut diagnostics = Vec::new();

        // Parse CryptoVerif output
        for line in combined.lines() {
            let trimmed = line.trim();

            // Check for successful verification
            if trimmed.contains("RESULT") || trimmed.contains("proved") || trimmed.contains("true")
            {
                diagnostics.push(format!("✓ {}", trimmed));
            }

            // Check for failures
            if trimmed.contains("CANNOT") || trimmed.contains("false") || trimmed.contains("Error")
            {
                diagnostics.push(format!("✗ {}", trimmed));
            }

            // Capture game transformations
            if trimmed.contains("Doing")
                || trimmed.contains("Game")
                || trimmed.contains("transformation")
            {
                diagnostics.push(trimmed.to_string());
            }

            // Capture query results
            if trimmed.contains("Query") || trimmed.contains("query") {
                diagnostics.push(trimmed.to_string());
            }
        }

        // Check for explicit success
        if combined.contains("RESULT secret")
            || combined.contains("is true")
            || combined.contains("Query proved")
        {
            return (VerificationStatus::Proven, diagnostics);
        }

        // Check for explicit failure
        if combined.contains("CANNOT prove")
            || combined.contains("is false")
            || combined.contains("Query could not be proved")
        {
            return (VerificationStatus::Disproven, diagnostics);
        }

        // Check for successful run without explicit failure
        if success && !combined.contains("Error") && !combined.contains("CANNOT") {
            return (VerificationStatus::Proven, diagnostics);
        }

        // Check for timeout
        if combined.contains("timeout") || combined.contains("Timeout") {
            return (
                VerificationStatus::Unknown {
                    reason: "Verification timed out".to_string(),
                },
                diagnostics,
            );
        }

        // Check for parse errors
        if combined.contains("Syntax error") || combined.contains("Parse error") {
            return (
                VerificationStatus::Unknown {
                    reason: "Parse error in CryptoVerif specification".to_string(),
                },
                diagnostics,
            );
        }

        // Check exit status
        if !success {
            let error_lines: Vec<_> = combined
                .lines()
                .filter(|l| l.contains("Error") || l.contains("error"))
                .take(3)
                .collect();

            if !error_lines.is_empty() {
                return (
                    VerificationStatus::Unknown {
                        reason: format!("CryptoVerif error: {}", error_lines.join("; ")),
                    },
                    diagnostics,
                );
            }

            return (
                VerificationStatus::Unknown {
                    reason: "CryptoVerif returned non-zero exit code".to_string(),
                },
                diagnostics,
            );
        }

        (
            VerificationStatus::Unknown {
                reason: "Could not parse CryptoVerif output".to_string(),
            },
            diagnostics,
        )
    }

    /// Parse counterexample from CryptoVerif output
    fn parse_counterexample(stdout: &str, stderr: &str) -> StructuredCounterexample {
        let mut ce = StructuredCounterexample::new();
        let combined = format!("{}\n{}", stdout, stderr);
        ce.raw = Some(combined.clone());

        // Extract failed checks
        ce.failed_checks = Self::extract_failed_checks(&combined);

        // Extract attack trace values
        ce.witness = Self::extract_attack_values(&combined);

        ce
    }

    /// Extract failed checks from CryptoVerif output
    fn extract_failed_checks(output: &str) -> Vec<FailedCheck> {
        let mut checks = Vec::new();

        for line in output.lines() {
            let trimmed = line.trim();

            if trimmed.contains("CANNOT")
                || trimmed.contains("false")
                || (trimmed.contains("Query") && trimmed.contains("not"))
            {
                let check_type = if trimmed.contains("secret") {
                    "cv_secrecy"
                } else if trimmed.contains("correspondence") {
                    "cv_correspondence"
                } else if trimmed.contains("event") {
                    "cv_event"
                } else {
                    "cv_query"
                };

                let (location, description) = Self::parse_error_location(trimmed);

                checks.push(FailedCheck {
                    check_id: check_type.to_string(),
                    description,
                    location,
                    function: None,
                });
            }
        }

        checks
    }

    /// Parse error location from CryptoVerif error line
    fn parse_error_location(line: &str) -> (Option<SourceLocation>, String) {
        // CryptoVerif format: "File "file.cv", line N, character C: message"
        if line.contains("File \"") {
            if let Some(start) = line.find("File \"") {
                let rest = &line[start + 6..];
                if let Some(end) = rest.find('"') {
                    let file = rest[..end].to_string();
                    let after_file = &rest[end + 1..];

                    if let Some(line_start) = after_file.find("line ") {
                        let line_rest = &after_file[line_start + 5..];
                        let num_str: String = line_rest
                            .chars()
                            .take_while(|c| c.is_ascii_digit())
                            .collect();
                        if let Ok(line_num) = num_str.parse::<u32>() {
                            // Find the message after the location info
                            let message = if let Some(colon) = after_file.rfind(':') {
                                after_file[colon + 1..].trim().to_string()
                            } else {
                                line.to_string()
                            };

                            return (
                                Some(SourceLocation {
                                    file,
                                    line: line_num,
                                    column: None,
                                }),
                                message,
                            );
                        }
                    }
                }
            }
        }

        (None, line.to_string())
    }

    /// Extract attack/witness values from CryptoVerif output
    fn extract_attack_values(output: &str) -> HashMap<String, CounterexampleValue> {
        let mut values = HashMap::new();
        let mut in_attack = false;

        for line in output.lines() {
            let trimmed = line.trim();

            // Look for attack section
            if trimmed.contains("Attack") || trimmed.contains("attack") {
                in_attack = true;
                continue;
            }

            if in_attack {
                // End of attack section
                if trimmed.is_empty() || trimmed.starts_with("---") || trimmed.contains("Game") {
                    in_attack = false;
                    continue;
                }

                // Parse "var = value" patterns
                if let Some(eq_pos) = trimmed.find(" = ") {
                    let var_name = trimmed[..eq_pos].trim().to_string();
                    let value_str = trimmed[eq_pos + 3..].trim();
                    values.insert(var_name, Self::parse_cv_value(value_str));
                }
            }
        }

        values
    }

    /// Parse a CryptoVerif value string
    fn parse_cv_value(value_str: &str) -> CounterexampleValue {
        let trimmed = value_str.trim();

        // Boolean
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
                type_hint: None,
            };
        }

        // Default to string
        CounterexampleValue::String(trimmed.to_string())
    }
}

#[async_trait]
impl VerificationBackend for CryptoVerifBackend {
    fn id(&self) -> BackendId {
        BackendId::CryptoVerif
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::SecurityProtocol]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let cv_path = self.detect().await.map_err(BackendError::Unavailable)?;

        // Create temp directory
        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp directory: {}", e))
        })?;

        let cv_file = temp_dir.path().join("spec.cv");
        let cv_code = self.generate_cv_code(spec);

        debug!("Generated CryptoVerif code:\n{}", cv_code);

        tokio::fs::write(&cv_file, &cv_code).await.map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write CryptoVerif file: {}", e))
        })?;

        // Build command
        let mut cmd = Command::new(&cv_path);
        cmd.arg(&cv_file)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .current_dir(temp_dir.path());

        // Add mode-specific options
        match self.config.mode {
            CryptoVerifMode::Auto => {
                // Default mode
            }
            CryptoVerifMode::Interactive => {
                cmd.arg("-interact");
            }
            CryptoVerifMode::TypeCheck => {
                cmd.arg("-typecheck");
            }
        }

        // Library directory
        if let Some(ref lib_dir) = self.config.lib_dir {
            cmd.arg("-lib").arg(lib_dir);
        }

        // Verbose output
        if self.config.verbose {
            cmd.arg("-v");
        }

        // Extra args
        for arg in &self.config.extra_args {
            cmd.arg(arg);
        }

        let output = tokio::time::timeout(self.config.timeout, cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| {
                BackendError::VerificationFailed(format!("Failed to run cryptoverif: {}", e))
            })?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        debug!("CryptoVerif stdout: {}", stdout);
        debug!("CryptoVerif stderr: {}", stderr);

        let (status, diagnostics) = self.parse_output(&stdout, &stderr, output.status.success());

        // Generate counterexample for failures
        let counterexample = if matches!(status, VerificationStatus::Disproven) {
            Some(Self::parse_counterexample(&stdout, &stderr))
        } else {
            None
        };

        Ok(BackendResult {
            backend: BackendId::CryptoVerif,
            status,
            proof: None,
            counterexample,
            diagnostics,
            time_taken: start.elapsed(),
        })
    }

    async fn health_check(&self) -> HealthStatus {
        match self.detect().await {
            Ok(_) => HealthStatus::Healthy,
            Err(r) => HealthStatus::Unavailable { reason: r },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_id() {
        assert_eq!(CryptoVerifBackend::new().id(), BackendId::CryptoVerif);
    }

    #[test]
    fn default_config() {
        let config = CryptoVerifConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert_eq!(config.mode, CryptoVerifMode::Auto);
        assert!(!config.verbose);
    }

    #[test]
    fn supports_security() {
        let backend = CryptoVerifBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::SecurityProtocol));
    }

    #[test]
    fn sanitize_name() {
        assert_eq!(
            CryptoVerifBackend::sanitize_name("Hello-World"),
            "hello_world"
        );
        assert_eq!(CryptoVerifBackend::sanitize_name("test:prop"), "test_prop");
    }

    #[test]
    fn parse_proved_output() {
        let backend = CryptoVerifBackend::new();
        let stdout = "RESULT secret secret_key is true.\nQuery proved.";
        let (status, diag) = backend.parse_output(stdout, "", true);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(!diag.is_empty());
    }

    #[test]
    fn parse_fail_output() {
        let backend = CryptoVerifBackend::new();
        let stdout = "CANNOT prove secret secret_key\nQuery could not be proved";
        let (status, _diag) = backend.parse_output(stdout, "", false);
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[test]
    fn parse_error_location() {
        let line = "File \"spec.cv\", line 10, character 5: Error here";
        let (loc, desc) = CryptoVerifBackend::parse_error_location(line);
        assert!(loc.is_some());
        let loc = loc.unwrap();
        assert_eq!(loc.file, "spec.cv");
        assert_eq!(loc.line, 10);
        assert!(desc.contains("Error"));
    }

    #[test]
    fn extract_failed_checks() {
        let output = "CANNOT prove secret key\ncorrespondence is false\nQuery not proved";
        let checks = CryptoVerifBackend::extract_failed_checks(output);
        assert_eq!(checks.len(), 3);
        assert_eq!(checks[0].check_id, "cv_secrecy");
        assert_eq!(checks[1].check_id, "cv_correspondence");
    }

    #[test]
    fn parse_cv_values() {
        assert!(matches!(
            CryptoVerifBackend::parse_cv_value("true"),
            CounterexampleValue::Bool(true)
        ));
        assert!(matches!(
            CryptoVerifBackend::parse_cv_value("42"),
            CounterexampleValue::Int { value: 42, .. }
        ));
    }

    #[test]
    fn generate_cv_empty_spec() {
        use dashprove_usl::ast::Spec;

        let backend = CryptoVerifBackend::new();
        let spec = TypedSpec {
            spec: Spec {
                types: vec![],
                properties: vec![],
            },
            type_info: HashMap::new(),
        };
        let code = backend.generate_cv_code(&spec);
        assert!(code.contains("Generated by DashProve"));
        assert!(code.contains("type key"));
        assert!(code.contains("query secret"));
        assert!(code.contains("process"));
    }

    #[test]
    fn generate_cv_with_types() {
        use dashprove_usl::ast::{Spec, TypeDef};

        let backend = CryptoVerifBackend::new();
        let spec = TypedSpec {
            spec: Spec {
                types: vec![TypeDef {
                    name: "SessionKey".to_string(),
                    fields: vec![],
                }],
                properties: vec![],
            },
            type_info: HashMap::new(),
        };
        let code = backend.generate_cv_code(&spec);
        assert!(code.contains("type sessionkey"));
    }

    #[test]
    fn config_with_lib() {
        let config = CryptoVerifConfig {
            lib_dir: Some(PathBuf::from("/opt/cv/lib")),
            verbose: true,
            ..Default::default()
        };
        let backend = CryptoVerifBackend::with_config(config);
        assert!(backend.config.verbose);
        assert_eq!(backend.config.lib_dir, Some(PathBuf::from("/opt/cv/lib")));
    }
}
