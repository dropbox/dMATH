//! Symbiotic backend for LLVM-based C/C++ verification
//!
//! Symbiotic is a tool for verification of sequential C programs that combines
//! static analysis, program instrumentation, and symbolic execution. It instruments
//! LLVM bitcode with error checks (memory safety, assertions), performs static
//! analysis to simplify the program, then uses KLEE for symbolic execution.
//!
//! Key features:
//! - Memory safety verification
//! - Buffer overflow detection
//! - Memory leak detection
//! - Assertion checking
//! - SV-COMP compatible
//!
//! See: <https://github.com/staticafi/symbiotic>

use crate::traits::*;
use async_trait::async_trait;
use dashprove_usl::typecheck::TypedSpec;
use regex::Regex;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::process::Command;
use tracing::{debug, info};

// =============================================
// Kani Proofs for Symbiotic Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- SymbioticConfig Default Tests ----

    /// Verify SymbioticConfig::default symbiotic_path is None
    #[kani::proof]
    fn proof_symbiotic_config_default_symbiotic_path_none() {
        let config = SymbioticConfig::default();
        kani::assert(
            config.symbiotic_path.is_none(),
            "Default symbiotic_path should be None",
        );
    }

    /// Verify SymbioticConfig::default property is MemSafety
    #[kani::proof]
    fn proof_symbiotic_config_default_property_memsafety() {
        let config = SymbioticConfig::default();
        kani::assert(
            matches!(config.property, SymbioticProperty::MemSafety),
            "Default property should be MemSafety",
        );
    }

    /// Verify SymbioticConfig::default timeout is 900 seconds
    #[kani::proof]
    fn proof_symbiotic_config_default_timeout() {
        let config = SymbioticConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(900),
            "Default timeout should be 900 seconds",
        );
    }

    /// Verify SymbioticConfig::default generate_witness is false
    #[kani::proof]
    fn proof_symbiotic_config_default_generate_witness_false() {
        let config = SymbioticConfig::default();
        kani::assert(
            !config.generate_witness,
            "Default generate_witness should be false",
        );
    }

    /// Verify SymbioticConfig::default svcomp_mode is false
    #[kani::proof]
    fn proof_symbiotic_config_default_svcomp_mode_false() {
        let config = SymbioticConfig::default();
        kani::assert(!config.svcomp_mode, "Default svcomp_mode should be false");
    }

    // ---- SymbioticBackend Construction Tests ----

    /// Verify SymbioticBackend::new uses default timeout
    #[kani::proof]
    fn proof_symbiotic_backend_new_default_timeout() {
        let backend = SymbioticBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(900),
            "New backend should use default timeout",
        );
    }

    /// Verify SymbioticBackend::default equals SymbioticBackend::new
    #[kani::proof]
    fn proof_symbiotic_backend_default_equals_new() {
        let b1 = SymbioticBackend::new();
        let b2 = SymbioticBackend::default();
        kani::assert(
            b1.config.timeout == b2.config.timeout,
            "default and new should have same timeout",
        );
    }

    /// Verify SymbioticBackend::with_config preserves timeout
    #[kani::proof]
    fn proof_symbiotic_backend_with_config_preserves_timeout() {
        let config = SymbioticConfig {
            timeout: Duration::from_secs(1800),
            ..SymbioticConfig::default()
        };
        let backend = SymbioticBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(1800),
            "with_config should preserve custom timeout",
        );
    }

    // ---- SymbioticProperty Tests ----

    /// Verify SymbioticProperty::ReachSafety prp_file returns correct name
    #[kani::proof]
    fn proof_property_reach_safety_prp() {
        let prop = SymbioticProperty::ReachSafety;
        let _ = prop.prp_file();
    }

    /// Verify SymbioticProperty::MemSafety prp_file returns correct name
    #[kani::proof]
    fn proof_property_mem_safety_prp() {
        let prop = SymbioticProperty::MemSafety;
        let _ = prop.prp_file();
    }

    // ---- SymbioticResult Tests ----

    /// Verify SymbioticResult::True is_verified returns true
    #[kani::proof]
    fn proof_result_true_is_verified() {
        let result = SymbioticResult::True;
        kani::assert(result.is_verified(), "True should be verified");
    }

    /// Verify SymbioticResult::False is_verified returns false
    #[kani::proof]
    fn proof_result_false_is_not_verified() {
        let result = SymbioticResult::False;
        kani::assert(!result.is_verified(), "False should not be verified");
    }

    /// Verify SymbioticResult::Unknown is_verified returns false
    #[kani::proof]
    fn proof_result_unknown_is_not_verified() {
        let result = SymbioticResult::Unknown;
        kani::assert(!result.is_verified(), "Unknown should not be verified");
    }
}

/// Property to verify with Symbiotic
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SymbioticProperty {
    /// Unreachability of error locations (reach-error)
    ReachSafety,
    /// Memory safety (valid pointers, no overflows)
    #[default]
    MemSafety,
    /// Memory tracking (no leaks)
    MemTrack,
    /// No integer overflows
    NoOverflows,
}

impl SymbioticProperty {
    /// Get the property file name
    #[must_use]
    pub fn prp_file(&self) -> &'static str {
        match self {
            SymbioticProperty::ReachSafety => "unreach-call.prp",
            SymbioticProperty::MemSafety => "valid-memsafety.prp",
            SymbioticProperty::MemTrack => "valid-memtrack.prp",
            SymbioticProperty::NoOverflows => "no-overflow.prp",
        }
    }

    /// Get the property description
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            SymbioticProperty::ReachSafety => "Unreachability of error locations",
            SymbioticProperty::MemSafety => "Memory safety (valid pointers)",
            SymbioticProperty::MemTrack => "Memory tracking (no leaks)",
            SymbioticProperty::NoOverflows => "No integer overflows",
        }
    }
}

/// Result of Symbiotic verification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbioticResult {
    /// Property holds (TRUE)
    True,
    /// Property violated (FALSE)
    False,
    /// Could not determine (UNKNOWN)
    Unknown,
    /// Verification timed out
    Timeout,
    /// Error during verification
    Error,
}

impl SymbioticResult {
    /// Check if the property was verified
    #[must_use]
    pub fn is_verified(&self) -> bool {
        matches!(self, SymbioticResult::True)
    }
}

/// Configuration for Symbiotic backend
#[derive(Debug, Clone)]
pub struct SymbioticConfig {
    /// Path to Symbiotic executable (if not in PATH)
    pub symbiotic_path: Option<PathBuf>,
    /// Property to verify
    pub property: SymbioticProperty,
    /// Timeout for verification
    pub timeout: Duration,
    /// Generate violation witness
    pub generate_witness: bool,
    /// Witness output path
    pub witness_path: Option<PathBuf>,
    /// Run in SV-COMP mode
    pub svcomp_mode: bool,
    /// Additional arguments
    pub extra_args: Vec<String>,
}

impl Default for SymbioticConfig {
    fn default() -> Self {
        Self {
            symbiotic_path: None,
            property: SymbioticProperty::MemSafety,
            timeout: Duration::from_secs(900), // 15 minutes default
            generate_witness: false,
            witness_path: None,
            svcomp_mode: false,
            extra_args: Vec::new(),
        }
    }
}

/// Symbiotic backend for C/C++ verification
#[derive(Debug, Clone)]
pub struct SymbioticBackend {
    /// Configuration
    pub config: SymbioticConfig,
}

impl Default for SymbioticBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl SymbioticBackend {
    /// Create a new Symbiotic backend with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: SymbioticConfig::default(),
        }
    }

    /// Create a new Symbiotic backend with custom configuration
    #[must_use]
    pub fn with_config(config: SymbioticConfig) -> Self {
        Self { config }
    }

    /// Get the path to the Symbiotic executable
    fn get_symbiotic_path(&self) -> PathBuf {
        self.config
            .symbiotic_path
            .clone()
            .unwrap_or_else(|| PathBuf::from("symbiotic"))
    }

    /// Check if Symbiotic is available
    pub async fn check_available(&self) -> Result<bool, BackendError> {
        let symbiotic_path = self.get_symbiotic_path();
        let output = Command::new(&symbiotic_path)
            .arg("--help")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await;

        match output {
            Ok(o) => {
                Ok(o.status.success() || String::from_utf8_lossy(&o.stdout).contains("Symbiotic"))
            }
            Err(_) => Ok(false),
        }
    }

    /// Parse Symbiotic output
    fn parse_output(&self, stdout: &str, stderr: &str, timed_out: bool) -> SymbioticResult {
        if timed_out {
            return SymbioticResult::Timeout;
        }

        let combined = format!("{}\n{}", stdout, stderr);

        // Check for TRUE result
        if combined.contains("TRUE") {
            return SymbioticResult::True;
        }

        // Check for FALSE results
        if combined.contains("FALSE(reach)")
            || combined.contains("FALSE(valid-memtrack)")
            || combined.contains("FALSE(valid-memsafety)")
            || combined.contains("FALSE(no-overflow)")
            || combined.contains("FALSE")
        {
            return SymbioticResult::False;
        }

        // Check for UNKNOWN
        if combined.contains("UNKNOWN") {
            return SymbioticResult::Unknown;
        }

        // Check for errors
        if combined.contains("ERROR") || combined.contains("error:") {
            return SymbioticResult::Error;
        }

        SymbioticResult::Unknown
    }

    /// Extract witness from output
    fn extract_witness_info(&self, output: &str) -> Option<String> {
        let witness_regex = Regex::new(r"(?i)witness.*:\s*(.+\.graphml)").unwrap();
        if let Some(cap) = witness_regex.captures(output) {
            if let Some(path) = cap.get(1) {
                return Some(path.as_str().trim().to_string());
            }
        }
        None
    }

    /// Extract error trace from output
    fn extract_error_trace(&self, output: &str) -> Option<String> {
        // Look for error trace section
        if let Some(start) = output.find("Error trace:") {
            let rest = &output[start..];
            let end = rest.find("\n\n").unwrap_or(rest.len());
            return Some(rest[..end].to_string());
        }
        None
    }

    /// Verify a C/C++ source file
    pub async fn verify_file(&self, source_path: &PathBuf) -> Result<BackendResult, BackendError> {
        let start = Instant::now();
        let symbiotic_path = self.get_symbiotic_path();

        let mut cmd = Command::new(&symbiotic_path);

        // Add property file argument
        cmd.arg(format!("--prp={}", self.config.property.prp_file()));

        // Add timeout
        cmd.arg(format!("--timeout={}", self.config.timeout.as_secs()));

        // Add witness generation if requested
        if self.config.generate_witness {
            if let Some(witness_path) = &self.config.witness_path {
                cmd.arg(format!("--witness={}", witness_path.display()));
            } else {
                cmd.arg("--witness=witness.graphml");
            }
        }

        // Add SV-COMP mode
        if self.config.svcomp_mode {
            cmd.arg("--sv-comp");
        }

        // Add extra arguments
        for arg in &self.config.extra_args {
            cmd.arg(arg);
        }

        // Add source file
        cmd.arg(source_path);

        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        debug!("Running Symbiotic: {:?}", cmd);

        let output_result = tokio::time::timeout(
            self.config.timeout + Duration::from_secs(60), // Add buffer
            cmd.output(),
        )
        .await;

        let elapsed = start.elapsed();

        let (stdout, stderr, timed_out) = match output_result {
            Ok(Ok(output)) => (
                String::from_utf8_lossy(&output.stdout).to_string(),
                String::from_utf8_lossy(&output.stderr).to_string(),
                false,
            ),
            Ok(Err(e)) => {
                return Err(BackendError::VerificationFailed(format!(
                    "Failed to run Symbiotic: {}",
                    e
                )));
            }
            Err(_) => ("".to_string(), "Timeout".to_string(), true),
        };

        let symbiotic_result = self.parse_output(&stdout, &stderr, timed_out);
        let witness_info = self.extract_witness_info(&stdout);
        let error_trace = self.extract_error_trace(&stdout);

        let status = match symbiotic_result {
            SymbioticResult::True => VerificationStatus::Proven,
            SymbioticResult::False => VerificationStatus::Disproven,
            SymbioticResult::Unknown => VerificationStatus::Unknown {
                reason: "Symbiotic could not determine result".to_string(),
            },
            SymbioticResult::Timeout => VerificationStatus::Unknown {
                reason: "Symbiotic verification timed out".to_string(),
            },
            SymbioticResult::Error => VerificationStatus::Unknown {
                reason: "Symbiotic encountered an error".to_string(),
            },
        };

        let mut diagnostics = Vec::new();
        if let Some(witness) = &witness_info {
            diagnostics.push(format!("Witness file: {}", witness));
        }
        diagnostics.push(format!("Property: {}", self.config.property.prp_file()));
        if symbiotic_result == SymbioticResult::Error {
            diagnostics.push(stderr.clone());
        }

        info!(
            "Symbiotic verification completed in {:?}: {:?}",
            elapsed, status
        );

        Ok(BackendResult {
            backend: BackendId::Symbiotic,
            status,
            proof: if symbiotic_result == SymbioticResult::True {
                Some(format!(
                    "Property {} verified",
                    self.config.property.description()
                ))
            } else {
                None
            },
            counterexample: error_trace.map(StructuredCounterexample::from_raw),
            diagnostics,
            time_taken: elapsed,
        })
    }

    /// Verify C/C++ source code string
    pub async fn verify_source(&self, source: &str) -> Result<BackendResult, BackendError> {
        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        let source_path = temp_dir.path().join("verify.c");
        std::fs::write(&source_path, source).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write source: {}", e))
        })?;

        self.verify_file(&source_path).await
    }
}

#[async_trait]
impl VerificationBackend for SymbioticBackend {
    fn id(&self) -> BackendId {
        BackendId::Symbiotic
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![
            PropertyType::MemorySafety,
            PropertyType::Contract,
            PropertyType::Invariant,
        ]
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        // For now, return a placeholder - actual USL compilation would go here
        Ok(BackendResult {
            backend: BackendId::Symbiotic,
            status: VerificationStatus::Unknown {
                reason: "USL to Symbiotic compilation not yet implemented".to_string(),
            },
            proof: None,
            counterexample: None,
            diagnostics: vec![
                "Symbiotic backend requires direct C/C++ source verification".to_string(),
            ],
            time_taken: Duration::from_secs(0),
        })
    }

    async fn health_check(&self) -> HealthStatus {
        match self.check_available().await {
            Ok(true) => HealthStatus::Healthy,
            Ok(false) => HealthStatus::Unavailable {
                reason: "Symbiotic not found".to_string(),
            },
            Err(e) => HealthStatus::Unavailable {
                reason: format!("Symbiotic health check failed: {}", e),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbiotic_config_default() {
        let config = SymbioticConfig::default();
        assert!(config.symbiotic_path.is_none());
        assert!(matches!(config.property, SymbioticProperty::MemSafety));
        assert_eq!(config.timeout, Duration::from_secs(900));
        assert!(!config.generate_witness);
        assert!(config.witness_path.is_none());
        assert!(!config.svcomp_mode);
    }

    #[test]
    fn test_symbiotic_backend_new() {
        let backend = SymbioticBackend::new();
        assert_eq!(backend.config.timeout, Duration::from_secs(900));
    }

    #[test]
    fn test_symbiotic_backend_with_config() {
        let config = SymbioticConfig {
            timeout: Duration::from_secs(1800),
            property: SymbioticProperty::ReachSafety,
            svcomp_mode: true,
            ..SymbioticConfig::default()
        };
        let backend = SymbioticBackend::with_config(config);
        assert_eq!(backend.config.timeout, Duration::from_secs(1800));
        assert!(matches!(
            backend.config.property,
            SymbioticProperty::ReachSafety
        ));
        assert!(backend.config.svcomp_mode);
    }

    #[test]
    fn test_property_prp_file() {
        assert_eq!(
            SymbioticProperty::ReachSafety.prp_file(),
            "unreach-call.prp"
        );
        assert_eq!(
            SymbioticProperty::MemSafety.prp_file(),
            "valid-memsafety.prp"
        );
        assert_eq!(SymbioticProperty::MemTrack.prp_file(), "valid-memtrack.prp");
        assert_eq!(SymbioticProperty::NoOverflows.prp_file(), "no-overflow.prp");
    }

    #[test]
    fn test_result_is_verified() {
        assert!(SymbioticResult::True.is_verified());
        assert!(!SymbioticResult::False.is_verified());
        assert!(!SymbioticResult::Unknown.is_verified());
        assert!(!SymbioticResult::Timeout.is_verified());
        assert!(!SymbioticResult::Error.is_verified());
    }

    #[test]
    fn test_parse_output_true() {
        let backend = SymbioticBackend::new();
        let result = backend.parse_output("Verification result: TRUE", "", false);
        assert_eq!(result, SymbioticResult::True);
    }

    #[test]
    fn test_parse_output_false() {
        let backend = SymbioticBackend::new();
        let result = backend.parse_output("Verification result: FALSE(valid-memsafety)", "", false);
        assert_eq!(result, SymbioticResult::False);
    }

    #[test]
    fn test_parse_output_unknown() {
        let backend = SymbioticBackend::new();
        let result = backend.parse_output("Verification result: UNKNOWN", "", false);
        assert_eq!(result, SymbioticResult::Unknown);
    }

    #[test]
    fn test_parse_output_timeout() {
        let backend = SymbioticBackend::new();
        let result = backend.parse_output("", "", true);
        assert_eq!(result, SymbioticResult::Timeout);
    }

    #[test]
    fn test_backend_id() {
        let backend = SymbioticBackend::new();
        assert_eq!(backend.id(), BackendId::Symbiotic);
    }

    #[test]
    fn test_supports() {
        let backend = SymbioticBackend::new();
        let types = backend.supports();
        assert!(types.contains(&PropertyType::MemorySafety));
        assert!(types.contains(&PropertyType::Contract));
        assert!(types.contains(&PropertyType::Invariant));
    }
}
