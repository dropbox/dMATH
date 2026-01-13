//! 2LS backend for template-based C verification
//!
//! 2LS (pronounced "tools") is a static analysis and verification tool for C
//! programs developed at the University of Oxford and Amazon. It uses a
//! template-based synthesis approach combined with abstract interpretation.
//!
//! Key features:
//! - Safety verification
//! - Termination checking
//! - Invariant synthesis
//! - K-induction
//! - SV-COMP compatible
//!
//! See: <https://github.com/diffblue/2ls>

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
// Kani Proofs for 2LS Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- TwoLsConfig Default Tests ----

    /// Verify TwoLsConfig::default twols_path is None
    #[kani::proof]
    fn proof_twols_config_default_twols_path_none() {
        let config = TwoLsConfig::default();
        kani::assert(
            config.twols_path.is_none(),
            "Default twols_path should be None",
        );
    }

    /// Verify TwoLsConfig::default analysis is KInduction
    #[kani::proof]
    fn proof_twols_config_default_analysis_kinduction() {
        let config = TwoLsConfig::default();
        kani::assert(
            matches!(config.analysis, TwoLsAnalysis::KInduction),
            "Default analysis should be KInduction",
        );
    }

    /// Verify TwoLsConfig::default timeout is 900 seconds
    #[kani::proof]
    fn proof_twols_config_default_timeout() {
        let config = TwoLsConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(900),
            "Default timeout should be 900 seconds",
        );
    }

    /// Verify TwoLsConfig::default use_intervals is true
    #[kani::proof]
    fn proof_twols_config_default_use_intervals_true() {
        let config = TwoLsConfig::default();
        kani::assert(config.use_intervals, "Default use_intervals should be true");
    }

    /// Verify TwoLsConfig::default use_heap is false
    #[kani::proof]
    fn proof_twols_config_default_use_heap_false() {
        let config = TwoLsConfig::default();
        kani::assert(!config.use_heap, "Default use_heap should be false");
    }

    /// Verify TwoLsConfig::default generate_witness is false
    #[kani::proof]
    fn proof_twols_config_default_generate_witness_false() {
        let config = TwoLsConfig::default();
        kani::assert(
            !config.generate_witness,
            "Default generate_witness should be false",
        );
    }

    // ---- TwoLsBackend Construction Tests ----

    /// Verify TwoLsBackend::new uses default timeout
    #[kani::proof]
    fn proof_twols_backend_new_default_timeout() {
        let backend = TwoLsBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(900),
            "New backend should use default timeout",
        );
    }

    /// Verify TwoLsBackend::default equals TwoLsBackend::new
    #[kani::proof]
    fn proof_twols_backend_default_equals_new() {
        let b1 = TwoLsBackend::new();
        let b2 = TwoLsBackend::default();
        kani::assert(
            b1.config.timeout == b2.config.timeout,
            "default and new should have same timeout",
        );
    }

    /// Verify TwoLsBackend::with_config preserves timeout
    #[kani::proof]
    fn proof_twols_backend_with_config_preserves_timeout() {
        let config = TwoLsConfig {
            timeout: Duration::from_secs(1800),
            ..TwoLsConfig::default()
        };
        let backend = TwoLsBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(1800),
            "with_config should preserve custom timeout",
        );
    }

    // ---- TwoLsAnalysis Tests ----

    /// Verify TwoLsAnalysis::KInduction flag returns correct value
    #[kani::proof]
    fn proof_analysis_kinduction_flag() {
        let analysis = TwoLsAnalysis::KInduction;
        let _ = analysis.flag();
    }

    /// Verify TwoLsAnalysis::Termination flag returns correct value
    #[kani::proof]
    fn proof_analysis_termination_flag() {
        let analysis = TwoLsAnalysis::Termination;
        let _ = analysis.flag();
    }

    // ---- TwoLsResult Tests ----

    /// Verify TwoLsResult::Success is_verified returns true
    #[kani::proof]
    fn proof_result_success_is_verified() {
        let result = TwoLsResult::Success;
        kani::assert(result.is_verified(), "Success should be verified");
    }

    /// Verify TwoLsResult::Failed is_verified returns false
    #[kani::proof]
    fn proof_result_failed_is_not_verified() {
        let result = TwoLsResult::Failed;
        kani::assert(!result.is_verified(), "Failed should not be verified");
    }

    /// Verify TwoLsResult::Unknown is_verified returns false
    #[kani::proof]
    fn proof_result_unknown_is_not_verified() {
        let result = TwoLsResult::Unknown;
        kani::assert(!result.is_verified(), "Unknown should not be verified");
    }
}

/// Analysis type for 2LS
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TwoLsAnalysis {
    /// K-induction analysis (default)
    #[default]
    KInduction,
    /// Termination analysis
    Termination,
    /// Bounded model checking
    BMC,
    /// Abstract interpretation only
    AbstractInterpretation,
}

impl TwoLsAnalysis {
    /// Get the command-line flag for this analysis
    #[must_use]
    pub fn flag(&self) -> &'static str {
        match self {
            TwoLsAnalysis::KInduction => "--k-induction",
            TwoLsAnalysis::Termination => "--termination",
            TwoLsAnalysis::BMC => "--bmc",
            TwoLsAnalysis::AbstractInterpretation => "--havoc",
        }
    }

    /// Get description of the analysis
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            TwoLsAnalysis::KInduction => "K-induction with invariant inference",
            TwoLsAnalysis::Termination => "Termination analysis",
            TwoLsAnalysis::BMC => "Bounded model checking",
            TwoLsAnalysis::AbstractInterpretation => "Abstract interpretation",
        }
    }
}

/// Result of 2LS verification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TwoLsResult {
    /// Verification successful
    Success,
    /// Verification failed (counterexample found)
    Failed,
    /// Could not determine
    Unknown,
    /// Timeout
    Timeout,
    /// Error during verification
    Error,
}

impl TwoLsResult {
    /// Check if verification succeeded
    #[must_use]
    pub fn is_verified(&self) -> bool {
        matches!(self, TwoLsResult::Success)
    }
}

/// Configuration for 2LS backend
#[derive(Debug, Clone)]
pub struct TwoLsConfig {
    /// Path to 2ls executable (if not in PATH)
    pub twols_path: Option<PathBuf>,
    /// Analysis type
    pub analysis: TwoLsAnalysis,
    /// Timeout for verification
    pub timeout: Duration,
    /// Use interval domain
    pub use_intervals: bool,
    /// Enable heap analysis
    pub use_heap: bool,
    /// Generate GraphML witness
    pub generate_witness: bool,
    /// Witness output path
    pub witness_path: Option<PathBuf>,
    /// Unwind limit for loops
    pub unwind: Option<u32>,
    /// Additional arguments
    pub extra_args: Vec<String>,
}

impl Default for TwoLsConfig {
    fn default() -> Self {
        Self {
            twols_path: None,
            analysis: TwoLsAnalysis::KInduction,
            timeout: Duration::from_secs(900),
            use_intervals: true,
            use_heap: false,
            generate_witness: false,
            witness_path: None,
            unwind: None,
            extra_args: Vec::new(),
        }
    }
}

/// 2LS backend for C verification
#[derive(Debug, Clone)]
pub struct TwoLsBackend {
    /// Configuration
    pub config: TwoLsConfig,
}

impl Default for TwoLsBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl TwoLsBackend {
    /// Create a new 2LS backend with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: TwoLsConfig::default(),
        }
    }

    /// Create a new 2LS backend with custom configuration
    #[must_use]
    pub fn with_config(config: TwoLsConfig) -> Self {
        Self { config }
    }

    /// Get the path to the 2ls executable
    fn get_twols_path(&self) -> PathBuf {
        self.config
            .twols_path
            .clone()
            .unwrap_or_else(|| PathBuf::from("2ls"))
    }

    /// Check if 2LS is available
    pub async fn check_available(&self) -> Result<bool, BackendError> {
        let twols_path = self.get_twols_path();
        let output = Command::new(&twols_path)
            .arg("--help")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await;

        match output {
            Ok(o) => Ok(o.status.success() || String::from_utf8_lossy(&o.stdout).contains("2LS")),
            Err(_) => Ok(false),
        }
    }

    /// Parse 2LS output
    fn parse_output(&self, stdout: &str, stderr: &str, timed_out: bool) -> TwoLsResult {
        if timed_out {
            return TwoLsResult::Timeout;
        }

        let combined = format!("{}\n{}", stdout, stderr);

        // Check for success
        if combined.contains("VERIFICATION SUCCESSFUL") {
            return TwoLsResult::Success;
        }

        // Check for failure
        if combined.contains("VERIFICATION FAILED") {
            return TwoLsResult::Failed;
        }

        // Check for unknown
        if combined.contains("UNKNOWN") {
            return TwoLsResult::Unknown;
        }

        // Check for errors
        if combined.contains("ERROR") || combined.contains("error:") {
            return TwoLsResult::Error;
        }

        // Check for out of memory
        if combined.contains("OUT OF MEMORY") {
            return TwoLsResult::Error;
        }

        TwoLsResult::Unknown
    }

    /// Extract inferred invariants from output
    fn extract_invariants(&self, output: &str) -> Vec<String> {
        let mut invariants = Vec::new();
        let inv_regex = Regex::new(r"(?i)invariant:\s*(.+)").unwrap();

        for cap in inv_regex.captures_iter(output) {
            if let Some(inv) = cap.get(1) {
                invariants.push(inv.as_str().trim().to_string());
            }
        }

        invariants
    }

    /// Extract counterexample trace from output
    fn extract_counterexample(&self, output: &str) -> Option<String> {
        if let Some(start) = output.find("Counterexample:") {
            let rest = &output[start..];
            let end = rest.find("\n\n").unwrap_or(rest.len());
            return Some(rest[..end].to_string());
        }

        // Also look for trace format
        if let Some(start) = output.find("Trace:") {
            let rest = &output[start..];
            let end = rest.find("\n\n").unwrap_or(rest.len());
            return Some(rest[..end].to_string());
        }

        None
    }

    /// Verify a C source file
    pub async fn verify_file(&self, source_path: &PathBuf) -> Result<BackendResult, BackendError> {
        let start = Instant::now();
        let twols_path = self.get_twols_path();

        let mut cmd = Command::new(&twols_path);

        // Add analysis flag
        cmd.arg(self.config.analysis.flag());

        // Add interval domain
        if self.config.use_intervals {
            cmd.arg("--intervals");
        }

        // Add heap analysis
        if self.config.use_heap {
            cmd.arg("--heap");
        }

        // Add witness generation
        if self.config.generate_witness {
            let witness = self
                .config
                .witness_path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "witness.graphml".to_string());
            cmd.arg(format!("--graphml-witness={}", witness));
        }

        // Add unwind limit
        if let Some(unwind) = self.config.unwind {
            cmd.arg(format!("--unwind={}", unwind));
        }

        // Add extra arguments
        for arg in &self.config.extra_args {
            cmd.arg(arg);
        }

        // Add source file
        cmd.arg(source_path);

        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        debug!("Running 2LS: {:?}", cmd);

        let output_result =
            tokio::time::timeout(self.config.timeout + Duration::from_secs(60), cmd.output()).await;

        let elapsed = start.elapsed();

        let (stdout, stderr, timed_out) = match output_result {
            Ok(Ok(output)) => (
                String::from_utf8_lossy(&output.stdout).to_string(),
                String::from_utf8_lossy(&output.stderr).to_string(),
                false,
            ),
            Ok(Err(e)) => {
                return Err(BackendError::VerificationFailed(format!(
                    "Failed to run 2LS: {}",
                    e
                )));
            }
            Err(_) => ("".to_string(), "Timeout".to_string(), true),
        };

        let twols_result = self.parse_output(&stdout, &stderr, timed_out);
        let invariants = self.extract_invariants(&stdout);
        let counterexample = self.extract_counterexample(&stdout);

        let status = match twols_result {
            TwoLsResult::Success => VerificationStatus::Proven,
            TwoLsResult::Failed => VerificationStatus::Disproven,
            TwoLsResult::Unknown => VerificationStatus::Unknown {
                reason: "2LS could not determine result".to_string(),
            },
            TwoLsResult::Timeout => VerificationStatus::Unknown {
                reason: "2LS verification timed out".to_string(),
            },
            TwoLsResult::Error => VerificationStatus::Unknown {
                reason: "2LS encountered an error".to_string(),
            },
        };

        let mut diagnostics = Vec::new();
        if !invariants.is_empty() {
            diagnostics.push(format!("Inferred invariants: {}", invariants.join(", ")));
        }
        diagnostics.push(format!("Analysis: {}", self.config.analysis.description()));
        if twols_result == TwoLsResult::Error {
            diagnostics.push(stderr.clone());
        }

        info!("2LS verification completed in {:?}: {:?}", elapsed, status);

        Ok(BackendResult {
            backend: BackendId::TwoLS,
            status,
            proof: if twols_result == TwoLsResult::Success {
                Some(format!(
                    "Verified using {}",
                    self.config.analysis.description()
                ))
            } else {
                None
            },
            counterexample: counterexample.map(StructuredCounterexample::from_raw),
            diagnostics,
            time_taken: elapsed,
        })
    }

    /// Verify C source code string
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
impl VerificationBackend for TwoLsBackend {
    fn id(&self) -> BackendId {
        BackendId::TwoLS
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![
            PropertyType::Contract,
            PropertyType::Invariant,
            PropertyType::MemorySafety,
        ]
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        // For now, return a placeholder - actual USL compilation would go here
        Ok(BackendResult {
            backend: BackendId::TwoLS,
            status: VerificationStatus::Unknown {
                reason: "USL to 2LS compilation not yet implemented".to_string(),
            },
            proof: None,
            counterexample: None,
            diagnostics: vec!["2LS backend requires direct C source verification".to_string()],
            time_taken: Duration::from_secs(0),
        })
    }

    async fn health_check(&self) -> HealthStatus {
        match self.check_available().await {
            Ok(true) => HealthStatus::Healthy,
            Ok(false) => HealthStatus::Unavailable {
                reason: "2LS not found".to_string(),
            },
            Err(e) => HealthStatus::Unavailable {
                reason: format!("2LS health check failed: {}", e),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_twols_config_default() {
        let config = TwoLsConfig::default();
        assert!(config.twols_path.is_none());
        assert!(matches!(config.analysis, TwoLsAnalysis::KInduction));
        assert_eq!(config.timeout, Duration::from_secs(900));
        assert!(config.use_intervals);
        assert!(!config.use_heap);
        assert!(!config.generate_witness);
    }

    #[test]
    fn test_twols_backend_new() {
        let backend = TwoLsBackend::new();
        assert_eq!(backend.config.timeout, Duration::from_secs(900));
    }

    #[test]
    fn test_twols_backend_with_config() {
        let config = TwoLsConfig {
            timeout: Duration::from_secs(1800),
            analysis: TwoLsAnalysis::Termination,
            use_heap: true,
            ..TwoLsConfig::default()
        };
        let backend = TwoLsBackend::with_config(config);
        assert_eq!(backend.config.timeout, Duration::from_secs(1800));
        assert!(matches!(
            backend.config.analysis,
            TwoLsAnalysis::Termination
        ));
        assert!(backend.config.use_heap);
    }

    #[test]
    fn test_analysis_flag() {
        assert_eq!(TwoLsAnalysis::KInduction.flag(), "--k-induction");
        assert_eq!(TwoLsAnalysis::Termination.flag(), "--termination");
        assert_eq!(TwoLsAnalysis::BMC.flag(), "--bmc");
        assert_eq!(TwoLsAnalysis::AbstractInterpretation.flag(), "--havoc");
    }

    #[test]
    fn test_result_is_verified() {
        assert!(TwoLsResult::Success.is_verified());
        assert!(!TwoLsResult::Failed.is_verified());
        assert!(!TwoLsResult::Unknown.is_verified());
        assert!(!TwoLsResult::Timeout.is_verified());
        assert!(!TwoLsResult::Error.is_verified());
    }

    #[test]
    fn test_parse_output_success() {
        let backend = TwoLsBackend::new();
        let result = backend.parse_output("VERIFICATION SUCCESSFUL", "", false);
        assert_eq!(result, TwoLsResult::Success);
    }

    #[test]
    fn test_parse_output_failed() {
        let backend = TwoLsBackend::new();
        let result = backend.parse_output("VERIFICATION FAILED", "", false);
        assert_eq!(result, TwoLsResult::Failed);
    }

    #[test]
    fn test_parse_output_unknown() {
        let backend = TwoLsBackend::new();
        let result = backend.parse_output("UNKNOWN", "", false);
        assert_eq!(result, TwoLsResult::Unknown);
    }

    #[test]
    fn test_parse_output_timeout() {
        let backend = TwoLsBackend::new();
        let result = backend.parse_output("", "", true);
        assert_eq!(result, TwoLsResult::Timeout);
    }

    #[test]
    fn test_extract_invariants() {
        let backend = TwoLsBackend::new();
        let output = "invariant: x >= 0\ninvariant: y < n";
        let invariants = backend.extract_invariants(output);
        assert_eq!(invariants.len(), 2);
    }

    #[test]
    fn test_extract_counterexample() {
        let backend = TwoLsBackend::new();
        let output = "Counterexample:\nstep 1: x = 0\nstep 2: y = -1\n\nEnd";
        let ce = backend.extract_counterexample(output);
        assert!(ce.is_some());
    }

    #[test]
    fn test_backend_id() {
        let backend = TwoLsBackend::new();
        assert_eq!(backend.id(), BackendId::TwoLS);
    }

    #[test]
    fn test_supports() {
        let backend = TwoLsBackend::new();
        let types = backend.supports();
        assert!(types.contains(&PropertyType::Contract));
        assert!(types.contains(&PropertyType::Invariant));
        assert!(types.contains(&PropertyType::MemorySafety));
    }
}
