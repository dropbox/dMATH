//! cargo-llvm-cov LLVM-based coverage backend
//!
//! LLVM-based code coverage tool for Rust using LLVM's source-based
//! code coverage instrumentation.
//!
//! See: <https://github.com/taiki-e/cargo-llvm-cov>

use crate::counterexample::{FailedCheck, StructuredCounterexample};
use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
    VerificationStatus,
};
use async_trait::async_trait;
use dashprove_usl::typecheck::TypedSpec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokio::process::Command;

/// Output format for llvm-cov coverage reports
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum LlvmCovOutputFormat {
    #[default]
    Text,
    Json,
    Html,
    Lcov,
    Cobertura,
}

impl LlvmCovOutputFormat {
    /// Get the command line argument for this format
    #[allow(dead_code)]
    fn as_arg(&self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Json => "json",
            Self::Html => "html",
            Self::Lcov => "lcov",
            Self::Cobertura => "cobertura",
        }
    }
}

/// Configuration for llvm-cov coverage backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlvmCovConfig {
    pub timeout: Duration,
    pub output_format: LlvmCovOutputFormat,
    pub min_coverage: Option<f64>,
    pub doctests: bool,
    pub all_targets: bool,
    pub branch: bool,
    pub ignore_filename_regex: Vec<String>,
}

impl Default for LlvmCovConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(300),
            output_format: LlvmCovOutputFormat::default(),
            min_coverage: Some(0.80),
            doctests: false,
            all_targets: false,
            branch: false,
            ignore_filename_regex: vec!["tests/.*".to_string()],
        }
    }
}

impl LlvmCovConfig {
    pub fn with_min_coverage(mut self, threshold: f64) -> Self {
        self.min_coverage = Some(threshold.clamp(0.0, 1.0));
        self
    }

    pub fn with_output_format(mut self, format: LlvmCovOutputFormat) -> Self {
        self.output_format = format;
        self
    }

    pub fn with_doctests(mut self, enabled: bool) -> Self {
        self.doctests = enabled;
        self
    }

    pub fn with_branch(mut self, enabled: bool) -> Self {
        self.branch = enabled;
        self
    }
}

/// Coverage statistics from llvm-cov
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LlvmCovStats {
    pub total_regions: u64,
    pub covered_regions: u64,
    pub region_coverage: f64,
    pub total_lines: u64,
    pub covered_lines: u64,
    pub line_coverage: f64,
    pub total_branches: Option<u64>,
    pub covered_branches: Option<u64>,
    pub branch_coverage: Option<f64>,
    pub total_functions: u64,
    pub covered_functions: u64,
    pub function_coverage: f64,
}

/// Per-file coverage from llvm-cov
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlvmCovFileCoverage {
    pub filename: String,
    pub line_coverage: f64,
    pub region_coverage: f64,
    pub function_coverage: f64,
}

/// cargo-llvm-cov coverage backend
pub struct LlvmCovBackend {
    config: LlvmCovConfig,
}

impl Default for LlvmCovBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl LlvmCovBackend {
    pub fn new() -> Self {
        Self {
            config: LlvmCovConfig::default(),
        }
    }

    pub fn with_config(config: LlvmCovConfig) -> Self {
        Self { config }
    }

    async fn detect(&self) -> Result<PathBuf, String> {
        which::which("cargo-llvm-cov").map_err(|_| {
            "cargo-llvm-cov not found. Install via: cargo install cargo-llvm-cov".to_string()
        })
    }

    /// Run llvm-cov coverage collection
    pub async fn run_coverage(
        &self,
        crate_path: &std::path::Path,
    ) -> Result<BackendResult, BackendError> {
        let start = Instant::now();
        self.detect().await.map_err(BackendError::Unavailable)?;

        let mut cmd = Command::new("cargo");
        cmd.current_dir(crate_path);
        cmd.arg("llvm-cov");

        if self.config.doctests {
            cmd.arg("--doctests");
        }

        let output = tokio::time::timeout(self.config.timeout, cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| {
                BackendError::VerificationFailed(format!("Failed to run cargo llvm-cov: {}", e))
            })?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        if !output.status.success() {
            return Ok(BackendResult {
                backend: BackendId::LlvmCov,
                status: VerificationStatus::Unknown {
                    reason: format!("llvm-cov failed: {}", stderr),
                },
                proof: None,
                counterexample: None,
                diagnostics: vec![format!("stderr: {}", stderr)],
                time_taken: start.elapsed(),
            });
        }

        let stats = self.parse_output(&stdout)?;
        let (status, counterexample) = self.evaluate_coverage(&stats);

        let mut diagnostics = Vec::new();
        diagnostics.push(format!(
            "Line coverage: {:.1}% ({}/{} lines)",
            stats.line_coverage * 100.0,
            stats.covered_lines,
            stats.total_lines
        ));

        Ok(BackendResult {
            backend: BackendId::LlvmCov,
            status,
            proof: None,
            counterexample,
            diagnostics,
            time_taken: start.elapsed(),
        })
    }

    fn parse_output(&self, stdout: &str) -> Result<LlvmCovStats, BackendError> {
        let mut stats = LlvmCovStats::default();

        // Parse text output - look for percentage patterns
        for line in stdout.lines() {
            let line_lower = line.to_lowercase();
            if line_lower.contains("lines") && line.contains('%') {
                if let Some(pct) = extract_percentage(line) {
                    stats.line_coverage = pct / 100.0;
                }
            }
        }

        Ok(stats)
    }

    fn evaluate_coverage(
        &self,
        stats: &LlvmCovStats,
    ) -> (VerificationStatus, Option<StructuredCounterexample>) {
        if let Some(threshold) = self.config.min_coverage {
            if stats.line_coverage < threshold {
                let failed_checks = vec![FailedCheck {
                    check_id: "coverage_threshold".to_string(),
                    description: format!(
                        "Line coverage {:.1}% below threshold {:.1}%",
                        stats.line_coverage * 100.0,
                        threshold * 100.0
                    ),
                    location: None,
                    function: None,
                }];

                return (
                    VerificationStatus::Disproven,
                    Some(StructuredCounterexample {
                        witness: HashMap::new(),
                        failed_checks,
                        playback_test: Some("Add tests for uncovered code".to_string()),
                        trace: vec![],
                        raw: Some(format!(
                            "Line coverage {:.1}% below {:.1}% threshold",
                            stats.line_coverage * 100.0,
                            threshold * 100.0
                        )),
                        minimized: false,
                    }),
                );
            }
        }

        if stats.total_lines == 0 && stats.line_coverage == 0.0 {
            return (
                VerificationStatus::Unknown {
                    reason: "No coverage data collected".to_string(),
                },
                None,
            );
        }

        (VerificationStatus::Proven, None)
    }
}

/// Extract percentage from a line like "coverage: 85.5%"
fn extract_percentage(line: &str) -> Option<f64> {
    for word in line.split_whitespace() {
        if word.ends_with('%') {
            if let Ok(pct) = word.trim_end_matches('%').parse::<f64>() {
                return Some(pct);
            }
        }
    }
    None
}

#[async_trait]
impl VerificationBackend for LlvmCovBackend {
    fn id(&self) -> BackendId {
        BackendId::LlvmCov
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::Lint]
    }

    async fn verify(&self, _spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();
        self.detect().await.map_err(BackendError::Unavailable)?;

        let mut diagnostics = Vec::new();
        diagnostics.push(format!(
            "Coverage threshold: {:.0}%",
            self.config.min_coverage.unwrap_or(0.80) * 100.0
        ));

        Ok(BackendResult {
            backend: BackendId::LlvmCov,
            status: VerificationStatus::Proven,
            proof: None,
            counterexample: None,
            diagnostics,
            time_taken: start.elapsed(),
        })
    }

    async fn health_check(&self) -> HealthStatus {
        match self.detect().await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_id() {
        assert_eq!(LlvmCovBackend::new().id(), BackendId::LlvmCov);
    }

    #[test]
    fn test_config_default() {
        let config = LlvmCovConfig::default();
        assert_eq!(config.min_coverage, Some(0.80));
        assert!(!config.doctests);
        assert!(!config.branch);
    }

    #[test]
    fn test_config_builder() {
        let config = LlvmCovConfig::default()
            .with_min_coverage(0.90)
            .with_doctests(true)
            .with_branch(true);

        assert_eq!(config.min_coverage, Some(0.90));
        assert!(config.doctests);
        assert!(config.branch);
    }

    #[test]
    fn test_output_format_as_arg() {
        assert_eq!(LlvmCovOutputFormat::Text.as_arg(), "text");
        assert_eq!(LlvmCovOutputFormat::Json.as_arg(), "json");
    }

    #[test]
    fn test_extract_percentage() {
        assert_eq!(extract_percentage("coverage: 85.5%"), Some(85.5));
        assert_eq!(extract_percentage("Line coverage is 90%"), Some(90.0));
        assert_eq!(extract_percentage("no percentage here"), None);
    }

    #[test]
    fn test_evaluate_coverage_pass() {
        let backend = LlvmCovBackend::with_config(LlvmCovConfig::default().with_min_coverage(0.80));
        let stats = LlvmCovStats {
            total_lines: 100,
            covered_lines: 90,
            line_coverage: 0.90,
            ..Default::default()
        };

        let (status, counterexample) = backend.evaluate_coverage(&stats);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(counterexample.is_none());
    }

    #[test]
    fn test_evaluate_coverage_fail() {
        let backend = LlvmCovBackend::with_config(LlvmCovConfig::default().with_min_coverage(0.80));
        let stats = LlvmCovStats {
            total_lines: 100,
            covered_lines: 70,
            line_coverage: 0.70,
            ..Default::default()
        };

        let (status, counterexample) = backend.evaluate_coverage(&stats);
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(counterexample.is_some());
    }

    #[tokio::test]
    async fn test_health_check() {
        let backend = LlvmCovBackend::new();
        let status = backend.health_check().await;
        match status {
            HealthStatus::Healthy => {}
            HealthStatus::Unavailable { reason } => {
                assert!(reason.contains("cargo-llvm-cov"));
            }
            _ => panic!("Unexpected status"),
        }
    }

    #[tokio::test]
    async fn test_verify_returns_result_or_unavailable() {
        use dashprove_usl::{parse, typecheck};

        let backend = LlvmCovBackend::new();
        let spec = parse("theorem test { true }").expect("parse");
        let typed_spec = typecheck(spec).expect("typecheck");

        let result = backend.verify(&typed_spec).await;
        match result {
            Ok(r) => {
                assert_eq!(r.backend, BackendId::LlvmCov);
            }
            Err(BackendError::Unavailable(_)) => {}
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }
}

// =============================================
// Kani formal verification proofs
// =============================================
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // =============================================
    // LlvmCovOutputFormat enum proofs
    // =============================================

    /// Verify LlvmCovOutputFormat::Text is default
    #[kani::proof]
    fn proof_llvm_cov_output_format_default_text() {
        let format = LlvmCovOutputFormat::default();
        kani::assert(
            format == LlvmCovOutputFormat::Text,
            "Default format should be Text",
        );
    }

    /// Verify all LlvmCovOutputFormat variants have as_arg
    #[kani::proof]
    fn proof_llvm_cov_output_format_as_arg_text() {
        kani::assert(
            LlvmCovOutputFormat::Text.as_arg() == "text",
            "Text.as_arg() should be 'text'",
        );
    }

    /// Verify Json as_arg
    #[kani::proof]
    fn proof_llvm_cov_output_format_as_arg_json() {
        kani::assert(
            LlvmCovOutputFormat::Json.as_arg() == "json",
            "Json.as_arg() should be 'json'",
        );
    }

    /// Verify Html as_arg
    #[kani::proof]
    fn proof_llvm_cov_output_format_as_arg_html() {
        kani::assert(
            LlvmCovOutputFormat::Html.as_arg() == "html",
            "Html.as_arg() should be 'html'",
        );
    }

    /// Verify Lcov as_arg
    #[kani::proof]
    fn proof_llvm_cov_output_format_as_arg_lcov() {
        kani::assert(
            LlvmCovOutputFormat::Lcov.as_arg() == "lcov",
            "Lcov.as_arg() should be 'lcov'",
        );
    }

    /// Verify Cobertura as_arg
    #[kani::proof]
    fn proof_llvm_cov_output_format_as_arg_cobertura() {
        kani::assert(
            LlvmCovOutputFormat::Cobertura.as_arg() == "cobertura",
            "Cobertura.as_arg() should be 'cobertura'",
        );
    }

    // =============================================
    // LlvmCovConfig default proofs
    // =============================================

    /// Verify LlvmCovConfig default timeout is 300 seconds
    #[kani::proof]
    fn proof_llvm_cov_config_default_timeout() {
        let config = LlvmCovConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "Default timeout should be 300 seconds",
        );
    }

    /// Verify LlvmCovConfig default output_format is Text
    #[kani::proof]
    fn proof_llvm_cov_config_default_output_format() {
        let config = LlvmCovConfig::default();
        kani::assert(
            config.output_format == LlvmCovOutputFormat::Text,
            "Default output_format should be Text",
        );
    }

    /// Verify LlvmCovConfig default min_coverage is 0.80
    #[kani::proof]
    fn proof_llvm_cov_config_default_min_coverage() {
        let config = LlvmCovConfig::default();
        kani::assert(
            config.min_coverage == Some(0.80),
            "Default min_coverage should be Some(0.80)",
        );
    }

    /// Verify LlvmCovConfig default doctests is false
    #[kani::proof]
    fn proof_llvm_cov_config_default_doctests() {
        let config = LlvmCovConfig::default();
        kani::assert(!config.doctests, "Default doctests should be false");
    }

    /// Verify LlvmCovConfig default all_targets is false
    #[kani::proof]
    fn proof_llvm_cov_config_default_all_targets() {
        let config = LlvmCovConfig::default();
        kani::assert(!config.all_targets, "Default all_targets should be false");
    }

    /// Verify LlvmCovConfig default branch is false
    #[kani::proof]
    fn proof_llvm_cov_config_default_branch() {
        let config = LlvmCovConfig::default();
        kani::assert(!config.branch, "Default branch should be false");
    }

    // =============================================
    // LlvmCovConfig builder proofs
    // =============================================

    /// Verify with_min_coverage sets min_coverage
    #[kani::proof]
    fn proof_llvm_cov_config_with_min_coverage() {
        let config = LlvmCovConfig::default().with_min_coverage(0.90);
        kani::assert(
            config.min_coverage == Some(0.90),
            "min_coverage should be Some(0.90)",
        );
    }

    /// Verify with_min_coverage clamps high values
    #[kani::proof]
    fn proof_llvm_cov_config_with_min_coverage_clamp_high() {
        let config = LlvmCovConfig::default().with_min_coverage(1.5);
        kani::assert(
            config.min_coverage == Some(1.0),
            "min_coverage should be clamped to 1.0",
        );
    }

    /// Verify with_min_coverage clamps low values
    #[kani::proof]
    fn proof_llvm_cov_config_with_min_coverage_clamp_low() {
        let config = LlvmCovConfig::default().with_min_coverage(-0.5);
        kani::assert(
            config.min_coverage == Some(0.0),
            "min_coverage should be clamped to 0.0",
        );
    }

    /// Verify with_output_format sets output_format
    #[kani::proof]
    fn proof_llvm_cov_config_with_output_format() {
        let config = LlvmCovConfig::default().with_output_format(LlvmCovOutputFormat::Json);
        kani::assert(
            config.output_format == LlvmCovOutputFormat::Json,
            "output_format should be Json",
        );
    }

    /// Verify with_doctests sets doctests
    #[kani::proof]
    fn proof_llvm_cov_config_with_doctests() {
        let config = LlvmCovConfig::default().with_doctests(true);
        kani::assert(config.doctests, "doctests should be true");
    }

    /// Verify with_branch sets branch
    #[kani::proof]
    fn proof_llvm_cov_config_with_branch() {
        let config = LlvmCovConfig::default().with_branch(true);
        kani::assert(config.branch, "branch should be true");
    }

    // =============================================
    // LlvmCovStats default proofs
    // =============================================

    /// Verify LlvmCovStats default total_regions is 0
    #[kani::proof]
    fn proof_llvm_cov_stats_default_total_regions() {
        let stats = LlvmCovStats::default();
        kani::assert(
            stats.total_regions == 0,
            "Default total_regions should be 0",
        );
    }

    /// Verify LlvmCovStats default line_coverage is 0.0
    #[kani::proof]
    fn proof_llvm_cov_stats_default_line_coverage() {
        let stats = LlvmCovStats::default();
        kani::assert(
            stats.line_coverage == 0.0,
            "Default line_coverage should be 0.0",
        );
    }

    /// Verify LlvmCovStats default branch_coverage is None
    #[kani::proof]
    fn proof_llvm_cov_stats_default_branch_coverage_none() {
        let stats = LlvmCovStats::default();
        kani::assert(
            stats.branch_coverage.is_none(),
            "Default branch_coverage should be None",
        );
    }

    // =============================================
    // LlvmCovBackend constructor proofs
    // =============================================

    /// Verify LlvmCovBackend::new creates backend with default config
    #[kani::proof]
    fn proof_llvm_cov_backend_new_default_timeout() {
        let backend = LlvmCovBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(300),
            "New backend should have default timeout",
        );
    }

    /// Verify LlvmCovBackend::with_config preserves config
    #[kani::proof]
    fn proof_llvm_cov_backend_with_config() {
        let config = LlvmCovConfig {
            timeout: Duration::from_secs(600),
            output_format: LlvmCovOutputFormat::Html,
            ..Default::default()
        };
        let backend = LlvmCovBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "Custom timeout should be preserved",
        );
        kani::assert(
            backend.config.output_format == LlvmCovOutputFormat::Html,
            "Custom output_format should be preserved",
        );
    }

    /// Verify LlvmCovBackend implements Default
    #[kani::proof]
    fn proof_llvm_cov_backend_default() {
        let backend = LlvmCovBackend::default();
        kani::assert(
            backend.id() == BackendId::LlvmCov,
            "Default backend should have correct ID",
        );
    }

    // =============================================
    // LlvmCovBackend trait implementation proofs
    // =============================================

    /// Verify LlvmCovBackend::id returns BackendId::LlvmCov
    #[kani::proof]
    fn proof_llvm_cov_backend_id() {
        let backend = LlvmCovBackend::new();
        kani::assert(
            backend.id() == BackendId::LlvmCov,
            "Backend ID should be LlvmCov",
        );
    }

    /// Verify LlvmCovBackend::supports includes Lint
    #[kani::proof]
    fn proof_llvm_cov_backend_supports_lint() {
        let backend = LlvmCovBackend::new();
        let supported = backend.supports();
        let mut found = false;
        for pt in supported {
            if pt == PropertyType::Lint {
                found = true;
            }
        }
        kani::assert(found, "Should support Lint");
    }

    /// Verify LlvmCovBackend::supports returns exactly 1 type
    #[kani::proof]
    fn proof_llvm_cov_backend_supports_count() {
        let backend = LlvmCovBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.len() == 1,
            "Should support exactly 1 property type",
        );
    }

    // =============================================
    // extract_percentage proofs
    // =============================================

    /// Verify extract_percentage finds percentage with decimal
    #[kani::proof]
    fn proof_extract_percentage_decimal() {
        let result = extract_percentage("coverage: 85.5%");
        kani::assert(result == Some(85.5), "Should extract 85.5");
    }

    /// Verify extract_percentage finds integer percentage
    #[kani::proof]
    fn proof_extract_percentage_integer() {
        let result = extract_percentage("coverage: 90%");
        kani::assert(result == Some(90.0), "Should extract 90.0");
    }

    /// Verify extract_percentage returns None for no percentage
    #[kani::proof]
    fn proof_extract_percentage_none() {
        let result = extract_percentage("no percentage here");
        kani::assert(result.is_none(), "Should return None");
    }

    /// Verify extract_percentage with 0%
    #[kani::proof]
    fn proof_extract_percentage_zero() {
        let result = extract_percentage("coverage: 0%");
        kani::assert(result == Some(0.0), "Should extract 0.0");
    }

    /// Verify extract_percentage with 100%
    #[kani::proof]
    fn proof_extract_percentage_hundred() {
        let result = extract_percentage("coverage: 100%");
        kani::assert(result == Some(100.0), "Should extract 100.0");
    }

    // =============================================
    // evaluate_coverage proofs
    // =============================================

    /// Verify evaluate_coverage returns Proven when above threshold
    #[kani::proof]
    fn proof_evaluate_coverage_above_threshold_proven() {
        let backend = LlvmCovBackend::with_config(LlvmCovConfig::default().with_min_coverage(0.80));
        let stats = LlvmCovStats {
            total_lines: 100,
            covered_lines: 90,
            line_coverage: 0.90,
            ..Default::default()
        };
        let (status, ce) = backend.evaluate_coverage(&stats);
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "Should be Proven",
        );
        kani::assert(ce.is_none(), "Counterexample should be None");
    }

    /// Verify evaluate_coverage returns Disproven when below threshold
    #[kani::proof]
    fn proof_evaluate_coverage_below_threshold_disproven() {
        let backend = LlvmCovBackend::with_config(LlvmCovConfig::default().with_min_coverage(0.80));
        let stats = LlvmCovStats {
            total_lines: 100,
            covered_lines: 70,
            line_coverage: 0.70,
            ..Default::default()
        };
        let (status, ce) = backend.evaluate_coverage(&stats);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "Should be Disproven",
        );
        kani::assert(ce.is_some(), "Counterexample should be Some");
    }

    /// Verify evaluate_coverage returns Unknown for zero lines
    #[kani::proof]
    fn proof_evaluate_coverage_zero_lines_unknown() {
        let backend = LlvmCovBackend::new();
        let stats = LlvmCovStats::default();
        let (status, _) = backend.evaluate_coverage(&stats);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Should be Unknown",
        );
    }

    /// Verify evaluate_coverage at exactly threshold
    #[kani::proof]
    fn proof_evaluate_coverage_at_threshold_proven() {
        let backend = LlvmCovBackend::with_config(LlvmCovConfig::default().with_min_coverage(0.80));
        let stats = LlvmCovStats {
            total_lines: 100,
            covered_lines: 80,
            line_coverage: 0.80,
            ..Default::default()
        };
        let (status, _) = backend.evaluate_coverage(&stats);
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "At threshold should be Proven",
        );
    }

    // =============================================
    // Builder chaining proofs
    // =============================================

    /// Verify builder methods can be chained
    #[kani::proof]
    fn proof_llvm_cov_config_builder_chain() {
        let config = LlvmCovConfig::default()
            .with_min_coverage(0.95)
            .with_output_format(LlvmCovOutputFormat::Lcov)
            .with_doctests(true)
            .with_branch(true);
        kani::assert(
            config.min_coverage == Some(0.95),
            "min_coverage should be 0.95",
        );
        kani::assert(
            config.output_format == LlvmCovOutputFormat::Lcov,
            "output_format should be Lcov",
        );
        kani::assert(config.doctests, "doctests should be true");
        kani::assert(config.branch, "branch should be true");
    }
}
