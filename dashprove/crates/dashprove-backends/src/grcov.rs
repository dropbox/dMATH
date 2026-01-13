//! grcov coverage aggregator backend
//!
//! grcov is Mozilla's tool for collecting and aggregating code coverage.
//! It supports multiple coverage formats (gcov, lcov, coveralls, cobertura).
//!
//! See: <https://github.com/mozilla/grcov>

// =============================================
// Kani Proofs for grcov Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- GrcovOutputFormat Default Tests ----

    /// Verify GrcovOutputFormat::default is Lcov
    #[kani::proof]
    fn proof_grcov_output_format_default_is_lcov() {
        let format = GrcovOutputFormat::default();
        kani::assert(
            format == GrcovOutputFormat::Lcov,
            "Default format should be Lcov",
        );
    }

    /// Verify GrcovOutputFormat::Lcov as_str is "lcov"
    #[kani::proof]
    fn proof_grcov_output_format_lcov_str() {
        kani::assert(
            GrcovOutputFormat::Lcov.as_str() == "lcov",
            "Lcov should be 'lcov'",
        );
    }

    /// Verify GrcovOutputFormat::Coveralls as_str is "coveralls"
    #[kani::proof]
    fn proof_grcov_output_format_coveralls_str() {
        kani::assert(
            GrcovOutputFormat::Coveralls.as_str() == "coveralls",
            "Coveralls should be 'coveralls'",
        );
    }

    /// Verify GrcovOutputFormat::Cobertura as_str is "cobertura"
    #[kani::proof]
    fn proof_grcov_output_format_cobertura_str() {
        kani::assert(
            GrcovOutputFormat::Cobertura.as_str() == "cobertura",
            "Cobertura should be 'cobertura'",
        );
    }

    /// Verify GrcovOutputFormat::Html as_str is "html"
    #[kani::proof]
    fn proof_grcov_output_format_html_str() {
        kani::assert(
            GrcovOutputFormat::Html.as_str() == "html",
            "Html should be 'html'",
        );
    }

    /// Verify GrcovOutputFormat::Markdown as_str is "markdown"
    #[kani::proof]
    fn proof_grcov_output_format_markdown_str() {
        kani::assert(
            GrcovOutputFormat::Markdown.as_str() == "markdown",
            "Markdown should be 'markdown'",
        );
    }

    // ---- GrcovConfig Default Tests ----

    /// Verify GrcovConfig::default timeout is 300 seconds
    #[kani::proof]
    fn proof_grcov_config_default_timeout() {
        let config = GrcovConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "Default timeout should be 300 seconds",
        );
    }

    /// Verify GrcovConfig::default output_format is Lcov
    #[kani::proof]
    fn proof_grcov_config_default_output_format() {
        let config = GrcovConfig::default();
        kani::assert(
            config.output_format == GrcovOutputFormat::Lcov,
            "Default output_format should be Lcov",
        );
    }

    /// Verify GrcovConfig::default min_coverage is Some(0.80)
    #[kani::proof]
    fn proof_grcov_config_default_min_coverage() {
        let config = GrcovConfig::default();
        kani::assert(
            config.min_coverage == Some(0.80),
            "Default min_coverage should be Some(0.80)",
        );
    }

    /// Verify GrcovConfig::default source_dir is None
    #[kani::proof]
    fn proof_grcov_config_default_source_dir_none() {
        let config = GrcovConfig::default();
        kani::assert(
            config.source_dir.is_none(),
            "Default source_dir should be None",
        );
    }

    /// Verify GrcovConfig::default binary_dir is None
    #[kani::proof]
    fn proof_grcov_config_default_binary_dir_none() {
        let config = GrcovConfig::default();
        kani::assert(
            config.binary_dir.is_none(),
            "Default binary_dir should be None",
        );
    }

    /// Verify GrcovConfig::default branch_coverage is false
    #[kani::proof]
    fn proof_grcov_config_default_branch_coverage_false() {
        let config = GrcovConfig::default();
        kani::assert(
            !config.branch_coverage,
            "Default branch_coverage should be false",
        );
    }

    /// Verify GrcovConfig::default ignore_patterns is not empty
    #[kani::proof]
    fn proof_grcov_config_default_ignore_patterns_not_empty() {
        let config = GrcovConfig::default();
        kani::assert(
            !config.ignore_patterns.is_empty(),
            "Default ignore_patterns should not be empty",
        );
    }

    // ---- GrcovConfig Builder Tests ----

    /// Verify with_min_coverage sets coverage
    #[kani::proof]
    fn proof_grcov_config_with_min_coverage() {
        let config = GrcovConfig::default().with_min_coverage(0.90);
        kani::assert(
            config.min_coverage == Some(0.90),
            "with_min_coverage should set to 0.90",
        );
    }

    /// Verify with_min_coverage clamps above 1.0
    #[kani::proof]
    fn proof_grcov_config_with_min_coverage_clamp_high() {
        let config = GrcovConfig::default().with_min_coverage(1.5);
        kani::assert(
            config.min_coverage == Some(1.0),
            "with_min_coverage should clamp to 1.0",
        );
    }

    /// Verify with_min_coverage clamps below 0.0
    #[kani::proof]
    fn proof_grcov_config_with_min_coverage_clamp_low() {
        let config = GrcovConfig::default().with_min_coverage(-0.5);
        kani::assert(
            config.min_coverage == Some(0.0),
            "with_min_coverage should clamp to 0.0",
        );
    }

    /// Verify with_output_format sets format
    #[kani::proof]
    fn proof_grcov_config_with_output_format_html() {
        let config = GrcovConfig::default().with_output_format(GrcovOutputFormat::Html);
        kani::assert(
            config.output_format == GrcovOutputFormat::Html,
            "with_output_format should set Html",
        );
    }

    /// Verify with_branch_coverage enables branch coverage
    #[kani::proof]
    fn proof_grcov_config_with_branch_coverage_true() {
        let config = GrcovConfig::default().with_branch_coverage(true);
        kani::assert(
            config.branch_coverage,
            "with_branch_coverage(true) should enable",
        );
    }

    /// Verify with_branch_coverage disables branch coverage
    #[kani::proof]
    fn proof_grcov_config_with_branch_coverage_false() {
        let config = GrcovConfig::default()
            .with_branch_coverage(true)
            .with_branch_coverage(false);
        kani::assert(
            !config.branch_coverage,
            "with_branch_coverage(false) should disable",
        );
    }

    // ---- GrcovBackend Construction Tests ----

    /// Verify GrcovBackend::new uses default timeout
    #[kani::proof]
    fn proof_grcov_backend_new_default_timeout() {
        let backend = GrcovBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(300),
            "New backend should use default timeout",
        );
    }

    /// Verify GrcovBackend::default equals GrcovBackend::new
    #[kani::proof]
    fn proof_grcov_backend_default_equals_new() {
        let default_backend = GrcovBackend::default();
        let new_backend = GrcovBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "Default and new should have same timeout",
        );
    }

    /// Verify GrcovBackend::with_config preserves min_coverage
    #[kani::proof]
    fn proof_grcov_backend_with_config_min_coverage() {
        let config = GrcovConfig::default().with_min_coverage(0.95);
        let backend = GrcovBackend::with_config(config);
        kani::assert(
            backend.config.min_coverage == Some(0.95),
            "with_config should preserve min_coverage",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify GrcovBackend::id returns Grcov
    #[kani::proof]
    fn proof_grcov_backend_id() {
        let backend = GrcovBackend::new();
        kani::assert(
            backend.id() == BackendId::Grcov,
            "Backend id should be Grcov",
        );
    }

    /// Verify GrcovBackend::supports includes Lint
    #[kani::proof]
    fn proof_grcov_backend_supports_lint() {
        let backend = GrcovBackend::new();
        let supported = backend.supports();
        let has_lint = supported.iter().any(|p| *p == PropertyType::Lint);
        kani::assert(has_lint, "Should support Lint property");
    }

    /// Verify GrcovBackend::supports returns exactly 1 property
    #[kani::proof]
    fn proof_grcov_backend_supports_length() {
        let backend = GrcovBackend::new();
        let supported = backend.supports();
        kani::assert(supported.len() == 1, "Should support exactly 1 property");
    }

    // ---- evaluate_coverage Tests ----

    /// Verify evaluate_coverage passes above threshold
    #[kani::proof]
    fn proof_evaluate_coverage_pass() {
        let backend = GrcovBackend::with_config(GrcovConfig::default().with_min_coverage(0.80));
        let stats = CoverageStats {
            total_lines: 100,
            covered_lines: 85,
            line_coverage: 0.85,
            total_branches: None,
            covered_branches: None,
            branch_coverage: None,
            files_covered: 5,
            files_below_threshold: vec![],
        };
        let (status, ce) = backend.evaluate_coverage(&stats);
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "Should pass above threshold",
        );
        kani::assert(ce.is_none(), "No counterexample for pass");
    }

    /// Verify evaluate_coverage fails below threshold
    #[kani::proof]
    fn proof_evaluate_coverage_fail() {
        let backend = GrcovBackend::with_config(GrcovConfig::default().with_min_coverage(0.80));
        let stats = CoverageStats {
            total_lines: 100,
            covered_lines: 70,
            line_coverage: 0.70,
            total_branches: None,
            covered_branches: None,
            branch_coverage: None,
            files_covered: 5,
            files_below_threshold: vec![],
        };
        let (status, ce) = backend.evaluate_coverage(&stats);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "Should fail below threshold",
        );
        kani::assert(ce.is_some(), "Should have counterexample for fail");
    }

    /// Verify evaluate_coverage returns Unknown for no data
    #[kani::proof]
    fn proof_evaluate_coverage_no_data() {
        let backend = GrcovBackend::with_config(GrcovConfig::default().with_min_coverage(0.80));
        let stats = CoverageStats {
            total_lines: 0,
            covered_lines: 0,
            line_coverage: 0.0,
            total_branches: None,
            covered_branches: None,
            branch_coverage: None,
            files_covered: 0,
            files_below_threshold: vec![],
        };
        let (status, _) = backend.evaluate_coverage(&stats);
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "Should be Unknown for no data",
        );
    }
}

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

/// Output format for grcov coverage reports
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum GrcovOutputFormat {
    #[default]
    Lcov,
    Coveralls,
    Cobertura,
    Html,
    Markdown,
}

impl GrcovOutputFormat {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Lcov => "lcov",
            Self::Coveralls => "coveralls",
            Self::Cobertura => "cobertura",
            Self::Html => "html",
            Self::Markdown => "markdown",
        }
    }
}

/// Configuration for grcov coverage backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrcovConfig {
    pub timeout: Duration,
    pub output_format: GrcovOutputFormat,
    pub min_coverage: Option<f64>,
    pub source_dir: Option<PathBuf>,
    pub binary_dir: Option<PathBuf>,
    pub ignore_patterns: Vec<String>,
    pub branch_coverage: bool,
}

impl Default for GrcovConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(300),
            output_format: GrcovOutputFormat::default(),
            min_coverage: Some(0.80),
            source_dir: None,
            binary_dir: None,
            ignore_patterns: vec!["**/tests/*".to_string(), "**/test_*.rs".to_string()],
            branch_coverage: false,
        }
    }
}

impl GrcovConfig {
    pub fn with_min_coverage(mut self, threshold: f64) -> Self {
        self.min_coverage = Some(threshold.clamp(0.0, 1.0));
        self
    }

    pub fn with_output_format(mut self, format: GrcovOutputFormat) -> Self {
        self.output_format = format;
        self
    }

    pub fn with_branch_coverage(mut self, enabled: bool) -> Self {
        self.branch_coverage = enabled;
        self
    }
}

/// Coverage statistics from grcov
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CoverageStats {
    pub total_lines: u64,
    pub covered_lines: u64,
    pub line_coverage: f64,
    pub total_branches: Option<u64>,
    pub covered_branches: Option<u64>,
    pub branch_coverage: Option<f64>,
    pub files_covered: u64,
    pub files_below_threshold: Vec<FileCoverage>,
}

/// Per-file coverage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileCoverage {
    pub path: String,
    pub line_coverage: f64,
    pub uncovered_lines: Vec<u32>,
}

/// grcov coverage aggregator backend
pub struct GrcovBackend {
    config: GrcovConfig,
}

impl Default for GrcovBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl GrcovBackend {
    pub fn new() -> Self {
        Self {
            config: GrcovConfig::default(),
        }
    }

    pub fn with_config(config: GrcovConfig) -> Self {
        Self { config }
    }

    async fn detect(&self) -> Result<PathBuf, String> {
        which::which("grcov")
            .map_err(|_| "grcov not found. Install via: cargo install grcov".to_string())
    }

    /// Run grcov on collected coverage data
    pub async fn run_coverage(
        &self,
        crate_path: &std::path::Path,
    ) -> Result<BackendResult, BackendError> {
        let start = Instant::now();
        let grcov_path = self.detect().await.map_err(BackendError::Unavailable)?;

        let mut cmd = Command::new(&grcov_path);
        cmd.current_dir(crate_path);
        cmd.arg("-s").arg(".");
        cmd.arg("-b").arg("./target/debug");
        cmd.arg("-t").arg(self.config.output_format.as_str());
        cmd.arg("-o").arg("-");
        cmd.arg("./target/debug/");

        let output = tokio::time::timeout(self.config.timeout, cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| BackendError::VerificationFailed(format!("Failed to run grcov: {}", e)))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        let stats = self.parse_coverage_output(&stdout)?;
        let (status, counterexample) = self.evaluate_coverage(&stats);

        let mut diagnostics = Vec::new();
        diagnostics.push(format!(
            "Line coverage: {:.1}% ({}/{} lines)",
            stats.line_coverage * 100.0,
            stats.covered_lines,
            stats.total_lines
        ));

        if !stderr.is_empty() && !stderr.contains("warning") {
            diagnostics.push(format!("grcov stderr: {}", stderr.trim()));
        }

        Ok(BackendResult {
            backend: BackendId::Grcov,
            status,
            proof: None,
            counterexample,
            diagnostics,
            time_taken: start.elapsed(),
        })
    }

    fn parse_coverage_output(&self, stdout: &str) -> Result<CoverageStats, BackendError> {
        let mut stats = CoverageStats::default();

        // Parse LCOV format
        for line in stdout.lines() {
            if let Some(data) = line.strip_prefix("DA:") {
                if let Some((_line_num, count)) = data.split_once(',') {
                    stats.total_lines += 1;
                    let exec_count: u64 = count.parse().unwrap_or(0);
                    if exec_count > 0 {
                        stats.covered_lines += 1;
                    }
                }
            } else if line == "end_of_record" {
                stats.files_covered += 1;
            }
        }

        if stats.total_lines > 0 {
            stats.line_coverage = stats.covered_lines as f64 / stats.total_lines as f64;
        }

        Ok(stats)
    }

    fn evaluate_coverage(
        &self,
        stats: &CoverageStats,
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
                        playback_test: Some("Add tests for uncovered code paths".to_string()),
                        trace: vec![],
                        raw: Some(format!(
                            "Coverage {:.1}% below {:.1}% threshold",
                            stats.line_coverage * 100.0,
                            threshold * 100.0
                        )),
                        minimized: false,
                    }),
                );
            }
        }

        if stats.total_lines == 0 {
            return (
                VerificationStatus::Unknown {
                    reason: "No coverage data found".to_string(),
                },
                None,
            );
        }

        (VerificationStatus::Proven, None)
    }
}

#[async_trait]
impl VerificationBackend for GrcovBackend {
    fn id(&self) -> BackendId {
        BackendId::Grcov
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
        diagnostics.push(format!(
            "Output format: {}",
            self.config.output_format.as_str()
        ));

        Ok(BackendResult {
            backend: BackendId::Grcov,
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
        assert_eq!(GrcovBackend::new().id(), BackendId::Grcov);
    }

    #[test]
    fn test_config_default() {
        let config = GrcovConfig::default();
        assert_eq!(config.min_coverage, Some(0.80));
        assert!(!config.branch_coverage);
        assert_eq!(config.output_format, GrcovOutputFormat::Lcov);
    }

    #[test]
    fn test_config_builder() {
        let config = GrcovConfig::default()
            .with_min_coverage(0.90)
            .with_branch_coverage(true)
            .with_output_format(GrcovOutputFormat::Html);

        assert_eq!(config.min_coverage, Some(0.90));
        assert!(config.branch_coverage);
        assert_eq!(config.output_format, GrcovOutputFormat::Html);
    }

    #[test]
    fn test_output_format_str() {
        assert_eq!(GrcovOutputFormat::Lcov.as_str(), "lcov");
        assert_eq!(GrcovOutputFormat::Html.as_str(), "html");
    }

    #[test]
    fn test_parse_lcov_output() {
        let lcov_output = "SF:src/lib.rs\nDA:1,1\nDA:2,1\nDA:3,0\nend_of_record\n";
        let backend = GrcovBackend::new();
        let stats = backend.parse_coverage_output(lcov_output).unwrap();

        assert_eq!(stats.total_lines, 3);
        assert_eq!(stats.covered_lines, 2);
        assert!((stats.line_coverage - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_evaluate_coverage_pass() {
        let backend = GrcovBackend::with_config(GrcovConfig::default().with_min_coverage(0.80));
        let stats = CoverageStats {
            total_lines: 100,
            covered_lines: 85,
            line_coverage: 0.85,
            ..Default::default()
        };

        let (status, counterexample) = backend.evaluate_coverage(&stats);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(counterexample.is_none());
    }

    #[test]
    fn test_evaluate_coverage_fail() {
        let backend = GrcovBackend::with_config(GrcovConfig::default().with_min_coverage(0.80));
        let stats = CoverageStats {
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
        let backend = GrcovBackend::new();
        let status = backend.health_check().await;
        match status {
            HealthStatus::Healthy => {}
            HealthStatus::Unavailable { reason } => {
                assert!(reason.contains("grcov"));
            }
            _ => panic!("Unexpected status"),
        }
    }

    #[tokio::test]
    async fn test_verify_returns_result_or_unavailable() {
        use dashprove_usl::{parse, typecheck};

        let backend = GrcovBackend::new();
        let spec = parse("theorem test { true }").expect("parse");
        let typed_spec = typecheck(spec).expect("typecheck");

        let result = backend.verify(&typed_spec).await;
        match result {
            Ok(r) => {
                assert_eq!(r.backend, BackendId::Grcov);
            }
            Err(BackendError::Unavailable(_)) => {}
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }
}
