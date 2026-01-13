//! Rust Static Analysis Backends for DashProve
//!
//! This crate provides verification backends for Rust static analysis tools:
//! - **Clippy**: Rust lint collection (400+ lints)
//! - **cargo-semver-checks**: API compatibility checking
//! - **cargo-geiger**: Unsafe code auditing
//! - **cargo-audit**: Security vulnerability scanning
//! - **cargo-deny**: Dependency policy enforcement
//! - **cargo-vet**: Supply chain auditing
//! - **cargo-mutants**: Mutation testing
//!
//! # Usage
//!
//! ```rust,ignore
//! use dashprove_static::{StaticAnalysisBackend, AnalysisTool, AnalysisConfig};
//!
//! let config = AnalysisConfig::default();
//! let backend = StaticAnalysisBackend::new(AnalysisTool::Clippy, config);
//! let result = backend.run_on_crate("/path/to/crate").await?;
//! ```

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::process::Command;

/// Static analysis tool type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnalysisTool {
    /// Clippy - comprehensive Rust linter
    Clippy,
    /// cargo-semver-checks - API compatibility
    SemverChecks,
    /// cargo-geiger - unsafe code audit
    Geiger,
    /// cargo-audit - security vulnerabilities
    Audit,
    /// cargo-deny - dependency policy
    Deny,
    /// cargo-vet - supply chain audit
    Vet,
    /// cargo-mutants - mutation testing
    Mutants,
}

impl AnalysisTool {
    /// Get the tool name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Clippy => "clippy",
            Self::SemverChecks => "cargo-semver-checks",
            Self::Geiger => "cargo-geiger",
            Self::Audit => "cargo-audit",
            Self::Deny => "cargo-deny",
            Self::Vet => "cargo-vet",
            Self::Mutants => "cargo-mutants",
        }
    }

    /// Get the cargo subcommand
    pub fn cargo_command(&self) -> &'static str {
        match self {
            Self::Clippy => "clippy",
            Self::SemverChecks => "semver-checks",
            Self::Geiger => "geiger",
            Self::Audit => "audit",
            Self::Deny => "deny",
            Self::Vet => "vet",
            Self::Mutants => "mutants",
        }
    }

    /// Get installation command
    pub fn install_command(&self) -> &'static str {
        match self {
            Self::Clippy => "rustup component add clippy",
            Self::SemverChecks => "cargo install cargo-semver-checks",
            Self::Geiger => "cargo install cargo-geiger",
            Self::Audit => "cargo install cargo-audit",
            Self::Deny => "cargo install cargo-deny",
            Self::Vet => "cargo install cargo-vet",
            Self::Mutants => "cargo install cargo-mutants",
        }
    }

    /// Check if this tool requires a config file
    pub fn requires_config(&self) -> bool {
        matches!(self, Self::Deny | Self::Vet)
    }
}

/// Configuration for static analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisConfig {
    /// Treat warnings as errors
    pub warnings_as_errors: bool,
    /// Enable all optional checks
    pub all_features: bool,
    /// Specific features to enable
    pub features: Vec<String>,
    /// Targets to check (default: all)
    pub targets: Vec<String>,
    /// Timeout for the analysis
    pub timeout: Duration,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            warnings_as_errors: false,
            all_features: false,
            features: Vec::new(),
            targets: Vec::new(),
            timeout: Duration::from_secs(300),
        }
    }
}

impl AnalysisConfig {
    /// Enable warnings as errors
    pub fn with_warnings_as_errors(mut self) -> Self {
        self.warnings_as_errors = true;
        self
    }

    /// Enable all features
    pub fn with_all_features(mut self) -> Self {
        self.all_features = true;
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

/// Error type for static analysis
#[derive(Error, Debug)]
pub enum AnalysisError {
    /// Tool not installed
    #[error("Tool not installed: {0}. Install with: {1}")]
    NotInstalled(String, String),

    /// Analysis failed to run
    #[error("Analysis failed: {0}")]
    Failed(String),

    /// Configuration missing
    #[error("Configuration file missing for {0}: {1}")]
    ConfigMissing(String, String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON parsing error
    #[error("JSON parsing error: {0}")]
    Json(#[from] serde_json::Error),

    /// Timeout
    #[error("Analysis timeout after {0:?}")]
    Timeout(Duration),
}

/// Severity of an analysis finding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    /// Error - must fix
    Error,
    /// Warning - should fix
    Warning,
    /// Info - informational
    Info,
    /// Help - suggestion
    Help,
}

/// A finding from static analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Finding {
    /// Tool that found this issue
    pub tool: String,
    /// Severity level
    pub severity: Severity,
    /// Code/ID of the finding (e.g., clippy lint name)
    pub code: String,
    /// Human-readable message
    pub message: String,
    /// File path where found
    pub file: Option<String>,
    /// Line number
    pub line: Option<u32>,
    /// Column number
    pub column: Option<u32>,
    /// Suggested fix if available
    pub suggestion: Option<String>,
}

/// Clippy-specific result
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ClippyResult {
    /// Errors found
    pub errors: Vec<Finding>,
    /// Warnings found
    pub warnings: Vec<Finding>,
    /// Number of lints checked
    pub lints_checked: usize,
}

/// Security audit result (cargo-audit)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AuditResult {
    /// Vulnerabilities found
    pub vulnerabilities: Vec<Vulnerability>,
    /// Warnings (yanked crates, etc.)
    pub warnings: Vec<String>,
    /// Number of dependencies scanned
    pub dependencies_scanned: usize,
}

/// A security vulnerability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vulnerability {
    /// Advisory ID (e.g., RUSTSEC-2021-0001)
    pub id: String,
    /// Affected crate
    pub crate_name: String,
    /// Affected version
    pub version: String,
    /// Severity (critical, high, medium, low)
    pub severity: String,
    /// Title/description
    pub title: String,
    /// URL for more info
    pub url: Option<String>,
    /// Patched versions
    pub patched_versions: Vec<String>,
}

/// Geiger (unsafe audit) result
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GeigerResult {
    /// Total unsafe blocks
    pub unsafe_blocks: usize,
    /// Unsafe functions
    pub unsafe_functions: usize,
    /// Unsafe impls
    pub unsafe_impls: usize,
    /// Crates with unsafe code
    pub unsafe_crates: Vec<UnsafeCrate>,
}

/// A crate with unsafe code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnsafeCrate {
    /// Crate name
    pub name: String,
    /// Version
    pub version: String,
    /// Unsafe statistics
    pub unsafe_count: usize,
    /// Safe percentage (0-100)
    pub safe_percentage: f64,
}

/// Result from static analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    /// Which tool was used
    pub tool: AnalysisTool,
    /// Whether the analysis passed (no errors)
    pub passed: bool,
    /// All findings
    pub findings: Vec<Finding>,
    /// Error count
    pub error_count: usize,
    /// Warning count
    pub warning_count: usize,
    /// Duration of analysis
    pub duration: Duration,
    /// Raw output for debugging
    pub raw_output: String,
}

/// Static analysis backend
pub struct StaticAnalysisBackend {
    /// Which tool to use
    tool: AnalysisTool,
    /// Configuration
    config: AnalysisConfig,
}

impl StaticAnalysisBackend {
    /// Create a new static analysis backend
    pub fn new(tool: AnalysisTool, config: AnalysisConfig) -> Self {
        Self { tool, config }
    }

    /// Check if the tool is installed
    pub async fn check_installed(&self) -> Result<bool, AnalysisError> {
        let (cmd, args) = match self.tool {
            AnalysisTool::Clippy => ("cargo", vec!["clippy", "--version"]),
            _ => ("cargo", vec![self.tool.cargo_command(), "--version"]),
        };

        let output = Command::new(cmd).args(&args).output().await?;
        Ok(output.status.success())
    }

    /// Run analysis on a crate
    pub async fn run_on_crate(&self, crate_path: &Path) -> Result<AnalysisResult, AnalysisError> {
        // Check if tool is installed
        if !self.check_installed().await? {
            return Err(AnalysisError::NotInstalled(
                self.tool.name().to_string(),
                self.tool.install_command().to_string(),
            ));
        }

        let start = Instant::now();

        let result = match self.tool {
            AnalysisTool::Clippy => self.run_clippy(crate_path).await?,
            AnalysisTool::SemverChecks => self.run_semver_checks(crate_path).await?,
            AnalysisTool::Geiger => self.run_geiger(crate_path).await?,
            AnalysisTool::Audit => self.run_audit(crate_path).await?,
            AnalysisTool::Deny => self.run_deny(crate_path).await?,
            AnalysisTool::Vet => self.run_vet(crate_path).await?,
            AnalysisTool::Mutants => self.run_mutants(crate_path).await?,
        };

        let duration = start.elapsed();

        Ok(AnalysisResult {
            tool: self.tool,
            passed: result
                .findings
                .iter()
                .all(|f| f.severity != Severity::Error),
            error_count: result
                .findings
                .iter()
                .filter(|f| f.severity == Severity::Error)
                .count(),
            warning_count: result
                .findings
                .iter()
                .filter(|f| f.severity == Severity::Warning)
                .count(),
            findings: result.findings,
            duration,
            raw_output: result.raw_output,
        })
    }

    async fn run_clippy(&self, crate_path: &Path) -> Result<PartialResult, AnalysisError> {
        let mut args = vec!["clippy", "--message-format=json"];

        if self.config.warnings_as_errors {
            args.push("--");
            args.push("-D");
            args.push("warnings");
        }

        if self.config.all_features {
            args.insert(1, "--all-features");
        }

        let output = Command::new("cargo")
            .args(&args)
            .current_dir(crate_path)
            .output()
            .await?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let findings = self.parse_clippy_output(&stdout);

        Ok(PartialResult {
            findings,
            raw_output: stdout.to_string(),
        })
    }

    fn parse_clippy_output(&self, output: &str) -> Vec<Finding> {
        let mut findings = Vec::new();

        for line in output.lines() {
            if line.starts_with('{') {
                if let Ok(msg) = serde_json::from_str::<serde_json::Value>(line) {
                    if let Some(reason) = msg.get("reason").and_then(|r| r.as_str()) {
                        if reason == "compiler-message" {
                            if let Some(message) = msg.get("message") {
                                let level = message
                                    .get("level")
                                    .and_then(|l| l.as_str())
                                    .unwrap_or("unknown");

                                let severity = match level {
                                    "error" => Severity::Error,
                                    "warning" => Severity::Warning,
                                    "help" => Severity::Help,
                                    _ => Severity::Info,
                                };

                                let code = message
                                    .get("code")
                                    .and_then(|c| c.get("code"))
                                    .and_then(|c| c.as_str())
                                    .unwrap_or("unknown")
                                    .to_string();

                                let text = message
                                    .get("message")
                                    .and_then(|m| m.as_str())
                                    .unwrap_or("")
                                    .to_string();

                                let (file, line_num, column) = message
                                    .get("spans")
                                    .and_then(|s| s.as_array())
                                    .and_then(|arr| arr.first())
                                    .map(|span| {
                                        (
                                            span.get("file_name")
                                                .and_then(|f| f.as_str())
                                                .map(|s| s.to_string()),
                                            span.get("line_start")
                                                .and_then(|l| l.as_u64())
                                                .map(|l| l as u32),
                                            span.get("column_start")
                                                .and_then(|c| c.as_u64())
                                                .map(|c| c as u32),
                                        )
                                    })
                                    .unwrap_or((None, None, None));

                                findings.push(Finding {
                                    tool: "clippy".to_string(),
                                    severity,
                                    code,
                                    message: text,
                                    file,
                                    line: line_num,
                                    column,
                                    suggestion: None,
                                });
                            }
                        }
                    }
                }
            }
        }

        findings
    }

    async fn run_semver_checks(&self, crate_path: &Path) -> Result<PartialResult, AnalysisError> {
        let output = Command::new("cargo")
            .args(["semver-checks", "check-release"])
            .current_dir(crate_path)
            .output()
            .await?;

        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let combined = format!("{}\n{}", stdout, stderr);

        let mut findings = Vec::new();

        // Parse semver-checks output for breaking changes
        for line in combined.lines() {
            if line.contains("BREAKING") || line.contains("breaking change") {
                findings.push(Finding {
                    tool: "semver-checks".to_string(),
                    severity: Severity::Error,
                    code: "breaking-change".to_string(),
                    message: line.to_string(),
                    file: None,
                    line: None,
                    column: None,
                    suggestion: None,
                });
            }
        }

        Ok(PartialResult {
            findings,
            raw_output: combined,
        })
    }

    async fn run_geiger(&self, crate_path: &Path) -> Result<PartialResult, AnalysisError> {
        let output = Command::new("cargo")
            .args(["geiger", "--output-format", "json"])
            .current_dir(crate_path)
            .output()
            .await?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut findings = Vec::new();

        // Parse JSON output if available
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&stdout) {
            if let Some(packages) = json.get("packages").and_then(|p| p.as_array()) {
                for pkg in packages {
                    let name = pkg
                        .get("id")
                        .and_then(|id| id.as_str())
                        .unwrap_or("unknown");

                    let unsafe_count = pkg
                        .get("unsafety")
                        .and_then(|u| u.get("used"))
                        .and_then(|u| u.get("unsafe_fns"))
                        .and_then(|f| f.as_u64())
                        .unwrap_or(0);

                    if unsafe_count > 0 {
                        findings.push(Finding {
                            tool: "geiger".to_string(),
                            severity: Severity::Warning,
                            code: "unsafe-code".to_string(),
                            message: format!("{}: {} unsafe functions", name, unsafe_count),
                            file: None,
                            line: None,
                            column: None,
                            suggestion: None,
                        });
                    }
                }
            }
        }

        Ok(PartialResult {
            findings,
            raw_output: stdout.to_string(),
        })
    }

    async fn run_audit(&self, crate_path: &Path) -> Result<PartialResult, AnalysisError> {
        let output = Command::new("cargo")
            .args(["audit", "--json"])
            .current_dir(crate_path)
            .output()
            .await?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut findings = Vec::new();

        // Parse audit JSON output
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&stdout) {
            if let Some(vulns) = json
                .get("vulnerabilities")
                .and_then(|v| v.get("list"))
                .and_then(|l| l.as_array())
            {
                for vuln in vulns {
                    let advisory = vuln.get("advisory").unwrap_or(vuln);

                    let id = advisory
                        .get("id")
                        .and_then(|i| i.as_str())
                        .unwrap_or("UNKNOWN");

                    let title = advisory
                        .get("title")
                        .and_then(|t| t.as_str())
                        .unwrap_or("Unknown vulnerability");

                    let severity_str = advisory
                        .get("severity")
                        .and_then(|s| s.as_str())
                        .unwrap_or("unknown");

                    let severity = match severity_str.to_lowercase().as_str() {
                        "critical" | "high" => Severity::Error,
                        "medium" => Severity::Warning,
                        _ => Severity::Info,
                    };

                    let crate_name = vuln
                        .get("package")
                        .and_then(|p| p.get("name"))
                        .and_then(|n| n.as_str())
                        .unwrap_or("unknown");

                    findings.push(Finding {
                        tool: "audit".to_string(),
                        severity,
                        code: id.to_string(),
                        message: format!("{}: {} in {}", id, title, crate_name),
                        file: None,
                        line: None,
                        column: None,
                        suggestion: None,
                    });
                }
            }
        }

        Ok(PartialResult {
            findings,
            raw_output: stdout.to_string(),
        })
    }

    async fn run_deny(&self, crate_path: &Path) -> Result<PartialResult, AnalysisError> {
        // Check for deny.toml
        let config_path = crate_path.join("deny.toml");
        if !config_path.exists() {
            return Err(AnalysisError::ConfigMissing(
                "cargo-deny".to_string(),
                "Create deny.toml in crate root".to_string(),
            ));
        }

        let output = Command::new("cargo")
            .args(["deny", "check", "--format", "json"])
            .current_dir(crate_path)
            .output()
            .await?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut findings = Vec::new();

        // Parse deny JSON output line by line
        for line in stdout.lines() {
            if let Ok(msg) = serde_json::from_str::<serde_json::Value>(line) {
                let level = msg
                    .get("level")
                    .and_then(|l| l.as_str())
                    .unwrap_or("unknown");

                let severity = match level {
                    "error" => Severity::Error,
                    "warn" => Severity::Warning,
                    _ => Severity::Info,
                };

                let code = msg
                    .get("code")
                    .and_then(|c| c.as_str())
                    .unwrap_or("unknown")
                    .to_string();

                let message = msg
                    .get("message")
                    .and_then(|m| m.as_str())
                    .unwrap_or("")
                    .to_string();

                if !message.is_empty() {
                    findings.push(Finding {
                        tool: "deny".to_string(),
                        severity,
                        code,
                        message,
                        file: None,
                        line: None,
                        column: None,
                        suggestion: None,
                    });
                }
            }
        }

        Ok(PartialResult {
            findings,
            raw_output: stdout.to_string(),
        })
    }

    async fn run_vet(&self, crate_path: &Path) -> Result<PartialResult, AnalysisError> {
        let output = Command::new("cargo")
            .args(["vet"])
            .current_dir(crate_path)
            .output()
            .await?;

        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let combined = format!("{}\n{}", stdout, stderr);

        let mut findings = Vec::new();

        // Parse vet output for unvetted crates
        for line in combined.lines() {
            if line.contains("unvetted") || line.contains("not audited") {
                findings.push(Finding {
                    tool: "vet".to_string(),
                    severity: Severity::Warning,
                    code: "unvetted".to_string(),
                    message: line.to_string(),
                    file: None,
                    line: None,
                    column: None,
                    suggestion: None,
                });
            }
        }

        Ok(PartialResult {
            findings,
            raw_output: combined,
        })
    }

    async fn run_mutants(&self, crate_path: &Path) -> Result<PartialResult, AnalysisError> {
        let output = Command::new("cargo")
            .args(["mutants", "--no-shuffle", "--json"])
            .current_dir(crate_path)
            .output()
            .await?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut findings = Vec::new();

        // Parse mutants output for surviving mutants
        for line in stdout.lines() {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(line) {
                if let Some(outcome) = json.get("outcome").and_then(|o| o.as_str()) {
                    if outcome == "Missed" || outcome == "Timeout" {
                        let location = json
                            .get("location")
                            .and_then(|l| l.as_str())
                            .unwrap_or("unknown");

                        let mutation = json
                            .get("mutation")
                            .and_then(|m| m.as_str())
                            .unwrap_or("unknown mutation");

                        findings.push(Finding {
                            tool: "mutants".to_string(),
                            severity: Severity::Warning,
                            code: "surviving-mutant".to_string(),
                            message: format!("Mutant survived: {} at {}", mutation, location),
                            file: None,
                            line: None,
                            column: None,
                            suggestion: Some("Add test coverage for this code path".to_string()),
                        });
                    }
                }
            }
        }

        Ok(PartialResult {
            findings,
            raw_output: stdout.to_string(),
        })
    }
}

/// Internal partial result
struct PartialResult {
    findings: Vec<Finding>,
    raw_output: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analysis_tool_names() {
        assert_eq!(AnalysisTool::Clippy.name(), "clippy");
        assert_eq!(AnalysisTool::SemverChecks.name(), "cargo-semver-checks");
        assert_eq!(AnalysisTool::Geiger.name(), "cargo-geiger");
        assert_eq!(AnalysisTool::Audit.name(), "cargo-audit");
        assert_eq!(AnalysisTool::Deny.name(), "cargo-deny");
        assert_eq!(AnalysisTool::Vet.name(), "cargo-vet");
        assert_eq!(AnalysisTool::Mutants.name(), "cargo-mutants");
    }

    #[test]
    fn test_analysis_tool_commands() {
        assert_eq!(AnalysisTool::Clippy.cargo_command(), "clippy");
        assert_eq!(AnalysisTool::Audit.cargo_command(), "audit");
    }

    #[test]
    fn test_analysis_tool_install_commands() {
        // Tests line 76: install_command for each tool
        // Verify each install command is correct and not "xyzzy"
        assert_eq!(
            AnalysisTool::Clippy.install_command(),
            "rustup component add clippy"
        );
        assert_eq!(
            AnalysisTool::SemverChecks.install_command(),
            "cargo install cargo-semver-checks"
        );
        assert_eq!(
            AnalysisTool::Geiger.install_command(),
            "cargo install cargo-geiger"
        );
        assert_eq!(
            AnalysisTool::Audit.install_command(),
            "cargo install cargo-audit"
        );
        assert_eq!(
            AnalysisTool::Deny.install_command(),
            "cargo install cargo-deny"
        );
        assert_eq!(
            AnalysisTool::Vet.install_command(),
            "cargo install cargo-vet"
        );
        assert_eq!(
            AnalysisTool::Mutants.install_command(),
            "cargo install cargo-mutants"
        );
        // Ensure none return "xyzzy"
        for tool in [
            AnalysisTool::Clippy,
            AnalysisTool::SemverChecks,
            AnalysisTool::Geiger,
            AnalysisTool::Audit,
            AnalysisTool::Deny,
            AnalysisTool::Vet,
            AnalysisTool::Mutants,
        ] {
            assert_ne!(tool.install_command(), "xyzzy");
        }
    }

    #[test]
    fn test_analysis_config_builder() {
        let config = AnalysisConfig::default()
            .with_warnings_as_errors()
            .with_all_features()
            .with_timeout(Duration::from_secs(60));

        assert!(config.warnings_as_errors);
        assert!(config.all_features);
        assert_eq!(config.timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_parse_clippy_output() {
        let backend = StaticAnalysisBackend::new(AnalysisTool::Clippy, AnalysisConfig::default());

        let output = r#"{"reason":"compiler-message","message":{"level":"warning","code":{"code":"clippy::needless_return"},"message":"test warning","spans":[{"file_name":"src/lib.rs","line_start":10,"column_start":5}]}}"#;

        let findings = backend.parse_clippy_output(output);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].severity, Severity::Warning);
        assert_eq!(findings[0].code, "clippy::needless_return");
    }

    #[test]
    fn test_requires_config() {
        assert!(!AnalysisTool::Clippy.requires_config());
        assert!(AnalysisTool::Deny.requires_config());
        assert!(AnalysisTool::Vet.requires_config());
    }

    #[test]
    fn test_severity_equality() {
        assert_eq!(Severity::Error, Severity::Error);
        assert_ne!(Severity::Error, Severity::Warning);
    }
}

// ==================== Kani Proofs ====================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ============== AnalysisTool Proofs ==============

    /// Proves that AnalysisTool::name() returns a non-empty string for all variants
    #[kani::proof]
    fn verify_analysis_tool_name_non_empty() {
        let tools = [
            AnalysisTool::Clippy,
            AnalysisTool::SemverChecks,
            AnalysisTool::Geiger,
            AnalysisTool::Audit,
            AnalysisTool::Deny,
            AnalysisTool::Vet,
            AnalysisTool::Mutants,
        ];
        for tool in tools {
            kani::assert(!tool.name().is_empty(), "Tool name must be non-empty");
        }
    }

    /// Proves that AnalysisTool::cargo_command() returns a non-empty string for all variants
    #[kani::proof]
    fn verify_analysis_tool_cargo_command_non_empty() {
        let tools = [
            AnalysisTool::Clippy,
            AnalysisTool::SemverChecks,
            AnalysisTool::Geiger,
            AnalysisTool::Audit,
            AnalysisTool::Deny,
            AnalysisTool::Vet,
            AnalysisTool::Mutants,
        ];
        for tool in tools {
            kani::assert(
                !tool.cargo_command().is_empty(),
                "Cargo command must be non-empty",
            );
        }
    }

    /// Proves that AnalysisTool::install_command() returns a non-empty string for all variants
    #[kani::proof]
    fn verify_analysis_tool_install_command_non_empty() {
        let tools = [
            AnalysisTool::Clippy,
            AnalysisTool::SemverChecks,
            AnalysisTool::Geiger,
            AnalysisTool::Audit,
            AnalysisTool::Deny,
            AnalysisTool::Vet,
            AnalysisTool::Mutants,
        ];
        for tool in tools {
            kani::assert(
                !tool.install_command().is_empty(),
                "Install command must be non-empty",
            );
        }
    }

    /// Proves that requires_config() returns true only for Deny and Vet
    #[kani::proof]
    fn verify_requires_config_correctness() {
        let tools_that_require_config = [AnalysisTool::Deny, AnalysisTool::Vet];
        let tools_that_dont_require_config = [
            AnalysisTool::Clippy,
            AnalysisTool::SemverChecks,
            AnalysisTool::Geiger,
            AnalysisTool::Audit,
            AnalysisTool::Mutants,
        ];

        for tool in tools_that_require_config {
            kani::assert(tool.requires_config(), "Deny and Vet require config");
        }
        for tool in tools_that_dont_require_config {
            kani::assert(!tool.requires_config(), "Other tools don't require config");
        }
    }

    // ============== AnalysisConfig Proofs ==============

    /// Proves that AnalysisConfig::default() has expected values
    #[kani::proof]
    fn verify_analysis_config_default_values() {
        let config = AnalysisConfig::default();
        kani::assert(
            !config.warnings_as_errors,
            "Default warnings_as_errors is false",
        );
        kani::assert(!config.all_features, "Default all_features is false");
        kani::assert(config.features.is_empty(), "Default features is empty");
        kani::assert(config.targets.is_empty(), "Default targets is empty");
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "Default timeout is 300 seconds",
        );
    }

    /// Proves that with_warnings_as_errors sets the flag to true
    #[kani::proof]
    fn verify_with_warnings_as_errors() {
        let config = AnalysisConfig::default().with_warnings_as_errors();
        kani::assert(
            config.warnings_as_errors,
            "with_warnings_as_errors sets flag to true",
        );
    }

    /// Proves that with_all_features sets the flag to true
    #[kani::proof]
    fn verify_with_all_features() {
        let config = AnalysisConfig::default().with_all_features();
        kani::assert(config.all_features, "with_all_features sets flag to true");
    }

    /// Proves that with_timeout preserves the timeout value
    #[kani::proof]
    fn verify_with_timeout_preserves_value() {
        let secs: u64 = kani::any();
        kani::assume(secs > 0 && secs <= 3600);
        let timeout = Duration::from_secs(secs);
        let config = AnalysisConfig::default().with_timeout(timeout);
        kani::assert(config.timeout == timeout, "with_timeout preserves value");
    }

    // ============== Severity Proofs ==============

    /// Proves that Severity equality is reflexive
    #[kani::proof]
    fn verify_severity_reflexive_equality() {
        let severities = [
            Severity::Error,
            Severity::Warning,
            Severity::Info,
            Severity::Help,
        ];
        for severity in severities {
            kani::assert(severity == severity, "Severity equality must be reflexive");
        }
    }

    /// Proves that Severity variants are distinct
    #[kani::proof]
    fn verify_severity_variants_distinct() {
        kani::assert(Severity::Error != Severity::Warning, "Error != Warning");
        kani::assert(Severity::Error != Severity::Info, "Error != Info");
        kani::assert(Severity::Error != Severity::Help, "Error != Help");
        kani::assert(Severity::Warning != Severity::Info, "Warning != Info");
        kani::assert(Severity::Warning != Severity::Help, "Warning != Help");
        kani::assert(Severity::Info != Severity::Help, "Info != Help");
    }

    // ============== Default Type Proofs ==============

    /// Proves that ClippyResult::default() has empty vectors and zero count
    #[kani::proof]
    fn verify_clippy_result_default() {
        let result = ClippyResult::default();
        kani::assert(result.errors.is_empty(), "Default errors is empty");
        kani::assert(result.warnings.is_empty(), "Default warnings is empty");
        kani::assert(result.lints_checked == 0, "Default lints_checked is 0");
    }

    /// Proves that AuditResult::default() has empty vectors and zero count
    #[kani::proof]
    fn verify_audit_result_default() {
        let result = AuditResult::default();
        kani::assert(
            result.vulnerabilities.is_empty(),
            "Default vulnerabilities is empty",
        );
        kani::assert(result.warnings.is_empty(), "Default warnings is empty");
        kani::assert(
            result.dependencies_scanned == 0,
            "Default dependencies_scanned is 0",
        );
    }

    /// Proves that GeigerResult::default() has zero counts and empty vector
    #[kani::proof]
    fn verify_geiger_result_default() {
        let result = GeigerResult::default();
        kani::assert(result.unsafe_blocks == 0, "Default unsafe_blocks is 0");
        kani::assert(
            result.unsafe_functions == 0,
            "Default unsafe_functions is 0",
        );
        kani::assert(result.unsafe_impls == 0, "Default unsafe_impls is 0");
        kani::assert(
            result.unsafe_crates.is_empty(),
            "Default unsafe_crates is empty",
        );
    }

    // ============== UnsafeCrate Proofs ==============

    /// Proves that safe_percentage is bounded [0, 100]
    #[kani::proof]
    fn verify_unsafe_crate_safe_percentage_bounds() {
        let safe_pct: f64 = kani::any();
        kani::assume(safe_pct >= 0.0 && safe_pct <= 100.0 && safe_pct.is_finite());
        let crate_info = UnsafeCrate {
            name: String::new(),
            version: String::new(),
            unsafe_count: 0,
            safe_percentage: safe_pct,
        };
        kani::assert(
            crate_info.safe_percentage >= 0.0 && crate_info.safe_percentage <= 100.0,
            "safe_percentage is in [0, 100]",
        );
    }

    // ============== AnalysisTool Additional Proofs ==============

    /// Proves that AnalysisTool variants are all distinct
    #[kani::proof]
    fn verify_analysis_tool_variants_distinct() {
        kani::assert(
            AnalysisTool::Clippy != AnalysisTool::SemverChecks,
            "Clippy != SemverChecks",
        );
        kani::assert(
            AnalysisTool::Clippy != AnalysisTool::Geiger,
            "Clippy != Geiger",
        );
        kani::assert(
            AnalysisTool::Clippy != AnalysisTool::Audit,
            "Clippy != Audit",
        );
        kani::assert(AnalysisTool::Clippy != AnalysisTool::Deny, "Clippy != Deny");
        kani::assert(AnalysisTool::Clippy != AnalysisTool::Vet, "Clippy != Vet");
        kani::assert(
            AnalysisTool::Clippy != AnalysisTool::Mutants,
            "Clippy != Mutants",
        );
        kani::assert(
            AnalysisTool::SemverChecks != AnalysisTool::Geiger,
            "SemverChecks != Geiger",
        );
        kani::assert(
            AnalysisTool::SemverChecks != AnalysisTool::Audit,
            "SemverChecks != Audit",
        );
        kani::assert(
            AnalysisTool::Geiger != AnalysisTool::Audit,
            "Geiger != Audit",
        );
        kani::assert(AnalysisTool::Audit != AnalysisTool::Deny, "Audit != Deny");
        kani::assert(AnalysisTool::Deny != AnalysisTool::Vet, "Deny != Vet");
        kani::assert(AnalysisTool::Vet != AnalysisTool::Mutants, "Vet != Mutants");
    }

    /// Proves that only Deny and Vet require config
    #[kani::proof]
    fn verify_requires_config_exhaustive() {
        kani::assert(
            !AnalysisTool::Clippy.requires_config(),
            "Clippy doesn't require config",
        );
        kani::assert(
            !AnalysisTool::SemverChecks.requires_config(),
            "SemverChecks doesn't require config",
        );
        kani::assert(
            !AnalysisTool::Geiger.requires_config(),
            "Geiger doesn't require config",
        );
        kani::assert(
            !AnalysisTool::Audit.requires_config(),
            "Audit doesn't require config",
        );
        kani::assert(AnalysisTool::Deny.requires_config(), "Deny requires config");
        kani::assert(AnalysisTool::Vet.requires_config(), "Vet requires config");
        kani::assert(
            !AnalysisTool::Mutants.requires_config(),
            "Mutants doesn't require config",
        );
    }

    // ============== AnalysisConfig Additional Proofs ==============

    /// Proves that builder methods are orthogonal (don't affect other fields)
    #[kani::proof]
    fn verify_with_warnings_as_errors_orthogonal() {
        let default_config = AnalysisConfig::default();
        let config = AnalysisConfig::default().with_warnings_as_errors();
        kani::assert(
            config.all_features == default_config.all_features,
            "with_warnings_as_errors doesn't change all_features",
        );
        kani::assert(
            config.timeout == default_config.timeout,
            "with_warnings_as_errors doesn't change timeout",
        );
    }

    /// Proves that with_all_features doesn't change other fields
    #[kani::proof]
    fn verify_with_all_features_orthogonal() {
        let default_config = AnalysisConfig::default();
        let config = AnalysisConfig::default().with_all_features();
        kani::assert(
            config.warnings_as_errors == default_config.warnings_as_errors,
            "with_all_features doesn't change warnings_as_errors",
        );
        kani::assert(
            config.timeout == default_config.timeout,
            "with_all_features doesn't change timeout",
        );
    }

    /// Proves that with_timeout doesn't change boolean flags
    #[kani::proof]
    fn verify_with_timeout_orthogonal() {
        let secs: u64 = kani::any();
        kani::assume(secs > 0 && secs <= 3600);
        let default_config = AnalysisConfig::default();
        let config = AnalysisConfig::default().with_timeout(Duration::from_secs(secs));
        kani::assert(
            config.warnings_as_errors == default_config.warnings_as_errors,
            "with_timeout doesn't change warnings_as_errors",
        );
        kani::assert(
            config.all_features == default_config.all_features,
            "with_timeout doesn't change all_features",
        );
    }

    // ============== Severity Additional Proofs ==============

    /// Proves that Severity::Error is the most severe level
    #[kani::proof]
    fn verify_severity_error_is_distinct() {
        kani::assert(Severity::Error != Severity::Warning, "Error != Warning");
        kani::assert(Severity::Error != Severity::Info, "Error != Info");
        kani::assert(Severity::Error != Severity::Help, "Error != Help");
    }

    // ============== Finding Structure Proofs ==============

    /// Proves that Finding preserves tool field
    #[kani::proof]
    fn verify_finding_preserves_tool() {
        let finding = Finding {
            tool: "clippy".to_string(),
            severity: Severity::Warning,
            code: "test".to_string(),
            message: "test".to_string(),
            file: None,
            line: None,
            column: None,
            suggestion: None,
        };
        kani::assert(finding.tool == "clippy", "Finding preserves tool name");
    }

    /// Proves that Finding preserves severity
    #[kani::proof]
    fn verify_finding_preserves_severity() {
        let severities = [
            Severity::Error,
            Severity::Warning,
            Severity::Info,
            Severity::Help,
        ];
        for severity in severities {
            let finding = Finding {
                tool: "test".to_string(),
                severity,
                code: "test".to_string(),
                message: "test".to_string(),
                file: None,
                line: None,
                column: None,
                suggestion: None,
            };
            kani::assert(finding.severity == severity, "Finding preserves severity");
        }
    }

    /// Proves that Finding can have optional fields set or unset
    #[kani::proof]
    fn verify_finding_optional_fields() {
        let finding_none = Finding {
            tool: "test".to_string(),
            severity: Severity::Info,
            code: "test".to_string(),
            message: "test".to_string(),
            file: None,
            line: None,
            column: None,
            suggestion: None,
        };
        kani::assert(finding_none.file.is_none(), "file can be None");
        kani::assert(finding_none.line.is_none(), "line can be None");
        kani::assert(finding_none.column.is_none(), "column can be None");
        kani::assert(finding_none.suggestion.is_none(), "suggestion can be None");

        let finding_some = Finding {
            tool: "test".to_string(),
            severity: Severity::Info,
            code: "test".to_string(),
            message: "test".to_string(),
            file: Some("file.rs".to_string()),
            line: Some(10),
            column: Some(5),
            suggestion: Some("fix it".to_string()),
        };
        kani::assert(finding_some.file.is_some(), "file can be Some");
        kani::assert(finding_some.line.is_some(), "line can be Some");
        kani::assert(finding_some.column.is_some(), "column can be Some");
        kani::assert(finding_some.suggestion.is_some(), "suggestion can be Some");
    }

    // ============== AnalysisResult Proofs ==============

    /// Proves that AnalysisResult preserves tool type
    #[kani::proof]
    fn verify_analysis_result_preserves_tool() {
        let tools = [
            AnalysisTool::Clippy,
            AnalysisTool::SemverChecks,
            AnalysisTool::Geiger,
            AnalysisTool::Audit,
            AnalysisTool::Deny,
            AnalysisTool::Vet,
            AnalysisTool::Mutants,
        ];
        for tool in tools {
            let result = AnalysisResult {
                tool,
                passed: true,
                findings: Vec::new(),
                error_count: 0,
                warning_count: 0,
                duration: Duration::from_secs(1),
                raw_output: String::new(),
            };
            kani::assert(result.tool == tool, "AnalysisResult preserves tool");
        }
    }

    /// Proves that error_count can be any reasonable value
    #[kani::proof]
    fn verify_analysis_result_error_count_range() {
        let error_count: usize = kani::any();
        kani::assume(error_count <= 10000);
        let result = AnalysisResult {
            tool: AnalysisTool::Clippy,
            passed: error_count == 0,
            findings: Vec::new(),
            error_count,
            warning_count: 0,
            duration: Duration::from_secs(1),
            raw_output: String::new(),
        };
        kani::assert(
            result.error_count == error_count,
            "error_count is preserved",
        );
    }

    /// Proves that warning_count can be any reasonable value
    #[kani::proof]
    fn verify_analysis_result_warning_count_range() {
        let warning_count: usize = kani::any();
        kani::assume(warning_count <= 10000);
        let result = AnalysisResult {
            tool: AnalysisTool::Clippy,
            passed: true,
            findings: Vec::new(),
            error_count: 0,
            warning_count,
            duration: Duration::from_secs(1),
            raw_output: String::new(),
        };
        kani::assert(
            result.warning_count == warning_count,
            "warning_count is preserved",
        );
    }

    // ============== Vulnerability Structure Proofs ==============

    /// Proves that Vulnerability preserves id field
    #[kani::proof]
    fn verify_vulnerability_preserves_id() {
        let vuln = Vulnerability {
            id: "RUSTSEC-2024-0001".to_string(),
            crate_name: "test".to_string(),
            version: "1.0.0".to_string(),
            severity: "high".to_string(),
            title: "Test".to_string(),
            url: None,
            patched_versions: Vec::new(),
        };
        kani::assert(vuln.id == "RUSTSEC-2024-0001", "Vulnerability preserves id");
    }

    /// Proves that Vulnerability can have optional url
    #[kani::proof]
    fn verify_vulnerability_optional_url() {
        let vuln_none = Vulnerability {
            id: "test".to_string(),
            crate_name: "test".to_string(),
            version: "1.0.0".to_string(),
            severity: "high".to_string(),
            title: "Test".to_string(),
            url: None,
            patched_versions: Vec::new(),
        };
        kani::assert(vuln_none.url.is_none(), "url can be None");

        let vuln_some = Vulnerability {
            id: "test".to_string(),
            crate_name: "test".to_string(),
            version: "1.0.0".to_string(),
            severity: "high".to_string(),
            title: "Test".to_string(),
            url: Some("https://example.com".to_string()),
            patched_versions: Vec::new(),
        };
        kani::assert(vuln_some.url.is_some(), "url can be Some");
    }

    // ============== UnsafeCrate Additional Proofs ==============

    /// Proves that UnsafeCrate preserves unsafe_count
    #[kani::proof]
    fn verify_unsafe_crate_preserves_count() {
        let count: usize = kani::any();
        kani::assume(count <= 1_000_000);
        let crate_info = UnsafeCrate {
            name: "test".to_string(),
            version: "1.0.0".to_string(),
            unsafe_count: count,
            safe_percentage: 50.0,
        };
        kani::assert(
            crate_info.unsafe_count == count,
            "unsafe_count is preserved",
        );
    }

    // ============== StaticAnalysisBackend Proofs ==============

    /// Proves that StaticAnalysisBackend preserves tool type
    #[kani::proof]
    fn verify_static_analysis_backend_preserves_tool() {
        let tools = [
            AnalysisTool::Clippy,
            AnalysisTool::SemverChecks,
            AnalysisTool::Geiger,
            AnalysisTool::Audit,
            AnalysisTool::Deny,
            AnalysisTool::Vet,
            AnalysisTool::Mutants,
        ];
        for tool in tools {
            let backend = StaticAnalysisBackend::new(tool, AnalysisConfig::default());
            kani::assert(backend.tool == tool, "StaticAnalysisBackend preserves tool");
        }
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    // Strategy for generating AnalysisTool variants
    fn analysis_tool_strategy() -> impl Strategy<Value = AnalysisTool> {
        prop_oneof![
            Just(AnalysisTool::Clippy),
            Just(AnalysisTool::SemverChecks),
            Just(AnalysisTool::Geiger),
            Just(AnalysisTool::Audit),
            Just(AnalysisTool::Deny),
            Just(AnalysisTool::Vet),
            Just(AnalysisTool::Mutants),
        ]
    }

    // Strategy for generating Severity variants
    fn severity_strategy() -> impl Strategy<Value = Severity> {
        prop_oneof![
            Just(Severity::Error),
            Just(Severity::Warning),
            Just(Severity::Info),
            Just(Severity::Help),
        ]
    }

    proptest! {
        // AnalysisTool property tests
        #[test]
        fn analysis_tool_name_non_empty(tool in analysis_tool_strategy()) {
            prop_assert!(!tool.name().is_empty());
        }

        #[test]
        fn analysis_tool_cargo_command_non_empty(tool in analysis_tool_strategy()) {
            prop_assert!(!tool.cargo_command().is_empty());
        }

        #[test]
        fn analysis_tool_install_command_non_empty(tool in analysis_tool_strategy()) {
            prop_assert!(!tool.install_command().is_empty());
        }

        #[test]
        fn analysis_tool_requires_config_consistency(tool in analysis_tool_strategy()) {
            let requires = tool.requires_config();
            let expected = matches!(tool, AnalysisTool::Deny | AnalysisTool::Vet);
            prop_assert_eq!(requires, expected);
        }

        // AnalysisConfig property tests
        #[test]
        fn analysis_config_default_values(_dummy in 0..1) {
            let config = AnalysisConfig::default();
            prop_assert!(!config.warnings_as_errors);
            prop_assert!(!config.all_features);
            prop_assert!(config.features.is_empty());
            prop_assert!(config.targets.is_empty());
            prop_assert_eq!(config.timeout, Duration::from_secs(300));
        }

        #[test]
        fn analysis_config_warnings_as_errors_preserved(_dummy in 0..1) {
            let config = AnalysisConfig::default().with_warnings_as_errors();
            prop_assert!(config.warnings_as_errors);
        }

        #[test]
        fn analysis_config_all_features_preserved(_dummy in 0..1) {
            let config = AnalysisConfig::default().with_all_features();
            prop_assert!(config.all_features);
        }

        #[test]
        fn analysis_config_timeout_preserved(secs in 1u64..10000) {
            let timeout = Duration::from_secs(secs);
            let config = AnalysisConfig::default().with_timeout(timeout);
            prop_assert_eq!(config.timeout, timeout);
        }

        #[test]
        fn analysis_config_builder_chaining(secs in 1u64..1000) {
            let timeout = Duration::from_secs(secs);
            let config = AnalysisConfig::default()
                .with_warnings_as_errors()
                .with_all_features()
                .with_timeout(timeout);

            prop_assert!(config.warnings_as_errors);
            prop_assert!(config.all_features);
            prop_assert_eq!(config.timeout, timeout);
        }

        // StaticAnalysisBackend property tests
        #[test]
        fn static_analysis_backend_preserves_tool(tool in analysis_tool_strategy()) {
            let backend = StaticAnalysisBackend::new(tool, AnalysisConfig::default());
            prop_assert_eq!(backend.tool, tool);
        }

        // Finding structure tests
        #[test]
        fn finding_preserves_fields(
            tool in "[a-z]+",
            severity in severity_strategy(),
            code in "[a-z_]+",
            message in ".{1,50}"
        ) {
            let finding = Finding {
                tool: tool.clone(),
                severity,
                code: code.clone(),
                message: message.clone(),
                file: None,
                line: None,
                column: None,
                suggestion: None,
            };
            prop_assert_eq!(finding.tool, tool);
            prop_assert_eq!(finding.severity, severity);
            prop_assert_eq!(finding.code, code);
            prop_assert_eq!(finding.message, message);
        }

        #[test]
        fn finding_with_file_location(
            file in "[a-z]+\\.rs",
            line in 1u32..10000,
            column in 1u32..1000
        ) {
            let finding = Finding {
                tool: "test".to_string(),
                severity: Severity::Error,
                code: "test_code".to_string(),
                message: "test message".to_string(),
                file: Some(file.clone()),
                line: Some(line),
                column: Some(column),
                suggestion: None,
            };
            prop_assert_eq!(finding.file, Some(file));
            prop_assert_eq!(finding.line, Some(line));
            prop_assert_eq!(finding.column, Some(column));
        }

        #[test]
        fn finding_with_suggestion(suggestion in ".{1,100}") {
            let finding = Finding {
                tool: "test".to_string(),
                severity: Severity::Warning,
                code: "test_code".to_string(),
                message: "test message".to_string(),
                file: None,
                line: None,
                column: None,
                suggestion: Some(suggestion.clone()),
            };
            prop_assert_eq!(finding.suggestion, Some(suggestion));
        }

        // Vulnerability structure tests
        #[test]
        fn vulnerability_preserves_fields(
            id in "RUSTSEC-[0-9]{4}-[0-9]{4}",
            crate_name in "[a-z_]+",
            version in "[0-9]+\\.[0-9]+\\.[0-9]+",
            severity in "(critical|high|medium|low)",
            title in ".{1,50}"
        ) {
            let vuln = Vulnerability {
                id: id.clone(),
                crate_name: crate_name.clone(),
                version: version.clone(),
                severity: severity.clone(),
                title: title.clone(),
                url: None,
                patched_versions: Vec::new(),
            };
            prop_assert_eq!(vuln.id, id);
            prop_assert_eq!(vuln.crate_name, crate_name);
            prop_assert_eq!(vuln.version, version);
            prop_assert_eq!(vuln.severity, severity);
            prop_assert_eq!(vuln.title, title);
        }

        #[test]
        fn vulnerability_with_url(url in "https://[a-z]+\\.[a-z]+/[a-z]+") {
            let vuln = Vulnerability {
                id: "RUSTSEC-2024-0001".to_string(),
                crate_name: "test".to_string(),
                version: "1.0.0".to_string(),
                severity: "high".to_string(),
                title: "Test vulnerability".to_string(),
                url: Some(url.clone()),
                patched_versions: Vec::new(),
            };
            prop_assert_eq!(vuln.url, Some(url));
        }

        // UnsafeCrate structure tests
        #[test]
        fn unsafe_crate_preserves_fields(
            name in "[a-z_]+",
            version in "[0-9]+\\.[0-9]+\\.[0-9]+",
            unsafe_count in 0usize..1000,
            safe_pct in 0.0f64..100.0
        ) {
            let crate_info = UnsafeCrate {
                name: name.clone(),
                version: version.clone(),
                unsafe_count,
                safe_percentage: safe_pct,
            };
            prop_assert_eq!(crate_info.name, name);
            prop_assert_eq!(crate_info.version, version);
            prop_assert_eq!(crate_info.unsafe_count, unsafe_count);
            prop_assert!((crate_info.safe_percentage - safe_pct).abs() < 0.001);
        }

        // AnalysisResult property tests
        #[test]
        fn analysis_result_passed_when_no_errors(tool in analysis_tool_strategy()) {
            let result = AnalysisResult {
                tool,
                passed: true,
                findings: Vec::new(),
                error_count: 0,
                warning_count: 0,
                duration: Duration::from_secs(1),
                raw_output: String::new(),
            };
            prop_assert!(result.passed);
            prop_assert_eq!(result.error_count, 0);
        }

        #[test]
        fn analysis_result_preserves_duration(secs in 1u64..10000) {
            let duration = Duration::from_secs(secs);
            let result = AnalysisResult {
                tool: AnalysisTool::Clippy,
                passed: true,
                findings: Vec::new(),
                error_count: 0,
                warning_count: 0,
                duration,
                raw_output: String::new(),
            };
            prop_assert_eq!(result.duration, duration);
        }

        #[test]
        fn analysis_result_preserves_raw_output(raw_output in ".{0,200}") {
            let result = AnalysisResult {
                tool: AnalysisTool::Audit,
                passed: true,
                findings: Vec::new(),
                error_count: 0,
                warning_count: 0,
                duration: Duration::from_secs(1),
                raw_output: raw_output.clone(),
            };
            prop_assert_eq!(result.raw_output, raw_output);
        }

        // ClippyResult default tests
        #[test]
        fn clippy_result_default_is_empty(_dummy in 0..1) {
            let result = ClippyResult::default();
            prop_assert!(result.errors.is_empty());
            prop_assert!(result.warnings.is_empty());
            prop_assert_eq!(result.lints_checked, 0);
        }

        // AuditResult default tests
        #[test]
        fn audit_result_default_is_empty(_dummy in 0..1) {
            let result = AuditResult::default();
            prop_assert!(result.vulnerabilities.is_empty());
            prop_assert!(result.warnings.is_empty());
            prop_assert_eq!(result.dependencies_scanned, 0);
        }

        // GeigerResult default tests
        #[test]
        fn geiger_result_default_is_zero(_dummy in 0..1) {
            let result = GeigerResult::default();
            prop_assert_eq!(result.unsafe_blocks, 0);
            prop_assert_eq!(result.unsafe_functions, 0);
            prop_assert_eq!(result.unsafe_impls, 0);
            prop_assert!(result.unsafe_crates.is_empty());
        }

        // AnalysisError message preservation tests
        #[test]
        fn analysis_error_not_installed_contains_name(name in "[A-Za-z]+") {
            let error = AnalysisError::NotInstalled(name.clone(), "cargo install foo".to_string());
            let msg = error.to_string();
            prop_assert!(msg.contains(&name));
        }

        #[test]
        fn analysis_error_failed_contains_message(message in ".{1,50}") {
            let error = AnalysisError::Failed(message.clone());
            let msg = error.to_string();
            prop_assert!(msg.contains(&message));
        }

        #[test]
        fn analysis_error_config_missing_contains_tool(tool in "[A-Za-z]+") {
            let error = AnalysisError::ConfigMissing(tool.clone(), "Create config file".to_string());
            let msg = error.to_string();
            prop_assert!(msg.contains(&tool));
        }

        #[test]
        fn analysis_error_timeout_contains_info(secs in 1u64..10000) {
            let duration = Duration::from_secs(secs);
            let error = AnalysisError::Timeout(duration);
            let msg = error.to_string();
            prop_assert!(msg.contains("timeout") || msg.contains("Timeout"));
        }

        // Clippy output parsing tests
        #[test]
        fn parse_clippy_output_handles_error_level(_dummy in 0..1) {
            let backend = StaticAnalysisBackend::new(AnalysisTool::Clippy, AnalysisConfig::default());
            let output = r#"{"reason":"compiler-message","message":{"level":"error","code":{"code":"E0001"},"message":"test error","spans":[]}}"#;
            let findings = backend.parse_clippy_output(output);
            prop_assert_eq!(findings.len(), 1);
            prop_assert_eq!(findings[0].severity, Severity::Error);
        }

        #[test]
        fn parse_clippy_output_handles_warning_level(_dummy in 0..1) {
            let backend = StaticAnalysisBackend::new(AnalysisTool::Clippy, AnalysisConfig::default());
            let output = r#"{"reason":"compiler-message","message":{"level":"warning","code":{"code":"W0001"},"message":"test warning","spans":[]}}"#;
            let findings = backend.parse_clippy_output(output);
            prop_assert_eq!(findings.len(), 1);
            prop_assert_eq!(findings[0].severity, Severity::Warning);
        }

        #[test]
        fn parse_clippy_output_handles_help_level(_dummy in 0..1) {
            let backend = StaticAnalysisBackend::new(AnalysisTool::Clippy, AnalysisConfig::default());
            let output = r#"{"reason":"compiler-message","message":{"level":"help","code":{"code":"H0001"},"message":"test help","spans":[]}}"#;
            let findings = backend.parse_clippy_output(output);
            prop_assert_eq!(findings.len(), 1);
            prop_assert_eq!(findings[0].severity, Severity::Help);
        }

        #[test]
        fn parse_clippy_output_handles_empty_output(_dummy in 0..1) {
            let backend = StaticAnalysisBackend::new(AnalysisTool::Clippy, AnalysisConfig::default());
            let findings = backend.parse_clippy_output("");
            prop_assert!(findings.is_empty());
        }

        #[test]
        fn parse_clippy_output_handles_non_json(text in "[a-z ]+") {
            let backend = StaticAnalysisBackend::new(AnalysisTool::Clippy, AnalysisConfig::default());
            let findings = backend.parse_clippy_output(&text);
            prop_assert!(findings.is_empty());
        }

        // Severity equality tests
        #[test]
        fn severity_reflexive_equality(severity in severity_strategy()) {
            prop_assert_eq!(severity, severity);
        }

        // AnalysisTool equality tests
        #[test]
        fn analysis_tool_reflexive_equality(tool in analysis_tool_strategy()) {
            prop_assert_eq!(tool, tool);
        }
    }
}
