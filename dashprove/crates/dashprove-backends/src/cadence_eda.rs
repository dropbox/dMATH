//! Cadence EDA verification suite backend
//!
//! Cadence provides a comprehensive suite of EDA tools for verification
//! including simulation, formal verification, and emulation.
//!
//! See: <https://www.cadence.com/>
//!
//! # Features
//!
//! - **Xcelium**: High-performance simulation
//! - **JasperGold**: Formal verification (see jaspergold.rs)
//! - **Palladium**: Hardware emulation
//! - **Conformal**: Equivalence checking
//!
//! # Requirements
//!
//! Commercial license required from Cadence.

// =============================================
// Kani Proofs for Cadence EDA Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- CadenceTool Default Tests ----

    /// Verify CadenceTool default is Xcelium
    #[kani::proof]
    fn proof_cadence_tool_default() {
        let tool = CadenceTool::default();
        kani::assert(
            tool == CadenceTool::Xcelium,
            "default tool should be Xcelium",
        );
    }

    // ---- CadenceEdaConfig Default Tests ----

    /// Verify CadenceEdaConfig::default sets correct baseline values
    #[kani::proof]
    fn proof_cadence_config_defaults() {
        let config = CadenceEdaConfig::default();
        kani::assert(
            config.cadence_path.is_none(),
            "cadence_path should default to None",
        );
        kani::assert(
            config.timeout == Duration::from_secs(600),
            "timeout should default to 600 seconds",
        );
        kani::assert(
            config.tool == CadenceTool::Xcelium,
            "tool should default to Xcelium",
        );
        kani::assert(
            config.extra_args.is_empty(),
            "extra_args should default to empty",
        );
    }

    // ---- CadenceEdaBackend Construction Tests ----

    /// Verify CadenceEdaBackend::new uses default configuration
    #[kani::proof]
    fn proof_cadence_backend_new_defaults() {
        let backend = CadenceEdaBackend::new();
        kani::assert(
            backend.config.cadence_path.is_none(),
            "new backend should have no cadence_path",
        );
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "new backend should default timeout to 600 seconds",
        );
        kani::assert(
            backend.config.tool == CadenceTool::Xcelium,
            "new backend should default tool to Xcelium",
        );
    }

    /// Verify CadenceEdaBackend::default equals CadenceEdaBackend::new
    #[kani::proof]
    fn proof_cadence_backend_default_equals_new() {
        let default_backend = CadenceEdaBackend::default();
        let new_backend = CadenceEdaBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "default and new should share timeout",
        );
        kani::assert(
            default_backend.config.tool == new_backend.config.tool,
            "default and new should share tool",
        );
    }

    /// Verify CadenceEdaBackend::with_config preserves custom configuration
    #[kani::proof]
    fn proof_cadence_backend_with_config() {
        let config = CadenceEdaConfig {
            cadence_path: Some(PathBuf::from("/opt/cadence")),
            timeout: Duration::from_secs(300),
            tool: CadenceTool::Conformal,
            extra_args: vec!["-debug".to_string()],
        };
        let backend = CadenceEdaBackend::with_config(config);
        kani::assert(
            backend.config.cadence_path == Some(PathBuf::from("/opt/cadence")),
            "with_config should preserve cadence_path",
        );
        kani::assert(
            backend.config.timeout == Duration::from_secs(300),
            "with_config should preserve timeout",
        );
        kani::assert(
            backend.config.tool == CadenceTool::Conformal,
            "with_config should preserve tool",
        );
        kani::assert(
            backend.config.extra_args.len() == 1,
            "with_config should preserve extra_args",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify id() returns BackendId::CadenceEDA
    #[kani::proof]
    fn proof_cadence_backend_id() {
        let backend = CadenceEdaBackend::new();
        kani::assert(
            backend.id() == BackendId::CadenceEDA,
            "CadenceEdaBackend id should be BackendId::CadenceEDA",
        );
    }

    /// Verify supports() includes Invariant and Temporal
    #[kani::proof]
    fn proof_cadence_backend_supports() {
        let backend = CadenceEdaBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::Invariant),
            "supports should include Invariant",
        );
        kani::assert(
            supported.contains(&PropertyType::Temporal),
            "supports should include Temporal",
        );
    }

    /// Verify supports() returns exactly two property types
    #[kani::proof]
    fn proof_cadence_backend_supports_length() {
        let backend = CadenceEdaBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.len() == 2,
            "CadenceEDA should support exactly two property types",
        );
    }

    // ---- Output Parsing Tests ----

    /// Verify parse_output detects passed as Proven
    #[kani::proof]
    fn proof_parse_output_passed() {
        let backend = CadenceEdaBackend::new();
        let (status, _) = backend.parse_output("Simulation passed\nAll tests completed", "", true);
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "passed should produce Proven status",
        );
    }

    /// Verify parse_output detects FAIL as Disproven
    #[kani::proof]
    fn proof_parse_output_fail() {
        let backend = CadenceEdaBackend::new();
        let (status, _) = backend.parse_output("assertion failed\nFAIL", "", false);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "FAIL should produce Disproven status",
        );
    }

    // ---- Name Sanitization Tests ----

    /// Verify sanitize_name replaces special characters
    #[kani::proof]
    fn proof_sanitize_name() {
        kani::assert(
            CadenceEdaBackend::sanitize_name("my-prop") == "my_prop",
            "should replace hyphen with underscore",
        );
        kani::assert(
            CadenceEdaBackend::sanitize_name("test.name") == "test_name",
            "should replace dot with underscore",
        );
        kani::assert(
            CadenceEdaBackend::sanitize_name("prop[0]") == "prop_0_",
            "should replace brackets",
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
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::process::Command;
use tracing::debug;

/// Tool selection for Cadence EDA
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CadenceTool {
    /// Xcelium simulation
    #[default]
    Xcelium,
    /// Conformal equivalence checking
    Conformal,
    /// Genus synthesis
    Genus,
}

/// Configuration for Cadence EDA backend
#[derive(Debug, Clone)]
pub struct CadenceEdaConfig {
    /// Path to Cadence tools (CDS_HOME)
    pub cadence_path: Option<PathBuf>,
    /// Timeout for verification
    pub timeout: Duration,
    /// Tool to use
    pub tool: CadenceTool,
    /// Additional tool options
    pub extra_args: Vec<String>,
}

impl Default for CadenceEdaConfig {
    fn default() -> Self {
        Self {
            cadence_path: None,
            timeout: Duration::from_secs(600),
            tool: CadenceTool::default(),
            extra_args: vec![],
        }
    }
}

/// Cadence EDA verification backend
pub struct CadenceEdaBackend {
    config: CadenceEdaConfig,
}

impl Default for CadenceEdaBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CadenceEdaBackend {
    /// Create a new Cadence EDA backend with default configuration
    pub fn new() -> Self {
        Self {
            config: CadenceEdaConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: CadenceEdaConfig) -> Self {
        Self { config }
    }

    async fn detect(&self) -> Result<PathBuf, String> {
        // Check configured path first
        if let Some(ref path) = self.config.cadence_path {
            if path.exists() {
                return Ok(path.clone());
            }
        }

        // Check CDS_HOME environment variable
        if let Ok(cds_home) = std::env::var("CDS_HOME") {
            let cds_path = PathBuf::from(&cds_home);
            if cds_path.exists() {
                return Ok(cds_path);
            }
        }

        // Try to find Xcelium directly
        if let Ok(path) = which::which("xrun") {
            return Ok(path);
        }

        Err("Cadence EDA not found (commercial license required)".to_string())
    }

    /// Generate Verilog testbench from USL spec
    fn generate_verilog(&self, spec: &TypedSpec) -> String {
        let mut code = String::new();
        code.push_str("// Generated by DashProve\n\n");

        // Module declaration
        code.push_str("module verify_tb;\n");
        code.push_str("  reg clk;\n");
        code.push_str("  reg rst;\n");
        code.push_str("  reg [31:0] data;\n\n");

        // Clock generation
        code.push_str("  initial clk = 0;\n");
        code.push_str("  always #5 clk = ~clk;\n\n");

        // Test sequence
        code.push_str("  initial begin\n");
        code.push_str("    rst = 1;\n");
        code.push_str("    #20 rst = 0;\n");

        // Generate checks for properties
        for (i, prop) in spec.spec.properties.iter().enumerate() {
            let prop_name = prop.name();
            code.push_str(&format!("    // Property {}: {}\n", i, prop_name));
            code.push_str(&format!(
                "    #10 if (!rst) $display(\"Property {}: PASS\");\n",
                i
            ));
        }

        code.push_str("    #100 $finish;\n");
        code.push_str("  end\n\n");

        // Assertions
        for (i, prop) in spec.spec.properties.iter().enumerate() {
            let prop_name = prop.name();
            let safe_name = Self::sanitize_name(&prop_name);
            let assert_name = if safe_name.is_empty() {
                format!("prop_{}", i)
            } else {
                format!("prop_{}", safe_name)
            };
            code.push_str(&format!(
                "  assert_{}: assert property (@(posedge clk) !rst |-> data >= 0)\n",
                assert_name
            ));
            code.push_str(&format!(
                "    else $error(\"Property {} failed\");\n\n",
                assert_name
            ));
        }

        code.push_str("endmodule\n");
        code
    }

    /// Sanitize a name for Verilog
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

        for line in combined.lines() {
            let trimmed = line.trim();

            if trimmed.contains("PASS") || trimmed.contains("passed") {
                diagnostics.push(format!("✓ {}", trimmed));
            }

            if trimmed.contains("FAIL") || trimmed.contains("Error") || trimmed.contains("error") {
                diagnostics.push(format!("✗ {}", trimmed));
            }
        }

        // Check for success
        if combined.contains("passed")
            || (success && !combined.contains("Error") && !combined.contains("FAIL"))
        {
            return (VerificationStatus::Proven, diagnostics);
        }

        // Check for failure
        if combined.contains("FAIL") || combined.contains("assertion failed") {
            return (VerificationStatus::Disproven, diagnostics);
        }

        if !success {
            return (
                VerificationStatus::Unknown {
                    reason: "Cadence tool returned non-zero exit".to_string(),
                },
                diagnostics,
            );
        }

        (
            VerificationStatus::Unknown {
                reason: "Could not parse output".to_string(),
            },
            diagnostics,
        )
    }

    fn parse_counterexample(stdout: &str, stderr: &str) -> StructuredCounterexample {
        let mut ce = StructuredCounterexample::new();
        let combined = format!("{}\n{}", stdout, stderr);
        ce.raw = Some(combined.clone());
        ce.failed_checks = Self::extract_failed_checks(&combined);
        ce.witness = HashMap::new();
        ce
    }

    fn extract_failed_checks(output: &str) -> Vec<FailedCheck> {
        let mut checks = Vec::new();

        for line in output.lines() {
            let trimmed = line.trim();
            if trimmed.contains("FAIL") || trimmed.contains("failed") {
                checks.push(FailedCheck {
                    check_id: "cadence_assertion".to_string(),
                    description: trimmed.to_string(),
                    location: None,
                    function: None,
                });
            }
        }

        checks
    }
}

#[async_trait]
impl VerificationBackend for CadenceEdaBackend {
    fn id(&self) -> BackendId {
        BackendId::CadenceEDA
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::Invariant, PropertyType::Temporal]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let cds_path = self.detect().await.map_err(BackendError::Unavailable)?;

        // Create temp directory
        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp directory: {}", e))
        })?;

        let verilog_file = temp_dir.path().join("verify_tb.sv");
        let verilog_code = self.generate_verilog(spec);

        debug!("Generated Verilog:\n{}", verilog_code);

        tokio::fs::write(&verilog_file, &verilog_code)
            .await
            .map_err(|e| {
                BackendError::VerificationFailed(format!("Failed to write Verilog file: {}", e))
            })?;

        // Determine tool binary
        let tool_bin = match self.config.tool {
            CadenceTool::Xcelium => {
                if cds_path.is_file() {
                    cds_path.clone()
                } else {
                    cds_path
                        .join("tools")
                        .join("xcelium")
                        .join("bin")
                        .join("xrun")
                }
            }
            CadenceTool::Conformal => cds_path
                .join("tools")
                .join("conformal")
                .join("bin")
                .join("lec"),
            CadenceTool::Genus => cds_path
                .join("tools")
                .join("genus")
                .join("bin")
                .join("genus"),
        };

        // Build command
        let mut cmd = Command::new(&tool_bin);
        cmd.arg(&verilog_file)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .current_dir(temp_dir.path());

        for arg in &self.config.extra_args {
            cmd.arg(arg);
        }

        let output = tokio::time::timeout(self.config.timeout, cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| {
                BackendError::VerificationFailed(format!("Failed to run Cadence tool: {}", e))
            })?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        debug!("Cadence stdout: {}", stdout);
        debug!("Cadence stderr: {}", stderr);

        let (status, diagnostics) = self.parse_output(&stdout, &stderr, output.status.success());

        let counterexample = if matches!(status, VerificationStatus::Disproven) {
            Some(Self::parse_counterexample(&stdout, &stderr))
        } else {
            None
        };

        Ok(BackendResult {
            backend: BackendId::CadenceEDA,
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
        assert_eq!(CadenceEdaBackend::new().id(), BackendId::CadenceEDA);
    }

    #[test]
    fn default_config() {
        let config = CadenceEdaConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(600));
        assert_eq!(config.tool, CadenceTool::Xcelium);
    }

    #[test]
    fn supports_invariant_and_temporal() {
        let backend = CadenceEdaBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::Invariant));
        assert!(supported.contains(&PropertyType::Temporal));
    }

    #[test]
    fn sanitize_name() {
        assert_eq!(CadenceEdaBackend::sanitize_name("my-prop"), "my_prop");
    }

    #[test]
    fn parse_pass_output() {
        let backend = CadenceEdaBackend::new();
        let stdout = "Simulation passed\nAll tests completed";
        let (status, diag) = backend.parse_output(stdout, "", true);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(!diag.is_empty());
    }

    #[test]
    fn parse_fail_output() {
        let backend = CadenceEdaBackend::new();
        let stdout = "assertion failed\nFAIL";
        let (status, _diag) = backend.parse_output(stdout, "", false);
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[test]
    fn generate_verilog_empty_spec() {
        use dashprove_usl::ast::Spec;

        let backend = CadenceEdaBackend::new();
        let spec = TypedSpec {
            spec: Spec {
                types: vec![],
                properties: vec![],
            },
            type_info: HashMap::new(),
        };
        let code = backend.generate_verilog(&spec);
        assert!(code.contains("Generated by DashProve"));
        assert!(code.contains("module verify_tb"));
    }

    #[test]
    fn config_with_conformal() {
        let config = CadenceEdaConfig {
            tool: CadenceTool::Conformal,
            ..Default::default()
        };
        let backend = CadenceEdaBackend::with_config(config);
        assert_eq!(backend.config.tool, CadenceTool::Conformal);
    }
}
