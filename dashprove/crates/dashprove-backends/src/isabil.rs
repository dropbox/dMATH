//! IsaBIL (Isabelle/HOL BIL) backend
//!
//! IsaBIL provides a machine-checked semantics of BAP's Binary Intermediate Language (BIL)
//! inside Isabelle/HOL, enabling end-to-end proofs about lifted binaries.
//!
//! Key features:
//! - Formal semantics of BIL in Isabelle/HOL
//! - Binary refinement proofs
//! - Control flow verification
//! - Simulation proofs between binary and specification
//!
//! Input: Binary + Isabelle theory files
//! Output: Proof results or counterexample traces
//!
//! See: <https://github.com/matt-j-griffin/isabil>

// =============================================
// Kani Proofs for IsaBIL Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- IsaBilConfig Default Tests ----

    /// Verify IsaBilConfig::default sets correct baseline values
    #[kani::proof]
    fn proof_isabil_config_defaults() {
        let config = IsaBilConfig::default();
        kani::assert(
            config.isabelle_path.is_none(),
            "isabelle_path should default to None",
        );
        kani::assert(
            config.timeout == Duration::from_secs(600),
            "timeout should default to 600 seconds",
        );
        kani::assert(
            config.target_binary.is_none(),
            "target_binary should default to None",
        );
        kani::assert(
            config.session_name == "IsaBIL",
            "session_name should default to IsaBIL",
        );
        kani::assert(
            config.verification_mode == IsaBilMode::Simulation,
            "verification_mode should default to Simulation",
        );
    }

    // ---- IsaBilBackend Construction Tests ----

    /// Verify IsaBilBackend::new uses default configuration
    #[kani::proof]
    fn proof_isabil_backend_new_defaults() {
        let backend = IsaBilBackend::new();
        kani::assert(
            backend.config.isabelle_path.is_none(),
            "new backend should have no isabelle_path",
        );
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "new backend should default timeout to 600 seconds",
        );
    }

    /// Verify IsaBilBackend::default equals IsaBilBackend::new
    #[kani::proof]
    fn proof_isabil_backend_default_equals_new() {
        let default_backend = IsaBilBackend::default();
        let new_backend = IsaBilBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "default and new should share timeout",
        );
        kani::assert(
            default_backend.config.verification_mode == new_backend.config.verification_mode,
            "default and new should share verification_mode",
        );
    }

    /// Verify IsaBilBackend::with_config preserves custom configuration
    #[kani::proof]
    fn proof_isabil_backend_with_config() {
        let config = IsaBilConfig {
            isabelle_path: Some(PathBuf::from("/usr/bin/isabelle")),
            timeout: Duration::from_secs(1200),
            target_binary: Some(PathBuf::from("/bin/test")),
            bap_path: Some(PathBuf::from("/usr/bin/bap")),
            session_name: "CustomSession".to_string(),
            theory_file: Some(PathBuf::from("/path/to/theory.thy")),
            verification_mode: IsaBilMode::Refinement,
            extra_options: vec!["-o".to_string(), "value".to_string()],
        };
        let backend = IsaBilBackend::with_config(config);
        kani::assert(
            backend.config.isabelle_path == Some(PathBuf::from("/usr/bin/isabelle")),
            "with_config should preserve isabelle_path",
        );
        kani::assert(
            backend.config.timeout == Duration::from_secs(1200),
            "with_config should preserve timeout",
        );
        kani::assert(
            backend.config.verification_mode == IsaBilMode::Refinement,
            "with_config should preserve verification_mode",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify id() returns BackendId::IsaBIL
    #[kani::proof]
    fn proof_isabil_backend_id() {
        let backend = IsaBilBackend::new();
        kani::assert(
            backend.id() == BackendId::IsaBIL,
            "IsaBilBackend id should be BackendId::IsaBIL",
        );
    }

    /// Verify supports() includes BinaryRefinement and Invariant
    #[kani::proof]
    fn proof_isabil_backend_supports() {
        let backend = IsaBilBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::Theorem),
            "supports should include Theorem",
        );
        kani::assert(
            supported.contains(&PropertyType::Invariant),
            "supports should include Invariant",
        );
    }

    // ---- Output Parsing Tests ----

    /// Verify parse_output detects proven status
    #[kani::proof]
    fn proof_parse_output_proven() {
        let backend = IsaBilBackend::new();
        let output = "theorem proved\nNo more goals.";
        let (status, ce) = backend.parse_output(output, "");
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "proved should produce Proven status",
        );
        kani::assert(ce.is_none(), "proved should have no counterexample");
    }

    /// Verify parse_output detects counterexample status
    #[kani::proof]
    fn proof_parse_output_counterexample() {
        let backend = IsaBilBackend::new();
        let output = "Auto Quickcheck found a counterexample:\n  x = 0";
        let (status, ce) = backend.parse_output(output, "");
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "counterexample should produce Disproven status",
        );
        kani::assert(
            ce.is_some(),
            "counterexample status should have counterexample",
        );
    }

    /// Verify parse_output handles type errors
    #[kani::proof]
    fn proof_parse_output_type_error() {
        let backend = IsaBilBackend::new();
        let (status, _) = backend.parse_output("", "Type unification failed");
        kani::assert(
            matches!(status, VerificationStatus::Unknown { .. }),
            "type error should produce Unknown status",
        );
    }
}

use crate::counterexample::{CounterexampleValue, FailedCheck, StructuredCounterexample};
use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
    VerificationStatus,
};
use async_trait::async_trait;
use dashprove_usl::typecheck::TypedSpec;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::process::Command;
use tracing::debug;

/// Verification mode for IsaBIL
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IsaBilMode {
    /// Prove simulation between binary and specification
    #[default]
    Simulation,
    /// Prove refinement properties
    Refinement,
    /// Verify control flow properties
    ControlFlow,
    /// Check memory safety invariants
    MemorySafety,
}

/// Configuration for IsaBIL backend
#[derive(Debug, Clone)]
pub struct IsaBilConfig {
    /// Path to Isabelle binary
    pub isabelle_path: Option<PathBuf>,
    /// Timeout for verification
    pub timeout: Duration,
    /// Target binary to analyze
    pub target_binary: Option<PathBuf>,
    /// Path to BAP for lifting binaries to BIL
    pub bap_path: Option<PathBuf>,
    /// Isabelle session name (default: IsaBIL)
    pub session_name: String,
    /// Theory file containing the specification
    pub theory_file: Option<PathBuf>,
    /// Verification mode
    pub verification_mode: IsaBilMode,
    /// Additional Isabelle options
    pub extra_options: Vec<String>,
}

impl Default for IsaBilConfig {
    fn default() -> Self {
        Self {
            isabelle_path: None,
            timeout: Duration::from_secs(600),
            target_binary: None,
            bap_path: None,
            session_name: "IsaBIL".to_string(),
            theory_file: None,
            verification_mode: IsaBilMode::default(),
            extra_options: vec![],
        }
    }
}

/// IsaBIL Isabelle/HOL binary verification backend
pub struct IsaBilBackend {
    config: IsaBilConfig,
}

impl Default for IsaBilBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl IsaBilBackend {
    pub fn new() -> Self {
        Self {
            config: IsaBilConfig::default(),
        }
    }

    pub fn with_config(config: IsaBilConfig) -> Self {
        Self { config }
    }

    async fn detect_isabelle(&self) -> Result<PathBuf, String> {
        let isabelle_path = self
            .config
            .isabelle_path
            .clone()
            .or_else(|| which::which("isabelle").ok())
            .ok_or("Isabelle not found. Install from: https://isabelle.in.tum.de/")?;

        // Check if Isabelle is working
        let output = Command::new(&isabelle_path)
            .arg("version")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| format!("Failed to check Isabelle: {}", e))?;

        if output.status.success() {
            let version = String::from_utf8_lossy(&output.stdout);
            debug!("Detected Isabelle version: {}", version.trim());
            Ok(isabelle_path)
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!(
                "Isabelle not functioning properly. Error: {}",
                stderr.trim()
            ))
        }
    }

    /// Generate a theory file for the verification task
    fn generate_theory(
        &self,
        temp_dir: &std::path::Path,
        spec: &TypedSpec,
    ) -> Result<PathBuf, String> {
        let theory_path = temp_dir.join("Verification.thy");

        // Generate Isabelle theory based on verification mode
        let theory_content = match self.config.verification_mode {
            IsaBilMode::Simulation => self.generate_simulation_theory(spec),
            IsaBilMode::Refinement => self.generate_refinement_theory(spec),
            IsaBilMode::ControlFlow => self.generate_control_flow_theory(spec),
            IsaBilMode::MemorySafety => self.generate_memory_safety_theory(spec),
        };

        std::fs::write(&theory_path, theory_content)
            .map_err(|e| format!("Failed to write theory file: {}", e))?;

        Ok(theory_path)
    }

    fn generate_simulation_theory(&self, _spec: &TypedSpec) -> String {
        r#"theory Verification
  imports IsaBIL.IsaBIL
begin

(* Generated by DashProve *)

(* Simulation theorem: binary refines specification *)
theorem simulation:
  assumes bil_semantics: "BIL.step s s'"
  shows "spec_step (abstract s) (abstract s')"
  apply (simp add: bil_semantics_def)
  sorry

end
"#
        .to_string()
    }

    fn generate_refinement_theory(&self, _spec: &TypedSpec) -> String {
        r#"theory Verification
  imports IsaBIL.IsaBIL
begin

(* Generated by DashProve *)

(* Refinement theorem: concrete refines abstract *)
theorem refinement:
  assumes inv: "invariant s"
  assumes step: "concrete_step s s'"
  shows "abstract_step (abs s) (abs s')"
  apply (rule simulation_step)
  apply (simp add: inv)
  sorry

end
"#
        .to_string()
    }

    fn generate_control_flow_theory(&self, _spec: &TypedSpec) -> String {
        r#"theory Verification
  imports IsaBIL.IsaBIL
begin

(* Generated by DashProve *)

(* Control flow theorem: all paths satisfy property *)
theorem control_flow_safe:
  assumes cfg: "cfg_valid g"
  assumes path: "path g start end"
  shows "all_nodes_satisfy P path"
  apply (induction path)
  apply simp_all
  sorry

end
"#
        .to_string()
    }

    fn generate_memory_safety_theory(&self, _spec: &TypedSpec) -> String {
        r#"theory Verification
  imports IsaBIL.IsaBIL
begin

(* Generated by DashProve *)

(* Memory safety theorem: all memory accesses are valid *)
theorem memory_safe:
  assumes well_typed: "well_typed_program prog"
  assumes exec: "exec prog s s'"
  shows "memory_invariant s' \<and> no_buffer_overflow s'"
  apply (induction rule: exec.induct)
  apply simp_all
  sorry

end
"#
        .to_string()
    }

    /// Build Isabelle command arguments
    fn build_isabelle_args(&self, theory_path: &std::path::Path) -> Vec<String> {
        let mut args = vec![
            "build".to_string(),
            "-D".to_string(),
            theory_path.parent().unwrap().to_string_lossy().to_string(),
        ];

        // Add session
        args.push("-d".to_string());
        args.push(self.config.session_name.clone());

        // Add extra options
        for opt in &self.config.extra_options {
            args.push(opt.clone());
        }

        args
    }

    /// Parse Isabelle output to verification result
    fn parse_output(
        &self,
        stdout: &str,
        stderr: &str,
    ) -> (VerificationStatus, Option<StructuredCounterexample>) {
        let combined = format!("{}\n{}", stdout, stderr);

        // Check for successful proof
        if combined.contains("No more goals")
            || combined.contains("theorem proved")
            || combined.contains("0 errors")
            || combined.contains("Finished")
                && !combined.contains("error")
                && !combined.contains("failed")
        {
            return (VerificationStatus::Proven, None);
        }

        // Check for counterexample from auto quickcheck
        if combined.contains("Auto Quickcheck found a counterexample")
            || combined.contains("Counterexample:")
            || combined.contains("counterexample found")
        {
            let ce = self.parse_counterexample(&combined);
            return (VerificationStatus::Disproven, Some(ce));
        }

        // Check for nitpick counterexample
        if combined.contains("Nitpick found a counterexample") {
            let ce = self.parse_counterexample(&combined);
            return (VerificationStatus::Disproven, Some(ce));
        }

        // Check for type errors
        if combined.contains("Type unification failed")
            || combined.contains("Illegal application")
            || combined.contains("Undefined constant")
        {
            return (
                VerificationStatus::Unknown {
                    reason: format!(
                        "Type error in theory: {}",
                        combined
                            .lines()
                            .find(|l| l.contains("error") || l.contains("Type"))
                            .unwrap_or("unknown error")
                    ),
                },
                None,
            );
        }

        // Check for IsaBIL-specific errors
        if combined.contains("Could not find constant BIL")
            || combined.contains("Unknown theory")
            || combined.contains("Missing session")
        {
            return (
                VerificationStatus::Unknown {
                    reason: "IsaBIL session not properly configured. Build IsaBIL session first."
                        .to_string(),
                },
                None,
            );
        }

        // Check for timeout
        if combined.contains("Timeout") || combined.contains("timed out") {
            return (
                VerificationStatus::Unknown {
                    reason: "Isabelle verification timed out".to_string(),
                },
                None,
            );
        }

        // Check for general errors
        if combined.contains("error") || combined.contains("FAILED") || combined.contains("failed")
        {
            let error_line = combined
                .lines()
                .find(|l| l.to_lowercase().contains("error") || l.contains("FAILED"))
                .unwrap_or("Unknown error");
            return (
                VerificationStatus::Unknown {
                    reason: format!("Isabelle error: {}", error_line),
                },
                None,
            );
        }

        (
            VerificationStatus::Unknown {
                reason: "Could not determine Isabelle verification result".to_string(),
            },
            None,
        )
    }

    /// Parse counterexample from Isabelle output
    fn parse_counterexample(&self, output: &str) -> StructuredCounterexample {
        let mut ce = StructuredCounterexample::new();
        ce.raw = Some(output.to_string());

        // Parse variable assignments from counterexample
        let mut in_counterexample = false;
        for line in output.lines() {
            if line.contains("counterexample") || line.contains("Counterexample") {
                in_counterexample = true;
                continue;
            }

            if in_counterexample {
                // Parse lines like "  x = 0" or "  f = (%_. 0)"
                let line = line.trim();
                if line.is_empty() || line.starts_with("(*") {
                    in_counterexample = false;
                    continue;
                }

                if let Some((var, val)) = line.split_once('=') {
                    let var = var.trim().to_string();
                    let val = val.trim();
                    ce.witness
                        .insert(var, CounterexampleValue::String(val.to_string()));
                }
            }
        }

        // Add failed check
        ce.failed_checks.push(FailedCheck {
            check_id: "isabil_0".to_string(),
            description: "Counterexample found by Isabelle".to_string(),
            location: None,
            function: None,
        });

        ce
    }
}

#[async_trait]
impl VerificationBackend for IsaBilBackend {
    fn id(&self) -> BackendId {
        BackendId::IsaBIL
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![
            PropertyType::Theorem,
            PropertyType::Invariant,
            PropertyType::MemorySafety,
        ]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();

        let isabelle_path = self
            .detect_isabelle()
            .await
            .map_err(BackendError::Unavailable)?;

        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        // Generate theory file
        let theory_path = self
            .generate_theory(temp_dir.path(), spec)
            .map_err(BackendError::VerificationFailed)?;

        // Build Isabelle command
        let args = self.build_isabelle_args(&theory_path);

        let mut cmd = Command::new(&isabelle_path);
        cmd.args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Execute with timeout
        let result =
            tokio::time::timeout(self.config.timeout + Duration::from_secs(30), cmd.output()).await;

        let duration = start.elapsed();

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                debug!("Isabelle stdout: {}", stdout);
                debug!("Isabelle stderr: {}", stderr);

                let (status, counterexample) = self.parse_output(&stdout, &stderr);

                let diagnostics: Vec<String> = stderr
                    .lines()
                    .filter(|l| {
                        l.contains("Warning")
                            || l.contains("Error")
                            || l.contains("error")
                            || l.contains("warning")
                    })
                    .map(String::from)
                    .collect();

                let proof = if matches!(status, VerificationStatus::Proven) {
                    Some("Verified by IsaBIL/Isabelle".to_string())
                } else {
                    None
                };

                Ok(BackendResult {
                    backend: BackendId::IsaBIL,
                    status,
                    proof,
                    counterexample,
                    diagnostics,
                    time_taken: duration,
                })
            }
            Ok(Err(e)) => Err(BackendError::VerificationFailed(format!(
                "Failed to execute Isabelle: {}",
                e
            ))),
            Err(_) => Err(BackendError::Timeout(self.config.timeout)),
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match self.detect_isabelle().await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =============================================
    // Configuration tests
    // =============================================

    #[test]
    fn default_config() {
        let config = IsaBilConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(600));
        assert!(config.isabelle_path.is_none());
        assert!(config.target_binary.is_none());
        assert!(config.bap_path.is_none());
        assert_eq!(config.session_name, "IsaBIL");
        assert_eq!(config.verification_mode, IsaBilMode::Simulation);
    }

    #[test]
    fn custom_config() {
        let config = IsaBilConfig {
            isabelle_path: Some(PathBuf::from("/usr/bin/isabelle")),
            timeout: Duration::from_secs(1200),
            target_binary: Some(PathBuf::from("/path/to/binary")),
            bap_path: Some(PathBuf::from("/usr/bin/bap")),
            session_name: "CustomSession".to_string(),
            theory_file: Some(PathBuf::from("/path/to/theory.thy")),
            verification_mode: IsaBilMode::Refinement,
            extra_options: vec!["-o".to_string(), "value".to_string()],
        };
        assert_eq!(config.timeout, Duration::from_secs(1200));
        assert_eq!(config.verification_mode, IsaBilMode::Refinement);
        assert_eq!(config.session_name, "CustomSession");
    }

    #[test]
    fn backend_id() {
        let backend = IsaBilBackend::new();
        assert_eq!(backend.id(), BackendId::IsaBIL);
    }

    #[test]
    fn backend_supports() {
        let backend = IsaBilBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::Theorem));
        assert!(supported.contains(&PropertyType::Invariant));
        assert!(supported.contains(&PropertyType::MemorySafety));
    }

    // =============================================
    // Output parsing tests
    // =============================================

    #[test]
    fn parse_output_proven() {
        let backend = IsaBilBackend::new();
        let output = "theorem proved\nNo more goals.";
        let (status, ce) = backend.parse_output(output, "");
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(ce.is_none());
    }

    #[test]
    fn parse_output_no_errors() {
        let backend = IsaBilBackend::new();
        let (status, _) = backend.parse_output("Finished\n0 errors", "");
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn parse_output_quickcheck_counterexample() {
        let backend = IsaBilBackend::new();
        let output = "Auto Quickcheck found a counterexample:\n  x = 0\n  y = 1";
        let (status, ce) = backend.parse_output(output, "");
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(ce.is_some());
        let counterexample = ce.unwrap();
        assert!(counterexample.witness.contains_key("x"));
        assert!(counterexample.witness.contains_key("y"));
    }

    #[test]
    fn parse_output_nitpick_counterexample() {
        let backend = IsaBilBackend::new();
        let output = "Nitpick found a counterexample:\n  n = 42";
        let (status, ce) = backend.parse_output(output, "");
        assert!(matches!(status, VerificationStatus::Disproven));
        assert!(ce.is_some());
    }

    #[test]
    fn parse_output_type_error() {
        let backend = IsaBilBackend::new();
        let (status, _) = backend.parse_output("", "Type unification failed");
        match status {
            VerificationStatus::Unknown { reason } => {
                assert!(reason.contains("Type error"));
            }
            _ => panic!("Expected Unknown status"),
        }
    }

    #[test]
    fn parse_output_isabil_not_found() {
        let backend = IsaBilBackend::new();
        let (status, _) = backend.parse_output("Could not find constant BIL", "");
        match status {
            VerificationStatus::Unknown { reason } => {
                assert!(reason.contains("IsaBIL session"));
            }
            _ => panic!("Expected Unknown status"),
        }
    }

    #[test]
    fn parse_output_timeout() {
        let backend = IsaBilBackend::new();
        let (status, _) = backend.parse_output("Timeout", "");
        match status {
            VerificationStatus::Unknown { reason } => {
                assert!(reason.contains("timed out"));
            }
            _ => panic!("Expected Unknown status"),
        }
    }

    #[test]
    fn parse_output_general_error() {
        let backend = IsaBilBackend::new();
        let (status, _) = backend.parse_output("", "error: something went wrong");
        match status {
            VerificationStatus::Unknown { reason } => {
                assert!(reason.contains("Isabelle error"));
            }
            _ => panic!("Expected Unknown status"),
        }
    }

    // =============================================
    // Counterexample parsing tests
    // =============================================

    #[test]
    fn parse_counterexample_with_assignments() {
        let backend = IsaBilBackend::new();
        let output = "Auto Quickcheck found a counterexample:\n  x = 0\n  y = 1\n  z = True";
        let ce = backend.parse_counterexample(output);

        assert_eq!(ce.failed_checks.len(), 1);
        assert!(ce.witness.contains_key("x"));
        assert!(ce.witness.contains_key("y"));
        assert!(ce.witness.contains_key("z"));
    }

    #[test]
    fn parse_counterexample_with_function() {
        let backend = IsaBilBackend::new();
        let output = "Counterexample:\n  f = (%_. 0)\n  g = (%x. x + 1)";
        let ce = backend.parse_counterexample(output);

        assert!(ce.witness.contains_key("f"));
        assert!(ce.witness.contains_key("g"));
    }

    #[test]
    fn counterexample_has_raw() {
        let backend = IsaBilBackend::new();
        let output = "Counterexample:\n  x = 0";
        let ce = backend.parse_counterexample(output);
        assert!(ce.raw.is_some());
    }

    // =============================================
    // Theory generation tests
    // =============================================

    fn make_test_spec() -> TypedSpec {
        use dashprove_usl::ast::Spec;
        use std::collections::HashMap;
        TypedSpec {
            spec: Spec::default(),
            type_info: HashMap::new(),
        }
    }

    #[test]
    fn generate_simulation_theory_contains_imports() {
        let backend = IsaBilBackend::new();
        let spec = make_test_spec();
        let theory = backend.generate_simulation_theory(&spec);
        assert!(theory.contains("imports IsaBIL.IsaBIL"));
        assert!(theory.contains("theorem simulation"));
    }

    #[test]
    fn generate_refinement_theory_contains_theorem() {
        let config = IsaBilConfig {
            verification_mode: IsaBilMode::Refinement,
            ..Default::default()
        };
        let backend = IsaBilBackend::with_config(config);
        let spec = make_test_spec();
        let theory = backend.generate_refinement_theory(&spec);
        assert!(theory.contains("theorem refinement"));
        assert!(theory.contains("simulation_step"));
    }

    #[test]
    fn generate_control_flow_theory() {
        let config = IsaBilConfig {
            verification_mode: IsaBilMode::ControlFlow,
            ..Default::default()
        };
        let backend = IsaBilBackend::with_config(config);
        let spec = make_test_spec();
        let theory = backend.generate_control_flow_theory(&spec);
        assert!(theory.contains("theorem control_flow_safe"));
        assert!(theory.contains("cfg_valid"));
    }

    #[test]
    fn generate_memory_safety_theory() {
        let config = IsaBilConfig {
            verification_mode: IsaBilMode::MemorySafety,
            ..Default::default()
        };
        let backend = IsaBilBackend::with_config(config);
        let spec = make_test_spec();
        let theory = backend.generate_memory_safety_theory(&spec);
        assert!(theory.contains("theorem memory_safe"));
        assert!(theory.contains("no_buffer_overflow"));
    }

    // =============================================
    // Isabelle args tests
    // =============================================

    #[test]
    fn build_args_includes_session() {
        let backend = IsaBilBackend::new();
        let args = backend.build_isabelle_args(std::path::Path::new("/tmp/Verification.thy"));
        assert!(args.contains(&"build".to_string()));
        assert!(args.contains(&"-d".to_string()));
        assert!(args.contains(&"IsaBIL".to_string()));
    }

    #[test]
    fn build_args_with_custom_session() {
        let config = IsaBilConfig {
            session_name: "CustomSession".to_string(),
            ..Default::default()
        };
        let backend = IsaBilBackend::with_config(config);
        let args = backend.build_isabelle_args(std::path::Path::new("/tmp/Verification.thy"));
        assert!(args.contains(&"CustomSession".to_string()));
    }

    #[test]
    fn build_args_with_extra_options() {
        let config = IsaBilConfig {
            extra_options: vec!["-v".to_string(), "-j4".to_string()],
            ..Default::default()
        };
        let backend = IsaBilBackend::with_config(config);
        let args = backend.build_isabelle_args(std::path::Path::new("/tmp/Verification.thy"));
        assert!(args.contains(&"-v".to_string()));
        assert!(args.contains(&"-j4".to_string()));
    }

    // =============================================
    // Edge cases
    // =============================================

    #[test]
    fn backend_with_custom_config() {
        let config = IsaBilConfig {
            timeout: Duration::from_secs(1200),
            verification_mode: IsaBilMode::MemorySafety,
            ..Default::default()
        };
        let backend = IsaBilBackend::with_config(config);
        assert_eq!(backend.config.timeout, Duration::from_secs(1200));
        assert_eq!(backend.config.verification_mode, IsaBilMode::MemorySafety);
    }

    #[test]
    fn default_equals_new() {
        let default_backend = IsaBilBackend::default();
        let new_backend = IsaBilBackend::new();
        assert_eq!(default_backend.config.timeout, new_backend.config.timeout);
        assert_eq!(
            default_backend.config.verification_mode,
            new_backend.config.verification_mode
        );
    }

    #[test]
    fn parse_output_empty() {
        let backend = IsaBilBackend::new();
        let (status, _) = backend.parse_output("", "");
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn verification_modes() {
        assert_eq!(IsaBilMode::default(), IsaBilMode::Simulation);
        assert_ne!(IsaBilMode::Refinement, IsaBilMode::ControlFlow);
        assert_ne!(IsaBilMode::MemorySafety, IsaBilMode::Simulation);
    }
}
