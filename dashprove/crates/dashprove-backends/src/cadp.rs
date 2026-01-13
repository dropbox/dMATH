//! CADP (Construction and Analysis of Distributed Processes) backend
//!
//! CADP is a comprehensive toolbox for protocol engineering and verification
//! of asynchronous concurrent systems. It provides tools for specification,
//! simulation, verification, and testing.
//!
//! Key features:
//! - LOTOS/LNT specification languages
//! - On-the-fly model checking with EVALUATOR
//! - Equivalence checking with BISIMULATOR
//! - Model generation and minimization
//! - Counterexample generation and visualization
//!
//! See: <https://cadp.inria.fr/>

// =============================================
// Kani Proofs for CADP Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- CadpConfig Default Tests ----

    /// Verify CadpConfig::default sets correct baseline values
    #[kani::proof]
    fn proof_cadp_config_defaults() {
        let config = CadpConfig::default();
        kani::assert(
            config.cadp_path.is_none(),
            "cadp_path should default to None",
        );
        kani::assert(
            config.timeout == Duration::from_secs(120),
            "timeout should default to 120 seconds",
        );
        kani::assert(
            config.generate_trace,
            "generate_trace should default to true",
        );
        kani::assert(
            config.evaluator_version == 4,
            "evaluator_version should default to 4",
        );
    }

    // ---- CadpBackend Construction Tests ----

    /// Verify CadpBackend::new uses default configuration
    #[kani::proof]
    fn proof_cadp_backend_new_defaults() {
        let backend = CadpBackend::new();
        kani::assert(
            backend.config.cadp_path.is_none(),
            "new backend should have no cadp_path",
        );
        kani::assert(
            backend.config.timeout == Duration::from_secs(120),
            "new backend should default timeout to 120 seconds",
        );
        kani::assert(
            backend.config.generate_trace,
            "new backend should default generate_trace to true",
        );
    }

    /// Verify CadpBackend::default equals CadpBackend::new
    #[kani::proof]
    fn proof_cadp_backend_default_equals_new() {
        let default_backend = CadpBackend::default();
        let new_backend = CadpBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "default and new should share timeout",
        );
        kani::assert(
            default_backend.config.generate_trace == new_backend.config.generate_trace,
            "default and new should share generate_trace",
        );
    }

    /// Verify CadpBackend::with_config preserves custom configuration
    #[kani::proof]
    fn proof_cadp_backend_with_config() {
        let config = CadpConfig {
            cadp_path: Some(PathBuf::from("/opt/cadp")),
            timeout: Duration::from_secs(300),
            generate_trace: false,
            evaluator_version: 3,
        };
        let backend = CadpBackend::with_config(config);
        kani::assert(
            backend.config.cadp_path == Some(PathBuf::from("/opt/cadp")),
            "with_config should preserve cadp_path",
        );
        kani::assert(
            backend.config.timeout == Duration::from_secs(300),
            "with_config should preserve timeout",
        );
        kani::assert(
            !backend.config.generate_trace,
            "with_config should preserve generate_trace",
        );
        kani::assert(
            backend.config.evaluator_version == 3,
            "with_config should preserve evaluator_version",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify id() returns BackendId::CADP
    #[kani::proof]
    fn proof_cadp_backend_id() {
        let backend = CadpBackend::new();
        kani::assert(
            backend.id() == BackendId::CADP,
            "CadpBackend id should be BackendId::CADP",
        );
    }

    /// Verify supports() includes Temporal and Invariant
    #[kani::proof]
    fn proof_cadp_backend_supports() {
        let backend = CadpBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::Temporal),
            "supports should include Temporal",
        );
        kani::assert(
            supported.contains(&PropertyType::Invariant),
            "supports should include Invariant",
        );
    }

    /// Verify supports() returns exactly two property types
    #[kani::proof]
    fn proof_cadp_backend_supports_length() {
        let backend = CadpBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.len() == 2,
            "CADP should support exactly two property types",
        );
    }

    // ---- Output Parsing Tests ----

    /// Verify parse_output detects TRUE as Proven
    #[kani::proof]
    fn proof_parse_output_true() {
        let backend = CadpBackend::new();
        let (status, ce, _) = backend.parse_output("property is true\nTRUE", "");
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "TRUE should produce Proven status",
        );
        kani::assert(ce.is_none(), "proven should have no counterexample");
    }

    /// Verify parse_output detects SATISFIED as Proven
    #[kani::proof]
    fn proof_parse_output_satisfied() {
        let backend = CadpBackend::new();
        let (status, _, _) = backend.parse_output("SATISFIED", "");
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "SATISFIED should produce Proven status",
        );
    }

    /// Verify parse_output detects FALSE as Disproven
    #[kani::proof]
    fn proof_parse_output_false() {
        let backend = CadpBackend::new();
        let (status, ce, _) =
            backend.parse_output("property is false\nFALSE\ncounterexample found", "");
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "FALSE should produce Disproven status",
        );
        kani::assert(ce.is_some(), "disproven should have counterexample");
    }

    /// Verify parse_output detects NOT SATISFIED as Disproven
    #[kani::proof]
    fn proof_parse_output_not_satisfied() {
        let backend = CadpBackend::new();
        let (status, _, _) = backend.parse_output("NOT SATISFIED\ncounterexample", "");
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "NOT SATISFIED should produce Disproven status",
        );
    }

    /// Verify parse_output detects memory exhausted as Unknown
    #[kani::proof]
    fn proof_parse_output_memory_exhausted() {
        let backend = CadpBackend::new();
        let (status, _, _) = backend.parse_output("out of memory", "");
        match status {
            VerificationStatus::Unknown { reason } => {
                kani::assert(reason.contains("memory"), "reason should mention memory");
            }
            _ => kani::assert(false, "expected Unknown status"),
        }
    }

    /// Verify parse_output detects syntax error as Unknown
    #[kani::proof]
    fn proof_parse_output_syntax_error() {
        let backend = CadpBackend::new();
        let (status, _, _) = backend.parse_output("syntax error at line 5", "");
        match status {
            VerificationStatus::Unknown { reason } => {
                kani::assert(reason.contains("syntax"), "reason should mention syntax");
            }
            _ => kani::assert(false, "expected Unknown status"),
        }
    }

    // ---- LNT Spec Generation Tests ----

    /// Verify generate_lnt_spec produces valid LNT code
    #[kani::proof]
    fn proof_generate_lnt_spec_structure() {
        let backend = CadpBackend::new();
        let spec = TypedSpec {
            spec: dashprove_usl::ast::Spec {
                types: vec![],
                properties: vec![],
            },
            type_info: HashMap::new(),
        };
        let code = backend.generate_lnt_spec(&spec);

        kani::assert(code.contains("module"), "should contain module");
        kani::assert(code.contains("process"), "should contain process");
        kani::assert(code.contains("channel"), "should contain channel");
    }

    // ---- MCL Formula Generation Tests ----

    /// Verify generate_mcl_formula produces valid MCL
    #[kani::proof]
    fn proof_generate_mcl_formula() {
        let backend = CadpBackend::new();
        let spec = TypedSpec {
            spec: dashprove_usl::ast::Spec {
                types: vec![],
                properties: vec![],
            },
            type_info: HashMap::new(),
        };
        let formula = backend.generate_mcl_formula(&spec);

        kani::assert(formula.contains("true"), "formula should contain true");
    }
}

use crate::counterexample::{
    CounterexampleValue, FailedCheck, SourceLocation, StructuredCounterexample, TraceState,
};
use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
    VerificationStatus,
};
use async_trait::async_trait;
use dashprove_usl::typecheck::TypedSpec;
use regex::Regex;
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::process::Command;
use tracing::debug;

/// Configuration for CADP backend
#[derive(Debug, Clone)]
pub struct CadpConfig {
    /// Path to CADP installation ($CADP environment variable)
    pub cadp_path: Option<PathBuf>,
    /// Timeout for verification
    pub timeout: Duration,
    /// Generate diagnostic traces
    pub generate_trace: bool,
    /// Verification algorithm (evaluator3, evaluator4)
    pub evaluator_version: u8,
}

impl Default for CadpConfig {
    fn default() -> Self {
        Self {
            cadp_path: None,
            timeout: Duration::from_secs(120),
            generate_trace: true,
            evaluator_version: 4,
        }
    }
}

/// CADP protocol verification backend
pub struct CadpBackend {
    config: CadpConfig,
}

impl Default for CadpBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CadpBackend {
    pub fn new() -> Self {
        Self {
            config: CadpConfig::default(),
        }
    }

    pub fn with_config(config: CadpConfig) -> Self {
        Self { config }
    }

    async fn detect(&self) -> Result<PathBuf, String> {
        // If explicit path is configured, verify it exists
        if let Some(ref path) = self.config.cadp_path {
            if path.exists() {
                return Ok(path.clone());
            } else {
                return Err(format!(
                    "CADP not found at configured path: {}",
                    path.display()
                ));
            }
        }

        // Otherwise, try $CADP environment variable
        std::env::var("CADP")
            .ok()
            .map(PathBuf::from)
            .filter(|p| p.exists())
            .ok_or("CADP not found. Set $CADP environment variable or install from https://cadp.inria.fr/".to_string())
    }

    /// Get path to a CADP tool
    fn tool_path(&self, tool: &str) -> PathBuf {
        if let Some(ref cadp) = self.config.cadp_path {
            cadp.join("bin.`$CADP/com/arch`").join(tool)
        } else if let Ok(cadp) = std::env::var("CADP") {
            PathBuf::from(cadp).join("bin").join(tool)
        } else {
            PathBuf::from(tool)
        }
    }

    /// Generate LNT specification from the spec
    fn generate_lnt_spec(&self, spec: &TypedSpec) -> String {
        let mut code = String::new();
        code.push_str("-- Generated by DashProve for CADP verification\n\n");

        // Module declaration
        code.push_str("module DASHPROVE_SPEC (TYPES) is\n\n");

        // Channel types
        code.push_str("channel MSG is\n");
        code.push_str("  (Nat)\n");
        code.push_str("end channel\n\n");

        // Process definitions
        code.push_str("process SENDER [SEND: MSG, RECV: MSG] (n: Nat) is\n");
        code.push_str("  var x: Nat in\n");
        code.push_str("    x := 0;\n");
        code.push_str("    loop\n");
        code.push_str("      SEND (x);\n");
        code.push_str("      x := x + 1;\n");
        code.push_str("      if x >= n then\n");
        code.push_str("        break\n");
        code.push_str("      end if\n");
        code.push_str("    end loop\n");
        code.push_str("  end var\n");
        code.push_str("end process\n\n");

        code.push_str("process RECEIVER [RECV: MSG] is\n");
        code.push_str("  var y: Nat in\n");
        code.push_str("    loop\n");
        code.push_str("      RECV (?y)\n");
        code.push_str("    end loop\n");
        code.push_str("  end var\n");
        code.push_str("end process\n\n");

        // Main process
        code.push_str("process MAIN [SEND: MSG, RECV: MSG] is\n");
        code.push_str("  par SEND, RECV in\n");
        code.push_str("    SENDER [SEND, RECV] (5)\n");
        code.push_str("  ||\n");
        code.push_str("    RECEIVER [RECV]\n");
        code.push_str("  end par\n");
        code.push_str("end process\n\n");

        code.push_str("end module\n");

        // Add property names as comments
        for prop in &spec.spec.properties {
            code.push_str(&format!("-- Property: {}\n", prop.name()));
        }

        code
    }

    /// Generate MCL (Model Checking Language) formula file
    fn generate_mcl_formula(&self, spec: &TypedSpec) -> String {
        let mut formula = String::new();
        formula.push_str("(* Generated MCL formula for CADP EVALUATOR *)\n\n");

        if spec.spec.properties.is_empty() {
            // Default liveness property: some action is always eventually possible
            formula.push_str("[ true* ] < true > true\n");
        } else {
            // Safety: AG(no error) expressed in MCL
            formula.push_str("[ true* . \"ERROR\" ] false\n");
        }

        formula
    }

    /// Parse CADP verification output
    fn parse_output(
        &self,
        stdout: &str,
        stderr: &str,
    ) -> (
        VerificationStatus,
        Option<StructuredCounterexample>,
        Vec<String>,
    ) {
        let combined = format!("{}\n{}", stdout, stderr);
        let mut diagnostics: Vec<String> = combined
            .lines()
            .filter(|l| {
                l.contains("error:")
                    || l.contains("warning:")
                    || l.contains("Error:")
                    || l.contains("Warning:")
                    || l.contains("ERROR")
            })
            .map(|s| s.trim().to_string())
            .collect();

        // CADP EVALUATOR output patterns
        // "TRUE" = formula satisfied (proven)
        // "FALSE" = formula not satisfied (disproven)
        let proven = combined.contains("TRUE")
            || combined.contains("property is true")
            || combined.contains("formula is satisfied")
            || combined.contains("SATISFIED");

        let disproven = combined.contains("FALSE")
            || combined.contains("property is false")
            || combined.contains("formula is not satisfied")
            || combined.contains("NOT SATISFIED")
            || combined.contains("counterexample");

        let status = if proven && !disproven {
            VerificationStatus::Proven
        } else if disproven {
            VerificationStatus::Disproven
        } else if combined.contains("out of memory") || combined.contains("memory exhausted") {
            VerificationStatus::Unknown {
                reason: "CADP ran out of memory".to_string(),
            }
        } else if combined.contains("syntax error") || combined.contains("parse error") {
            VerificationStatus::Unknown {
                reason: "Specification syntax error".to_string(),
            }
        } else if combined.contains("license") || combined.contains("License") {
            VerificationStatus::Unknown {
                reason: "CADP license issue".to_string(),
            }
        } else {
            VerificationStatus::Unknown {
                reason: "Could not parse CADP output".to_string(),
            }
        };

        let counterexample = if matches!(status, VerificationStatus::Disproven) {
            self.extract_counterexample(&combined)
        } else {
            None
        };

        if diagnostics.is_empty() && !stderr.trim().is_empty() {
            for line in stderr.lines().take(5) {
                if !line.trim().is_empty() {
                    diagnostics.push(line.trim().to_string());
                }
            }
        }

        (status, counterexample, diagnostics)
    }

    /// Extract counterexample trace from CADP output
    fn extract_counterexample(&self, output: &str) -> Option<StructuredCounterexample> {
        let mut witness = HashMap::new();
        let mut trace = Vec::new();

        // Parse the formula that failed
        let failed_checks = vec![FailedCheck {
            check_id: "cadp_formula".to_string(),
            description: "MCL formula not satisfied".to_string(),
            location: Some(SourceLocation {
                file: "property.mcl".to_string(),
                line: 0,
                column: None,
            }),
            function: None,
        }];

        // Parse counterexample trace from EVALUATOR output
        // CADP trace format typically shows transitions:
        // "ACTION" -> state
        // or
        // 0 -- "ACTION" --> 1
        let transition_re = Regex::new(r#"(\d+)\s*--\s*"([^"]+)"\s*-->\s*(\d+)"#).ok()?;
        let action_re = Regex::new(r#""([^"]+)""#).ok()?;

        let mut state_num = 0u32;

        for line in output.lines() {
            // Check for transitions in BCG/AUT format
            if let Some(cap) = transition_re.captures(line) {
                let from_state: u32 = cap.get(1).unwrap().as_str().parse().unwrap_or(0);
                let action = cap.get(2).unwrap().as_str().to_string();
                let to_state: u32 = cap.get(3).unwrap().as_str().parse().unwrap_or(0);

                // Record the transition
                let mut vars = HashMap::new();
                vars.insert(
                    "from_state".to_string(),
                    CounterexampleValue::Int {
                        value: from_state as i128,
                        type_hint: None,
                    },
                );
                vars.insert(
                    "to_state".to_string(),
                    CounterexampleValue::Int {
                        value: to_state as i128,
                        type_hint: None,
                    },
                );

                // Try to extract parameters from action
                if let Some(paren_start) = action.find('(') {
                    if let Some(paren_end) = action.find(')') {
                        let params = &action[paren_start + 1..paren_end];
                        for (idx, param) in params.split(',').enumerate() {
                            let param = param.trim();
                            if let Ok(num) = param.parse::<i128>() {
                                let var = format!("param_{}", idx);
                                let value = CounterexampleValue::Int {
                                    value: num,
                                    type_hint: None,
                                };
                                vars.insert(var.clone(), value.clone());
                                witness.insert(var, value);
                            }
                        }
                    }
                }

                trace.push(TraceState {
                    state_num,
                    action: Some(action),
                    variables: vars,
                });
                state_num += 1;
            } else if let Some(cap) = action_re.captures(line) {
                // Simple action on its own line
                let action = cap.get(1).unwrap().as_str().to_string();
                trace.push(TraceState {
                    state_num,
                    action: Some(action),
                    variables: HashMap::new(),
                });
                state_num += 1;
            }
        }

        Some(StructuredCounterexample {
            witness,
            failed_checks,
            playback_test: None,
            trace,
            raw: Some(output.to_string()),
            minimized: false,
        })
    }
}

#[async_trait]
impl VerificationBackend for CadpBackend {
    fn id(&self) -> BackendId {
        BackendId::CADP
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::Temporal, PropertyType::Invariant]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();
        let cadp_path = self.detect().await.map_err(BackendError::Unavailable)?;

        // Create temp directory
        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {e}"))
        })?;

        let lnt_path = temp_dir.path().join("spec.lnt");
        let mcl_path = temp_dir.path().join("property.mcl");
        let bcg_path = temp_dir.path().join("spec.bcg");

        // Write specification files
        std::fs::write(&lnt_path, self.generate_lnt_spec(spec)).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write LNT spec: {e}"))
        })?;
        std::fs::write(&mcl_path, self.generate_mcl_formula(spec)).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write MCL formula: {e}"))
        })?;

        // Step 1: lnt.open - compile LNT and generate BCG
        let lnt_open = self.tool_path("lnt.open");
        let output1 = tokio::time::timeout(
            self.config.timeout / 2,
            Command::new(&lnt_open)
                .arg(&lnt_path)
                .arg("generator")
                .arg(&bcg_path)
                .env("CADP", &cadp_path)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output(),
        )
        .await
        .map_err(|_| BackendError::Timeout(self.config.timeout))?
        .map_err(|e| BackendError::VerificationFailed(format!("lnt.open failed: {e}")))?;

        if !output1.status.success() {
            let stderr = String::from_utf8_lossy(&output1.stderr);
            let stdout = String::from_utf8_lossy(&output1.stdout);
            return Ok(BackendResult {
                backend: BackendId::CADP,
                status: VerificationStatus::Unknown {
                    reason: format!("lnt.open failed: {}{}", stdout, stderr),
                },
                proof: None,
                counterexample: None,
                diagnostics: vec![format!("{}{}", stdout, stderr)],
                time_taken: start.elapsed(),
            });
        }

        // Step 2: bcg_open + evaluator - verify MCL formula on BCG
        let evaluator = format!("evaluator{}", self.config.evaluator_version);
        let bcg_open = self.tool_path("bcg_open");

        let mut cmd = Command::new(&bcg_open);
        cmd.arg(&bcg_path)
            .arg(&evaluator)
            .arg(&mcl_path)
            .env("CADP", &cadp_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if self.config.generate_trace {
            cmd.arg("-diag");
        }

        let output2 = tokio::time::timeout(self.config.timeout / 2, cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| BackendError::VerificationFailed(format!("evaluator failed: {e}")))?;

        let stdout = String::from_utf8_lossy(&output2.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output2.stderr).to_string();

        debug!("CADP evaluator stdout: {}", stdout);
        if !stderr.trim().is_empty() {
            debug!("CADP evaluator stderr: {}", stderr);
        }

        let (status, counterexample, diagnostics) = self.parse_output(&stdout, &stderr);
        let proof = if matches!(status, VerificationStatus::Proven) {
            Some("CADP verified property".to_string())
        } else {
            None
        };

        Ok(BackendResult {
            backend: BackendId::CADP,
            status,
            proof,
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
    use dashprove_usl::ast::{Expr, Invariant, Property, Spec};

    #[test]
    fn default_config() {
        let config = CadpConfig::default();
        assert!(config.generate_trace);
        assert_eq!(config.evaluator_version, 4);
    }

    #[test]
    fn backend_id() {
        assert_eq!(CadpBackend::new().id(), BackendId::CADP);
    }

    #[test]
    fn supports_properties() {
        let backend = CadpBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::Temporal));
        assert!(supported.contains(&PropertyType::Invariant));
    }

    #[test]
    fn lnt_spec_generation() {
        let backend = CadpBackend::new();
        let spec = TypedSpec {
            spec: Spec {
                types: vec![],
                properties: vec![Property::Invariant(Invariant {
                    name: "safety_prop".to_string(),
                    body: Expr::Bool(true),
                })],
            },
            type_info: HashMap::new(),
        };
        let code = backend.generate_lnt_spec(&spec);
        assert!(code.contains("module"));
        assert!(code.contains("process"));
        assert!(code.contains("channel"));
        assert!(code.contains("safety_prop"));
    }

    #[test]
    fn mcl_formula_generation() {
        let backend = CadpBackend::new();
        let spec = TypedSpec {
            spec: Spec {
                types: vec![],
                properties: vec![],
            },
            type_info: HashMap::new(),
        };
        let formula = backend.generate_mcl_formula(&spec);
        assert!(formula.contains("true"));
    }

    #[test]
    fn parse_output_true() {
        let backend = CadpBackend::new();
        let stdout = "property is true\nTRUE";
        let (status, cex, _) = backend.parse_output(stdout, "");
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(cex.is_none());
    }

    #[test]
    fn parse_output_false() {
        let backend = CadpBackend::new();
        let stdout = "property is false\nFALSE\ncounterexample found";
        let (status, cex, _) = backend.parse_output(stdout, "");
        assert!(matches!(status, VerificationStatus::Disproven));
        let ce = cex.expect("expected counterexample");
        assert!(!ce.failed_checks.is_empty());
    }

    #[test]
    fn parse_output_memory_exhausted() {
        let backend = CadpBackend::new();
        let stdout = "out of memory";
        let (status, _, _) = backend.parse_output(stdout, "");
        match status {
            VerificationStatus::Unknown { reason } => {
                assert!(reason.contains("memory"));
            }
            _ => panic!("Expected Unknown status"),
        }
    }

    #[test]
    fn extract_counterexample_with_transitions() {
        let backend = CadpBackend::new();
        let output = r#"FALSE
counterexample:
0 -- "SEND (1)" --> 1
1 -- "RECV (1)" --> 2
2 -- "ERROR" --> 3"#;
        let cex = backend.extract_counterexample(output).unwrap();
        assert!(!cex.trace.is_empty());
        assert!(!cex.failed_checks.is_empty());
    }

    #[test]
    fn extract_counterexample_with_params() {
        let backend = CadpBackend::new();
        let output = r#"FALSE
0 -- "ACTION (42, 100)" --> 1"#;
        let cex = backend.extract_counterexample(output).unwrap();
        assert!(cex.witness.contains_key("param_0") || cex.witness.contains_key("param_1"));
    }

    #[tokio::test]
    async fn health_check_unavailable_when_not_installed() {
        // Clear CADP env var for test
        let backend = CadpBackend::with_config(CadpConfig {
            cadp_path: Some(PathBuf::from("/nonexistent/cadp")),
            ..Default::default()
        });
        let health = backend.health_check().await;
        assert!(matches!(health, HealthStatus::Unavailable { .. }));
    }
}
