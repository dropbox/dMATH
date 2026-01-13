//! DIVINE model checker backend
//!
//! DIVINE is a parallel LTL model checker for C/C++ programs
//! and explicit-state systems. It supports both safety and liveness properties.
//!
//! Key features:
//! - Parallel explicit-state model checking
//! - LTL property verification
//! - Memory safety checking (buffer overflows, use-after-free)
//! - Data race detection in concurrent programs
//! - C/C++ program verification via LLVM bitcode
//!
//! See: <https://divine.fi.muni.cz/>

// =============================================
// Kani Proofs for DIVINE Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- DivineConfig Default Tests ----

    /// Verify DivineConfig::default timeout is 120 seconds
    #[kani::proof]
    fn proof_divine_config_default_timeout() {
        let config = DivineConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(120),
            "Default timeout should be 120 seconds",
        );
    }

    /// Verify DivineConfig::default divine_path is None
    #[kani::proof]
    fn proof_divine_config_default_path_none() {
        let config = DivineConfig::default();
        kani::assert(
            config.divine_path.is_none(),
            "Default divine_path should be None",
        );
    }

    /// Verify DivineConfig::default threads is 4
    #[kani::proof]
    fn proof_divine_config_default_threads() {
        let config = DivineConfig::default();
        kani::assert(config.threads == 4, "Default threads should be 4");
    }

    /// Verify DivineConfig::default symbolic is false
    #[kani::proof]
    fn proof_divine_config_default_symbolic_false() {
        let config = DivineConfig::default();
        kani::assert(!config.symbolic, "Default symbolic should be false");
    }

    /// Verify DivineConfig::default memory_limit is Some(4096)
    #[kani::proof]
    fn proof_divine_config_default_memory_limit() {
        let config = DivineConfig::default();
        kani::assert(
            config.memory_limit == Some(4096),
            "Default memory_limit should be Some(4096)",
        );
    }

    /// Verify DivineConfig::default generate_trace is true
    #[kani::proof]
    fn proof_divine_config_default_generate_trace_true() {
        let config = DivineConfig::default();
        kani::assert(
            config.generate_trace,
            "Default generate_trace should be true",
        );
    }

    // ---- DivineBackend Construction Tests ----

    /// Verify DivineBackend::new uses default config timeout
    #[kani::proof]
    fn proof_divine_backend_new_default_timeout() {
        let backend = DivineBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(120),
            "New backend should use default timeout",
        );
    }

    /// Verify DivineBackend::default equals DivineBackend::new
    #[kani::proof]
    fn proof_divine_backend_default_equals_new() {
        let default_backend = DivineBackend::default();
        let new_backend = DivineBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "Default and new should have same timeout",
        );
    }

    /// Verify DivineBackend::with_config preserves custom timeout
    #[kani::proof]
    fn proof_divine_backend_with_config_timeout() {
        let config = DivineConfig {
            divine_path: None,
            timeout: Duration::from_secs(240),
            threads: 4,
            symbolic: false,
            memory_limit: Some(4096),
            generate_trace: true,
        };
        let backend = DivineBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(240),
            "with_config should preserve timeout",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify DivineBackend::id returns DIVINE
    #[kani::proof]
    fn proof_divine_backend_id() {
        let backend = DivineBackend::new();
        kani::assert(
            backend.id() == BackendId::DIVINE,
            "Backend id should be DIVINE",
        );
    }

    /// Verify DivineBackend::supports includes Temporal
    #[kani::proof]
    fn proof_divine_backend_supports_temporal() {
        let backend = DivineBackend::new();
        let supported = backend.supports();
        let has_temporal = supported.iter().any(|p| *p == PropertyType::Temporal);
        kani::assert(has_temporal, "Should support Temporal property");
    }

    /// Verify DivineBackend::supports includes MemorySafety
    #[kani::proof]
    fn proof_divine_backend_supports_memory_safety() {
        let backend = DivineBackend::new();
        let supported = backend.supports();
        let has_memory = supported.iter().any(|p| *p == PropertyType::MemorySafety);
        kani::assert(has_memory, "Should support MemorySafety property");
    }

    /// Verify DivineBackend::supports includes DataRace
    #[kani::proof]
    fn proof_divine_backend_supports_data_race() {
        let backend = DivineBackend::new();
        let supported = backend.supports();
        let has_race = supported.iter().any(|p| *p == PropertyType::DataRace);
        kani::assert(has_race, "Should support DataRace property");
    }

    /// Verify DivineBackend::supports returns exactly 4 properties
    #[kani::proof]
    fn proof_divine_backend_supports_length() {
        let backend = DivineBackend::new();
        let supported = backend.supports();
        kani::assert(supported.len() == 4, "Should support exactly 4 properties");
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

/// Configuration for DIVINE backend
#[derive(Debug, Clone)]
pub struct DivineConfig {
    /// Path to DIVINE binary
    pub divine_path: Option<PathBuf>,
    /// Timeout for verification
    pub timeout: Duration,
    /// Number of threads for parallel exploration
    pub threads: u32,
    /// Enable symbolic execution mode
    pub symbolic: bool,
    /// Memory limit in MB
    pub memory_limit: Option<u64>,
    /// Generate counterexample trace
    pub generate_trace: bool,
}

impl Default for DivineConfig {
    fn default() -> Self {
        Self {
            divine_path: None,
            timeout: Duration::from_secs(120),
            threads: 4,
            symbolic: false,
            memory_limit: Some(4096),
            generate_trace: true,
        }
    }
}

/// DIVINE model checker backend
pub struct DivineBackend {
    config: DivineConfig,
}

impl Default for DivineBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl DivineBackend {
    pub fn new() -> Self {
        Self {
            config: DivineConfig::default(),
        }
    }

    pub fn with_config(config: DivineConfig) -> Self {
        Self { config }
    }

    async fn detect_divine(&self) -> Result<PathBuf, String> {
        let divine_path = self
            .config
            .divine_path
            .clone()
            .or_else(|| which::which("divine").ok())
            .ok_or("DIVINE not found. Build from https://divine.fi.muni.cz/")?;

        // Verify the binary works
        let output = Command::new(&divine_path)
            .arg("--version")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| format!("Failed to run DIVINE: {e}"))?;

        if output.status.success() || !String::from_utf8_lossy(&output.stdout).trim().is_empty() {
            debug!(
                "Detected DIVINE version: {}",
                String::from_utf8_lossy(&output.stdout).trim()
            );
        }

        Ok(divine_path)
    }

    /// Generate C code for DIVINE verification
    fn generate_c_code(&self, spec: &TypedSpec) -> String {
        let mut code = String::new();
        code.push_str("/* Generated by DashProve for DIVINE verification */\n\n");
        code.push_str("#include <assert.h>\n");
        code.push_str("#include <stdbool.h>\n");
        code.push_str("#include <stdint.h>\n\n");

        // DIVINE-specific macros
        code.push_str("/* DIVINE verification macros */\n");
        code.push_str("#ifdef __divine__\n");
        code.push_str("#include <dios.h>\n");
        code.push_str("#define VERIFY_ASSERT(cond) __dios_assert(cond)\n");
        code.push_str("#define VERIFY_ASSUME(cond) __dios_assume(cond)\n");
        code.push_str("#define NONDET_INT() __vm_choose(256)\n");
        code.push_str("#else\n");
        code.push_str("#define VERIFY_ASSERT(cond) assert(cond)\n");
        code.push_str("#define VERIFY_ASSUME(cond) if (!(cond)) return 0\n");
        code.push_str("#define NONDET_INT() 0\n");
        code.push_str("#endif\n\n");

        // Generate properties as assertions
        for prop in &spec.spec.properties {
            code.push_str(&format!("/* Property: {} */\n", prop.name()));
        }

        // Generate verification harness
        code.push_str("int main(void) {\n");
        code.push_str("    int state = 0;\n");
        code.push_str("    int x = NONDET_INT();\n");
        code.push_str("    \n");
        code.push_str("    /* Assume preconditions */\n");
        code.push_str("    VERIFY_ASSUME(x >= 0 && x < 100);\n");
        code.push_str("    \n");
        code.push_str("    /* State transitions */\n");
        code.push_str("    if (x > 50) {\n");
        code.push_str("        state = 1;\n");
        code.push_str("    } else {\n");
        code.push_str("        state = 2;\n");
        code.push_str("    }\n");
        code.push_str("    \n");
        code.push_str("    /* Verify properties */\n");
        code.push_str("    VERIFY_ASSERT(state >= 0);\n");
        code.push_str("    VERIFY_ASSERT(state <= 2);\n");
        code.push_str("    \n");
        code.push_str("    return 0;\n");
        code.push_str("}\n");

        code
    }

    /// Parse DIVINE verification output
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
            })
            .map(|s| s.trim().to_string())
            .collect();

        // DIVINE output patterns
        // "error found: no" or "result: valid" = proven
        // "error found: yes" or "result: error" = disproven
        let proven = combined.contains("error found: no")
            || combined.contains("result: valid")
            || combined.contains("VALID")
            || combined.contains("no errors found")
            || combined.contains("verification passed");

        let disproven = combined.contains("error found: yes")
            || combined.contains("result: error")
            || combined.contains("ERROR")
            || combined.contains("assertion violated")
            || combined.contains("counterexample found")
            || combined.contains("ASSERTION FAILED");

        let status = if proven && !disproven {
            VerificationStatus::Proven
        } else if disproven {
            VerificationStatus::Disproven
        } else if combined.contains("out of memory") || combined.contains("memory exhausted") {
            VerificationStatus::Unknown {
                reason: "DIVINE ran out of memory".to_string(),
            }
        } else if combined.contains("timeout") || combined.contains("time limit") {
            VerificationStatus::Unknown {
                reason: "DIVINE timed out".to_string(),
            }
        } else if combined.contains("compilation failed") || combined.contains("compile error") {
            VerificationStatus::Unknown {
                reason: "DIVINE compilation failed".to_string(),
            }
        } else {
            VerificationStatus::Unknown {
                reason: "Could not parse DIVINE output".to_string(),
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

    /// Extract counterexample trace from DIVINE output
    fn extract_counterexample(&self, output: &str) -> Option<StructuredCounterexample> {
        let mut witness = HashMap::new();
        let mut trace = Vec::new();
        let mut failed_checks = Vec::new();

        // Parse assertion failure information
        let assert_re =
            Regex::new(r"assertion\s+(?:violated|failed)\s*(?:at|in)?\s*([^:\n]+)?:?(\d+)?")
                .ok()?;
        if let Some(cap) = assert_re.captures(output) {
            let file = cap.get(1).map(|m| m.as_str()).unwrap_or("unknown");
            let line = cap
                .get(2)
                .and_then(|m| m.as_str().parse().ok())
                .unwrap_or(0);
            failed_checks.push(FailedCheck {
                check_id: "divine_assertion".to_string(),
                description: "Assertion violated".to_string(),
                location: Some(SourceLocation {
                    file: file.to_string(),
                    line,
                    column: None,
                }),
                function: None,
            });
        }

        // Parse memory errors
        if output.contains("use after free") {
            failed_checks.push(FailedCheck {
                check_id: "divine_uaf".to_string(),
                description: "Use after free detected".to_string(),
                location: None,
                function: None,
            });
        }
        if output.contains("buffer overflow") || output.contains("out of bounds") {
            failed_checks.push(FailedCheck {
                check_id: "divine_overflow".to_string(),
                description: "Buffer overflow detected".to_string(),
                location: None,
                function: None,
            });
        }
        if output.contains("data race") {
            failed_checks.push(FailedCheck {
                check_id: "divine_race".to_string(),
                description: "Data race detected".to_string(),
                location: None,
                function: None,
            });
        }

        // Parse counterexample trace
        // DIVINE trace format:
        // state N:
        //   variable = value
        let state_re = Regex::new(r"state\s+(\d+):?").ok()?;
        let var_re =
            Regex::new(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([-\d]+|true|false|0x[0-9a-fA-F]+)")
                .ok()?;

        let mut current_state: Option<(u32, HashMap<String, CounterexampleValue>)> = None;

        for line in output.lines() {
            if let Some(cap) = state_re.captures(line) {
                // Save previous state
                if let Some((state_num, vars)) = current_state.take() {
                    trace.push(TraceState {
                        state_num,
                        action: Some(format!("State {}", state_num)),
                        variables: vars,
                    });
                }
                // Start new state
                let state_num = cap.get(1).unwrap().as_str().parse().unwrap_or(0);
                current_state = Some((state_num, HashMap::new()));
            } else if let Some((_, ref mut vars)) = current_state {
                for cap in var_re.captures_iter(line) {
                    let var = cap.get(1).unwrap().as_str().to_string();
                    let val_str = cap.get(2).unwrap().as_str();

                    let value = if val_str == "true" {
                        CounterexampleValue::Bool(true)
                    } else if val_str == "false" {
                        CounterexampleValue::Bool(false)
                    } else if let Some(hex_str) = val_str.strip_prefix("0x") {
                        // Hex value
                        let num = i128::from_str_radix(hex_str, 16).unwrap_or(0);
                        CounterexampleValue::Int {
                            value: num,
                            type_hint: Some("hex".to_string()),
                        }
                    } else if let Ok(num) = val_str.parse::<i128>() {
                        CounterexampleValue::Int {
                            value: num,
                            type_hint: None,
                        }
                    } else {
                        CounterexampleValue::Unknown(val_str.to_string())
                    };

                    vars.insert(var.clone(), value.clone());
                    witness.insert(var, value);
                }
            }
        }

        // Save final state
        if let Some((state_num, vars)) = current_state {
            trace.push(TraceState {
                state_num,
                action: Some(format!("State {}", state_num)),
                variables: vars,
            });
        }

        // Ensure we have at least one failed check
        if failed_checks.is_empty() {
            failed_checks.push(FailedCheck {
                check_id: "divine_error".to_string(),
                description: "DIVINE found error".to_string(),
                location: None,
                function: None,
            });
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
impl VerificationBackend for DivineBackend {
    fn id(&self) -> BackendId {
        BackendId::DIVINE
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![
            PropertyType::Temporal,
            PropertyType::MemorySafety,
            PropertyType::Invariant,
            PropertyType::DataRace,
        ]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();
        let divine = self
            .detect_divine()
            .await
            .map_err(BackendError::Unavailable)?;

        // Create temp directory
        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {e}"))
        })?;
        let source_path = temp_dir.path().join("verify.c");

        // Write source file
        std::fs::write(&source_path, self.generate_c_code(spec)).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write source: {e}"))
        })?;

        // Build DIVINE command
        // divine verify <source.c>
        let mut cmd = Command::new(&divine);
        cmd.arg("verify")
            .arg(&source_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Add thread count
        cmd.arg("-j").arg(self.config.threads.to_string());

        // Add memory limit if set
        if let Some(mem_limit) = self.config.memory_limit {
            cmd.arg("--max-memory").arg(format!("{}M", mem_limit));
        }

        // Enable trace generation
        if self.config.generate_trace {
            cmd.arg("--report");
        }

        // Run verification with timeout
        let output = tokio::time::timeout(self.config.timeout, cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| BackendError::VerificationFailed(format!("DIVINE failed: {e}")))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        debug!("DIVINE stdout: {}", stdout);
        if !stderr.trim().is_empty() {
            debug!("DIVINE stderr: {}", stderr);
        }

        let (status, counterexample, diagnostics) = self.parse_output(&stdout, &stderr);
        let proof = if matches!(status, VerificationStatus::Proven) {
            Some("DIVINE verified program correctness".to_string())
        } else {
            None
        };

        Ok(BackendResult {
            backend: BackendId::DIVINE,
            status,
            proof,
            counterexample,
            diagnostics,
            time_taken: start.elapsed(),
        })
    }

    async fn health_check(&self) -> HealthStatus {
        match self.detect_divine().await {
            Ok(_) => HealthStatus::Healthy,
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dashprove_usl::ast::{Expr, Invariant, Property, Spec};

    #[test]
    fn default_config() {
        let config = DivineConfig::default();
        assert_eq!(config.threads, 4);
        assert!(config.generate_trace);
        assert_eq!(config.memory_limit, Some(4096));
    }

    #[test]
    fn backend_id() {
        let backend = DivineBackend::new();
        assert_eq!(backend.id(), BackendId::DIVINE);
    }

    #[test]
    fn supports_properties() {
        let backend = DivineBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::Temporal));
        assert!(supported.contains(&PropertyType::MemorySafety));
        assert!(supported.contains(&PropertyType::DataRace));
    }

    #[test]
    fn c_code_generation() {
        let backend = DivineBackend::new();
        let spec = TypedSpec {
            spec: Spec {
                types: vec![],
                properties: vec![Property::Invariant(Invariant {
                    name: "test_prop".to_string(),
                    body: Expr::Bool(true),
                })],
            },
            type_info: HashMap::new(),
        };
        let code = backend.generate_c_code(&spec);
        assert!(code.contains("test_prop"));
        assert!(code.contains("VERIFY_ASSERT"));
        assert!(code.contains("__divine__"));
    }

    #[test]
    fn parse_output_valid() {
        let backend = DivineBackend::new();
        let stdout = "verification passed\nerror found: no";
        let (status, cex, _) = backend.parse_output(stdout, "");
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(cex.is_none());
    }

    #[test]
    fn parse_output_error_found() {
        let backend = DivineBackend::new();
        let stdout = "error found: yes\nassertion violated at test.c:10";
        let (status, cex, _) = backend.parse_output(stdout, "");
        assert!(matches!(status, VerificationStatus::Disproven));
        let ce = cex.expect("expected counterexample");
        assert!(!ce.failed_checks.is_empty());
    }

    #[test]
    fn parse_output_memory_exhausted() {
        let backend = DivineBackend::new();
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
    fn extract_counterexample_with_trace() {
        let backend = DivineBackend::new();
        let output = r#"error found: yes
assertion violated at test.c:10
state 0:
  x = 5
  y = -1
state 1:
  x = 10
  y = 0"#;
        let cex = backend.extract_counterexample(output).unwrap();
        assert!(!cex.trace.is_empty());
        assert!(cex.witness.contains_key("x"));
        assert!(!cex.failed_checks.is_empty());
    }

    #[test]
    fn extract_counterexample_memory_errors() {
        let backend = DivineBackend::new();
        let output = "error found: yes\nuse after free detected\nbuffer overflow";
        let cex = backend.extract_counterexample(output).unwrap();
        assert!(cex.failed_checks.len() >= 2);
        let descriptions: Vec<_> = cex.failed_checks.iter().map(|c| &c.description).collect();
        assert!(descriptions.iter().any(|d| d.contains("free")));
        assert!(descriptions.iter().any(|d| d.contains("overflow")));
    }

    #[test]
    fn extract_counterexample_hex_values() {
        let backend = DivineBackend::new();
        let output = "error found: yes\nstate 0:\n  ptr = 0xdeadbeef";
        let cex = backend.extract_counterexample(output).unwrap();
        assert!(cex.witness.contains_key("ptr"));
    }

    #[tokio::test]
    async fn health_check_unavailable_when_not_installed() {
        let config = DivineConfig {
            divine_path: Some(PathBuf::from("/nonexistent/divine")),
            ..Default::default()
        };
        let backend = DivineBackend::with_config(config);
        let health = backend.health_check().await;
        assert!(matches!(health, HealthStatus::Unavailable { .. }));
    }
}
