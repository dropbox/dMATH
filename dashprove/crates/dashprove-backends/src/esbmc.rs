//! ESBMC (Efficient SMT-based Bounded Model Checker) backend
//!
//! ESBMC is a context-bounded model checker for C/C++ programs.
//! It verifies safety properties using SMT solvers (Z3, Boolector, etc.)
//! with bounded model checking techniques.
//!
//! Key features:
//! - Memory safety (buffer overflows, null pointer dereferences)
//! - Assertion checking
//! - Deadlock detection
//! - Division by zero detection
//! - Array bounds checking
//!
//! See: <http://www.esbmc.org/>

// =============================================
// Kani Proofs for ESBMC Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- EsbmcConfig Default Tests ----

    /// Verify EsbmcConfig::default timeout is 120 seconds
    #[kani::proof]
    fn proof_esbmc_config_default_timeout() {
        let config = EsbmcConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(120),
            "Default timeout should be 120 seconds",
        );
    }

    /// Verify EsbmcConfig::default esbmc_path is None
    #[kani::proof]
    fn proof_esbmc_config_default_path_none() {
        let config = EsbmcConfig::default();
        kani::assert(
            config.esbmc_path.is_none(),
            "Default esbmc_path should be None",
        );
    }

    /// Verify EsbmcConfig::default unwind is 10
    #[kani::proof]
    fn proof_esbmc_config_default_unwind() {
        let config = EsbmcConfig::default();
        kani::assert(config.unwind == 10, "Default unwind should be 10");
    }

    /// Verify EsbmcConfig::default memory_safety is true
    #[kani::proof]
    fn proof_esbmc_config_default_memory_safety_true() {
        let config = EsbmcConfig::default();
        kani::assert(config.memory_safety, "Default memory_safety should be true");
    }

    /// Verify EsbmcConfig::default overflow_check is true
    #[kani::proof]
    fn proof_esbmc_config_default_overflow_check_true() {
        let config = EsbmcConfig::default();
        kani::assert(
            config.overflow_check,
            "Default overflow_check should be true",
        );
    }

    /// Verify EsbmcConfig::default bounds_check is true
    #[kani::proof]
    fn proof_esbmc_config_default_bounds_check_true() {
        let config = EsbmcConfig::default();
        kani::assert(config.bounds_check, "Default bounds_check should be true");
    }

    /// Verify EsbmcConfig::default div_by_zero_check is true
    #[kani::proof]
    fn proof_esbmc_config_default_div_by_zero_true() {
        let config = EsbmcConfig::default();
        kani::assert(
            config.div_by_zero_check,
            "Default div_by_zero_check should be true",
        );
    }

    /// Verify EsbmcConfig::default solver is "z3"
    #[kani::proof]
    fn proof_esbmc_config_default_solver() {
        let config = EsbmcConfig::default();
        kani::assert(config.solver == "z3", "Default solver should be z3");
    }

    /// Verify EsbmcConfig::default witness is true
    #[kani::proof]
    fn proof_esbmc_config_default_witness_true() {
        let config = EsbmcConfig::default();
        kani::assert(config.witness, "Default witness should be true");
    }

    // ---- EsbmcBackend Construction Tests ----

    /// Verify EsbmcBackend::new uses default config timeout
    #[kani::proof]
    fn proof_esbmc_backend_new_default_timeout() {
        let backend = EsbmcBackend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(120),
            "New backend should use default timeout",
        );
    }

    /// Verify EsbmcBackend::default equals EsbmcBackend::new
    #[kani::proof]
    fn proof_esbmc_backend_default_equals_new() {
        let default_backend = EsbmcBackend::default();
        let new_backend = EsbmcBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "Default and new should have same timeout",
        );
    }

    /// Verify EsbmcBackend::with_config preserves custom timeout
    #[kani::proof]
    fn proof_esbmc_backend_with_config_timeout() {
        let config = EsbmcConfig {
            esbmc_path: None,
            timeout: Duration::from_secs(240),
            unwind: 10,
            memory_safety: true,
            overflow_check: true,
            bounds_check: true,
            div_by_zero_check: true,
            solver: "z3".to_string(),
            witness: true,
        };
        let backend = EsbmcBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(240),
            "with_config should preserve timeout",
        );
    }

    /// Verify EsbmcBackend::with_config preserves custom unwind
    #[kani::proof]
    fn proof_esbmc_backend_with_config_unwind() {
        let config = EsbmcConfig {
            esbmc_path: None,
            timeout: Duration::from_secs(120),
            unwind: 20,
            memory_safety: true,
            overflow_check: true,
            bounds_check: true,
            div_by_zero_check: true,
            solver: "z3".to_string(),
            witness: true,
        };
        let backend = EsbmcBackend::with_config(config);
        kani::assert(
            backend.config.unwind == 20,
            "with_config should preserve unwind",
        );
    }

    // ---- Backend Trait Tests ----

    /// Verify EsbmcBackend::id returns ESBMC
    #[kani::proof]
    fn proof_esbmc_backend_id() {
        let backend = EsbmcBackend::new();
        kani::assert(
            backend.id() == BackendId::ESBMC,
            "Backend id should be ESBMC",
        );
    }

    /// Verify EsbmcBackend::supports includes MemorySafety
    #[kani::proof]
    fn proof_esbmc_backend_supports_memory_safety() {
        let backend = EsbmcBackend::new();
        let supported = backend.supports();
        let has_memory = supported.iter().any(|p| *p == PropertyType::MemorySafety);
        kani::assert(has_memory, "Should support MemorySafety property");
    }

    /// Verify EsbmcBackend::supports includes Invariant
    #[kani::proof]
    fn proof_esbmc_backend_supports_invariant() {
        let backend = EsbmcBackend::new();
        let supported = backend.supports();
        let has_invariant = supported.iter().any(|p| *p == PropertyType::Invariant);
        kani::assert(has_invariant, "Should support Invariant property");
    }

    /// Verify EsbmcBackend::supports includes Contract
    #[kani::proof]
    fn proof_esbmc_backend_supports_contract() {
        let backend = EsbmcBackend::new();
        let supported = backend.supports();
        let has_contract = supported.iter().any(|p| *p == PropertyType::Contract);
        kani::assert(has_contract, "Should support Contract property");
    }

    /// Verify EsbmcBackend::supports returns exactly 3 properties
    #[kani::proof]
    fn proof_esbmc_backend_supports_length() {
        let backend = EsbmcBackend::new();
        let supported = backend.supports();
        kani::assert(supported.len() == 3, "Should support exactly 3 properties");
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

/// Configuration for ESBMC backend
#[derive(Debug, Clone)]
pub struct EsbmcConfig {
    /// Path to ESBMC binary
    pub esbmc_path: Option<PathBuf>,
    /// Timeout for verification
    pub timeout: Duration,
    /// Unwind bound for loops
    pub unwind: u32,
    /// Enable memory safety checks
    pub memory_safety: bool,
    /// Enable overflow checks
    pub overflow_check: bool,
    /// Enable array bounds checking
    pub bounds_check: bool,
    /// Enable division by zero checks
    pub div_by_zero_check: bool,
    /// SMT solver to use (z3, boolector, mathsat, yices)
    pub solver: String,
    /// Generate counterexample witness
    pub witness: bool,
}

impl Default for EsbmcConfig {
    fn default() -> Self {
        Self {
            esbmc_path: None,
            timeout: Duration::from_secs(120),
            unwind: 10,
            memory_safety: true,
            overflow_check: true,
            bounds_check: true,
            div_by_zero_check: true,
            solver: "z3".to_string(),
            witness: true,
        }
    }
}

/// ESBMC model checker backend
pub struct EsbmcBackend {
    config: EsbmcConfig,
}

impl Default for EsbmcBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl EsbmcBackend {
    pub fn new() -> Self {
        Self {
            config: EsbmcConfig::default(),
        }
    }

    pub fn with_config(config: EsbmcConfig) -> Self {
        Self { config }
    }

    async fn detect_esbmc(&self) -> Result<PathBuf, String> {
        if let Some(path) = &self.config.esbmc_path {
            if path.exists() {
                return Ok(path.clone());
            }
        }

        if let Ok(path) = which::which("esbmc") {
            // Verify it's working
            let output = Command::new(&path)
                .arg("--version")
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output()
                .await
                .map_err(|e| format!("Failed to execute ESBMC: {}", e))?;

            let stdout = String::from_utf8_lossy(&output.stdout);
            if stdout.contains("ESBMC") || output.status.success() {
                debug!("Detected ESBMC: {}", stdout.trim());
                return Ok(path);
            }
        }

        // Try common paths
        let home = std::env::var("HOME").unwrap_or_default();
        for base in &[
            "/usr/local/bin/esbmc",
            "/opt/esbmc/bin/esbmc",
            &format!("{}/esbmc/bin/esbmc", home),
            &format!("{}/.local/bin/esbmc", home),
        ] {
            let p = PathBuf::from(base);
            if p.exists() {
                return Ok(p);
            }
        }

        Err("ESBMC not found. Build from http://www.esbmc.org/".to_string())
    }

    /// Generate C code from USL spec for ESBMC verification
    fn generate_c_code(&self, spec: &TypedSpec) -> String {
        let mut code = String::new();

        code.push_str("// Generated by DashProve for ESBMC verification\n");
        code.push_str("#include <assert.h>\n");
        code.push_str("#include <stdlib.h>\n");
        code.push('\n');
        code.push_str("// ESBMC nondet functions\n");
        code.push_str("int __VERIFIER_nondet_int(void);\n");
        code.push_str("unsigned int __VERIFIER_nondet_uint(void);\n");
        code.push_str("void __VERIFIER_assume(int);\n");
        code.push_str("void __VERIFIER_error(void);\n");
        code.push('\n');

        code.push_str("int main(void) {\n");

        if spec.spec.properties.is_empty() {
            // Default: check for common safety properties
            code.push_str("    int x = __VERIFIER_nondet_int();\n");
            code.push_str("    int y = __VERIFIER_nondet_int();\n");
            code.push_str("    \n");
            code.push_str("    // Assume valid inputs\n");
            code.push_str("    __VERIFIER_assume(y != 0);\n");
            code.push_str("    \n");
            code.push_str("    // Check overflow\n");
            code.push_str("    int sum = x + y;\n");
            code.push_str("    \n");
            code.push_str("    // Check division\n");
            code.push_str("    int div = x / y;\n");
            code.push_str("    \n");
            code.push_str("    // Array bounds check\n");
            code.push_str("    int arr[10];\n");
            code.push_str("    int idx = __VERIFIER_nondet_int();\n");
            code.push_str("    __VERIFIER_assume(idx >= 0 && idx < 10);\n");
            code.push_str("    arr[idx] = 42;\n");
            code.push_str("    \n");
            code.push_str("    return 0;\n");
        } else {
            for (idx, prop) in spec.spec.properties.iter().enumerate() {
                code.push_str(&format!("    // Property {}: {}\n", idx, prop.name()));
                code.push_str("    int value = __VERIFIER_nondet_int();\n");
                code.push_str(&format!(
                    "    assert(value >= 0); // Property: {}\n",
                    prop.name()
                ));
            }
            code.push_str("    return 0;\n");
        }

        code.push_str("}\n");
        code
    }

    /// Parse ESBMC output
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
        let mut diagnostics = Vec::new();

        // Collect warnings
        for line in combined.lines() {
            if line.contains("warning:") || line.contains("error:") && !line.contains("ERROR:") {
                diagnostics.push(line.trim().to_string());
            }
        }

        // ESBMC output patterns:
        // "VERIFICATION SUCCESSFUL" - all properties hold
        // "VERIFICATION FAILED" - counterexample found
        // "VERIFICATION UNKNOWN" - timeout or undecidable

        let successful = combined.contains("VERIFICATION SUCCESSFUL");
        let failed = combined.contains("VERIFICATION FAILED");

        let status = if successful && !failed {
            VerificationStatus::Proven
        } else if failed {
            VerificationStatus::Disproven
        } else if combined.contains("VERIFICATION UNKNOWN") {
            VerificationStatus::Unknown {
                reason: "ESBMC could not determine result".to_string(),
            }
        } else if combined.contains("Timed out") || combined.contains("timeout") {
            VerificationStatus::Unknown {
                reason: "ESBMC timed out".to_string(),
            }
        } else if combined.contains("Out of memory") || combined.contains("memory") {
            VerificationStatus::Unknown {
                reason: "ESBMC ran out of memory".to_string(),
            }
        } else if combined.contains("Parsing error") || combined.contains("syntax error") {
            VerificationStatus::Unknown {
                reason: "ESBMC encountered a parsing error".to_string(),
            }
        } else {
            VerificationStatus::Unknown {
                reason: "Could not parse ESBMC output".to_string(),
            }
        };

        let counterexample = if matches!(status, VerificationStatus::Disproven) {
            self.extract_counterexample(&combined)
        } else {
            None
        };

        (status, counterexample, diagnostics)
    }

    /// Extract counterexample from ESBMC output
    fn extract_counterexample(&self, output: &str) -> Option<StructuredCounterexample> {
        let mut witness = HashMap::new();
        let mut trace = Vec::new();
        let mut failed_checks = Vec::new();

        // Parse violation type
        // Pattern: "Violated property:" or specific error types
        if output.contains("array bounds violated") || output.contains("array index out of bounds")
        {
            failed_checks.push(FailedCheck {
                check_id: "esbmc_bounds".to_string(),
                description: "Array bounds violation".to_string(),
                location: None,
                function: None,
            });
        }

        if output.contains("division by zero") {
            failed_checks.push(FailedCheck {
                check_id: "esbmc_div_zero".to_string(),
                description: "Division by zero".to_string(),
                location: None,
                function: None,
            });
        }

        if output.contains("dereference failure") || output.contains("NULL pointer") {
            failed_checks.push(FailedCheck {
                check_id: "esbmc_null_deref".to_string(),
                description: "Null pointer dereference".to_string(),
                location: None,
                function: None,
            });
        }

        if output.contains("overflow") {
            failed_checks.push(FailedCheck {
                check_id: "esbmc_overflow".to_string(),
                description: "Integer overflow".to_string(),
                location: None,
                function: None,
            });
        }

        if output.contains("assertion") {
            failed_checks.push(FailedCheck {
                check_id: "esbmc_assertion".to_string(),
                description: "Assertion violation".to_string(),
                location: None,
                function: None,
            });
        }

        // Parse location
        // Pattern: "file input.c line N column M function F"
        let loc_re = Regex::new(
            r"file\s+(\S+)\s+line\s+(\d+)(?:\s+column\s+(\d+))?(?:\s+function\s+(\w+))?",
        )
        .ok()?;
        if let Some(cap) = loc_re.captures(output) {
            let file = cap
                .get(1)
                .map(|m| m.as_str().to_string())
                .unwrap_or_default();
            let line: u32 = cap
                .get(2)
                .and_then(|m| m.as_str().parse().ok())
                .unwrap_or(0);
            let col = cap.get(3).and_then(|m| m.as_str().parse().ok());
            let func = cap.get(4).map(|m| m.as_str().to_string());

            // Update first failed check with location
            if !failed_checks.is_empty() {
                failed_checks[0].location = Some(SourceLocation {
                    file,
                    line,
                    column: col,
                });
                failed_checks[0].function = func;
            }
        }

        // Parse counterexample values
        // ESBMC format: "State N file X line Y function Z"
        // followed by variable assignments
        let state_re = Regex::new(r"State\s+(\d+)").ok()?;
        let assign_re = Regex::new(r"(\w+)=(-?\d+)").ok()?;
        let hex_re = Regex::new(r"(\w+)=0x([0-9a-fA-F]+)").ok()?;

        let mut current_state: Option<u32> = None;
        let mut current_vars = HashMap::new();

        for line in output.lines() {
            // Check for state marker
            if let Some(cap) = state_re.captures(line) {
                // Save previous state
                if let Some(state_num) = current_state {
                    if !current_vars.is_empty() {
                        trace.push(TraceState {
                            state_num,
                            action: None,
                            variables: current_vars.clone(),
                        });
                    }
                }

                current_state = cap.get(1).and_then(|m| m.as_str().parse().ok());
                current_vars.clear();
            }

            // Parse hex assignments
            for cap in hex_re.captures_iter(line) {
                let var = cap.get(1).unwrap().as_str().to_string();
                let val_str = cap.get(2).unwrap().as_str();
                if let Ok(val) = u128::from_str_radix(val_str, 16) {
                    let value = CounterexampleValue::UInt {
                        value: val,
                        type_hint: None,
                    };
                    current_vars.insert(var.clone(), value.clone());
                    witness.insert(var, value);
                }
            }

            // Parse decimal assignments
            for cap in assign_re.captures_iter(line) {
                let var = cap.get(1).unwrap().as_str().to_string();
                // Skip if already parsed as hex
                if current_vars.contains_key(&var) {
                    continue;
                }
                let val_str = cap.get(2).unwrap().as_str();
                if let Ok(val) = val_str.parse::<i128>() {
                    let value = CounterexampleValue::Int {
                        value: val,
                        type_hint: None,
                    };
                    current_vars.insert(var.clone(), value.clone());
                    witness.insert(var, value);
                }
            }
        }

        // Save final state
        if let Some(state_num) = current_state {
            if !current_vars.is_empty() {
                trace.push(TraceState {
                    state_num,
                    action: None,
                    variables: current_vars,
                });
            }
        }

        // Ensure we have at least one failed check
        if failed_checks.is_empty() {
            failed_checks.push(FailedCheck {
                check_id: "esbmc_failure".to_string(),
                description: "ESBMC found a counterexample".to_string(),
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
impl VerificationBackend for EsbmcBackend {
    fn id(&self) -> BackendId {
        BackendId::ESBMC
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![
            PropertyType::MemorySafety,
            PropertyType::Invariant,
            PropertyType::Contract,
        ]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();
        let esbmc = self
            .detect_esbmc()
            .await
            .map_err(BackendError::Unavailable)?;

        // Create temp directory
        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {e}"))
        })?;
        let source_path = temp_dir.path().join("input.c");

        // Write source
        std::fs::write(&source_path, self.generate_c_code(spec)).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write C source: {e}"))
        })?;

        // Build ESBMC command
        let mut cmd = Command::new(&esbmc);
        cmd.arg(&source_path)
            .arg("--unwind")
            .arg(self.config.unwind.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Add solver option
        match self.config.solver.as_str() {
            "z3" => {
                cmd.arg("--z3");
            }
            "boolector" => {
                cmd.arg("--boolector");
            }
            "mathsat" => {
                cmd.arg("--mathsat");
            }
            "yices" => {
                cmd.arg("--yices");
            }
            _ => {
                cmd.arg("--z3");
            }
        }

        // Add property checks
        if self.config.memory_safety {
            cmd.arg("--memory-leak-check");
        }
        if self.config.overflow_check {
            cmd.arg("--overflow-check");
        }
        if self.config.bounds_check {
            cmd.arg("--bounds-check");
        }
        if self.config.div_by_zero_check {
            cmd.arg("--div-by-zero-check");
        }
        if self.config.witness {
            cmd.arg("--witness-output")
                .arg(temp_dir.path().join("witness.graphml"));
        }

        // Run with timeout
        let output = tokio::time::timeout(self.config.timeout, cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| BackendError::VerificationFailed(format!("ESBMC failed: {e}")))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        debug!("ESBMC stdout: {}", stdout);
        if !stderr.trim().is_empty() {
            debug!("ESBMC stderr: {}", stderr);
        }

        let (status, counterexample, diagnostics) = self.parse_output(&stdout, &stderr);
        let proof = if matches!(status, VerificationStatus::Proven) {
            Some("ESBMC verified all properties within bounds".to_string())
        } else {
            None
        };

        Ok(BackendResult {
            backend: BackendId::ESBMC,
            status,
            proof,
            counterexample,
            diagnostics,
            time_taken: start.elapsed(),
        })
    }

    async fn health_check(&self) -> HealthStatus {
        match self.detect_esbmc().await {
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
        let config = EsbmcConfig::default();
        assert_eq!(config.unwind, 10);
        assert!(config.memory_safety);
        assert!(config.overflow_check);
        assert_eq!(config.solver, "z3");
    }

    #[test]
    fn backend_id() {
        let backend = EsbmcBackend::new();
        assert_eq!(backend.id(), BackendId::ESBMC);
    }

    #[test]
    fn supports_properties() {
        let backend = EsbmcBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::MemorySafety));
        assert!(supported.contains(&PropertyType::Invariant));
    }

    #[test]
    fn c_code_generation_empty_spec() {
        let backend = EsbmcBackend::new();
        let spec = TypedSpec {
            spec: Spec {
                types: vec![],
                properties: vec![],
            },
            type_info: HashMap::new(),
        };
        let code = backend.generate_c_code(&spec);
        assert!(code.contains("__VERIFIER_nondet_int"));
        assert!(code.contains("__VERIFIER_assume"));
        assert!(code.contains("int arr[10]"));
    }

    #[test]
    fn c_code_generation_with_property() {
        let backend = EsbmcBackend::new();
        let spec = TypedSpec {
            spec: Spec {
                types: vec![],
                properties: vec![Property::Invariant(Invariant {
                    name: "positive".to_string(),
                    body: Expr::Bool(true),
                })],
            },
            type_info: HashMap::new(),
        };
        let code = backend.generate_c_code(&spec);
        assert!(code.contains("positive"));
        assert!(code.contains("assert"));
    }

    #[test]
    fn parse_output_successful() {
        let backend = EsbmcBackend::new();
        let stdout = "Starting bounded model checking\nVERIFICATION SUCCESSFUL";
        let (status, cex, _) = backend.parse_output(stdout, "");
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(cex.is_none());
    }

    #[test]
    fn parse_output_failed() {
        let backend = EsbmcBackend::new();
        let stdout = "VERIFICATION FAILED\narray bounds violated\nfile input.c line 15";
        let (status, cex, _) = backend.parse_output(stdout, "");
        assert!(matches!(status, VerificationStatus::Disproven));
        let ce = cex.expect("expected counterexample");
        assert!(!ce.failed_checks.is_empty());
    }

    #[test]
    fn parse_output_unknown() {
        let backend = EsbmcBackend::new();
        let stdout = "VERIFICATION UNKNOWN";
        let (status, _, _) = backend.parse_output(stdout, "");
        match status {
            VerificationStatus::Unknown { reason } => {
                assert!(reason.contains("determine"));
            }
            _ => panic!("Expected Unknown status"),
        }
    }

    #[test]
    fn parse_output_timeout() {
        let backend = EsbmcBackend::new();
        let stdout = "Timed out";
        let (status, _, _) = backend.parse_output(stdout, "");
        match status {
            VerificationStatus::Unknown { reason } => {
                assert!(reason.to_lowercase().contains("timeout") || reason.contains("timed"));
            }
            _ => panic!("Expected Unknown status"),
        }
    }

    #[test]
    fn extract_counterexample_bounds() {
        let backend = EsbmcBackend::new();
        let output =
            "VERIFICATION FAILED\narray bounds violated\nfile input.c line 15 column 5 function main";
        let cex = backend.extract_counterexample(output).unwrap();
        assert!(cex
            .failed_checks
            .iter()
            .any(|c| c.check_id == "esbmc_bounds"));
        assert!(cex.failed_checks[0].location.is_some());
    }

    #[test]
    fn extract_counterexample_div_zero() {
        let backend = EsbmcBackend::new();
        let output = "VERIFICATION FAILED\ndivision by zero";
        let cex = backend.extract_counterexample(output).unwrap();
        assert!(cex
            .failed_checks
            .iter()
            .any(|c| c.check_id == "esbmc_div_zero"));
    }

    #[test]
    fn extract_counterexample_null_deref() {
        let backend = EsbmcBackend::new();
        let output = "VERIFICATION FAILED\ndereference failure: NULL pointer";
        let cex = backend.extract_counterexample(output).unwrap();
        assert!(cex
            .failed_checks
            .iter()
            .any(|c| c.check_id == "esbmc_null_deref"));
    }

    #[test]
    fn extract_counterexample_overflow() {
        let backend = EsbmcBackend::new();
        let output = "VERIFICATION FAILED\narithmetic overflow";
        let cex = backend.extract_counterexample(output).unwrap();
        assert!(cex
            .failed_checks
            .iter()
            .any(|c| c.check_id == "esbmc_overflow"));
    }

    #[test]
    fn extract_counterexample_with_values() {
        let backend = EsbmcBackend::new();
        let output =
            "VERIFICATION FAILED\nState 1\n  x=42\n  y=-10\n  ptr=0x1000\nState 2\n  x=100";
        let cex = backend.extract_counterexample(output).unwrap();
        assert!(cex.witness.contains_key("x"));
        assert!(cex.witness.contains_key("y"));
        assert!(!cex.trace.is_empty());
    }

    #[test]
    fn extract_counterexample_hex_values() {
        let backend = EsbmcBackend::new();
        let output = "VERIFICATION FAILED\nState 1\naddr=0xDEADBEEF";
        let cex = backend.extract_counterexample(output).unwrap();
        assert!(cex.witness.contains_key("addr"));
    }

    #[tokio::test]
    async fn health_check_unavailable() {
        let config = EsbmcConfig {
            esbmc_path: Some(PathBuf::from("/nonexistent/esbmc")),
            ..Default::default()
        };
        let backend = EsbmcBackend::with_config(config);
        let health = backend.health_check().await;
        assert!(matches!(health, HealthStatus::Unavailable { .. }));
    }
}
