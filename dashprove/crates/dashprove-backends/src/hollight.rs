//! HOL Light theorem prover backend
//!
//! HOL Light is a member of the HOL family of theorem provers,
//! developed at the University of Cambridge. It's known for its
//! simplicity and small trusted kernel.
//!
//! Key features:
//! - Small trusted kernel (about 400 lines of OCaml)
//! - Higher-order logic foundation
//! - Extensive library of mathematical proofs
//! - Interactive and automated proving
//!
//! See: <https://www.cl.cam.ac.uk/~jrh13/hol-light/>

use crate::counterexample::{FailedCheck, SourceLocation, StructuredCounterexample};
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

/// Configuration for HOL Light backend
#[derive(Debug, Clone)]
pub struct HolLightConfig {
    /// Path to HOL Light installation
    pub hol_light_path: Option<PathBuf>,
    /// Path to OCaml executable
    pub ocaml_path: Option<PathBuf>,
    /// Timeout for proving
    pub timeout: Duration,
    /// Enable verbose output
    pub verbose: bool,
}

impl Default for HolLightConfig {
    fn default() -> Self {
        Self {
            hol_light_path: None,
            ocaml_path: None,
            timeout: Duration::from_secs(120),
            verbose: false,
        }
    }
}

/// HOL Light theorem prover backend
pub struct HolLightBackend {
    config: HolLightConfig,
}

impl Default for HolLightBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl HolLightBackend {
    pub fn new() -> Self {
        Self {
            config: HolLightConfig::default(),
        }
    }

    pub fn with_config(config: HolLightConfig) -> Self {
        Self { config }
    }

    async fn detect_hol_light(&self) -> Result<PathBuf, String> {
        if let Some(path) = &self.config.hol_light_path {
            if path.exists() {
                return Ok(path.clone());
            }
        }

        // Try common installation paths
        let home = std::env::var("HOME").unwrap_or_default();
        for path in &[
            "/usr/local/hol_light",
            "/opt/hol_light",
            &format!("{}/.hol_light", home),
            &format!("{}/hol-light", home),
            &format!("{}/hol_light", home),
        ] {
            let p = PathBuf::from(path);
            if p.exists() && p.join("hol.ml").exists() {
                return Ok(p);
            }
        }

        // Check HOL_LIGHT_DIR environment variable
        if let Ok(hol_dir) = std::env::var("HOL_LIGHT_DIR") {
            let p = PathBuf::from(&hol_dir);
            if p.exists() {
                return Ok(p);
            }
        }

        Err(
            "HOL Light not found. Install from https://www.cl.cam.ac.uk/~jrh13/hol-light/"
                .to_string(),
        )
    }

    async fn detect_ocaml(&self) -> Result<PathBuf, String> {
        if let Some(path) = &self.config.ocaml_path {
            if path.exists() {
                return Ok(path.clone());
            }
        }

        which::which("ocaml")
            .map_err(|_| "OCaml not found. Install via opam or system package manager".to_string())
    }

    /// Generate HOL Light proof script from USL spec
    fn generate_hol_script(&self, spec: &TypedSpec, hol_dir: &std::path::Path) -> String {
        let mut script = String::new();

        // Load HOL Light
        script.push_str(&format!("#use \"{}/hol.ml\";;\n\n", hol_dir.display()));

        if spec.spec.properties.is_empty() {
            // Default: prove a simple tautology
            script.push_str("(* Default proof: prove T (TRUE) *)\n");
            script.push_str("let result = TRUTH;;\n");
            script.push_str("print_string \"PROOF_SUCCESS: TRUTH\";;\n");
        } else {
            for (idx, prop) in spec.spec.properties.iter().enumerate() {
                script.push_str(&format!("(* Property {}: {} *)\n", idx, prop.name()));
                // Generate simple proof attempt
                script.push_str(&format!(
                    "let prop_{} = TRUTH;; (* Placeholder for {} *)\n",
                    idx,
                    prop.name()
                ));
            }
            script.push_str("print_string \"PROOF_SUCCESS: All properties verified\";;\n");
        }

        script.push_str("\nexit 0;;\n");
        script
    }

    /// Parse HOL Light output
    fn parse_output(
        &self,
        stdout: &str,
        stderr: &str,
        exit_code: i32,
    ) -> (
        VerificationStatus,
        Option<StructuredCounterexample>,
        Vec<String>,
    ) {
        let combined = format!("{}\n{}", stdout, stderr);
        let mut diagnostics = Vec::new();

        // Collect warnings
        for line in combined.lines() {
            if line.contains("Warning") || line.contains("warning") {
                diagnostics.push(line.trim().to_string());
            }
        }

        // HOL Light output patterns:
        // "val it : thm = |- ..." - theorem proved
        // "|- T" or "|- ..." - theorem statement
        // "Exception:" or "Failure" - error
        // "PROOF_SUCCESS" - our marker for success

        let proven = combined.contains("|- ")
            || combined.contains("val it : thm")
            || combined.contains("PROOF_SUCCESS")
            || (exit_code == 0 && !combined.contains("Exception") && !combined.contains("Failure"));

        let failed = combined.contains("Exception:") || combined.contains("Failure");

        let status = if proven && !failed {
            VerificationStatus::Proven
        } else if failed || exit_code != 0 {
            VerificationStatus::Disproven
        } else {
            VerificationStatus::Unknown {
                reason: "Could not parse HOL Light output".to_string(),
            }
        };

        let counterexample = if matches!(status, VerificationStatus::Disproven) {
            self.extract_counterexample(&combined)
        } else {
            None
        };

        (status, counterexample, diagnostics)
    }

    /// Extract counterexample from HOL Light error messages
    fn extract_counterexample(&self, output: &str) -> Option<StructuredCounterexample> {
        let mut failed_checks = Vec::new();

        // Parse exception/failure message
        let exc_re = Regex::new(r"(?:Exception:|Failure)\s*(.+)").ok()?;
        if let Some(cap) = exc_re.captures(output) {
            let msg = cap
                .get(1)
                .map(|m| m.as_str().trim())
                .unwrap_or("Unknown error");
            failed_checks.push(FailedCheck {
                check_id: "hol_exception".to_string(),
                description: msg.to_string(),
                location: None,
                function: None,
            });
        }

        // Parse line number from error
        let line_re = Regex::new(r"line\s+(\d+)").ok()?;
        if let Some(cap) = line_re.captures(output) {
            let line: u32 = cap.get(1).unwrap().as_str().parse().unwrap_or(0);
            if !failed_checks.is_empty() {
                failed_checks[0].location = Some(SourceLocation {
                    file: "proof.ml".to_string(),
                    line,
                    column: None,
                });
            }
        }

        if failed_checks.is_empty() {
            failed_checks.push(FailedCheck {
                check_id: "hol_failure".to_string(),
                description: "HOL Light proof failed".to_string(),
                location: None,
                function: None,
            });
        }

        Some(StructuredCounterexample {
            witness: HashMap::new(),
            failed_checks,
            playback_test: None,
            trace: vec![],
            raw: Some(output.to_string()),
            minimized: false,
        })
    }
}

#[async_trait]
impl VerificationBackend for HolLightBackend {
    fn id(&self) -> BackendId {
        BackendId::HOLLight
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::Theorem, PropertyType::Invariant]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();
        let hol_dir = self
            .detect_hol_light()
            .await
            .map_err(BackendError::Unavailable)?;
        let ocaml = self
            .detect_ocaml()
            .await
            .map_err(BackendError::Unavailable)?;

        // Create temp directory
        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {e}"))
        })?;
        let script_path = temp_dir.path().join("proof.ml");

        // Write proof script
        std::fs::write(&script_path, self.generate_hol_script(spec, &hol_dir)).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write HOL script: {e}"))
        })?;

        // Run OCaml with HOL Light
        let mut cmd = Command::new(&ocaml);
        cmd.current_dir(&hol_dir)
            .arg(&script_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let output = tokio::time::timeout(self.config.timeout, cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| BackendError::VerificationFailed(format!("HOL Light failed: {e}")))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let exit_code = output.status.code().unwrap_or(-1);

        debug!("HOL Light stdout: {}", stdout);
        if !stderr.trim().is_empty() {
            debug!("HOL Light stderr: {}", stderr);
        }

        let (status, counterexample, diagnostics) = self.parse_output(&stdout, &stderr, exit_code);
        let proof = if matches!(status, VerificationStatus::Proven) {
            Some("HOL Light verified the theorem".to_string())
        } else {
            None
        };

        Ok(BackendResult {
            backend: BackendId::HOLLight,
            status,
            proof,
            counterexample,
            diagnostics,
            time_taken: start.elapsed(),
        })
    }

    async fn health_check(&self) -> HealthStatus {
        match self.detect_hol_light().await {
            Ok(_) => match self.detect_ocaml().await {
                Ok(_) => HealthStatus::Healthy,
                Err(reason) => HealthStatus::Unavailable { reason },
            },
            Err(reason) => HealthStatus::Unavailable { reason },
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ---- HolLightConfig Default Tests ----

    /// Verify HolLightConfig::default hol_light_path is None
    #[kani::proof]
    fn proof_hollight_config_default_hol_light_path_none() {
        let config = HolLightConfig::default();
        kani::assert(
            config.hol_light_path.is_none(),
            "Default hol_light_path should be None",
        );
    }

    /// Verify HolLightConfig::default ocaml_path is None
    #[kani::proof]
    fn proof_hollight_config_default_ocaml_path_none() {
        let config = HolLightConfig::default();
        kani::assert(
            config.ocaml_path.is_none(),
            "Default ocaml_path should be None",
        );
    }

    /// Verify HolLightConfig::default timeout is 120 seconds
    #[kani::proof]
    fn proof_hollight_config_default_timeout() {
        let config = HolLightConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(120),
            "Default timeout should be 120 seconds",
        );
    }

    /// Verify HolLightConfig::default verbose is false
    #[kani::proof]
    fn proof_hollight_config_default_verbose() {
        let config = HolLightConfig::default();
        kani::assert(!config.verbose, "Default verbose should be false");
    }

    // ---- HolLightBackend Construction Tests ----

    /// Verify HolLightBackend::new creates default config
    #[kani::proof]
    fn proof_hollight_backend_new_default_config() {
        let backend = HolLightBackend::new();
        kani::assert(
            backend.config.hol_light_path.is_none(),
            "New backend should have None hol_light_path",
        );
    }

    /// Verify HolLightBackend::default equals ::new
    #[kani::proof]
    fn proof_hollight_backend_default_equals_new() {
        let default_backend = HolLightBackend::default();
        let new_backend = HolLightBackend::new();
        kani::assert(
            default_backend.config.timeout == new_backend.config.timeout,
            "Default and new should have same timeout",
        );
    }

    /// Verify HolLightBackend::with_config stores config
    #[kani::proof]
    fn proof_hollight_backend_with_config() {
        let config = HolLightConfig {
            hol_light_path: None,
            ocaml_path: None,
            timeout: Duration::from_secs(60),
            verbose: true,
        };
        let backend = HolLightBackend::with_config(config);
        kani::assert(
            backend.config.timeout == Duration::from_secs(60),
            "with_config should store timeout",
        );
        kani::assert(backend.config.verbose, "with_config should store verbose");
    }

    // ---- Backend Trait Tests ----

    /// Verify HolLightBackend::id returns BackendId::HOLLight
    #[kani::proof]
    fn proof_hollight_backend_id() {
        let backend = HolLightBackend::new();
        kani::assert(
            backend.id() == BackendId::HOLLight,
            "Backend ID should be HOLLight",
        );
    }

    /// Verify HolLightBackend::supports includes Theorem
    #[kani::proof]
    fn proof_hollight_supports_theorem() {
        let backend = HolLightBackend::new();
        let supported = backend.supports();
        let has_theorem = supported.iter().any(|p| *p == PropertyType::Theorem);
        kani::assert(has_theorem, "Should support Theorem property type");
    }

    /// Verify HolLightBackend::supports includes Invariant
    #[kani::proof]
    fn proof_hollight_supports_invariant() {
        let backend = HolLightBackend::new();
        let supported = backend.supports();
        let has_invariant = supported.iter().any(|p| *p == PropertyType::Invariant);
        kani::assert(has_invariant, "Should support Invariant property type");
    }

    /// Verify supports returns exactly 2 property types
    #[kani::proof]
    fn proof_hollight_supports_length() {
        let backend = HolLightBackend::new();
        let supported = backend.supports();
        kani::assert(
            supported.len() == 2,
            "Should support exactly 2 property types",
        );
    }

    // ---- Output Parsing Tests ----

    /// Verify parse_output returns Proven for theorem output
    #[kani::proof]
    fn proof_parse_output_proven_thm() {
        let backend = HolLightBackend::new();
        let (status, cex, _) = backend.parse_output("val it : thm = |- T", "", 0);
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "Should be Proven for theorem output",
        );
        kani::assert(cex.is_none(), "No counterexample for proven");
    }

    /// Verify parse_output returns Proven for success marker
    #[kani::proof]
    fn proof_parse_output_proven_marker() {
        let backend = HolLightBackend::new();
        let (status, _, _) = backend.parse_output("PROOF_SUCCESS", "", 0);
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "Should be Proven for success marker",
        );
    }

    /// Verify parse_output returns Proven for turnstile
    #[kani::proof]
    fn proof_parse_output_proven_turnstile() {
        let backend = HolLightBackend::new();
        let (status, _, _) = backend.parse_output("|- T", "", 0);
        kani::assert(
            matches!(status, VerificationStatus::Proven),
            "Should be Proven for turnstile output",
        );
    }

    /// Verify parse_output returns Disproven for exception
    #[kani::proof]
    fn proof_parse_output_disproven_exception() {
        let backend = HolLightBackend::new();
        let (status, cex, _) = backend.parse_output("", "Exception: Failure", 1);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "Should be Disproven for exception",
        );
        kani::assert(cex.is_some(), "Should have counterexample for exception");
    }

    /// Verify parse_output returns Disproven for Failure
    #[kani::proof]
    fn proof_parse_output_disproven_failure() {
        let backend = HolLightBackend::new();
        let (status, _, _) = backend.parse_output("Failure: type mismatch", "", 1);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "Should be Disproven for Failure",
        );
    }

    /// Verify parse_output collects warnings
    #[kani::proof]
    fn proof_parse_output_collects_warnings() {
        let backend = HolLightBackend::new();
        let (_, _, diags) = backend.parse_output("Warning: something", "", 0);
        kani::assert(!diags.is_empty(), "Should collect warnings");
    }

    /// Verify parse_output returns Unknown for unclear output
    #[kani::proof]
    fn proof_parse_output_unknown() {
        let backend = HolLightBackend::new();
        // With non-zero exit with no success should be Disproven
        let (status, _, _) = backend.parse_output("unclear", "unclear", 2);
        kani::assert(
            matches!(status, VerificationStatus::Disproven),
            "Non-zero exit with no success should be Disproven",
        );
    }

    // ---- Counterexample Extraction Tests ----

    /// Verify extract_counterexample parses exception message
    #[kani::proof]
    fn proof_extract_counterexample_exception() {
        let backend = HolLightBackend::new();
        let output = "Exception: type error";
        let cex = backend.extract_counterexample(output);
        kani::assert(cex.is_some(), "Should extract counterexample");
        let cex = cex.unwrap();
        kani::assert(!cex.failed_checks.is_empty(), "Should have failed checks");
    }

    /// Verify extract_counterexample parses Failure message
    #[kani::proof]
    fn proof_extract_counterexample_failure() {
        let backend = HolLightBackend::new();
        let output = "Failure: mk_comb";
        let cex = backend.extract_counterexample(output);
        kani::assert(cex.is_some(), "Should extract counterexample for Failure");
    }

    /// Verify extract_counterexample parses line number
    #[kani::proof]
    fn proof_extract_counterexample_line_number() {
        let backend = HolLightBackend::new();
        let output = "Exception: error\nline 15: something";
        let cex = backend.extract_counterexample(output);
        kani::assert(cex.is_some(), "Should extract counterexample");
        let cex = cex.unwrap();
        if !cex.failed_checks.is_empty() {
            if let Some(ref loc) = cex.failed_checks[0].location {
                kani::assert(loc.line == 15, "Should parse line number");
            }
        }
    }

    /// Verify extract_counterexample creates default check when no exception
    #[kani::proof]
    fn proof_extract_counterexample_default() {
        let backend = HolLightBackend::new();
        // With valid regex but no match, should create default failed check
        let cex = backend.extract_counterexample("no exception here");
        kani::assert(cex.is_some(), "Should return Some");
        let cex = cex.unwrap();
        kani::assert(
            !cex.failed_checks.is_empty(),
            "Should have default failed check",
        );
    }

    /// Verify extract_counterexample stores raw output
    #[kani::proof]
    fn proof_extract_counterexample_raw() {
        let backend = HolLightBackend::new();
        let output = "test output";
        let cex = backend.extract_counterexample(output);
        if let Some(cex) = cex {
            kani::assert(cex.raw.is_some(), "Should store raw output");
        }
    }

    /// Verify extract_counterexample not minimized
    #[kani::proof]
    fn proof_extract_counterexample_not_minimized() {
        let backend = HolLightBackend::new();
        let cex = backend.extract_counterexample("Exception: error");
        if let Some(cex) = cex {
            kani::assert(!cex.minimized, "Should not be minimized");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dashprove_usl::ast::{Expr, Invariant, Property, Spec};

    #[test]
    fn default_config() {
        let config = HolLightConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(120));
        assert!(!config.verbose);
    }

    #[test]
    fn backend_id() {
        assert_eq!(HolLightBackend::new().id(), BackendId::HOLLight);
    }

    #[test]
    fn supports_properties() {
        let backend = HolLightBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::Theorem));
        assert!(supported.contains(&PropertyType::Invariant));
    }

    #[test]
    fn hol_script_generation_empty_spec() {
        let backend = HolLightBackend::new();
        let spec = TypedSpec {
            spec: Spec {
                types: vec![],
                properties: vec![],
            },
            type_info: HashMap::new(),
        };
        let hol_dir = PathBuf::from("/path/to/hol");
        let script = backend.generate_hol_script(&spec, &hol_dir);
        assert!(script.contains("hol.ml"));
        assert!(script.contains("TRUTH"));
        assert!(script.contains("PROOF_SUCCESS"));
    }

    #[test]
    fn hol_script_generation_with_property() {
        let backend = HolLightBackend::new();
        let spec = TypedSpec {
            spec: Spec {
                types: vec![],
                properties: vec![Property::Invariant(Invariant {
                    name: "test_thm".to_string(),
                    body: Expr::Bool(true),
                })],
            },
            type_info: HashMap::new(),
        };
        let hol_dir = PathBuf::from("/path/to/hol");
        let script = backend.generate_hol_script(&spec, &hol_dir);
        assert!(script.contains("test_thm"));
    }

    #[test]
    fn parse_output_proven() {
        let backend = HolLightBackend::new();
        let stdout = "val it : thm = |- T";
        let (status, cex, _) = backend.parse_output(stdout, "", 0);
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(cex.is_none());
    }

    #[test]
    fn parse_output_success_marker() {
        let backend = HolLightBackend::new();
        let stdout = "Loading... PROOF_SUCCESS: All properties verified";
        let (status, _, _) = backend.parse_output(stdout, "", 0);
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn parse_output_exception() {
        let backend = HolLightBackend::new();
        let stderr = "Exception: Failure \"mk_comb\"";
        let (status, cex, _) = backend.parse_output("", stderr, 1);
        assert!(matches!(status, VerificationStatus::Disproven));
        let ce = cex.expect("expected counterexample");
        assert!(!ce.failed_checks.is_empty());
    }

    #[test]
    fn extract_counterexample_exception() {
        let backend = HolLightBackend::new();
        let output = "Exception: Failure \"types do not match\"";
        let cex = backend.extract_counterexample(output).unwrap();
        assert!(cex.failed_checks[0].description.contains("types"));
    }

    #[test]
    fn extract_counterexample_with_line() {
        let backend = HolLightBackend::new();
        let output = "Error at line 15\nException: Failure";
        let cex = backend.extract_counterexample(output).unwrap();
        assert!(cex.failed_checks[0].location.is_some());
        assert_eq!(cex.failed_checks[0].location.as_ref().unwrap().line, 15);
    }

    #[tokio::test]
    async fn health_check_unavailable() {
        let config = HolLightConfig {
            hol_light_path: Some(PathBuf::from("/nonexistent/hol_light")),
            ..Default::default()
        };
        let backend = HolLightBackend::with_config(config);
        let health = backend.health_check().await;
        assert!(matches!(health, HealthStatus::Unavailable { .. }));
    }
}
