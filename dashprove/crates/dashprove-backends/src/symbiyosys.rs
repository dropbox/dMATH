//! SymbiYosys formal verification frontend for Yosys
//!
//! SymbiYosys is a front-end driver for Yosys-based formal verification.
//!
//! See: <https://symbiyosys.readthedocs.io/>

use crate::counterexample::{FailedCheck, SourceLocation, StructuredCounterexample};
use crate::traits::{
    BackendError, BackendId, BackendResult, HealthStatus, PropertyType, VerificationBackend,
    VerificationStatus,
};
use crate::yosys::YosysBackend;
use async_trait::async_trait;
use dashprove_usl::typecheck::TypedSpec;
use regex::Regex;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::process::Command;
use tracing::debug;

#[derive(Debug, Clone)]
pub struct SymbiYosysConfig {
    pub sby_path: Option<PathBuf>,
    pub timeout: Duration,
    /// Verification mode (bmc, prove, cover)
    pub mode: String,
    /// Depth bound for BMC
    pub depth: Option<u32>,
}

impl Default for SymbiYosysConfig {
    fn default() -> Self {
        Self {
            sby_path: None,
            timeout: Duration::from_secs(180),
            mode: "bmc".to_string(),
            depth: Some(50),
        }
    }
}

pub struct SymbiYosysBackend {
    config: SymbiYosysConfig,
}
impl Default for SymbiYosysBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl SymbiYosysBackend {
    pub fn new() -> Self {
        Self {
            config: SymbiYosysConfig::default(),
        }
    }
    pub fn with_config(config: SymbiYosysConfig) -> Self {
        Self { config }
    }

    async fn detect(&self) -> Result<PathBuf, String> {
        self.config
            .sby_path
            .clone()
            .or_else(|| which::which("sby").ok())
            .ok_or("SymbiYosys not found".to_string())
    }

    fn write_design_files(
        &self,
        spec: &TypedSpec,
        dir: &Path,
    ) -> Result<(PathBuf, PathBuf), BackendError> {
        let yosys_backend = YosysBackend::new();
        let design = yosys_backend.generate_verilog(spec);
        let design_path = dir.join("design.v");
        std::fs::write(&design_path, design).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write design.v: {e}"))
        })?;

        let sby_path = dir.join("job.sby");
        std::fs::write(&sby_path, self.render_sby_file(&design_path)).map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to write job.sby: {e}"))
        })?;

        Ok((design_path, sby_path))
    }

    fn render_sby_file(&self, design_path: &Path) -> String {
        let mut sby = String::new();
        sby.push_str("[options]\n");
        sby.push_str(&format!("mode {}\n", self.config.mode));
        if let Some(depth) = self.config.depth {
            sby.push_str(&format!("depth {}\n", depth));
        }
        sby.push_str("wait on\n\n");

        sby.push_str("[engines]\n");
        sby.push_str("smtbmc\n\n");

        sby.push_str("[script]\n");
        sby.push_str(&format!("read_verilog {}\n", design_path.display()));
        sby.push_str("prep -top top\n");
        sby.push_str("chformal -assume -early\n\n");

        sby.push_str("[files]\n");
        sby.push_str(&format!("{}\n", design_path.display()));
        sby
    }

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
            .filter(|l| l.contains("SBY") || l.contains("engine_"))
            .map(|s| s.to_string())
            .collect();

        let status = if combined.contains("Status: passed") {
            VerificationStatus::Proven
        } else if combined.contains("Status: failed") {
            VerificationStatus::Disproven
        } else {
            VerificationStatus::Unknown {
                reason: "SymbiYosys did not report status".to_string(),
            }
        };

        let counterexample = if matches!(status, VerificationStatus::Disproven) {
            self.extract_counterexample(stdout)
        } else {
            None
        };

        if diagnostics.is_empty() && !stderr.trim().is_empty() {
            diagnostics.push(stderr.to_string());
        }

        (status, counterexample, diagnostics)
    }

    fn extract_counterexample(&self, stdout: &str) -> Option<StructuredCounterexample> {
        let mut witness = HashMap::new();
        let re = Regex::new(r"(?P<name>[A-Za-z0-9_\.]+)\s*=\s*(?P<value>[\w'xbhd\.]+)")
            .expect("regex compiles");

        for cap in re.captures_iter(stdout) {
            let name = cap.name("name").unwrap().as_str().to_string();
            let raw_val = cap.name("value").unwrap().as_str();
            let value = YosysBackend::parse_value(raw_val);
            witness.insert(name, value);
        }

        if witness.is_empty() {
            return None;
        }

        let failed_checks = vec![FailedCheck {
            check_id: "symbiyosys_assert".to_string(),
            description: "SymbiYosys assertion failure".to_string(),
            location: Some(SourceLocation {
                file: "symbiyosys".to_string(),
                line: 0,
                column: None,
            }),
            function: None,
        }];

        Some(StructuredCounterexample {
            witness,
            failed_checks,
            playback_test: None,
            trace: Vec::new(),
            raw: Some(stdout.to_string()),
            minimized: false,
        })
    }
}

#[async_trait]
impl VerificationBackend for SymbiYosysBackend {
    fn id(&self) -> BackendId {
        BackendId::SymbiYosys
    }
    fn supports(&self) -> Vec<PropertyType> {
        vec![PropertyType::Invariant, PropertyType::Temporal]
    }
    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let start = Instant::now();
        let sby = self.detect().await.map_err(BackendError::Unavailable)?;
        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to create temp dir: {e}"))
        })?;

        let (_, sby_file) = self.write_design_files(spec, temp_dir.path())?;

        let mut cmd = Command::new(&sby);
        cmd.arg("-f")
            .arg(&sby_file)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let output = tokio::time::timeout(self.config.timeout, cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| BackendError::VerificationFailed(format!("SymbiYosys failed: {e}")))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        debug!("SymbiYosys stdout: {}", stdout);
        if !stderr.trim().is_empty() {
            debug!("SymbiYosys stderr: {}", stderr);
        }

        let (status, counterexample, diagnostics) = self.parse_output(&stdout, &stderr);
        let proof = if matches!(status, VerificationStatus::Proven) {
            Some("SymbiYosys proved properties".to_string())
        } else {
            None
        };

        Ok(BackendResult {
            backend: BackendId::SymbiYosys,
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

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ===== SymbiYosysConfig defaults =====

    #[kani::proof]
    fn verify_config_defaults_timeout() {
        let config = SymbiYosysConfig::default();
        assert!(config.timeout == Duration::from_secs(180));
    }

    #[kani::proof]
    fn verify_config_defaults_mode() {
        let config = SymbiYosysConfig::default();
        assert!(config.mode == "bmc");
    }

    #[kani::proof]
    fn verify_config_defaults_depth() {
        let config = SymbiYosysConfig::default();
        assert!(config.depth == Some(50));
    }

    #[kani::proof]
    fn verify_config_defaults_path() {
        let config = SymbiYosysConfig::default();
        assert!(config.sby_path.is_none());
    }

    // ===== Backend construction =====

    #[kani::proof]
    fn verify_backend_new_uses_defaults() {
        let backend = SymbiYosysBackend::new();
        assert!(backend.config.timeout == Duration::from_secs(180));
        assert!(backend.config.mode == "bmc");
    }

    #[kani::proof]
    fn verify_backend_default_matches_new() {
        let b1 = SymbiYosysBackend::new();
        let b2 = SymbiYosysBackend::default();
        assert!(b1.config.timeout == b2.config.timeout);
        assert!(b1.config.mode == b2.config.mode);
    }

    #[kani::proof]
    fn verify_backend_with_config_preserves_values() {
        let config = SymbiYosysConfig {
            sby_path: Some(PathBuf::from("/usr/bin/sby")),
            timeout: Duration::from_secs(60),
            mode: "prove".to_string(),
            depth: Some(100),
        };
        let backend = SymbiYosysBackend::with_config(config);
        assert!(backend.config.timeout == Duration::from_secs(60));
        assert!(backend.config.mode == "prove");
        assert!(backend.config.depth == Some(100));
    }

    // ===== ID and supports =====

    #[kani::proof]
    fn verify_backend_id() {
        let backend = SymbiYosysBackend::new();
        assert!(matches!(backend.id(), BackendId::SymbiYosys));
    }

    #[kani::proof]
    fn verify_supports_invariant_temporal() {
        let backend = SymbiYosysBackend::new();
        let supported = backend.supports();
        assert!(supported.contains(&PropertyType::Invariant));
        assert!(supported.contains(&PropertyType::Temporal));
        assert!(supported.len() == 2);
    }

    // ===== SBY file rendering =====

    #[kani::proof]
    fn verify_render_sby_contains_options() {
        let backend = SymbiYosysBackend::new();
        let rendered = backend.render_sby_file(Path::new("design.v"));
        assert!(rendered.contains("[options]"));
    }

    #[kani::proof]
    fn verify_render_sby_contains_mode() {
        let backend = SymbiYosysBackend::new();
        let rendered = backend.render_sby_file(Path::new("design.v"));
        assert!(rendered.contains("mode bmc"));
    }

    #[kani::proof]
    fn verify_render_sby_contains_depth() {
        let backend = SymbiYosysBackend::new();
        let rendered = backend.render_sby_file(Path::new("design.v"));
        assert!(rendered.contains("depth 50"));
    }

    #[kani::proof]
    fn verify_render_sby_contains_engines() {
        let backend = SymbiYosysBackend::new();
        let rendered = backend.render_sby_file(Path::new("design.v"));
        assert!(rendered.contains("[engines]"));
        assert!(rendered.contains("smtbmc"));
    }

    #[kani::proof]
    fn verify_render_sby_contains_script() {
        let backend = SymbiYosysBackend::new();
        let rendered = backend.render_sby_file(Path::new("design.v"));
        assert!(rendered.contains("[script]"));
        assert!(rendered.contains("read_verilog"));
    }

    // ===== Output parsing =====

    #[kani::proof]
    fn verify_parse_output_passed() {
        let backend = SymbiYosysBackend::new();
        let (status, _, _) = backend.parse_output("Status: passed", "");
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[kani::proof]
    fn verify_parse_output_failed() {
        let backend = SymbiYosysBackend::new();
        let (status, _, _) = backend.parse_output("Status: failed\nstate=1'b0", "");
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[kani::proof]
    fn verify_parse_output_unknown() {
        let backend = SymbiYosysBackend::new();
        let (status, _, _) = backend.parse_output("timeout", "");
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }

    // ===== Counterexample extraction =====

    #[kani::proof]
    fn verify_extract_counterexample_none_when_no_values() {
        let backend = SymbiYosysBackend::new();
        let result = backend.extract_counterexample("Status: failed\n");
        assert!(result.is_none());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::counterexample::CounterexampleValue;
    use dashprove_usl::ast::{Expr, Invariant, Property, Spec};
    use std::collections::HashMap;

    #[test]
    fn backend_id() {
        assert_eq!(SymbiYosysBackend::new().id(), BackendId::SymbiYosys);
    }

    #[test]
    fn render_sby_contains_sections() {
        let backend = SymbiYosysBackend::new();
        let rendered = backend.render_sby_file(Path::new("design.v"));
        assert!(rendered.contains("[options]"));
        assert!(rendered.contains("read_verilog design.v"));
    }

    #[test]
    fn parse_output_counterexample() {
        let backend = SymbiYosysBackend::new();
        let stdout = "
SBY 10:00:00 [job] Status: failed
engine_0: Model:
state=1'b0
";
        let (status, cex, _) = backend.parse_output(stdout, "");
        assert!(matches!(status, VerificationStatus::Disproven));
        let ce = cex.expect("counterexample expected");
        assert_eq!(ce.witness["state"], CounterexampleValue::Bool(false));
    }

    #[test]
    fn parse_output_verified() {
        let backend = SymbiYosysBackend::new();
        let stdout = "Status: passed\n";
        let (status, cex, _) = backend.parse_output(stdout, "");
        assert!(matches!(status, VerificationStatus::Proven));
        assert!(cex.is_none());
    }

    #[test]
    fn write_design_files_uses_yosys_renderer() {
        let backend = SymbiYosysBackend::new();
        let spec = TypedSpec {
            spec: Spec {
                types: vec![],
                properties: vec![Property::Invariant(Invariant {
                    name: "p".into(),
                    body: Expr::Bool(true),
                })],
            },
            type_info: HashMap::new(),
        };
        let temp = TempDir::new().unwrap();
        let (design, sby) = backend.write_design_files(&spec, temp.path()).unwrap();
        let contents = std::fs::read_to_string(design).unwrap();
        assert!(contents.contains("assert"));
        let sby_content = std::fs::read_to_string(sby).unwrap();
        assert!(sby_content.contains("mode bmc"));
    }
}
