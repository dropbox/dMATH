//! LEAN 4 backend implementation
//!
//! This backend executes LEAN 4 specifications using the lake build system.
//! LEAN 4 is a theorem prover that can verify mathematical properties.

// Allow match on Result for clearer control flow in path detection
#![allow(clippy::single_match_else)]

use crate::traits::*;
use crate::util::expand_home_dir;
use async_trait::async_trait;
use dashprove_usl::{compile_to_lean, typecheck::TypedSpec};
use regex::Regex;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::process::Command;
use tracing::{debug, info, warn};

// =============================================
// Kani Proofs for LEAN 4 Backend
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify Lean4Config::default timeout is 300 seconds
    #[kani::proof]
    fn proof_lean4_config_default_timeout() {
        let config = Lean4Config::default();
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "Default timeout should be 300 seconds",
        );
    }

    /// Verify Lean4Config::default does not use mathlib
    #[kani::proof]
    fn proof_lean4_config_default_no_mathlib() {
        let config = Lean4Config::default();
        kani::assert(!config.use_mathlib, "Default should not use Mathlib");
    }

    /// Verify Lean4Config::default lake_path is None
    #[kani::proof]
    fn proof_lean4_config_default_lake_path_none() {
        let config = Lean4Config::default();
        kani::assert(
            config.lake_path.is_none(),
            "Default lake_path should be None",
        );
    }

    /// Verify Lean4Config::default lean_path is None
    #[kani::proof]
    fn proof_lean4_config_default_lean_path_none() {
        let config = Lean4Config::default();
        kani::assert(
            config.lean_path.is_none(),
            "Default lean_path should be None",
        );
    }

    /// Verify Lean4Backend::new uses default config
    #[kani::proof]
    fn proof_lean4_backend_new_uses_defaults() {
        let backend = Lean4Backend::new();
        kani::assert(
            backend.config.timeout == Duration::from_secs(300),
            "New backend should use default timeout",
        );
    }

    /// Verify Lean4Backend::with_config preserves config
    #[kani::proof]
    fn proof_lean4_backend_with_config() {
        let config = Lean4Config {
            lake_path: Some(PathBuf::from("/custom/lake")),
            lean_path: None,
            timeout: Duration::from_secs(600),
            use_mathlib: true,
        };
        let backend = Lean4Backend::with_config(config.clone());
        kani::assert(
            backend.config.timeout == Duration::from_secs(600),
            "Backend should use custom timeout",
        );
        kani::assert(
            backend.config.use_mathlib,
            "Backend should use mathlib flag",
        );
    }

    /// Verify generate_lakefile includes project name
    #[kani::proof]
    fn proof_generate_lakefile_contains_project_name() {
        let backend = Lean4Backend::new();
        let lakefile = backend.generate_lakefile("TestProject");
        kani::assert(
            lakefile.contains("TestProject"),
            "Lakefile should contain project name",
        );
    }

    /// Verify generate_lakefile without mathlib has no mathlib reference
    #[kani::proof]
    fn proof_generate_lakefile_no_mathlib() {
        let backend = Lean4Backend::new();
        let lakefile = backend.generate_lakefile("Test");
        kani::assert(
            !lakefile.contains("mathlib"),
            "Default lakefile should not have mathlib",
        );
    }

    /// Verify generate_lakefile with mathlib includes mathlib
    #[kani::proof]
    fn proof_generate_lakefile_with_mathlib() {
        let config = Lean4Config {
            use_mathlib: true,
            ..Default::default()
        };
        let backend = Lean4Backend::with_config(config);
        let lakefile = backend.generate_lakefile("Test");
        kani::assert(
            lakefile.contains("mathlib"),
            "Lakefile should include mathlib",
        );
    }

    /// Verify generate_toolchain returns lean4 version string
    #[kani::proof]
    fn proof_generate_toolchain_format() {
        let backend = Lean4Backend::new();
        let toolchain = backend.generate_toolchain();
        kani::assert(
            toolchain.starts_with("leanprover/lean4:"),
            "Toolchain should start with leanprover/lean4:",
        );
    }

    /// Verify strip_mathlib_imports removes import lines
    #[kani::proof]
    fn proof_strip_mathlib_imports_removes_imports() {
        let backend = Lean4Backend::new();
        let code = "import Mathlib.Data.Set.Basic\ntheorem foo := trivial";
        let stripped = backend.strip_mathlib_imports(code);
        kani::assert(
            !stripped.contains("import Mathlib"),
            "Should remove Mathlib imports",
        );
        kani::assert(stripped.contains("theorem foo"), "Should preserve theorem");
    }

    /// Verify strip_mathlib_imports preserves non-mathlib imports
    #[kani::proof]
    fn proof_strip_mathlib_imports_preserves_other() {
        let backend = Lean4Backend::new();
        let code = "import Std\ntheorem foo := trivial";
        let stripped = backend.strip_mathlib_imports(code);
        kani::assert(
            stripped.contains("import Std"),
            "Should preserve Std import",
        );
    }

    /// Verify estimate_completion returns 100 for no sorry
    #[kani::proof]
    fn proof_estimate_completion_no_sorry() {
        let backend = Lean4Backend::new();
        let output = "theorem a := trivial\ntheorem b := trivial";
        let completion = backend.estimate_completion(output);
        kani::assert(completion == 100.0, "No sorry should mean 100% completion");
    }

    /// Verify estimate_completion handles sorry
    #[kani::proof]
    fn proof_estimate_completion_with_sorry() {
        let backend = Lean4Backend::new();
        let output = "theorem a := sorry\ntheorem b := trivial";
        let completion = backend.estimate_completion(output);
        // One sorry out of two theorems = ~50%
        kani::assert(
            completion >= 0.0 && completion <= 100.0,
            "Completion should be in valid range",
        );
    }

    /// Verify id() returns Lean4
    #[kani::proof]
    fn proof_backend_id_is_lean4() {
        let backend = Lean4Backend::new();
        kani::assert(backend.id() == BackendId::Lean4, "ID should be Lean4");
    }

    /// Verify supports() includes Theorem
    #[kani::proof]
    fn proof_supports_includes_theorem() {
        let backend = Lean4Backend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::Theorem),
            "Should support Theorem",
        );
    }

    /// Verify supports() includes Invariant
    #[kani::proof]
    fn proof_supports_includes_invariant() {
        let backend = Lean4Backend::new();
        let supported = backend.supports();
        kani::assert(
            supported.contains(&PropertyType::Invariant),
            "Should support Invariant",
        );
    }

    /// Verify LakeDetection::NotFound preserves reason
    #[kani::proof]
    fn proof_lake_detection_not_found() {
        let reason = "lake not found".to_string();
        let detection = LakeDetection::NotFound(reason.clone());
        if let LakeDetection::NotFound(r) = detection {
            kani::assert(r == reason, "NotFound should preserve reason");
        } else {
            kani::assert(false, "Should be NotFound variant");
        }
    }

    /// Verify LakeDetection::Available stores paths
    #[kani::proof]
    fn proof_lake_detection_available() {
        let lake_path = PathBuf::from("/bin/lake");
        let lean_path = PathBuf::from("/bin/lean");
        let detection = LakeDetection::Available {
            lake_path: lake_path.clone(),
            lean_path: lean_path.clone(),
        };
        if let LakeDetection::Available {
            lake_path: l,
            lean_path: n,
        } = detection
        {
            kani::assert(l == lake_path, "Should store lake_path");
            kani::assert(n == lean_path, "Should store lean_path");
        } else {
            kani::assert(false, "Should be Available variant");
        }
    }
}

/// Configuration for LEAN 4 backend
#[derive(Debug, Clone)]
pub struct Lean4Config {
    /// Path to lake executable (if not in PATH)
    pub lake_path: Option<PathBuf>,
    /// Path to lean executable (if not in PATH)
    pub lean_path: Option<PathBuf>,
    /// Timeout for verification
    pub timeout: Duration,
    /// Whether to use Mathlib (requires internet/cache on first use)
    pub use_mathlib: bool,
}

impl Default for Lean4Config {
    fn default() -> Self {
        Self {
            lake_path: None,
            lean_path: None,
            timeout: Duration::from_secs(300), // 5 minutes
            use_mathlib: false,                // Mathlib requires longer setup, disabled by default
        }
    }
}

/// LEAN 4 verification backend using lake build system
pub struct Lean4Backend {
    config: Lean4Config,
}

#[derive(Debug, Clone)]
enum LakeDetection {
    /// Lake available as a standalone command
    Available {
        lake_path: PathBuf,
        lean_path: PathBuf,
    },
    /// Lake/LEAN not found
    NotFound(String),
}

impl Lean4Backend {
    /// Create a new LEAN 4 backend with default configuration
    pub fn new() -> Self {
        Self {
            config: Lean4Config::default(),
        }
    }

    /// Create a new LEAN 4 backend with custom configuration
    pub fn with_config(config: Lean4Config) -> Self {
        Self { config }
    }

    /// Detect lake/lean installation
    async fn detect_lake(&self) -> LakeDetection {
        // 1. Check if lake_path is explicitly configured
        let lake_path = if let Some(ref path) = self.config.lake_path {
            if path.exists() {
                path.clone()
            } else {
                return LakeDetection::NotFound(format!(
                    "Configured lake path does not exist: {:?}",
                    path
                ));
            }
        } else {
            // Check for lake in PATH
            match which::which("lake") {
                Ok(path) => path,
                Err(_) => {
                    // Check common installation locations
                    let common_paths = [
                        expand_home_dir("~/.elan/bin/lake"),
                        Some(PathBuf::from("/usr/local/bin/lake")),
                        Some(PathBuf::from("/opt/homebrew/bin/lake")),
                    ];

                    let mut found = None;
                    for path in common_paths.into_iter().flatten() {
                        if path.exists() {
                            found = Some(path);
                            break;
                        }
                    }

                    match found {
                        Some(path) => path,
                        None => {
                            return LakeDetection::NotFound(
                                "lake not found. Install LEAN 4 via elan: https://leanprover-community.github.io/get_started.html".to_string(),
                            );
                        }
                    }
                }
            }
        };

        // 2. Find lean executable
        let lean_path = if let Some(ref path) = self.config.lean_path {
            if path.exists() {
                path.clone()
            } else {
                return LakeDetection::NotFound(format!(
                    "Configured lean path does not exist: {:?}",
                    path
                ));
            }
        } else {
            // Check for lean in PATH
            match which::which("lean") {
                Ok(path) => path,
                Err(_) => {
                    // Check common installation locations
                    let common_paths = [
                        expand_home_dir("~/.elan/bin/lean"),
                        Some(PathBuf::from("/usr/local/bin/lean")),
                        Some(PathBuf::from("/opt/homebrew/bin/lean")),
                    ];

                    let mut found = None;
                    for path in common_paths.into_iter().flatten() {
                        if path.exists() {
                            found = Some(path);
                            break;
                        }
                    }

                    match found {
                        Some(path) => path,
                        None => {
                            return LakeDetection::NotFound(
                                "lean not found. Install LEAN 4 via elan: https://leanprover-community.github.io/get_started.html".to_string(),
                            );
                        }
                    }
                }
            }
        };

        // 3. Verify lake works
        let version_check = Command::new(&lake_path)
            .arg("--version")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await;

        match version_check {
            Ok(output) if output.status.success() => {
                let version = String::from_utf8_lossy(&output.stdout);
                debug!("Found lake version: {}", version.trim());
                LakeDetection::Available {
                    lake_path,
                    lean_path,
                }
            }
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                LakeDetection::NotFound(format!("lake --version failed: {}", stderr))
            }
            Err(e) => LakeDetection::NotFound(format!("Failed to execute lake: {}", e)),
        }
    }

    /// Generate lakefile.toml for the project
    fn generate_lakefile(&self, project_name: &str) -> String {
        let mut lakefile = String::new();
        lakefile.push_str(&format!(
            r#"name = "{}"
version = "0.1.0"
defaultTargets = ["{}"]

[[lean_lib]]
name = "{}"
"#,
            project_name, project_name, project_name
        ));

        // Add Mathlib dependency if configured
        if self.config.use_mathlib {
            lakefile.push_str(
                r#"
[[require]]
name = "mathlib"
scope = "leanprover-community"
rev = "master"
"#,
            );
        }

        lakefile
    }

    /// Generate lean-toolchain file
    fn generate_toolchain(&self) -> String {
        // Use a stable LEAN 4 version
        "leanprover/lean4:v4.13.0".to_string()
    }

    /// Write LEAN 4 project files to temp directory
    async fn write_project(&self, spec: &TypedSpec, dir: &Path) -> Result<PathBuf, BackendError> {
        let compiled = compile_to_lean(spec);
        let project_name = compiled.module_name.as_deref().unwrap_or("USLSpec");

        // Write lakefile.toml
        let lakefile_content = self.generate_lakefile(project_name);
        let lakefile_path = dir.join("lakefile.toml");
        tokio::fs::write(&lakefile_path, &lakefile_content)
            .await
            .map_err(|e| {
                BackendError::CompilationFailed(format!("Failed to write lakefile.toml: {}", e))
            })?;

        // Write lean-toolchain
        let toolchain_content = self.generate_toolchain();
        let toolchain_path = dir.join("lean-toolchain");
        tokio::fs::write(&toolchain_path, &toolchain_content)
            .await
            .map_err(|e| {
                BackendError::CompilationFailed(format!("Failed to write lean-toolchain: {}", e))
            })?;

        // Prepare LEAN source code - remove Mathlib imports if not using Mathlib
        let lean_code = if self.config.use_mathlib {
            compiled.code.clone()
        } else {
            // Remove Mathlib imports and use basic types
            self.strip_mathlib_imports(&compiled.code)
        };

        // Write main LEAN file
        let lean_path = dir.join(format!("{}.lean", project_name));
        tokio::fs::write(&lean_path, &lean_code)
            .await
            .map_err(|e| {
                BackendError::CompilationFailed(format!("Failed to write LEAN file: {}", e))
            })?;

        debug!("Written lakefile.toml to {:?}", lakefile_path);
        debug!("Written lean-toolchain to {:?}", toolchain_path);
        debug!("Written LEAN spec to {:?}", lean_path);

        Ok(lean_path)
    }

    /// Strip Mathlib imports from generated code for standalone verification
    fn strip_mathlib_imports(&self, code: &str) -> String {
        code.lines()
            .filter(|line| !line.starts_with("import Mathlib"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Execute lake build and capture output
    async fn run_lake_build(
        &self,
        detection: &LakeDetection,
        project_dir: &Path,
    ) -> Result<LakeOutput, BackendError> {
        let start = Instant::now();

        let lake_path = match detection {
            LakeDetection::Available { lake_path, .. } => lake_path,
            LakeDetection::NotFound(reason) => {
                return Err(BackendError::Unavailable(reason.clone()));
            }
        };

        let mut cmd = Command::new(lake_path);
        cmd.arg("build");
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        cmd.current_dir(project_dir);

        let output = tokio::time::timeout(self.config.timeout, cmd.output())
            .await
            .map_err(|_| BackendError::Timeout(self.config.timeout))?
            .map_err(|e| {
                BackendError::VerificationFailed(format!("Failed to execute lake build: {}", e))
            })?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let duration = start.elapsed();

        debug!("lake build stdout:\n{}", stdout);
        if !stderr.is_empty() {
            debug!("lake build stderr:\n{}", stderr);
        }

        Ok(LakeOutput {
            stdout,
            stderr,
            exit_code: output.status.code(),
            duration,
        })
    }

    /// Parse lake build output into verification result
    fn parse_output(&self, output: &LakeOutput) -> BackendResult {
        let combined = format!("{}\n{}", output.stdout, output.stderr);

        // Check for successful build (no errors)
        if output.exit_code == Some(0) && !combined.contains("error:") {
            // Check if there are any 'sorry' placeholders left
            let has_sorry = combined.contains("declaration uses 'sorry'")
                || combined.contains("contains sorry");

            if has_sorry {
                return BackendResult {
                    backend: BackendId::Lean4,
                    status: VerificationStatus::Partial {
                        verified_percentage: self.estimate_completion(&combined),
                    },
                    proof: Some(
                        "Build succeeded but proofs contain 'sorry' placeholders".to_string(),
                    ),
                    counterexample: None,
                    diagnostics: self.extract_diagnostics(&combined),
                    time_taken: output.duration,
                };
            }

            return BackendResult {
                backend: BackendId::Lean4,
                status: VerificationStatus::Proven,
                proof: Some("All theorems verified successfully".to_string()),
                counterexample: None,
                diagnostics: self.extract_diagnostics(&combined),
                time_taken: output.duration,
            };
        }

        // Check for type errors (disproven)
        if combined.contains("type mismatch") || combined.contains("failed to synthesize") {
            let error_msg = self.extract_error(&combined);
            return BackendResult {
                backend: BackendId::Lean4,
                status: VerificationStatus::Disproven,
                proof: None,
                counterexample: Some(StructuredCounterexample::from_raw(error_msg)),
                diagnostics: self.extract_diagnostics(&combined),
                time_taken: output.duration,
            };
        }

        // Check for elaboration errors (could be disproven or error)
        if combined.contains("error:") {
            let error_msg = self.extract_error(&combined);

            // Certain errors indicate the property is falsifiable
            // Real LEAN 4 output shows "tactic 'rfl' failed" or similar patterns
            if combined.contains("failed to prove")
                || combined.contains("failed to show")
                || combined.contains("unsolved goals")
                || combined.contains("tactic") && combined.contains("failed")
            {
                return BackendResult {
                    backend: BackendId::Lean4,
                    status: VerificationStatus::Disproven,
                    proof: None,
                    counterexample: Some(StructuredCounterexample::from_raw(error_msg)),
                    diagnostics: self.extract_diagnostics(&combined),
                    time_taken: output.duration,
                };
            }

            // Other errors are compilation/setup issues
            return BackendResult {
                backend: BackendId::Lean4,
                status: VerificationStatus::Unknown {
                    reason: format!("LEAN 4 error: {}", error_msg),
                },
                proof: None,
                counterexample: None,
                diagnostics: self.extract_diagnostics(&combined),
                time_taken: output.duration,
            };
        }

        // Check for warnings about sorry
        if combined.contains("warning:") && combined.contains("sorry") {
            return BackendResult {
                backend: BackendId::Lean4,
                status: VerificationStatus::Partial {
                    verified_percentage: self.estimate_completion(&combined),
                },
                proof: Some("Build succeeded with warnings about incomplete proofs".to_string()),
                counterexample: None,
                diagnostics: self.extract_diagnostics(&combined),
                time_taken: output.duration,
            };
        }

        // Unknown result
        BackendResult {
            backend: BackendId::Lean4,
            status: VerificationStatus::Unknown {
                reason: "Could not determine verification result".to_string(),
            },
            proof: None,
            counterexample: None,
            diagnostics: vec![combined],
            time_taken: output.duration,
        }
    }

    fn extract_error(&self, output: &str) -> String {
        let mut errors = Vec::new();

        for line in output.lines() {
            if line.contains("error:") || line.contains("Error:") {
                errors.push(line.to_string());
            }
        }

        if errors.is_empty() {
            // Try to extract any line after "error"
            let mut in_error = false;
            for line in output.lines() {
                if line.contains("error") {
                    in_error = true;
                }
                if in_error {
                    errors.push(line.to_string());
                    if errors.len() >= 5 {
                        break;
                    }
                }
            }
        }

        if errors.is_empty() {
            "Unknown error".to_string()
        } else {
            errors.join("\n")
        }
    }

    fn extract_diagnostics(&self, output: &str) -> Vec<String> {
        let mut diagnostics = Vec::new();

        // Extract build time
        let time_re = Regex::new(r"built in (\d+(?:\.\d+)?)\s*(?:s|ms)").ok();
        if let Some(re) = time_re {
            if let Some(caps) = re.captures(output) {
                diagnostics.push(format!(
                    "Build time: {}",
                    caps.get(1).map(|m| m.as_str()).unwrap_or("?")
                ));
            }
        }

        // Count warnings
        let warning_count = output.matches("warning:").count();
        if warning_count > 0 {
            diagnostics.push(format!("{} warning(s)", warning_count));
        }

        // Count errors
        let error_count = output.matches("error:").count();
        if error_count > 0 {
            diagnostics.push(format!("{} error(s)", error_count));
        }

        diagnostics
    }

    fn estimate_completion(&self, output: &str) -> f64 {
        // Count sorry occurrences vs total theorems
        let sorry_count = output.matches("sorry").count();
        let theorem_count = output.matches("theorem").count().max(1);

        // If we have sorry placeholders, estimate based on ratio
        if sorry_count > 0 {
            let proven = (theorem_count as f64 - sorry_count as f64).max(0.0);
            (proven / theorem_count as f64) * 100.0
        } else {
            100.0
        }
    }
}

impl Default for Lean4Backend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VerificationBackend for Lean4Backend {
    fn id(&self) -> BackendId {
        BackendId::Lean4
    }

    fn supports(&self) -> Vec<PropertyType> {
        vec![
            PropertyType::Theorem,
            PropertyType::Invariant,
            PropertyType::Refinement,
        ]
    }

    async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError> {
        let detection = self.detect_lake().await;

        if let LakeDetection::NotFound(reason) = &detection {
            return Err(BackendError::Unavailable(reason.clone()));
        }

        // Create temp directory for project
        let temp_dir = TempDir::new().map_err(|e| {
            BackendError::CompilationFailed(format!("Failed to create temp dir: {}", e))
        })?;

        // Write project files
        let _lean_path = self.write_project(spec, temp_dir.path()).await?;

        // Run lake build
        let output = self.run_lake_build(&detection, temp_dir.path()).await?;

        // Parse results
        let result = self.parse_output(&output);

        Ok(result)
    }

    async fn health_check(&self) -> HealthStatus {
        let detection = self.detect_lake().await;
        match detection {
            LakeDetection::Available {
                lake_path,
                lean_path,
            } => {
                info!(
                    "LEAN 4 available: lake={:?}, lean={:?}",
                    lake_path, lean_path
                );
                HealthStatus::Healthy
            }
            LakeDetection::NotFound(reason) => {
                warn!("LEAN 4 not available: {}", reason);
                HealthStatus::Unavailable { reason }
            }
        }
    }
}

/// lake build execution output
struct LakeOutput {
    stdout: String,
    stderr: String,
    exit_code: Option<i32>,
    duration: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;
    use dashprove_usl::{parse, typecheck};

    fn make_typed_spec(input: &str) -> TypedSpec {
        let spec = parse(input).expect("parse failed");
        typecheck(spec).expect("typecheck failed")
    }

    #[test]
    fn test_lakefile_generation() {
        let backend = Lean4Backend::new();
        let lakefile = backend.generate_lakefile("TestProject");

        assert!(lakefile.contains(r#"name = "TestProject""#));
        assert!(lakefile.contains(r#"[[lean_lib]]"#));
        // Default config doesn't use Mathlib
        assert!(!lakefile.contains("mathlib"));
    }

    #[test]
    fn test_lakefile_with_mathlib() {
        let config = Lean4Config {
            use_mathlib: true,
            ..Default::default()
        };
        let backend = Lean4Backend::with_config(config);
        let lakefile = backend.generate_lakefile("TestProject");

        assert!(lakefile.contains("mathlib"));
        assert!(lakefile.contains("leanprover-community"));
    }

    #[test]
    fn test_toolchain_generation() {
        let backend = Lean4Backend::new();
        let toolchain = backend.generate_toolchain();

        assert!(toolchain.starts_with("leanprover/lean4:"));
    }

    #[test]
    fn test_strip_mathlib_imports() {
        let backend = Lean4Backend::new();
        let code = r#"import Mathlib.Data.Set.Basic
import Mathlib.Data.List.Basic

namespace Test
theorem foo : True := by trivial
end Test"#;

        let stripped = backend.strip_mathlib_imports(code);
        assert!(!stripped.contains("import Mathlib"));
        assert!(stripped.contains("namespace Test"));
        assert!(stripped.contains("theorem foo"));
    }

    #[test]
    fn test_parse_success_output() {
        let backend = Lean4Backend::new();
        let output = LakeOutput {
            stdout: "Build completed successfully\nAll theorems verified".to_string(),
            stderr: String::new(),
            exit_code: Some(0),
            duration: Duration::from_secs(5),
        };

        let result = backend.parse_output(&output);
        assert!(matches!(result.status, VerificationStatus::Proven));
    }

    #[test]
    fn test_parse_sorry_warning() {
        let backend = Lean4Backend::new();
        let output = LakeOutput {
            stdout: "warning: declaration uses 'sorry'\ntheorem test := sorry\n".to_string(),
            stderr: String::new(),
            exit_code: Some(0),
            duration: Duration::from_secs(3),
        };

        let result = backend.parse_output(&output);
        assert!(matches!(result.status, VerificationStatus::Partial { .. }));
    }

    #[test]
    fn test_parse_type_error() {
        let backend = Lean4Backend::new();
        let output = LakeOutput {
            stdout: String::new(),
            stderr: "error: type mismatch\n  expected Nat\n  got Bool".to_string(),
            exit_code: Some(1),
            duration: Duration::from_secs(2),
        };

        let result = backend.parse_output(&output);
        assert!(matches!(result.status, VerificationStatus::Disproven));
        assert!(result.counterexample.is_some());
    }

    #[test]
    fn test_parse_elaboration_error() {
        let backend = Lean4Backend::new();
        let output = LakeOutput {
            stdout: String::new(),
            stderr: "error: failed to synthesize\n  instance\n    Add String".to_string(),
            exit_code: Some(1),
            duration: Duration::from_secs(1),
        };

        let result = backend.parse_output(&output);
        assert!(matches!(result.status, VerificationStatus::Disproven));
    }

    #[test]
    fn test_parse_proof_failure() {
        let backend = Lean4Backend::new();
        let output = LakeOutput {
            stdout: String::new(),
            stderr: "error: unsolved goals\n⊢ False".to_string(),
            exit_code: Some(1),
            duration: Duration::from_secs(1),
        };

        let result = backend.parse_output(&output);
        assert!(matches!(result.status, VerificationStatus::Disproven));
    }

    #[test]
    fn test_extract_diagnostics() {
        let backend = Lean4Backend::new();
        let output = "warning: unused variable\nwarning: sorry\nerror: type mismatch";
        let diagnostics = backend.extract_diagnostics(output);

        assert!(diagnostics.iter().any(|d| d.contains("2 warning")));
        assert!(diagnostics.iter().any(|d| d.contains("1 error")));
    }

    #[test]
    fn test_estimate_completion() {
        let backend = Lean4Backend::new();

        // No sorry - 100%
        let output = "theorem a := trivial\ntheorem b := trivial";
        assert_eq!(backend.estimate_completion(output), 100.0);

        // One sorry out of two theorems - 50%
        let output = "theorem a := sorry\ntheorem b := trivial";
        let completion = backend.estimate_completion(output);
        assert!(completion > 40.0 && completion < 60.0);
    }

    #[tokio::test]
    async fn test_health_check_reports_status() {
        let backend = Lean4Backend::new();
        let status = backend.health_check().await;

        // Should report some status (Healthy or Unavailable depending on system)
        match status {
            HealthStatus::Healthy => println!("LEAN 4 is available"),
            HealthStatus::Unavailable { reason } => println!("LEAN 4 not available: {}", reason),
            HealthStatus::Degraded { reason } => println!("LEAN 4 degraded: {}", reason),
        }
    }

    #[tokio::test]
    async fn test_verify_returns_result_or_unavailable() {
        let backend = Lean4Backend::new();
        let input = r#"
            theorem test {
                forall x: Bool . x or not x
            }
        "#;
        let spec = make_typed_spec(input);

        let result = backend.verify(&spec).await;
        match result {
            Ok(r) => {
                println!("Verification result: {:?}", r.status);
            }
            Err(BackendError::Unavailable(reason)) => {
                println!("Backend unavailable: {}", reason);
            }
            Err(e) => {
                panic!("Unexpected error: {:?}", e);
            }
        }
    }

    #[test]
    fn test_typed_spec_generates_valid_lean() {
        let input = r#"
            theorem excluded_middle {
                forall p: Bool . p or not p
            }
        "#;
        let spec = make_typed_spec(input);
        let compiled = compile_to_lean(&spec);

        // Verify the generated LEAN code structure
        assert!(compiled.code.contains("namespace USLSpec"));
        assert!(compiled.code.contains("theorem excluded_middle"));
        assert!(compiled.code.contains("end USLSpec"));
    }
}
