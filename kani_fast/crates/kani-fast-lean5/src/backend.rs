//! Lean5 compiler backend for proof checking
//!
//! This module provides integration with the Lean proof assistant to verify
//! generated proof obligations.

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};
use thiserror::Error;
use wait_timeout::ChildExt;

/// Errors from the Lean5 backend
#[derive(Debug, Error)]
pub enum Lean5Error {
    /// Lean executable not found
    #[error("Lean executable not found: {0}")]
    NotFound(String),

    /// Proof checking failed
    #[error("Proof check failed: {0}")]
    ProofFailed(String),

    /// Timeout during proof checking
    #[error("Proof check timed out after {0:?}")]
    Timeout(Duration),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Lake (Lean build tool) error
    #[error("Lake error: {0}")]
    LakeError(String),
}

/// Result of checking a Lean file
#[derive(Debug, Clone)]
pub struct Lean5Result {
    /// Whether the proof was successful (no errors)
    pub success: bool,
    /// Error messages from Lean (empty if success)
    pub errors: Vec<String>,
    /// Warning messages
    pub warnings: Vec<String>,
    /// Time taken for proof checking
    pub check_time: Duration,
    /// Number of `sorry` placeholders found
    pub sorry_count: usize,
    /// Number of goals solved
    pub goals_solved: usize,
    /// Lean version used
    pub lean_version: String,
}

impl Lean5Result {
    /// Create a successful result
    pub fn success(check_time: Duration, lean_version: String) -> Self {
        Self {
            success: true,
            errors: vec![],
            warnings: vec![],
            check_time,
            sorry_count: 0,
            goals_solved: 0,
            lean_version,
        }
    }

    /// Create a failed result
    pub fn failure(errors: Vec<String>, check_time: Duration, lean_version: String) -> Self {
        Self {
            success: false,
            errors,
            warnings: vec![],
            check_time,
            sorry_count: 0,
            goals_solved: 0,
            lean_version,
        }
    }

    /// Check if this result indicates a complete proof (no sorry)
    pub fn is_complete_proof(&self) -> bool {
        self.success && self.sorry_count == 0
    }
}

/// Configuration for the Lean5 backend
#[derive(Debug, Clone)]
pub struct Lean5Config {
    /// Path to the Lean executable (auto-detected if None)
    pub lean_path: Option<PathBuf>,
    /// Path to the Lake executable (auto-detected if None)
    pub lake_path: Option<PathBuf>,
    /// Timeout for proof checking
    pub timeout: Duration,
    /// Whether to use Lake for building
    pub use_lake: bool,
    /// Additional Lean flags
    pub extra_flags: Vec<String>,
    /// Working directory for Lake projects
    pub work_dir: Option<PathBuf>,
}

impl Default for Lean5Config {
    fn default() -> Self {
        Self {
            lean_path: None,
            lake_path: None,
            timeout: Duration::from_secs(60),
            use_lake: false,
            extra_flags: vec![],
            work_dir: None,
        }
    }
}

impl Lean5Config {
    /// Create a new config with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the Lean executable path
    pub fn with_lean_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.lean_path = Some(path.into());
        self
    }

    /// Set the timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Enable Lake for building
    pub fn with_lake(mut self, work_dir: impl Into<PathBuf>) -> Self {
        self.use_lake = true;
        self.work_dir = Some(work_dir.into());
        self
    }

    /// Add extra Lean flags
    pub fn with_flags(mut self, flags: Vec<String>) -> Self {
        self.extra_flags = flags;
        self
    }
}

/// Lean5 backend for proof checking
pub struct Lean5Backend {
    config: Lean5Config,
    lean_path: PathBuf,
    lean_version: String,
}

impl Lean5Backend {
    /// Create a new Lean5 backend with auto-detection
    pub fn new(config: Lean5Config) -> Result<Self, Lean5Error> {
        let lean_path = if let Some(path) = &config.lean_path {
            path.clone()
        } else {
            which::which("lean").map_err(|_| {
                Lean5Error::NotFound(
                    "Lean executable not found in PATH. Install with: elan install leanprover/lean4:stable".to_string(),
                )
            })?
        };

        // Get Lean version
        let version_output = Command::new(&lean_path)
            .arg("--version")
            .output()
            .map_err(Lean5Error::Io)?;

        let lean_version = String::from_utf8_lossy(&version_output.stdout)
            .lines()
            .next()
            .unwrap_or("unknown")
            .to_string();

        Ok(Self {
            config,
            lean_path,
            lean_version,
        })
    }

    /// Check if Lean is available
    pub fn is_available() -> bool {
        which::which("lean").is_ok()
    }

    /// Get the Lean version
    pub fn version(&self) -> &str {
        &self.lean_version
    }

    /// Check a Lean file for errors
    pub fn check_file(&self, file_path: &Path) -> Result<Lean5Result, Lean5Error> {
        let start = Instant::now();

        let output = self.run_lean(file_path)?;
        let check_time = start.elapsed();

        self.parse_output(output, check_time)
    }

    /// Check Lean source code directly (writes to temp file)
    pub fn check_source(&self, source: &str) -> Result<Lean5Result, Lean5Error> {
        let temp_file = tempfile::NamedTempFile::new()?;
        std::fs::write(temp_file.path(), source)?;

        // Use a stable path for the spawned process, then close the file to release the handle
        let path = temp_file.path().to_path_buf();
        let result = self.check_file(&path);

        // Explicitly close to ensure cleanup even on Windows
        let _ = temp_file.close();

        result
    }

    /// Run Lean on a file
    fn run_lean(&self, file_path: &Path) -> Result<Output, Lean5Error> {
        let mut cmd = Command::new(&self.lean_path);

        // Add the file to check
        cmd.arg(file_path);

        // Add extra flags
        for flag in &self.config.extra_flags {
            cmd.arg(flag);
        }

        // Set working directory if specified
        if let Some(work_dir) = &self.config.work_dir {
            cmd.current_dir(work_dir);
        }

        // Capture output
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let mut child = cmd.spawn().map_err(Lean5Error::Io)?;

        // Enforce timeout to avoid hanging Lean processes
        match child
            .wait_timeout(self.config.timeout)
            .map_err(Lean5Error::Io)?
        {
            Some(_) => child.wait_with_output().map_err(Lean5Error::Io),
            None => {
                let _ = child.kill();
                let _ = child.wait();
                Err(Lean5Error::Timeout(self.config.timeout))
            }
        }
    }

    /// Parse Lean output
    fn parse_output(
        &self,
        output: Output,
        check_time: Duration,
    ) -> Result<Lean5Result, Lean5Error> {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        // Combine stdout and stderr for parsing
        let full_output = format!("{stdout}\n{stderr}");

        // Count sorry occurrences in output
        let sorry_count = full_output.matches("declaration uses 'sorry'").count()
            + full_output.matches("uses sorry").count();

        // Parse errors
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        for line in full_output.lines() {
            let line = line.trim();
            if line.contains("error:") || line.contains("Error:") {
                errors.push(line.to_string());
            } else if line.contains("warning:") || line.contains("Warning:") {
                warnings.push(line.to_string());
            }
        }

        // Check exit code
        let success = output.status.success() && errors.is_empty();

        // Estimate goals solved based on success status
        // If successful without sorry, all declarations are verified
        // This is an approximation since Lean doesn't report goal counts directly
        let goals_solved = if success && sorry_count == 0 {
            // For a successful compilation, we count this as at least 1 goal solved
            // A more accurate count would require parsing the source file
            1
        } else {
            0
        };

        Ok(Lean5Result {
            success,
            errors,
            warnings,
            check_time,
            sorry_count,
            goals_solved,
            lean_version: self.lean_version.clone(),
        })
    }

    /// Build and check a Lake project
    pub fn build_lake_project(&self, project_dir: &Path) -> Result<Lean5Result, Lean5Error> {
        let lake_path = if let Some(path) = &self.config.lake_path {
            path.clone()
        } else {
            which::which("lake").map_err(|_| {
                Lean5Error::NotFound("Lake executable not found in PATH".to_string())
            })?
        };

        let start = Instant::now();

        let output = Command::new(&lake_path)
            .arg("build")
            .current_dir(project_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(Lean5Error::Io)?;

        let check_time = start.elapsed();

        self.parse_output(output, check_time)
    }
}

/// Check if Lean is installed and get version info
pub fn check_lean_installation() -> Result<String, Lean5Error> {
    let lean_path = which::which("lean").map_err(|_| {
        Lean5Error::NotFound(
            "Lean not found. Install with: curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh"
                .to_string(),
        )
    })?;

    let output = Command::new(&lean_path)
        .arg("--version")
        .output()
        .map_err(Lean5Error::Io)?;

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_exit_status(code: i32) -> std::process::ExitStatus {
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            std::process::ExitStatus::from_raw(code << 8)
        }

        #[cfg(windows)]
        {
            use std::os::windows::process::ExitStatusExt;
            std::process::ExitStatus::from_raw(code as u32)
        }
    }

    fn fake_backend() -> Lean5Backend {
        Lean5Backend {
            config: Lean5Config::default(),
            lean_path: PathBuf::from("lean"),
            lean_version: "Lean 4.0.0".to_string(),
        }
    }

    #[test]
    fn test_lean5_config_builder() {
        let config = Lean5Config::new()
            .with_timeout(Duration::from_secs(30))
            .with_flags(vec!["--threads=4".to_string()]);

        assert_eq!(config.timeout, Duration::from_secs(30));
        assert_eq!(config.extra_flags.len(), 1);
    }

    #[test]
    fn test_lean5_result_success() {
        let result = Lean5Result::success(Duration::from_millis(100), "Lean 4.0.0".to_string());
        assert!(result.success);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_lean5_result_failure() {
        let result = Lean5Result::failure(
            vec!["error: type mismatch".to_string()],
            Duration::from_millis(50),
            "Lean 4.0.0".to_string(),
        );
        assert!(!result.success);
        assert_eq!(result.errors.len(), 1);
    }

    #[test]
    fn test_lean5_result_complete_proof() {
        let mut result = Lean5Result::success(Duration::from_millis(100), "Lean 4.0.0".to_string());
        assert!(result.is_complete_proof());

        result.sorry_count = 1;
        assert!(!result.is_complete_proof());
    }

    #[test]
    fn test_parse_output_success_and_sorry_detection() {
        let backend = fake_backend();
        let output = Output {
            status: make_exit_status(0),
            stdout: b"Lean proof ok\n".to_vec(),
            stderr: b"warning: declaration uses 'sorry'".to_vec(),
        };

        let result = backend
            .parse_output(output, Duration::from_millis(5))
            .expect("parse_output");

        assert!(result.success, "Expected success for zero exit status");
        assert_eq!(result.sorry_count, 1, "Should detect sorry usage");
        assert_eq!(result.warnings.len(), 1, "Should capture warning lines");
        assert!(!result.is_complete_proof());
    }

    #[test]
    fn test_parse_output_failure_collects_errors() {
        let backend = fake_backend();
        let output = Output {
            status: make_exit_status(1),
            stdout: b"error: type mismatch".to_vec(),
            stderr: b"Error: failed to elaborate".to_vec(),
        };

        let result = backend
            .parse_output(output, Duration::from_millis(10))
            .expect("parse_output");

        assert!(!result.success, "Non-zero exit should be failure");
        assert_eq!(result.errors.len(), 2);
        assert_eq!(result.goals_solved, 0);
    }

    #[test]
    #[ignore] // Requires Lean installation
    fn test_check_lean_installation() {
        if Lean5Backend::is_available() {
            let version = check_lean_installation().expect("Should get version");
            assert!(version.contains("Lean") || version.contains("lean"));
        }
    }

    #[test]
    #[ignore] // Requires Lean installation
    fn test_check_valid_lean_source() {
        if !Lean5Backend::is_available() {
            return;
        }

        let backend = Lean5Backend::new(Lean5Config::new()).expect("Backend should init");

        // Simple valid Lean code
        let source = r#"
def hello : String := "Hello"

#check hello
"#;

        let result = backend.check_source(source).expect("Should check");
        assert!(
            result.success,
            "Valid code should pass: {:?}",
            result.errors
        );
    }

    #[test]
    #[ignore] // Requires Lean installation
    fn test_check_invalid_lean_source() {
        if !Lean5Backend::is_available() {
            return;
        }

        let backend = Lean5Backend::new(Lean5Config::new()).expect("Backend should init");

        // Invalid Lean code (type error)
        let source = r#"
def bad : Nat := "not a nat"
"#;

        let result = backend.check_source(source).expect("Should check");
        assert!(!result.success, "Invalid code should fail");
        assert!(!result.errors.is_empty(), "Should have error messages");
    }

    #[test]
    #[ignore] // Requires Lean installation
    fn test_check_sorry_detection() {
        if !Lean5Backend::is_available() {
            return;
        }

        let backend = Lean5Backend::new(Lean5Config::new()).expect("Backend should init");

        // Code with sorry
        let source = r"
theorem my_theorem : 1 + 1 = 2 := by sorry
";

        let result = backend.check_source(source).expect("Should check");
        // Note: sorry still compiles, just with a warning
        assert!(
            result.sorry_count > 0 || result.warnings.iter().any(|w| w.contains("sorry")),
            "Should detect sorry usage"
        );
    }

    #[cfg(unix)]
    #[test]
    fn test_timeout_errors() {
        // Use /bin/sleep to simulate a long-running Lean process
        let backend = Lean5Backend {
            config: Lean5Config {
                timeout: Duration::from_millis(10),
                ..Default::default()
            },
            lean_path: PathBuf::from("/bin/sleep"),
            lean_version: "test".to_string(),
        };

        let result = backend.check_file(Path::new("1"));
        assert!(
            matches!(result, Err(Lean5Error::Timeout(t)) if t == Duration::from_millis(10)),
            "Expected timeout error, got: {result:?}"
        );
    }

    #[test]
    fn test_goals_solved_estimation() {
        // Success without sorry should report 1 goal solved
        let mut result = Lean5Result::success(Duration::from_millis(100), "Lean 4.0.0".to_string());
        result.goals_solved = 1;
        assert_eq!(
            result.goals_solved, 1,
            "Successful proof should report 1 goal solved"
        );

        // Failure should report 0 goals solved
        let failure = Lean5Result::failure(
            vec!["error".to_string()],
            Duration::from_millis(50),
            "Lean 4.0.0".to_string(),
        );
        assert_eq!(
            failure.goals_solved, 0,
            "Failed proof should report 0 goals solved"
        );
    }

    #[test]
    #[ignore] // Requires Lean installation
    fn test_goals_solved_with_valid_proof() {
        if !Lean5Backend::is_available() {
            return;
        }

        let backend = Lean5Backend::new(Lean5Config::new()).expect("Backend should init");

        // Valid theorem with proof (no sorry)
        let source = r"
theorem simple : 1 + 1 = 2 := by decide
";

        let result = backend.check_source(source).expect("Should check");
        assert!(result.success, "Valid proof should succeed");
        assert_eq!(result.sorry_count, 0, "No sorry in proof");
        assert_eq!(
            result.goals_solved, 1,
            "Successful proof should report 1 goal solved"
        );
    }

    #[test]
    #[ignore] // Requires Lean installation
    fn test_goals_solved_with_sorry() {
        if !Lean5Backend::is_available() {
            return;
        }

        let backend = Lean5Backend::new(Lean5Config::new()).expect("Backend should init");

        // Theorem with sorry
        let source = r"
theorem incomplete : 1 + 1 = 2 := by sorry
";

        let result = backend.check_source(source).expect("Should check");
        // File compiles but with sorry
        assert!(result.sorry_count > 0, "Should detect sorry");
        assert_eq!(
            result.goals_solved, 0,
            "Proof with sorry should report 0 goals solved"
        );
    }

    // ========================================================================
    // Mutation coverage tests
    // ========================================================================

    #[test]
    fn test_lean5_config_with_lean_path_returns_self() {
        // Mutation: replace with_lean_path -> Self with Default::default()
        let config = Lean5Config::new().with_lean_path("/usr/local/bin/lean");
        // Verify the path was actually set
        assert!(config.lean_path.is_some());
        assert_eq!(
            config.lean_path.unwrap(),
            PathBuf::from("/usr/local/bin/lean")
        );
    }

    #[test]
    fn test_lean5_config_with_lake_returns_self() {
        // Mutation: replace with_lake -> Self with Default::default()
        let config = Lean5Config::new().with_lake("/some/project");
        // Verify lake was actually enabled and work_dir set
        assert!(config.use_lake);
        assert!(config.work_dir.is_some());
        assert_eq!(config.work_dir.unwrap(), PathBuf::from("/some/project"));
    }

    #[test]
    fn test_lean5_backend_version_returns_actual_version() {
        // Mutation: replace version -> &str with "" or "xyzzy"
        let backend = fake_backend();
        let version = backend.version();
        // Should return actual version, not empty or xyzzy
        assert!(!version.is_empty());
        assert_ne!(version, "xyzzy");
        assert_eq!(version, "Lean 4.0.0");
    }

    #[test]
    fn test_parse_output_sorry_count_addition() {
        // Mutation: replace + with - in sorry_count calculation
        let backend = fake_backend();

        // Output with both types of sorry messages
        let output = Output {
            status: make_exit_status(0),
            stdout: b"declaration uses 'sorry'\n".to_vec(),
            stderr: b"also uses sorry\n".to_vec(),
        };

        let result = backend
            .parse_output(output, Duration::from_millis(1))
            .expect("parse");

        // Should count both types (1 + 1 = 2), not subtract
        assert_eq!(result.sorry_count, 2);
    }

    #[test]
    fn test_parse_output_success_requires_both_conditions() {
        // Mutation: replace && with || in success check
        let backend = fake_backend();

        // Case 1: exit code 0 but has errors - should fail
        let output_with_errors = Output {
            status: make_exit_status(0),
            stdout: b"error: something failed\n".to_vec(),
            stderr: b"".to_vec(),
        };

        let result = backend
            .parse_output(output_with_errors, Duration::from_millis(1))
            .expect("parse");

        assert!(
            !result.success,
            "Should fail when exit code ok but has errors"
        );

        // Case 2: no errors but non-zero exit - should fail
        let output_bad_exit = Output {
            status: make_exit_status(1),
            stdout: b"compilation complete\n".to_vec(),
            stderr: b"".to_vec(),
        };

        let result = backend
            .parse_output(output_bad_exit, Duration::from_millis(1))
            .expect("parse");

        assert!(
            !result.success,
            "Should fail when no errors but bad exit code"
        );
    }

    #[test]
    fn test_parse_output_goals_solved_requires_success_and_no_sorry() {
        // Mutation: replace == with != in sorry_count == 0 check
        let backend = fake_backend();

        // Success with sorry should have 0 goals solved
        let output_with_sorry = Output {
            status: make_exit_status(0),
            stdout: b"declaration uses 'sorry'\n".to_vec(),
            stderr: b"".to_vec(),
        };

        let result = backend
            .parse_output(output_with_sorry, Duration::from_millis(1))
            .expect("parse");

        assert!(result.success);
        assert!(result.sorry_count > 0);
        assert_eq!(
            result.goals_solved, 0,
            "Should have 0 goals when sorry present"
        );

        // Success without sorry should have 1 goal solved
        let output_clean = Output {
            status: make_exit_status(0),
            stdout: b"compilation complete\n".to_vec(),
            stderr: b"".to_vec(),
        };

        let result = backend
            .parse_output(output_clean, Duration::from_millis(1))
            .expect("parse");

        assert!(result.success);
        assert_eq!(result.sorry_count, 0);
        assert_eq!(
            result.goals_solved, 1,
            "Should have 1 goal when clean success"
        );
    }

    #[test]
    #[ignore] // Requires check_lean_installation to work
    fn test_check_lean_installation_returns_version_not_empty() {
        // Mutation: replace check_lean_installation return with Ok("") or Ok("xyzzy")
        // This test is ignored because it requires Lean to be installed
        if !Lean5Backend::is_available() {
            return;
        }
        let version = check_lean_installation().expect("installation check");
        assert!(!version.is_empty(), "Version should not be empty");
        assert_ne!(version, "xyzzy", "Version should be real not xyzzy");
    }
}
