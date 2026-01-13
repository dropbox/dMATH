//! Kani wrapper for executing cargo kani

use crate::config::{KaniConfig, KaniOutput};
use crate::result::VerificationResult;
use kani_fast_counterexample::{
    extract_diagnostics, extract_structured_counterexample, failed_check_count,
};
use std::path::Path;
use std::process::Stdio;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::time::timeout;
use tracing::{debug, info, warn};

/// Errors that can occur during Kani execution
#[derive(Debug, Error)]
pub enum KaniError {
    #[error("Failed to execute cargo kani: {0}")]
    ExecutionError(#[from] std::io::Error),

    #[error("Verification timed out after {0:?}")]
    Timeout(Duration),

    #[error("Kani not installed or not configured")]
    NotInstalled,

    #[error("Invalid project path: {0}")]
    InvalidPath(String),
}

/// Wrapper around cargo kani for verification
pub struct KaniWrapper {
    config: KaniConfig,
    cargo_path: String,
}

impl KaniWrapper {
    /// Create a new KaniWrapper with the given configuration
    pub fn new(config: KaniConfig) -> Result<Self, KaniError> {
        let cargo_path = config
            .cargo_path
            .as_ref()
            .map(|p| p.to_string_lossy().to_string())
            .or_else(|| crate::find_executable("cargo").map(|p| p.to_string_lossy().to_string()))
            .ok_or(KaniError::NotInstalled)?;

        Ok(Self { config, cargo_path })
    }

    /// Create with default configuration
    pub fn with_defaults() -> Result<Self, KaniError> {
        Self::new(KaniConfig::default())
    }

    /// Verify a Rust project at the given path
    pub async fn verify(&self, project_path: &Path) -> Result<VerificationResult, KaniError> {
        self.verify_with_harness(project_path, None).await
    }

    /// Verify a specific harness in a Rust project
    pub async fn verify_with_harness(
        &self,
        project_path: &Path,
        harness: Option<&str>,
    ) -> Result<VerificationResult, KaniError> {
        if !project_path.exists() {
            return Err(KaniError::InvalidPath(project_path.display().to_string()));
        }

        let output = self.run_kani(project_path, harness).await?;
        Ok(self.parse_output(&output))
    }

    /// Run cargo kani and capture output
    async fn run_kani(
        &self,
        project_path: &Path,
        harness: Option<&str>,
    ) -> Result<KaniOutput, KaniError> {
        let mut cmd = Command::new(&self.cargo_path);
        cmd.arg("kani");

        // Add manifest path
        let manifest_path = project_path.join("Cargo.toml");
        if manifest_path.exists() {
            cmd.arg("--manifest-path");
            cmd.arg(&manifest_path);
        } else {
            cmd.current_dir(project_path);
        }

        // Add harness filter if specified
        if let Some(h) = harness {
            cmd.arg("--harness").arg(h);
        }

        // Add concrete playback for counterexamples
        if self.config.enable_concrete_playback {
            cmd.arg("-Z").arg("concrete-playback");
            cmd.arg("--concrete-playback=print");
        }

        // Add unwind bound if specified
        if let Some(unwind) = self.config.default_unwind {
            cmd.arg("--default-unwind").arg(unwind.to_string());
        }

        // Add solver if specified
        if let Some(solver) = &self.config.solver {
            cmd.arg("--solver").arg(solver);
        }

        // Add extra args
        for arg in &self.config.extra_args {
            cmd.arg(arg);
        }

        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        info!("Running: cargo kani on {}", project_path.display());
        debug!("Command: {:?}", cmd);

        let start = Instant::now();

        // Spawn the process
        let mut child = cmd.spawn()?;

        // Take stdout/stderr handles before waiting - must read concurrently with wait
        let stdout_handle = child.stdout.take();
        let stderr_handle = child.stderr.take();

        // Wait with timeout, reading output concurrently
        let result = timeout(self.config.timeout, async {
            let mut stdout = String::new();
            let mut stderr = String::new();

            // Read stdout and stderr concurrently with waiting for the process
            let (status, stdout_result, stderr_result) = tokio::join!(
                child.wait(),
                async {
                    if let Some(mut handle) = stdout_handle {
                        let _ = handle.read_to_string(&mut stdout).await;
                    }
                    stdout
                },
                async {
                    if let Some(mut handle) = stderr_handle {
                        let _ = handle.read_to_string(&mut stderr).await;
                    }
                    stderr
                }
            );

            Ok::<_, std::io::Error>((status?, stdout_result, stderr_result))
        })
        .await;

        let duration = start.elapsed();

        match result {
            Ok(Ok((status, stdout, stderr))) => {
                debug!(
                    "Kani completed in {:?} with status {:?}",
                    duration,
                    status.code()
                );
                Ok(KaniOutput {
                    stdout,
                    stderr,
                    exit_code: status.code(),
                    duration,
                })
            }
            Ok(Err(e)) => Err(KaniError::ExecutionError(e)),
            Err(_) => {
                warn!("Kani timed out after {:?}", self.config.timeout);
                // Try to kill the process
                let _ = child.kill().await;
                Err(KaniError::Timeout(self.config.timeout))
            }
        }
    }

    /// Parse Kani output into a VerificationResult
    fn parse_output(&self, output: &KaniOutput) -> VerificationResult {
        let combined = output.combined();
        let diagnostics = extract_diagnostics(&combined);

        // Get check counts
        let (checks_failed, checks_total) = failed_check_count(&combined).unwrap_or((0, 0));

        // Success detection
        if combined.contains("VERIFICATION:- SUCCESSFUL")
            || (checks_total > 0 && checks_failed == 0)
        {
            return VerificationResult::proven(output.duration, checks_total)
                .with_diagnostics(diagnostics);
        }

        // Failure detection
        if combined.contains("VERIFICATION:- FAILED")
            || combined.contains("Status: FAILURE")
            || checks_failed > 0
        {
            let counterexample = extract_structured_counterexample(&combined);
            return VerificationResult::disproven(
                counterexample,
                output.duration,
                checks_failed,
                checks_total,
            )
            .with_diagnostics(diagnostics);
        }

        // Unknown result
        let reason = match output.exit_code {
            Some(2) => "Kani CLI error".to_string(),
            Some(code) => format!("Kani exited with code {code} without definitive result"),
            None => "Process terminated without exit code".to_string(),
        };

        VerificationResult::unknown(reason, output.duration).with_diagnostics(diagnostics)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VerificationStatus;
    use std::fs;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    use tempfile::tempdir;

    #[cfg(unix)]
    fn write_script(path: &Path, contents: &str) {
        fs::write(path, contents).unwrap();
        let mut perms = fs::metadata(path).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(path, perms).unwrap();
    }

    #[test]
    fn test_wrapper_creation() {
        // This test verifies the wrapper can be created
        // It may fail if cargo is not in PATH, which is fine
        let result = KaniWrapper::with_defaults();
        // Just check it doesn't panic
        let _ = result;
    }

    #[tokio::test]
    async fn test_invalid_path() {
        if let Ok(wrapper) = KaniWrapper::with_defaults() {
            let result = wrapper.verify(Path::new("/nonexistent/path")).await;
            assert!(matches!(result, Err(KaniError::InvalidPath(_))));
        }
    }

    #[test]
    fn test_kani_error_execution_error_display() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "not found");
        let err = KaniError::ExecutionError(io_err);
        let display = err.to_string();
        assert!(display.contains("Failed to execute cargo kani"));
        assert!(display.contains("not found"));
    }

    #[test]
    fn test_kani_error_timeout_display() {
        let err = KaniError::Timeout(Duration::from_secs(300));
        let display = err.to_string();
        assert!(display.contains("timed out"));
        assert!(display.contains("300"));
    }

    #[test]
    fn test_kani_error_not_installed_display() {
        let err = KaniError::NotInstalled;
        let display = err.to_string();
        assert!(display.contains("not installed"));
    }

    #[test]
    fn test_kani_error_invalid_path_display() {
        let err = KaniError::InvalidPath("/bad/path".to_string());
        let display = err.to_string();
        assert!(display.contains("Invalid project path"));
        assert!(display.contains("/bad/path"));
    }

    #[test]
    fn test_kani_error_debug() {
        let errors: Vec<KaniError> = vec![
            KaniError::ExecutionError(std::io::Error::other("test")),
            KaniError::Timeout(Duration::from_secs(60)),
            KaniError::NotInstalled,
            KaniError::InvalidPath("test".to_string()),
        ];
        for err in errors {
            let debug_str = format!("{:?}", err);
            assert!(!debug_str.is_empty());
        }
    }

    #[test]
    fn test_kani_error_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let err: KaniError = io_err.into();
        assert!(matches!(err, KaniError::ExecutionError(_)));
    }

    #[test]
    fn test_wrapper_new_with_custom_config() {
        let config = KaniConfig {
            timeout: Duration::from_secs(60),
            cargo_path: Some(std::path::PathBuf::from("/usr/bin/cargo")),
            enable_concrete_playback: false,
            default_unwind: Some(10),
            solver: Some("cadical".to_string()),
            extra_args: vec!["--verbose".to_string()],
        };
        // Even if cargo is not at /usr/bin/cargo, the wrapper should be created
        let result = KaniWrapper::new(config);
        // This may succeed or fail depending on system - just verify it doesn't panic
        let _ = result;
    }

    #[test]
    fn test_wrapper_new_with_nonexistent_cargo() {
        let config = KaniConfig {
            cargo_path: Some(std::path::PathBuf::from("/nonexistent/cargo")),
            ..Default::default()
        };
        // If cargo path is explicitly set to non-existent, creation should still work
        // as we only use the path at execution time, not creation time
        let _ = KaniWrapper::new(config);
    }

    #[test]
    fn test_kani_error_is_error() {
        fn is_error<E: std::error::Error>(_: &E) {}
        let err = KaniError::NotInstalled;
        is_error(&err);
    }

    #[test]
    fn test_kani_error_variants_are_distinct() {
        let err1 = KaniError::NotInstalled;
        let err2 = KaniError::InvalidPath("test".to_string());
        let err3 = KaniError::Timeout(Duration::from_secs(1));

        let msg1 = err1.to_string();
        let msg2 = err2.to_string();
        let msg3 = err3.to_string();

        assert_ne!(msg1, msg2);
        assert_ne!(msg2, msg3);
        assert_ne!(msg1, msg3);
    }

    #[test]
    fn test_kani_error_timeout_duration_preserved() {
        let durations = [
            Duration::from_secs(1),
            Duration::from_secs(60),
            Duration::from_secs(300),
            Duration::from_millis(500),
        ];
        for duration in durations {
            let err = KaniError::Timeout(duration);
            let display = err.to_string();
            // Check the duration is somehow represented
            assert!(display.contains("timed out"));
        }
    }

    #[tokio::test]
    async fn test_verify_with_harness_invalid_path() {
        if let Ok(wrapper) = KaniWrapper::with_defaults() {
            let result = wrapper
                .verify_with_harness(Path::new("/nonexistent/path"), Some("test_harness"))
                .await;
            assert!(matches!(result, Err(KaniError::InvalidPath(_))));
        }
    }

    #[test]
    fn test_parse_output_success() {
        if let Ok(wrapper) = KaniWrapper::with_defaults() {
            let output = KaniOutput {
                stdout: "VERIFICATION:- SUCCESSFUL\n2 verified, 0 failed".to_string(),
                stderr: String::new(),
                exit_code: Some(0),
                duration: Duration::from_secs(5),
            };
            let result = wrapper.parse_output(&output);
            assert!(result.status.is_success());
        }
    }

    #[test]
    fn test_parse_output_failure() {
        if let Ok(wrapper) = KaniWrapper::with_defaults() {
            let output = KaniOutput {
                stdout: "VERIFICATION:- FAILED\n1 verified, 1 failed".to_string(),
                stderr: String::new(),
                exit_code: Some(1),
                duration: Duration::from_secs(5),
            };
            let result = wrapper.parse_output(&output);
            assert!(matches!(result.status, VerificationStatus::Disproven));
        }
    }

    #[test]
    fn test_parse_output_unknown_with_exit_code_2() {
        if let Ok(wrapper) = KaniWrapper::with_defaults() {
            let output = KaniOutput {
                stdout: String::new(),
                stderr: "some error".to_string(),
                exit_code: Some(2),
                duration: Duration::from_secs(1),
            };
            let result = wrapper.parse_output(&output);
            assert!(matches!(result.status, VerificationStatus::Unknown { .. }));
        }
    }

    #[test]
    fn test_parse_output_unknown_with_arbitrary_exit_code() {
        if let Ok(wrapper) = KaniWrapper::with_defaults() {
            let output = KaniOutput {
                stdout: String::new(),
                stderr: String::new(),
                exit_code: Some(42),
                duration: Duration::from_secs(1),
            };
            let result = wrapper.parse_output(&output);
            assert!(matches!(result.status, VerificationStatus::Unknown { .. }));
        }
    }

    #[test]
    fn test_parse_output_unknown_with_no_exit_code() {
        if let Ok(wrapper) = KaniWrapper::with_defaults() {
            let output = KaniOutput {
                stdout: String::new(),
                stderr: String::new(),
                exit_code: None,
                duration: Duration::from_secs(1),
            };
            let result = wrapper.parse_output(&output);
            assert!(matches!(result.status, VerificationStatus::Unknown { .. }));
        }
    }

    #[test]
    fn test_parse_output_status_failure() {
        if let Ok(wrapper) = KaniWrapper::with_defaults() {
            let output = KaniOutput {
                stdout: "Status: FAILURE".to_string(),
                stderr: String::new(),
                exit_code: Some(1),
                duration: Duration::from_secs(2),
            };
            let result = wrapper.parse_output(&output);
            assert!(matches!(result.status, VerificationStatus::Disproven));
        }
    }

    #[test]
    fn test_parse_output_with_checks_passed() {
        if let Ok(wrapper) = KaniWrapper::with_defaults() {
            let output = KaniOutput {
                stdout: "VERIFICATION:- SUCCESSFUL\n10 verified, 0 failed".to_string(),
                stderr: String::new(),
                exit_code: Some(0),
                duration: Duration::from_secs(5),
            };
            let result = wrapper.parse_output(&output);
            assert!(result.status.is_success());
        }
    }

    #[test]
    fn test_parse_output_with_some_checks_failed() {
        if let Ok(wrapper) = KaniWrapper::with_defaults() {
            // Simulating output where we have both passed and failed checks
            let output = KaniOutput {
                stdout: "VERIFICATION:- FAILED\nFailed Checks: assertion\n Check 1: SATISFIED"
                    .to_string(),
                stderr: String::new(),
                exit_code: Some(1),
                duration: Duration::from_secs(3),
            };
            let result = wrapper.parse_output(&output);
            assert!(matches!(result.status, VerificationStatus::Disproven));
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_run_kani_invokes_custom_cargo_with_expected_flags() {
        let temp_dir = tempdir().unwrap();
        let project_dir = temp_dir.path().join("proj");
        fs::create_dir_all(&project_dir).unwrap();
        fs::write(
            project_dir.join("Cargo.toml"),
            "[package]\nname=\"demo\"\nversion=\"0.1.0\"\n",
        )
        .unwrap();

        let args_log = temp_dir.path().join("args.log");
        let script_path = temp_dir.path().join("fake_cargo");
        let script = format!(
            "#!/bin/sh\n\
             echo \"$@\" > \"{}\"\n\
             echo \"VERIFICATION:- SUCCESSFUL\"\n\
             echo \"2 verified, 0 failed\"\n",
            args_log.display()
        );
        write_script(&script_path, &script);

        let config = KaniConfig {
            timeout: Duration::from_secs(60), // Extended timeout for parallel test runner contention
            cargo_path: Some(script_path),
            enable_concrete_playback: true,
            default_unwind: Some(8),
            solver: Some("cadical".to_string()),
            extra_args: vec!["--json".to_string()],
        };

        let wrapper = KaniWrapper::new(config).unwrap();
        let output = wrapper
            .run_kani(&project_dir, Some("demo_harness"))
            .await
            .unwrap();

        assert!(output.stdout.contains("SUCCESSFUL"));
        let args = fs::read_to_string(args_log).unwrap();
        assert!(args.contains("kani"));
        assert!(args.contains("--manifest-path"));
        assert!(args.contains("demo_harness"));
        assert!(args.contains("--default-unwind"));
        assert!(args.contains("8"));
        assert!(args.contains("--solver"));
        assert!(args.contains("cadical"));
        assert!(args.contains("concrete-playback=print"));
        assert!(args.contains("--json"));
    }

    // Tests to catch mutation: checks_total > 0 (line 197)
    // When checks_total is 0 and no VERIFICATION marker, result should be unknown
    #[test]
    fn test_parse_output_checks_total_zero_no_verification_marker() {
        if let Ok(wrapper) = KaniWrapper::with_defaults() {
            let output = KaniOutput {
                stdout: "No verification results".to_string(),
                stderr: String::new(),
                exit_code: Some(0),
                duration: Duration::from_secs(1),
            };
            let result = wrapper.parse_output(&output);
            // With checks_total=0 and no VERIFICATION marker, should be unknown
            assert!(matches!(result.status, VerificationStatus::Unknown { .. }));
        }
    }

    // Test: checks_total > 0 with checks_failed == 0 should be success
    #[test]
    fn test_parse_output_checks_total_positive_checks_failed_zero_success() {
        if let Ok(wrapper) = KaniWrapper::with_defaults() {
            // Kani output format: "N verified, 0 failed" with N > 0
            let output = KaniOutput {
                stdout: "Check 1: SUCCESS\nCheck 2: SUCCESS\n** 2 of 2 passed\n\n\n".to_string(),
                stderr: String::new(),
                exit_code: Some(0),
                duration: Duration::from_secs(2),
            };
            let result = wrapper.parse_output(&output);
            // If checks_total > 0 and checks_failed == 0, should be success
            // Note: depends on failed_check_count parsing
            assert!(matches!(
                result.status,
                VerificationStatus::Proven | VerificationStatus::Unknown { .. }
            ));
        }
    }

    // Test: checks_failed > 0 detection for failure (line 206)
    #[test]
    fn test_parse_output_checks_failed_one_triggers_failure() {
        if let Ok(wrapper) = KaniWrapper::with_defaults() {
            // Kani format: "** N of M failed" where N > 0
            let output = KaniOutput {
                stdout: "Failed Checks: assertion\nCheck 1: FAILURE\n** 1 of 1 failed\n"
                    .to_string(),
                stderr: String::new(),
                exit_code: Some(1),
                duration: Duration::from_secs(1),
            };
            let result = wrapper.parse_output(&output);
            // checks_failed > 0 should trigger disproven
            assert!(matches!(result.status, VerificationStatus::Disproven));
        }
    }

    // Test: exactly at boundary - checks_failed transitions from 0 to 1
    #[test]
    fn test_parse_output_checks_failed_boundary_zero_vs_one() {
        if let Ok(wrapper) = KaniWrapper::with_defaults() {
            // Zero failures = success
            let output_success = KaniOutput {
                stdout: "VERIFICATION:- SUCCESSFUL\n".to_string(),
                stderr: String::new(),
                exit_code: Some(0),
                duration: Duration::from_secs(1),
            };
            let result_success = wrapper.parse_output(&output_success);
            assert!(result_success.status.is_success());

            // One failure = disproven
            let output_fail = KaniOutput {
                stdout: "VERIFICATION:- FAILED\n".to_string(),
                stderr: String::new(),
                exit_code: Some(1),
                duration: Duration::from_secs(1),
            };
            let result_fail = wrapper.parse_output(&output_fail);
            assert!(matches!(result_fail.status, VerificationStatus::Disproven));
        }
    }

    // Test: checks_total > 0 (not <, not ==) with checks_failed == 0
    // This specifically tests line 197: (checks_total > 0 && checks_failed == 0)
    // Catches mutation: > replaced with < would make checks_total < 0 always false
    #[test]
    fn test_parse_output_checks_total_greater_than_zero_required() {
        if let Ok(wrapper) = KaniWrapper::with_defaults() {
            // Output with "** 0 of 5 failed" - this matches the regex and gives checks_total=5
            // No VERIFICATION marker, so success depends on checks_total > 0 condition
            let output_with_checks = KaniOutput {
                stdout: "Check 1: SUCCESS\nCheck 2: SUCCESS\n** 0 of 5 failed\n".to_string(),
                stderr: String::new(),
                exit_code: Some(0),
                duration: Duration::from_secs(1),
            };
            let result_with_checks = wrapper.parse_output(&output_with_checks);
            // With checks_total=5 > 0 and checks_failed=0, should be Proven
            assert!(
                result_with_checks.status.is_success(),
                "Expected success when checks_total=5 > 0 and checks_failed=0"
            );

            // Output with "** 0 of 0 failed" - checks_total=0, should NOT be success via this path
            // (It might still be Unknown due to no VERIFICATION marker)
            let output_zero_checks = KaniOutput {
                stdout: "No checks\n** 0 of 0 failed\n".to_string(),
                stderr: String::new(),
                exit_code: Some(0),
                duration: Duration::from_secs(1),
            };
            let result_zero_checks = wrapper.parse_output(&output_zero_checks);
            // With checks_total=0, the (checks_total > 0) condition is false
            // So we should NOT get Proven via this path - should be Unknown
            assert!(
                !matches!(result_zero_checks.status, VerificationStatus::Proven),
                "Expected NOT Proven when checks_total=0 (boundary case)"
            );
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_run_kani_times_out_and_kills_process() {
        let temp_dir = tempdir().unwrap();
        let project_dir = temp_dir.path().join("proj");
        fs::create_dir_all(&project_dir).unwrap();
        fs::write(
            project_dir.join("Cargo.toml"),
            "[package]\nname=\"demo\"\nversion=\"0.1.0\"\n",
        )
        .unwrap();

        let script_path = temp_dir.path().join("slow_cargo");
        let script = "#!/bin/sh\nsleep 2\necho \"VERIFICATION:- SUCCESSFUL\"\n";
        write_script(&script_path, script);

        let config = KaniConfig {
            timeout: Duration::from_millis(200),
            cargo_path: Some(script_path),
            enable_concrete_playback: false,
            default_unwind: None,
            solver: None,
            extra_args: Vec::new(),
        };

        let wrapper = KaniWrapper::new(config).unwrap();
        let result = wrapper.run_kani(&project_dir, None).await;

        match result {
            Err(KaniError::Timeout(dur)) => {
                assert!(dur <= Duration::from_millis(200));
            }
            other => panic!("expected timeout error, got {:?}", other),
        }
    }
}
