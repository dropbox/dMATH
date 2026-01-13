//! Kani execution logic

use super::config::{KaniConfig, KaniDetection, KaniOutput};
use crate::traits::BackendError;
use std::path::Path;
use std::process::Stdio;
use std::time::Instant;
use tokio::process::Command;
use tracing::debug;

/// Run cargo kani on the generated harness project
pub async fn run_kani(
    config: &KaniConfig,
    detection: &KaniDetection,
    manifest_path: &Path,
) -> Result<KaniOutput, BackendError> {
    let cargo_path = match detection {
        KaniDetection::Available { cargo_path } => cargo_path,
        KaniDetection::NotFound(reason) => return Err(BackendError::Unavailable(reason.clone())),
    };

    let mut cmd = Command::new(cargo_path);
    cmd.arg("kani").arg("--manifest-path").arg(manifest_path);

    if config.enable_concrete_playback {
        cmd.arg("-Z")
            .arg("concrete-playback")
            .arg("--concrete-playback=print");
    }

    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    cmd.current_dir(manifest_path.parent().unwrap_or(Path::new(".")));

    let start = Instant::now();
    let output = tokio::time::timeout(config.timeout, cmd.output())
        .await
        .map_err(|_| BackendError::Timeout(config.timeout))?
        .map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to execute cargo kani: {}", e))
        })?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let duration = start.elapsed();

    debug!("cargo kani stdout:\n{}", stdout);
    if !stderr.is_empty() {
        debug!("cargo kani stderr:\n{}", stderr);
    }

    Ok(KaniOutput {
        stdout,
        stderr,
        exit_code: output.status.code(),
        duration,
    })
}
