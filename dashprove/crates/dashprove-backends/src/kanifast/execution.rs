//! Kani Fast execution logic

use super::config::{KaniFastConfig, KaniFastDetection, KaniFastOutput};
use crate::counterexample::StructuredCounterexample;
use crate::traits::{BackendError, BackendId, BackendResult, VerificationStatus};
use dashprove_usl::typecheck::TypedSpec;
use std::path::Path;
use std::process::Stdio;
use std::time::{Duration, Instant};
use tokio::process::Command;
use tracing::debug;

/// Run kani-fast CLI on the project
pub async fn run_kani_fast(
    config: &KaniFastConfig,
    detection: &KaniFastDetection,
    manifest_path: &Path,
) -> Result<KaniFastOutput, BackendError> {
    let cli_path = match detection {
        KaniFastDetection::Available { cli_path } => cli_path,
        KaniFastDetection::NotFound(reason) => {
            return Err(BackendError::Unavailable(reason.clone()))
        }
    };

    let mut cmd = Command::new(cli_path);
    cmd.arg("verify");

    // Add project path
    if let Some(parent) = manifest_path.parent() {
        cmd.arg(parent);
    }

    // Add verification mode flag
    if let Some(mode_flag) = config.mode.as_flag() {
        cmd.arg(mode_flag);
    }

    // Add AI flag if enabled
    if config.enable_ai {
        cmd.arg("--ai");
        cmd.arg("--ai-max-attempts")
            .arg(config.ai_max_attempts.to_string());
    }

    // Add explain flag if enabled
    if config.enable_explain {
        cmd.arg("--explain");
    }

    // Add Lean5 flag if enabled
    if config.enable_lean5 {
        cmd.arg("--lean5");
    }

    // Add specific harness if specified
    if let Some(ref harness) = config.harness {
        cmd.arg("--harness").arg(harness);
    }

    // Add parallel solvers count if specified
    if let Some(solvers) = config.parallel_solvers {
        cmd.arg("--solvers").arg(solvers.to_string());
    }

    // Request JSON output for structured parsing
    cmd.arg("--json");

    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let start = Instant::now();
    let output = tokio::time::timeout(config.timeout, cmd.output())
        .await
        .map_err(|_| BackendError::Timeout(config.timeout))?
        .map_err(|e| {
            BackendError::VerificationFailed(format!("Failed to execute kani-fast: {}", e))
        })?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let duration = start.elapsed();

    debug!("kani-fast stdout:\n{}", stdout);
    if !stderr.is_empty() {
        debug!("kani-fast stderr:\n{}", stderr);
    }

    Ok(KaniFastOutput {
        stdout,
        stderr,
        exit_code: output.status.code(),
        duration,
    })
}

/// Parse kani-fast output into BackendResult
pub fn parse_output(output: &KaniFastOutput, _spec: &TypedSpec) -> BackendResult {
    // Try to parse JSON output first
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&output.stdout) {
        return parse_json_output(&json, output.duration);
    }

    // Fall back to text parsing
    parse_text_output(output)
}

pub(crate) fn parse_json_output(json: &serde_json::Value, duration: Duration) -> BackendResult {
    let status_str = json
        .get("status")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    let status = match status_str {
        "proven" | "verified" | "success" => VerificationStatus::Proven,
        "disproven" | "failed" | "counterexample" => VerificationStatus::Disproven,
        "timeout" => VerificationStatus::Unknown {
            reason: "Verification timed out".to_string(),
        },
        "error" => VerificationStatus::Unknown {
            reason: json
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown error")
                .to_string(),
        },
        _ => VerificationStatus::Unknown {
            reason: format!("Unknown status: {}", status_str),
        },
    };

    let proof = json.get("proof").and_then(|v| v.as_str()).map(String::from);

    let counterexample = json.get("counterexample").and_then(parse_counterexample);

    let diagnostics = json
        .get("diagnostics")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    BackendResult {
        backend: BackendId::KaniFast,
        status,
        proof,
        counterexample,
        diagnostics,
        time_taken: duration,
    }
}

fn parse_counterexample(json: &serde_json::Value) -> Option<StructuredCounterexample> {
    let mut ce = StructuredCounterexample::default();

    // Parse witness values
    if let Some(witness) = json.get("witness").and_then(|v| v.as_object()) {
        for (key, value) in witness {
            if let Some(val) = parse_counterexample_value(value) {
                ce.witness.insert(key.clone(), val);
            }
        }
    }

    // Parse failed checks
    if let Some(checks) = json.get("failed_checks").and_then(|v| v.as_array()) {
        for check in checks {
            if let Some(failed) = parse_failed_check(check) {
                ce.failed_checks.push(failed);
            }
        }
    }

    // Parse playback test
    if let Some(playback) = json.get("playback_test").and_then(|v| v.as_str()) {
        ce.playback_test = Some(playback.to_string());
    }

    // Parse raw text if available
    if let Some(raw) = json.get("raw").and_then(|v| v.as_str()) {
        ce.raw = Some(raw.to_string());
    }

    // Parse explanation if available (from --explain flag)
    if let Some(explanation) = json.get("explanation").and_then(|v| v.as_str()) {
        // Store explanation in raw field for now
        if ce.raw.is_none() {
            ce.raw = Some(explanation.to_string());
        }
    }

    if ce.witness.is_empty() && ce.failed_checks.is_empty() && ce.raw.is_none() {
        None
    } else {
        Some(ce)
    }
}

fn parse_counterexample_value(
    json: &serde_json::Value,
) -> Option<crate::counterexample::CounterexampleValue> {
    use crate::counterexample::CounterexampleValue;

    match json {
        serde_json::Value::Bool(b) => Some(CounterexampleValue::Bool(*b)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(CounterexampleValue::Int {
                    value: i as i128,
                    type_hint: None,
                })
            } else if let Some(u) = n.as_u64() {
                Some(CounterexampleValue::UInt {
                    value: u as u128,
                    type_hint: None,
                })
            } else {
                n.as_f64().map(|f| CounterexampleValue::Float { value: f })
            }
        }
        serde_json::Value::String(s) => Some(CounterexampleValue::String(s.clone())),
        serde_json::Value::Array(arr) => {
            let values: Vec<_> = arr.iter().filter_map(parse_counterexample_value).collect();
            Some(CounterexampleValue::Sequence(values))
        }
        serde_json::Value::Object(obj) => {
            // Check if it has a type hint
            if let Some(ty) = obj.get("type").and_then(|v| v.as_str()) {
                if let Some(val) = obj.get("value") {
                    match ty {
                        "int" | "i8" | "i16" | "i32" | "i64" | "i128" | "isize" => {
                            if let Some(i) = val.as_i64() {
                                return Some(CounterexampleValue::Int {
                                    value: i as i128,
                                    type_hint: Some(ty.to_string()),
                                });
                            }
                        }
                        "uint" | "u8" | "u16" | "u32" | "u64" | "u128" | "usize" => {
                            if let Some(u) = val.as_u64() {
                                return Some(CounterexampleValue::UInt {
                                    value: u as u128,
                                    type_hint: Some(ty.to_string()),
                                });
                            }
                        }
                        "bytes" => {
                            if let Some(arr) = val.as_array() {
                                let bytes: Vec<u8> = arr
                                    .iter()
                                    .filter_map(|v| v.as_u64().map(|n| n as u8))
                                    .collect();
                                return Some(CounterexampleValue::Bytes(bytes));
                            }
                        }
                        _ => {}
                    }
                }
            }

            // Parse as record
            let fields: std::collections::HashMap<String, _> = obj
                .iter()
                .filter_map(|(k, v)| parse_counterexample_value(v).map(|val| (k.clone(), val)))
                .collect();
            Some(CounterexampleValue::Record(fields))
        }
        serde_json::Value::Null => None,
    }
}

fn parse_failed_check(json: &serde_json::Value) -> Option<crate::counterexample::FailedCheck> {
    use crate::counterexample::{FailedCheck, SourceLocation};

    let check_id = json
        .get("check_id")
        .or_else(|| json.get("id"))
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let description = json
        .get("description")
        .or_else(|| json.get("message"))
        .and_then(|v| v.as_str())
        .unwrap_or("Unknown check failure")
        .to_string();

    let location = json.get("location").and_then(|loc| {
        let file = loc.get("file").and_then(|v| v.as_str())?;
        let line = loc.get("line").and_then(|v| v.as_u64())? as u32;
        let column = loc.get("column").and_then(|v| v.as_u64()).map(|c| c as u32);
        Some(SourceLocation {
            file: file.to_string(),
            line,
            column,
        })
    });

    let function = json
        .get("function")
        .and_then(|v| v.as_str())
        .map(String::from);

    Some(FailedCheck {
        check_id,
        description,
        location,
        function,
    })
}

fn parse_text_output(output: &KaniFastOutput) -> BackendResult {
    let combined = format!("{}\n{}", output.stdout, output.stderr);

    // Check for success indicators
    if combined.contains("VERIFIED")
        || combined.contains("All checks passed")
        || combined.contains("Proven")
    {
        return BackendResult {
            backend: BackendId::KaniFast,
            status: VerificationStatus::Proven,
            proof: Some("All Kani Fast checks passed".to_string()),
            counterexample: None,
            diagnostics: vec![],
            time_taken: output.duration,
        };
    }

    // Check for failure indicators
    if combined.contains("FAILED")
        || combined.contains("VERIFICATION FAILED")
        || combined.contains("counterexample")
    {
        let ce = StructuredCounterexample {
            raw: Some(output.stdout.clone()),
            ..Default::default()
        };

        return BackendResult {
            backend: BackendId::KaniFast,
            status: VerificationStatus::Disproven,
            proof: None,
            counterexample: Some(ce),
            diagnostics: vec![],
            time_taken: output.duration,
        };
    }

    // Check for timeout
    if combined.contains("timeout") || combined.contains("TIMEOUT") {
        return BackendResult {
            backend: BackendId::KaniFast,
            status: VerificationStatus::Unknown {
                reason: "Verification timed out".to_string(),
            },
            proof: None,
            counterexample: None,
            diagnostics: vec!["Timeout during verification".to_string()],
            time_taken: output.duration,
        };
    }

    // Check for errors
    if output.exit_code != Some(0) || combined.contains("error") {
        let reason = if !output.stderr.is_empty() {
            output
                .stderr
                .lines()
                .next()
                .unwrap_or("Unknown error")
                .to_string()
        } else {
            "Verification failed with unknown error".to_string()
        };

        return BackendResult {
            backend: BackendId::KaniFast,
            status: VerificationStatus::Unknown { reason },
            proof: None,
            counterexample: None,
            diagnostics: output.stderr.lines().map(String::from).collect(),
            time_taken: output.duration,
        };
    }

    // Unknown state
    BackendResult {
        backend: BackendId::KaniFast,
        status: VerificationStatus::Unknown {
            reason: "Could not determine verification result".to_string(),
        },
        proof: None,
        counterexample: None,
        diagnostics: vec![output.stdout.clone()],
        time_taken: output.duration,
    }
}
