//! TLC output parsing

use crate::traits::{BackendId, BackendResult, VerificationStatus};

use super::execution::TlcOutput;
use super::trace::parse_trace;
use super::util::{extract_coverage, extract_diagnostics, extract_error};

/// Parse TLC output into verification result
pub fn parse_output(output: &TlcOutput) -> BackendResult {
    let combined = format!("{}\n{}", output.stdout, output.stderr);

    // Check for successful verification
    if combined.contains("Model checking completed. No error has been found.") {
        return BackendResult {
            backend: BackendId::TlaPlus,
            status: VerificationStatus::Proven,
            proof: Some("Model checking completed successfully".to_string()),
            counterexample: None,
            diagnostics: extract_diagnostics(&combined),
            time_taken: output.duration,
        };
    }

    // Check for invariant violations
    if combined.contains("Invariant") && combined.contains("is violated") {
        let counterexample = parse_trace(&combined);
        return BackendResult {
            backend: BackendId::TlaPlus,
            status: VerificationStatus::Disproven,
            proof: None,
            counterexample: Some(counterexample),
            diagnostics: extract_diagnostics(&combined),
            time_taken: output.duration,
        };
    }

    // Check for property violations
    if combined.contains("Error: Temporal properties were violated.")
        || combined.contains("is violated.")
    {
        let counterexample = parse_trace(&combined);
        return BackendResult {
            backend: BackendId::TlaPlus,
            status: VerificationStatus::Disproven,
            proof: None,
            counterexample: Some(counterexample),
            diagnostics: extract_diagnostics(&combined),
            time_taken: output.duration,
        };
    }

    // Check for deadlock
    if combined.contains("Deadlock reached") {
        let mut counterexample = parse_trace(&combined);
        counterexample
            .failed_checks
            .push(crate::traits::FailedCheck {
                check_id: "deadlock".to_string(),
                description: "Deadlock detected".to_string(),
                location: None,
                function: None,
            });
        return BackendResult {
            backend: BackendId::TlaPlus,
            status: VerificationStatus::Disproven,
            proof: None,
            counterexample: Some(counterexample),
            diagnostics: extract_diagnostics(&combined),
            time_taken: output.duration,
        };
    }

    // Check for parsing/semantic errors
    // Real TLC output uses "***Parse Error***" and "Parsing or semantic analysis failed"
    if combined.contains("Semantic error")
        || combined.contains("***Parse Error***")
        || combined.contains("Parsing or semantic analysis failed")
    {
        let error_msg = extract_error(&combined);
        return BackendResult {
            backend: BackendId::TlaPlus,
            status: VerificationStatus::Unknown {
                reason: format!("TLC error: {}", error_msg),
            },
            proof: None,
            counterexample: None,
            diagnostics: vec![error_msg],
            time_taken: output.duration,
        };
    }

    // Check for state space issues
    if combined.contains("too many states") || combined.contains("Out of memory") {
        return BackendResult {
            backend: BackendId::TlaPlus,
            status: VerificationStatus::Unknown {
                reason: "State space too large".to_string(),
            },
            proof: None,
            counterexample: None,
            diagnostics: extract_diagnostics(&combined),
            time_taken: output.duration,
        };
    }

    // Partial results
    if let Some(percentage) = extract_coverage(&combined) {
        return BackendResult {
            backend: BackendId::TlaPlus,
            status: VerificationStatus::Partial {
                verified_percentage: percentage,
            },
            proof: None,
            counterexample: None,
            diagnostics: extract_diagnostics(&combined),
            time_taken: output.duration,
        };
    }

    // Unknown result
    BackendResult {
        backend: BackendId::TlaPlus,
        status: VerificationStatus::Unknown {
            reason: "Could not determine verification result".to_string(),
        },
        proof: None,
        counterexample: None,
        diagnostics: vec![combined],
        time_taken: output.duration,
    }
}
