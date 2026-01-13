//! Storm output parsing

use crate::traits::VerificationStatus;

/// Parse Storm output to determine verification status
pub fn parse_output(stdout: &str, _stderr: &str) -> VerificationStatus {
    // Storm outputs results like "Result: true" or "Result: 0.95"
    if let Some(line) = stdout.lines().find(|l| l.contains("Result:")) {
        if line.contains("true") {
            return VerificationStatus::Proven;
        }
        if line.contains("false") {
            return VerificationStatus::Disproven;
        }
        // Parse probability value
        if let Some(prob) = line.split_whitespace().last() {
            if let Ok(p) = prob.parse::<f64>() {
                return VerificationStatus::Partial {
                    verified_percentage: p * 100.0,
                };
            }
        }
    }

    VerificationStatus::Unknown {
        reason: "Could not parse Storm output".to_string(),
    }
}
