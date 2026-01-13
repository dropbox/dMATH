//! PRISM output parsing

use crate::traits::VerificationStatus;

/// Parse PRISM output to determine verification status
pub fn parse_output(stdout: &str, _stderr: &str) -> VerificationStatus {
    // PRISM outputs results in various formats
    // "Result: true" for boolean queries
    // "Result: 0.95" for quantitative queries
    // Property satisfied/not satisfied

    // Check for explicit satisfaction results first
    if stdout.contains("Property satisfied") {
        return VerificationStatus::Proven;
    }
    if stdout.contains("Property NOT satisfied") {
        return VerificationStatus::Disproven;
    }

    // Parse numerical results
    if let Some(line) = stdout.lines().find(|l| l.contains("Result:")) {
        let result_part = line.split("Result:").nth(1).unwrap_or("").trim();

        if result_part.eq_ignore_ascii_case("true") {
            return VerificationStatus::Proven;
        }
        if result_part.eq_ignore_ascii_case("false") {
            return VerificationStatus::Disproven;
        }

        // Try parsing as probability
        if let Ok(prob) = result_part.parse::<f64>() {
            // For probability queries, return as partial result
            return VerificationStatus::Partial {
                verified_percentage: prob * 100.0,
            };
        }
    }

    // Check for errors
    if stdout.contains("Error") || stdout.contains("error") {
        return VerificationStatus::Unknown {
            reason: stdout
                .lines()
                .find(|l| l.contains("Error") || l.contains("error"))
                .unwrap_or("Unknown error")
                .to_string(),
        };
    }

    VerificationStatus::Unknown {
        reason: "Could not parse PRISM output".to_string(),
    }
}
