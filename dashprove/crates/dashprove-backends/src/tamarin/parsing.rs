//! Tamarin output parsing

use crate::traits::VerificationStatus;

/// Parse Tamarin output to determine verification status
pub fn parse_output(stdout: &str, stderr: &str) -> VerificationStatus {
    let combined = format!("{}\n{}", stdout, stderr);

    if combined.contains("verified") || combined.contains("analysis complete") {
        // Check for individual lemma status
        let proven = combined.matches("verified").count();
        let total = combined.matches("analyzing lemma").count();

        if total > 0 && proven == total {
            VerificationStatus::Proven
        } else if proven > 0 {
            VerificationStatus::Partial {
                verified_percentage: (proven as f64 / total as f64) * 100.0,
            }
        } else {
            VerificationStatus::Disproven
        }
    } else if combined.contains("attack found") || combined.contains("falsified") {
        VerificationStatus::Disproven
    } else if combined.contains("timeout") {
        VerificationStatus::Unknown {
            reason: "Verification timed out".to_string(),
        }
    } else {
        VerificationStatus::Unknown {
            reason: "Could not parse Tamarin output".to_string(),
        }
    }
}
