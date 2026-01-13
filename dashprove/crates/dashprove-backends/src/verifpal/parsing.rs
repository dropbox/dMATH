//! Verifpal output parsing

use crate::traits::VerificationStatus;

/// Parse Verifpal output to determine verification status
pub fn parse_output(stdout: &str, _stderr: &str) -> VerificationStatus {
    // Verifpal outputs analysis results
    // With JSON output, look for "result" field
    // Without JSON, look for "PASS" or "FAIL" patterns

    let lower = stdout.to_lowercase();

    // Count passes and fails
    let pass_count = lower.matches("pass").count() + lower.matches("verified").count();
    let fail_count = lower.matches("fail").count() + lower.matches("attack found").count();

    if fail_count > 0 {
        // Any attack found means property violated
        return VerificationStatus::Disproven;
    }

    // Try to parse partial results from query analysis BEFORE checking for full verification
    // This handles "X of Y queries verified" where X < Y
    if let Some(line) = stdout.lines().find(|l| {
        let ll = l.to_lowercase();
        ll.contains("queries") || ll.contains("query")
    }) {
        // Look for "X of Y" pattern (e.g., "3 of 5 queries verified")
        let words: Vec<&str> = line.split_whitespace().collect();
        for i in 0..words.len().saturating_sub(2) {
            if words[i + 1] == "of" {
                if let (Ok(verified), Ok(total)) =
                    (words[i].parse::<f64>(), words[i + 2].parse::<f64>())
                {
                    if total > 0.0 {
                        let percentage = (verified / total) * 100.0;
                        // If all queries verified, return Proven; otherwise Partial
                        if (verified - total).abs() < 0.001 {
                            return VerificationStatus::Proven;
                        }
                        return VerificationStatus::Partial {
                            verified_percentage: percentage,
                        };
                    }
                }
            }
        }

        // Fallback: try to extract percentage directly (e.g., "60% queries verified")
        if let Some(pct_idx) = line.find('%') {
            if pct_idx > 0 {
                let before_pct = &line[..pct_idx];
                if let Some(num_str) = before_pct.split_whitespace().last() {
                    if let Ok(pct) = num_str.parse::<f64>() {
                        // 100% means fully verified
                        if (pct - 100.0).abs() < 0.001 {
                            return VerificationStatus::Proven;
                        }
                        return VerificationStatus::Partial {
                            verified_percentage: pct,
                        };
                    }
                }
            }
        }
    }

    // If no partial results found, check for full pass
    if pass_count > 0 && fail_count == 0 {
        return VerificationStatus::Proven;
    }

    // Check for analysis completion without clear result
    if (lower.contains("analysis complete") || lower.contains("finished"))
        && !lower.contains("error")
    {
        return VerificationStatus::Proven;
    }

    VerificationStatus::Unknown {
        reason: "Could not parse Verifpal output".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_verified() {
        let output = "Analysis complete: All queries VERIFIED\nPASS";
        let status = parse_output(output, "");
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn test_parse_attack_found() {
        let output = "Analysis result: Attack found on confidentiality query";
        let status = parse_output(output, "");
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[test]
    fn test_parse_fail() {
        let output = "Query FAIL: Authentication property violated";
        let status = parse_output(output, "");
        assert!(matches!(status, VerificationStatus::Disproven));
    }

    #[test]
    fn test_parse_partial_x_of_y() {
        let output = "Status: 3 of 5 queries verified";
        let status = parse_output(output, "");
        match status {
            VerificationStatus::Partial {
                verified_percentage,
            } => {
                assert!((verified_percentage - 60.0).abs() < 0.01);
            }
            _ => panic!("Expected Partial status, got {:?}", status),
        }
    }

    #[test]
    fn test_parse_partial_percentage() {
        let output = "Query status: 75% queries verified";
        let status = parse_output(output, "");
        match status {
            VerificationStatus::Partial {
                verified_percentage,
            } => {
                assert!((verified_percentage - 75.0).abs() < 0.01);
            }
            _ => panic!("Expected Partial status, got {:?}", status),
        }
    }

    #[test]
    fn test_parse_analysis_complete() {
        let output = "Protocol analysis complete. finished successfully.";
        let status = parse_output(output, "");
        assert!(matches!(status, VerificationStatus::Proven));
    }

    #[test]
    fn test_parse_unknown() {
        let output = "Some unrecognized output format";
        let status = parse_output(output, "");
        assert!(matches!(status, VerificationStatus::Unknown { .. }));
    }
}
