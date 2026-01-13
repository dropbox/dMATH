//! Kani output parsing
//!
//! Functions for parsing cargo kani output into structured verification results.

use super::config::KaniOutput;
use crate::traits::{
    BackendId, BackendResult, CounterexampleValue, FailedCheck, SourceLocation,
    StructuredCounterexample, VerificationStatus,
};
use regex::Regex;
use std::collections::HashMap;

// =============================================
// Kani Proofs for Parsing Functions
// =============================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify parse_byte_array returns empty vec for empty string
    #[kani::proof]
    fn proof_parse_byte_array_empty_string() {
        let result = parse_byte_array("");
        kani::assert(
            result == Some(Vec::new()),
            "Empty string should return empty vec",
        );
    }

    /// Verify parse_byte_array returns empty vec for whitespace-only string
    #[kani::proof]
    fn proof_parse_byte_array_whitespace() {
        let result = parse_byte_array("   ");
        kani::assert(
            result == Some(Vec::new()),
            "Whitespace should return empty vec",
        );
    }

    /// Verify parse_byte_array parses single byte
    #[kani::proof]
    fn proof_parse_byte_array_single_byte() {
        let result = parse_byte_array("42");
        kani::assert(
            result == Some(vec![42u8]),
            "Single byte should parse correctly",
        );
    }

    /// Verify parse_byte_array parses multiple bytes
    #[kani::proof]
    fn proof_parse_byte_array_multiple_bytes() {
        let result = parse_byte_array("1, 2, 3");
        kani::assert(
            result == Some(vec![1u8, 2u8, 3u8]),
            "Multiple bytes should parse",
        );
    }

    /// Verify parse_byte_array rejects values > 255
    #[kani::proof]
    fn proof_parse_byte_array_rejects_large_values() {
        let result = parse_byte_array("256");
        kani::assert(result.is_none(), "Values > 255 should fail");
    }

    /// Verify parse_byte_array rejects negative values
    #[kani::proof]
    fn proof_parse_byte_array_rejects_negative() {
        let result = parse_byte_array("-1");
        kani::assert(result.is_none(), "Negative values should fail");
    }

    /// Verify parse_counterexample_value parses unsigned int
    #[kani::proof]
    fn proof_parse_counterexample_value_uint() {
        let result = parse_counterexample_value("42", None);
        if let CounterexampleValue::UInt { value, .. } = result {
            kani::assert(value == 42, "Should parse as UInt(42)");
        } else {
            kani::assert(false, "Should be UInt variant");
        }
    }

    /// Verify parse_counterexample_value parses signed int
    #[kani::proof]
    fn proof_parse_counterexample_value_negative_int() {
        let result = parse_counterexample_value("-42", None);
        if let CounterexampleValue::Int { value, .. } = result {
            kani::assert(value == -42, "Should parse as Int(-42)");
        } else {
            kani::assert(false, "Should be Int variant");
        }
    }

    /// Verify parse_counterexample_value parses bool true
    #[kani::proof]
    fn proof_parse_counterexample_value_bool_true() {
        let result = parse_counterexample_value("true", None);
        kani::assert(
            result == CounterexampleValue::Bool(true),
            "Should parse as Bool(true)",
        );
    }

    /// Verify parse_counterexample_value parses bool false
    #[kani::proof]
    fn proof_parse_counterexample_value_bool_false() {
        let result = parse_counterexample_value("false", None);
        kani::assert(
            result == CounterexampleValue::Bool(false),
            "Should parse as Bool(false)",
        );
    }

    /// Verify parse_counterexample_value is case-insensitive for booleans
    #[kani::proof]
    fn proof_parse_counterexample_value_bool_case_insensitive() {
        let result = parse_counterexample_value("TRUE", None);
        kani::assert(
            result == CounterexampleValue::Bool(true),
            "TRUE should parse as Bool(true)",
        );
    }

    /// Verify parse_counterexample_value preserves type hint
    #[kani::proof]
    fn proof_parse_counterexample_value_type_hint() {
        let hint = Some("u32".to_string());
        let result = parse_counterexample_value("42", hint.clone());
        if let CounterexampleValue::UInt { type_hint, .. } = result {
            kani::assert(type_hint == hint, "Type hint should be preserved");
        } else {
            kani::assert(false, "Should be UInt variant");
        }
    }

    /// Verify parse_counterexample_value returns Unknown for non-parseable values
    #[kani::proof]
    fn proof_parse_counterexample_value_unknown() {
        let result = parse_counterexample_value("not_a_number", None);
        if let CounterexampleValue::Unknown(s) = result {
            kani::assert(s == "not_a_number", "Should preserve original string");
        } else {
            kani::assert(false, "Should be Unknown variant");
        }
    }

    /// Verify parse_counterexample_value parses float
    #[kani::proof]
    fn proof_parse_counterexample_value_float() {
        let result = parse_counterexample_value("3.14", None);
        if let CounterexampleValue::Float { value } = result {
            // Approximate comparison for float
            kani::assert(value > 3.13 && value < 3.15, "Should parse as Float(~3.14)");
        } else {
            kani::assert(false, "Should be Float variant");
        }
    }

    /// Verify parse_counterexample_value parses zero
    #[kani::proof]
    fn proof_parse_counterexample_value_zero() {
        let result = parse_counterexample_value("0", None);
        if let CounterexampleValue::UInt { value, .. } = result {
            kani::assert(value == 0, "Should parse as UInt(0)");
        } else {
            kani::assert(false, "Should be UInt variant");
        }
    }

    /// Verify extract_diagnostics returns empty vec for empty string
    #[kani::proof]
    fn proof_extract_diagnostics_empty() {
        let diagnostics = extract_diagnostics("");
        kani::assert(
            diagnostics.is_empty(),
            "Empty string should produce empty diagnostics",
        );
    }

    /// Verify extract_diagnostics extracts failed check count
    #[kani::proof]
    fn proof_extract_diagnostics_with_failure() {
        let diagnostics = extract_diagnostics("**  2 of 5 failed");
        kani::assert(!diagnostics.is_empty(), "Should extract failure count");
        kani::assert(
            diagnostics.iter().any(|d| d.contains("2 of 5")),
            "Should contain failure info",
        );
    }

    /// Verify failed_check_count returns None for no match
    #[kani::proof]
    fn proof_failed_check_count_no_match() {
        let result = failed_check_count("no matches here");
        kani::assert(result.is_none(), "Should return None when no match");
    }

    /// Verify failed_check_count parses correctly
    #[kani::proof]
    fn proof_failed_check_count_valid() {
        let result = failed_check_count("**  3 of 10 failed");
        if let Some((failed, total)) = result {
            kani::assert(failed == 3, "Failed count should be 3");
            kani::assert(total == 10, "Total count should be 10");
        } else {
            kani::assert(false, "Should parse check count");
        }
    }

    /// Verify failed_check_count handles zero failures
    #[kani::proof]
    fn proof_failed_check_count_zero() {
        let result = failed_check_count("**  0 of 5 failed");
        if let Some((failed, total)) = result {
            kani::assert(failed == 0, "Failed count should be 0");
            kani::assert(total == 5, "Total count should be 5");
        } else {
            kani::assert(false, "Should parse check count");
        }
    }

    /// Verify extract_playback_test returns None for no playback
    #[kani::proof]
    fn proof_extract_playback_test_none() {
        let result = extract_playback_test("no playback here");
        kani::assert(result.is_none(), "Should return None when no playback test");
    }

    /// Verify extract_raw_counterexample returns None for clean output
    #[kani::proof]
    fn proof_extract_raw_counterexample_none() {
        let result = extract_raw_counterexample("VERIFICATION:- SUCCESSFUL");
        kani::assert(
            result.is_none(),
            "Should return None for successful verification",
        );
    }

    /// Verify extract_structured_counterexample creates valid structure
    #[kani::proof]
    fn proof_extract_structured_counterexample_empty() {
        let ce = extract_structured_counterexample("");
        kani::assert(
            ce.witness.is_empty(),
            "Empty input should produce empty witness",
        );
        kani::assert(
            ce.failed_checks.is_empty(),
            "Empty input should produce no failed checks",
        );
    }
}

/// Parse cargo kani output into a structured backend result
pub fn parse_output(output: &KaniOutput) -> BackendResult {
    let combined = format!("{}\n{}", output.stdout, output.stderr);
    let diagnostics = extract_diagnostics(&combined);

    // Success detection
    if combined.contains("VERIFICATION:- SUCCESSFUL")
        || failed_check_count(&combined)
            .map(|(failed, _)| failed == 0)
            .unwrap_or(false)
    {
        return BackendResult {
            backend: BackendId::Kani,
            status: VerificationStatus::Proven,
            proof: Some("All Kani checks passed".to_string()),
            counterexample: None,
            diagnostics,
            time_taken: output.duration,
        };
    }

    // Failure detection
    if combined.contains("VERIFICATION:- FAILED")
        || combined.contains("Status: FAILURE")
        || failed_check_count(&combined)
            .map(|(failed, _)| failed > 0)
            .unwrap_or(false)
    {
        let counterexample = extract_structured_counterexample(&combined);
        return BackendResult {
            backend: BackendId::Kani,
            status: VerificationStatus::Disproven,
            proof: None,
            counterexample: Some(counterexample),
            diagnostics,
            time_taken: output.duration,
        };
    }

    // Unknown result (errors, setup issues, or timeouts)
    let reason = match output.exit_code {
        Some(2) => "Kani reported a CLI error".to_string(),
        Some(code) => format!("Kani exited with code {} without a definitive result", code),
        None => "Could not determine verification result".to_string(),
    };

    BackendResult {
        backend: BackendId::Kani,
        status: VerificationStatus::Unknown { reason },
        proof: None,
        counterexample: None,
        diagnostics,
        time_taken: output.duration,
    }
}

/// Extract the count of failed checks from output
pub fn failed_check_count(output: &str) -> Option<(usize, usize)> {
    let re = Regex::new(r"\*\*\s+(\d+)\s+of\s+(\d+)\s+failed").ok()?;
    let caps = re.captures(output)?;
    let failed = caps.get(1)?.as_str().parse().ok()?;
    let total = caps.get(2)?.as_str().parse().ok()?;
    Some((failed, total))
}

/// Extract a structured counterexample with parsed values and failed checks
pub fn extract_structured_counterexample(output: &str) -> StructuredCounterexample {
    let mut ce = StructuredCounterexample::new();

    // Extract failed checks from RESULTS section
    ce.failed_checks = extract_failed_checks(output);

    // Extract concrete playback test code
    ce.playback_test = extract_playback_test(output);

    // Extract witness values from "Counterexample values:" section
    ce.witness = extract_witness_values(output);

    // Store raw output for backwards compatibility
    ce.raw = extract_raw_counterexample(output);

    ce
}

/// Extract failed check details from Kani output
pub fn extract_failed_checks(output: &str) -> Vec<FailedCheck> {
    let mut checks = Vec::new();

    // Pattern: Check N: check_id
    //          - Status: FAILURE
    //          - Description: "..."
    //          - Location: file:line:col in function fn_name
    let check_re = Regex::new(
        r#"Check \d+:\s*(\S+)\s*\n\s+-\s+Status:\s*FAILURE\s*\n\s+-\s+Description:\s*"([^"]+)"\s*\n\s+-\s+Location:\s*([^:]+):(\d+):(\d+)\s+in function\s+(\S+)"#
    ).ok();

    if let Some(re) = check_re {
        for caps in re.captures_iter(output) {
            let check_id = caps
                .get(1)
                .map(|m| m.as_str().to_string())
                .unwrap_or_default();
            let description = caps
                .get(2)
                .map(|m| m.as_str().to_string())
                .unwrap_or_default();
            let file = caps
                .get(3)
                .map(|m| m.as_str().to_string())
                .unwrap_or_default();
            let line: u32 = caps
                .get(4)
                .and_then(|m| m.as_str().parse().ok())
                .unwrap_or(0);
            let column: u32 = caps
                .get(5)
                .and_then(|m| m.as_str().parse().ok())
                .unwrap_or(0);
            let function = caps.get(6).map(|m| m.as_str().to_string());

            checks.push(FailedCheck {
                check_id,
                description,
                location: Some(SourceLocation {
                    file,
                    line,
                    column: Some(column),
                }),
                function,
            });
        }
    }

    // Fallback: extract from "Failed Checks:" section
    if checks.is_empty() {
        if let Some(start) = output.find("Failed Checks:") {
            let section = &output[start..];
            // Pattern: Failed Checks: description
            //          File: "file", line N, in function
            if let Some(desc_end) = section.find('\n') {
                let description = section["Failed Checks:".len()..desc_end].trim().to_string();

                // Try to find location info
                let location_re =
                    Regex::new(r#"File:\s*"([^"]+)",\s*line\s*(\d+),\s*in\s+(\S+)"#).ok();
                let (location, function) = if let Some(re) = location_re {
                    if let Some(caps) = re.captures(section) {
                        let file = caps
                            .get(1)
                            .map(|m| m.as_str().to_string())
                            .unwrap_or_default();
                        let line: u32 = caps
                            .get(2)
                            .and_then(|m| m.as_str().parse().ok())
                            .unwrap_or(0);
                        let func = caps.get(3).map(|m| m.as_str().to_string());
                        (
                            Some(SourceLocation {
                                file,
                                line,
                                column: None,
                            }),
                            func,
                        )
                    } else {
                        (None, None)
                    }
                } else {
                    (None, None)
                };

                if !description.is_empty() {
                    checks.push(FailedCheck {
                        check_id: String::new(),
                        description,
                        location,
                        function,
                    });
                }
            }
        }
    }

    checks
}

/// Extract concrete playback test code
pub fn extract_playback_test(output: &str) -> Option<String> {
    if let Some(start) = output.find("Concrete playback unit test") {
        // Find the code block
        if let Some(code_start) = output[start..].find("```") {
            let after_marker = start + code_start + 3;
            if let Some(code_end) = output[after_marker..].find("```") {
                return Some(
                    output[after_marker..after_marker + code_end]
                        .trim()
                        .to_string(),
                );
            }
        }
        // Fallback: take everything from "Concrete playback" to end or next section
        let end = output[start..]
            .find("Counterexample values:")
            .unwrap_or(output.len() - start);
        return Some(output[start..start + end].trim().to_string());
    }
    None
}

/// Extract witness values from Counterexample values section
pub fn extract_witness_values(output: &str) -> HashMap<String, CounterexampleValue> {
    let mut witness = HashMap::new();

    if let Some(start) = output.find("Counterexample values:") {
        let section = &output[start + "Counterexample values:".len()..];

        // Pattern: - var = value (type_info)
        // e.g., "- a = 4294967295 (u32::MAX)"
        // e.g., "- b = 0"
        let value_re = Regex::new(r"-\s*(\w+)\s*=\s*(\S+)(?:\s*\(([^)]+)\))?").ok();

        if let Some(re) = value_re {
            for caps in re.captures_iter(section) {
                let var_name = caps
                    .get(1)
                    .map(|m| m.as_str().to_string())
                    .unwrap_or_default();
                let value_str = caps.get(2).map(|m| m.as_str()).unwrap_or("");
                let type_hint = caps.get(3).map(|m| m.as_str().to_string());

                let value = parse_counterexample_value(value_str, type_hint);
                if !var_name.is_empty() {
                    witness.insert(var_name, value);
                }
            }
        }
    }

    // Also try to extract from concrete_vals in playback test
    if witness.is_empty() {
        extract_concrete_vals_witness(output, &mut witness);
    }

    // Extract raw byte arrays for debugging (stored as bytes_arg0, bytes_arg1, etc.)
    extract_raw_byte_arrays(output, &mut witness);

    witness
}

/// Extract witness values from concrete_vals section when no explicit values are present
fn extract_concrete_vals_witness(output: &str, witness: &mut HashMap<String, CounterexampleValue>) {
    if let Some(start) = output.find("concrete_vals:") {
        let section = &output[start..];
        // Look for comments like "// 4294967295" followed by vec![...]
        let comment_re = Regex::new(r"//\s*(-?\d+)\s*\n\s*vec!\[([^\]]+)\]").ok();
        if let Some(re) = comment_re {
            for (i, caps) in re.captures_iter(section).enumerate() {
                let value_str = caps.get(1).map(|m| m.as_str()).unwrap_or("");
                let value = parse_counterexample_value(value_str, None);
                witness.insert(format!("arg{}", i), value);
            }
        }
    }
}

/// Extract raw byte arrays from vec![...] literals for low-level debugging
fn extract_raw_byte_arrays(output: &str, witness: &mut HashMap<String, CounterexampleValue>) {
    if let Some(start) = output.find("concrete_vals:") {
        let section = &output[start..];
        // Pattern: inner vec![n1, n2, n3, ...] containing only numeric values
        // Skip the outer vec![...] which contains the inner vecs
        // Look for vec![<digits and commas only>] to match inner byte arrays
        let vec_re = Regex::new(r"vec!\[([\d,\s]+)\]").ok();
        if let Some(re) = vec_re {
            let mut arg_idx = 0;
            for caps in re.captures_iter(section) {
                let bytes_str = caps.get(1).map(|m| m.as_str()).unwrap_or("");
                // Only include if it looks like a byte array (has at least one digit)
                if bytes_str.chars().any(|c| c.is_ascii_digit()) {
                    if let Some(bytes) = parse_byte_array(bytes_str) {
                        witness.insert(
                            format!("bytes_arg{}", arg_idx),
                            CounterexampleValue::Bytes(bytes),
                        );
                        arg_idx += 1;
                    }
                }
            }
        }
    }
}

/// Parse a comma-separated byte array from vec![...] content
pub fn parse_byte_array(bytes_str: &str) -> Option<Vec<u8>> {
    if bytes_str.trim().is_empty() {
        return Some(Vec::new());
    }

    let bytes: Result<Vec<u8>, _> = bytes_str
        .split(',')
        .map(|s| s.trim().parse::<u8>())
        .collect();

    bytes.ok()
}

/// Parse a string value into a CounterexampleValue
pub fn parse_counterexample_value(
    value_str: &str,
    type_hint: Option<String>,
) -> CounterexampleValue {
    // Try parsing as unsigned integer
    if let Ok(v) = value_str.parse::<u128>() {
        return CounterexampleValue::UInt {
            value: v,
            type_hint,
        };
    }

    // Try parsing as signed integer
    if let Ok(v) = value_str.parse::<i128>() {
        return CounterexampleValue::Int {
            value: v,
            type_hint,
        };
    }

    // Try parsing as float
    if let Ok(v) = value_str.parse::<f64>() {
        return CounterexampleValue::Float { value: v };
    }

    // Try parsing as boolean
    match value_str.to_lowercase().as_str() {
        "true" => return CounterexampleValue::Bool(true),
        "false" => return CounterexampleValue::Bool(false),
        _ => {}
    }

    // Return as unknown string
    CounterexampleValue::Unknown(value_str.to_string())
}

/// Extract raw counterexample text for backwards compatibility
pub fn extract_raw_counterexample(output: &str) -> Option<String> {
    if let Some(start) = output.find("Concrete playback unit test") {
        if let Some(end) = output[start..].find("Counterexample values:") {
            let snippet = &output[start..start + end];
            if !snippet.trim().is_empty() {
                return Some(snippet.trim().to_string());
            }
        } else {
            return Some(output[start..].trim().to_string());
        }
    }

    if let Some(start) = output.find("Counterexample values:") {
        return Some(output[start..].trim().to_string());
    }

    if let Some(start) = output.find("Failed Checks:") {
        let section: Vec<&str> = output[start..].lines().take(5).collect();
        return Some(section.join("\n"));
    }

    for line in output.lines() {
        if line.contains("Status: FAILURE") || line.contains("assertion") {
            return Some(line.trim().to_string());
        }
    }

    None
}

/// Extract diagnostic messages from output
pub fn extract_diagnostics(output: &str) -> Vec<String> {
    let mut diagnostics = Vec::new();

    if let Some((failed, total)) = failed_check_count(output) {
        diagnostics.push(format!("{} of {} checks failed", failed, total));
    }

    for line in output.lines() {
        if line.contains("Failed Checks:") || line.contains("Location:") {
            diagnostics.push(line.trim().to_string());
        }
    }

    diagnostics
}
