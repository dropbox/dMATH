//! Kani output parsing
//!
//! Functions for parsing cargo kani output into structured verification results.

use crate::types::{CounterexampleValue, FailedCheck, SourceLocation, StructuredCounterexample};
use lazy_static::lazy_static;
use regex::Regex;
use std::collections::HashMap;

// Pre-compiled regexes for Kani output parsing (compiled once, reused)
lazy_static! {
    /// Matches: ** N of M failed
    static ref RE_FAILED_COUNT: Regex = Regex::new(r"\*\*\s+(\d+)\s+of\s+(\d+)\s+failed").unwrap();
    /// Matches failed check details
    static ref RE_CHECK_DETAIL: Regex = Regex::new(
        r#"Check \d+:\s*(\S+)\s*\n\s+-\s+Status:\s*FAILURE\s*\n\s+-\s+Description:\s*"([^"]+)"\s*\n\s+-\s+Location:\s*([^:]+):(\d+):(\d+)\s+in function\s+(\S+)"#
    ).unwrap();
    /// Matches: File: "...", line N, in function_name
    static ref RE_FILE_LOCATION: Regex = Regex::new(
        r#"File:\s*"([^"]+)",\s*line\s*(\d+),\s*in\s+(\S+)"#
    ).unwrap();
    /// Matches: - var_name = value (type_hint)
    static ref RE_WITNESS_VALUE: Regex = Regex::new(r"-\s*(\w+)\s*=\s*(\S+)(?:\s*\(([^)]+)\))?").unwrap();
    /// Matches: // value \n vec![...]
    static ref RE_CONCRETE_VALS: Regex = Regex::new(r"//\s*(-?\d+)\s*\n\s*vec!\[([^\]]+)\]").unwrap();
    /// Matches: vec![byte, byte, ...]
    static ref RE_VEC_BYTES: Regex = Regex::new(r"vec!\[([\d,\s]+)\]").unwrap();
}

/// Extract the count of failed checks from output
pub fn failed_check_count(output: &str) -> Option<(usize, usize)> {
    let caps = RE_FAILED_COUNT.captures(output)?;
    let failed = caps.get(1)?.as_str().parse().ok()?;
    let total = caps.get(2)?.as_str().parse().ok()?;
    Some((failed, total))
}

/// Extract a structured counterexample with parsed values and failed checks
pub fn extract_structured_counterexample(output: &str) -> StructuredCounterexample {
    let mut ce = StructuredCounterexample::new();

    ce.failed_checks = extract_failed_checks(output);
    ce.playback_test = extract_playback_test(output);
    ce.witness = extract_witness_values(output);
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
    for caps in RE_CHECK_DETAIL.captures_iter(output) {
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

    // Fallback: extract from "Failed Checks:" section
    if checks.is_empty() {
        if let Some(start) = output.find("Failed Checks:") {
            let section = &output[start..];
            if let Some(desc_end) = section.find('\n') {
                let description = section["Failed Checks:".len()..desc_end].trim().to_string();

                let (location, function) = if let Some(caps) = RE_FILE_LOCATION.captures(section) {
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

        for caps in RE_WITNESS_VALUE.captures_iter(section) {
            let var_name = caps
                .get(1)
                .map(|m| m.as_str().to_string())
                .unwrap_or_default();
            let value_str = caps.get(2).map_or("", |m| m.as_str());
            let type_hint = caps.get(3).map(|m| m.as_str().to_string());

            let value = parse_counterexample_value(value_str, type_hint);
            if !var_name.is_empty() {
                witness.insert(var_name, value);
            }
        }
    }

    if witness.is_empty() {
        extract_concrete_vals_witness(output, &mut witness);
    }

    extract_raw_byte_arrays(output, &mut witness);

    witness
}

/// Extract witness values from concrete_vals section
fn extract_concrete_vals_witness(output: &str, witness: &mut HashMap<String, CounterexampleValue>) {
    if let Some(start) = output.find("concrete_vals:") {
        let section = &output[start..];
        for (i, caps) in RE_CONCRETE_VALS.captures_iter(section).enumerate() {
            let value_str = caps.get(1).map_or("", |m| m.as_str());
            let value = parse_counterexample_value(value_str, None);
            witness.insert(format!("arg{i}"), value);
        }
    }
}

/// Extract raw byte arrays from vec![...] literals
fn extract_raw_byte_arrays(output: &str, witness: &mut HashMap<String, CounterexampleValue>) {
    if let Some(start) = output.find("concrete_vals:") {
        let section = &output[start..];
        let mut arg_idx = 0;
        for caps in RE_VEC_BYTES.captures_iter(section) {
            let bytes_str = caps.get(1).map_or("", |m| m.as_str());
            if bytes_str.chars().any(|c| c.is_ascii_digit()) {
                if let Some(bytes) = parse_byte_array(bytes_str) {
                    witness.insert(
                        format!("bytes_arg{arg_idx}"),
                        CounterexampleValue::Bytes(bytes),
                    );
                    arg_idx += 1;
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
        diagnostics.push(format!("{failed} of {total} checks failed"));
    }

    for line in output.lines() {
        if line.contains("Failed Checks:") || line.contains("Location:") {
            diagnostics.push(line.trim().to_string());
        }
    }

    diagnostics
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== parse_byte_array tests ====================

    #[test]
    fn test_parse_byte_array_empty() {
        assert_eq!(parse_byte_array(""), Some(Vec::new()));
        assert_eq!(parse_byte_array("   "), Some(Vec::new()));
    }

    #[test]
    fn test_parse_byte_array_single() {
        assert_eq!(parse_byte_array("42"), Some(vec![42u8]));
    }

    #[test]
    fn test_parse_byte_array_multiple() {
        assert_eq!(parse_byte_array("1, 2, 3"), Some(vec![1u8, 2u8, 3u8]));
    }

    #[test]
    fn test_parse_byte_array_invalid() {
        assert_eq!(parse_byte_array("256"), None);
        assert_eq!(parse_byte_array("-1"), None);
    }

    #[test]
    fn test_parse_byte_array_with_whitespace() {
        assert_eq!(
            parse_byte_array("  1,   2  ,3  "),
            Some(vec![1u8, 2u8, 3u8])
        );
    }

    #[test]
    fn test_parse_byte_array_zeros() {
        assert_eq!(
            parse_byte_array("0, 0, 0, 0"),
            Some(vec![0u8, 0u8, 0u8, 0u8])
        );
    }

    #[test]
    fn test_parse_byte_array_max_values() {
        assert_eq!(parse_byte_array("255, 255"), Some(vec![255u8, 255u8]));
    }

    // ==================== parse_counterexample_value tests ====================

    #[test]
    fn test_parse_counterexample_value_uint() {
        match parse_counterexample_value("42", None) {
            CounterexampleValue::UInt { value, .. } => assert_eq!(value, 42),
            _ => panic!("Expected UInt"),
        }
    }

    #[test]
    fn test_parse_counterexample_value_uint_with_type_hint() {
        match parse_counterexample_value("42", Some("u32".to_string())) {
            CounterexampleValue::UInt { value, type_hint } => {
                assert_eq!(value, 42);
                assert_eq!(type_hint, Some("u32".to_string()));
            }
            _ => panic!("Expected UInt"),
        }
    }

    #[test]
    fn test_parse_counterexample_value_uint_large() {
        match parse_counterexample_value("340282366920938463463374607431768211455", None) {
            CounterexampleValue::UInt { value, .. } => assert_eq!(value, u128::MAX),
            _ => panic!("Expected UInt"),
        }
    }

    #[test]
    fn test_parse_counterexample_value_int() {
        match parse_counterexample_value("-42", None) {
            CounterexampleValue::Int { value, .. } => assert_eq!(value, -42),
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn test_parse_counterexample_value_int_with_type_hint() {
        match parse_counterexample_value("-42", Some("i64".to_string())) {
            CounterexampleValue::Int { value, type_hint } => {
                assert_eq!(value, -42);
                assert_eq!(type_hint, Some("i64".to_string()));
            }
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn test_parse_counterexample_value_int_min() {
        let min_str = i128::MIN.to_string();
        match parse_counterexample_value(&min_str, None) {
            CounterexampleValue::Int { value, .. } => assert_eq!(value, i128::MIN),
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn test_parse_counterexample_value_float() {
        match parse_counterexample_value("1.234", None) {
            CounterexampleValue::Float { value } => assert!((value - 1.234).abs() < 0.001),
            _ => panic!("Expected Float"),
        }
    }

    #[test]
    fn test_parse_counterexample_value_float_negative() {
        match parse_counterexample_value("-1.234", None) {
            CounterexampleValue::Float { value } => assert!((value + 1.234).abs() < 0.001),
            _ => panic!("Expected Float"),
        }
    }

    #[test]
    fn test_parse_counterexample_value_float_scientific() {
        match parse_counterexample_value("1e10", None) {
            CounterexampleValue::Float { value } => assert!((value - 1e10).abs() < 1.0),
            _ => panic!("Expected Float"),
        }
    }

    #[test]
    fn test_parse_counterexample_value_bool() {
        assert_eq!(
            parse_counterexample_value("true", None),
            CounterexampleValue::Bool(true)
        );
        assert_eq!(
            parse_counterexample_value("false", None),
            CounterexampleValue::Bool(false)
        );
        assert_eq!(
            parse_counterexample_value("TRUE", None),
            CounterexampleValue::Bool(true)
        );
    }

    #[test]
    fn test_parse_counterexample_value_bool_mixed_case() {
        assert_eq!(
            parse_counterexample_value("True", None),
            CounterexampleValue::Bool(true)
        );
        assert_eq!(
            parse_counterexample_value("FALSE", None),
            CounterexampleValue::Bool(false)
        );
        assert_eq!(
            parse_counterexample_value("FaLsE", None),
            CounterexampleValue::Bool(false)
        );
    }

    #[test]
    fn test_parse_counterexample_value_unknown() {
        match parse_counterexample_value("unknown_value", None) {
            CounterexampleValue::Unknown(s) => assert_eq!(s, "unknown_value"),
            _ => panic!("Expected Unknown"),
        }
    }

    #[test]
    fn test_parse_counterexample_value_pointer() {
        match parse_counterexample_value("0x7fffffff", None) {
            CounterexampleValue::Unknown(s) => assert_eq!(s, "0x7fffffff"),
            _ => panic!("Expected Unknown for hex pointer"),
        }
    }

    // ==================== failed_check_count tests ====================

    #[test]
    fn test_failed_check_count() {
        assert_eq!(failed_check_count("**  0 of 5 failed"), Some((0, 5)));
        assert_eq!(failed_check_count("**  3 of 10 failed"), Some((3, 10)));
        assert_eq!(failed_check_count("no matches here"), None);
    }

    #[test]
    fn test_failed_check_count_all_failed() {
        assert_eq!(failed_check_count("**  5 of 5 failed"), Some((5, 5)));
    }

    #[test]
    fn test_failed_check_count_large_numbers() {
        assert_eq!(
            failed_check_count("**  100 of 500 failed"),
            Some((100, 500))
        );
    }

    #[test]
    fn test_failed_check_count_single() {
        assert_eq!(failed_check_count("**  1 of 1 failed"), Some((1, 1)));
    }

    #[test]
    fn test_failed_check_count_in_context() {
        let output = r"
        Running verification...
        **  2 of 10 failed
        Check summary
        ";
        assert_eq!(failed_check_count(output), Some((2, 10)));
    }

    // ==================== extract_failed_checks tests ====================

    #[test]
    fn test_extract_failed_checks_empty() {
        let output = "No checks found";
        let checks = extract_failed_checks(output);
        assert!(checks.is_empty());
    }

    #[test]
    fn test_extract_failed_checks_check_n_format() {
        let output = r#"
Check 1: overflow_check
 - Status: FAILURE
 - Description: "arithmetic overflow on addition"
 - Location: src/main.rs:42:5 in function test_fn
"#;
        let checks = extract_failed_checks(output);
        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].check_id, "overflow_check");
        assert_eq!(checks[0].description, "arithmetic overflow on addition");
        assert_eq!(checks[0].location.as_ref().unwrap().file, "src/main.rs");
        assert_eq!(checks[0].location.as_ref().unwrap().line, 42);
        assert_eq!(checks[0].location.as_ref().unwrap().column, Some(5));
        assert_eq!(checks[0].function, Some("test_fn".to_string()));
    }

    #[test]
    fn test_extract_failed_checks_multiple() {
        let output = r#"
Check 1: overflow_check
 - Status: FAILURE
 - Description: "arithmetic overflow"
 - Location: src/main.rs:42:5 in function test_fn

Check 2: assert_check
 - Status: FAILURE
 - Description: "assertion failed"
 - Location: src/lib.rs:100:10 in function helper
"#;
        let checks = extract_failed_checks(output);
        assert_eq!(checks.len(), 2);
        assert_eq!(checks[0].check_id, "overflow_check");
        assert_eq!(checks[1].check_id, "assert_check");
        assert_eq!(checks[1].location.as_ref().unwrap().line, 100);
    }

    #[test]
    fn test_extract_failed_checks_fallback_format() {
        let output = r#"
Failed Checks: assertion failed: x > 0
File: "src/main.rs", line 42, in test_function
"#;
        let checks = extract_failed_checks(output);
        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].description, "assertion failed: x > 0");
        assert_eq!(checks[0].location.as_ref().unwrap().file, "src/main.rs");
        assert_eq!(checks[0].location.as_ref().unwrap().line, 42);
        assert_eq!(checks[0].function, Some("test_function".to_string()));
    }

    #[test]
    fn test_extract_failed_checks_fallback_no_location() {
        let output = r"
Failed Checks: simple assertion failed
Some other output
";
        let checks = extract_failed_checks(output);
        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].description, "simple assertion failed");
        assert!(checks[0].location.is_none());
    }

    #[test]
    fn test_extract_failed_checks_success_only() {
        let output = r#"
Check 1: overflow_check
 - Status: SUCCESS
 - Description: "no overflow"
 - Location: src/main.rs:42:5 in function test_fn
"#;
        let checks = extract_failed_checks(output);
        assert!(checks.is_empty());
    }

    // ==================== extract_playback_test tests ====================

    #[test]
    fn test_extract_playback_test_none() {
        let output = "No playback test here";
        assert!(extract_playback_test(output).is_none());
    }

    #[test]
    fn test_extract_playback_test_with_code_block() {
        let output = r"
Concrete playback unit test for test_fn:
```
#[test]
fn kani_concrete_playback_test_fn_1() {
    let input = 42u32;
    test_fn(input);
}
```
Some other output
";
        let playback = extract_playback_test(output);
        assert!(playback.is_some());
        let test_code = playback.unwrap();
        assert!(test_code.contains("#[test]"));
        assert!(test_code.contains("kani_concrete_playback"));
    }

    #[test]
    fn test_extract_playback_test_with_rust_marker() {
        let output = r"
Concrete playback unit test for test_fn:
```rust
#[test]
fn kani_test() {
    let x = 5;
}
```
";
        let playback = extract_playback_test(output);
        assert!(playback.is_some());
        // Note: The code will include "rust\n#[test]..." because we start after ```
        let test_code = playback.unwrap();
        assert!(test_code.contains("#[test]"));
    }

    #[test]
    fn test_extract_playback_test_ends_at_counterexample() {
        let output = r"
Concrete playback unit test for test_fn:
Some test information
Test setup details
Counterexample values:
- x = 42
";
        let playback = extract_playback_test(output);
        assert!(playback.is_some());
        let test_text = playback.unwrap();
        assert!(!test_text.contains("Counterexample values"));
    }

    #[test]
    fn test_extract_playback_test_no_code_block() {
        let output = r"
Concrete playback unit test for test_fn:
Generated test code here without backticks
";
        let playback = extract_playback_test(output);
        assert!(playback.is_some());
    }

    #[test]
    fn test_extract_playback_test_content_with_counterexample_boundary() {
        // This test catches the `start + end` → `start * end` mutation
        // The content between start and "Counterexample values:" must be extracted correctly
        let output = "Concrete playback unit test for foo:\nlet x = 1;\nlet y = 2;\nCounterexample values:\nx=3";
        let playback = extract_playback_test(output);
        assert!(playback.is_some());
        let text = playback.unwrap();
        // The text should contain the setup code
        assert!(text.contains("let x = 1"));
        assert!(text.contains("let y = 2"));
        // It should NOT include Counterexample values section
        assert!(!text.contains("Counterexample values"));
    }

    // ==================== extract_witness_values tests ====================

    #[test]
    fn test_extract_witness_values_empty() {
        let output = "No counterexample values";
        let witness = extract_witness_values(output);
        assert!(witness.is_empty());
    }

    #[test]
    fn test_extract_witness_values_single() {
        let output = r"
Counterexample values:
- x = 42
";
        let witness = extract_witness_values(output);
        assert_eq!(witness.len(), 1);
        match witness.get("x") {
            Some(CounterexampleValue::UInt { value, .. }) => assert_eq!(*value, 42),
            _ => panic!("Expected UInt for x"),
        }
    }

    #[test]
    fn test_extract_witness_values_multiple() {
        let output = r"
Counterexample values:
- x = 42
- y = -10
- flag = true
";
        let witness = extract_witness_values(output);
        assert_eq!(witness.len(), 3);
        match witness.get("x") {
            Some(CounterexampleValue::UInt { value, .. }) => assert_eq!(*value, 42),
            _ => panic!("Expected UInt for x"),
        }
        match witness.get("y") {
            Some(CounterexampleValue::Int { value, .. }) => assert_eq!(*value, -10),
            _ => panic!("Expected Int for y"),
        }
        match witness.get("flag") {
            Some(CounterexampleValue::Bool(b)) => assert!(*b),
            _ => panic!("Expected Bool for flag"),
        }
    }

    #[test]
    fn test_extract_witness_values_with_type_hints() {
        let output = r"
Counterexample values:
- count = 255 (u8)
- offset = -1000 (i32)
";
        let witness = extract_witness_values(output);
        match witness.get("count") {
            Some(CounterexampleValue::UInt { value, type_hint }) => {
                assert_eq!(*value, 255);
                assert_eq!(*type_hint, Some("u8".to_string()));
            }
            _ => panic!("Expected UInt for count"),
        }
        match witness.get("offset") {
            Some(CounterexampleValue::Int { value, type_hint }) => {
                assert_eq!(*value, -1000);
                assert_eq!(*type_hint, Some("i32".to_string()));
            }
            _ => panic!("Expected Int for offset"),
        }
    }

    #[test]
    fn test_extract_witness_values_floats() {
        let output = r"
Counterexample values:
- ratio = 1.5
- negative = -2.5
";
        let witness = extract_witness_values(output);
        match witness.get("ratio") {
            Some(CounterexampleValue::Float { value }) => assert!((value - 1.5).abs() < 0.001),
            _ => panic!("Expected Float for ratio"),
        }
    }

    // ==================== extract_concrete_vals_witness tests ====================

    #[test]
    fn test_extract_concrete_vals_witness() {
        let output = r"
concrete_vals:
// 42
vec![42, 0, 0, 0]
// -5
vec![251, 255, 255, 255]
";
        let mut witness = HashMap::new();
        extract_concrete_vals_witness(output, &mut witness);
        assert_eq!(witness.len(), 2);
        match witness.get("arg0") {
            Some(CounterexampleValue::UInt { value, .. }) => assert_eq!(*value, 42),
            _ => panic!("Expected UInt for arg0"),
        }
        match witness.get("arg1") {
            Some(CounterexampleValue::Int { value, .. }) => assert_eq!(*value, -5),
            _ => panic!("Expected Int for arg1"),
        }
    }

    #[test]
    fn test_extract_concrete_vals_witness_empty() {
        let output = "No concrete vals here";
        let mut witness = HashMap::new();
        extract_concrete_vals_witness(output, &mut witness);
        assert!(witness.is_empty());
    }

    #[test]
    fn test_extract_concrete_vals_witness_single() {
        let output = r"
concrete_vals:
// 100
vec![100, 0, 0, 0]
";
        let mut witness = HashMap::new();
        extract_concrete_vals_witness(output, &mut witness);
        assert_eq!(witness.len(), 1);
    }

    // ==================== extract_raw_byte_arrays tests ====================

    #[test]
    fn test_extract_raw_byte_arrays() {
        let output = r"
concrete_vals:
vec![1, 2, 3, 4]
vec![255, 0, 128]
";
        let mut witness = HashMap::new();
        extract_raw_byte_arrays(output, &mut witness);
        assert_eq!(witness.len(), 2);
        match witness.get("bytes_arg0") {
            Some(CounterexampleValue::Bytes(bytes)) => assert_eq!(*bytes, vec![1u8, 2, 3, 4]),
            _ => panic!("Expected Bytes for bytes_arg0"),
        }
        match witness.get("bytes_arg1") {
            Some(CounterexampleValue::Bytes(bytes)) => assert_eq!(*bytes, vec![255u8, 0, 128]),
            _ => panic!("Expected Bytes for bytes_arg1"),
        }
    }

    #[test]
    fn test_extract_raw_byte_arrays_empty() {
        let output = "No concrete_vals here";
        let mut witness = HashMap::new();
        extract_raw_byte_arrays(output, &mut witness);
        assert!(witness.is_empty());
    }

    #[test]
    fn test_extract_raw_byte_arrays_no_digits() {
        let output = r"
concrete_vals:
vec![]
";
        let mut witness = HashMap::new();
        extract_raw_byte_arrays(output, &mut witness);
        // Empty vec![] won't match because there are no digits
        assert!(witness.is_empty());
    }

    // ==================== extract_raw_counterexample tests ====================

    #[test]
    fn test_extract_raw_counterexample_none() {
        let output = "Nothing relevant here";
        assert!(extract_raw_counterexample(output).is_none());
    }

    #[test]
    fn test_extract_raw_counterexample_playback() {
        let output = r"
Concrete playback unit test for function:
#[test]
fn test() {}
Counterexample values:
- x = 42
";
        let raw = extract_raw_counterexample(output);
        assert!(raw.is_some());
        let text = raw.unwrap();
        assert!(text.contains("Concrete playback"));
        assert!(!text.contains("Counterexample values"));
    }

    #[test]
    fn test_extract_raw_counterexample_playback_no_values() {
        let output = r"
Concrete playback unit test for function:
#[test]
fn test() {}
End of output
";
        let raw = extract_raw_counterexample(output);
        assert!(raw.is_some());
        assert!(raw.unwrap().contains("Concrete playback"));
    }

    #[test]
    fn test_extract_raw_counterexample_values_only() {
        let output = r"
Some output
Counterexample values:
- x = 42
- y = 100
";
        let raw = extract_raw_counterexample(output);
        assert!(raw.is_some());
        let text = raw.unwrap();
        assert!(text.starts_with("Counterexample values:"));
    }

    #[test]
    fn test_extract_raw_counterexample_failed_checks() {
        let output = r"
Failed Checks: assertion failed
Location: file.rs:10
Details here
More info
Extra line
";
        let raw = extract_raw_counterexample(output);
        assert!(raw.is_some());
        let text = raw.unwrap();
        assert!(text.contains("Failed Checks"));
        // Takes first 5 lines
    }

    #[test]
    fn test_extract_raw_counterexample_status_failure() {
        let output = r"
Check completed
Status: FAILURE - assertion failed at line 10
Done
";
        let raw = extract_raw_counterexample(output);
        assert!(raw.is_some());
        assert!(raw.unwrap().contains("Status: FAILURE"));
    }

    #[test]
    fn test_extract_raw_counterexample_assertion_line() {
        let output = r"
Running tests
assertion `x > 0` failed
Completed
";
        let raw = extract_raw_counterexample(output);
        assert!(raw.is_some());
        assert!(raw.unwrap().contains("assertion"));
    }

    // ==================== extract_diagnostics tests ====================

    #[test]
    fn test_extract_diagnostics_empty() {
        let output = "No diagnostics here";
        let diagnostics = extract_diagnostics(output);
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn test_extract_diagnostics_with_count() {
        let output = "**  2 of 5 failed";
        let diagnostics = extract_diagnostics(output);
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0], "2 of 5 checks failed");
    }

    #[test]
    fn test_extract_diagnostics_with_failed_checks() {
        let output = r"
Failed Checks: overflow detected
Location: src/main.rs:42
";
        let diagnostics = extract_diagnostics(output);
        assert!(diagnostics.len() >= 2);
        assert!(diagnostics.iter().any(|d| d.contains("Failed Checks")));
        assert!(diagnostics.iter().any(|d| d.contains("Location")));
    }

    #[test]
    fn test_extract_diagnostics_mixed() {
        let output = r"
**  1 of 3 failed
Failed Checks: assertion error
Location: test.rs:10
";
        let diagnostics = extract_diagnostics(output);
        assert!(diagnostics.len() >= 3);
        assert!(diagnostics
            .iter()
            .any(|d| d.contains("1 of 3 checks failed")));
    }

    // ==================== extract_structured_counterexample tests ====================

    #[test]
    fn test_extract_structured_counterexample_empty() {
        let output = "No counterexample info";
        let ce = extract_structured_counterexample(output);
        assert!(ce.failed_checks.is_empty());
        assert!(ce.playback_test.is_none());
        assert!(ce.witness.is_empty());
        assert!(ce.raw.is_none());
    }

    #[test]
    fn test_extract_structured_counterexample_full() {
        let output = r#"
Check 1: overflow_check
 - Status: FAILURE
 - Description: "overflow detected"
 - Location: src/main.rs:42:5 in function test_fn

Concrete playback unit test for test_fn:
```
#[test]
fn kani_test() {}
```
Counterexample values:
- x = 42
- y = -10
"#;
        let ce = extract_structured_counterexample(output);
        assert_eq!(ce.failed_checks.len(), 1);
        assert_eq!(ce.failed_checks[0].check_id, "overflow_check");
        assert!(ce.playback_test.is_some());
        assert_eq!(ce.witness.len(), 2);
        assert!(ce.raw.is_some());
    }

    #[test]
    fn test_extract_structured_counterexample_partial_checks_only() {
        let output = r#"
Check 1: bounds_check
 - Status: FAILURE
 - Description: "index out of bounds"
 - Location: src/lib.rs:100:3 in function get_item
"#;
        let ce = extract_structured_counterexample(output);
        assert_eq!(ce.failed_checks.len(), 1);
        assert!(ce.playback_test.is_none());
        assert!(ce.witness.is_empty());
    }

    #[test]
    fn test_extract_structured_counterexample_partial_witness_only() {
        let output = r"
Counterexample values:
- index = 999
- length = 10
";
        let ce = extract_structured_counterexample(output);
        assert!(ce.failed_checks.is_empty());
        assert_eq!(ce.witness.len(), 2);
        assert!(ce.raw.is_some());
    }

    #[test]
    fn test_extract_structured_counterexample_with_concrete_vals() {
        let output = r"
concrete_vals:
// 42
vec![42, 0, 0, 0]
vec![1, 2, 3]
";
        let ce = extract_structured_counterexample(output);
        // Should have arg0 from concrete_vals and bytes from raw arrays
        assert!(!ce.witness.is_empty());
    }

    #[test]
    fn test_extract_structured_counterexample_multiple_checks() {
        let output = r#"
Check 1: check_a
 - Status: FAILURE
 - Description: "error a"
 - Location: a.rs:1:1 in function fa

Check 2: check_b
 - Status: FAILURE
 - Description: "error b"
 - Location: b.rs:2:2 in function fb

Check 3: check_c
 - Status: FAILURE
 - Description: "error c"
 - Location: c.rs:3:3 in function fc
"#;
        let ce = extract_structured_counterexample(output);
        assert_eq!(ce.failed_checks.len(), 3);
        assert_eq!(ce.failed_checks[0].check_id, "check_a");
        assert_eq!(ce.failed_checks[1].check_id, "check_b");
        assert_eq!(ce.failed_checks[2].check_id, "check_c");
    }
}
