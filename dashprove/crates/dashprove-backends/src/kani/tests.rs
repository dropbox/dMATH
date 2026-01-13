//! Tests for Kani backend

use super::config::KaniOutput;
use super::parsing::{extract_structured_counterexample, parse_byte_array, parse_output};
use crate::traits::{
    CounterexampleValue, FailedCheck, SourceLocation, StructuredCounterexample, VerificationStatus,
};
use std::time::Duration;

fn make_output(text: &str) -> KaniOutput {
    KaniOutput {
        stdout: text.to_string(),
        stderr: String::new(),
        exit_code: Some(0),
        duration: Duration::from_secs(1),
    }
}

const KANI_PASS_OUTPUT: &str = include_str!("../../../../examples/kani/OUTPUT_pass.txt");
const KANI_FAIL_OUTPUT: &str = include_str!("../../../../examples/kani/OUTPUT_fail.txt");
const KANI_COUNTEREXAMPLE: &str =
    include_str!("../../../../examples/kani/OUTPUT_counterexample.txt");

#[test]
fn parse_success_output_marks_proven() {
    let output = make_output(KANI_PASS_OUTPUT);
    let result = parse_output(&output);
    assert!(matches!(result.status, VerificationStatus::Proven));
    assert!(result
        .diagnostics
        .iter()
        .any(|d| d.contains("0 of 8 checks failed")));
}

#[test]
fn parse_failure_output_marks_disproven() {
    let mut output = make_output(KANI_FAIL_OUTPUT);
    output.exit_code = Some(1);
    let result = parse_output(&output);
    assert!(matches!(result.status, VerificationStatus::Disproven));
    // Check that structured counterexample has raw text with the error message
    assert!(result
        .counterexample
        .as_ref()
        .and_then(|c| c.raw.as_ref())
        .map(|r| r.contains("attempt to divide by zero"))
        .unwrap_or(false));
}

#[test]
fn extract_counterexample_from_concrete_playback() {
    let ce = extract_structured_counterexample(KANI_COUNTEREXAMPLE);
    // Check playback test is extracted
    assert!(ce.playback_test.is_some());
    assert!(ce.playback_test.as_ref().unwrap().contains("concrete_vals"));
}

#[test]
fn extract_witness_values_from_counterexample() {
    let ce = extract_structured_counterexample(KANI_COUNTEREXAMPLE);
    // Should extract witness values: a = 4294967295 (u32::MAX), b = 0
    assert!(
        !ce.witness.is_empty(),
        "Expected witness values to be extracted"
    );

    // Check that 'a' was extracted with correct value
    if let Some(CounterexampleValue::UInt { value, type_hint }) = ce.witness.get("a") {
        assert_eq!(*value, 4294967295u128);
        assert!(type_hint
            .as_ref()
            .map(|t| t.contains("u32"))
            .unwrap_or(false));
    } else {
        panic!("Expected 'a' to be an unsigned integer value");
    }

    // Check that 'b' was extracted with value 0
    if let Some(CounterexampleValue::UInt { value, .. }) = ce.witness.get("b") {
        assert_eq!(*value, 0u128);
    } else {
        panic!("Expected 'b' to be an unsigned integer value");
    }
}

#[test]
fn extract_failed_checks_from_failure_output() {
    let ce = extract_structured_counterexample(KANI_FAIL_OUTPUT);
    // Should extract the failed check description
    assert!(
        !ce.failed_checks.is_empty(),
        "Expected failed checks to be extracted"
    );

    let check = &ce.failed_checks[0];
    assert!(
        check.description.contains("attempt to divide by zero")
            || check.description.contains("divide")
    );
}

#[test]
fn structured_counterexample_summary() {
    let mut ce = StructuredCounterexample::new();
    ce.failed_checks.push(FailedCheck {
        check_id: "test::assertion.1".to_string(),
        description: "division by zero".to_string(),
        location: Some(SourceLocation {
            file: "src/lib.rs".to_string(),
            line: 10,
            column: Some(5),
        }),
        function: Some("test::divide".to_string()),
    });
    ce.witness.insert(
        "x".to_string(),
        CounterexampleValue::UInt {
            value: 42,
            type_hint: None,
        },
    );

    let summary = ce.summary();
    assert!(summary.contains("Failed: division by zero"));
    assert!(summary.contains("x = 42"));
}

#[test]
fn counterexample_value_display() {
    let int_val = CounterexampleValue::Int {
        value: -42,
        type_hint: Some("i32".to_string()),
    };
    assert_eq!(format!("{}", int_val), "-42 (i32)");

    let uint_val = CounterexampleValue::UInt {
        value: 255,
        type_hint: None,
    };
    assert_eq!(format!("{}", uint_val), "255");

    let bool_val = CounterexampleValue::Bool(true);
    assert_eq!(format!("{}", bool_val), "true");

    let unknown_val = CounterexampleValue::Unknown("??".to_string());
    assert_eq!(format!("{}", unknown_val), "??");
}

#[test]
fn source_location_display() {
    let loc = SourceLocation {
        file: "src/main.rs".to_string(),
        line: 42,
        column: Some(10),
    };
    assert_eq!(format!("{}", loc), "src/main.rs:42:10");

    let loc_no_col = SourceLocation {
        file: "src/lib.rs".to_string(),
        line: 100,
        column: None,
    };
    assert_eq!(format!("{}", loc_no_col), "src/lib.rs:100");
}

#[test]
fn extract_raw_byte_arrays_from_counterexample() {
    let ce = extract_structured_counterexample(KANI_COUNTEREXAMPLE);

    // Should extract bytes_arg0 and bytes_arg1 with raw byte arrays
    assert!(
        ce.witness.contains_key("bytes_arg0"),
        "Expected bytes_arg0 to be extracted"
    );
    assert!(
        ce.witness.contains_key("bytes_arg1"),
        "Expected bytes_arg1 to be extracted"
    );

    // Check bytes_arg0 = [255, 255, 255, 255] (u32::MAX)
    if let Some(CounterexampleValue::Bytes(bytes)) = ce.witness.get("bytes_arg0") {
        assert_eq!(bytes, &[255, 255, 255, 255]);
    } else {
        panic!("Expected bytes_arg0 to be Bytes variant");
    }

    // Check bytes_arg1 = [0, 0, 0, 0] (0)
    if let Some(CounterexampleValue::Bytes(bytes)) = ce.witness.get("bytes_arg1") {
        assert_eq!(bytes, &[0, 0, 0, 0]);
    } else {
        panic!("Expected bytes_arg1 to be Bytes variant");
    }
}

#[test]
fn parse_byte_array_handles_various_formats() {
    // Empty array
    assert_eq!(parse_byte_array(""), Some(vec![]));
    assert_eq!(parse_byte_array("  "), Some(vec![]));

    // Single byte
    assert_eq!(parse_byte_array("42"), Some(vec![42]));

    // Multiple bytes with spaces
    assert_eq!(parse_byte_array("1, 2, 3"), Some(vec![1, 2, 3]));

    // No spaces
    assert_eq!(parse_byte_array("1,2,3"), Some(vec![1, 2, 3]));

    // Invalid values return None
    assert_eq!(parse_byte_array("256"), None); // > 255
    assert_eq!(parse_byte_array("-1"), None); // negative
    assert_eq!(parse_byte_array("abc"), None); // non-numeric
}

#[test]
fn failed_check_display() {
    let check = FailedCheck {
        check_id: "test".to_string(),
        description: "division by zero".to_string(),
        location: Some(SourceLocation {
            file: "src/lib.rs".to_string(),
            line: 10,
            column: Some(5),
        }),
        function: Some("my_function".to_string()),
    };

    let display = format!("{}", check);
    assert!(display.contains("division by zero"));
    assert!(display.contains("src/lib.rs:10:5"));
    assert!(display.contains("my_function"));
}

#[tokio::test]
async fn write_inline_project_creates_valid_structure() {
    use super::project::write_inline_project;
    use dashprove_usl::{parse, typecheck};

    // Simple Rust code
    let code = r#"
pub fn identity(x: u32) -> u32 {
    x
}
"#;

    // Simple contract
    let spec_source = r#"
contract identity(x: Int) -> Int {
    ensures { result == x }
}
"#;

    let spec = parse(spec_source).expect("parse");
    let typed_spec = typecheck(spec).expect("typecheck");

    let (temp_dir, manifest_path) = write_inline_project(code, &typed_spec)
        .await
        .expect("write_inline_project");

    // Check that manifest was created
    assert!(manifest_path.exists());

    // Check that lib.rs was created
    let lib_path = temp_dir.path().join("src/lib.rs");
    assert!(lib_path.exists());

    // Read and verify lib.rs contains both user code and harness
    let lib_content = std::fs::read_to_string(&lib_path).expect("read lib.rs");
    assert!(lib_content.contains("pub fn identity"));
    assert!(lib_content.contains("#[kani::proof]"));
    assert!(lib_content.contains("verify_identity"));
}

#[tokio::test]
async fn write_inline_project_combines_code_and_contracts() {
    use super::project::write_inline_project;
    use dashprove_usl::{parse, typecheck};

    let code = r#"
pub fn bounded_add(a: u32, b: u32) -> Option<u32> {
    a.checked_add(b)
}
"#;

    let spec_source = r#"
contract bounded_add(a: Int, b: Int) -> Result<Int> {
    ensures { result >= a or result >= b }
}
"#;

    let spec = parse(spec_source).expect("parse");
    let typed_spec = typecheck(spec).expect("typecheck");

    let (temp_dir, _manifest_path) = write_inline_project(code, &typed_spec)
        .await
        .expect("write_inline_project");

    let lib_content =
        std::fs::read_to_string(temp_dir.path().join("src/lib.rs")).expect("read lib.rs");

    // Should have the user code
    assert!(lib_content.contains("pub fn bounded_add"));
    assert!(lib_content.contains("checked_add"));

    // Should have the generated harness
    assert!(lib_content.contains("#[cfg(kani)]"));
    assert!(lib_content.contains("#[kani::proof]"));
    assert!(lib_content.contains("fn verify_bounded_add"));

    // Should have preconditions (kani::assume) and postconditions (kani::assert)
    assert!(lib_content.contains("kani::assert"));
}

#[test]
fn kani_backend_supports_contracts() {
    use super::KaniBackend;
    use crate::traits::{PropertyType, VerificationBackend};

    let backend = KaniBackend::new();
    let supports = backend.supports();
    assert!(supports.contains(&PropertyType::Contract));
}
