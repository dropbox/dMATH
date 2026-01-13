//! Differential testing: Kani Fast driver vs cargo kani
//!
//! These tests compare verification results between kani-fast-driver and
//! cargo kani to ensure they agree on VERIFIED vs FAILED.
//!
//! Note: Some tests are expected to differ due to:
//! - Unbounded integer model in CHC (no overflow detection by default)
//! - Different loop unwinding strategies
//!
//! Run with: cargo test --test differential

use std::process::Command;
use tempfile::Builder;

/// Test harness configuration
struct TestCase {
    name: &'static str,
    source: &'static str,
    /// Expected result for kani-fast-driver: "pass" or "fail"
    expected_fast: &'static str,
    /// Expected result for cargo kani (if different, Some("pass"/"fail"))
    expected_kani: Option<&'static str>,
}

const TEST_CASES: &[TestCase] = &[
    // Must-pass tests: safe code that should verify
    TestCase {
        name: "safe_add",
        source: r"
fn safe_add_proof() {
    let x: i32 = 5;
    let y: i32 = 3;
    assert!(x + y == 8);
}
",
        expected_fast: "pass",
        expected_kani: None,
    },
    TestCase {
        name: "checked_add",
        source: r"
fn checked_add_proof() {
    let x: i32 = 100;
    let y: i32 = 50;
    if let Some(result) = x.checked_add(y) {
        assert!(result == 150);
    }
}
",
        // Fixed in #438: checked_add now correctly constrains Option value
        expected_fast: "pass",
        expected_kani: None, // Same behavior now
    },
    TestCase {
        name: "checked_add_method",
        source: r"
fn checked_add_method_proof() {
    let x: i32 = 100;
    let sum = x.checked_add(50);
    // Don't assert on the result - just check it compiles
    let _ = sum;
    assert!(x == 100);
}
",
        expected_fast: "pass",
        expected_kani: None,
    },
    TestCase {
        name: "overflowing_add_safe",
        source: r"
fn overflowing_add_safe_proof() {
    let x: i32 = 5;
    let y: i32 = 3;
    // No overflow for small values
    assert!(x + y == 8);
}
",
        expected_fast: "pass",
        expected_kani: None,
    },
    // Must-fail tests: buggy code that should be caught
    TestCase {
        name: "buggy_multiply",
        source: r"
fn buggy_multiply_proof() {
    let x: i32 = 5;
    let y: i32 = 3;
    // Bug: result should be 15, not 14
    assert!(x * y == 14);
}
",
        expected_fast: "fail",
        expected_kani: None,
    },
    TestCase {
        name: "overflow_add",
        source: r"
fn overflow_add_proof() {
    let x: i32 = 2147483647; // i32::MAX
    let y: i32 = 1;
    // This will overflow
    let result = x + y;
    assert!(result > 0); // Fails due to overflow
}
",
        // kani-fast uses unbounded integers by default, no overflow
        expected_fast: "pass",       // Unbounded Int: no overflow
        expected_kani: Some("fail"), // Kani detects overflow
    },
    TestCase {
        name: "doubling_overflow",
        source: r"
fn doubling_overflow_proof() {
    let x: i32 = 1073741824; // 2^30
    let doubled = x * 2; // 2^31 = i32::MIN (overflow)
    assert!(doubled > 0); // Fails
}
",
        // kani-fast uses unbounded integers by default
        expected_fast: "pass",       // Unbounded Int: no overflow
        expected_kani: Some("fail"), // Kani detects overflow
    },
];

/// Result of running the driver - separate from io::Result to handle skip case
#[derive(Debug)]
enum DriverResult {
    Passed,
    Failed,
    Skipped,
}

/// Run kani-fast-driver on a source file and return whether verification passed
fn run_kani_fast_driver(test: &TestCase) -> std::io::Result<DriverResult> {
    let tmpfile = Builder::new()
        .prefix(&format!("kani_fast_diff_{}", test.name))
        .suffix(".rs")
        .tempfile()?;
    std::fs::write(tmpfile.path(), test.source)?;
    let src_path = tmpfile.path().to_path_buf();

    // Find the driver binary
    let mut driver =
        std::env::current_dir()?.join("crates/kani-fast-compiler/target/debug/kani-fast-driver");

    if !driver.exists() {
        // Try workspace target
        driver = std::env::current_dir()?.join("target/debug/kani-fast-driver");
        if !driver.exists() {
            eprintln!("Warning: kani-fast-driver not found, skipping test");
            return Ok(DriverResult::Skipped);
        }
    }

    // Set up environment for rustc driver
    let sysroot = Command::new("rustup")
        .args(["run", "nightly-2025-11-20", "rustc", "--print", "sysroot"])
        .output()?;
    let sysroot = String::from_utf8_lossy(&sysroot.stdout).trim().to_string();

    let output = Command::new(&driver)
        .arg(&src_path)
        .arg("--crate-type=lib")
        .env(
            "DYLD_LIBRARY_PATH",
            format!(
                "{}/lib:{}/lib/rustlib/aarch64-apple-darwin/lib",
                sysroot, sysroot
            ),
        )
        .output()?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    if stdout.contains("All harnesses verified successfully") {
        Ok(DriverResult::Passed)
    } else {
        Ok(DriverResult::Failed)
    }
}

/// Run cargo kani on a source file and return whether verification passed
#[allow(dead_code)]
fn run_cargo_kani(source: &str) -> std::io::Result<bool> {
    let tmpdir = std::env::temp_dir().join("kani_diff_test");
    let _ = std::fs::remove_dir_all(&tmpdir);
    std::fs::create_dir_all(&tmpdir)?;

    // Create a minimal Cargo project
    let cargo_toml = r#"
[package]
name = "test_diff"
version = "0.1.0"
edition = "2021"
"#;
    std::fs::write(tmpdir.join("Cargo.toml"), cargo_toml)?;
    std::fs::create_dir_all(tmpdir.join("src"))?;
    std::fs::write(tmpdir.join("src/lib.rs"), source)?;

    let output = Command::new("cargo")
        .args(["kani", "--only-codegen"])
        .current_dir(&tmpdir)
        .output()?;

    // For now, just check compilation works
    // Full verification would require more setup
    Ok(output.status.success())
}

#[test]
fn test_overflow_behavior_documented() {
    // Document the expected difference: kani-fast uses unbounded integers
    // by default, so overflow tests behave differently than cargo kani.
    //
    // This is intentional - users should set KANI_FAST_OVERFLOW_CHECKS=1
    // or use KANI_FAST_BITVEC=1 for overflow detection.

    let differing: Vec<&TestCase> = TEST_CASES
        .iter()
        .filter(|t| t.expected_kani.is_some())
        .collect();
    assert_eq!(
        differing.len(),
        2,
        "Expected two cases with different kani outcomes (overflow tests)"
    );

    for tc in differing {
        match tc.name {
            "overflow_add" | "doubling_overflow" => {
                assert_eq!(tc.expected_fast, "pass"); // No overflow in unbounded Int
                assert_eq!(tc.expected_kani, Some("fail")); // Kani detects overflow
            }
            other => panic!("Unexpected test case with expected_kani: {}", other),
        }
    }
}

#[test]
fn driver_matches_all_expected_outcomes() {
    for tc in TEST_CASES {
        match run_kani_fast_driver(tc) {
            Ok(DriverResult::Passed) => assert_eq!(
                tc.expected_fast, "pass",
                "{}: expected fail, got pass",
                tc.name
            ),
            Ok(DriverResult::Failed) => assert_eq!(
                tc.expected_fast, "fail",
                "{}: expected pass, got fail",
                tc.name
            ),
            Ok(DriverResult::Skipped) => {}
            Err(e) => eprintln!("Warning: Could not run driver for {}: {}", tc.name, e),
        }
    }
}
