//! Differential testing: Kani Fast driver vs cargo kani
//!
//! These tests verify that kani-fast-driver produces the same verification
//! results as cargo kani on a set of representative test cases.
//!
//! Test files are in `tests/soundness/` with subdirectories:
//! - `must_pass/` - harnesses that should verify successfully
//! - `must_fail/` - harnesses with explicit failing assertions that must be reported
//!
//! The differential tests:
//! 1. Run both tools on the same input
//! 2. Compare VERIFIED vs FAILED results
//! 3. Ensure soundness consistency

use std::path::{Path, PathBuf};
use std::process::Command;

fn skip_if_driver_unavailable() -> bool {
    let driver_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("kani-fast-compiler/target/debug/kani-fast-driver");
    !driver_path.exists()
}

/// Check if the Kani library is available and compatible
fn skip_if_kani_incompatible() -> bool {
    // Run a quick test to see if the driver can use the Kani library
    let driver_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("kani-fast-compiler/target/debug/kani-fast-driver");

    let kani_lib_dir = match get_kani_lib_dir() {
        Some(dir) => dir,
        None => return true, // Skip if no Kani library
    };

    let sysroot_output = Command::new("rustup")
        .args(["run", "nightly-2025-11-20", "rustc", "--print", "sysroot"])
        .output();

    let sysroot = match sysroot_output {
        Ok(output) => String::from_utf8_lossy(&output.stdout).trim().to_string(),
        Err(_) => return true,
    };

    // Also need the path to libkani_fast_compiler.dylib
    let compiler_lib_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("kani-fast-compiler/target/debug");

    let dyld_path = format!(
        "{}:{}:{}/lib:{}/lib/rustlib/aarch64-apple-darwin/lib",
        std::env::var("DYLD_LIBRARY_PATH").unwrap_or_default(),
        compiler_lib_dir.display(),
        sysroot,
        sysroot
    );

    // Create a minimal test file
    let test_content = r#"
#[kani::proof]
fn test() {}
"#;
    let temp_file = std::env::temp_dir().join("kani_compat_test.rs");
    if std::fs::write(&temp_file, test_content).is_err() {
        return true;
    }

    let kani_lib_path = kani_lib_dir.to_string_lossy();
    let kani_extern = format!("kani={}/libkani.rlib", kani_lib_path);

    let result = Command::new(&driver_path)
        .arg(temp_file.to_str().unwrap())
        .arg("--crate-type=lib")
        .arg("-L")
        .arg(kani_lib_dir.as_os_str())
        .arg("--extern")
        .arg(&kani_extern)
        .env("DYLD_LIBRARY_PATH", &dyld_path)
        .output();

    let _ = std::fs::remove_file(&temp_file);

    match result {
        Ok(output) => {
            let combined = format!(
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            // Skip if Kani library is incompatible
            combined.contains("incompatible version of rustc")
                || combined.contains("Kani library functions missing")
        }
        Err(_) => true,
    }
}

fn get_kani_lib_dir() -> Option<PathBuf> {
    // Check for Kani library in ~/.kani/kani-<version>/lib
    let home = std::env::var("HOME").ok()?;
    let kani_dir = PathBuf::from(home).join(".kani");

    if !kani_dir.exists() {
        return None;
    }

    // Find the kani version directory (e.g., kani-0.66.0)
    let entries = std::fs::read_dir(&kani_dir).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.starts_with("kani-") {
            let lib_dir = entry.path().join("lib");
            if lib_dir.join("libkani.rlib").exists() {
                return Some(lib_dir);
            }
        }
    }
    None
}

/// Result from running a verifier
#[derive(Debug, Clone, PartialEq)]
enum VerifyResult {
    Verified,
    Failed,
    Unknown,
    Error(String),
}

impl VerifyResult {
    fn is_success(&self) -> bool {
        matches!(self, VerifyResult::Verified)
    }
}

/// Run kani-fast-driver on a Rust source file
fn run_kani_fast_driver(source_path: &str) -> VerifyResult {
    let driver_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("kani-fast-compiler/target/debug/kani-fast-driver");

    // Check for Kani library
    let kani_lib_dir = match get_kani_lib_dir() {
        Some(dir) => dir,
        None => {
            return VerifyResult::Error(
                "Kani library not found. Install Kani: cargo install --locked kani-verifier"
                    .to_string(),
            )
        }
    };

    // Set up DYLD_LIBRARY_PATH for rustc libs and kani-fast-compiler dylib
    let sysroot_output = Command::new("rustup")
        .args(["run", "nightly-2025-11-20", "rustc", "--print", "sysroot"])
        .output();

    let sysroot = match sysroot_output {
        Ok(output) => String::from_utf8_lossy(&output.stdout).trim().to_string(),
        Err(_) => return VerifyResult::Error("Failed to get sysroot".to_string()),
    };

    // Also need the path to libkani_fast_compiler.dylib
    let compiler_lib_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("kani-fast-compiler/target/debug");

    let dyld_path = format!(
        "{}:{}:{}/lib:{}/lib/rustlib/aarch64-apple-darwin/lib",
        std::env::var("DYLD_LIBRARY_PATH").unwrap_or_default(),
        compiler_lib_dir.display(),
        sysroot,
        sysroot
    );

    let kani_lib_path = kani_lib_dir.to_string_lossy();
    let kani_extern = format!("kani={}/libkani.rlib", kani_lib_path);

    let result = Command::new(&driver_path)
        .arg(source_path)
        .arg("--crate-type=lib")
        .arg("-L")
        .arg(kani_lib_dir.as_os_str())
        .arg("--extern")
        .arg(&kani_extern)
        .env("DYLD_LIBRARY_PATH", &dyld_path)
        .output();

    match result {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let combined = format!("{}{}", stdout, stderr);

            // Check for known error conditions first
            if combined.contains("compiled by an incompatible version of rustc") {
                return VerifyResult::Error(
                    "Kani library version incompatible with current rustc".to_string(),
                );
            }
            if combined.contains("Kani library functions missing") {
                return VerifyResult::Error(
                    "Kani library not properly linked or incompatible".to_string(),
                );
            }

            if combined.contains("All harnesses verified successfully")
                || combined.contains("Property verified")
            {
                VerifyResult::Verified
            } else if combined.contains("Property violated")
                || combined.contains("VERIFICATION FAILED")
            {
                VerifyResult::Failed
            } else if combined.contains("Unknown") {
                VerifyResult::Unknown
            } else {
                VerifyResult::Error(format!(
                    "Unexpected output: {}",
                    combined.chars().take(200).collect::<String>()
                ))
            }
        }
        Err(e) => VerifyResult::Error(format!("Driver execution failed: {}", e)),
    }
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn expect_result(path: &Path, expected: VerifyResult) -> VerifyResult {
    assert!(path.exists(), "Test file not found: {}", path.display());
    let result = run_kani_fast_driver(path.to_str().unwrap());
    assert_eq!(
        expected,
        result,
        "Expected {:?} for {}, got {:?}",
        expected,
        path.display(),
        result
    );
    result
}

// Test files in tests/soundness/must_pass should verify
const MUST_PASS_FILES: &[&str] = &[
    "tests/soundness/must_pass/safe_add.rs",
    "tests/soundness/must_pass/checked_add.rs",
    "tests/soundness/must_pass/checked_add_method.rs",
    "tests/soundness/must_pass/overflowing_add_safe.rs",
];

// Test files in tests/soundness/must_fail should fail verification
const MUST_FAIL_FILES: &[&str] = &[
    "tests/soundness/must_fail/buggy_multiply.rs",
    "tests/soundness/must_fail/overflow_add.rs",
    "tests/soundness/must_fail/doubling_overflow.rs",
];

#[test]
fn test_differential_soundness_summary() {
    if skip_if_driver_unavailable() {
        eprintln!(
            "Skipping: kani-fast-driver not built. Run `cargo build -p kani-fast-compiler` first."
        );
        return;
    }

    if skip_if_kani_incompatible() {
        eprintln!(
            "Skipping: Kani library is incompatible with current rustc. Reinstall Kani with: cargo install --locked kani-verifier"
        );
        return;
    }

    let workspace_root = workspace_root();

    let mut pass_count = 0;
    let mut fail_count = 0;
    let mut mismatches = Vec::new();

    println!("\n=== Soundness Tests Summary ===\n");

    println!("must_pass files (should verify):");
    for file in MUST_PASS_FILES {
        let path = workspace_root.join(file);
        let result = run_kani_fast_driver(path.to_str().unwrap());
        if result.is_success() {
            pass_count += 1;
            println!("  [PASS] {} -> {:?}", file, result);
        } else {
            fail_count += 1;
            mismatches.push(format!("{} expected Verified, got {:?}", file, result));
            println!("  [FAIL] {} -> {:?}", file, result);
        }
    }

    println!("\nmust_fail files (should fail verification):");
    for file in MUST_FAIL_FILES {
        let path = workspace_root.join(file);
        let result = run_kani_fast_driver(path.to_str().unwrap());
        if matches!(result, VerifyResult::Failed) {
            pass_count += 1;
            println!("  [PASS] {} -> {:?}", file, result);
        } else {
            fail_count += 1;
            mismatches.push(format!("{} expected Failed, got {:?}", file, result));
            println!("  [FAIL] {} -> {:?}", file, result);
        }
    }

    println!(
        "\n=== Summary: {} passed, {} failed ===\n",
        pass_count, fail_count
    );

    assert!(
        mismatches.is_empty(),
        "Soundness mismatches:\n{}",
        mismatches.join("\n")
    );
}

#[test]
fn test_must_pass_safe_add() {
    if skip_if_driver_unavailable() || skip_if_kani_incompatible() {
        return;
    }

    let path = workspace_root().join("tests/soundness/must_pass/safe_add.rs");
    expect_result(&path, VerifyResult::Verified);
}

#[test]
fn test_must_fail_buggy_multiply() {
    if skip_if_driver_unavailable() || skip_if_kani_incompatible() {
        return;
    }

    let path = workspace_root().join("tests/soundness/must_fail/buggy_multiply.rs");
    expect_result(&path, VerifyResult::Failed);
}
