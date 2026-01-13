// REGRESSION TEST: Verification without Kani library
//
// This test ensures that kani-fast-driver can verify simple Rust code
// WITHOUT requiring `extern crate kani;` in the source.
//
// CRITICAL: This bug has been re-introduced 5 times (#204, #216, #226, #308)
// via strict Kani library enforcement. DO NOT add any code that requires
// the Kani library to be present.
//
// See: apply_kani_transforms() in codegen_chc/mod.rs for the graceful handling.

/// Simple proof harness - no Kani intrinsics
fn simple_no_kani_proof() {
    let x = 5i32;
    let y = 3i32;
    assert!(x + y == 8);
}

/// Another proof without Kani - tests multiple assertions
fn multiple_assertions_no_kani_proof() {
    let a = 10i32;
    let b = 5i32;

    // All of these should pass without Kani library
    assert!(a > b);
    assert!(a - b == 5);
    assert!(a + b == 15);
}

/// Proof with conditional logic - no Kani
fn conditional_no_kani_proof() {
    let x = 7i32;
    let result = if x > 5 { x - 5 } else { 0 };
    assert!(result == 2);
}

fn main() {}
