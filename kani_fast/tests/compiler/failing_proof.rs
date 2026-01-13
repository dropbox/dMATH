// Test harness with a failing proof
// This file tests that the CHC verification correctly detects violations

/// This proof should FAIL - the assertion is false
fn false_assertion_proof() {
    let x = 5;
    let y = 10;
    // This assertion is false: 5 > 10
    assert!(x > y);
}

/// This proof should PASS - the assertion is true
fn true_assertion_proof() {
    let x = 10;
    let y = 5;
    // This assertion is true: 10 > 5
    assert!(x > y);
}

fn main() {}
