// Simple test harness for kani-fast-compiler
// This file tests basic CHC verification through the rustc driver

/// Proof harness: verify simple arithmetic (no function calls)
/// Named with _proof suffix for automatic detection
fn simple_add_proof() {
    let x = 5;
    let y = 0;
    let result = x + y;
    assert_eq!(result, x);
}

/// Another proof harness: verify basic arithmetic
fn arithmetic_proof() {
    let a = 3;
    let b = 4;
    let c = a + b;
    assert_eq!(c, 7);
}

fn main() {}
