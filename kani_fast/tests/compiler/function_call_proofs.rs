// Test CHC encoding for function calls
//
// NOTE: As of iteration #133, user-defined function calls are automatically inlined
// before CHC encoding. This allows the solver to reason about the full semantics
// of helper functions. See inline_test.rs for tests of automatic inlining.
//
// This file contains manually inlined tests that were created before automatic
// inlining was implemented. They still work and serve as regression tests.

/// Test with inlined computation (manual inlining) - should verify
fn inlined_add_proof() {
    let a = 3;
    let b = 5;
    // Manually inline: result = a + b
    let result = a + b;
    assert!(result == 8);
}

/// Test with inlined wrong assertion - should fail
fn inlined_wrong_proof() {
    let a = 3;
    let b = 5;
    let result = a + b;
    assert!(result == 10); // Wrong - should be 8
}

/// Test computation with multiple steps (manually inlined) - should verify
fn inlined_compute_proof() {
    let x = 5;
    // Manual inline of: doubled = x * 2; incremented = doubled + 1
    let doubled = x * 2;
    let incremented = doubled + 1;
    assert!(incremented == 11);
}

/// Test multiple computations (manually inlined) - should verify
fn inlined_multiple_proof() {
    let a = 1 + 2;  // inline add(1, 2)
    let b = 3 + 4;  // inline add(3, 4)
    let total = a + b;  // inline add(a, b)
    assert!(total == 10);
}

/// Test conditional computation (manually inlined) - should verify
fn inlined_conditional_proof() {
    let x = 5;
    // Manual inline of conditional function call
    let result = if x > 0 { x + 10 } else { 0 };
    assert!(result == 15);
}

/// Test chain of computations (manually inlined) - should verify
fn inlined_chain_proof() {
    let a = 42;  // inline get_constant()
    let b = a + 8;  // inline add(a, 8)
    assert!(b == 50);
}

fn main() {}
