// Test CHC encoding with local helper functions
//
// This tests whether the MIR we receive has inlined functions or not.
// We want to verify properties about functions that call other functions.

/// A simple helper function
#[inline(always)]
fn add(a: i32, b: i32) -> i32 {
    a + b
}

/// Helper that doubles a value
#[inline(always)]
fn double(x: i32) -> i32 {
    x * 2
}

/// Test calling a helper function - should verify if inlining works
fn inline_add_proof() {
    let result = add(3, 5);
    assert!(result == 8);
}

/// Test calling multiple helpers - should verify if inlining works
fn inline_chain_proof() {
    let x = 5;
    let doubled = double(x);
    let result = add(doubled, 1);
    assert!(result == 11);
}

/// Test with wrong assertion - should fail
fn inline_wrong_proof() {
    let result = add(3, 5);
    assert!(result == 10); // Wrong - should be 8
}

fn main() {}
