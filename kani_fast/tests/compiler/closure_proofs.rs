// Test CHC encoding for closures
// These tests verify that closures are properly inlined and verified
//
// NOTE: Currently, closure inlining works for closures with 0-1 parameters.
// Closures with 2+ parameters have a known limitation where the tuple of
// arguments is not properly unpacked during inlining.

/// Test closure capturing environment (no params) - should verify
fn closure_capture_proof() {
    let x = 10i32;
    let get_x = || x;
    assert!(get_x() == 10);
}

/// Test closure with multiple calls (single param) - should verify
fn closure_multiple_calls_proof() {
    let double = |x: i32| x * 2;
    let a = double(5);
    let b = double(10);
    assert!(a == 10);
    assert!(b == 20);
}

/// Test nested closure calls (single param) - should verify
fn closure_nested_proof() {
    let inc = |x: i32| x + 1;
    let result = inc(inc(inc(0)));
    assert!(result == 3);
}

/// Test closure modifying captured mutable reference - should verify
fn closure_modify_captured_proof() {
    let mut x = 5i32;
    {
        let mut add_to_x = |n: i32| x = x + n;
        add_to_x(10);
    }
    assert!(x == 15);
}

fn main() {}
