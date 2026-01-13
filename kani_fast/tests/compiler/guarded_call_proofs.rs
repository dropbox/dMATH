// Regression tests for path-condition tracking at call sites
// Functions ending with `_proof` are verified via `kani-fast chc --all-proofs`

fn requires_positive(x: i32) {
    assert!(x > 0);
}

/// Guarded by branch condition: call should be allowed only when x > 0
fn guarded_call_proof(x: i32) {
    if x > 0 {
        requires_positive(x);
        assert!(x > 0);
    }
}

/// No guard: x could be 0 or negative, so this proof should fail
fn unguarded_call_proof(x: i32) {
    requires_positive(x);
    assert!(x > 0);
}

/// Call inside a loop body with an inner guard
fn loop_guarded_call_proof() {
    let mut i = 0;
    while i < 3 {
        if i > 0 {
            requires_positive(i);
            assert!(i > 0);
        }
        i += 1;
    }
}

/// Loop call without guard: fails on the first iteration (i = 0)
fn loop_unguarded_call_proof() {
    let mut i = 0;
    while i < 3 {
        requires_positive(i);
        assert!(i > 0);
        i += 1;
    }
}

fn main() {}
