//! Simple loop test for hybrid mode
//!
//! This should verify with CHC (unbounded proof).
//! No Kani fallback should be needed.
//!
//! Run with:
//!   kani-fast chc tests/hybrid_mode/simple_loop_proof.rs --hybrid

fn simple_loop_proof() {
    let mut x: i32 = 0;
    let mut i: i32 = 0;
    while i < 10 {
        x = x + 1;
        i = i + 1;
    }
    assert!(x >= 0);
}
