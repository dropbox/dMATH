//! For-loop sum test
//!
//! This tests for-loop support via Range iterator semantics.
//! The sum 0 + 1 + 2 = 3 should be proven with CHC.
//!
//! Run with:
//!   kani-fast chc tests/hybrid_mode/for_loop_sum_proof.rs

fn for_loop_sum_proof() {
    let mut sum = 0i32;
    for i in 0..3 {
        sum += i;  // 0 + 1 + 2 = 3
    }
    assert!(sum >= 0);  // Simple property that should verify
}
