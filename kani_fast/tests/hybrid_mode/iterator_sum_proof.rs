//! Test case for hybrid mode
//!
//! This test uses iterators which CHC cannot handle (returns unknown)
//! but Kani BMC can verify with bounded unrolling.
//!
//! Run with:
//!   kani-fast chc tests/hybrid_mode/iterator_sum_proof.rs --hybrid

fn iterator_sum_proof() {
    // Iterators are uninterpreted functions in CHC - this will return unknown
    // But Kani can verify this with BMC
    let arr = [1, 2, 3, 4, 5];
    let sum: i32 = arr.iter().sum();
    assert!(sum == 15);
}
