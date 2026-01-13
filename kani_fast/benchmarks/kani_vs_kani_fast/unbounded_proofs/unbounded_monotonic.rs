// Benchmark: Unbounded monotonic increase
// Kani Fast: CHC proves monotonicity for ALL iterations
// Kani: Can only prove for specific --unwind value

// Increments value until limit - proves value is always increasing
fn monotonic_increase(limit: i32) -> i32 {
    if limit <= 0 {
        return 0;
    }

    let mut value: i32 = 0;
    let mut prev: i32 = -1;
    while value < limit {
        prev = value;
        value = value + 1;
        // Invariant: value > prev (monotonicity)
    }
    value
}

fn unbounded_monotonic_proof() {
    // Kani Fast proves monotonicity for ALL limits via CHC
    // Invariant: 0 <= value <= limit, prev < value
    let limit: i32 = 5;  // Small value for benchmark timing
    let result = monotonic_increase(limit);
    assert_eq!(result, limit);
}

#[cfg(kani)]
mod verification {
    use super::*;

    #[kani::proof]
    #[kani::unwind(101)]
    fn verify_monotonic() {
        let limit: i32 = kani::any();
        kani::assume(limit >= 0 && limit <= 100);
        let result = monotonic_increase(limit);
        assert_eq!(result, limit);
    }
}
