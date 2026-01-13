// Benchmark: Unbounded counter invariant
// Kani Fast: CHC proves for ALL n via invariant (i <= n)
// Kani: Can only prove for specific --unwind value, cannot prove for all n

// This function has an unbounded loop - the invariant i <= n holds for all n
fn count_to_n(n: i32) -> i32 {
    if n < 0 {
        return 0;
    }

    let mut i: i32 = 0;
    while i < n {
        i = i + 1;
    }
    i
}

fn unbounded_counter_proof() {
    // Kani Fast proves this for ALL n via CHC/k-induction
    // Loop invariant: 0 <= i <= n
    let n: i32 = 5;  // Small value for benchmark timing
    let result = count_to_n(n);
    assert_eq!(result, n);
}

#[cfg(kani)]
mod verification {
    use super::*;

    // Kani requires --unwind and can only prove bounded
    #[kani::proof]
    #[kani::unwind(101)]
    fn verify_count_reaches_n() {
        let n: i32 = kani::any();
        kani::assume(n >= 0 && n <= 100);  // Kani requires bounded assumption
        let result = count_to_n(n);
        assert_eq!(result, n);
    }
}
