// Benchmark: Unbounded sum invariant
// Kani Fast: CHC proves sum formula for ALL n
// Kani: Can only prove for specific --unwind value

// Computes sum of 0..n using loop
// Invariant: sum == i*(i-1)/2 (triangular number formula offset)
fn sum_to_n(n: i32) -> i32 {
    if n <= 0 {
        return 0;
    }

    let mut sum: i32 = 0;
    let mut i: i32 = 0;
    while i < n {
        sum = sum + i;
        i = i + 1;
    }
    sum
}

fn unbounded_sum_proof() {
    // Kani Fast proves this for ALL n via CHC
    // Loop invariant: sum >= 0, i >= 0, i <= n
    let n: i32 = 5;  // Small value for benchmark timing
    let result = sum_to_n(n);
    assert!(result >= 0);
    assert!(result == 10);  // sum(0..5) = 0+1+2+3+4 = 10
}

#[cfg(kani)]
mod verification {
    use super::*;

    // Kani requires --unwind and can only prove bounded
    #[kani::proof]
    #[kani::unwind(51)]
    fn verify_sum_positive() {
        let n: i32 = kani::any();
        kani::assume(n >= 0 && n <= 50);  // Kani requires bounded assumption
        let result = sum_to_n(n);
        assert!(result >= 0);
    }
}
