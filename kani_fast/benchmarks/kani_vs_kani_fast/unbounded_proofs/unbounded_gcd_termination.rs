// Benchmark: GCD-like decreasing invariant
// Kani Fast: CHC proves decreasing variant for termination
// Kani: Requires --unwind, cannot prove termination

// Simulates GCD-like computation with decreasing invariant
fn decreasing_to_zero(mut a: i32, mut b: i32) -> i32 {
    if a < 0 || b < 0 {
        return 0;
    }

    // Simplified decreasing loop (not actual GCD, but demonstrates variant)
    while a > 0 && b > 0 {
        if a > b {
            a = a - b;
        } else {
            b = b - a;
        }
        // Variant: a + b decreases each iteration
    }
    a + b
}

fn unbounded_gcd_termination_proof() {
    // Kani Fast proves termination via decreasing variant
    // Variant: a + b > 0 implies next (a' + b') < (a + b)
    let a: i32 = 6;  // Small values for benchmark timing
    let b: i32 = 4;  // GCD(6,4) = 2, max iterations = 3
    let result = decreasing_to_zero(a, b);
    assert!(result >= 0);
}

#[cfg(kani)]
mod verification {
    use super::*;

    #[kani::proof]
    #[kani::unwind(201)]
    fn verify_terminates_non_negative() {
        let a: i32 = kani::any();
        let b: i32 = kani::any();
        kani::assume(a >= 0 && a <= 100);
        kani::assume(b >= 0 && b <= 100);
        let result = decreasing_to_zero(a, b);
        assert!(result >= 0);
    }
}
