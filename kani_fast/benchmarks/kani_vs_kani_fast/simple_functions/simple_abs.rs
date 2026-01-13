// Benchmark: Absolute value
// Kani Fast: CHC proves instantly
// Kani: SAT encoding overhead

fn abs_safe(x: i32) -> i32 {
    // Avoid overflow on i32::MIN
    if x == i32::MIN {
        i32::MAX
    } else if x < 0 {
        -x
    } else {
        x
    }
}

fn simple_abs_proof() {
    let x: i32 = -5;
    let result = abs_safe(x);
    assert!(result == 5);
    assert!(result >= 0);
}

#[cfg(kani)]
mod verification {
    use super::*;

    #[kani::proof]
    fn verify_abs_non_negative() {
        let x: i32 = kani::any();
        let result = abs_safe(x);
        assert!(result >= 0);
    }
}
