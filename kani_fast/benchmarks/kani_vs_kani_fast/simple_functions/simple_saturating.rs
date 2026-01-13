// Benchmark: Saturating arithmetic
// Kani Fast: CHC proves instantly with intrinsic support
// Kani: SAT encoding overhead

fn saturating_add_i32(a: i32, b: i32) -> i32 {
    a.saturating_add(b)
}

fn saturating_sub_i32(a: i32, b: i32) -> i32 {
    a.saturating_sub(b)
}

fn simple_saturating_proof() {
    // Normal case
    let result = saturating_add_i32(5, 10);
    assert!(result == 15);

    // Saturation case
    let result = saturating_add_i32(i32::MAX, 1);
    assert!(result == i32::MAX);

    let result = saturating_sub_i32(i32::MIN, 1);
    assert!(result == i32::MIN);
}

#[cfg(kani)]
mod verification {
    use super::*;

    #[kani::proof]
    fn verify_saturating_add_no_overflow() {
        let a: i32 = kani::any();
        let b: i32 = kani::any();
        kani::assume(b >= 0);
        let result = saturating_add_i32(a, b);
        assert!(result >= a || result == i32::MAX);
    }

    #[kani::proof]
    fn verify_saturating_sub_no_underflow() {
        let a: i32 = kani::any();
        let b: i32 = kani::any();
        kani::assume(b >= 0);
        let result = saturating_sub_i32(a, b);
        assert!(result <= a || result == i32::MIN);
    }
}
