// Benchmark: Clamp value to range
// Kani Fast: CHC proves instantly
// Kani: SAT encoding overhead

fn clamp(value: i32, min: i32, max: i32) -> i32 {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

fn simple_clamp_proof() {
    let value: i32 = 15;
    let min: i32 = 0;
    let max: i32 = 10;
    let result = clamp(value, min, max);
    assert!(result == 10);
    assert!(result >= min);
    assert!(result <= max);
}

#[cfg(kani)]
mod verification {
    use super::*;

    #[kani::proof]
    fn verify_clamp_in_range() {
        let value: i32 = kani::any();
        let min: i32 = kani::any();
        let max: i32 = kani::any();
        kani::assume(min <= max);

        let result = clamp(value, min, max);
        assert!(result >= min && result <= max);
    }
}
