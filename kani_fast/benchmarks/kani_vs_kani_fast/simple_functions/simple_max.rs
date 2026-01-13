// Benchmark: Simple max function
// Kani Fast: CHC proves instantly
// Kani: SAT encoding overhead

fn max(a: i32, b: i32) -> i32 {
    if a > b { a } else { b }
}

fn simple_max_proof() {
    let a: i32 = 5;
    let b: i32 = 10;
    let m = max(a, b);
    assert!(m == 10);
    assert!(m >= a);
    assert!(m >= b);
}

#[cfg(kani)]
mod verification {
    use super::*;

    #[kani::proof]
    fn verify_max_returns_larger() {
        let a: i32 = kani::any();
        let b: i32 = kani::any();
        let m = max(a, b);
        assert!(m >= a && m >= b);
    }

    #[kani::proof]
    fn verify_max_is_one_of_inputs() {
        let a: i32 = kani::any();
        let b: i32 = kani::any();
        let m = max(a, b);
        assert!(m == a || m == b);
    }
}
