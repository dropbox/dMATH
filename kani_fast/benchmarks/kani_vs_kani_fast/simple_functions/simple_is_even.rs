// Benchmark: Even/odd check
// Kani Fast: CHC proves instantly (bitwise via bitvec mode)
// Kani: SAT encoding overhead

fn is_even(x: i32) -> bool {
    (x & 1) == 0
}

fn is_odd(x: i32) -> bool {
    (x & 1) == 1
}

fn simple_is_even_proof() {
    assert!(is_even(4));
    assert!(is_odd(5));
    assert!(!is_even(3));
    assert!(!is_odd(4));
}

#[cfg(kani)]
mod verification {
    use super::*;

    #[kani::proof]
    fn verify_even_odd_exclusive() {
        let x: i32 = kani::any();
        assert!(is_even(x) != is_odd(x));
    }
}
