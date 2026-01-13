// Benchmark: Swap values via tuple
// Kani Fast: CHC proves instantly
// Kani: SAT encoding overhead

fn swap(a: i32, b: i32) -> (i32, i32) {
    (b, a)
}

fn simple_swap_proof() {
    let a: i32 = 5;
    let b: i32 = 10;
    let (x, y) = swap(a, b);
    assert!(x == 10);
    assert!(y == 5);
}

#[cfg(kani)]
mod verification {
    use super::*;

    #[kani::proof]
    fn verify_swap_correctness() {
        let a: i32 = kani::any();
        let b: i32 = kani::any();
        let (x, y) = swap(a, b);
        assert!(x == b && y == a);
    }

    #[kani::proof]
    fn verify_swap_involution() {
        let a: i32 = kani::any();
        let b: i32 = kani::any();
        let (x, y) = swap(a, b);
        let (p, q) = swap(x, y);
        assert!(p == a && q == b);
    }
}
