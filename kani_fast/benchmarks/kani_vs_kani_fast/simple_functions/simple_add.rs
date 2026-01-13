// Benchmark: Simple addition
// Kani Fast: CHC proves instantly (~50ms)
// Kani: CBMC SAT encoding (~5-10s startup)

fn add(a: i32, b: i32) -> i32 {
    a.wrapping_add(b)
}

// Kani Fast driver recognizes functions ending in _proof
fn simple_add_proof() {
    let a: i32 = 5;
    let b: i32 = 10;
    assert!(add(a, b) == add(b, a));  // Commutativity
    assert!(add(a, b) == 15);
}

// Kani harness (for cargo kani)
#[cfg(kani)]
mod verification {
    use super::*;

    #[kani::proof]
    fn verify_add_commutative() {
        let a: i32 = kani::any();
        let b: i32 = kani::any();
        assert!(add(a, b) == add(b, a));
    }
}
