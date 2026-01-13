// Benchmark: Sign function
// Kani Fast: CHC proves instantly
// Kani: SAT encoding overhead

fn sign(x: i32) -> i32 {
    if x > 0 {
        1
    } else if x < 0 {
        -1
    } else {
        0
    }
}

fn simple_sign_proof() {
    assert!(sign(5) == 1);
    assert!(sign(-5) == -1);
    assert!(sign(0) == 0);
}

#[cfg(kani)]
mod verification {
    use super::*;

    #[kani::proof]
    fn verify_sign_values() {
        let x: i32 = kani::any();
        let s = sign(x);
        assert!(s == -1 || s == 0 || s == 1);
    }

    #[kani::proof]
    fn verify_sign_semantics() {
        let x: i32 = kani::any();
        let s = sign(x);
        if x > 0 { assert!(s == 1); }
        if x < 0 { assert!(s == -1); }
        if x == 0 { assert!(s == 0); }
    }
}
