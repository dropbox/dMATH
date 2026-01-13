// Benchmark: Minimum of three values
// Kani Fast: CHC proves instantly
// Kani: SAT encoding overhead

fn min3(a: i32, b: i32, c: i32) -> i32 {
    let min_ab = if a < b { a } else { b };
    if min_ab < c { min_ab } else { c }
}

fn simple_min3_proof() {
    let a: i32 = 5;
    let b: i32 = 3;
    let c: i32 = 7;
    let m = min3(a, b, c);
    assert!(m == 3);
}

#[cfg(kani)]
mod verification {
    use super::*;

    #[kani::proof]
    fn verify_min3_is_smallest() {
        let a: i32 = kani::any();
        let b: i32 = kani::any();
        let c: i32 = kani::any();
        let m = min3(a, b, c);
        assert!(m <= a && m <= b && m <= c);
    }

    #[kani::proof]
    fn verify_min3_is_one_of_inputs() {
        let a: i32 = kani::any();
        let b: i32 = kani::any();
        let c: i32 = kani::any();
        let m = min3(a, b, c);
        assert!(m == a || m == b || m == c);
    }
}
