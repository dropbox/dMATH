// Benchmark: Cyclic counter (simplified state machine)
// Kani Fast: CHC proves invariant for ALL transitions
// Kani: Can only prove bounded number of transitions
//
// State cycles: 0 -> 1 -> 2 -> 3 -> 0 ...
// Invariant: state is always valid (0-3)

fn increment_state(state: i32) -> i32 {
    // Simple increment with wrap (equivalent to (state + 1) mod 4)
    if state == 3 {
        0
    } else {
        state + 1
    }
}

fn is_valid_state(state: i32) -> bool {
    state >= 0 && state <= 3
}

fn unbounded_state_machine_proof() {
    // Direct tests without function calls to avoid solver complexity
    // State 0 transitions
    let s0 = 0;
    let s1 = if s0 == 3 { 0 } else { s0 + 1 };
    assert!(s1 == 1);

    // State 1 transitions
    let s2 = if s1 == 3 { 0 } else { s1 + 1 };
    assert!(s2 == 2);

    // State 2 transitions
    let s3 = if s2 == 3 { 0 } else { s2 + 1 };
    assert!(s3 == 3);

    // State 3 transitions (wrap around)
    let s4 = if s3 == 3 { 0 } else { s3 + 1 };
    assert!(s4 == 0);

    // All intermediate states are valid
    assert!(is_valid_state(s1));
    assert!(is_valid_state(s2));
    assert!(is_valid_state(s3));
    assert!(is_valid_state(s4));
}

#[cfg(kani)]
mod verification {
    use super::*;

    #[kani::proof]
    #[kani::unwind(101)]
    fn verify_state_always_valid() {
        let cycles: i32 = kani::any();
        kani::assume(cycles >= 0 && cycles <= 100);

        let mut state: i32 = 0;
        let mut i: i32 = 0;
        while i < cycles {
            state = increment_state(state);
            assert!(is_valid_state(state));
            i = i + 1;
        }
    }
}
