// Test harness for loop verification
// Tests that CHC encoding correctly handles loops and loop invariants

/// Simple loop that increments a counter - should verify
/// The loop invariant is: counter >= 0 && counter <= 10
fn simple_counter_loop_proof() {
    let mut counter = 0;
    while counter < 10 {
        counter += 1;
    }
    // After the loop, counter should be exactly 10
    assert!(counter == 10);
}

/// Loop that accumulates a sum - should verify
/// Sum of 1..=5 should be 15
fn sum_loop_proof() {
    let mut sum = 0;
    let mut i = 1;
    while i <= 5 {
        sum += i;
        i += 1;
    }
    assert!(sum == 15);
}

/// Loop that should fail - incorrect assertion
fn incorrect_loop_proof() {
    let mut x = 0;
    while x < 5 {
        x += 1;
    }
    // Wrong assertion: x is 5, not 10
    assert!(x == 10);
}

/// Loop with early exit - should verify
fn early_exit_loop_proof() {
    let mut found = false;
    let mut i = 0;
    while i < 100 {
        if i == 42 {
            found = true;
            break;
        }
        i += 1;
    }
    // We found 42 somewhere between 0 and 99
    assert!(found || i == 100);
}

fn main() {}
