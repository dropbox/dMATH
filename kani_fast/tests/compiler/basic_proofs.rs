// Test basic CHC verification
// Simple proofs that work with the current CHC encoding

/// Simple true assertion
fn always_true_proof() {
    let x = 5;
    assert!(x == 5);
}

/// Simple false assertion - should fail
fn always_false_proof() {
    let x = 5;
    assert!(x == 10); // This should fail
}

/// Arithmetic that should pass
fn arithmetic_correct_proof() {
    let a = 10;
    let b = 20;
    let c = a + b;
    assert!(c == 30);
}

/// Arithmetic that should fail
fn arithmetic_wrong_proof() {
    let a = 10;
    let b = 20;
    let c = a + b;
    assert!(c == 99); // Wrong - should fail
}

/// Boolean logic proof
fn boolean_logic_proof() {
    let x = true;
    let y = false;
    let result = x && !y;
    assert!(result);
}

fn main() {}
