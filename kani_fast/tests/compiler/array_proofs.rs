// Test CHC encoding for arrays
// These tests verify that the CHC encoding correctly handles fixed-size arrays
//
// Note: Array verification can be expensive for CHC solvers due to quantified
// array operations. Tests are kept simple to ensure reasonable verification times.

/// Test basic array creation and indexing - should verify
fn array_creation_proof() {
    let arr: [i32; 3] = [10, 20, 30];
    assert!(arr[0] == 10);
}

/// Test array with wrong assertion - should fail
fn array_wrong_assertion_proof() {
    let arr: [i32; 2] = [5, 10];
    assert!(arr[0] == 10); // Wrong - arr[0] is 5, not 10
}

/// Test array second element - should verify
fn array_second_element_proof() {
    let arr: [i32; 2] = [42, 84];
    assert!(arr[1] == 84);
}

fn main() {}
