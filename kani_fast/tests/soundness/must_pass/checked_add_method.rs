// This function uses the checked_add method which returns Option<u8>.
// The function safely handles potential overflow by returning 0 on None.
// CHC verification must succeed.

fn safe_add_with_checked(a: u8, b: u8) -> u8 {
    match a.checked_add(b) {
        Some(v) => v,
        None => 0,
    }
}

/// Proof harness: checked_add method handles overflow safely
fn checked_add_method_proof() {
    let a: u8 = 100;
    let b: u8 = 100;
    let result = safe_add_with_checked(a, b);
    // 100 + 100 = 200 which fits, so result should be 200
    assert!(result == 200);
}
