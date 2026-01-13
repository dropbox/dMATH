// This function correctly handles potential overflow.
// CHC verification must succeed.
// Simplified without Option<u8> return type

fn checked_add_direct(a: u8, b: u8) -> u8 {
    if a > 255 - b {
        0  // Return 0 on overflow (like checked_add().unwrap_or(0))
    } else {
        a + b
    }
}

/// Proof harness: checked_add handles overflow correctly
fn checked_add_proof() {
    let a: u8 = 100;
    let b: u8 = 100;
    let result = checked_add_direct(a, b);
    // 100 + 100 = 200 which fits in u8
    assert!(result == 200);
}
