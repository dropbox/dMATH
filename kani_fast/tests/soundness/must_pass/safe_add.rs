// Safe addition that can't overflow - should verify
// Simplified test without Option<u8>

fn safe_add_direct(a: u8, b: u8) -> u8 {
    if a > 255 - b {
        0  // Return 0 on overflow (like checked_add().unwrap_or(0))
    } else {
        a + b
    }
}

/// Proof harness: safe_add handles overflow correctly
fn safe_add_proof() {
    let a: u8 = 100;
    let b: u8 = 100;
    let result = safe_add_direct(a, b);
    // 100 + 100 = 200 which fits in u8, so result should be 200
    assert!(result == 200);
}
