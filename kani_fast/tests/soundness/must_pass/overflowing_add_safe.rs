// Soundness test: safe overflow handling
// This function correctly handles the overflow case
// Expected: VERIFIED
// Simplified without tuple return type

fn safe_add_with_overflow_check(a: u8, b: u8) -> u8 {
    // Manually check for overflow before adding
    if a > 255 - b {
        // Return 0 on overflow (safe default)
        0
    } else {
        a + b
    }
}

/// Proof harness: overflow check makes addition safe
fn overflowing_add_safe_proof() {
    let a: u8 = 100;
    let b: u8 = 100;
    let result = safe_add_with_overflow_check(a, b);
    // 100 + 100 = 200 which fits, so result should be 200
    assert!(result == 200);
}
