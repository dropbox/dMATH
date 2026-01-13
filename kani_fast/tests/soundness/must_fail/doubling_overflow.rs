// Simple doubling that can overflow - should fail verification
// No loops to avoid saturating_add optimization

fn doubling_overflow(n: u8) -> u8 {
    // Double the input three times
    // For n > 31, this overflows: 32 * 8 = 256 > 255
    let a = n + n; // Can overflow if n > 127
    let b = a + a; // Can overflow if n > 63
    let c = b + b; // Can overflow if n > 31
    c
}

/// Proof harness: doubling_overflow can overflow
fn doubling_overflow_proof() {
    let n: u8 = 40;
    // 40 * 8 = 320 > 255, so this overflows
    let result = doubling_overflow(n);
    let overflow = (n as u16) * 8 > u8::MAX as u16;
    assert!(!overflow);
    let _ = result;
}
