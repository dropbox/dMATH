// This function has a bug: multiplication can overflow.
// CHC verification must detect this and report a violation.

fn buggy_multiply(a: u8, b: u8) -> u8 {
    a * b // Can overflow!
}

/// Proof harness: buggy_multiply can overflow with arbitrary inputs
fn buggy_multiply_overflow_proof() {
    let a: u8 = 20;
    let b: u8 = 20;
    // This should fail: 20 * 20 = 400, which overflows u8
    let result = buggy_multiply(a, b);
    let overflow = (a as u16) * (b as u16) > u8::MAX as u16;
    // Fail if multiplication would overflow or wrap
    assert!(!overflow);
    let _ = result;
}
