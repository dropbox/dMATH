// This function has a bug: addition can overflow.
// CHC verification must detect this and report a violation.

fn overflow_add(a: u32, b: u32) -> u32 {
    a + b // Can overflow if a + b > u32::MAX
}

/// Proof harness: overflow_add can overflow
fn overflow_add_proof() {
    let a: u32 = 4294967295; // u32::MAX
    let b: u32 = 1;
    // This should fail: MAX + 1 overflows
    let result = overflow_add(a, b);
    let overflow = (a as u64) + (b as u64) > u32::MAX as u64;
    assert!(!overflow);
    let _ = result;
}
