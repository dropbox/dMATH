# shellcheck shell=bash
# Test definitions for category: regression
# Sourced by test_driver.sh; relies on shared helpers like run_test.

# Test 1: Simple arithmetic (WITHOUT Kani library)
# Purpose: Tests simple code without extern crate kani (regression: #204, #216, #226, #308).
# Category: regression
cat > "$TMPDIR/simple.rs" << 'EOF'
fn simple_proof() {
    let x = 5i32;
    let y = 3i32;
    assert!(x + y == 8);
}
EOF
run_test "simple_arithmetic" "$TMPDIR/simple.rs" "pass"
