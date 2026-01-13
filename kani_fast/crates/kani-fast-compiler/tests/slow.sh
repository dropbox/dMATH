# shellcheck shell=bash
# Test definitions for category: slow
# Sourced by test_driver.sh; relies on shared helpers like run_test.

# Test 170c: For loop with summation (slow - complex invariant)
# Purpose: Tests summation loop requiring complex invariant synthesis.
# Category: slow
# Note: Requires invariant sum = i*(i-1)/2, may timeout.
cat > "$TMPDIR/for_loop_sum.rs" << 'ENDOFFILE'
fn for_loop_sum_proof() {
    let mut sum = 0i32;
    for i in 0..5 {
        sum += i;  // 0 + 1 + 2 + 3 + 4 = 10
    }
    assert!(sum == 10);
}
ENDOFFILE
run_test "for_loop_sum" "$TMPDIR/for_loop_sum.rs" "fail" "slow"  # Timeout: complex invariant




# Test 249: GCD algorithm (simple case)
# Purpose: Tests GCD via subtraction method (known to timeout).
# Category: slow
# Note: Loop with 20 iterations causes CHC solver timeout.
cat > "$TMPDIR/gcd.rs" << 'ENDOFFILE'
fn gcd(mut a: i32, mut b: i32) -> i32 {
    // Simple subtraction-based GCD (limited iterations)
    let mut i = 0;
    while a != b && i < 20 {
        if a > b {
            a = a - b;
        } else {
            b = b - a;
        }
        i += 1;
    }
    a
}

fn gcd_proof() {
    assert!(gcd(12, 8) == 4);
    assert!(gcd(15, 10) == 5);
    assert!(gcd(6, 6) == 6);
}
ENDOFFILE
run_test "gcd" "$TMPDIR/gcd.rs" "fail" "slow"  # Expected timeout


# Test 264: Sum of squares (linear arithmetic only)
# Purpose: Tests accumulator pattern (known to timeout).
# Category: slow
# Note: Loop with array indexing causes CHC solver timeout.
cat > "$TMPDIR/sum_squares.rs" << 'ENDOFFILE'
fn sum_squares_proof() {
    // Pre-computed squares to avoid non-linear arithmetic
    let squares = [0, 1, 4, 9, 16, 25];
    let mut sum = 0i32;
    let mut i = 0usize;
    while i < 6 {
        sum += squares[i] as i32;
        i += 1;
    }
    assert!(sum == 55);  // 0 + 1 + 4 + 9 + 16 + 25 = 55
}
ENDOFFILE
run_test "sum_squares" "$TMPDIR/sum_squares.rs" "fail" "slow"  # Expected timeout


# Test 328b: Checked multiplication (non-linear)
# Purpose: Tests checked_mul with non-linear arithmetic (slow due to multiplication).
# Category: slow
# Note: Non-linear arithmetic causes timeout in fast mode, but correctly constrained.
cat > "$TMPDIR/checked_mul_nonlinear.rs" << 'ENDOFFILE'
fn checked_mul_nonlinear_proof() {
    let a: i32 = 7;
    let b: i32 = 6;
    if let Some(result) = a.checked_mul(b) {
        // Non-linear arithmetic is slow but correctly constrained
        assert!(result == 42);
    }
}
ENDOFFILE
run_test "checked_mul_nonlinear" "$TMPDIR/checked_mul_nonlinear.rs" "fail" "slow"  # Non-linear: times out in fast mode


# Test 76: Nested tuple destructuring (from features.sh)
# Purpose: Tests nested tuple pattern destructuring ((a,b),(c,d)).
# Category: slow
# Note: Semantically correct (fixed in #567) but CHC has 21 variables, times out in fast mode.
cat > "$TMPDIR/nested_tuple.rs" << 'ENDOFFILE'
fn nested_tuple_proof() {
    let nested = ((1i32, 2i32), (3i32, 4i32));
    let ((a, b), (c, d)) = nested;
    assert!(a == 1);
    assert!(b == 2);
    assert!(c == 3);
    assert!(d == 4);
    assert!(a + b + c + d == 10);
}
ENDOFFILE
run_test "nested_tuple_destructuring" "$TMPDIR/nested_tuple.rs" "fail" "slow"  # Large CHC: times out in fast mode


# Test 86: Array repeat (from features.sh, fixed in #565)
# Purpose: Tests array repeat syntax [val; n], now properly supported via SMT constant arrays.
# Category: slow
# Note: Fixed in #565 but very slow in debug mode (>60s). Passes in release mode ~15s.
cat > "$TMPDIR/array_repeat.rs" << 'ENDOFFILE'
fn array_repeat_proof() {
    let fives: [i32; 3] = [5; 3];
    assert!(fives[0] == 5);
    assert!(fives[1] == 5);
    assert!(fives[2] == 5);
}
ENDOFFILE
run_test "array_repeat" "$TMPDIR/array_repeat.rs" "fail" "slow"  # Very slow in debug mode


# Test 27: Multiple assertions in sequence (from features.sh)
# Purpose: Tests that multiple sequential assertions are all verified.
# Category: slow
# Note: 7 assertions cause PDR solver to reach iteration limit (~30s).
cat > "$TMPDIR/multi_assert.rs" << 'EOF'
fn multi_assert_proof() {
    let x = 10i32;
    assert!(x > 0);
    assert!(x < 100);
    assert!(x == 10);
    assert!(x >= 10);
    assert!(x <= 10);

    let y = x + 5;
    assert!(y == 15);
    assert!(y > x);
}
EOF
run_test "multiple_assertions" "$TMPDIR/multi_assert.rs" "fail" "slow"  # PDR limit: many assertions


# Test 195: assert_ne! macro (from features.sh)
# Purpose: Tests assert_ne! macro for inequality checks.
# Category: slow
# Note: assert_ne! macro expansion causes PDR solver timeout (~38s).
cat > "$TMPDIR/assert_ne.rs" << 'ENDOFFILE'
fn assert_ne_proof() {
    let x = 5i32;
    let y = 10i32;
    assert_ne!(x, y);  // x != y should pass
    assert!(x + y == 15);
}
ENDOFFILE
run_test "assert_ne" "$TMPDIR/assert_ne.rs" "fail" "slow"  # PDR limit: macro expansion


# Test 136: Move closure (from features.sh)
# Purpose: Tests move closures (limitation: return value unconstrained).
# Category: slow
# Note: Closure call causes complex CHC encoding, PDR returns unknown.
cat > "$TMPDIR/move_closure.rs" << 'ENDOFFILE'
fn move_closure_proof() {
    let x = 42i32;
    let get_x = move || x;  // Move captures x

    // Can't verify closure return, just test that it compiles
    let _result = get_x();
    assert!(x == 42);  // Original variable still verifiable
}
ENDOFFILE
run_test "move_closure" "$TMPDIR/move_closure.rs" "fail" "slow"  # PDR limit: closure complexity


# Test 6: Closure with capture (from features.sh)
# Purpose: Tests closures that capture variables from their environment.
# Category: slow
# Note: Closure capture causes complex CHC encoding.
cat > "$TMPDIR/closure_capture.rs" << 'EOF'
fn closure_capture_proof() {
    let base = 100i32;
    let add_base = |x: i32| x + base;
    let result = add_base(50);
    assert!(result == 150);
}
EOF
run_test "closure_capture" "$TMPDIR/closure_capture.rs" "fail" "slow"  # Closure: complex CHC


# Test 7: Count leading zeros - manual implementation (from features.sh)
# Purpose: Tests counting leading zero bits via multiple comparisons.
# Category: slow
# Note: Multiple if-branch conditions cause Z4 PDR timeout.
cat > "$TMPDIR/clz_simple.rs" << 'ENDOFFILE'
fn clz_simple_proof() {
    let x = 4i32;  // Binary: 0000...0100

    // Check if bit 2 is set (value 4 = 2^2)
    let bit2 = (x >> 2) & 1;
    assert!(bit2 == 1);

    // Count how many high bits are zero for small value
    let mut count = 0i32;
    if x < 128 { count += 25; }  // Skip to relevant bits
    if x < 64 { count += 1; }
    if x < 32 { count += 1; }
    if x < 16 { count += 1; }
    if x < 8 { count += 1; }

    assert!(count == 29);  // 32 - 3 = 29 leading zeros for value 4
}
ENDOFFILE
run_test "clz_simple" "$TMPDIR/clz_simple.rs" "fail" "slow"  # Multiple if branches cause timeout


# Test 8: Count trailing zeros - manual implementation (from features.sh)
# Purpose: Tests counting trailing zero bits via nested if-else chain.
# Category: slow
# Note: Nested if-else chain causes Z4 PDR timeout.
cat > "$TMPDIR/ctz_simple.rs" << 'ENDOFFILE'
fn ctz_simple_proof() {
    let x = 24i32;  // Binary: 11000

    // Count trailing zeros for lower 5 bits
    let count = if (x & 1) != 0 {
        0
    } else if (x & 2) != 0 {
        1
    } else if (x & 4) != 0 {
        2
    } else if (x & 8) != 0 {
        3
    } else if (x & 16) != 0 {
        4
    } else {
        5
    };

    assert!(count == 3);  // Three trailing zeros
}
ENDOFFILE
run_test "ctz_simple" "$TMPDIR/ctz_simple.rs" "fail" "slow"  # Nested if-else causes timeout
