# shellcheck shell=bash
# Test definitions for category: limitation
# Sourced by test_driver.sh; relies on shared helpers like run_test.
#
# Tests in this file document known limitations of CHC verification:
# - "fail": The limitation prevents verification (returns unsat/unknown)
# - "pass": Code compiles and runs but values are unconstrained (incomplete modeling)

# Test 158: Array element mutation - FIXED in #564
# Previously was a limitation but array index writes now properly encode using SMT store.
# MOVED to features.sh


# Test 169: Nested enum match soundness (known limitation)
# Purpose: Tests nested enums (known to produce CHC unsatisfiable).
# Category: limitation
# Note: Nested enum discriminant tracking is incomplete.
cat > "$TMPDIR/nested_enum_match_soundness.rs" << 'ENDOFFILE'
enum Outer {
    A(Inner),
    B,
}

enum Inner {
    X(i32),
    Y,
}

fn nested_enum_match_soundness_proof() {
    let _o = Outer::A(Inner::X(42));
    assert!(false);  // Expected to fail
}
ENDOFFILE
run_test "nested_enum_match_soundness" "$TMPDIR/nested_enum_match_soundness.rs" "fail" "limitation"


# Test 171: Non-linear arithmetic (known limitation)
# Purpose: Tests variable*variable multiplication (known to timeout).
# Category: limitation
# Note: Spacer struggles with non-linear arithmetic, returns unknown.
cat > "$TMPDIR/nonlinear_arith.rs" << 'ENDOFFILE'
fn nonlinear_arith_proof() {
    let n = 3i32;
    let m = 4i32;
    let product = n * m;  // Non-linear: variable * variable
    assert!(product == 12);
}
ENDOFFILE
run_test "nonlinear_arithmetic" "$TMPDIR/nonlinear_arith.rs" "fail" "limitation"  # Expected failure: non-linear arithmetic


# Test 173: Raw pointer creation (known limitation)
# Purpose: Tests raw pointer creation (is_null method unsupported).
# Category: limitation
# Note: Pointer created but is_null is a method call.
cat > "$TMPDIR/raw_pointer.rs" << 'ENDOFFILE'
fn raw_pointer_proof() {
    let x = 42i32;
    let ptr = &x as *const i32;
    // Can't dereference safely, but we can check pointer is non-null
    assert!(!ptr.is_null());
}
ENDOFFILE
run_test "raw_pointer" "$TMPDIR/raw_pointer.rs" "fail" "limitation"  # Expected failure: is_null is method call


# Test 184: Recursion (known limitation)
# Purpose: Tests recursive functions (modeled as uninterpreted).
# Category: limitation
# Note: Recursive calls return unconstrained values.
cat > "$TMPDIR/recursion.rs" << 'ENDOFFILE'
fn factorial(n: i32) -> i32 {
    if n <= 1 {
        1
    } else {
        n * factorial(n - 1)
    }
}

fn recursion_proof() {
    let f = factorial(5);
    assert!(f == 120);  // 5! = 120
}
ENDOFFILE
run_test "recursion" "$TMPDIR/recursion.rs" "fail" "limitation"  # Expected failure: recursion unsupported


# Test 187: String literal length (known limitation)
# Purpose: Tests str::len() method (unsupported - uninterpreted).
# Category: limitation
cat > "$TMPDIR/string_literal.rs" << 'ENDOFFILE'
fn string_literal_proof() {
    let s = "hello";
    assert!(s.len() == 5);
}
ENDOFFILE
run_test "string_literal" "$TMPDIR/string_literal.rs" "fail" "limitation"  # Expected failure: method call unsupported


# Test 327: Wrapping multiplication
# Purpose: Tests wrapping_mul intrinsic - expected timeout (non-linear).
# Category: limitation
cat > "$TMPDIR/wrapping_mul.rs" << 'ENDOFFILE'
fn wrapping_mul_proof() {
    let a: i32 = 7;
    let b: i32 = 6;
    let result = a.wrapping_mul(b);
    assert!(result == 42);
}
ENDOFFILE
run_test "wrapping_mul" "$TMPDIR/wrapping_mul.rs" "fail" "limitation"


# Test 327b: Wrapping multiplication soundness
# Purpose: Tests wrapping_mul soundness - expected failure (non-linear).
# Category: limitation
cat > "$TMPDIR/wrapping_mul_soundness.rs" << 'ENDOFFILE'
fn wrapping_mul_soundness_proof() {
    let a: i32 = 7;
    let b: i32 = 6;
    let result = a.wrapping_mul(b);
    // Non-linear arithmetic - expected failure
    assert!(result == 999);
}
ENDOFFILE
run_test "wrapping_mul_soundness" "$TMPDIR/wrapping_mul_soundness.rs" "fail" "limitation"


# Test 328: Checked multiplication
# Purpose: Tests checked_mul intrinsic - expected failure (non-linear).
# Category: limitation
cat > "$TMPDIR/checked_mul.rs" << 'ENDOFFILE'
fn checked_mul_proof() {
    let a: i32 = 7;
    let b: i32 = 6;
    let result = a.checked_mul(b);
    // Non-linear arithmetic - expected failure
    assert!(result.is_some());
}
ENDOFFILE
run_test "checked_mul" "$TMPDIR/checked_mul.rs" "fail" "limitation"


# Test 329: Overflowing multiplication
# Purpose: Tests overflowing_mul intrinsic - tuple ignored (unconstrained).
# Category: limitation
cat > "$TMPDIR/overflowing_mul.rs" << 'ENDOFFILE'
fn overflowing_mul_proof() {
    let a: i32 = 7;
    let b: i32 = 6;
    let (result, overflow) = a.overflowing_mul(b);
    // Basic check - result exists (unconstrained but non-failing)
    let _ = result;
    let _ = overflow;
}
ENDOFFILE
run_test "overflowing_mul" "$TMPDIR/overflowing_mul.rs" "pass" "limitation"


# Test 329b: Overflowing multiplication value assertion
# Purpose: Tests overflowing_mul tuple assertion - expected failure.
# Category: limitation
cat > "$TMPDIR/overflowing_mul_soundness.rs" << 'ENDOFFILE'
fn overflowing_mul_soundness_proof() {
    let a: i32 = 7;
    let b: i32 = 6;
    let (result, _overflow) = a.overflowing_mul(b);
    // Assertion on tuple field produces CHC unsatisfiable
    assert!(result == 42);
}
ENDOFFILE
run_test "overflowing_mul_soundness" "$TMPDIR/overflowing_mul_soundness.rs" "fail" "limitation"


# Test 331: Overflowing division
# Purpose: Tests overflowing_div intrinsic - tuple values unconstrained.
# Category: limitation
cat > "$TMPDIR/overflowing_div.rs" << 'ENDOFFILE'
fn overflowing_div_proof() {
    let a: i32 = 42;
    let (result, overflow) = a.overflowing_div(7);
    // Basic check - values exist (may be unconstrained)
    let _ = result;
    let _ = overflow;
}
ENDOFFILE
run_test "overflowing_div" "$TMPDIR/overflowing_div.rs" "pass" "limitation"


# Test 331b: Overflowing division value assertion
# Purpose: Tests overflowing_div tuple assertion - expected failure.
# Category: limitation
cat > "$TMPDIR/overflowing_div_soundness.rs" << 'ENDOFFILE'
fn overflowing_div_soundness_proof() {
    let a: i32 = 42;
    let (result, _overflow) = a.overflowing_div(7);
    // Assertion on tuple field produces CHC unsatisfiable
    assert!(result == 6);
}
ENDOFFILE
run_test "overflowing_div_soundness" "$TMPDIR/overflowing_div_soundness.rs" "fail" "limitation"


# Test 335: Overflowing remainder
# Purpose: Tests overflowing_rem intrinsic - tuple values unconstrained.
# Category: limitation
cat > "$TMPDIR/overflowing_rem.rs" << 'ENDOFFILE'
fn overflowing_rem_proof() {
    let a: i32 = 42;
    let (result, overflow) = a.overflowing_rem(5);
    // Basic check - values exist (may be unconstrained)
    let _ = result;
    let _ = overflow;
}
ENDOFFILE
run_test "overflowing_rem" "$TMPDIR/overflowing_rem.rs" "pass" "limitation"


# Test 335b: Overflowing remainder value assertion
# Purpose: Tests overflowing_rem tuple assertion - expected failure.
# Category: limitation
cat > "$TMPDIR/overflowing_rem_soundness.rs" << 'ENDOFFILE'
fn overflowing_rem_soundness_proof() {
    let a: i32 = 42;
    let (result, _overflow) = a.overflowing_rem(5);
    // Assertion on tuple field produces CHC unsatisfiable
    assert!(result == 2);
}
ENDOFFILE
run_test "overflowing_rem_soundness" "$TMPDIR/overflowing_rem_soundness.rs" "fail" "limitation"


# Test 337: Saturating multiplication
# Purpose: Tests saturating_mul intrinsic - expected failure (non-linear).
# Category: limitation
cat > "$TMPDIR/saturating_mul.rs" << 'ENDOFFILE'
fn saturating_mul_proof() {
    let a: i32 = 7;
    let b: i32 = 6;
    let result = a.saturating_mul(b);
    assert!(result == 42);
}
ENDOFFILE
run_test "saturating_mul" "$TMPDIR/saturating_mul.rs" "fail" "limitation"


# Test 337b: Saturating multiplication soundness
# Purpose: Tests saturating_mul soundness - expected failure (non-linear).
# Category: limitation
cat > "$TMPDIR/saturating_mul_soundness.rs" << 'ENDOFFILE'
fn saturating_mul_soundness_proof() {
    let a: i32 = 7;
    let b: i32 = 6;
    let result = a.saturating_mul(b);
    // WRONG: result is 42, not 999 - would fail if verifier could evaluate
    assert!(result == 999);
}
ENDOFFILE
run_test "saturating_mul_soundness" "$TMPDIR/saturating_mul_soundness.rs" "fail" "limitation"


# Test 341: Overflowing negation (signed)
# Purpose: Tests overflowing_neg intrinsic - tuple values unconstrained.
# Category: limitation
cat > "$TMPDIR/overflowing_neg.rs" << 'ENDOFFILE'
fn overflowing_neg_proof() {
    let a: i32 = 42;
    // overflowing_neg returns (i32, bool) where bool indicates overflow
    let (result, overflow) = a.overflowing_neg();
    // 42 can be negated without overflow
    // NOTE: We can't reliably assert on extracted tuple values
    let _ = result;
    let _ = overflow;
    assert!(true);  // Just test that it compiles
}
ENDOFFILE
run_test "overflowing_neg" "$TMPDIR/overflowing_neg.rs" "pass" "limitation"


# Test 346: Overflowing shift left
# Purpose: Tests overflowing_shl intrinsic - tuple values unconstrained.
# Category: limitation
cat > "$TMPDIR/overflowing_shl.rs" << 'ENDOFFILE'
fn overflowing_shl_proof() {
    let a: i32 = 1;
    let (result, overflow) = a.overflowing_shl(3);
    // Shift by 3 doesn't overflow
    let _ = result;
    let _ = overflow;
    assert!(true);  // Just test that it compiles
}
ENDOFFILE
run_test "overflowing_shl" "$TMPDIR/overflowing_shl.rs" "pass" "limitation"


# Test 347: Overflowing shift right
# Purpose: Tests overflowing_shr intrinsic - tuple values unconstrained.
# Category: limitation
cat > "$TMPDIR/overflowing_shr.rs" << 'ENDOFFILE'
fn overflowing_shr_proof() {
    let a: i32 = 8;
    let (result, overflow) = a.overflowing_shr(2);
    let _ = result;
    let _ = overflow;
    assert!(true);  // Just test that it compiles
}
ENDOFFILE
run_test "overflowing_shr" "$TMPDIR/overflowing_shr.rs" "pass" "limitation"


# Test 341b: Overflowing negation value assertion
# Purpose: Tests overflowing_neg tuple value assertion - expected failure.
# Category: limitation
# Note: Tuple extraction from overflowing_neg is unconstrained.
cat > "$TMPDIR/overflowing_neg_soundness.rs" << 'ENDOFFILE'
fn overflowing_neg_soundness_proof() {
    let a: i32 = 42;
    let (result, overflow) = a.overflowing_neg();
    let _ = result;
    assert!(!overflow);  // True (42 negates without overflow), but tuple unconstrained
}
ENDOFFILE
run_test "overflowing_neg_soundness" "$TMPDIR/overflowing_neg_soundness.rs" "fail" "limitation"  # Expected failure: tuple extraction unconstrained


# Test 346b: Overflowing shift left value assertion
# Purpose: Tests overflowing_shl tuple value assertion - expected failure.
# Category: limitation
# Note: Tuple extraction from overflowing_shl is unconstrained.
cat > "$TMPDIR/overflowing_shl_soundness.rs" << 'ENDOFFILE'
fn overflowing_shl_soundness_proof() {
    let a: i32 = 1;
    let (result, overflow) = a.overflowing_shl(3);
    let _ = result;
    assert!(!overflow);  // True (shift by 3 doesn't overflow), but tuple unconstrained
}
ENDOFFILE
run_test "overflowing_shl_soundness" "$TMPDIR/overflowing_shl_soundness.rs" "fail" "limitation"  # Expected failure: tuple extraction unconstrained


# Test 347b: Overflowing shift right value assertion
# Purpose: Tests overflowing_shr tuple value assertion - expected failure.
# Category: limitation
# Note: Tuple extraction from overflowing_shr is unconstrained.
cat > "$TMPDIR/overflowing_shr_soundness.rs" << 'ENDOFFILE'
fn overflowing_shr_soundness_proof() {
    let a: i32 = 8;
    let (result, overflow) = a.overflowing_shr(2);
    let _ = result;
    assert!(!overflow);  // True (shift by 2 doesn't overflow), but tuple unconstrained
}
ENDOFFILE
run_test "overflowing_shr_soundness" "$TMPDIR/overflowing_shr_soundness.rs" "fail" "limitation"  # Expected failure: tuple extraction unconstrained


# Test 350: Count zeros intrinsic
# Purpose: Tests count_zeros intrinsic - Z3 returns unknown.
# Category: limitation
cat > "$TMPDIR/count_zeros.rs" << 'ENDOFFILE'
fn count_zeros_proof() {
    let x: u32 = 0b1010101;  // 85, has 4 ones so 28 zeros in u32
    let count = x.count_zeros();
    assert!(count == 28);
}
ENDOFFILE
run_test "count_zeros" "$TMPDIR/count_zeros.rs" "fail" "limitation"  # Expected failure: bit intrinsic unsupported in integer mode


# Test 350b: Count zeros soundness
# Purpose: Tests count_zeros soundness - Z3 returns unknown.
# Category: limitation
cat > "$TMPDIR/count_zeros_soundness.rs" << 'ENDOFFILE'
fn count_zeros_soundness_proof() {
    let x: u32 = 0b1010101;
    let count = x.count_zeros();
    assert!(count == 999);
}
ENDOFFILE
run_test "count_zeros_soundness" "$TMPDIR/count_zeros_soundness.rs" "fail" "limitation"  # Expected failure: bit intrinsic unsupported in integer mode


# Test 98: Trait method dispatch via trait object (known limitation)
# Purpose: Tests trait object method dispatch (known limitation: unconstrained return).
# Category: limitation
# Note: Trait dispatch is modeled as uninterpreted function with unconstrained return.
cat > "$TMPDIR/trait_dispatch.rs" << 'ENDOFFILE'
trait Value {
    fn get(&self) -> i32;
}

struct Wrapper {
    value: i32,
}

impl Value for Wrapper {
    fn get(&self) -> i32 {
        self.value
    }
}

fn trait_dispatch_proof() {
    let w = Wrapper { value: 5 };
    let t: &dyn Value = &w;
    let v = t.get();

    // Only tautological properties are safe; v is unconstrained in CHC encoding.
    assert!(v == v);
}
ENDOFFILE
run_test "trait_dispatch" "$TMPDIR/trait_dispatch.rs" "fail" "limitation"


# Test 98b: Trait dispatch (known limitation)
# Purpose: Verifies solver correctly returns unknown for trait dispatch.
# Category: limitation
# Note: Trait method calls are uninterpreted, so solver returns unknown.
cat > "$TMPDIR/trait_dispatch_soundness.rs" << 'ENDOFFILE'
trait Value {
    fn get(&self) -> i32;
}

struct Wrapper {
    value: i32,
}

impl Value for Wrapper {
    fn get(&self) -> i32 {
        self.value
    }
}

fn trait_dispatch_soundness_proof() {
    let w = Wrapper { value: 5 };
    let t: &dyn Value = &w;

    // Returns unknown (limitation) - trait methods are uninterpreted
    assert!(t.get() == 999);
}
ENDOFFILE
run_test "trait_dispatch_soundness" "$TMPDIR/trait_dispatch_soundness.rs" "fail" "limitation"


# Test 139b: Derive PartialEq (known limitation)
# Purpose: Documents struct == returns unknown due to trait complexity.
# Category: limitation
# Note: PartialEq::eq is a trait method, uninterpreted by solver.
cat > "$TMPDIR/derive_partialeq_soundness.rs" << 'ENDOFFILE'
#[derive(PartialEq)]
struct Point {
    x: i32,
    y: i32,
}

fn derive_partialeq_soundness_proof() {
    let p1 = Point { x: 1, y: 2 };
    let p2 = Point { x: 1, y: 3 };
    assert!(p1 == p2);  // Struct == uses PartialEq::eq trait method - solver returns unknown
}
ENDOFFILE
run_test "derive_partialeq_soundness" "$TMPDIR/derive_partialeq_soundness.rs" "fail" "limitation"


# Test 263: Count set bits (population count) - while loop with bit shifts
# Purpose: Tests popcount via loop with bitwise operations.
# Category: limitation
# Note: While loop requires complex invariant synthesis (count + bit shift), times out.
cat > "$TMPDIR/popcount.rs" << 'ENDOFFILE'
fn popcount(mut n: u32) -> u32 {
    let mut count = 0u32;
    while n > 0 {
        count += n & 1;
        n >>= 1;
    }
    count
}

fn popcount_proof() {
    assert!(popcount(0) == 0);
    assert!(popcount(1) == 1);
    assert!(popcount(7) == 3);   // 0b111 = 3 bits
    assert!(popcount(15) == 4);  // 0b1111 = 4 bits
}
ENDOFFILE
run_test "popcount" "$TMPDIR/popcount.rs" "fail" "limitation"  # Timeout: while loop with bit shifts


# Test 174: Division operation (known limitation)
# Purpose: Tests integer division (a / b) with truncation toward zero.
# Category: limitation
# Note: CHC solver times out on division with bitvec encoding.
cat > "$TMPDIR/division.rs" << 'ENDOFFILE'
fn division_proof() {
    let a = 10i32;
    let b = 3i32;
    let q = a / b;  // Integer division: 10 / 3 = 3
    assert!(q == 3);
}
ENDOFFILE
run_test "division" "$TMPDIR/division.rs" "fail" "limitation"  # Timeout: bitvec division


# Test 308: Min/max update pattern (known limitation)
# Purpose: Tests conditional update of min/max trackers.
# Category: limitation
# Note: CHC solver times out on array indexing with many conditionals.
cat > "$TMPDIR/minmax_update.rs" << 'ENDOFFILE'
fn minmax_update_proof() {
    let mut min = 100i32;
    let mut max = 0i32;

    let values = [5i32, 10, 3, 8];

    // Update min/max for each value
    if values[0] < min { min = values[0]; }
    if values[0] > max { max = values[0]; }

    if values[1] < min { min = values[1]; }
    if values[1] > max { max = values[1]; }

    if values[2] < min { min = values[2]; }
    if values[2] > max { max = values[2]; }

    if values[3] < min { min = values[3]; }
    if values[3] > max { max = values[3]; }

    assert!(min == 3);
    assert!(max == 10);
}
ENDOFFILE
run_test "minmax_update" "$TMPDIR/minmax_update.rs" "fail" "limitation"  # Timeout: array conditionals
