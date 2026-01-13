# shellcheck shell=bash
# Test definitions for category: soundness
# Sourced by test_driver.sh; relies on shared helpers like run_test.

# Test 2: Failing assertion (should detect violation)
# Purpose: Tests that the verifier correctly detects assertion violations.
# Category: soundness
cat > "$TMPDIR/failing.rs" << 'EOF'
fn failing_proof() {
    let x = 5i32;
    assert!(x == 10);
}
EOF
run_test "failing_assertion" "$TMPDIR/failing.rs" "fail"


# Test 11b: Struct field soundness (regression test for issue #173)
# Purpose: Verifies that wrong struct field assertions correctly fail.
# Category: soundness
# This test MUST fail - p.x is 10, not 9999
# Previously, dead variable elimination caused undeclared variables in CHC,
# making Z3 treat the transition as vacuously true.
cat > "$TMPDIR/struct_soundness.rs" << 'EOF'
struct Point { x: i32, y: i32 }
fn struct_soundness_proof() {
    let p = Point { x: 10, y: 20 };
    assert!(p.x == 9999);  // MUST fail: p.x is 10, not 9999
}
EOF
run_test "struct_field_soundness" "$TMPDIR/struct_soundness.rs" "fail"


# Test 20b: Enum with data soundness (regression test)
# Purpose: Verifies that wrong enum data assertions correctly fail.
# Category: soundness
# This test MUST fail - it catches the soundness bug where enum field accesses
# were not properly tracked due to Downcast projection handling and missing
# field variable declarations.
cat > "$TMPDIR/enum_data_soundness.rs" << 'EOF'
enum Message {
    Quit,
    Number(i32),
}
fn enum_data_soundness_proof() {
    let m = Message::Number(42);
    let value = match m {
        Message::Quit => 0i32,
        Message::Number(n) => n,
    };
    assert!(value == 9999);  // MUST fail: value is 42, not 9999
}
EOF
run_test "enum_data_soundness" "$TMPDIR/enum_data_soundness.rs" "fail"


# Test 20c: Enum mutation soundness
# Purpose: Verifies that wrong enum variant assertions after mutation fail.
# Category: soundness
cat > "$TMPDIR/enum_mutate_soundness.rs" << 'EOF'
enum Color {
    Red,
    Green,
    Blue,
}

fn enum_mutate_soundness_proof() {
    let mut c = Color::Red;
    c = Color::Blue;
    let is_red = matches!(c, Color::Red);
    assert!(is_red);  // MUST fail: c is Blue, not Red
}
EOF
run_test "enum_mutation_soundness" "$TMPDIR/enum_mutate_soundness.rs" "fail"


# Test 20d: Enum multiple mutations soundness
# Purpose: Verifies detection of stale enum variant after multiple mutations.
# Category: soundness
cat > "$TMPDIR/enum_multi_mutate_soundness.rs" << 'EOF'
enum State {
    Init,
    Running,
    Done,
}

fn enum_multi_mutate_soundness_proof() {
    let mut s = State::Init;
    s = State::Running;
    s = State::Done;
    let is_init = matches!(s, State::Init);
    assert!(is_init);  // MUST fail: s is Done, not Init
}
EOF
run_test "enum_multi_mutation_soundness" "$TMPDIR/enum_multi_mutate_soundness.rs" "fail"


# Test 21b: Nested struct field soundness (regression test)
# Purpose: Verifies that wrong nested struct field assertions correctly fail.
# Category: soundness
# Regression: Nested struct field accesses like outer.inner.value were not
# properly propagated. Fix required declaring nested field variables.
cat > "$TMPDIR/nested_struct_soundness.rs" << 'EOF'
struct Inner { value: i32 }
struct Outer { inner: Inner }
fn nested_struct_soundness_proof() {
    let inner = Inner { value: 42 };
    let outer = Outer { inner };
    assert!(outer.inner.value == 9999);  // MUST fail: value is 42, not 9999
}
EOF
run_test "nested_struct_soundness" "$TMPDIR/nested_struct_soundness.rs" "fail"


# Test 40b: Unguarded call soundness - call without proper guard should fail
# Purpose: Verifies that precondition violations in called functions are caught.
# Category: soundness
cat > "$TMPDIR/unguarded_call_soundness.rs" << 'ENDOFFILE'
fn requires_positive(x: i32) {
    assert!(x > 0);
}

// x is 0, which violates the precondition of requires_positive
fn unguarded_call_soundness_proof() {
    let x = 0i32;
    requires_positive(x);  // Should fail: x == 0 violates assert!(x > 0)
}
ENDOFFILE
run_test "unguarded_call_soundness" "$TMPDIR/unguarded_call_soundness.rs" "fail"


# Test 44: Loop counter soundness
# Purpose: Verifies that wrong loop accumulator assertions correctly fail.
# Category: soundness
cat > "$TMPDIR/loop_counter_soundness.rs" << 'ENDOFFILE'
fn loop_counter_soundness_proof() {
    let mut sum = 0i32;
    let mut i = 0i32;
    while i < 5 {
        sum += 2;  // adds 2 each iteration: 0+2+2+2+2+2 = 10
        i += 1;
    }
    assert!(sum == 100);  // MUST fail: sum is 10, not 100
}
ENDOFFILE
run_test "loop_counter_soundness" "$TMPDIR/loop_counter_soundness.rs" "fail"


# Test 45: Function argument soundness
# Purpose: Verifies that wrong function return value assertions fail.
# Category: soundness
cat > "$TMPDIR/fn_arg_soundness.rs" << 'ENDOFFILE'
fn add(a: i32, b: i32) -> i32 { a + b }
fn fn_arg_soundness_proof() {
    let x = 5i32;
    let y = 7i32;
    let result = add(x, y);
    assert!(result == 999);  // MUST fail: result is 12, not 999
}
ENDOFFILE
run_test "fn_arg_soundness" "$TMPDIR/fn_arg_soundness.rs" "fail"


# Test 46: Array index soundness
# Purpose: Verifies that wrong array element assertions correctly fail.
# Category: soundness
cat > "$TMPDIR/array_soundness.rs" << 'ENDOFFILE'
fn array_soundness_proof() {
    let arr = [10i32, 20, 30];
    let second = arr[1];  // arr[1] == 20
    assert!(second == 9999);  // MUST fail: second is 20, not 9999
}
ENDOFFILE
run_test "array_index_soundness" "$TMPDIR/array_soundness.rs" "fail"


# Test 47: Conditional soundness - false branch
# Purpose: Verifies that assertions on wrong branch values correctly fail.
# Category: soundness
cat > "$TMPDIR/cond_soundness.rs" << 'ENDOFFILE'
fn cond_soundness_proof() {
    let x = 5i32;
    let y;
    if x >= 10 {
        y = 1i32;
    } else {
        y = 2i32;
    }
    assert!(y == 100);  // MUST fail: y is 2, not 100
}
ENDOFFILE
run_test "conditional_soundness" "$TMPDIR/cond_soundness.rs" "fail"


# Test 48: Nested function call soundness
# Purpose: Verifies that wrong nested call result assertions correctly fail.
# Category: soundness
cat > "$TMPDIR/nested_call_soundness.rs" << 'ENDOFFILE'
fn triple(x: i32) -> i32 { x * 3 }
fn double(x: i32) -> i32 { x * 2 }
fn nested_call_soundness_proof() {
    let x = 2i32;
    let result = double(triple(x));  // 2 * 3 = 6, 6 * 2 = 12
    assert!(result == 9999);  // MUST fail: result is 12, not 9999
}
ENDOFFILE
run_test "nested_call_soundness" "$TMPDIR/nested_call_soundness.rs" "fail"


# Test 49b: Shadowing soundness - wrong assertion
# Purpose: Verifies that wrong outer variable assertions after shadowing fail.
# Category: soundness
cat > "$TMPDIR/shadowing_soundness_fail.rs" << 'ENDOFFILE'
fn shadowing_soundness_fail_proof() {
    let x = 10i32;
    {
        let x = 20i32;
        assert!(x == 20);
    }
    assert!(x == 9999);  // MUST fail: outer x is 10, not 9999
}
ENDOFFILE
run_test "variable_shadowing_soundness" "$TMPDIR/shadowing_soundness_fail.rs" "fail"


# Test 50b: If-else chain soundness - wrong branch
# Purpose: Verifies that wrong if-else chain result assertions fail.
# Category: soundness
cat > "$TMPDIR/if_else_chain_soundness.rs" << 'ENDOFFILE'
fn if_else_chain_soundness_proof() {
    let x = 5i32;
    let result;
    if x < 0 {
        result = -1i32;
    } else if x == 0 {
        result = 0i32;
    } else if x < 10 {
        result = 1i32;
    } else {
        result = 2i32;
    }
    assert!(result == 9999);  // MUST fail: result is 1, not 9999
}
ENDOFFILE
run_test "if_else_chain_soundness" "$TMPDIR/if_else_chain_soundness.rs" "fail"


# Test 51b: Multiple return soundness - wrong value
# Purpose: Verifies that wrong multi-return function assertions fail.
# Category: soundness
cat > "$TMPDIR/multi_return_soundness.rs" << 'ENDOFFILE'
fn classify(x: i32) -> i32 {
    if x < 0 { return -1; }
    if x == 0 { return 0; }
    return 1;
}
fn multi_return_soundness_proof() {
    assert!(classify(0) == 9999);  // MUST fail: classify(0) == 0
}
ENDOFFILE
run_test "multi_return_soundness" "$TMPDIR/multi_return_soundness.rs" "fail"


# Test 52b: Nested linear arithmetic soundness - wrong result
# Purpose: Verifies that wrong nested arithmetic assertions fail.
# Category: soundness
cat > "$TMPDIR/nested_linear_soundness.rs" << 'ENDOFFILE'
fn nested_linear_soundness_proof() {
    let a = 10i32;
    let b = 20i32;
    let c = 5i32;
    let result = (a + b) - c + a;
    assert!(result == 9999);  // MUST fail: result is 35, not 9999
}
ENDOFFILE
run_test "nested_linear_arithmetic_soundness" "$TMPDIR/nested_linear_soundness.rs" "fail"


# Test 53b: Accumulator loop soundness - wrong result
# Purpose: Verifies that wrong accumulator loop assertions fail.
# Category: soundness
cat > "$TMPDIR/accumulator_loop_soundness.rs" << 'ENDOFFILE'
fn accumulator_loop_soundness_proof() {
    let mut sum = 0i32;
    let mut i = 1i32;
    while i <= 4 {
        sum = sum + i;
        i = i + 1;
    }
    assert!(sum == 9999);  // MUST fail: sum is 10, not 9999
}
ENDOFFILE
run_test "accumulator_loop_soundness" "$TMPDIR/accumulator_loop_soundness.rs" "fail"


# Test 54b: Chained conditionals soundness - wrong result
# Purpose: Verifies that wrong chained conditional assertions fail.
# Category: soundness
cat > "$TMPDIR/chained_cond_soundness.rs" << 'ENDOFFILE'
fn chained_cond_soundness_proof() {
    let a = 5i32;
    let mut x = 0i32;
    let mut y = 0i32;
    if a > 0 { x = 10; }
    if a > 3 { y = 20; }
    assert!(x + y == 9999);  // MUST fail: x+y is 30, not 9999
}
ENDOFFILE
run_test "chained_conditionals_soundness" "$TMPDIR/chained_cond_soundness.rs" "fail"


# Test 55b: Complex control flow soundness - wrong result
# Purpose: Verifies that wrong complex control flow assertions fail.
# Category: soundness
cat > "$TMPDIR/complex_control_soundness.rs" << 'ENDOFFILE'
fn max_linear(a: i32, b: i32) -> i32 {
    if a > b { a } else { b }
}
fn min_linear(a: i32, b: i32) -> i32 {
    if a < b { a } else { b }
}
fn complex_control_soundness_proof() {
    let x = 10i32;
    let y = 20i32;
    let z = 15i32;
    let result = max_linear(min_linear(x, y), z);
    assert!(result == 9999);  // MUST fail: result is 15, not 9999
}
ENDOFFILE
run_test "complex_control_flow_soundness" "$TMPDIR/complex_control_soundness.rs" "fail"


# Test 56b: Loop with break soundness - wrong result
# Purpose: Verifies that wrong loop-with-break assertions fail.
# Category: soundness
cat > "$TMPDIR/loop_break_soundness.rs" << 'ENDOFFILE'
fn loop_break_soundness_proof() {
    let mut sum = 0i32;
    let mut i = 0i32;
    loop {
        if i >= 5 {
            break;
        }
        sum += i;
        i += 1;
    }
    assert!(sum == 9999);  // MUST fail: sum is 10, not 9999
}
ENDOFFILE
run_test "loop_with_break_soundness" "$TMPDIR/loop_break_soundness.rs" "fail"


# Test 57b: Mutable reference function call soundness check
# Purpose: Verifies that stale value assertions after mut ref call fail.
# Category: soundness
cat > "$TMPDIR/mut_ref_call_soundness.rs" << 'ENDOFFILE'
fn increment(x: &mut i32) {
    *x += 1;
}

fn mut_ref_call_soundness_proof() {
    let mut val = 5i32;
    increment(&mut val);
    assert!(val == 5);  // SHOULD FAIL: val is now 6, not 5
}
ENDOFFILE
run_test "mutable_ref_soundness" "$TMPDIR/mut_ref_call_soundness.rs" "fail"


# Test 58b: Multiple mutable reference calls soundness check
# Purpose: Verifies that stale value after multiple mut ref calls fail.
# Category: soundness
cat > "$TMPDIR/multi_mut_ref_soundness.rs" << 'ENDOFFILE'
fn double(x: &mut i32) {
    *x *= 2;
}

fn multi_mut_ref_soundness_proof() {
    let mut val = 5i32;
    double(&mut val);
    double(&mut val);
    assert!(val == 5);  // SHOULD FAIL: val is now 20, not 5
}
ENDOFFILE
run_test "multi_mutable_ref_soundness" "$TMPDIR/multi_mut_ref_soundness.rs" "fail"


# Test 59b: Match on negative integers soundness - MOVED to unsound.sh
# This test reveals a soundness bug with negative integer matching.
# MOVED to unsound.sh


# Test 60b: Loop with continue soundness - wrong count
# Purpose: Verifies that wrong loop-with-continue count assertions fail.
# Category: soundness
cat > "$TMPDIR/loop_continue_soundness.rs" << 'ENDOFFILE'
fn loop_with_continue_soundness_proof() {
    let mut count = 0i32;
    let mut i = 0i32;
    while i < 5 {
        i += 1;
        if i == 3 {
            continue;
        }
        count += 1;
    }
    assert!(count == 9999);  // MUST fail: count is 4, not 9999
}
ENDOFFILE
run_test "loop_with_continue_soundness" "$TMPDIR/loop_continue_soundness.rs" "fail"


# Test 61b: Labeled break soundness - wrong result
# Purpose: Verifies that wrong labeled break result assertions fail.
# Category: soundness
cat > "$TMPDIR/labeled_break_soundness.rs" << 'ENDOFFILE'
fn labeled_break_soundness_proof() {
    let mut found = 0i32;
    let mut i = 0i32;
    'outer: while i < 3 {
        let mut j = 0i32;
        while j < 3 {
            if i == 1 && j == 1 {
                found = 1;
                break 'outer;
            }
            j = j + 1;
        }
        i = i + 1;
    }
    assert!(found == 9999);  // MUST fail: found is 1, not 9999
}
ENDOFFILE
run_test "labeled_break_soundness" "$TMPDIR/labeled_break_soundness.rs" "fail"


# Test 63b: If let soundness - wrong value
# Purpose: Verifies that wrong if-let extracted value assertions fail.
# Category: soundness
cat > "$TMPDIR/if_let_soundness.rs" << 'ENDOFFILE'
fn if_let_soundness_proof() {
    let x: Option<i32> = Some(42);
    if let Some(val) = x {
        assert!(val == 9999);  // MUST fail: val is 42, not 9999
    }
}
ENDOFFILE
run_test "if_let_soundness" "$TMPDIR/if_let_soundness.rs" "fail"


# Test 64b: While let soundness - wrong count
# Purpose: Verifies that wrong while-let count assertions fail.
# Category: soundness
cat > "$TMPDIR/while_let_soundness.rs" << 'ENDOFFILE'
fn while_let_soundness_proof() {
    let mut count = 0i32;
    let mut opt: Option<i32> = Some(42);
    while let Some(_x) = opt {
        count += 1;
        opt = None;
    }
    assert!(count == 9999);  // MUST fail: count is 1, not 9999
}
ENDOFFILE
run_test "while_let_soundness" "$TMPDIR/while_let_soundness.rs" "fail"


# Test 65b: Let else soundness - wrong value
# Purpose: Verifies that wrong let-else extracted value assertions fail.
# Category: soundness
cat > "$TMPDIR/let_else_soundness.rs" << 'ENDOFFILE'
fn let_else_soundness_proof() {
    let x: Option<i32> = Some(10);
    let Some(val) = x else {
        return;
    };
    assert!(val == 9999);  // MUST fail: val is 10, not 9999
}
ENDOFFILE
run_test "let_else_soundness" "$TMPDIR/let_else_soundness.rs" "fail"


# Test 66b: Module-level constants soundness - wrong value
# Purpose: Verifies that wrong const value assertions fail.
# Category: soundness
cat > "$TMPDIR/const_soundness.rs" << 'ENDOFFILE'
const MAX_VALUE: i32 = 100;

fn const_soundness_proof() {
    let x = MAX_VALUE;
    assert!(x == 9999);  // MUST fail: MAX_VALUE is 100, not 9999
}
ENDOFFILE
run_test "module_constants_soundness" "$TMPDIR/const_soundness.rs" "fail"


# Test 68b: Range types soundness - wrong value
# Purpose: Verifies that wrong Range field assertions fail.
# Category: soundness
cat > "$TMPDIR/range_soundness.rs" << 'ENDOFFILE'
fn range_soundness_proof() {
    let r = 5..10;
    assert!(r.end == 9999);  // MUST fail: r.end is 10, not 9999
}
ENDOFFILE
run_test "range_types_soundness" "$TMPDIR/range_soundness.rs" "fail"


# Test 71b: Struct methods soundness - wrong value
# Purpose: Verifies that wrong struct method return assertions fail.
# Category: soundness
cat > "$TMPDIR/struct_method_soundness.rs" << 'ENDOFFILE'
struct Counter {
    value: i32,
}

impl Counter {
    fn new(start: i32) -> Counter {
        Counter { value: start }
    }

    fn get(&self) -> i32 {
        self.value
    }
}

fn struct_method_soundness_proof() {
    let c = Counter::new(10);
    assert!(c.get() == 9999);  // MUST fail: c.get() returns 10, not 9999
}
ENDOFFILE
run_test "struct_methods_soundness" "$TMPDIR/struct_method_soundness.rs" "fail"


# Test 72b: Mutable struct methods soundness - wrong value
# Purpose: Verifies that stale struct field assertions after mutation fail.
# Category: soundness
cat > "$TMPDIR/struct_mut_method_soundness.rs" << 'ENDOFFILE'
struct Counter {
    value: i32,
}

impl Counter {
    fn increment(&mut self) {
        self.value += 1;
    }

    fn add(&mut self, amount: i32) {
        self.value += amount;
    }
}

fn struct_mut_method_soundness_proof() {
    let mut c = Counter { value: 0 };
    c.increment();
    c.add(10);
    assert!(c.value == 0);  // MUST fail: c.value is 11, not 0
}
ENDOFFILE
run_test "struct_mut_methods_soundness" "$TMPDIR/struct_mut_method_soundness.rs" "fail"


# Test 75b: Tuple destructuring soundness - wrong value
# Purpose: Verifies that wrong tuple destructured value assertions fail.
# Category: soundness
cat > "$TMPDIR/tuple_destruct_soundness.rs" << 'ENDOFFILE'
fn tuple_destruct_soundness_proof() {
    let pair = (10i32, 20i32);
    let (a, _b) = pair;
    assert!(a == 9999);  // MUST fail: a is 10, not 9999
}
ENDOFFILE
run_test "tuple_destructuring_soundness" "$TMPDIR/tuple_destruct_soundness.rs" "fail"


# Test 77b: Struct destructuring soundness - wrong value
# Purpose: Verifies that wrong struct destructured value assertions fail.
# Category: soundness
cat > "$TMPDIR/struct_destruct_soundness.rs" << 'ENDOFFILE'
struct Point { x: i32, y: i32 }

fn struct_destruct_soundness_proof() {
    let p = Point { x: 10, y: 20 };
    let Point { x: px, y: _ } = p;
    assert!(px == 9999);  // MUST fail: px is 10, not 9999
}
ENDOFFILE
run_test "struct_destructuring_soundness" "$TMPDIR/struct_destruct_soundness.rs" "fail"


# Test 78b: Range pattern soundness - value outside range
# Purpose: Verifies that wrong wildcard branch assertions fail.
# Category: soundness
cat > "$TMPDIR/range_pattern_soundness.rs" << 'ENDOFFILE'
fn range_pattern_soundness_proof() {
    let x = 15i32;  // Outside range 1..=10

    match x {
        1..=10 => assert!(x == 15),  // Should NOT reach here since 15 > 10
        _ => assert!(x == 999),  // MUST fail: x is 15, not 999
    }
}
ENDOFFILE
run_test "range_pattern_soundness" "$TMPDIR/range_pattern_soundness.rs" "fail"


# Test 79b: @ binding soundness - wrong value assertion
# Purpose: Verifies that wrong @ binding value assertions fail.
# Category: soundness
cat > "$TMPDIR/at_binding_soundness.rs" << 'ENDOFFILE'
fn at_binding_soundness_proof() {
    let x = Some(5i32);

    match x {
        Some(val @ 1..=10) => assert!(val == 999),  // MUST fail: val is 5, not 999
        _ => assert!(true),
    }
}
ENDOFFILE
run_test "at_binding_soundness" "$TMPDIR/at_binding_soundness.rs" "fail"


# Test 80b: Or-pattern soundness
# Purpose: Verifies that wrong or-pattern value assertions fail.
# Category: soundness
cat > "$TMPDIR/or_pattern_soundness.rs" << 'ENDOFFILE'
fn or_pattern_soundness_proof() {
    let x = 3i32;

    match x {
        1 | 2 | 3 => assert!(x == 999),  // MUST fail: x is 3, not 999
        _ => assert!(true),
    }
}
ENDOFFILE
run_test "or_pattern_soundness" "$TMPDIR/or_pattern_soundness.rs" "fail"


# Test 81b: ? operator soundness - calling with zero should fail
# Purpose: Verifies that Err branch assertions are checked.
# Category: soundness
cat > "$TMPDIR/question_op_soundness.rs" << 'ENDOFFILE'
fn might_fail(x: i32) -> Result<i32, ()> {
    if x > 0 { Ok(x * 2) } else { Err(()) }
}

fn question_op_soundness_proof() {
    // Call with 0 - should return Err
    let result = might_fail(0);
    match result {
        Ok(val) => assert!(val == 0),  // Should not reach here
        Err(_) => assert!(false),  // MUST fail: we always get Err from might_fail(0)
    }
}
ENDOFFILE
run_test "question_op_soundness" "$TMPDIR/question_op_soundness.rs" "fail"


# Test 82b: Const fn soundness
# Purpose: Verifies that wrong const fn return assertions fail.
# Category: soundness
cat > "$TMPDIR/const_fn_soundness.rs" << 'ENDOFFILE'
const fn add_const(a: i32, b: i32) -> i32 {
    a + b
}

fn const_fn_soundness_proof() {
    let sum = add_const(5, 7);
    assert!(sum == 999);  // MUST fail: 5 + 7 = 12, not 999
}
ENDOFFILE
run_test "const_fn_soundness" "$TMPDIR/const_fn_soundness.rs" "fail"


# Test 83b: Associated constants soundness
# Purpose: Verifies that wrong associated const assertions fail.
# Category: soundness
cat > "$TMPDIR/assoc_const_soundness.rs" << 'ENDOFFILE'
struct Circle {
    radius: i32,
}

impl Circle {
    const PI_APPROX: i32 = 3;
}

fn assoc_const_soundness_proof() {
    assert!(Circle::PI_APPROX == 999);  // MUST fail: PI_APPROX is 3, not 999
}
ENDOFFILE
run_test "assoc_const_soundness" "$TMPDIR/assoc_const_soundness.rs" "fail"


# Test 84b: Type alias soundness
# Purpose: Verifies that wrong type alias value assertions fail.
# Category: soundness
cat > "$TMPDIR/type_alias_soundness.rs" << 'ENDOFFILE'
type Score = i32;

fn type_alias_soundness_proof() {
    let score: Score = 100;
    assert!(score == 999);  // MUST fail: score is 100, not 999
}
ENDOFFILE
run_test "type_alias_soundness" "$TMPDIR/type_alias_soundness.rs" "fail"


# Test 85b: Nested function soundness
# Purpose: Verifies that wrong nested function return assertions fail.
# Category: soundness
cat > "$TMPDIR/nested_fn_soundness.rs" << 'ENDOFFILE'
fn nested_fn_soundness_proof() {
    fn add_inner(a: i32, b: i32) -> i32 {
        a + b
    }

    let sum = add_inner(5, 3);
    assert!(sum == 999);  // MUST fail: 5 + 3 = 8, not 999
}
ENDOFFILE
run_test "nested_fn_soundness" "$TMPDIR/nested_fn_soundness.rs" "fail"


# Test 87b: Parameter destructuring soundness
# Purpose: Verifies that wrong parameter destructuring assertions fail.
# Category: soundness
cat > "$TMPDIR/param_destructure_soundness.rs" << 'ENDOFFILE'
fn add_pair((a, b): (i32, i32)) -> i32 {
    a + b
}

fn param_destructure_soundness_proof() {
    let pair = (3i32, 7i32);
    let sum = add_pair(pair);
    assert!(sum == 9999);  // MUST fail: sum is 10, not 9999
}
ENDOFFILE
run_test "param_destructure_soundness" "$TMPDIR/param_destructure_soundness.rs" "fail"


# Test 88b: Builder Self soundness
# Purpose: Verifies that wrong builder pattern assertions fail.
# Category: soundness
cat > "$TMPDIR/builder_self_soundness.rs" << 'ENDOFFILE'
struct Point {
    x: i32,
    y: i32,
}

impl Point {
    fn new() -> Self {
        Point { x: 0, y: 0 }
    }

    fn with_x(self, x: i32) -> Self {
        Point { x, y: self.y }
    }

    fn with_y(self, y: i32) -> Self {
        Point { x: self.x, y }
    }
}

fn builder_self_soundness_proof() {
    let p = Point::new().with_x(5).with_y(10);
    assert!(p.x == 9999);  // MUST fail: x is 5, not 9999
}
ENDOFFILE
run_test "builder_self_soundness" "$TMPDIR/builder_self_soundness.rs" "fail"


# Test 91b: Default trait soundness
# Purpose: Verifies that wrong default value assertions fail.
# Category: soundness
cat > "$TMPDIR/default_impl_soundness.rs" << 'ENDOFFILE'
struct Counter {
    value: i32,
    max: i32,
}

impl Counter {
    fn default() -> Self {
        Counter { value: 0, max: 100 }
    }
}

fn default_impl_soundness_proof() {
    let c = Counter::default();
    assert!(c.value == 999);  // MUST fail: value is 0, not 999
}
ENDOFFILE
run_test "default_impl_soundness" "$TMPDIR/default_impl_soundness.rs" "fail"


# Test 92b: Multiple impl blocks soundness
# Purpose: Verifies that wrong multi-impl method assertions fail.
# Category: soundness
cat > "$TMPDIR/multi_impl_soundness.rs" << 'ENDOFFILE'
struct Stats {
    a: i32,
    b: i32,
}

impl Stats {
    fn new(x: i32, y: i32) -> Self {
        Stats { a: x, b: y }
    }
}

impl Stats {
    fn sum(&self) -> i32 {
        self.a + self.b
    }
}

fn multi_impl_soundness_proof() {
    let s = Stats::new(3, 4);
    assert!(s.sum() == 999);  // MUST fail: sum is 7, not 999
}
ENDOFFILE
run_test "multi_impl_soundness" "$TMPDIR/multi_impl_soundness.rs" "fail"


# Test 93b: Pair struct soundness
# Purpose: Verifies that wrong struct constructor assertions fail.
# Category: soundness
cat > "$TMPDIR/pair_struct_soundness.rs" << 'ENDOFFILE'
struct IntPair {
    first: i32,
    second: i32,
}

impl IntPair {
    fn new(a: i32, b: i32) -> Self {
        IntPair { first: a, second: b }
    }
}

fn pair_struct_soundness_proof() {
    let p = IntPair::new(10, 20);
    assert!(p.first == 999);  // MUST fail: first is 10, not 999
}
ENDOFFILE
run_test "pair_struct_soundness" "$TMPDIR/pair_struct_soundness.rs" "fail"


# Test 94b: Owned self soundness
# Purpose: Verifies that wrong consuming method assertions fail.
# Category: soundness
cat > "$TMPDIR/owned_self_soundness.rs" << 'ENDOFFILE'
struct Wrapper {
    value: i32,
}

impl Wrapper {
    fn new(v: i32) -> Self {
        Wrapper { value: v }
    }

    fn unwrap(self) -> i32 {
        self.value
    }
}

fn owned_self_soundness_proof() {
    let w = Wrapper::new(42);
    let v = w.unwrap();
    assert!(v == 999);  // MUST fail: v is 42, not 999
}
ENDOFFILE
run_test "owned_self_soundness" "$TMPDIR/owned_self_soundness.rs" "fail"


# Test 95b: Chained self types soundness
# Purpose: Verifies that wrong chained method assertions fail.
# Category: soundness
cat > "$TMPDIR/chained_self_types_soundness.rs" << 'ENDOFFILE'
struct Counter {
    value: i32,
}

impl Counter {
    fn new() -> Self {
        Counter { value: 0 }
    }

    fn increment(&mut self) {
        self.value += 1;
    }
}

fn chained_self_types_soundness_proof() {
    let mut c = Counter::new();
    c.increment();
    c.increment();
    assert!(c.value == 999);  // MUST fail: value is 2, not 999
}
ENDOFFILE
run_test "chained_self_types_soundness" "$TMPDIR/chained_self_types_soundness.rs" "fail"


# Test 96b: Unit struct soundness
# Purpose: Verifies wrong return value assertions on unit struct methods fail.
# Category: soundness
cat > "$TMPDIR/unit_struct_soundness.rs" << 'ENDOFFILE'
struct Unit;

impl Unit {
    fn value(&self) -> i32 {
        42
    }
}

fn unit_struct_soundness_proof() {
    let u = Unit;
    assert!(u.value() == 999);  // MUST fail: value is 42, not 999
}
ENDOFFILE
run_test "unit_struct_soundness" "$TMPDIR/unit_struct_soundness.rs" "fail"


# Test 97b: Tuple struct impl soundness
# Purpose: Verifies wrong assertions on tuple struct method return values fail.
# Category: soundness
cat > "$TMPDIR/tuple_struct_impl_soundness.rs" << 'ENDOFFILE'
struct Meters(i32);

impl Meters {
    fn new(m: i32) -> Self {
        Meters(m)
    }

    fn value(&self) -> i32 {
        self.0
    }

    fn double(&self) -> Self {
        Meters(self.0 * 2)
    }
}

fn tuple_struct_impl_soundness_proof() {
    let m = Meters::new(5);
    let doubled = m.double();
    assert!(doubled.value() == 999);  // MUST fail: doubled is 10, not 999
}
ENDOFFILE
run_test "tuple_struct_impl_soundness" "$TMPDIR/tuple_struct_impl_soundness.rs" "fail"


# Test 99b: Saturating arithmetic soundness
# Purpose: Verifies wrong saturating_add results are correctly rejected.
# Category: soundness
cat > "$TMPDIR/saturating_add_soundness.rs" << 'ENDOFFILE'
fn saturating_add_soundness_proof() {
    let a: i32 = 100;
    let b: i32 = 50;
    let result = a.saturating_add(b);
    // WRONG: result is 150, not 999 - should fail verification
    assert!(result == 999);
}
ENDOFFILE
run_test "saturating_add_soundness" "$TMPDIR/saturating_add_soundness.rs" "fail"


# Test 100b: Wrapping arithmetic soundness
# Purpose: Verifies wrong wrapping_add results are correctly rejected.
# Category: soundness
cat > "$TMPDIR/wrapping_add_soundness.rs" << 'ENDOFFILE'
fn wrapping_add_soundness_proof() {
    let a: i32 = 10;
    let b: i32 = 20;
    let result = a.wrapping_add(b);
    // WRONG: result is 30, not 999 - should fail verification
    assert!(result == 999);
}
ENDOFFILE
run_test "wrapping_add_soundness" "$TMPDIR/wrapping_add_soundness.rs" "fail"


# Test 101b: Min/max soundness
# Purpose: Verifies wrong min/max assertions are correctly rejected.
# Category: soundness
cat > "$TMPDIR/min_max_soundness.rs" << 'ENDOFFILE'
fn min_max_soundness_proof() {
    let a: i32 = 5;
    let b: i32 = 10;
    let minimum = if a < b { a } else { b };
    assert!(minimum == 999);  // MUST fail: minimum is 5
}
ENDOFFILE
run_test "min_max_soundness" "$TMPDIR/min_max_soundness.rs" "fail"


# Test 102b: Nested match soundness
# Purpose: Verifies wrong enum match extraction assertions fail.
# Category: soundness
cat > "$TMPDIR/nested_match_soundness.rs" << 'ENDOFFILE'
enum Outer {
    Inner(i32),
    Empty,
}

fn nested_match_soundness_proof() {
    let value = Outer::Inner(42);

    let result = match value {
        Outer::Inner(x) => x,
        Outer::Empty => 0,
    };

    assert!(result == 999);  // MUST fail: result is 42
}
ENDOFFILE
run_test "nested_match_soundness" "$TMPDIR/nested_match_soundness.rs" "fail"


# Test 103b: Absolute value soundness
# Purpose: Verifies wrong absolute value assertions are rejected.
# Category: soundness
cat > "$TMPDIR/abs_value_soundness.rs" << 'ENDOFFILE'
fn abs_value_soundness_proof() {
    let negative: i32 = -5;
    let neg_abs = if negative >= 0 { negative } else { -negative };
    assert!(neg_abs == -5);  // MUST fail: abs(-5) is 5, not -5
}
ENDOFFILE
run_test "abs_value_soundness" "$TMPDIR/abs_value_soundness.rs" "fail"


# Test 104b: Clamp soundness
# Purpose: Verifies wrong clamp result assertions are rejected.
# Category: soundness
cat > "$TMPDIR/clamp_soundness.rs" << 'ENDOFFILE'
fn clamp_soundness_proof() {
    let value: i32 = 150;
    let min_val: i32 = 0;
    let max_val: i32 = 100;

    let clamped = if value < min_val {
        min_val
    } else if value > max_val {
        max_val
    } else {
        value
    };

    assert!(clamped == 150);  // MUST fail: clamped value is 100
}
ENDOFFILE
run_test "clamp_soundness" "$TMPDIR/clamp_soundness.rs" "fail"


# Test 105b: Boolean short-circuit soundness
# Purpose: Verifies wrong short-circuit boolean assertions fail.
# Category: soundness
cat > "$TMPDIR/short_circuit_soundness.rs" << 'ENDOFFILE'
fn short_circuit_soundness_proof() {
    let a = true;
    let b = false;

    let and_result = a && b;  // true && false = false
    assert!(and_result);  // MUST fail: result is false
}
ENDOFFILE
run_test "short_circuit_soundness" "$TMPDIR/short_circuit_soundness.rs" "fail"


# Test 107b: Variable shadowing soundness
# Purpose: Verifies assertions on original value after shadowing fail.
# Category: soundness
cat > "$TMPDIR/shadowing_soundness.rs" << 'ENDOFFILE'
fn shadowing_soundness_proof() {
    let x = 5i32;
    let x = x + 10;
    assert!(x == 5);  // MUST fail: shadowed x is 15, not 5
}
ENDOFFILE
run_test "shadowing_soundness" "$TMPDIR/shadowing_soundness.rs" "fail"


# Test 108b: Struct update syntax soundness
# Purpose: Verifies wrong assertions on updated struct fields fail.
# Category: soundness
cat > "$TMPDIR/struct_update_soundness.rs" << 'ENDOFFILE'
struct Config {
    x: i32,
    y: i32,
}

fn struct_update_soundness_proof() {
    let base = Config { x: 1, y: 2 };
    let updated = Config { x: 100, ..base };
    assert!(updated.x == 1);  // MUST fail: x is 100, not 1
}
ENDOFFILE
run_test "struct_update_soundness" "$TMPDIR/struct_update_soundness.rs" "fail"


# Test 109b: Multiple return values soundness (fail - tuple returns now correctly verified)
# Purpose: Verifies wrong tuple return value assertions fail.
# Category: soundness
cat > "$TMPDIR/tuple_return_soundness.rs" << 'ENDOFFILE'
fn sum_and_diff(a: i32, b: i32) -> (i32, i32) {
    (a + b, a - b)
}

fn tuple_return_soundness_proof() {
    let (sum, _diff) = sum_and_diff(10, 3);
    assert!(sum == 12);  // MUST fail: sum is 13, not 12
}
ENDOFFILE
run_test "tuple_return_soundness" "$TMPDIR/tuple_return_soundness.rs" "fail"


# Test 110b: Unit return soundness
# Purpose: Verifies wrong assertions on mutated values fail.
# Category: soundness
cat > "$TMPDIR/unit_return_soundness.rs" << 'ENDOFFILE'
fn set_value(target: &mut i32, value: i32) {
    *target = value;
}

fn unit_return_soundness_proof() {
    let mut x = 0i32;
    set_value(&mut x, 42);
    assert!(x == 0);  // MUST fail: x is 42
}
ENDOFFILE
run_test "unit_return_soundness" "$TMPDIR/unit_return_soundness.rs" "fail"


# Test 111b: If let expression soundness
# Purpose: Verifies wrong if-let None branch assertions fail.
# Category: soundness
cat > "$TMPDIR/if_let_expr_soundness.rs" << 'ENDOFFILE'
fn if_let_expr_soundness_proof() {
    let x: Option<i32> = None;
    let result = if let Some(n) = x {
        n * 2
    } else {
        0
    };
    assert!(result == 42);  // MUST fail: result is 0
}
ENDOFFILE
run_test "if_let_expr_soundness" "$TMPDIR/if_let_expr_soundness.rs" "fail"


# Test 36b: Compound assignment soundness
# Purpose: Verifies wrong compound assignment results fail.
# Category: soundness
cat > "$TMPDIR/compound_assign_soundness.rs" << 'ENDOFFILE'
fn compound_assign_soundness_proof() {
    let mut x = 10i32;
    x += 5;
    assert!(x == 10);  // MUST fail: x is 15
}
ENDOFFILE
run_test "compound_assign_soundness" "$TMPDIR/compound_assign_soundness.rs" "fail"


# Test 113b: Loop break with value soundness
# Purpose: Verifies wrong loop break value assertions fail.
# Category: soundness
cat > "$TMPDIR/loop_break_value_soundness.rs" << 'ENDOFFILE'
fn loop_break_value_soundness_proof() {
    let mut i = 0i32;
    let result = loop {
        i += 1;
        if i == 5 {
            break i * 2;
        }
    };
    assert!(result == 5);  // MUST fail: result is 10
}
ENDOFFILE
run_test "loop_break_value_soundness" "$TMPDIR/loop_break_value_soundness.rs" "fail"


# Test 114b: Explicit array literal soundness
# Purpose: Verifies wrong array element assertions fail.
# Category: soundness
cat > "$TMPDIR/array_literal_soundness.rs" << 'ENDOFFILE'
fn array_literal_soundness_proof() {
    let arr = [1i32, 2, 3, 4, 5];
    assert!(arr[2] == 10);  // MUST fail: arr[2] is 3
}
ENDOFFILE
run_test "array_literal_soundness" "$TMPDIR/array_literal_soundness.rs" "fail"


# Test 115b: Const item soundness
# Purpose: Verifies wrong const value assertions fail.
# Category: soundness
cat > "$TMPDIR/const_item_soundness.rs" << 'ENDOFFILE'
const VALUE: i32 = 42;

fn const_item_soundness_proof() {
    assert!(VALUE == 0);  // MUST fail: VALUE is 42
}
ENDOFFILE
run_test "const_item_soundness" "$TMPDIR/const_item_soundness.rs" "fail"


# Test 116b: Integer widths soundness
# Purpose: Verifies wrong integer width value assertions fail.
# Category: soundness
cat > "$TMPDIR/int_widths_soundness.rs" << 'ENDOFFILE'
fn int_widths_soundness_proof() {
    let a: u8 = 200;
    assert!(a == 100);  // MUST fail: a is 200
}
ENDOFFILE
run_test "int_widths_soundness" "$TMPDIR/int_widths_soundness.rs" "fail"


# Test 117b: Labeled break with value soundness
# Purpose: Verifies wrong labeled break value assertions fail.
# Category: soundness
cat > "$TMPDIR/labeled_break_value_soundness.rs" << 'ENDOFFILE'
fn labeled_break_value_soundness_proof() {
    let result = 'outer: loop {
        let mut x = 0i32;
        loop {
            x += 1;
            if x == 3 {
                break 'outer x * 10;
            }
        }
    };
    assert!(result == 3);  // MUST fail: result is 30
}
ENDOFFILE
run_test "labeled_break_value_soundness" "$TMPDIR/labeled_break_value_soundness.rs" "fail"


# Test 118b: Arithmetic with constants soundness
# Purpose: Verifies wrong arithmetic with constants assertions fail.
# Category: soundness
cat > "$TMPDIR/const_arith_soundness.rs" << 'ENDOFFILE'
fn const_arith_soundness_proof() {
    let a: i32 = 10;
    let product = a * 4;
    assert!(product == 30);  // MUST fail: product is 40
}
ENDOFFILE
run_test "const_arith_soundness" "$TMPDIR/const_arith_soundness.rs" "fail"


# Test 119b: Simple enum unit variants soundness
# Purpose: Verifies wrong unit enum match assertions fail.
# Category: soundness
cat > "$TMPDIR/match_unit_enum_soundness.rs" << 'ENDOFFILE'
enum Color {
    Red,
    Green,
    Blue,
}

fn match_unit_enum_soundness_proof() {
    let c = Color::Green;
    let value = match c {
        Color::Red => 1,
        Color::Green => 2,
        Color::Blue => 3,
    };
    assert!(value == 1);  // MUST fail: value is 2
}
ENDOFFILE
run_test "match_unit_enum_soundness" "$TMPDIR/match_unit_enum_soundness.rs" "fail"


# Test 120b: Nested if-else expression soundness
# Purpose: Verifies wrong nested if-else value assertions fail.
# Category: soundness
cat > "$TMPDIR/nested_if_expr_soundness.rs" << 'ENDOFFILE'
fn nested_if_expr_soundness_proof() {
    let x = 15i32;
    let category = if x < 0 {
        -1
    } else if x < 10 {
        0
    } else if x < 20 {
        1
    } else {
        2
    };
    assert!(category == 0);  // MUST fail: category is 1
}
ENDOFFILE
run_test "nested_if_expr_soundness" "$TMPDIR/nested_if_expr_soundness.rs" "fail"


# Test 121b: Wildcard in let binding soundness
# Purpose: Verifies wrong wildcard tuple assertions fail.
# Category: soundness
cat > "$TMPDIR/wildcard_let_soundness.rs" << 'ENDOFFILE'
fn wildcard_let_soundness_proof() {
    let (a, _, c) = (1i32, 2i32, 3i32);
    assert!(a + c == 5);  // MUST fail: a + c is 4
}
ENDOFFILE
run_test "wildcard_let_soundness" "$TMPDIR/wildcard_let_soundness.rs" "fail"


# Test 122b: Wildcard in function parameter soundness
# Purpose: Verifies wrong assertions on used parameters fail.
# Category: soundness
cat > "$TMPDIR/wildcard_param_soundness.rs" << 'ENDOFFILE'
fn ignore_first(_unused: i32, used: i32) -> i32 {
    used * 2
}

fn wildcard_param_soundness_proof() {
    let result = ignore_first(999, 21);
    assert!(result == 999);  // MUST fail: result is 42
}
ENDOFFILE
run_test "wildcard_param_soundness" "$TMPDIR/wildcard_param_soundness.rs" "fail"


# Test 123b: Multi-field struct update soundness
# Purpose: Verifies wrong multi-field struct update assertions fail.
# Category: soundness
cat > "$TMPDIR/multi_field_update_soundness.rs" << 'ENDOFFILE'
struct Pair { a: i32, b: i32 }

fn multi_field_update_soundness_proof() {
    let mut p = Pair { a: 1, b: 2 };
    p.a = p.b;
    p.b = 10;
    assert!(p.a == 1);  // MUST fail: a was updated to 2
}
ENDOFFILE
run_test "multi_field_update_soundness" "$TMPDIR/multi_field_update_soundness.rs" "fail"


# Test 124b: Tuple struct pattern matching soundness
# Purpose: Verifies wrong tuple struct destructure assertions fail.
# Category: soundness
cat > "$TMPDIR/tuple_struct_destruct_soundness.rs" << 'ENDOFFILE'
struct Wrapper(i32, i32);

fn tuple_struct_destruct_soundness_proof() {
    let w = Wrapper(10, 20);
    let Wrapper(a, b) = w;
    assert!(a + b == 20);  // MUST fail: a + b is 30
}
ENDOFFILE
run_test "tuple_struct_destruct_soundness" "$TMPDIR/tuple_struct_destruct_soundness.rs" "fail"


# Test 125b: Chained boolean expressions soundness
# Purpose: Verifies wrong chained boolean assertions fail.
# Category: soundness
cat > "$TMPDIR/chained_bool_soundness.rs" << 'ENDOFFILE'
fn chained_bool_soundness_proof() {
    let a = true;
    let b = false;
    let c = true;

    let r = a && b;  // true && false = false
    assert!(r);  // MUST fail: r is false
}
ENDOFFILE
run_test "chained_bool_soundness" "$TMPDIR/chained_bool_soundness.rs" "fail"


# Test 126b: Const expression in array length soundness
# Purpose: Verifies wrong const array element assertions fail.
# Category: soundness
cat > "$TMPDIR/const_array_len_soundness.rs" << 'ENDOFFILE'
const SIZE: usize = 3;

fn const_array_len_soundness_proof() {
    let arr: [i32; SIZE] = [10, 20, 30];
    assert!(arr[1] == 10);  // MUST fail: arr[1] is 20
}
ENDOFFILE
run_test "const_array_len_soundness" "$TMPDIR/const_array_len_soundness.rs" "fail"


# Test 127b: If-let with chained operations soundness
# Purpose: Verifies wrong if-let chained operation assertions fail.
# Category: soundness
cat > "$TMPDIR/if_let_chain_soundness.rs" << 'ENDOFFILE'
enum MaybeInt {
    Some(i32),
    None,
}

fn if_let_chain_soundness_proof() {
    let v = MaybeInt::Some(21);
    let result = if let MaybeInt::Some(n) = v {
        n * 2
    } else {
        0
    };
    assert!(result == 21);  // MUST fail: result is 42
}
ENDOFFILE
run_test "if_let_chain_soundness" "$TMPDIR/if_let_chain_soundness.rs" "fail"


# Test 128b: Negation operator soundness
# Purpose: Verifies wrong negation assertions fail.
# Category: soundness
cat > "$TMPDIR/negation_soundness.rs" << 'ENDOFFILE'
fn negation_soundness_proof() {
    let x = 42i32;
    let neg_x = -x;
    assert!(neg_x == 42);  // MUST fail: neg_x is -42
}
ENDOFFILE
run_test "negation_soundness" "$TMPDIR/negation_soundness.rs" "fail"


# Test 130b: Multiple match arms soundness
# Purpose: Verifies wrong match arm result assertions fail.
# Category: soundness
cat > "$TMPDIR/match_same_result_soundness.rs" << 'ENDOFFILE'
fn classify(x: i32) -> i32 {
    match x {
        0 => 0,
        1 => 1,
        2 => 1,
        3 => 1,
        _ => 2,
    }
}

fn match_same_result_soundness_proof() {
    assert!(classify(2) == 2);  // MUST fail: classify(2) returns 1
}
ENDOFFILE
run_test "match_same_result_soundness" "$TMPDIR/match_same_result_soundness.rs" "fail"


# Test 132b: Where clause soundness
# Purpose: Verifies wrong generic identity assertions fail.
# Category: soundness
cat > "$TMPDIR/where_clause_soundness.rs" << 'ENDOFFILE'
fn identity<T>(x: T) -> T where T: Copy {
    x
}

fn where_clause_soundness_proof() {
    let a = identity(42i32);
    assert!(a == 0);  // MUST fail: a is 42
}
ENDOFFILE
run_test "where_clause_soundness" "$TMPDIR/where_clause_soundness.rs" "fail"


# Test 133b: Slice pattern soundness
# Purpose: Verifies wrong array destructure assertions fail (fixed #214).
# Category: soundness
cat > "$TMPDIR/slice_pattern_soundness.rs" << 'ENDOFFILE'
fn slice_pattern_soundness_proof() {
    let arr = [1, 2, 3];
    let [first, _second, _third] = arr;

    assert!(first == 999);  // first == 1, so this fails as expected
}
ENDOFFILE
run_test "slice_pattern_soundness" "$TMPDIR/slice_pattern_soundness.rs" "fail"  # Fixed in #214: now correctly constrained


# Test 134b: Derive Clone and Copy soundness
# Purpose: Verifies wrong copied struct field assertions fail.
# Category: soundness
cat > "$TMPDIR/derive_copy_soundness.rs" << 'ENDOFFILE'
#[derive(Clone, Copy)]
struct Point {
    x: i32,
    y: i32,
}

fn derive_copy_soundness_proof() {
    let p1 = Point { x: 10, y: 20 };
    let p2 = p1;
    assert!(p2.x == 20);  // MUST fail: p2.x is 10
}
ENDOFFILE
run_test "derive_copy_soundness" "$TMPDIR/derive_copy_soundness.rs" "fail"


# Test 135b: Exclusive range pattern soundness
# Purpose: Verifies wrong exclusive range match assertions fail.
# Category: soundness
cat > "$TMPDIR/exclusive_range_soundness.rs" << 'ENDOFFILE'
fn classify_digit(d: i32) -> i32 {
    match d {
        0..5 => 0,
        5..10 => 1,
        _ => 2,
    }
}

fn exclusive_range_soundness_proof() {
    assert!(classify_digit(5) == 0);  // MUST fail: 5 matches second arm (5..10)
}
ENDOFFILE
run_test "exclusive_range_soundness" "$TMPDIR/exclusive_range_soundness.rs" "fail"


# Test 138b: Derive Default soundness
# Purpose: Verifies Default::default() trait call fails (not unsound).
# Category: soundness
cat > "$TMPDIR/derive_default_soundness.rs" << 'ENDOFFILE'
#[derive(Default)]
struct Config {
    count: i32,
}

fn derive_default_soundness_proof() {
    let cfg: Config = Default::default();  // Trait method call
    assert!(cfg.count == 999);  // Verification fails (good - not unsound!)
}
ENDOFFILE
run_test "derive_default_soundness" "$TMPDIR/derive_default_soundness.rs" "fail"


# Test 140b: Numeric literals soundness
# Purpose: Verifies wrong numeric literal assertions fail.
# Category: soundness
cat > "$TMPDIR/numeric_literals_soundness.rs" << 'ENDOFFILE'
fn numeric_literals_soundness_proof() {
    let hex = 0xFF;
    assert!(hex == 256);  // MUST fail: 0xFF is 255
}
ENDOFFILE
run_test "numeric_literals_soundness" "$TMPDIR/numeric_literals_soundness.rs" "fail"


# Test 141b: Tuple indexing soundness
# Purpose: Verifies wrong tuple index assertions fail.
# Category: soundness
cat > "$TMPDIR/tuple_indexing_soundness.rs" << 'ENDOFFILE'
fn tuple_indexing_soundness_proof() {
    let pair = (10i32, 20i32);
    assert!(pair.0 == 20);  // MUST fail: pair.0 is 10
}
ENDOFFILE
run_test "tuple_indexing_soundness" "$TMPDIR/tuple_indexing_soundness.rs" "fail"


# Test 142b: usize/isize soundness
# Purpose: Verifies wrong usize/isize assertions fail.
# Category: soundness
cat > "$TMPDIR/usize_isize_soundness.rs" << 'ENDOFFILE'
fn usize_isize_soundness_proof() {
    let u: usize = 42;
    assert!(u == 0);  // MUST fail: u is 42
}
ENDOFFILE
run_test "usize_isize_soundness" "$TMPDIR/usize_isize_soundness.rs" "fail"


# Test 143b: Method chaining soundness
# Purpose: Verifies wrong method chaining sum assertions fail.
# Category: soundness
cat > "$TMPDIR/method_chaining_soundness.rs" << 'ENDOFFILE'
struct Counter {
    value: i32,
}

impl Counter {
    fn new() -> Counter {
        Counter { value: 0 }
    }

    fn add(&mut self, n: i32) {
        self.value += n;
    }

    fn get(&self) -> i32 {
        self.value
    }
}

fn method_chaining_soundness_proof() {
    let mut c = Counter::new();
    c.add(5);
    c.add(3);
    assert!(c.get() == 5);  // MUST fail: value is 8
}
ENDOFFILE
run_test "method_chaining_soundness" "$TMPDIR/method_chaining_soundness.rs" "fail"


# Test 144b: Underscore numbers soundness
# Purpose: Verifies wrong underscore number assertions fail.
# Category: soundness
cat > "$TMPDIR/underscore_numbers_soundness.rs" << 'ENDOFFILE'
fn underscore_numbers_soundness_proof() {
    let n = 1_000i32;
    assert!(n == 100);  // MUST fail: n is 1000
}
ENDOFFILE
run_test "underscore_numbers_soundness" "$TMPDIR/underscore_numbers_soundness.rs" "fail"


# Test 145b: Sequential mutable references soundness
# Purpose: Verifies wrong sequential mutation assertions fail.
# Category: soundness
cat > "$TMPDIR/sequential_mut_refs_soundness.rs" << 'ENDOFFILE'
struct Data {
    x: i32,
    y: i32,
}

impl Data {
    fn set_x(&mut self, val: i32) {
        self.x = val;
    }
}

fn sequential_mut_refs_soundness_proof() {
    let mut d = Data { x: 0, y: 0 };
    d.set_x(10);
    assert!(d.x == 0);  // MUST fail: d.x is now 10
}
ENDOFFILE
run_test "sequential_mut_refs_soundness" "$TMPDIR/sequential_mut_refs_soundness.rs" "fail"


# Test 146b: Reference field assignment soundness
# Purpose: Verifies wrong reference field assignment assertions fail.
# Category: soundness
cat > "$TMPDIR/ref_field_assign_soundness.rs" << 'ENDOFFILE'
struct Point {
    x: i32,
    y: i32,
}

fn modify_point(p: &mut Point) {
    p.x = 100;
}

fn ref_field_assign_soundness_proof() {
    let mut pt = Point { x: 0, y: 0 };
    modify_point(&mut pt);
    assert!(pt.x == 0);  // MUST fail: pt.x is now 100
}
ENDOFFILE
run_test "ref_field_assign_soundness" "$TMPDIR/ref_field_assign_soundness.rs" "fail"


# Test 147b: Nested expression calls soundness
# Purpose: Verifies wrong nested call assertions fail.
# Category: soundness
cat > "$TMPDIR/nested_expr_calls_soundness.rs" << 'ENDOFFILE'
fn add_five(x: i32) -> i32 { x + 5 }
fn add_one(x: i32) -> i32 { x + 1 }

fn nested_expr_calls_soundness_proof() {
    let result = add_one(add_five(3));  // 8 + 1 = 9
    assert!(result == 8);  // MUST fail: result is 9
}
ENDOFFILE
run_test "nested_expr_calls_soundness" "$TMPDIR/nested_expr_calls_soundness.rs" "fail"


# Test 148b: Zero-sized type soundness
# Purpose: Verifies wrong ZST method assertions fail.
# Category: soundness
cat > "$TMPDIR/zst_struct_soundness.rs" << 'ENDOFFILE'
struct Marker;

impl Marker {
    fn value(&self) -> i32 { 42 }
}

fn zst_struct_soundness_proof() {
    let m = Marker;
    assert!(m.value() == 0);  // MUST fail: value() returns 42
}
ENDOFFILE
run_test "zst_struct_soundness" "$TMPDIR/zst_struct_soundness.rs" "fail"


# Test 149b: Match on boolean soundness
# Purpose: Verifies wrong bool match assertions fail.
# Category: soundness
cat > "$TMPDIR/match_bool_soundness.rs" << 'ENDOFFILE'
fn bool_to_int(b: bool) -> i32 {
    match b {
        true => 1,
        false => 0,
    }
}

fn match_bool_soundness_proof() {
    assert!(bool_to_int(true) == 0);  // MUST fail: returns 1
}
ENDOFFILE
run_test "match_bool_soundness" "$TMPDIR/match_bool_soundness.rs" "fail"


# Test 150b: Multiple assignment soundness
# Purpose: Verifies wrong multiple assignment assertions fail.
# Category: soundness
cat > "$TMPDIR/multiple_assign_soundness.rs" << 'ENDOFFILE'
fn multiple_assign_soundness_proof() {
    let mut x = 1i32;
    x = 2;
    x = 3;
    assert!(x == 2);  // MUST fail: x is 3
}
ENDOFFILE
run_test "multiple_assign_soundness" "$TMPDIR/multiple_assign_soundness.rs" "fail"


# Test 151b: Explicit type soundness
# Purpose: Verifies wrong explicit type assertions fail.
# Category: soundness
cat > "$TMPDIR/explicit_type_soundness.rs" << 'ENDOFFILE'
fn explicit_type_soundness_proof() {
    let a: i32 = 10;
    assert!(a == 20);  // MUST fail: a is 10
}
ENDOFFILE
run_test "explicit_type_soundness" "$TMPDIR/explicit_type_soundness.rs" "fail"


# Test 152b: Field through mutable reference soundness
# Purpose: Verifies wrong field access through mut ref assertions fail.
# Category: soundness
cat > "$TMPDIR/field_through_mut_ref_soundness.rs" << 'ENDOFFILE'
struct Pair {
    a: i32,
    b: i32,
}

fn get_a(p: &mut Pair) -> i32 {
    p.a
}

fn field_through_mut_ref_soundness_proof() {
    let mut pair = Pair { a: 10, b: 20 };
    let a = get_a(&mut pair);
    assert!(a == 20);  // MUST fail: a is 10
}
ENDOFFILE
run_test "field_through_mut_ref_soundness" "$TMPDIR/field_through_mut_ref_soundness.rs" "fail"


# Test 153b: Integer comparison soundness
# Purpose: Verifies wrong comparison assertions fail.
# Category: soundness
cat > "$TMPDIR/int_comparisons_soundness.rs" << 'ENDOFFILE'
fn int_comparisons_soundness_proof() {
    let a = 10i32;
    let b = 20i32;
    assert!(a > b);  // MUST fail: 10 is not > 20
}
ENDOFFILE
run_test "int_comparisons_soundness" "$TMPDIR/int_comparisons_soundness.rs" "fail"


# Test 154b: Parameter arithmetic soundness
# Purpose: Verifies wrong parameter arithmetic assertions fail.
# Category: soundness
cat > "$TMPDIR/param_arithmetic_soundness.rs" << 'ENDOFFILE'
fn subtract(a: i32, b: i32) -> i32 {
    a - b
}

fn param_arithmetic_soundness_proof() {
    let result = subtract(10, 3);
    assert!(result == 3);  // MUST fail: 10 - 3 = 7
}
ENDOFFILE
run_test "param_arithmetic_soundness" "$TMPDIR/param_arithmetic_soundness.rs" "fail"


# Test 155b: Char type soundness (fixed in #223)
# Purpose: Verifies wrong char equality assertions fail.
# Category: soundness
cat > "$TMPDIR/char_type_soundness.rs" << 'ENDOFFILE'
fn char_type_soundness_proof() {
    let c: char = 'A';
    assert!(c == 'B');  // Should fail - 'A' != 'B'
}
ENDOFFILE
run_test "char_type_soundness" "$TMPDIR/char_type_soundness.rs" "fail"  # Soundness: correctly detects inequality


# Test 156b: Unsigned arithmetic soundness
# Purpose: Verifies wrong u32 subtraction assertions fail.
# Category: soundness
cat > "$TMPDIR/unsigned_arith_soundness.rs" << 'ENDOFFILE'
fn unsigned_arith_soundness_proof() {
    let a: u32 = 100;
    let b: u32 = 50;
    let diff = a - b;
    assert!(diff == 100);  // MUST fail: 100 - 50 = 50
}
ENDOFFILE
run_test "unsigned_arith_soundness" "$TMPDIR/unsigned_arith_soundness.rs" "fail"


# Test 157b: Associated function soundness
# Purpose: Verifies wrong associated function return value assertions fail.
# Category: soundness
cat > "$TMPDIR/assoc_fn_soundness.rs" << 'ENDOFFILE'
struct Math;

impl Math {
    fn double(x: i32) -> i32 {
        x * 2
    }
}

fn assoc_fn_soundness_proof() {
    let result = Math::double(5);
    assert!(result == 5);  // MUST fail: 5 * 2 = 10
}
ENDOFFILE
run_test "assoc_fn_soundness" "$TMPDIR/assoc_fn_soundness.rs" "fail"


# Test 159b: Multiple struct instances soundness
# Purpose: Verifies struct instances maintain separate field values.
# Category: soundness
cat > "$TMPDIR/multi_struct_soundness.rs" << 'ENDOFFILE'
struct Rect {
    w: i32,
    h: i32,
}

fn multi_struct_soundness_proof() {
    let r1 = Rect { w: 10, h: 20 };
    let r2 = Rect { w: 30, h: 40 };
    assert!(r1.w == r2.w);  // MUST fail: 10 != 30
}
ENDOFFILE
run_test "multi_struct_soundness" "$TMPDIR/multi_struct_soundness.rs" "fail"


# Test 161b: Single-variant enum soundness
# Purpose: Verifies wrong single-variant enum value assertions fail.
# Category: soundness
cat > "$TMPDIR/single_variant_soundness.rs" << 'ENDOFFILE'
enum Single {
    Only(i32),
}

fn get_value(s: Single) -> i32 {
    match s {
        Single::Only(v) => v,
    }
}

fn single_variant_soundness_proof() {
    let s = Single::Only(42);
    let v = get_value(s);
    assert!(v == 100);  // MUST fail: v is 42
}
ENDOFFILE
run_test "single_variant_soundness" "$TMPDIR/single_variant_soundness.rs" "fail"


# Test 162b: Logical condition soundness
# Purpose: Verifies && correctly short-circuits when second condition is false.
# Category: soundness
cat > "$TMPDIR/logical_cond_soundness.rs" << 'ENDOFFILE'
fn logical_cond_soundness_proof() {
    let a = 5i32;
    let b = -1i32;

    let result = if a > 0 && b > 0 {
        100
    } else {
        0
    };

    assert!(result == 100);  // MUST fail: b > 0 is false
}
ENDOFFILE
run_test "logical_cond_soundness" "$TMPDIR/logical_cond_soundness.rs" "fail"


# Test 163b: Multi-data enum soundness
# Purpose: Verifies wrong multi-data enum perimeter calculation fails.
# Category: soundness
cat > "$TMPDIR/multi_data_enum_soundness.rs" << 'ENDOFFILE'
enum Shape {
    Square(i32),
    Rectangle(i32, i32),
}

fn perimeter(s: Shape) -> i32 {
    match s {
        Shape::Square(side) => side * 4,
        Shape::Rectangle(w, h) => w * 2 + h * 2,
    }
}

fn multi_data_enum_soundness_proof() {
    let square = Shape::Square(10);
    let p = perimeter(square);
    assert!(p == 10);  // MUST fail: 10*4 = 40
}
ENDOFFILE
run_test "multi_data_enum_soundness" "$TMPDIR/multi_data_enum_soundness.rs" "fail"


# Test 164b: Boolean field soundness
# Purpose: Verifies wrong boolean field assertions fail.
# Category: soundness
cat > "$TMPDIR/bool_field_soundness.rs" << 'ENDOFFILE'
struct Config {
    enabled: bool,
    value: i32,
}

fn bool_field_soundness_proof() {
    let c = Config { enabled: false, value: 42 };
    assert!(c.enabled);  // MUST fail: enabled is false
}
ENDOFFILE
run_test "bool_field_soundness" "$TMPDIR/bool_field_soundness.rs" "fail"


# Test 165b: Integer bounds soundness
# Purpose: Verifies wrong sign assertions on i32::MAX fail.
# Category: soundness
cat > "$TMPDIR/int_bounds_soundness.rs" << 'ENDOFFILE'
fn int_bounds_soundness_proof() {
    let max = i32::MAX;
    assert!(max < 0);  // MUST fail: MAX is positive
}
ENDOFFILE
run_test "int_bounds_soundness" "$TMPDIR/int_bounds_soundness.rs" "fail"


# Test 165d: Boundary arithmetic - additive identity soundness
# Purpose: Verifies wrong additive identity assertions fail.
# Category: soundness
cat > "$TMPDIR/boundary_add_zero_soundness.rs" << 'ENDOFFILE'
fn boundary_add_zero_soundness_proof() {
    let x: i32 = 42;
    // WRONG: x + 0 is 42, not 100
    assert!(x + 0 == 100);
}
ENDOFFILE
run_test "boundary_add_zero_soundness" "$TMPDIR/boundary_add_zero_soundness.rs" "fail" "soundness"


# Test 165f: Boundary arithmetic - subtractive identity soundness
# Purpose: Verifies wrong subtractive identity assertions fail.
# Category: soundness
cat > "$TMPDIR/boundary_sub_zero_soundness.rs" << 'ENDOFFILE'
fn boundary_sub_zero_soundness_proof() {
    let x: i32 = 42;
    // WRONG: x - 0 is 42, not 0
    assert!(x - 0 == 0);
}
ENDOFFILE
run_test "boundary_sub_zero_soundness" "$TMPDIR/boundary_sub_zero_soundness.rs" "fail" "soundness"


# Test 165j: Boundary arithmetic - MIN wrapping soundness
# Purpose: Verifies wrong MIN boundary assertions fail.
# Category: soundness
cat > "$TMPDIR/boundary_min_wrapping_soundness.rs" << 'ENDOFFILE'
fn boundary_min_wrapping_soundness_proof() {
    let min: i32 = -2147483648;
    let result = min.wrapping_sub(1);
    // WRONG: wrapping at MIN goes to MAX, not 0
    assert!(result == 0);
}
ENDOFFILE
run_test "boundary_min_wrapping_soundness" "$TMPDIR/boundary_min_wrapping_soundness.rs" "fail" "soundness"


# Test 166b: Char field soundness
# Purpose: Verifies wrong char field assertions fail.
# Category: soundness
cat > "$TMPDIR/char_field_soundness.rs" << 'ENDOFFILE'
struct Letter {
    c: char,
    index: i32,
}

fn char_field_soundness_proof() {
    let l = Letter { c: 'A', index: 0 };
    assert!(l.c == 'B');  // MUST fail: char is 'A', not 'B'
}
ENDOFFILE
run_test "char_field_soundness" "$TMPDIR/char_field_soundness.rs" "fail"


# Test 167b: Empty struct soundness
# Purpose: Verifies wrong empty struct associated function assertions fail.
# Category: soundness
cat > "$TMPDIR/empty_struct_soundness.rs" << 'ENDOFFILE'
struct Empty {}

impl Empty {
    fn value() -> i32 {
        42
    }
}

fn empty_struct_soundness_proof() {
    let v = Empty::value();
    assert!(v == 0);  // MUST fail: v is 42
}
ENDOFFILE
run_test "empty_struct_soundness" "$TMPDIR/empty_struct_soundness.rs" "fail"


# Test 168b: While compound soundness
# Purpose: Verifies wrong while loop termination assertions fail.
# Category: soundness
cat > "$TMPDIR/while_compound_soundness.rs" << 'ENDOFFILE'
fn while_compound_soundness_proof() {
    let mut i = 0i32;
    while i < 5 && i >= 0 {
        i += 1;
    }
    assert!(i == 3);  // MUST fail: i is 5
}
ENDOFFILE
run_test "while_compound_soundness" "$TMPDIR/while_compound_soundness.rs" "fail"


# Test 170b: For loop soundness
# Purpose: Verifies for loop iteration bounds are correctly modeled.
# Category: soundness
cat > "$TMPDIR/for_loop_soundness.rs" << 'ENDOFFILE'
fn for_loop_soundness_proof() {
    let mut count = 0i32;
    for _ in 0..3 {
        count += 1;
    }
    assert!(count == 4);  // MUST fail: 3 iterations gives count=3, not 4
}
ENDOFFILE
run_test "for_loop_soundness" "$TMPDIR/for_loop_soundness.rs" "fail"


# Test 173b: Raw pointer soundness
# Purpose: Verifies raw pointer context with assert!(false) fails.
# Category: soundness
cat > "$TMPDIR/raw_pointer_soundness.rs" << 'ENDOFFILE'
fn raw_pointer_soundness_proof() {
    let x = 42i32;
    let _ptr = &x as *const i32;
    assert!(false);  // Expected to fail
}
ENDOFFILE
run_test "raw_pointer_soundness" "$TMPDIR/raw_pointer_soundness.rs" "fail"


# Test 174b: Division soundness
# Purpose: Verifies wrong integer division assertions fail.
# Category: soundness
cat > "$TMPDIR/division_soundness.rs" << 'ENDOFFILE'
fn division_soundness_proof() {
    let a = 10i32;
    let b = 3i32;
    let q = a / b;
    assert!(q == 4);  // MUST fail: 10/3 = 3, not 4
}
ENDOFFILE
run_test "division_soundness" "$TMPDIR/division_soundness.rs" "fail"


# Test 175b: Modulo soundness
# Purpose: Verifies wrong modulo assertions fail.
# Category: soundness
cat > "$TMPDIR/modulo_soundness.rs" << 'ENDOFFILE'
fn modulo_soundness_proof() {
    let a = 10i32;
    let b = 3i32;
    let r = a % b;
    assert!(r == 2);  // MUST fail: 10%3 = 1, not 2
}
ENDOFFILE
run_test "modulo_soundness" "$TMPDIR/modulo_soundness.rs" "fail"


# Test 176b: Bitwise NOT soundness
# Purpose: Verifies wrong bitwise NOT assertions fail.
# Category: soundness
cat > "$TMPDIR/bitwise_not_soundness.rs" << 'ENDOFFILE'
fn bitwise_not_soundness_proof() {
    let x = 0i32;
    let y = !x;
    assert!(y == 0);  // MUST fail: !0 = -1, not 0
}
ENDOFFILE
run_test "bitwise_not_soundness" "$TMPDIR/bitwise_not_soundness.rs" "fail"


# Test 177b: Abs fn soundness
# Purpose: Verifies wrong abs value function assertions fail.
# Category: soundness
cat > "$TMPDIR/abs_fn_soundness.rs" << 'ENDOFFILE'
fn abs(x: i32) -> i32 {
    if x >= 0 {
        return x;
    }
    return -x;
}

fn abs_fn_soundness_proof() {
    let a = abs(5);
    assert!(a == 10);  // MUST fail: abs(5) = 5, not 10
}
ENDOFFILE
run_test "abs_fn_soundness" "$TMPDIR/abs_fn_soundness.rs" "fail"


# Test 178b: Option Some soundness
# Purpose: Verifies wrong Option::Some extraction assertions fail.
# Category: soundness
cat > "$TMPDIR/option_some_soundness.rs" << 'ENDOFFILE'
fn option_some_soundness_proof() {
    let opt: Option<i32> = Some(42);
    let val = match opt {
        Some(v) => v,
        None => 0,
    };
    assert!(val == 0);  // MUST fail: val is 42, not 0
}
ENDOFFILE
run_test "option_some_soundness" "$TMPDIR/option_some_soundness.rs" "fail"


# Test 179b: Option None soundness
# Purpose: Verifies wrong Option::None default assertions fail.
# Category: soundness
cat > "$TMPDIR/option_none_soundness.rs" << 'ENDOFFILE'
fn option_none_soundness_proof() {
    let opt: Option<i32> = None;
    let val = match opt {
        Some(v) => v,
        None => -1,
    };
    assert!(val == 42);  // MUST fail: val is -1, not 42
}
ENDOFFILE
run_test "option_none_soundness" "$TMPDIR/option_none_soundness.rs" "fail"


# Test 180b: Result Ok soundness
# Purpose: Verifies wrong Result::Ok extraction assertions fail.
# Category: soundness
cat > "$TMPDIR/result_ok_soundness.rs" << 'ENDOFFILE'
fn result_ok_soundness_proof() {
    let res: Result<i32, i32> = Ok(42);
    let val = match res {
        Ok(v) => v,
        Err(_) => 0,
    };
    assert!(val == 0);  // MUST fail: val is 42, not 0
}
ENDOFFILE
run_test "result_ok_soundness" "$TMPDIR/result_ok_soundness.rs" "fail"


# Test 181b: Result Err soundness
# Purpose: Verifies wrong Result::Err extraction assertions fail.
# Category: soundness
cat > "$TMPDIR/result_err_soundness.rs" << 'ENDOFFILE'
fn result_err_soundness_proof() {
    let res: Result<i32, i32> = Err(99);
    let val = match res {
        Ok(_) => 0,
        Err(e) => e,
    };
    assert!(val == 0);  // MUST fail: val is 99, not 0
}
ENDOFFILE
run_test "result_err_soundness" "$TMPDIR/result_err_soundness.rs" "fail"


# Test 182b: Infinite loop soundness
# Purpose: Verifies wrong loop termination value assertions fail.
# Category: soundness
cat > "$TMPDIR/infinite_loop_soundness.rs" << 'ENDOFFILE'
fn infinite_loop_soundness_proof() {
    let mut i = 0i32;
    loop {
        i += 1;
        if i >= 5 {
            break;
        }
    }
    assert!(i == 10);  // MUST fail: i is 5, not 10
}
ENDOFFILE
run_test "infinite_loop_soundness" "$TMPDIR/infinite_loop_soundness.rs" "fail"


# Test 183b: Nested loops soundness
# Purpose: Verifies wrong nested loop total assertions fail.
# Category: soundness
cat > "$TMPDIR/nested_loops_soundness.rs" << 'ENDOFFILE'
fn nested_loops_soundness_proof() {
    let mut outer = 0i32;
    let mut total = 0i32;
    while outer < 3 {
        let mut inner = 0i32;
        while inner < 2 {
            total += 1;
            inner += 1;
        }
        outer += 1;
    }
    assert!(total == 5);  // MUST fail: total is 6, not 5
}
ENDOFFILE
run_test "nested_loops_soundness" "$TMPDIR/nested_loops_soundness.rs" "fail"


# Test 184b: Recursion soundness
# Purpose: Verifies assert!(false) with recursion context fails.
# Category: soundness
cat > "$TMPDIR/recursion_soundness.rs" << 'ENDOFFILE'
fn factorial(n: i32) -> i32 {
    if n <= 1 {
        1
    } else {
        n * factorial(n - 1)
    }
}

fn recursion_soundness_proof() {
    let _f = factorial(5);
    assert!(false);  // Expected to fail
}
ENDOFFILE
run_test "recursion_soundness" "$TMPDIR/recursion_soundness.rs" "fail"


# Test 185b: Static variable soundness
# Purpose: Verifies assert!(false) with static variable context fails.
# Category: soundness
cat > "$TMPDIR/static_var_soundness.rs" << 'ENDOFFILE'
static VALUE: i32 = 42;

fn static_var_soundness_proof() {
    let _v = VALUE;
    assert!(false);  // Expected to fail
}
ENDOFFILE
run_test "static_var_soundness" "$TMPDIR/static_var_soundness.rs" "fail"


# Test 186b: Const variable soundness
# Purpose: Verifies wrong const variable assertions fail.
# Category: soundness
cat > "$TMPDIR/const_var_soundness.rs" << 'ENDOFFILE'
const VALUE: i32 = 42;

fn const_var_soundness_proof() {
    let v = VALUE;
    assert!(v == 0);  // MUST fail: v is 42, not 0
}
ENDOFFILE
run_test "const_var_soundness" "$TMPDIR/const_var_soundness.rs" "fail"


# Test 187b: String literal soundness
# Purpose: Verifies assert!(false) with string context fails.
# Category: soundness
cat > "$TMPDIR/string_literal_soundness.rs" << 'ENDOFFILE'
fn string_literal_soundness_proof() {
    let _s = "hello";
    assert!(false);  // Expected to fail
}
ENDOFFILE
run_test "string_literal_soundness" "$TMPDIR/string_literal_soundness.rs" "fail"


# Test 188b: Array const size soundness
# Purpose: Verifies wrong array element assertions fail.
# Category: soundness
cat > "$TMPDIR/array_const_size_soundness.rs" << 'ENDOFFILE'
const SIZE: usize = 3;

fn array_const_size_soundness_proof() {
    let arr: [i32; SIZE] = [1, 2, 3];
    let first = arr[0];
    assert!(first == 99);  // MUST fail: first is 1, not 99
}
ENDOFFILE
run_test "array_const_size_soundness" "$TMPDIR/array_const_size_soundness.rs" "fail"


# Test 189b: Mixed tuple soundness
# Purpose: Verifies wrong tuple element assertions fail.
# Category: soundness
cat > "$TMPDIR/mixed_tuple_soundness.rs" << 'ENDOFFILE'
fn mixed_tuple_soundness_proof() {
    let t: (i32, bool, i32) = (42, true, 7);
    let a = t.0;
    assert!(a == 0);  // MUST fail: a is 42, not 0
}
ENDOFFILE
run_test "mixed_tuple_soundness" "$TMPDIR/mixed_tuple_soundness.rs" "fail"


# Test 190b: Expression block soundness
# Purpose: Verifies wrong expression block result assertions fail.
# Category: soundness
cat > "$TMPDIR/expr_block_soundness.rs" << 'ENDOFFILE'
fn expr_block_soundness_proof() {
    let x = {
        let a = 5i32;
        let b = 3i32;
        a + b
    };
    assert!(x == 0);  // MUST fail: x is 8, not 0
}
ENDOFFILE
run_test "expr_block_soundness" "$TMPDIR/expr_block_soundness.rs" "fail"


# Test 192b: Diverging soundness
# Purpose: Verifies wrong path-reachability assertions fail.
# Category: soundness
cat > "$TMPDIR/diverging_soundness.rs" << 'ENDOFFILE'
fn never_returns() -> ! {
    loop {}
}

fn diverging_soundness_proof() {
    let x = 5i32;
    if x < 0 {
        never_returns();
    }
    assert!(x < 0);  // MUST fail: x is 5, which is >= 0
}
ENDOFFILE
run_test "diverging_soundness" "$TMPDIR/diverging_soundness.rs" "fail"


# Test 193b: Type cast soundness
# Purpose: Verifies wrong type cast result assertions fail.
# Category: soundness
cat > "$TMPDIR/type_cast_soundness.rs" << 'ENDOFFILE'
fn type_cast_soundness_proof() {
    let x: i32 = 42;
    let y: i64 = x as i64;
    let z: i32 = y as i32;
    assert!(z == 0);  // MUST fail: z is 42, not 0
}
ENDOFFILE
run_test "type_cast_soundness" "$TMPDIR/type_cast_soundness.rs" "fail"


# Test 194b: Multi type struct soundness
# Purpose: Verifies wrong multi-type struct field assertions fail.
# Category: soundness
cat > "$TMPDIR/multi_type_struct_soundness.rs" << 'ENDOFFILE'
struct MultiType {
    int_val: i32,
    bool_val: bool,
    other_int: i64,
}

fn multi_type_struct_soundness_proof() {
    let s = MultiType {
        int_val: 42,
        bool_val: true,
        other_int: 100,
    };
    assert!(s.int_val == 0);  // MUST fail: int_val is 42, not 0
}
ENDOFFILE
run_test "multi_type_struct_soundness" "$TMPDIR/multi_type_struct_soundness.rs" "fail"


# Test 196b: unreachable! soundness
# Purpose: Verifies unreachable!() path triggers panic correctly.
# Category: soundness
cat > "$TMPDIR/unreachable_soundness.rs" << 'ENDOFFILE'
fn unreachable_soundness_proof() {
    let x = 0i32;  // Will hit the unreachable! path
    let result = if x > 0 {
        x * 2
    } else if x < 0 {
        x * -2
    } else {
        unreachable!()  // This path IS reachable with x = 0
    };
    assert!(result == 0);  // MUST fail: unreachable! panics
}
ENDOFFILE
run_test "unreachable_soundness" "$TMPDIR/unreachable_soundness.rs" "fail"


# Test 197b: panic! soundness
# Purpose: Verifies panic! path triggers correctly on invalid input.
# Category: soundness
cat > "$TMPDIR/panic_soundness.rs" << 'ENDOFFILE'
fn add_if_positive(a: i32, b: i32) -> i32 {
    if b <= 0 {
        panic!("b must be positive")
    } else {
        a + b
    }
}

fn panic_soundness_proof() {
    let result = add_if_positive(10, 0);  // MUST fail: panic!
    assert!(result == 0);
}
ENDOFFILE
run_test "panic_soundness" "$TMPDIR/panic_soundness.rs" "fail"


# Test 198b: #[inline] soundness
# Purpose: Verifies wrong inline function result assertions fail.
# Category: soundness
cat > "$TMPDIR/inline_attr_soundness.rs" << 'ENDOFFILE'
#[inline]
fn inline_add(a: i32, b: i32) -> i32 {
    a + b
}

fn inline_attr_soundness_proof() {
    let sum = inline_add(3, 4);
    assert!(sum == 0);  // MUST fail: sum is 7
}
ENDOFFILE
run_test "inline_attr_soundness" "$TMPDIR/inline_attr_soundness.rs" "fail"


# Test 200b: Turbofish soundness
# Purpose: Verifies wrong generic function result assertions fail.
# Category: soundness
cat > "$TMPDIR/turbofish_soundness.rs" << 'ENDOFFILE'
fn identity<T>(x: T) -> T {
    x
}

fn turbofish_soundness_proof() {
    let x = identity::<i32>(42);
    assert!(x == 0);  // MUST fail: x is 42
}
ENDOFFILE
run_test "turbofish_soundness" "$TMPDIR/turbofish_soundness.rs" "fail"


# Test 201b: #[repr(C)] soundness
# Purpose: Verifies wrong repr(C) struct field assertions fail.
# Category: soundness
cat > "$TMPDIR/repr_c_soundness.rs" << 'ENDOFFILE'
#[repr(C)]
struct CPoint {
    x: i32,
    y: i32,
}

fn repr_c_soundness_proof() {
    let p = CPoint { x: 10, y: 20 };
    assert!(p.x == 0);  // MUST fail: x is 10
}
ENDOFFILE
run_test "repr_c_soundness" "$TMPDIR/repr_c_soundness.rs" "fail"


# Test 202b: pub visibility soundness
# Purpose: Verifies wrong pub function result assertions fail.
# Category: soundness
cat > "$TMPDIR/pub_visibility_soundness.rs" << 'ENDOFFILE'
pub struct PublicStruct {
    pub value: i32,
}

pub fn public_fn(x: i32) -> i32 {
    x * 2
}

fn pub_visibility_soundness_proof() {
    let s = PublicStruct { value: 21 };
    let result = public_fn(s.value);
    assert!(result == 0);  // MUST fail: result is 42
}
ENDOFFILE
run_test "pub_visibility_soundness" "$TMPDIR/pub_visibility_soundness.rs" "fail"


# Test 203b: Struct field shorthand soundness
# Purpose: Verifies wrong shorthand initialized field assertions fail.
# Category: soundness
cat > "$TMPDIR/field_shorthand_soundness.rs" << 'ENDOFFILE'
struct Point {
    x: i32,
    y: i32,
}

fn field_shorthand_soundness_proof() {
    let x = 10i32;
    let y = 20i32;
    let p = Point { x, y };
    assert!(p.x == 0);  // MUST fail: x is 10
}
ENDOFFILE
run_test "field_shorthand_soundness" "$TMPDIR/field_shorthand_soundness.rs" "fail"


# Test 204b: Module definition soundness
# Purpose: Verifies wrong module function result assertions fail.
# Category: soundness
cat > "$TMPDIR/module_def_soundness.rs" << 'ENDOFFILE'
mod math {
    pub fn add(a: i32, b: i32) -> i32 {
        a + b
    }
}

fn module_def_soundness_proof() {
    let result = math::add(3, 4);
    assert!(result == 0);  // MUST fail: result is 7
}
ENDOFFILE
run_test "module_def_soundness" "$TMPDIR/module_def_soundness.rs" "fail"


# Test 205b: Nested module soundness
# Purpose: Verifies wrong nested module function result assertions fail.
# Category: soundness
cat > "$TMPDIR/nested_module_soundness.rs" << 'ENDOFFILE'
mod outer {
    pub mod inner {
        pub fn add(a: i32, b: i32) -> i32 {
            a + b
        }
    }
}

fn nested_module_soundness_proof() {
    let result = outer::inner::add(20, 22);
    assert!(result == 0);  // MUST fail: result is 42
}
ENDOFFILE
run_test "nested_module_soundness" "$TMPDIR/nested_module_soundness.rs" "fail"


# Test 206b: Use alias soundness
# Purpose: Verifies wrong aliased function result assertions fail.
# Category: soundness
cat > "$TMPDIR/use_alias_soundness.rs" << 'ENDOFFILE'
mod operations {
    pub fn long_function_name(x: i32) -> i32 {
        x * 3
    }
}

use operations::long_function_name as triple;

fn use_alias_soundness_proof() {
    let result = triple(14);
    assert!(result == 0);  // MUST fail: result is 42
}
ENDOFFILE
run_test "use_alias_soundness" "$TMPDIR/use_alias_soundness.rs" "fail"


# Test 207b: Implicit return soundness
# Purpose: Verifies wrong implicit return value assertions fail.
# Category: soundness
cat > "$TMPDIR/implicit_return_soundness.rs" << 'ENDOFFILE'
fn compute(x: i32) -> i32 {
    if x > 0 {
        x * 2
    } else {
        0
    }
}

fn implicit_return_soundness_proof() {
    let result = compute(21);
    assert!(result == 0);  // MUST fail: result is 42
}
ENDOFFILE
run_test "implicit_return_soundness" "$TMPDIR/implicit_return_soundness.rs" "fail"


# Test 208b: Private fields soundness
# Purpose: Verifies wrong private field getter assertions fail.
# Category: soundness
cat > "$TMPDIR/private_fields_soundness.rs" << 'ENDOFFILE'
struct Counter {
    value: i32,
}

impl Counter {
    fn new(initial: i32) -> Self {
        Counter { value: initial }
    }

    fn get(&self) -> i32 {
        self.value
    }
}

fn private_fields_soundness_proof() {
    let c = Counter::new(42);
    assert!(c.get() == 0);  // MUST fail: value is 42
}
ENDOFFILE
run_test "private_fields_soundness" "$TMPDIR/private_fields_soundness.rs" "fail"


# Test 210b: Associated type soundness
# Purpose: Verifies wrong associated type struct field assertions fail.
# Category: soundness
cat > "$TMPDIR/assoc_type_soundness.rs" << 'ENDOFFILE'
trait Container {
    type Item;
    fn get_value(&self) -> i32;
}

struct IntBox {
    value: i32,
}

impl Container for IntBox {
    type Item = i32;
    fn get_value(&self) -> i32 {
        self.value
    }
}

fn assoc_type_soundness_proof() {
    let b = IntBox { value: 42 };
    assert!(b.value == 0);  // MUST fail: value is 42
}
ENDOFFILE
run_test "assoc_type_soundness" "$TMPDIR/assoc_type_soundness.rs" "fail"


# Test 211b: impl Trait soundness
# Purpose: Verifies wrong struct field computation assertions fail.
# Category: soundness
cat > "$TMPDIR/impl_trait_ret_soundness.rs" << 'ENDOFFILE'
struct Number {
    value: i32,
}

fn impl_trait_ret_soundness_proof() {
    let num = Number { value: 41 };
    assert!(num.value + 1 == 0);  // MUST fail: result is 42
}
ENDOFFILE
run_test "impl_trait_ret_soundness" "$TMPDIR/impl_trait_ret_soundness.rs" "fail"


# Test 212b: Multi trait impl soundness
# Purpose: Verifies wrong multi-trait field arithmetic assertions fail.
# Category: soundness
cat > "$TMPDIR/multi_trait_impl_soundness.rs" << 'ENDOFFILE'
struct Value {
    n: i32,
}

fn multi_trait_impl_soundness_proof() {
    let v = Value { n: 10 };
    assert!(v.n + 5 == 0);  // MUST fail: result is 15
}
ENDOFFILE
run_test "multi_trait_impl_soundness" "$TMPDIR/multi_trait_impl_soundness.rs" "fail"


# Test 213b: Static vs instance soundness
# Purpose: Verifies wrong static method result assertions fail.
# Category: soundness
cat > "$TMPDIR/static_vs_instance_soundness.rs" << 'ENDOFFILE'
struct Calculator;

impl Calculator {
    fn static_add(a: i32, b: i32) -> i32 {
        a + b
    }
}

fn static_vs_instance_soundness_proof() {
    let sum = Calculator::static_add(20, 22);
    assert!(sum == 0);  // MUST fail: sum is 42
}
ENDOFFILE
run_test "static_vs_instance_soundness" "$TMPDIR/static_vs_instance_soundness.rs" "fail"


# Test 215b: Trailing comma soundness
# Purpose: Verifies wrong trailing comma struct field assertions fail.
# Category: soundness
cat > "$TMPDIR/trailing_comma_soundness.rs" << 'ENDOFFILE'
struct Point {
    x: i32,
    y: i32,
}

fn trailing_comma_soundness_proof() {
    let p = Point {
        x: 10,
        y: 20,
    };
    assert!(p.x + p.y == 0);  // MUST fail: sum is 30
}
ENDOFFILE
run_test "trailing_comma_soundness" "$TMPDIR/trailing_comma_soundness.rs" "fail"


# Test 216b: Type alias soundness
# Purpose: Verifies wrong type aliased variable assertions fail.
# Category: soundness
cat > "$TMPDIR/type_alias_prim_soundness.rs" << 'ENDOFFILE'
type Integer = i32;

fn type_alias_prim_soundness_proof() {
    let x: Integer = 42;
    assert!(x == 0);  // MUST fail: x is 42
}
ENDOFFILE
run_test "type_alias_prim_soundness" "$TMPDIR/type_alias_prim_soundness.rs" "fail"


# Test 217b: Doc comment soundness
# Purpose: Verifies wrong documented function result assertions fail.
# Category: soundness
cat > "$TMPDIR/doc_comment_soundness.rs" << 'ENDOFFILE'
/// Adds two numbers
fn documented_add(a: i32, b: i32) -> i32 {
    a + b
}

fn doc_comment_soundness_proof() {
    let result = documented_add(20, 22);
    assert!(result == 0);  // MUST fail: result is 42
}
ENDOFFILE
run_test "doc_comment_soundness" "$TMPDIR/doc_comment_soundness.rs" "fail"


# Test 218b: Statement attribute soundness
# Purpose: Verifies wrong statement-attributed variable assertions fail.
# Category: soundness
cat > "$TMPDIR/stmt_attr_soundness.rs" << 'ENDOFFILE'
fn stmt_attr_soundness_proof() {
    #[allow(unused_variables)]
    let _unused = 42i32;

    let used = 10i32;
    assert!(used == 0);  // MUST fail: used is 10
}
ENDOFFILE
run_test "stmt_attr_soundness" "$TMPDIR/stmt_attr_soundness.rs" "fail"


# Test 219b: Never type soundness
# Purpose: Verifies divergent path triggers panic correctly.
# Category: soundness
cat > "$TMPDIR/never_type_soundness.rs" << 'ENDOFFILE'
fn always_panics() -> ! {
    panic!("this always panics")
}

fn safe_check(x: i32) -> i32 {
    if x >= 0 {
        x
    } else {
        always_panics()
    }
}

fn never_type_soundness_proof() {
    let result = safe_check(-1);  // Will call always_panics
    assert!(result == 0);  // MUST fail: panics before reaching here
}
ENDOFFILE
run_test "never_type_soundness" "$TMPDIR/never_type_soundness.rs" "fail"


# Test 220b: Self return soundness
# Purpose: Verifies wrong Self-returning constructor assertions fail.
# Category: soundness
cat > "$TMPDIR/self_return_soundness.rs" << 'ENDOFFILE'
struct Builder {
    value: i32,
}

impl Builder {
    fn with_value(value: i32) -> Self {
        Builder { value }
    }
}

fn self_return_soundness_proof() {
    let b = Builder::with_value(42);
    assert!(b.value == 0);  // MUST fail: value is 42
}
ENDOFFILE
run_test "self_return_soundness" "$TMPDIR/self_return_soundness.rs" "fail"


# Test 221b: #[must_use] soundness
# Purpose: Verifies wrong must_use function result assertions fail.
# Category: soundness
cat > "$TMPDIR/must_use_soundness.rs" << 'ENDOFFILE'
#[must_use]
fn compute_value(x: i32) -> i32 {
    x * 2
}

fn must_use_soundness_proof() {
    let result = compute_value(21);
    assert!(result == 0);  // MUST fail: result is 42
}
ENDOFFILE
run_test "must_use_soundness" "$TMPDIR/must_use_soundness.rs" "fail"


# Test 222b: let-else enum soundness
# Purpose: Verifies wrong let-else else-branch result assertions fail.
# Category: soundness
cat > "$TMPDIR/let_else_enum_soundness.rs" << 'ENDOFFILE'
enum Value {
    Some(i32),
    None,
}

fn extract_value(v: Value) -> i32 {
    let Value::Some(x) = v else {
        return 0;
    };
    x
}

fn let_else_enum_soundness_proof() {
    let v = Value::None;
    let result = extract_value(v);
    assert!(result == 1);  // MUST fail: result is 0
}
ENDOFFILE
run_test "let_else_enum_soundness" "$TMPDIR/let_else_enum_soundness.rs" "fail"


# Test 224b: Inclusive range soundness
# Purpose: Verifies wrong inclusive range match result assertions fail.
# Category: soundness
cat > "$TMPDIR/inclusive_range_soundness.rs" << 'ENDOFFILE'
fn categorize(n: i32) -> i32 {
    match n {
        0..=9 => 1,
        10..=99 => 2,
        _ => 3,
    }
}

fn inclusive_range_soundness_proof() {
    assert!(categorize(5) == 2);  // MUST fail: 5 gives 1
}
ENDOFFILE
run_test "inclusive_range_soundness" "$TMPDIR/inclusive_range_soundness.rs" "fail"


# Test 225b: Multi-pattern soundness
# Purpose: Verifies wrong multi-pattern match result assertions fail.
# Category: soundness
cat > "$TMPDIR/multi_pattern_soundness.rs" << 'ENDOFFILE'
enum Suit {
    Hearts,
    Diamonds,
    Clubs,
    Spades,
}

fn is_black(s: &Suit) -> bool {
    match s {
        Suit::Clubs | Suit::Spades => true,
        Suit::Hearts | Suit::Diamonds => false,
    }
}

fn multi_pattern_soundness_proof() {
    let s = Suit::Hearts;
    let result = is_black(&s);
    assert!(result);  // MUST fail: Hearts is not black
}
ENDOFFILE
run_test "multi_pattern_soundness" "$TMPDIR/multi_pattern_soundness.rs" "fail"


# Test 226b: Default fn soundness
# Purpose: Verifies wrong default constructor field assertions fail.
# Category: soundness
cat > "$TMPDIR/default_fn_soundness.rs" << 'ENDOFFILE'
struct Config {
    timeout: i32,
    retries: i32,
}

fn default_config() -> Config {
    Config {
        timeout: 30,
        retries: 3,
    }
}

fn default_fn_soundness_proof() {
    let c = default_config();
    assert!(c.timeout == 0);  // MUST fail: timeout is 30
}
ENDOFFILE
run_test "default_fn_soundness" "$TMPDIR/default_fn_soundness.rs" "fail"


# Test 227b: Visibility soundness
# Purpose: Verifies wrong pub struct field assertions fail.
# Category: soundness
cat > "$TMPDIR/visibility_soundness.rs" << 'ENDOFFILE'
mod inner {
    pub struct PublicStruct {
        pub public_field: i32,
    }

    impl PublicStruct {
        pub fn new(v: i32) -> Self {
            PublicStruct { public_field: v }
        }
    }
}

fn visibility_soundness_proof() {
    let s = inner::PublicStruct::new(10);
    assert!(s.public_field == 0);  // MUST fail: public_field is 10
}
ENDOFFILE
run_test "visibility_soundness" "$TMPDIR/visibility_soundness.rs" "fail"


# Test 228b: Nested if-let soundness
# Purpose: Verifies wrong nested if-let else branch assertions fail.
# Category: soundness
cat > "$TMPDIR/nested_if_let_soundness.rs" << 'ENDOFFILE'
enum Outer {
    A(i32),
    B,
}

fn nested_if_let_soundness_proof() {
    let outer = Outer::B;
    let result = if let Outer::A(n) = outer {
        n
    } else {
        -1
    };
    assert!(result == 42);  // MUST fail: result is -1 (B branch)
}
ENDOFFILE
run_test "nested_if_let_soundness" "$TMPDIR/nested_if_let_soundness.rs" "fail"


# Test 229b: While complex soundness
# Purpose: Verifies wrong while loop termination value assertions fail.
# Category: soundness
cat > "$TMPDIR/while_complex_soundness.rs" << 'ENDOFFILE'
fn while_complex_soundness_proof() {
    let mut x = 0i32;
    while x < 3 {
        x += 1;
    }
    assert!(x == 0);  // MUST fail: x is 3
}
ENDOFFILE
run_test "while_complex_soundness" "$TMPDIR/while_complex_soundness.rs" "fail"


# Test 230b: Struct array field soundness
# Purpose: Verifies wrong struct array element assertions fail.
# Category: soundness
cat > "$TMPDIR/struct_array_field_soundness.rs" << 'ENDOFFILE'
struct Buffer {
    data: [i32; 3],
    len: i32,
}

fn struct_array_field_soundness_proof() {
    let buf = Buffer {
        data: [1, 2, 3],
        len: 3,
    };
    assert!(buf.data[0] == 0);  // MUST fail: data[0] is 1
}
ENDOFFILE
run_test "struct_array_field_soundness" "$TMPDIR/struct_array_field_soundness.rs" "fail"


# Test 231b: Multi impl block soundness
# Purpose: Verifies wrong multi-impl method result assertions fail.
# Category: soundness
cat > "$TMPDIR/multi_impl_block_soundness.rs" << 'ENDOFFILE'
struct Calculator {
    value: i32,
}

impl Calculator {
    fn new(v: i32) -> Self {
        Calculator { value: v }
    }

    fn get(&self) -> i32 {
        self.value
    }
}

fn multi_impl_block_soundness_proof() {
    let calc = Calculator::new(42);
    assert!(calc.get() == 0);  // MUST fail: value is 42
}
ENDOFFILE
run_test "multi_impl_block_soundness" "$TMPDIR/multi_impl_block_soundness.rs" "fail"


# Test 232b: Early return nested soundness
# Purpose: Verifies wrong early return ordering assertions fail.
# Category: soundness
cat > "$TMPDIR/early_return_nested_soundness.rs" << 'ENDOFFILE'
fn find_positive(a: i32, b: i32) -> i32 {
    if a > 0 {
        return a;
    }
    if b > 0 {
        return b;
    }
    0
}

fn early_return_nested_soundness_proof() {
    let result = find_positive(5, 10);
    assert!(result == 10);  // MUST fail: result is 5 (first positive)
}
ENDOFFILE
run_test "early_return_nested_soundness" "$TMPDIR/early_return_nested_soundness.rs" "fail"


# Test 233b: Enum tuple variant soundness
# Purpose: Verifies wrong enum tuple variant extraction assertions fail.
# Category: soundness
cat > "$TMPDIR/enum_tuple_variant_soundness.rs" << 'ENDOFFILE'
#[derive(Clone, Copy)]
enum Value {
    Single(i32),
    Pair(i32, i32),
}

fn first_value(v: Value) -> i32 {
    match v {
        Value::Single(x) => x,
        Value::Pair(x, _) => x,
    }
}

fn enum_tuple_variant_soundness_proof() {
    let v = Value::Single(42);
    let result = first_value(v);
    assert!(result == 99);  // MUST fail: result is 42
}
ENDOFFILE
run_test "enum_tuple_variant_soundness" "$TMPDIR/enum_tuple_variant_soundness.rs" "fail"


# Test 234b: Struct from fn soundness
# Purpose: Verifies wrong function-initialized struct field assertions fail.
# Category: soundness
cat > "$TMPDIR/struct_from_fn_soundness.rs" << 'ENDOFFILE'
struct Point {
    x: i32,
    y: i32,
}

fn compute_x() -> i32 { 10 }

fn struct_from_fn_soundness_proof() {
    let p = Point {
        x: compute_x(),
        y: 20,
    };
    assert!(p.x == 0);  // MUST fail: x is 10
}
ENDOFFILE
run_test "struct_from_fn_soundness" "$TMPDIR/struct_from_fn_soundness.rs" "fail"


# Test 235b: Bool negation soundness
# Purpose: Verifies wrong boolean negation assertions fail.
# Category: soundness
cat > "$TMPDIR/bool_negation_soundness.rs" << 'ENDOFFILE'
fn bool_negation_soundness_proof() {
    let t = true;
    assert!(!t);  // MUST fail: !true is false
}
ENDOFFILE
run_test "bool_negation_soundness" "$TMPDIR/bool_negation_soundness.rs" "fail"


# Test 236b: Deep expr soundness
# Purpose: Verifies wrong nested expression result assertions fail.
# Category: soundness
cat > "$TMPDIR/deep_expr_soundness.rs" << 'ENDOFFILE'
fn deep_expr_soundness_proof() {
    let a = 1i32;
    let b = 2i32;
    let result = (a + b) + (a + b);  // 3 + 3 = 6
    assert!(result == 0);  // MUST fail: result is 6
}
ENDOFFILE
run_test "deep_expr_soundness" "$TMPDIR/deep_expr_soundness.rs" "fail"


# Test 237b: Multi param method soundness
# Purpose: Verifies wrong multi-param constructor field assertions fail.
# Category: soundness
cat > "$TMPDIR/multi_param_method_soundness.rs" << 'ENDOFFILE'
struct Rect {
    width: i32,
    height: i32,
}

impl Rect {
    fn new(w: i32, h: i32) -> Self {
        Rect { width: w, height: h }
    }
}

fn multi_param_method_soundness_proof() {
    let r = Rect::new(10, 20);
    assert!(r.width == 0);  // MUST fail: width is 10
}
ENDOFFILE
run_test "multi_param_method_soundness" "$TMPDIR/multi_param_method_soundness.rs" "fail"


# Test 238b: Const expr soundness
# Purpose: Verifies wrong const variable assertions fail.
# Category: soundness
cat > "$TMPDIR/const_expr_soundness.rs" << 'ENDOFFILE'
const VALUE: i32 = 42;

fn const_expr_soundness_proof() {
    let x = VALUE;
    assert!(x == 0);  // MUST fail: x is 42
}
ENDOFFILE
run_test "const_expr_soundness" "$TMPDIR/const_expr_soundness.rs" "fail"


# Test 239b: Enum discriminant soundness
# Purpose: Verifies wrong enum discriminant match assertions fail.
# Category: soundness
cat > "$TMPDIR/enum_discriminant_soundness.rs" << 'ENDOFFILE'
enum Status {
    Active,
    Inactive,
}

fn is_active(s: &Status) -> bool {
    match s {
        Status::Active => true,
        _ => false,
    }
}

fn enum_discriminant_soundness_proof() {
    let inactive = Status::Inactive;
    assert!(is_active(&inactive));  // MUST fail: inactive returns false
}
ENDOFFILE
run_test "enum_discriminant_soundness" "$TMPDIR/enum_discriminant_soundness.rs" "fail"


# Test 240b: Field mutation method soundness
# Purpose: Verifies wrong post-mutation field value assertions fail.
# Category: soundness
cat > "$TMPDIR/field_mutation_method_soundness.rs" << 'ENDOFFILE'
struct Counter {
    value: i32,
}

impl Counter {
    fn new() -> Self {
        Counter { value: 0 }
    }

    fn increment(&mut self) {
        self.value += 1;
    }

    fn get(&self) -> i32 {
        self.value
    }
}

fn field_mutation_method_soundness_proof() {
    let mut c = Counter::new();
    c.increment();
    assert!(c.get() == 0);  // MUST fail: value is 1
}
ENDOFFILE
run_test "field_mutation_method_soundness" "$TMPDIR/field_mutation_method_soundness.rs" "fail"


# Test 241b: Decrement soundness
# Purpose: Verifies wrong decrement result assertions fail.
# Category: soundness
cat > "$TMPDIR/decrement_soundness.rs" << 'ENDOFFILE'
fn decrement_soundness_proof() {
    let mut x = 10i32;
    x -= 1;
    assert!(x == 10);  // MUST fail: x is 9
}
ENDOFFILE
run_test "decrement_soundness" "$TMPDIR/decrement_soundness.rs" "fail"


# Test 242b: Boolean XOR soundness
# Purpose: Verifies wrong XOR result assertions fail.
# Category: soundness
cat > "$TMPDIR/bool_xor_soundness.rs" << 'ENDOFFILE'
fn xor(a: bool, b: bool) -> bool {
    (a || b) && !(a && b)
}

fn bool_xor_soundness_proof() {
    assert!(xor(true, true));  // MUST fail: XOR of same values is false
}
ENDOFFILE
run_test "bool_xor_soundness" "$TMPDIR/bool_xor_soundness.rs" "fail"


# Test 243b: Negative arithmetic soundness
# Purpose: Verifies wrong negative arithmetic result assertions fail.
# Category: soundness
cat > "$TMPDIR/neg_arith_soundness.rs" << 'ENDOFFILE'
fn neg_arith_soundness_proof() {
    let a = -10i32;
    let b = 5i32;
    let sum = a + b;
    assert!(sum == 0);  // MUST fail: sum is -5
}
ENDOFFILE
run_test "neg_arith_soundness" "$TMPDIR/neg_arith_soundness.rs" "fail"


# Test 244b: Loop countdown soundness
# Purpose: Verifies wrong countdown termination value assertions fail.
# Category: soundness
cat > "$TMPDIR/loop_countdown_soundness.rs" << 'ENDOFFILE'
fn loop_countdown_soundness_proof() {
    let mut count = 5i32;
    while count > 0 {
        count -= 1;
    }
    assert!(count == 1);  // MUST fail: count is 0
}
ENDOFFILE
run_test "loop_countdown_soundness" "$TMPDIR/loop_countdown_soundness.rs" "fail"


# Test 245b: Branch return soundness
# Purpose: Verifies wrong branch return value assertions fail.
# Category: soundness
cat > "$TMPDIR/branch_return_soundness.rs" << 'ENDOFFILE'
fn classify(x: i32) -> i32 {
    if x < 0 {
        return -1;
    } else if x > 0 {
        return 1;
    }
    0
}

fn branch_return_soundness_proof() {
    assert!(classify(-5) == 1);  // MUST fail: classify(-5) returns -1
}
ENDOFFILE
run_test "branch_return_soundness" "$TMPDIR/branch_return_soundness.rs" "fail"


# Test 246b: Power of two soundness
# Purpose: Verifies wrong power-of-two check assertions fail.
# Category: soundness
cat > "$TMPDIR/power_of_two_soundness.rs" << 'ENDOFFILE'
fn is_power_of_two(x: i32) -> bool {
    if x <= 0 {
        return false;
    }
    let mask = x - 1;
    (x & mask) == 0
}

fn power_of_two_soundness_proof() {
    assert!(is_power_of_two(3));  // MUST fail: 3 is not a power of 2
}
ENDOFFILE
run_test "power_of_two_soundness" "$TMPDIR/power_of_two_soundness.rs" "fail"


# Test 247b: Even/odd soundness
# Purpose: Verifies wrong even/odd check assertions fail.
# Category: soundness
cat > "$TMPDIR/even_odd_soundness.rs" << 'ENDOFFILE'
fn is_even(x: i32) -> bool {
    x % 2 == 0
}

fn even_odd_soundness_proof() {
    assert!(is_even(3));  // MUST fail: 3 is odd
}
ENDOFFILE
run_test "even_odd_soundness" "$TMPDIR/even_odd_soundness.rs" "fail"


# Test 248b: Sign check soundness
# Purpose: Verifies wrong sign check assertions fail.
# Category: soundness
cat > "$TMPDIR/sign_check_soundness.rs" << 'ENDOFFILE'
fn sign(x: i32) -> i32 {
    if x > 0 {
        1
    } else if x < 0 {
        -1
    } else {
        0
    }
}

fn sign_check_soundness_proof() {
    assert!(sign(-5) == 1);  // MUST fail: sign(-5) is -1
}
ENDOFFILE
run_test "sign_check_soundness" "$TMPDIR/sign_check_soundness.rs" "fail"


# Test 249b: GCD soundness
# Purpose: Verifies wrong GCD result assertions fail.
# Category: soundness
cat > "$TMPDIR/gcd_soundness.rs" << 'ENDOFFILE'
fn gcd(mut a: i32, mut b: i32) -> i32 {
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

fn gcd_soundness_proof() {
    assert!(gcd(12, 8) == 3);  // MUST fail: gcd(12,8) = 4
}
ENDOFFILE
run_test "gcd_soundness" "$TMPDIR/gcd_soundness.rs" "fail"


# Test 250b: Array sum soundness
# Purpose: Verifies wrong array sum assertions fail.
# Category: soundness
cat > "$TMPDIR/array_sum_soundness.rs" << 'ENDOFFILE'
fn array_sum_soundness_proof() {
    let arr = [1i32, 2, 3, 4, 5];
    let sum = arr[0] + arr[1] + arr[2] + arr[3] + arr[4];
    assert!(sum == 10);  // MUST fail: sum is 15
}
ENDOFFILE
run_test "array_sum_soundness" "$TMPDIR/array_sum_soundness.rs" "fail"


# Test 251b: Struct equality soundness
# Purpose: Verifies wrong struct equality assertions fail.
# Category: soundness
cat > "$TMPDIR/struct_eq_soundness.rs" << 'ENDOFFILE'
struct Point {
    x: i32,
    y: i32,
}

fn points_equal(a: &Point, b: &Point) -> bool {
    a.x == b.x && a.y == b.y
}

fn struct_eq_soundness_proof() {
    let p1 = Point { x: 5, y: 10 };
    let p2 = Point { x: 5, y: 20 };
    assert!(points_equal(&p1, &p2));  // MUST fail: different y values
}
ENDOFFILE
run_test "struct_eq_soundness" "$TMPDIR/struct_eq_soundness.rs" "fail"


# Test 252b: Fibonacci soundness
# Purpose: Verifies wrong Fibonacci result assertions fail.
# Category: soundness
cat > "$TMPDIR/fibonacci_soundness.rs" << 'ENDOFFILE'
fn fib(n: i32) -> i32 {
    if n <= 1 {
        return n;
    }
    let mut a = 0i32;
    let mut b = 1i32;
    let mut i = 2i32;
    while i <= n {
        let tmp = a + b;
        a = b;
        b = tmp;
        i += 1;
    }
    b
}

fn fibonacci_soundness_proof() {
    assert!(fib(5) == 6);  // MUST fail: fib(5) = 5
}
ENDOFFILE
run_test "fibonacci_soundness" "$TMPDIR/fibonacci_soundness.rs" "fail"


# Test 253b: Temp variable soundness
# Purpose: Verifies wrong temp variable assertions fail.
# Category: soundness
cat > "$TMPDIR/temp_var_soundness.rs" << 'ENDOFFILE'
fn temp_var_soundness_proof() {
    let a = 10i32;
    let tmp = a;
    assert!(tmp == 0);  // MUST fail: tmp is 10
}
ENDOFFILE
run_test "temp_var_soundness" "$TMPDIR/temp_var_soundness.rs" "fail"


# Test 254b: Conditional assignment soundness
# Purpose: Verifies wrong conditional assignment assertions fail.
# Category: soundness
cat > "$TMPDIR/cond_assign_soundness.rs" << 'ENDOFFILE'
fn cond_assign_soundness_proof() {
    let condition = false;
    let x = if condition { 100 } else { 200 };
    assert!(x == 100);  // MUST fail: x is 200
}
ENDOFFILE
run_test "cond_assign_soundness" "$TMPDIR/cond_assign_soundness.rs" "fail"


# Test 255b: XOR assign soundness
# Purpose: Verifies wrong XOR assign result assertions fail.
# Category: soundness
cat > "$TMPDIR/xor_assign_soundness.rs" << 'ENDOFFILE'
fn xor_assign_soundness_proof() {
    let mut x = 0b1010i32;
    x ^= 0b1100;
    assert!(x == 0b1010);  // MUST fail: x is 0b0110
}
ENDOFFILE
run_test "xor_assign_soundness" "$TMPDIR/xor_assign_soundness.rs" "fail"


# Test 256b: AND op soundness
# Purpose: Verifies wrong bitwise AND result assertions fail.
# Category: soundness
cat > "$TMPDIR/and_op_soundness.rs" << 'ENDOFFILE'
fn and_op_soundness_proof() {
    let a = 15i32;
    let b = 5i32;
    let result = a & b;
    assert!(result == 15);  // MUST fail: result is 5
}
ENDOFFILE
run_test "and_op_soundness" "$TMPDIR/and_op_soundness.rs" "fail"


# Test 257b: OR op soundness
# Purpose: Verifies wrong bitwise OR result assertions fail.
# Category: soundness
cat > "$TMPDIR/or_op_soundness.rs" << 'ENDOFFILE'
fn or_op_soundness_proof() {
    let a = 3i32;
    let b = 12i32;
    let result = a | b;
    assert!(result == 3);  // MUST fail: result is 15
}
ENDOFFILE
run_test "or_op_soundness" "$TMPDIR/or_op_soundness.rs" "fail"


# Test 258b: Shift left assign soundness
# Purpose: Verifies wrong shift left result assertions fail.
# Category: soundness
cat > "$TMPDIR/shl_assign_soundness.rs" << 'ENDOFFILE'
fn shl_assign_soundness_proof() {
    let mut x = 1i32;
    x <<= 3;
    assert!(x == 1);  // MUST fail: x is 8
}
ENDOFFILE
run_test "shl_assign_soundness" "$TMPDIR/shl_assign_soundness.rs" "fail"


# Test 259b: Shift right assign soundness
# Purpose: Verifies wrong shift right result assertions fail.
# Category: soundness
cat > "$TMPDIR/shr_assign_soundness.rs" << 'ENDOFFILE'
fn shr_assign_soundness_proof() {
    let mut x = 16i32;
    x >>= 2;
    assert!(x == 16);  // MUST fail: x is 4
}
ENDOFFILE
run_test "shr_assign_soundness" "$TMPDIR/shr_assign_soundness.rs" "fail"


# Test 261b: Min three soundness
# Purpose: Verifies wrong min-three result assertions fail.
# Category: soundness
cat > "$TMPDIR/min_three_soundness.rs" << 'ENDOFFILE'
fn min_three(a: i32, b: i32, c: i32) -> i32 {
    if a <= b && a <= c {
        a
    } else if b <= c {
        b
    } else {
        c
    }
}

fn min_three_soundness_proof() {
    assert!(min_three(1, 2, 3) == 2);  // MUST fail: min is 1
}
ENDOFFILE
run_test "min_three_soundness" "$TMPDIR/min_three_soundness.rs" "fail"


# Test 262b: Max three soundness
# Purpose: Verifies wrong max-three result assertions fail.
# Category: soundness
cat > "$TMPDIR/max_three_soundness.rs" << 'ENDOFFILE'
fn max_three(a: i32, b: i32, c: i32) -> i32 {
    if a >= b && a >= c {
        a
    } else if b >= c {
        b
    } else {
        c
    }
}

fn max_three_soundness_proof() {
    assert!(max_three(1, 2, 3) == 1);  // MUST fail: max is 3
}
ENDOFFILE
run_test "max_three_soundness" "$TMPDIR/max_three_soundness.rs" "fail"


# Test 263b: Popcount soundness
# Purpose: Verifies wrong popcount result assertions fail.
# Category: soundness
cat > "$TMPDIR/popcount_soundness.rs" << 'ENDOFFILE'
fn popcount(mut n: u32) -> u32 {
    let mut count = 0u32;
    while n > 0 {
        count += n & 1;
        n >>= 1;
    }
    count
}

fn popcount_soundness_proof() {
    assert!(popcount(7) == 2);  // MUST fail: popcount(7) = 3
}
ENDOFFILE
run_test "popcount_soundness" "$TMPDIR/popcount_soundness.rs" "fail"


# Test 264b: Sum squares soundness
# Purpose: Verifies wrong sum-squares result assertions fail.
# Category: soundness
cat > "$TMPDIR/sum_squares_soundness.rs" << 'ENDOFFILE'
fn sum_squares_soundness_proof() {
    let squares = [0, 1, 4, 9, 16, 25];
    let mut sum = 0i32;
    let mut i = 0usize;
    while i < 6 {
        sum += squares[i] as i32;
        i += 1;
    }
    assert!(sum == 100);  // MUST fail: sum is 55
}
ENDOFFILE
run_test "sum_squares_soundness" "$TMPDIR/sum_squares_soundness.rs" "fail"


# Test 268b: Count down exit soundness
# Purpose: Verifies wrong count-down result assertions fail.
# Category: soundness
cat > "$TMPDIR/count_down_exit_soundness.rs" << 'ENDOFFILE'
fn count_until(start: i32, target: i32) -> i32 {
    let mut current = start;
    let mut steps = 0i32;
    while current > 0 {
        if current == target {
            return steps;
        }
        current -= 1;
        steps += 1;
    }
    steps
}

fn count_down_exit_soundness_proof() {
    assert!(count_until(10, 5) == 10);  // MUST fail: steps is 5
}
ENDOFFILE
run_test "count_down_exit_soundness" "$TMPDIR/count_down_exit_soundness.rs" "fail"


# Test 269b: Clip range soundness
# Purpose: Verifies wrong clip-range assertions fail.
# Category: soundness
cat > "$TMPDIR/clip_range_soundness.rs" << 'ENDOFFILE'
fn clip(value: i32, min_val: i32, max_val: i32) -> i32 {
    if value < min_val {
        min_val
    } else if value > max_val {
        max_val
    } else {
        value
    }
}

fn clip_range_soundness_proof() {
    assert!(clip(-5, 0, 10) == -5);  // MUST fail: result is 0
}
ENDOFFILE
run_test "clip_range_soundness" "$TMPDIR/clip_range_soundness.rs" "fail"


# Test 271b: Two's complement soundness
# Purpose: Verifies wrong two's complement assertions fail.
# Category: soundness
cat > "$TMPDIR/twos_complement_soundness.rs" << 'ENDOFFILE'
fn negate_twos(n: i32) -> i32 {
    !n + 1
}

fn twos_complement_soundness_proof() {
    assert!(negate_twos(5) == 5);  // MUST fail: result is -5
}
ENDOFFILE
run_test "twos_complement_soundness" "$TMPDIR/twos_complement_soundness.rs" "fail"


# Test 272b: Struct defaults soundness
# Purpose: Verifies wrong struct-defaults assertions fail.
# Category: soundness
cat > "$TMPDIR/struct_defaults_soundness.rs" << 'ENDOFFILE'
struct Config {
    width: i32,
    height: i32,
    depth: i32,
}

impl Config {
    fn new() -> Self {
        Config { width: 800, height: 600, depth: 24 }
    }
}

fn struct_defaults_soundness_proof() {
    let default = Config::new();
    assert!(default.width == 0);  // MUST fail: width is 800
}
ENDOFFILE
run_test "struct_defaults_soundness" "$TMPDIR/struct_defaults_soundness.rs" "fail"


# Test 273b: Fluent chain soundness
# Purpose: Verifies wrong fluent-chain assertions fail.
# Category: soundness
cat > "$TMPDIR/fluent_chain_soundness.rs" << 'ENDOFFILE'
struct Counter {
    value: i32,
}

impl Counter {
    fn new() -> Counter {
        Counter { value: 0 }
    }

    fn increment(&self) -> Counter {
        Counter { value: self.value + 1 }
    }
}

fn fluent_chain_soundness_proof() {
    let c = Counter::new().increment().increment();
    assert!(c.value == 0);  // MUST fail: value is 2
}
ENDOFFILE
run_test "fluent_chain_soundness" "$TMPDIR/fluent_chain_soundness.rs" "fail"


# Test 274b: Nested cond soundness
# Purpose: Verifies wrong nested-conditional assertions fail.
# Category: soundness
cat > "$TMPDIR/nested_cond_soundness.rs" << 'ENDOFFILE'
fn classify(x: i32, y: i32) -> i32 {
    if x > 0 {
        if y > 0 { 1 } else if y < 0 { 2 } else { 3 }
    } else if x < 0 {
        if y > 0 { 4 } else if y < 0 { 5 } else { 6 }
    } else {
        if y > 0 { 7 } else if y < 0 { 8 } else { 9 }
    }
}

fn nested_cond_soundness_proof() {
    assert!(classify(1, 1) == 5);  // MUST fail: result is 1
}
ENDOFFILE
run_test "nested_cond_soundness" "$TMPDIR/nested_cond_soundness.rs" "fail"


# Test 275b: Sum range soundness
# Purpose: Verifies wrong sum-range assertions fail.
# Category: soundness
cat > "$TMPDIR/sum_range_soundness.rs" << 'ENDOFFILE'
fn sum_range(a: i32, b: i32) -> i32 {
    let mut sum = 0i32;
    let mut i = a;
    while i <= b {
        sum += i;
        i += 1;
    }
    sum
}

fn sum_range_soundness_proof() {
    assert!(sum_range(1, 5) == 10);  // MUST fail: sum is 15
}
ENDOFFILE
run_test "sum_range_soundness" "$TMPDIR/sum_range_soundness.rs" "fail"


# Test 276b: Div round zero soundness
# Purpose: Verifies wrong division assertions fail.
# Category: soundness
cat > "$TMPDIR/div_round_zero_soundness.rs" << 'ENDOFFILE'
fn div_round_zero_soundness_proof() {
    assert!(7i32 / 3 == 3);  // MUST fail: result is 2
}
ENDOFFILE
run_test "div_round_zero_soundness" "$TMPDIR/div_round_zero_soundness.rs" "fail"


# Test 277b: Mod negative soundness
# Purpose: Verifies wrong modulo assertions fail.
# Category: soundness
cat > "$TMPDIR/mod_negative_soundness.rs" << 'ENDOFFILE'
fn mod_negative_soundness_proof() {
    assert!((-7i32) % 3 == 1);  // MUST fail: result is -1
}
ENDOFFILE
run_test "mod_negative_soundness" "$TMPDIR/mod_negative_soundness.rs" "fail"


# Test 280b: Point3D soundness
# Purpose: Verifies wrong Point3D assertions fail.
# Category: soundness
cat > "$TMPDIR/point3d_soundness.rs" << 'ENDOFFILE'
struct Point3D {
    x: i32,
    y: i32,
    z: i32,
}

impl Point3D {
    fn add(&self, other: &Point3D) -> Point3D {
        Point3D {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

fn point3d_soundness_proof() {
    let a = Point3D { x: 1, y: 2, z: 3 };
    let b = Point3D { x: 4, y: 5, z: 6 };
    let sum = a.add(&b);
    assert!(sum.x == 0);  // MUST fail: sum.x is 5
}
ENDOFFILE
run_test "point3d_soundness" "$TMPDIR/point3d_soundness.rs" "fail"


# Test 281b: Swap temp soundness
# Purpose: Verifies wrong swap-temp assertions fail.
# Category: soundness
cat > "$TMPDIR/swap_temp_soundness.rs" << 'ENDOFFILE'
fn swap_temp_soundness_proof() {
    let mut a = 10i32;
    let mut b = 20i32;
    let temp = a;
    assert!(temp == 10);
    a = b;
    assert!(a == 20);
    b = temp;
    assert!(a == 10);  // MUST fail: a is still 20
}
ENDOFFILE
run_test "swap_temp_soundness" "$TMPDIR/swap_temp_soundness.rs" "fail"


# Test 282b: XOR swap soundness
# Purpose: Verifies wrong XOR-swap assertions fail.
# Category: soundness
cat > "$TMPDIR/xor_swap_soundness.rs" << 'ENDOFFILE'
fn xor_swap_soundness_proof() {
    let mut a = 5i32;
    let mut b = 3i32;
    a = a ^ b;
    assert!(a == 6);
    b = a ^ b;
    assert!(b == 5);
    a = a ^ b;
    assert!(a == 5);  // MUST fail: a is 3 after swap
}
ENDOFFILE
run_test "xor_swap_soundness" "$TMPDIR/xor_swap_soundness.rs" "fail"


# Test 283b: CLZ simple soundness
# Purpose: Verifies wrong CLZ assertions fail.
# Category: soundness
cat > "$TMPDIR/clz_simple_soundness.rs" << 'ENDOFFILE'
fn clz_simple_soundness_proof() {
    let x = 4i32;
    let mut count = 0i32;
    if x < 128 { count += 25; }
    if x < 64 { count += 1; }
    if x < 32 { count += 1; }
    if x < 16 { count += 1; }
    if x < 8 { count += 1; }
    assert!(count == 0);  // MUST fail: count is 29
}
ENDOFFILE
run_test "clz_simple_soundness" "$TMPDIR/clz_simple_soundness.rs" "fail"


# Test 284b: Sign extend soundness
# Purpose: Verifies wrong sign-extend assertions fail.
# Category: soundness
cat > "$TMPDIR/sign_extend_soundness.rs" << 'ENDOFFILE'
fn sign_extend_soundness_proof() {
    let byte: i32 = 0xFF;
    let extended = if (byte & 0x80) != 0 {
        byte | !0xFF
    } else {
        byte & 0xFF
    };
    assert!(extended == 255);  // MUST fail: extended is -1
}
ENDOFFILE
run_test "sign_extend_soundness" "$TMPDIR/sign_extend_soundness.rs" "fail"


# Test 285b: Bit field soundness
# Purpose: Verifies wrong bit-field assertions fail.
# Category: soundness
cat > "$TMPDIR/bit_field_soundness.rs" << 'ENDOFFILE'
fn bit_field_soundness_proof() {
    let packed = 0b_0011_1010_i32;
    let field = (packed >> 2) & 0b111;
    assert!(field == 0);  // MUST fail: field is 6
}
ENDOFFILE
run_test "bit_field_soundness" "$TMPDIR/bit_field_soundness.rs" "fail"


# Test 286b: Set bit soundness
# Purpose: Verifies wrong set-bit assertions fail.
# Category: soundness
cat > "$TMPDIR/set_bit_soundness.rs" << 'ENDOFFILE'
fn set_bit_soundness_proof() {
    let mut x = 0i32;
    x = x | (1 << 3);
    x = x | (1 << 1);
    assert!(x == 0);  // MUST fail: x is 10
}
ENDOFFILE
run_test "set_bit_soundness" "$TMPDIR/set_bit_soundness.rs" "fail"


# Test 287b: Clear bit soundness
# Purpose: Verifies wrong clear-bit assertions fail.
# Category: soundness
cat > "$TMPDIR/clear_bit_soundness.rs" << 'ENDOFFILE'
fn clear_bit_soundness_proof() {
    let mut x = 15i32;
    x = x & !(1 << 2);
    x = x & !(1 << 0);
    assert!(x == 15);  // MUST fail: x is 10
}
ENDOFFILE
run_test "clear_bit_soundness" "$TMPDIR/clear_bit_soundness.rs" "fail"


# Test 288b: Toggle bit soundness
# Purpose: Verifies wrong toggle-bit assertions fail.
# Category: soundness
cat > "$TMPDIR/toggle_bit_soundness.rs" << 'ENDOFFILE'
fn toggle_bit_soundness_proof() {
    let mut x = 5i32;
    x = x ^ (1 << 1);
    assert!(x == 5);  // MUST fail: x is 7 after toggle
}
ENDOFFILE
run_test "toggle_bit_soundness" "$TMPDIR/toggle_bit_soundness.rs" "fail"


# Test 289b: Check bit soundness
# Purpose: Verifies wrong check-bit assertions fail.
# Category: soundness
cat > "$TMPDIR/check_bit_soundness.rs" << 'ENDOFFILE'
fn check_bit_soundness_proof() {
    let x = 10i32;
    let bit1 = (x >> 1) & 1;
    assert!(bit1 == 0);  // MUST fail: bit1 is 1
}
ENDOFFILE
run_test "check_bit_soundness" "$TMPDIR/check_bit_soundness.rs" "fail"


# Test 290b: Rotate left soundness
# Purpose: Verifies wrong rotate-left assertions fail.
# Category: soundness
cat > "$TMPDIR/rotate_left_soundness.rs" << 'ENDOFFILE'
fn rotate_left_soundness_proof() {
    let x: i32 = 1;
    let rot = (x << 3) | ((x as u32 >> 29) as i32);
    assert!(rot == 1);  // MUST fail: rot is 8
}
ENDOFFILE
run_test "rotate_left_soundness" "$TMPDIR/rotate_left_soundness.rs" "fail"


# Test 291b: Integer average soundness
# Purpose: Verifies wrong int-average assertions fail.
# Category: soundness
cat > "$TMPDIR/int_avg_soundness.rs" << 'ENDOFFILE'
fn int_avg_soundness_proof() {
    let a = 10i32;
    let b = 20i32;
    let avg = (a & b) + ((a ^ b) >> 1);
    assert!(avg == 10);  // MUST fail: avg is 15
}
ENDOFFILE
run_test "int_avg_soundness" "$TMPDIR/int_avg_soundness.rs" "fail"


# Test 292b: Round pow2 soundness
# Purpose: Verifies wrong round-pow2 assertions fail.
# Category: soundness
cat > "$TMPDIR/round_pow2_soundness.rs" << 'ENDOFFILE'
fn round_pow2_soundness_proof() {
    let x = 13i32;
    let result = if x >= 8 { 8 } else if x >= 4 { 4 } else if x >= 2 { 2 } else { 1 };
    assert!(result == 16);  // MUST fail: result is 8
}
ENDOFFILE
run_test "round_pow2_soundness" "$TMPDIR/round_pow2_soundness.rs" "fail"


# Test 293b: Byte pack soundness
# Purpose: Verifies wrong byte-pack assertions fail.
# Category: soundness
cat > "$TMPDIR/byte_pack_soundness.rs" << 'ENDOFFILE'
fn byte_pack_soundness_proof() {
    let hi: i32 = 0xAB;
    let lo: i32 = 0xCD;
    let packed = (hi << 8) | lo;
    assert!(packed == 0xCDAB);  // MUST fail: packed is 0xABCD
}
ENDOFFILE
run_test "byte_pack_soundness" "$TMPDIR/byte_pack_soundness.rs" "fail"


# Test 294b: Byte unpack soundness
# Purpose: Verifies wrong byte-unpack assertions fail.
# Category: soundness
cat > "$TMPDIR/byte_unpack_soundness.rs" << 'ENDOFFILE'
fn byte_unpack_soundness_proof() {
    let packed: i32 = 0xABCD;
    let hi = (packed >> 8) & 0xFF;
    assert!(hi == 0xCD);  // MUST fail: hi is 0xAB
}
ENDOFFILE
run_test "byte_unpack_soundness" "$TMPDIR/byte_unpack_soundness.rs" "fail"


# Test 295b: Aligned check soundness
# Purpose: Verifies wrong alignment-check assertions fail.
# Category: soundness
cat > "$TMPDIR/aligned_check_soundness.rs" << 'ENDOFFILE'
fn aligned_check_soundness_proof() {
    let addr = 16i32;
    let aligned = (addr & 7) == 0;
    assert!(!aligned);  // MUST fail: addr 16 is aligned to 8
}
ENDOFFILE
run_test "aligned_check_soundness" "$TMPDIR/aligned_check_soundness.rs" "fail"


# Test 296b: Align up soundness
# Purpose: Verifies wrong align-up assertions fail.
# Category: soundness
cat > "$TMPDIR/align_up_soundness.rs" << 'ENDOFFILE'
fn align_up_soundness_proof() {
    let addr = 13i32;
    let align = 8i32;
    let mask = align - 1;
    let aligned = (addr + mask) & !mask;
    assert!(aligned == 8);  // MUST fail: aligned is 16
}
ENDOFFILE
run_test "align_up_soundness" "$TMPDIR/align_up_soundness.rs" "fail"


# Test 297b: Mod exp step soundness
# Purpose: Verifies wrong mod-exp assertions fail.
# Category: soundness
cat > "$TMPDIR/mod_exp_step_soundness.rs" << 'ENDOFFILE'
fn mod_exp_step_soundness_proof() {
    let base = 3i32;
    let exp = 2i32;
    let modulus = 7i32;
    let result = if exp == 2 && base == 3 && modulus == 7 { 2 } else { 0 };
    assert!(result == 9);  // MUST fail: result is 2
}
ENDOFFILE
run_test "mod_exp_step_soundness" "$TMPDIR/mod_exp_step_soundness.rs" "fail"


# Test 298b: Ternary bool soundness
# Purpose: Verifies wrong ternary-bool assertions fail.
# Category: soundness
cat > "$TMPDIR/ternary_bool_soundness.rs" << 'ENDOFFILE'
fn ternary_bool_soundness_proof() {
    let ternary = -1i32;
    let is_negative = ternary < 0;
    assert!(!is_negative);  // MUST fail: -1 is negative
}
ENDOFFILE
run_test "ternary_bool_soundness" "$TMPDIR/ternary_bool_soundness.rs" "fail"


# Test 299b: Lerp int soundness
# Purpose: Verifies wrong lerp assertions fail.
# Category: soundness
cat > "$TMPDIR/lerp_int_soundness.rs" << 'ENDOFFILE'
fn lerp_int_soundness_proof() {
    let a = 0i32;
    let b = 100i32;
    let t = 50i32;
    let result = if t == 0 { a } else if t == 50 { (a + b) / 2 } else if t == 100 { b } else { a };
    assert!(result == 100);  // MUST fail: result is 50
}
ENDOFFILE
run_test "lerp_int_soundness" "$TMPDIR/lerp_int_soundness.rs" "fail"


# Test 300b: State machine soundness
# Purpose: Verifies wrong state-machine assertions fail.
# Category: soundness
cat > "$TMPDIR/state_machine_soundness.rs" << 'ENDOFFILE'
fn state_machine_soundness_proof() {
    let mut state = 0i32;
    if state == 0 { state = 1; }
    if state == 1 { state = 2; }
    assert!(state == 0);  // MUST fail: state is 2
}
ENDOFFILE
run_test "state_machine_soundness" "$TMPDIR/state_machine_soundness.rs" "fail"


# Test 301b: Priority encode soundness
# Purpose: Verifies wrong priority-encode assertions fail.
# Category: soundness
cat > "$TMPDIR/priority_encode_soundness.rs" << 'ENDOFFILE'
fn priority_encode_soundness_proof() {
    let x = 12i32;
    let pos = if (x & 1) != 0 { 0 } else if (x & 2) != 0 { 1 } else if (x & 4) != 0 { 2 } else if (x & 8) != 0 { 3 } else { -1 };
    assert!(pos == 3);  // MUST fail: pos is 2
}
ENDOFFILE
run_test "priority_encode_soundness" "$TMPDIR/priority_encode_soundness.rs" "fail"


# Test 302b: Gray code soundness
# Purpose: Verifies wrong gray-code assertions fail.
# Category: soundness
cat > "$TMPDIR/gray_code_soundness.rs" << 'ENDOFFILE'
fn gray_code_soundness_proof() {
    let binary = 5i32;
    let gray = binary ^ (binary >> 1);
    assert!(gray == 5);  // MUST fail: gray is 7
}
ENDOFFILE
run_test "gray_code_soundness" "$TMPDIR/gray_code_soundness.rs" "fail"


# Test 303b: Saturate add soundness
# Purpose: Verifies wrong saturate-add assertions fail.
# Category: soundness
cat > "$TMPDIR/saturate_add_soundness.rs" << 'ENDOFFILE'
fn saturate_add_soundness_proof() {
    let a = 100i32;
    let b = 50i32;
    let max_val = 120i32;
    let sum = a + b;
    let result = if sum > max_val { max_val } else { sum };
    assert!(result == 150);  // MUST fail: result is 120
}
ENDOFFILE
run_test "saturate_add_soundness" "$TMPDIR/saturate_add_soundness.rs" "fail"


# Test 304b: Saturate sub soundness
# Purpose: Verifies wrong saturate-sub assertions fail.
# Category: soundness
cat > "$TMPDIR/saturate_sub_soundness.rs" << 'ENDOFFILE'
fn saturate_sub_soundness_proof() {
    let a = 10i32;
    let b = 25i32;
    let min_val = 0i32;
    let diff = a - b;
    let result = if diff < min_val { min_val } else { diff };
    assert!(result == -15);  // MUST fail: result is 0
}
ENDOFFILE
run_test "saturate_sub_soundness" "$TMPDIR/saturate_sub_soundness.rs" "fail"


# Test 305b: Ring index soundness
# Purpose: Verifies wrong ring-index assertions fail.
# Category: soundness
cat > "$TMPDIR/ring_index_soundness.rs" << 'ENDOFFILE'
fn ring_index_soundness_proof() {
    let size = 8i32;
    let mut idx = 7i32;
    idx = (idx + 1) & (size - 1);
    assert!(idx == 8);  // MUST fail: idx is 0 (wrapped)
}
ENDOFFILE
run_test "ring_index_soundness" "$TMPDIR/ring_index_soundness.rs" "fail"


# Test 306b: Bounds inclusive soundness
# Purpose: Verifies wrong bounds-inclusive assertions fail.
# Category: soundness
cat > "$TMPDIR/bounds_inclusive_soundness.rs" << 'ENDOFFILE'
fn bounds_inclusive_soundness_proof() {
    let x = 15i32;
    let lo = 1i32;
    let hi = 10i32;
    let in_range = x >= lo && x <= hi;
    assert!(in_range);  // MUST fail: 15 is not in [1, 10]
}
ENDOFFILE
run_test "bounds_inclusive_soundness" "$TMPDIR/bounds_inclusive_soundness.rs" "fail"


# Test 307b: Bounds exclusive soundness
# Purpose: Verifies wrong bounds-exclusive assertions fail.
# Category: soundness
cat > "$TMPDIR/bounds_exclusive_soundness.rs" << 'ENDOFFILE'
fn bounds_exclusive_soundness_proof() {
    let x = 10i32;
    let lo = 1i32;
    let hi = 10i32;
    let in_range = x > lo && x < hi;
    assert!(in_range);  // MUST fail: 10 is not in (1, 10)
}
ENDOFFILE
run_test "bounds_exclusive_soundness" "$TMPDIR/bounds_exclusive_soundness.rs" "fail"


# Test 308b: Minmax update soundness
# Purpose: Verifies wrong minmax-update assertions fail.
# Category: soundness
cat > "$TMPDIR/minmax_update_soundness.rs" << 'ENDOFFILE'
fn minmax_update_soundness_proof() {
    let mut min = 100i32;
    let mut max = 0i32;
    let values = [5i32, 10, 3, 8];
    if values[0] < min { min = values[0]; }
    if values[0] > max { max = values[0]; }
    if values[1] < min { min = values[1]; }
    if values[1] > max { max = values[1]; }
    if values[2] < min { min = values[2]; }
    if values[2] > max { max = values[2]; }
    if values[3] < min { min = values[3]; }
    if values[3] > max { max = values[3]; }
    assert!(min == 5);  // MUST fail: min is 3
}
ENDOFFILE
run_test "minmax_update_soundness" "$TMPDIR/minmax_update_soundness.rs" "fail"


# Test 309b: Flag register soundness
# Purpose: Verifies wrong flag-register assertions fail.
# Category: soundness
cat > "$TMPDIR/flag_register_soundness.rs" << 'ENDOFFILE'
fn flag_register_soundness_proof() {
    const FLAG_A: i32 = 1;
    const FLAG_B: i32 = 2;
    let flags = FLAG_A;
    let has_b = (flags & FLAG_B) != 0;
    assert!(has_b);  // MUST fail: FLAG_B not set
}
ENDOFFILE
run_test "flag_register_soundness" "$TMPDIR/flag_register_soundness.rs" "fail"


# Test 310b: ASCII digit soundness
# Purpose: Verifies wrong ASCII-digit assertions fail.
# Category: soundness
cat > "$TMPDIR/ascii_digit_soundness.rs" << 'ENDOFFILE'
fn ascii_digit_soundness_proof() {
    let c = 65i32;  // ASCII 'A'
    let is_digit = c >= 48 && c <= 57;
    assert!(is_digit);  // MUST fail: 'A' is not a digit
}
ENDOFFILE
run_test "ascii_digit_soundness" "$TMPDIR/ascii_digit_soundness.rs" "fail"


# Test 311b: ASCII letter soundness
# Purpose: Verifies wrong ASCII-letter assertions fail.
# Category: soundness
cat > "$TMPDIR/ascii_letter_soundness.rs" << 'ENDOFFILE'
fn ascii_letter_soundness_proof() {
    let c = 48i32;  // ASCII '0'
    let is_upper = c >= 65 && c <= 90;
    let is_lower = c >= 97 && c <= 122;
    let is_letter = is_upper || is_lower;
    assert!(is_letter);  // MUST fail: '0' is not a letter
}
ENDOFFILE
run_test "ascii_letter_soundness" "$TMPDIR/ascii_letter_soundness.rs" "fail"


# Test 312b: ASCII case soundness
# Purpose: Verifies wrong ASCII-case assertions fail.
# Category: soundness
cat > "$TMPDIR/ascii_case_soundness.rs" << 'ENDOFFILE'
fn ascii_case_soundness_proof() {
    let upper = 65i32;
    let lower = if upper >= 65 && upper <= 90 { upper + 32 } else { upper };
    assert!(lower == 65);  // MUST fail: lower is 97
}
ENDOFFILE
run_test "ascii_case_soundness" "$TMPDIR/ascii_case_soundness.rs" "fail"


# Test 313b: Digit to int soundness
# Purpose: Verifies wrong digit-to-int assertions fail.
# Category: soundness
cat > "$TMPDIR/digit_to_int_soundness.rs" << 'ENDOFFILE'
fn digit_to_int_soundness_proof() {
    let c = 55i32;
    let value = c - 48;
    assert!(value == 55);  // MUST fail: value is 7
}
ENDOFFILE
run_test "digit_to_int_soundness" "$TMPDIR/digit_to_int_soundness.rs" "fail"


# Test 314b: Int to digit soundness
# Purpose: Verifies wrong int-to-digit assertions fail.
# Category: soundness
cat > "$TMPDIR/int_to_digit_soundness.rs" << 'ENDOFFILE'
fn int_to_digit_soundness_proof() {
    let value = 3i32;
    let c = value + 48;
    assert!(c == 3);  // MUST fail: c is 51
}
ENDOFFILE
run_test "int_to_digit_soundness" "$TMPDIR/int_to_digit_soundness.rs" "fail"


# Test 315b: Hex to int soundness
# Purpose: Verifies wrong hex-to-int assertions fail.
# Category: soundness
cat > "$TMPDIR/hex_to_int_soundness.rs" << 'ENDOFFILE'
fn hex_to_int_soundness_proof() {
    let c = 66i32;
    let value = if c >= 48 && c <= 57 { c - 48 } else if c >= 65 && c <= 70 { c - 65 + 10 } else if c >= 97 && c <= 102 { c - 97 + 10 } else { -1 };
    assert!(value == 16);  // MUST fail: value is 11
}
ENDOFFILE
run_test "hex_to_int_soundness" "$TMPDIR/hex_to_int_soundness.rs" "fail"


# Test 316b: Parity soundness
# Purpose: Verifies wrong parity assertions fail.
# Category: soundness
cat > "$TMPDIR/parity_soundness.rs" << 'ENDOFFILE'
fn parity_soundness_proof() {
    let x = 7i32;
    let b0 = (x >> 0) & 1;
    let b1 = (x >> 1) & 1;
    let b2 = (x >> 2) & 1;
    let b3 = (x >> 3) & 1;
    let parity = b0 ^ b1 ^ b2 ^ b3;
    assert!(parity == 0);  // MUST fail: parity is 1
}
ENDOFFILE
run_test "parity_soundness" "$TMPDIR/parity_soundness.rs" "fail"


# Test 317b: CRC step soundness
# Purpose: Verifies wrong CRC-step assertions fail.
# Category: soundness
cat > "$TMPDIR/crc_step_soundness.rs" << 'ENDOFFILE'
fn crc_step_soundness_proof() {
    let mut crc = 0xFFi32;
    let data = 0x5Ai32;
    let poly = 0x31i32;
    crc = crc ^ data;
    if (crc & 0x80) != 0 {
        crc = (crc << 1) ^ poly;
    } else {
        crc = crc << 1;
    }
    let result = crc & 0xFF;
    assert!(result == 0xFF);  // MUST fail: result is 0x7B
}
ENDOFFILE
run_test "crc_step_soundness" "$TMPDIR/crc_step_soundness.rs" "fail"


# Test 318b: Bit reverse4 soundness
# Purpose: Verifies wrong bit-reverse assertions fail.
# Category: soundness
cat > "$TMPDIR/bit_reverse4_soundness.rs" << 'ENDOFFILE'
fn bit_reverse4_soundness_proof() {
    let x = 0b1011i32;
    let b0 = (x >> 0) & 1;
    let b1 = (x >> 1) & 1;
    let b2 = (x >> 2) & 1;
    let b3 = (x >> 3) & 1;
    let reversed = (b0 << 3) | (b1 << 2) | (b2 << 1) | (b3 << 0);
    assert!(reversed == 0b1011);  // MUST fail: reversed is 0b1101
}
ENDOFFILE
run_test "bit_reverse4_soundness" "$TMPDIR/bit_reverse4_soundness.rs" "fail"


# Test 319b: CTZ simple soundness
# Purpose: Verifies wrong CTZ assertions fail.
# Category: soundness
cat > "$TMPDIR/ctz_simple_soundness.rs" << 'ENDOFFILE'
fn ctz_simple_soundness_proof() {
    let x = 24i32;
    let count = if (x & 1) != 0 { 0 } else if (x & 2) != 0 { 1 } else if (x & 4) != 0 { 2 } else if (x & 8) != 0 { 3 } else if (x & 16) != 0 { 4 } else { 5 };
    assert!(count == 0);  // MUST fail: count is 3
}
ENDOFFILE
run_test "ctz_simple_soundness" "$TMPDIR/ctz_simple_soundness.rs" "fail"


# Test 320b: Cond negate soundness
# Purpose: Verifies wrong cond-negate assertions fail.
# Category: soundness
cat > "$TMPDIR/cond_negate_soundness.rs" << 'ENDOFFILE'
fn cond_negate_soundness_proof() {
    let x = 42i32;
    let should_negate = true;
    let mask = if should_negate { -1i32 } else { 0i32 };
    let result = (x ^ mask) - mask;
    assert!(result == 42);  // MUST fail: result is -42
}
ENDOFFILE
run_test "cond_negate_soundness" "$TMPDIR/cond_negate_soundness.rs" "fail"


# Test 322b: Overflowing add soundness
# Purpose: Verifies wrong wrapping_add assertions fail.
# Category: soundness
cat > "$TMPDIR/overflowing_add_soundness.rs" << 'ENDOFFILE'
fn overflowing_add_soundness_proof() {
    let a: i32 = 100;
    let b: i32 = 50;
    let result = a.wrapping_add(b);
    assert!(result == 999);  // MUST fail: result is 150
}
ENDOFFILE
run_test "overflowing_add_soundness" "$TMPDIR/overflowing_add_soundness.rs" "fail"


# Test 324b: Overflowing sub soundness
# Purpose: Verifies wrong wrapping_sub assertions fail.
# Category: soundness
cat > "$TMPDIR/overflowing_sub_soundness.rs" << 'ENDOFFILE'
fn overflowing_sub_soundness_proof() {
    let a: i32 = 100;
    let b: i32 = 50;
    let result = a.wrapping_sub(b);
    assert!(result == 999);  // MUST fail: result is 50
}
ENDOFFILE
run_test "overflowing_sub_soundness" "$TMPDIR/overflowing_sub_soundness.rs" "fail"


# Test 325b: Saturating subtraction soundness
# Purpose: Verifies wrong saturating_sub assertions fail.
# Category: soundness
cat > "$TMPDIR/saturating_sub_soundness.rs" << 'ENDOFFILE'
fn saturating_sub_soundness_proof() {
    let a: i32 = 100;
    let b: i32 = 50;
    let result = a.saturating_sub(b);
    // WRONG: result is 50, not 999 - should fail verification
    assert!(result == 999);
}
ENDOFFILE
run_test "saturating_sub_soundness" "$TMPDIR/saturating_sub_soundness.rs" "fail"


# Test 326b: Wrapping subtraction soundness
# Purpose: Verifies wrong wrapping_sub assertions fail.
# Category: soundness
cat > "$TMPDIR/wrapping_sub_soundness.rs" << 'ENDOFFILE'
fn wrapping_sub_soundness_proof() {
    let a: i32 = 100;
    let b: i32 = 30;
    let result = a.wrapping_sub(b);
    // WRONG: result is 70, not 999 - should fail verification
    assert!(result == 999);
}
ENDOFFILE
run_test "wrapping_sub_soundness" "$TMPDIR/wrapping_sub_soundness.rs" "fail"


# Test 332b: Wrapping division soundness
# Purpose: Verifies wrong wrapping_div assertions fail.
# Category: soundness
cat > "$TMPDIR/wrapping_div_soundness.rs" << 'ENDOFFILE'
fn wrapping_div_soundness_proof() {
    let a: i32 = 42;
    let result = a.wrapping_div(7);
    // WRONG: result is 6, not 999 - should fail verification
    assert!(result == 999);
}
ENDOFFILE
run_test "wrapping_div_soundness" "$TMPDIR/wrapping_div_soundness.rs" "fail"


# Test 333b: Saturating division soundness
# Purpose: Verifies wrong saturating_div assertions fail.
# Category: soundness
cat > "$TMPDIR/saturating_div_soundness.rs" << 'ENDOFFILE'
fn saturating_div_soundness_proof() {
    let a: i32 = 42;
    let result = a.saturating_div(7);
    assert!(result == 999);
}
ENDOFFILE
run_test "saturating_div_soundness" "$TMPDIR/saturating_div_soundness.rs" "fail"


# Test 336b: Wrapping remainder soundness
# Purpose: Verifies wrong wrapping_rem assertions fail.
# Category: soundness
cat > "$TMPDIR/wrapping_rem_soundness.rs" << 'ENDOFFILE'
fn wrapping_rem_soundness_proof() {
    let a: i32 = 42;
    let result = a.wrapping_rem(5);
    // WRONG: result is 2, not 999 - should fail verification
    assert!(result == 999);
}
ENDOFFILE
run_test "wrapping_rem_soundness" "$TMPDIR/wrapping_rem_soundness.rs" "fail"


# Test 338b: Wrapping negation soundness (signed)
# Purpose: Verifies wrong wrapping_neg assertions fail.
# Category: soundness
cat > "$TMPDIR/wrapping_neg_soundness.rs" << 'ENDOFFILE'
fn wrapping_neg_soundness_proof() {
    let a: i32 = 42;
    let result = a.wrapping_neg();
    // WRONG: result is -42, not 999 - should fail verification
    assert!(result == 999);
}
ENDOFFILE
run_test "wrapping_neg_soundness" "$TMPDIR/wrapping_neg_soundness.rs" "fail"


# Test 339b: Wrapping negation soundness (unsigned)
# Purpose: Verifies wrong unsigned wrapping_neg assertions fail.
# Category: soundness
cat > "$TMPDIR/wrapping_neg_unsigned_soundness.rs" << 'ENDOFFILE'
fn wrapping_neg_unsigned_soundness_proof() {
    let a: u32 = 1;
    let result = a.wrapping_neg();
    // WRONG: result is u32::MAX, not 0 - should fail verification
    assert!(result == 0);
}
ENDOFFILE
run_test "wrapping_neg_unsigned_soundness" "$TMPDIR/wrapping_neg_unsigned_soundness.rs" "fail"


# Test 340b: Checked negation soundness
# Purpose: Verifies wrong checked_neg discriminant assertions fail.
# Category: soundness
cat > "$TMPDIR/checked_neg_soundness.rs" << 'ENDOFFILE'
fn checked_neg_soundness_proof() {
    let a: i32 = 42;
    let is_some = match a.checked_neg() {
        Some(_) => true,
        None => false,
    };
    // WRONG: is_some should be true, not false - should fail verification
    assert!(!is_some);
}
ENDOFFILE
run_test "checked_neg_soundness" "$TMPDIR/checked_neg_soundness.rs" "fail"


# Test 342b: Wrapping shift left soundness
# Purpose: Verifies wrong wrapping_shl assertions fail (fixed #213).
# Category: soundness
cat > "$TMPDIR/wrapping_shl_soundness.rs" << 'ENDOFFILE'
fn wrapping_shl_soundness_proof() {
    let a: i32 = 1;
    let result = a.wrapping_shl(3);
    // Result is unconstrained, so any assertion passes
    assert!(result == 999);
}
ENDOFFILE
run_test "wrapping_shl_soundness" "$TMPDIR/wrapping_shl_soundness.rs" "fail"  # Fixed in #213: now correctly constrained


# Test 343b: Wrapping shift right soundness
# Purpose: Verifies wrong wrapping_shr assertions fail (fixed #213).
# Category: soundness
cat > "$TMPDIR/wrapping_shr_soundness.rs" << 'ENDOFFILE'
fn wrapping_shr_soundness_proof() {
    let a: i32 = 8;
    let result = a.wrapping_shr(2);
    // Result is unconstrained, so any assertion passes
    assert!(result == 999);
}
ENDOFFILE
run_test "wrapping_shr_soundness" "$TMPDIR/wrapping_shr_soundness.rs" "fail"  # Fixed in #213: now correctly constrained


# Test 344b: Boundary arithmetic - MAX wrapping soundness
# Purpose: Verifies wrong wrapping_add at MAX boundary correctly fails.
# Category: soundness
# Note: Fixed in #442 - this test was incorrectly in unsound.sh
cat > "$TMPDIR/boundary_max_wrapping_soundness.rs" << 'ENDOFFILE'
fn boundary_max_wrapping_soundness_proof() {
    let max: i32 = 2147483647;
    let result = max.wrapping_add(1);
    // WRONG: wrapping at MAX goes to MIN (-2147483648), not 0
    assert!(result == 0);
}
ENDOFFILE
run_test "boundary_max_wrapping_soundness" "$TMPDIR/boundary_max_wrapping_soundness.rs" "fail"


# Test 345b: Boundary arithmetic - MAX saturating soundness
# Purpose: Verifies wrong saturating_add at MAX boundary correctly fails.
# Category: soundness
# Note: Fixed in #442 - this test was incorrectly in unsound.sh
cat > "$TMPDIR/boundary_max_saturating_soundness.rs" << 'ENDOFFILE'
fn boundary_max_saturating_soundness_proof() {
    let max: i32 = 2147483647;
    let result = max.saturating_add(1);
    // WRONG: saturating at MAX stays at MAX, not 0
    assert!(result == 0);
}
ENDOFFILE
run_test "boundary_max_saturating_soundness" "$TMPDIR/boundary_max_saturating_soundness.rs" "fail"


# Test 346b: Boundary arithmetic - MIN saturating soundness
# Purpose: Verifies wrong saturating_sub at MIN boundary correctly fails.
# Category: soundness
# Note: Fixed in #442 - this test was incorrectly in unsound.sh
cat > "$TMPDIR/boundary_min_saturating_soundness.rs" << 'ENDOFFILE'
fn boundary_min_saturating_soundness_proof() {
    let min: i32 = -2147483648;
    let result = min.saturating_sub(1);
    // WRONG: saturating at MIN stays at MIN, not 0
    assert!(result == 0);
}
ENDOFFILE
run_test "boundary_min_saturating_soundness" "$TMPDIR/boundary_min_saturating_soundness.rs" "fail"


# Test 347b: Boundary arithmetic - unsigned wrapping soundness
# Purpose: Verifies wrong u32 wrapping_add at MAX correctly fails.
# Category: soundness
# Note: Fixed in #442 - this test was incorrectly in unsound.sh
cat > "$TMPDIR/boundary_unsigned_soundness.rs" << 'ENDOFFILE'
fn boundary_unsigned_soundness_proof() {
    let max: u32 = 4294967295;
    let wrapped = max.wrapping_add(1);
    // WRONG: MAX + 1 wraps to 0, not 1
    assert!(wrapped == 1);
}
ENDOFFILE
run_test "boundary_unsigned_soundness" "$TMPDIR/boundary_unsigned_soundness.rs" "fail"


# Test 348b: Checked add soundness (FIXED in #438)
# Purpose: Tests that checked_add correctly extracts Option value.
# Category: soundness
# Note: Fixed in #438, moved from unsound.sh in #443
cat > "$TMPDIR/checked_add_soundness.rs" << 'ENDOFFILE'
fn checked_add_soundness_proof() {
    let a: i32 = 100;
    let b: i32 = 50;
    let result = match a.checked_add(b) {
        Some(v) => v,  // v = 150
        None => 0,
    };
    // Correctly constrained: result = 150, so this assertion fails
    assert!(result == 999);  // Should fail: result is 150
}
ENDOFFILE
run_test "checked_add_soundness" "$TMPDIR/checked_add_soundness.rs" "fail"


# Test 349b: Checked sub soundness (FIXED in #438)
# Purpose: Tests that checked_sub correctly extracts Option value.
# Category: soundness
# Note: Fixed in #438, moved from unsound.sh in #443
cat > "$TMPDIR/checked_sub_soundness.rs" << 'ENDOFFILE'
fn checked_sub_soundness_proof() {
    let a: i32 = 100;
    let b: i32 = 50;
    let result = match a.checked_sub(b) {
        Some(v) => v,  // v = 50
        None => 0,
    };
    // Correctly constrained: result = 50, so this assertion fails
    assert!(result == 999);  // Should fail: result is 50
}
ENDOFFILE
run_test "checked_sub_soundness" "$TMPDIR/checked_sub_soundness.rs" "fail"


# Test 350b: Checked division soundness (FIXED in #438)
# Purpose: Tests that checked_div correctly extracts Option value.
# Category: soundness
# Note: Fixed in #438, moved from unsound.sh in #443
cat > "$TMPDIR/checked_div_soundness.rs" << 'ENDOFFILE'
fn checked_div_soundness_proof() {
    let a: i32 = 42;
    if let Some(result) = a.checked_div(7) {
        // Correctly constrained: 42/7=6, not 999
        assert!(result == 999);  // Should fail: result is 6
    }
}
ENDOFFILE
run_test "checked_div_soundness" "$TMPDIR/checked_div_soundness.rs" "fail"


# Test 351b: Checked remainder soundness (FIXED in #438)
# Purpose: Tests that checked_rem correctly extracts Option value.
# Category: soundness
# Note: Fixed in #438, moved from unsound.sh in #443
cat > "$TMPDIR/checked_rem_soundness.rs" << 'ENDOFFILE'
fn checked_rem_soundness_proof() {
    let a: i32 = 42;
    if let Some(result) = a.checked_rem(5) {
        // Correctly constrained: 42%5=2, not 999
        assert!(result == 999);  // Should fail: result is 2
    }
}
ENDOFFILE
run_test "checked_rem_soundness" "$TMPDIR/checked_rem_soundness.rs" "fail"


# Test 352b: Checked shift left soundness (FIXED in #438)
# Purpose: Tests that checked_shl correctly sets discriminant.
# Category: soundness
# Note: Fixed in #438, moved from unsound.sh in #443
cat > "$TMPDIR/checked_shl_soundness.rs" << 'ENDOFFILE'
fn checked_shl_soundness_proof() {
    let a: i32 = 1;
    let is_some = match a.checked_shl(3) {
        Some(_) => true,
        None => false,
    };
    // Correctly constrained: shift by 3 < 32 bits, so result is Some
    assert!(!is_some);  // Should fail: is_some is true
}
ENDOFFILE
run_test "checked_shl_soundness" "$TMPDIR/checked_shl_soundness.rs" "fail"


# Test 353b: Checked shift right soundness (FIXED in #438)
# Purpose: Tests that checked_shr correctly sets discriminant.
# Category: soundness
# Note: Fixed in #438, moved from unsound.sh in #443
cat > "$TMPDIR/checked_shr_soundness.rs" << 'ENDOFFILE'
fn checked_shr_soundness_proof() {
    let a: i32 = 8;
    let is_some = match a.checked_shr(2) {
        Some(_) => true,
        None => false,
    };
    // Correctly constrained: shift by 2 < 32 bits, so result is Some
    assert!(!is_some);  // Should fail: is_some is true
}
ENDOFFILE
run_test "checked_shr_soundness" "$TMPDIR/checked_shr_soundness.rs" "fail"
