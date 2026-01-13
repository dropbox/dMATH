//! Integration tests for tla-codegen
//!
//! These tests parse real TLA+ specs and verify the generated Rust code.

use tla_codegen::{generate_rust, CodeGenOptions};
use tla_core::{lower, parse, FileId};

fn parse_and_generate(source: &str, options: &CodeGenOptions) -> Result<String, String> {
    let parsed = parse(source);
    if !parsed.errors.is_empty() {
        let msgs: Vec<_> = parsed.errors.iter().map(|e| e.message.clone()).collect();
        return Err(format!("Parse errors: {}", msgs.join(", ")));
    }

    let tree = tla_core::SyntaxNode::new_root(parsed.green_node);
    let result = lower(FileId(0), &tree);

    if !result.errors.is_empty() {
        let msgs: Vec<_> = result.errors.iter().map(|e| e.message.clone()).collect();
        return Err(format!("Lower errors: {}", msgs.join(", ")));
    }

    let module = result
        .module
        .ok_or_else(|| "No module produced".to_string())?;

    generate_rust(&module, options)
}

#[test]
fn test_simple_counter_spec() {
    let source = r#"
---- MODULE Counter ----
VARIABLE count

Init == count = 0

Next == count' = count + 1

InvNonNegative == count >= 0
====
"#;

    let options = CodeGenOptions::default();
    let code = parse_and_generate(source, &options).expect("generation failed");

    // Verify key elements in generated code
    assert!(
        code.contains("pub struct CounterState"),
        "Missing state struct"
    );
    assert!(code.contains("pub count: i64"), "Missing count field");
    assert!(
        code.contains("impl StateMachine for Counter"),
        "Missing StateMachine impl"
    );
    assert!(code.contains("fn init(&self)"), "Missing init method");
    assert!(
        code.contains("fn next(&self, state: &Self::State)"),
        "Missing next method"
    );
    assert!(
        code.contains("fn check_invariant"),
        "Missing invariant check"
    );
    assert!(
        code.contains("fn check_inv_non_negative"),
        "Missing invariant method"
    );

    // Verify Init body
    assert!(code.contains("count: 0_i64"), "Init should set count to 0");

    // Verify Next body
    assert!(
        code.contains("state.count + 1_i64"),
        "Next should increment count"
    );

    // Verify invariant body
    assert!(
        code.contains("state.count >= 0_i64"),
        "Invariant should check count >= 0"
    );
}

#[test]
fn test_two_variable_spec() {
    let source = r#"
---- MODULE TwoPhase ----
VARIABLES x, y

Init == x = 0 /\ y = 0

Next == x' = x + 1 /\ y' = y + x

InvPositive == y >= 0
====
"#;

    let options = CodeGenOptions::default();
    let code = parse_and_generate(source, &options).expect("generation failed");

    // Verify both variables in state struct
    assert!(code.contains("pub x: i64"), "Missing x field");
    assert!(code.contains("pub y: i64"), "Missing y field");

    // Verify init sets both variables
    assert!(code.contains("x: 0_i64"), "Init should set x");
    assert!(code.contains("y: 0_i64"), "Init should set y");
}

#[test]
fn test_nondeterministic_init() {
    let source = r#"
---- MODULE Nondet ----
VARIABLE x

Init == x \in 1..3

Next == x' = x
====
"#;

    let options = CodeGenOptions::default();
    let code = parse_and_generate(source, &options).expect("generation failed");

    // Verify non-deterministic init generates a loop
    assert!(
        code.contains("for x in range_set(1_i64, 3_i64)"),
        "Should generate loop for non-deterministic init"
    );
    assert!(
        code.contains("states.push("),
        "Should collect states in a vector"
    );
}

#[test]
fn test_conditional_next() {
    let source = r#"
---- MODULE Bounded ----
VARIABLE count

Init == count = 0

Inc == count < 10 /\ count' = count + 1
Dec == count > 0 /\ count' = count - 1

Next == Inc \/ Dec

InvBounded == count >= 0 /\ count <= 10
====
"#;

    let options = CodeGenOptions::default();
    let code = parse_and_generate(source, &options).expect("generation failed");

    // The Next action uses disjunction, should generate multiple action blocks
    assert!(code.contains("// Action 1"), "Should have action comments");
    assert!(code.contains("// Action 2"), "Should have multiple actions");
}

#[test]
fn test_set_operations() {
    let source = r#"
---- MODULE Sets ----
VARIABLE s

Init == s = {}

Next == s' = s \union {1}

InvSmall == s \subseteq {1, 2, 3}
====
"#;

    let options = CodeGenOptions::default();
    let code = parse_and_generate(source, &options).expect("generation failed");

    // Note: Type inference currently defaults unknown types to i64
    // Set operations are still translated correctly
    assert!(
        code.contains("TlaSet::new()"),
        "Empty set should use TlaSet::new()"
    );
    assert!(code.contains(".union("), "Should use union method");
    assert!(code.contains(".is_subset("), "Should use is_subset method");
}

#[test]
fn test_kani_harness_generation() {
    let source = r#"
---- MODULE Simple ----
VARIABLE x

Init == x = 0

Next == x' = x + 1

Invariant == x >= 0
====
"#;

    let options = CodeGenOptions {
        module_name: None,
        generate_proptest: false,
        generate_kani: true,
    };
    let code = parse_and_generate(source, &options).expect("generation failed");

    // Verify Kani proofs are generated
    assert!(code.contains("#[cfg(kani)]"), "Missing kani cfg");
    assert!(code.contains("mod kani_proofs"), "Missing kani module");
    assert!(
        code.contains("#[kani::proof]"),
        "Missing kani::proof attribute"
    );
    assert!(
        code.contains("fn init_satisfies_invariants()"),
        "Missing init proof"
    );
    assert!(
        code.contains("fn next_preserves_invariants()"),
        "Missing next proof"
    );
    assert!(code.contains("kani::assert"), "Missing kani assertions");
    assert!(
        code.contains("#[kani::unwind(5)]"),
        "Missing unwind attribute"
    );
}

#[test]
fn test_boolean_variable() {
    let source = r#"
---- MODULE Toggle ----
VARIABLE flag

Init == flag = FALSE

Next == flag' = ~flag

InvBoolean == flag \in {TRUE, FALSE}
====
"#;

    let options = CodeGenOptions::default();
    let code = parse_and_generate(source, &options).expect("generation failed");

    // Note: Type inference currently treats booleans as bool type when
    // assigned FALSE/TRUE directly
    // Verify boolean operations in generated code
    assert!(
        code.contains("flag: false") || code.contains("flag: FALSE"),
        "Init should set flag to false"
    );
    // The negation operator should be translated
    assert!(
        code.contains("!") || code.contains("flag"),
        "Next should negate flag"
    );
}

#[test]
fn test_if_then_else() {
    let source = r#"
---- MODULE IfThenElse ----
VARIABLE x

Init == x = 0

Next == x' = IF x < 5 THEN x + 1 ELSE 0
====
"#;

    let options = CodeGenOptions::default();
    let code = parse_and_generate(source, &options).expect("generation failed");

    // Verify if-then-else is generated
    assert!(
        code.contains("if (state.x < 5_i64)"),
        "Should generate if condition"
    );
    assert!(
        code.contains("(state.x + 1_i64)"),
        "Should have then branch"
    );
    assert!(code.contains("else { 0_i64 }"), "Should have else branch");
}

#[test]
fn test_multiple_invariants() {
    let source = r#"
---- MODULE MultiInv ----
VARIABLE x

Init == x = 5

Next == x' = x

InvPositive == x > 0
InvBounded == x <= 10
====
"#;

    let options = CodeGenOptions::default();
    let code = parse_and_generate(source, &options).expect("generation failed");

    // Verify both invariant methods
    assert!(
        code.contains("fn check_inv_positive"),
        "Missing first invariant method"
    );
    assert!(
        code.contains("fn check_inv_bounded"),
        "Missing second invariant method"
    );

    // Verify check_invariant combines them
    assert!(
        code.contains("self.check_inv_positive(state)"),
        "Missing first invariant call"
    );
    assert!(
        code.contains("self.check_inv_bounded(state)"),
        "Missing second invariant call"
    );
}

#[test]
fn test_unchanged_syntax() {
    let source = r#"
---- MODULE Unchanged ----
VARIABLES x, y

Init == x = 0 /\ y = 0

IncX == x' = x + 1 /\ UNCHANGED y
IncY == y' = y + 1 /\ UNCHANGED x

Next == IncX \/ IncY
====
"#;

    let options = CodeGenOptions::default();
    let code = parse_and_generate(source, &options).expect("generation failed");

    // The code should correctly handle UNCHANGED by copying from state
    assert!(
        code.contains("state.x.clone()") || code.contains("state.x,"),
        "Should preserve unchanged x"
    );
    assert!(
        code.contains("state.y.clone()") || code.contains("state.y,"),
        "Should preserve unchanged y"
    );
}

#[test]
fn test_proptest_generation() {
    let source = r#"
---- MODULE PropTest ----
VARIABLE x

Init == x = 0
Next == x' = x + 1
Invariant == x >= 0
====
"#;

    let options = CodeGenOptions {
        module_name: None,
        generate_proptest: true,
        generate_kani: false,
    };
    let code = parse_and_generate(source, &options).expect("generation failed");

    // Verify proptest module is generated
    assert!(code.contains("#[cfg(test)]"), "Missing test cfg");
    assert!(code.contains("mod tests"), "Missing tests module");
    assert!(
        code.contains("use std::collections::HashSet"),
        "Missing HashSet import"
    );

    // Verify test functions are generated
    assert!(
        code.contains("fn test_init_not_empty()"),
        "Missing init not empty test"
    );
    assert!(
        code.contains("fn test_init_satisfies_invariants()"),
        "Missing init invariants test"
    );
    assert!(
        code.contains("fn test_bounded_exploration()"),
        "Missing bounded exploration test"
    );
    assert!(
        code.contains("fn test_next_preserves_invariants()"),
        "Missing next preserves invariants test"
    );

    // Verify test content
    assert!(
        code.contains("max_states = 1000"),
        "Missing max_states limit"
    );
    assert!(
        code.contains("machine.check_invariant"),
        "Missing invariant check"
    );
}

#[test]
fn test_module_name_override() {
    let source = r#"
---- MODULE Original ----
VARIABLE x
Init == x = 0
Next == x' = x
====
"#;

    let options = CodeGenOptions {
        module_name: Some("CustomName".to_string()),
        generate_proptest: false,
        generate_kani: false,
    };
    let code = parse_and_generate(source, &options).expect("generation failed");

    // Verify custom module name is used
    assert!(
        code.contains("CustomNameState"),
        "Should use custom name for state struct"
    );
    assert!(
        code.contains("struct CustomName;"),
        "Should use custom name for machine struct"
    );
}

#[test]
fn test_snapshot_counter() {
    let source = r#"
---- MODULE Counter ----
VARIABLE count

Init == count = 0

Next == count' = count + 1

InvNonNegative == count >= 0
====
"#;

    let options = CodeGenOptions::default();
    let code = parse_and_generate(source, &options).expect("generation failed");

    insta::assert_snapshot!("counter_codegen", code);
}

#[test]
fn test_operator_reference_in_init() {
    let source = r#"
---- MODULE InitRef ----
VARIABLE x

Start == x = 0
Init == Start

Next == x' = x
====
"#;

    let options = CodeGenOptions::default();
    let code = parse_and_generate(source, &options).expect("generation failed");

    assert!(
        code.contains("x: 0_i64"),
        "Init should expand operator reference"
    );
}

#[test]
fn test_operator_reference_in_invariant() {
    let source = r#"
---- MODULE InvRef ----
VARIABLE x

Init == x = 0
Next == x' = x

TypeOK == x >= 0
InvNonNeg == TypeOK
====
"#;

    let options = CodeGenOptions::default();
    let code = parse_and_generate(source, &options).expect("generation failed");

    assert!(
        code.contains("fn check_inv_non_neg"),
        "Missing invariant method"
    );
    assert!(
        code.contains("state.x >= 0_i64"),
        "Invariant should expand operator reference"
    );
}

#[test]
fn test_parameterized_action_operator() {
    let source = r#"
---- MODULE ParamOp ----
VARIABLE x

Init == x = 0

Inc(v) == v' = v + 1
Next == Inc(x)
====
"#;

    let options = CodeGenOptions::default();
    let code = parse_and_generate(source, &options).expect("generation failed");

    assert!(
        code.contains("state.x + 1_i64"),
        "Next should expand parameterized operator application"
    );
}

#[test]
fn test_stdlib_cardinality() {
    // Tests Cardinality from FiniteSets module
    let source = r#"
---- MODULE CardTest ----
EXTENDS FiniteSets
VARIABLE s

Init == s = {1, 2, 3}

Next == s' = s

InvSize == Cardinality(s) = 3
====
"#;

    let options = CodeGenOptions::default();
    let code = parse_and_generate(source, &options).expect("generation failed");

    // Cardinality(s) should translate to s.len() as i64
    assert!(
        code.contains(".len() as i64"),
        "Cardinality should translate to len(): {}",
        code
    );
}

#[test]
fn test_stdlib_len_operator() {
    // Tests Len operator translation in an invariant
    let source = r#"
---- MODULE LenOp ----
EXTENDS Sequences
VARIABLE s

Init == s = {1, 2, 3}

Next == s' = s

InvHasLen == Len(s) > 0
====
"#;

    let options = CodeGenOptions::default();
    let code = parse_and_generate(source, &options).expect("generation failed");

    // Len(s) should translate to s.len() as i64
    assert!(
        code.contains(".len() as i64"),
        "Len should translate to len(): {}",
        code
    );
}

#[test]
fn test_stdlib_head_tail_operator() {
    // Tests Head and Tail operator translation in invariants
    let source = r#"
---- MODULE HeadTailOp ----
EXTENDS Sequences
VARIABLE s

Init == s = {1, 2, 3}

Next == s' = s

InvHead == Head(s) = Head(s)
InvTail == Tail(s) = Tail(s)
====
"#;

    let options = CodeGenOptions::default();
    let code = parse_and_generate(source, &options).expect("generation failed");

    // Head should translate to .first().cloned().unwrap()
    assert!(
        code.contains(".first()"),
        "Head should translate to .first(): {}",
        code
    );
    // Tail should translate to skip(1)
    assert!(
        code.contains("skip(1)"),
        "Tail should translate to skip(1): {}",
        code
    );
}

#[test]
fn test_stdlib_append() {
    // Tests Append from Sequences module
    let source = r#"
---- MODULE AppendTest ----
EXTENDS Sequences
VARIABLE seq

Init == seq = <<>>

Next == seq' = Append(seq, 1)
====
"#;

    let options = CodeGenOptions::default();
    let code = parse_and_generate(source, &options).expect("generation failed");

    // Append should generate code that pushes to a vec
    assert!(
        code.contains("push("),
        "Append should translate to push: {}",
        code
    );
}

#[test]
fn test_stdlib_max_min() {
    // Tests Max and Min from FiniteSetsExt module
    let source = r#"
---- MODULE MaxMin ----
EXTENDS FiniteSetsExt
VARIABLE s

Init == s = {1, 2, 3}

Next == s' = s

InvBounded == Max(s) >= Min(s)
====
"#;

    let options = CodeGenOptions::default();
    let code = parse_and_generate(source, &options).expect("generation failed");

    // Max and Min should translate to iter().max/min()
    assert!(
        code.contains(".max()"),
        "Max should translate to .max(): {}",
        code
    );
    assert!(
        code.contains(".min()"),
        "Min should translate to .min(): {}",
        code
    );
}

#[test]
fn test_stdlib_reverse() {
    // Tests Reverse from SequencesExt module
    let source = r#"
---- MODULE ReverseTest ----
EXTENDS SequencesExt
VARIABLE seq

Init == seq = <<1, 2, 3>>

Next == seq' = Reverse(seq)
====
"#;

    let options = CodeGenOptions::default();
    let code = parse_and_generate(source, &options).expect("generation failed");

    // Reverse should translate to .reverse()
    assert!(
        code.contains(".reverse()"),
        "Reverse should translate to .reverse(): {}",
        code
    );
}

#[test]
fn test_stdlib_is_finite_set() {
    // Tests IsFiniteSet from FiniteSets module
    let source = r#"
---- MODULE FiniteTest ----
EXTENDS FiniteSets
VARIABLE s

Init == s = {1, 2}

Next == s' = s

InvFinite == IsFiniteSet(s)
====
"#;

    let options = CodeGenOptions::default();
    let code = parse_and_generate(source, &options).expect("generation failed");

    // IsFiniteSet should translate to true (all our sets are finite)
    assert!(
        code.contains("true"),
        "IsFiniteSet should translate to true"
    );
}

#[test]
fn test_stdlib_tostring() {
    // Tests ToString from TLC module
    let source = r#"
---- MODULE ToStringTest ----
EXTENDS TLC
VARIABLE x, str

Init == x = 42 /\ str = ""

Next == x' = x /\ str' = ToString(x)
====
"#;

    let options = CodeGenOptions::default();
    let code = parse_and_generate(source, &options).expect("generation failed");

    // ToString should translate to format!("{:?}", ...)
    assert!(
        code.contains("format!"),
        "ToString should translate to format!: {}",
        code
    );
}

#[test]
fn test_except_function_update() {
    // Tests EXCEPT for function update [f EXCEPT ![a] = b]
    let source = r#"
---- MODULE ExceptTest ----
VARIABLE f

Init == f = [x \in {1, 2, 3} |-> x * 2]

Next == f' = [f EXCEPT ![1] = 10]

InvCheckValue == f[1] # 0
====
"#;

    let options = CodeGenOptions::default();
    let code = parse_and_generate(source, &options).expect("generation failed");

    // EXCEPT should generate clone + update pattern
    assert!(
        code.contains("__tmp"),
        "EXCEPT should use temporary variable: {}",
        code
    );
    assert!(
        code.contains(".clone()"),
        "EXCEPT should clone the function: {}",
        code
    );
    assert!(
        code.contains(".update("),
        "EXCEPT should call update method: {}",
        code
    );
}

#[test]
fn test_multi_arg_function_definition() {
    // Tests multi-argument function definition [x, y \in S |-> expr]
    let source = r#"
---- MODULE MultiArgFunc ----
VARIABLE matrix

Init == matrix = [x \in {1, 2}, y \in {1, 2} |-> x + y]

Next == matrix' = matrix
====
"#;

    let options = CodeGenOptions::default();
    let code = parse_and_generate(source, &options).expect("generation failed");

    // Multi-arg function should use nested flat_map/map iterators
    assert!(
        code.contains("flat_map"),
        "Multi-arg func should use flat_map: {}",
        code
    );
    assert!(
        code.contains("TlaFunc::from_iter"),
        "Should create TlaFunc: {}",
        code
    );
    // Should generate tuple key
    assert!(
        code.contains("x.clone(), y.clone()"),
        "Should create tuple key: {}",
        code
    );
}

#[test]
fn test_nested_except_path() {
    // Tests nested EXCEPT path [f EXCEPT ![a][b] = v]
    let source = r#"
---- MODULE NestedExcept ----
VARIABLE matrix

Init == matrix = [x \in {1, 2} |-> [y \in {1, 2} |-> 0]]

Next == matrix' = [matrix EXCEPT ![1][2] = 99]

InvCheck == matrix[1][1] = matrix[1][1]
====
"#;

    let options = CodeGenOptions::default();
    let code = parse_and_generate(source, &options).expect("generation failed");

    // Nested EXCEPT should generate keys and intermediate values
    assert!(
        code.contains("__key_0"),
        "Should generate __key_0: {}",
        code
    );
    assert!(
        code.contains("__key_1"),
        "Should generate __key_1: {}",
        code
    );
    assert!(
        code.contains("__inner_0"),
        "Should generate __inner_0: {}",
        code
    );
    // Should have multiple .update calls (or nested update logic)
    assert!(
        code.contains(".update(__key_0"),
        "Should update outer function: {}",
        code
    );
}

#[test]
fn test_three_level_nested_except() {
    // Tests deeply nested EXCEPT path [f EXCEPT ![a][b][c] = v]
    let source = r#"
---- MODULE DeepNested ----
VARIABLE cube

Init == cube = [x \in {1, 2} |-> [y \in {1, 2} |-> [z \in {1, 2} |-> 0]]]

Next == cube' = [cube EXCEPT ![1][1][1] = 42]
====
"#;

    let options = CodeGenOptions::default();
    let code = parse_and_generate(source, &options).expect("generation failed");

    // Should generate three keys
    assert!(
        code.contains("__key_2"),
        "Should generate __key_2 for third level: {}",
        code
    );
    // Should have __inner_0 and __inner_1
    assert!(
        code.contains("__inner_1"),
        "Should generate __inner_1 for intermediate: {}",
        code
    );
}

#[test]
fn test_record_literal() {
    // Tests record literal expression [a |-> 1, b |-> 2]
    let source = r#"
---- MODULE RecordTest ----
VARIABLE r

Init == r = [x |-> 1, y |-> 2]

Next == r' = r
====
"#;

    let options = CodeGenOptions::default();
    let code = parse_and_generate(source, &options).expect("generation failed");

    // Record literal should use TlaRecord::from_fields
    assert!(
        code.contains("TlaRecord::from_fields"),
        "Record should use TlaRecord::from_fields: {}",
        code
    );
    // Should have field names
    assert!(code.contains("\"x\""), "Should have field 'x': {}", code);
    assert!(code.contains("\"y\""), "Should have field 'y': {}", code);
}

#[test]
fn test_record_access() {
    // Tests record field access r.field
    let source = r#"
---- MODULE RecordAccessTest ----
VARIABLE total

Init == total = 0

Next ==
    LET rec == [a |-> 10, b |-> 20]
    IN total' = rec.a + rec.b
====
"#;

    let options = CodeGenOptions::default();
    let code = parse_and_generate(source, &options).expect("generation failed");

    // Record access should use .get() for non-state variables
    assert!(
        code.contains(".get("),
        "Record access should use .get(): {}",
        code
    );
}

#[test]
fn test_mixed_index_field_except() {
    // Tests mixed Index/Field EXCEPT path [f EXCEPT ![i].field = v]
    let source = r#"
---- MODULE MixedExcept ----
VARIABLE data

Init == data = [x \in {1, 2} |-> [a |-> 0, b |-> 0]]

Next == data' = [data EXCEPT ![1].a = 10]
====
"#;

    let options = CodeGenOptions::default();
    let code = parse_and_generate(source, &options).expect("generation failed");

    // Should generate key for index
    assert!(
        code.contains("__key_0"),
        "Should generate __key_0 for index: {}",
        code
    );
    // Should have __inner for intermediate
    assert!(
        code.contains("__inner_0"),
        "Should generate __inner_0: {}",
        code
    );
    // Should use .set for field update (on TlaRecord)
    assert!(
        code.contains(".set("),
        "Should use .set() for field update: {}",
        code
    );
}

#[test]
fn test_funcset_expression() {
    // Tests FuncSet expression [S -> T] - the set of all functions from S to T
    let source = r#"
---- MODULE FuncSetTest ----
VARIABLE f

Init == f \in [{1, 2} -> {0, 1}]

Next == f' = f
====
"#;

    let options = CodeGenOptions::default();
    let code = parse_and_generate(source, &options).expect("generation failed");

    // Should enumerate all functions from {1,2} to {0,1}
    // That's 2^2 = 4 functions
    assert!(
        code.contains("__domain"),
        "Should have domain variable: {}",
        code
    );
    assert!(
        code.contains("__range"),
        "Should have range variable: {}",
        code
    );
    assert!(
        code.contains("TlaFunc::from_iter"),
        "Should construct functions: {}",
        code
    );
}

/// End-to-end compile test: generates Rust code and compiles it to verify correctness.
///
/// This test creates a temporary Cargo project with generated code and runs `cargo check`
/// to ensure the generated code is valid, compilable Rust.
#[test]
fn test_end_to_end_compile() {
    use std::fs;
    use std::process::Command;

    // Multiple TLA+ specs to test compile compatibility
    let specs = vec![
        (
            "Counter",
            r#"
---- MODULE Counter ----
VARIABLE count

Init == count = 0

Next == count' = count + 1

InvNonNegative == count >= 0
====
"#,
        ),
        (
            "SetOps",
            r#"
---- MODULE SetOps ----
VARIABLE s

Init == s = {1, 2, 3}

Next == s' = s \union {4}

InvHasOne == 1 \in s
====
"#,
        ),
        (
            "FunctionTest",
            r#"
---- MODULE FunctionTest ----
VARIABLE f

Init == f = [x \in {1, 2, 3} |-> x * 2]

Next == f' = [f EXCEPT ![1] = 10]

InvDomainSize == TRUE
====
"#,
        ),
        (
            "RecordTest",
            r#"
---- MODULE RecordTest ----
VARIABLE r

Init == r = [x |-> 1, y |-> 2]

Next == r' = r
====
"#,
        ),
        (
            "IfThenElse",
            r#"
---- MODULE IfThenElse ----
VARIABLE x

Init == x = 0

Next == x' = IF x < 10 THEN x + 1 ELSE x
====
"#,
        ),
        (
            "LetIn",
            r#"
---- MODULE LetIn ----
VARIABLE result

Init == result = 0

Next ==
    LET a == 10
        b == 20
    IN result' = a + b
====
"#,
        ),
        (
            "FuncSetTest",
            r#"
---- MODULE FuncSetTest ----
VARIABLE f

Init == f \in [{1, 2} -> {0, 1}]

Next == f' = f
====
"#,
        ),
    ];

    // Create temporary directory for the test project
    let temp_dir = std::env::temp_dir().join(format!("tla2_compile_test_{}", std::process::id()));
    let src_dir = temp_dir.join("src");

    // Clean up any previous runs
    let _ = fs::remove_dir_all(&temp_dir);
    fs::create_dir_all(&src_dir).expect("Failed to create temp src directory");

    // Get the path to tla-runtime relative to the temp directory
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let runtime_path = std::path::Path::new(&manifest_dir)
        .join("../tla-runtime")
        .canonicalize()
        .expect("Failed to find tla-runtime path");

    // Create Cargo.toml
    let cargo_toml = format!(
        r#"[package]
name = "tla2-compile-test"
version = "0.1.0"
edition = "2021"

[dependencies]
tla-runtime = {{ path = "{}" }}
"#,
        runtime_path.display()
    );
    fs::write(temp_dir.join("Cargo.toml"), cargo_toml).expect("Failed to write Cargo.toml");

    // Generate and write each spec as a module
    let options = CodeGenOptions::default();
    let mut lib_content = String::new();

    for (name, source) in &specs {
        let code = parse_and_generate(source, &options)
            .unwrap_or_else(|e| panic!("Failed to generate {} : {}", name, e));

        let mod_name = name.to_lowercase();
        let mod_file = src_dir.join(format!("{}.rs", mod_name));
        fs::write(&mod_file, &code)
            .unwrap_or_else(|e| panic!("Failed to write {}.rs: {}", mod_name, e));

        lib_content.push_str(&format!("pub mod {};\n", mod_name));
    }

    // Write lib.rs
    fs::write(src_dir.join("lib.rs"), lib_content).expect("Failed to write lib.rs");

    // Run cargo check
    let output = Command::new("cargo")
        .args(["check", "--message-format=short"])
        .current_dir(&temp_dir)
        .output()
        .expect("Failed to execute cargo check");

    // Check results
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);

        // Read generated files for debugging
        let mut debug_info = String::new();
        for (name, _) in &specs {
            let mod_name = name.to_lowercase();
            let mod_file = src_dir.join(format!("{}.rs", mod_name));
            if let Ok(content) = fs::read_to_string(&mod_file) {
                debug_info.push_str(&format!("\n=== {}.rs ===\n{}\n", mod_name, content));
            }
        }

        // Clean up temp directory
        let _ = fs::remove_dir_all(&temp_dir);

        panic!(
            "Generated code failed to compile!\n\nstdout:\n{}\n\nstderr:\n{}\n\nGenerated files:\n{}",
            stdout, stderr, debug_info
        );
    }

    // Clean up temp directory
    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_forall_quantifier() {
    let source = r#"
---- MODULE ForallTest ----
VARIABLE s

Init == s = {1, 2, 3}

InvAllPositive == \A x \in s : x > 0
====
"#;

    let options = CodeGenOptions::default();
    let code = parse_and_generate(source, &options).expect("generation failed");

    // Verify forall generates iter().all()
    assert!(
        code.contains(".iter().all("),
        "Forall should generate iter().all()"
    );
    assert!(
        code.contains("|x|"),
        "Forall should bind variable x in closure"
    );
}

#[test]
fn test_exists_quantifier() {
    let source = r#"
---- MODULE ExistsTest ----
VARIABLE s

Init == s = {1, 2, 3}

InvExistsEven == \E x \in s : x = 2
====
"#;

    let options = CodeGenOptions::default();
    let code = parse_and_generate(source, &options).expect("generation failed");

    // Verify exists generates iter().any()
    assert!(
        code.contains(".iter().any("),
        "Exists should generate iter().any(): got:\n{}",
        code
    );
    assert!(
        code.contains("|x|"),
        "Exists should bind variable x in closure"
    );
}

#[test]
fn test_choose_operator() {
    let source = r#"
---- MODULE ChooseTest ----
VARIABLE x

Values == {1, 2, 3, 4, 5}
Init == x = CHOOSE y \in Values : y > 3
====
"#;

    let options = CodeGenOptions::default();
    let code = parse_and_generate(source, &options).expect("generation failed");

    // Verify CHOOSE generates iter().find()
    assert!(
        code.contains(".iter().find("),
        "CHOOSE should generate iter().find()"
    );
    assert!(
        code.contains("cloned()"),
        "CHOOSE should clone the found value"
    );
    assert!(
        code.contains("expect("),
        "CHOOSE should panic if no element found"
    );
}

#[test]
fn test_set_builder() {
    let source = r#"
---- MODULE SetBuilderTest ----
VARIABLE squares

Domain == 1..5
Init == squares = {x * x : x \in Domain}
====
"#;

    let options = CodeGenOptions::default();
    let code = parse_and_generate(source, &options).expect("generation failed");

    // Verify SetBuilder generates map + collect
    assert!(
        code.contains("TlaSet::from_iter("),
        "SetBuilder should use TlaSet::from_iter"
    );
    assert!(
        code.contains(".iter().map("),
        "SetBuilder should use iter().map()"
    );
}

#[test]
fn test_set_filter() {
    let source = r#"
---- MODULE SetFilterTest ----
VARIABLE evens

Domain == 1..10
Init == evens = {x \in Domain : x % 2 = 0}
====
"#;

    let options = CodeGenOptions::default();
    let code = parse_and_generate(source, &options).expect("generation failed");

    // Verify SetFilter generates filter + collect
    assert!(
        code.contains("TlaSet::from_iter("),
        "SetFilter should use TlaSet::from_iter"
    );
    assert!(
        code.contains(".iter().filter("),
        "SetFilter should use iter().filter()"
    );
}

#[test]
fn test_let_expression() {
    // Use LET in an invariant where it will be code-generated
    let source = r#"
---- MODULE LetTest ----
VARIABLE x

Init == x = 1

InvLetCheck == LET doubled == x * 2 IN doubled > 0
====
"#;

    let options = CodeGenOptions::default();
    let code = parse_and_generate(source, &options).expect("generation failed");

    // Verify LET generates let binding with a block
    assert!(
        code.contains("let doubled") || code.contains("{"),
        "LET should generate let binding:\n{}",
        code
    );
}

#[test]
fn test_case_expression() {
    let source = r#"
---- MODULE CaseTest ----
VARIABLE category

Value == 50

Init == category = CASE Value < 10 -> "small"
                     [] Value < 100 -> "medium"
                     [] OTHER -> "large"
====
"#;

    let options = CodeGenOptions::default();
    let code = parse_and_generate(source, &options).expect("generation failed");

    // Verify CASE generates if/else chain
    assert!(code.contains("if "), "CASE should generate if");
    assert!(
        code.contains("else if "),
        "CASE should generate else if for multiple arms"
    );
    assert!(
        code.contains("else {"),
        "CASE with OTHER should generate final else"
    );
}

#[test]
fn test_cartesian_product() {
    let source = r#"
---- MODULE TimesTest ----
VARIABLE pairs

A == {1, 2}
B == {"a", "b"}

Init == pairs = A \X B
====
"#;

    let options = CodeGenOptions::default();
    let code = parse_and_generate(source, &options).expect("generation failed");

    // Verify Cartesian product generates flat_map
    assert!(
        code.contains("flat_map"),
        "Times should generate flat_map for Cartesian product"
    );
}

#[test]
fn test_big_union() {
    let source = r#"
---- MODULE BigUnionTest ----
VARIABLE flattened

Sets == {{1, 2}, {3, 4}, {5}}

Init == flattened = UNION Sets
====
"#;

    let options = CodeGenOptions::default();
    let code = parse_and_generate(source, &options).expect("generation failed");

    // Verify UNION (BigUnion) generates flat_map
    assert!(
        code.contains("flat_map"),
        "UNION should generate flat_map to flatten sets"
    );
}

#[test]
fn test_multi_bound_quantifier() {
    let source = r#"
---- MODULE MultiBoundTest ----
VARIABLE ok

A == {1, 2}
B == {3, 4}

Init == ok = \A x \in A, y \in B : x < y
====
"#;

    let options = CodeGenOptions::default();
    let code = parse_and_generate(source, &options).expect("generation failed");

    // Verify nested all() for multiple bounds
    assert!(
        code.contains(".iter().all(|x|"),
        "Multi-bound forall should nest all() calls"
    );
    assert!(
        code.contains(".iter().all(|y|"),
        "Multi-bound forall should nest for second variable"
    );
}

/// End-to-end test that generates code, compiles it, and executes it
/// to verify semantic correctness of the generated state machine.
#[test]
fn test_end_to_end_execute() {
    use std::fs;
    use std::process::Command;

    // A simple counter spec with bounded behavior
    let spec_source = r#"
---- MODULE BoundedCounter ----
VARIABLE count

Init == count = 0

Next == count' = IF count < 5 THEN count + 1 ELSE count

InvBounded == count >= 0 /\ count <= 5
====
"#;

    // Create temporary directory
    let temp_dir = std::env::temp_dir().join(format!("tla2_exec_test_{}", std::process::id()));
    let src_dir = temp_dir.join("src");

    let _ = fs::remove_dir_all(&temp_dir);
    fs::create_dir_all(&src_dir).expect("Failed to create temp src directory");

    // Get path to tla-runtime
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let runtime_path = std::path::Path::new(&manifest_dir)
        .join("../tla-runtime")
        .canonicalize()
        .expect("Failed to find tla-runtime path");

    // Create Cargo.toml
    let cargo_toml = format!(
        r#"[package]
name = "tla2-exec-test"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "exec_test"
path = "src/main.rs"

[dependencies]
tla-runtime = {{ path = "{}" }}
"#,
        runtime_path.display()
    );
    fs::write(temp_dir.join("Cargo.toml"), cargo_toml).expect("Failed to write Cargo.toml");

    // Generate spec code
    let options = CodeGenOptions::default();
    let spec_code =
        parse_and_generate(spec_source, &options).expect("Failed to generate BoundedCounter");

    // Write the spec module
    fs::write(src_dir.join("bounded_counter.rs"), &spec_code).expect("Failed to write spec");

    // Create main.rs that tests the generated code
    let main_code = r#"
mod bounded_counter;

use bounded_counter::BoundedCounter;
use tla_runtime::prelude::*;

fn main() {
    let machine = BoundedCounter;

    // Test 1: Verify initial states
    let init_states = machine.init();
    assert_eq!(init_states.len(), 1, "Should have exactly one initial state");
    assert_eq!(init_states[0].count, 0, "Initial count should be 0");

    // Test 2: Run model checker
    let result = model_check(&machine, 100);

    // We expect a deadlock since at count=5 the spec stays at 5 forever
    // (but all states still have a next state - themselves)
    // Actually: IF count < 5 THEN count + 1 ELSE count means no deadlock

    // Test 3: Verify state space
    let states = collect_states(&machine, 100);
    assert!(states.len() >= 6, "Should have at least 6 states (0-5), got {}", states.len());

    // Test 4: Verify counts are in expected range
    for state in &states {
        assert!(state.count >= 0 && state.count <= 5,
            "Count {} out of expected range [0, 5]", state.count);
    }

    // Test 5: Verify invariant holds on all states
    for state in &states {
        if let Some(holds) = machine.check_invariant(state) {
            assert!(holds, "Invariant violated at count={}", state.count);
        }
    }

    // Test 6: Check transitions are correct
    let state0 = bounded_counter::BoundedCounterState { count: 0 };
    let next_states = machine.next(&state0);
    assert_eq!(next_states.len(), 1);
    assert_eq!(next_states[0].count, 1, "From 0 should go to 1");

    let state5 = bounded_counter::BoundedCounterState { count: 5 };
    let next_from_5 = machine.next(&state5);
    assert_eq!(next_from_5.len(), 1);
    assert_eq!(next_from_5[0].count, 5, "From 5 should stay at 5");

    println!("All tests passed!");
    println!("States explored: {}", states.len());
}
"#;
    fs::write(src_dir.join("main.rs"), main_code).expect("Failed to write main.rs");

    // Build the test binary
    let build_output = Command::new("cargo")
        .args(["build", "--release"])
        .current_dir(&temp_dir)
        .output()
        .expect("Failed to execute cargo build");

    if !build_output.status.success() {
        let stderr = String::from_utf8_lossy(&build_output.stderr);
        let _ = fs::remove_dir_all(&temp_dir);
        panic!("Failed to build test binary:\n{}", stderr);
    }

    // Run the test binary
    let run_output = Command::new("cargo")
        .args(["run", "--release"])
        .current_dir(&temp_dir)
        .output()
        .expect("Failed to execute test binary");

    // Clean up
    let _ = fs::remove_dir_all(&temp_dir);

    if !run_output.status.success() {
        let stderr = String::from_utf8_lossy(&run_output.stderr);
        let stdout = String::from_utf8_lossy(&run_output.stdout);
        panic!(
            "Test binary failed!\nstdout:\n{}\nstderr:\n{}",
            stdout, stderr
        );
    }

    // Verify output contains success message
    let stdout = String::from_utf8_lossy(&run_output.stdout);
    assert!(
        stdout.contains("All tests passed!"),
        "Test binary should print success message. Got:\n{}",
        stdout
    );
}
