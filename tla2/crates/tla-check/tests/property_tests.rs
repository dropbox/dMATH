//! Property-based tests for TLA+ evaluator
//!
//! These tests verify algebraic laws and semantic properties using proptest.
//! They ensure correctness of the evaluator across randomized inputs.

use im::OrdSet;
use num_bigint::BigInt;
use proptest::prelude::*;
use tla_check::value::{SortedSet, Value};
use tla_core::{lower, parse_to_syntax_tree, FileId};

// ============================================================================
// Helper functions
// ============================================================================

/// Evaluate a TLA+ expression string and return the result
fn eval_str(src: &str) -> Result<Value, String> {
    let module_src = format!("---- MODULE Test ----\n\nOp == {}\n\n====", src);
    let tree = parse_to_syntax_tree(&module_src);
    let lower_result = lower(FileId(0), &tree);
    let module = match lower_result.module {
        Some(m) => m,
        None => return Err(format!("Parse error: {:?}", lower_result.errors)),
    };

    for unit in &module.units {
        if let tla_core::ast::Unit::Operator(def) = &unit.node {
            if def.name.node == "Op" {
                let ctx = tla_check::eval::EvalCtx::new();
                return tla_check::eval::eval(&ctx, &def.body).map_err(|e| format!("{:?}", e));
            }
        }
    }
    Err("Op not found".to_string())
}

/// Create a Value::Set from a set of integers
fn int_set(values: &[i32]) -> Value {
    let set: OrdSet<Value> = values.iter().map(|&v| Value::int(v.into())).collect();
    Value::Set(SortedSet::from_ord_set(&set))
}

/// Convert Rust bool to TLA+ boolean string
fn tla_bool(b: bool) -> &'static str {
    if b {
        "TRUE"
    } else {
        "FALSE"
    }
}

// ============================================================================
// Boolean operator property tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    // --- AND (conjunction) properties ---

    #[test]
    fn prop_and_identity(a: bool) {
        // a /\ TRUE = a
        let result = eval_str(&format!(r#"{} /\ TRUE"#, tla_bool(a))).unwrap();
        prop_assert_eq!(result, Value::Bool(a));
    }

    #[test]
    fn prop_and_annihilation(a: bool) {
        // a /\ FALSE = FALSE
        let result = eval_str(&format!(r#"{} /\ FALSE"#, tla_bool(a))).unwrap();
        prop_assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn prop_and_commutativity(a: bool, b: bool) {
        // a /\ b = b /\ a
        let lhs = eval_str(&format!(r#"{} /\ {}"#, tla_bool(a), tla_bool(b))).unwrap();
        let rhs = eval_str(&format!(r#"{} /\ {}"#, tla_bool(b), tla_bool(a))).unwrap();
        prop_assert_eq!(lhs, rhs);
    }

    #[test]
    fn prop_and_associativity(a: bool, b: bool, c: bool) {
        // (a /\ b) /\ c = a /\ (b /\ c)
        let lhs = eval_str(&format!(r#"({} /\ {}) /\ {}"#, tla_bool(a), tla_bool(b), tla_bool(c))).unwrap();
        let rhs = eval_str(&format!(r#"{} /\ ({} /\ {})"#, tla_bool(a), tla_bool(b), tla_bool(c))).unwrap();
        prop_assert_eq!(lhs, rhs);
    }

    #[test]
    fn prop_and_idempotence(a: bool) {
        // a /\ a = a
        let result = eval_str(&format!(r#"{} /\ {}"#, tla_bool(a), tla_bool(a))).unwrap();
        prop_assert_eq!(result, Value::Bool(a));
    }

    // --- OR (disjunction) properties ---

    #[test]
    fn prop_or_identity(a: bool) {
        // a \/ FALSE = a
        let result = eval_str(&format!(r#"{} \/ FALSE"#, tla_bool(a))).unwrap();
        prop_assert_eq!(result, Value::Bool(a));
    }

    #[test]
    fn prop_or_annihilation(a: bool) {
        // a \/ TRUE = TRUE
        let result = eval_str(&format!(r#"{} \/ TRUE"#, tla_bool(a))).unwrap();
        prop_assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn prop_or_commutativity(a: bool, b: bool) {
        // a \/ b = b \/ a
        let lhs = eval_str(&format!(r#"{} \/ {}"#, tla_bool(a), tla_bool(b))).unwrap();
        let rhs = eval_str(&format!(r#"{} \/ {}"#, tla_bool(b), tla_bool(a))).unwrap();
        prop_assert_eq!(lhs, rhs);
    }

    #[test]
    fn prop_or_associativity(a: bool, b: bool, c: bool) {
        // (a \/ b) \/ c = a \/ (b \/ c)
        let lhs = eval_str(&format!(r#"({} \/ {}) \/ {}"#, tla_bool(a), tla_bool(b), tla_bool(c))).unwrap();
        let rhs = eval_str(&format!(r#"{} \/ ({} \/ {})"#, tla_bool(a), tla_bool(b), tla_bool(c))).unwrap();
        prop_assert_eq!(lhs, rhs);
    }

    #[test]
    fn prop_or_idempotence(a: bool) {
        // a \/ a = a
        let result = eval_str(&format!(r#"{} \/ {}"#, tla_bool(a), tla_bool(a))).unwrap();
        prop_assert_eq!(result, Value::Bool(a));
    }

    // --- NOT (negation) properties ---

    #[test]
    fn prop_not_involution(a: bool) {
        // ~~a = a
        let result = eval_str(&format!("~~{}", tla_bool(a))).unwrap();
        prop_assert_eq!(result, Value::Bool(a));
    }

    // --- De Morgan's Laws ---

    #[test]
    fn prop_de_morgan_and(a: bool, b: bool) {
        // ~(a /\ b) = ~a \/ ~b
        let lhs = eval_str(&format!(r#"~({} /\ {})"#, tla_bool(a), tla_bool(b))).unwrap();
        let rhs = eval_str(&format!(r#"~{} \/ ~{}"#, tla_bool(a), tla_bool(b))).unwrap();
        prop_assert_eq!(lhs, rhs);
    }

    #[test]
    fn prop_de_morgan_or(a: bool, b: bool) {
        // ~(a \/ b) = ~a /\ ~b
        let lhs = eval_str(&format!(r#"~({} \/ {})"#, tla_bool(a), tla_bool(b))).unwrap();
        let rhs = eval_str(&format!(r#"~{} /\ ~{}"#, tla_bool(a), tla_bool(b))).unwrap();
        prop_assert_eq!(lhs, rhs);
    }

    // --- Implication properties ---

    #[test]
    fn prop_implies_definition(a: bool, b: bool) {
        // (a => b) = (~a \/ b)
        let lhs = eval_str(&format!("{} => {}", tla_bool(a), tla_bool(b))).unwrap();
        let rhs = eval_str(&format!(r#"~{} \/ {}"#, tla_bool(a), tla_bool(b))).unwrap();
        prop_assert_eq!(lhs, rhs);
    }

    #[test]
    fn prop_implies_reflexive(a: bool) {
        // a => a = TRUE
        let result = eval_str(&format!("{} => {}", tla_bool(a), tla_bool(a))).unwrap();
        prop_assert_eq!(result, Value::Bool(true));
    }

    // --- Equivalence properties ---

    #[test]
    fn prop_equiv_reflexive(a: bool) {
        // a <=> a = TRUE
        let result = eval_str(&format!("{} <=> {}", tla_bool(a), tla_bool(a))).unwrap();
        prop_assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn prop_equiv_symmetric(a: bool, b: bool) {
        // (a <=> b) = (b <=> a)
        let lhs = eval_str(&format!("{} <=> {}", tla_bool(a), tla_bool(b))).unwrap();
        let rhs = eval_str(&format!("{} <=> {}", tla_bool(b), tla_bool(a))).unwrap();
        prop_assert_eq!(lhs, rhs);
    }

    // --- Absorption laws ---

    #[test]
    fn prop_absorption_and_or(a: bool, b: bool) {
        // a /\ (a \/ b) = a
        let result = eval_str(&format!(r#"{} /\ ({} \/ {})"#, tla_bool(a), tla_bool(a), tla_bool(b))).unwrap();
        prop_assert_eq!(result, Value::Bool(a));
    }

    #[test]
    fn prop_absorption_or_and(a: bool, b: bool) {
        // a \/ (a /\ b) = a
        let result = eval_str(&format!(r#"{} \/ ({} /\ {})"#, tla_bool(a), tla_bool(a), tla_bool(b))).unwrap();
        prop_assert_eq!(result, Value::Bool(a));
    }
}

// ============================================================================
// Set operator property tests
// ============================================================================

/// Strategy for small sets of integers (for performance)
fn small_int_set() -> impl Strategy<Value = Vec<i32>> {
    prop::collection::vec(-10i32..10, 0..8)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    // --- Union properties ---

    #[test]
    fn prop_union_commutativity(s in small_int_set(), t in small_int_set()) {
        // S \cup T = T \cup S
        let s_str = format!("{{{}}}", s.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let t_str = format!("{{{}}}", t.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let lhs = eval_str(&format!(r#"{} \cup {}"#, s_str, t_str)).unwrap();
        let rhs = eval_str(&format!(r#"{} \cup {}"#, t_str, s_str)).unwrap();
        prop_assert_eq!(lhs, rhs);
    }

    #[test]
    fn prop_union_associativity(s in small_int_set(), t in small_int_set(), u in small_int_set()) {
        // (S \cup T) \cup U = S \cup (T \cup U)
        let s_str = format!("{{{}}}", s.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let t_str = format!("{{{}}}", t.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let u_str = format!("{{{}}}", u.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let lhs = eval_str(&format!(r#"({} \cup {}) \cup {}"#, s_str, t_str, u_str)).unwrap();
        let rhs = eval_str(&format!(r#"{} \cup ({} \cup {})"#, s_str, t_str, u_str)).unwrap();
        prop_assert_eq!(lhs, rhs);
    }

    #[test]
    fn prop_union_identity(s in small_int_set()) {
        // S \cup {} = S
        let s_str = format!("{{{}}}", s.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let result = eval_str(&format!(r#"{} \cup {{}}"#, s_str)).unwrap();
        let expected = int_set(&s);
        prop_assert_eq!(result, expected);
    }

    #[test]
    fn prop_union_idempotence(s in small_int_set()) {
        // S \cup S = S
        let s_str = format!("{{{}}}", s.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let result = eval_str(&format!(r#"{} \cup {}"#, s_str, s_str)).unwrap();
        let expected = int_set(&s);
        prop_assert_eq!(result, expected);
    }

    // --- Intersection properties ---

    #[test]
    fn prop_intersect_commutativity(s in small_int_set(), t in small_int_set()) {
        // S \cap T = T \cap S
        let s_str = format!("{{{}}}", s.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let t_str = format!("{{{}}}", t.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let lhs = eval_str(&format!(r#"{} \cap {}"#, s_str, t_str)).unwrap();
        let rhs = eval_str(&format!(r#"{} \cap {}"#, t_str, s_str)).unwrap();
        prop_assert_eq!(lhs, rhs);
    }

    #[test]
    fn prop_intersect_associativity(s in small_int_set(), t in small_int_set(), u in small_int_set()) {
        // (S \cap T) \cap U = S \cap (T \cap U)
        let s_str = format!("{{{}}}", s.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let t_str = format!("{{{}}}", t.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let u_str = format!("{{{}}}", u.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let lhs = eval_str(&format!(r#"({} \cap {}) \cap {}"#, s_str, t_str, u_str)).unwrap();
        let rhs = eval_str(&format!(r#"{} \cap ({} \cap {})"#, s_str, t_str, u_str)).unwrap();
        prop_assert_eq!(lhs, rhs);
    }

    #[test]
    fn prop_intersect_idempotence(s in small_int_set()) {
        // S \cap S = S
        let s_str = format!("{{{}}}", s.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let result = eval_str(&format!(r#"{} \cap {}"#, s_str, s_str)).unwrap();
        let expected = int_set(&s);
        prop_assert_eq!(result, expected);
    }

    #[test]
    fn prop_intersect_empty(s in small_int_set()) {
        // S \cap {} = {}
        let s_str = format!("{{{}}}", s.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let result = eval_str(&format!(r#"{} \cap {{}}"#, s_str)).unwrap();
        prop_assert_eq!(result, Value::Set(SortedSet::new()));
    }

    // --- Set difference properties ---

    #[test]
    fn prop_setminus_self(s in small_int_set()) {
        // S \ S = {}
        let s_str = format!("{{{}}}", s.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let result = eval_str(&format!(r#"{} \ {}"#, s_str, s_str)).unwrap();
        prop_assert_eq!(result, Value::Set(SortedSet::new()));
    }

    #[test]
    fn prop_setminus_empty(s in small_int_set()) {
        // S \ {} = S
        let s_str = format!("{{{}}}", s.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let result = eval_str(&format!(r#"{} \ {{}}"#, s_str)).unwrap();
        let expected = int_set(&s);
        prop_assert_eq!(result, expected);
    }

    #[test]
    fn prop_empty_setminus(s in small_int_set()) {
        // {} \ S = {}
        let s_str = format!("{{{}}}", s.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let result = eval_str(&format!(r#"{{}} \ {}"#, s_str)).unwrap();
        prop_assert_eq!(result, Value::Set(SortedSet::new()));
    }

    // --- Subset properties ---

    #[test]
    fn prop_subset_reflexive(s in small_int_set()) {
        // S \subseteq S
        let s_str = format!("{{{}}}", s.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let result = eval_str(&format!(r#"{} \subseteq {}"#, s_str, s_str)).unwrap();
        prop_assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn prop_empty_subset_all(s in small_int_set()) {
        // {} \subseteq S
        let s_str = format!("{{{}}}", s.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let result = eval_str(&format!(r#"{{}} \subseteq {}"#, s_str)).unwrap();
        prop_assert_eq!(result, Value::Bool(true));
    }

    // --- Distributive laws ---

    #[test]
    fn prop_union_intersect_distribute(s in small_int_set(), t in small_int_set(), u in small_int_set()) {
        // S \cap (T \cup U) = (S \cap T) \cup (S \cap U)
        let s_str = format!("{{{}}}", s.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let t_str = format!("{{{}}}", t.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let u_str = format!("{{{}}}", u.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let lhs = eval_str(&format!(r#"{} \cap ({} \cup {})"#, s_str, t_str, u_str)).unwrap();
        let rhs = eval_str(&format!(r#"({} \cap {}) \cup ({} \cap {})"#, s_str, t_str, s_str, u_str)).unwrap();
        prop_assert_eq!(lhs, rhs);
    }

    #[test]
    fn prop_intersect_union_distribute(s in small_int_set(), t in small_int_set(), u in small_int_set()) {
        // S \cup (T \cap U) = (S \cup T) \cap (S \cup U)
        let s_str = format!("{{{}}}", s.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let t_str = format!("{{{}}}", t.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let u_str = format!("{{{}}}", u.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let lhs = eval_str(&format!(r#"{} \cup ({} \cap {})"#, s_str, t_str, u_str)).unwrap();
        let rhs = eval_str(&format!(r#"({} \cup {}) \cap ({} \cup {})"#, s_str, t_str, s_str, u_str)).unwrap();
        prop_assert_eq!(lhs, rhs);
    }

    // --- Cardinality properties ---

    #[test]
    fn prop_cardinality_union_intersect(s in small_int_set(), t in small_int_set()) {
        // |S \cup T| + |S \cap T| = |S| + |T|
        let s_str = format!("{{{}}}", s.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let t_str = format!("{{{}}}", t.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));

        let union_card = eval_str(&format!(r#"Cardinality({} \cup {})"#, s_str, t_str)).unwrap();
        let intersect_card = eval_str(&format!(r#"Cardinality({} \cap {})"#, s_str, t_str)).unwrap();
        let s_card = eval_str(&format!(r#"Cardinality({})"#, s_str)).unwrap();
        let t_card = eval_str(&format!(r#"Cardinality({})"#, t_str)).unwrap();

        let lhs = union_card.as_int().expect("Expected integer") + intersect_card.as_int().expect("Expected integer");
        let rhs = s_card.as_int().expect("Expected integer") + t_card.as_int().expect("Expected integer");
        prop_assert_eq!(lhs, rhs);
    }
}

// ============================================================================
// Arithmetic property tests
// ============================================================================

/// Strategy for small integers (avoid overflow in parsing)
fn small_int() -> impl Strategy<Value = i32> {
    -1000i32..1000
}

/// Strategy for positive integers (for division)
fn positive_int() -> impl Strategy<Value = i32> {
    1i32..100
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    // --- Addition properties ---

    #[test]
    fn prop_add_commutativity(a in small_int(), b in small_int()) {
        // a + b = b + a
        let lhs = eval_str(&format!("{} + {}", a, b)).unwrap();
        let rhs = eval_str(&format!("{} + {}", b, a)).unwrap();
        prop_assert_eq!(lhs, rhs);
    }

    #[test]
    fn prop_add_associativity(a in small_int(), b in small_int(), c in small_int()) {
        // (a + b) + c = a + (b + c)
        let lhs = eval_str(&format!("({} + {}) + {}", a, b, c)).unwrap();
        let rhs = eval_str(&format!("{} + ({} + {})", a, b, c)).unwrap();
        prop_assert_eq!(lhs, rhs);
    }

    #[test]
    fn prop_add_identity(a in small_int()) {
        // a + 0 = a
        let result = eval_str(&format!("{} + 0", a)).unwrap();
        prop_assert_eq!(result, Value::int(a.into()));
    }

    #[test]
    fn prop_add_inverse(a in small_int()) {
        // a + (-a) = 0
        let result = eval_str(&format!("{} + ({})", a, -a)).unwrap();
        prop_assert_eq!(result, Value::int(0));
    }

    // --- Multiplication properties ---

    #[test]
    fn prop_mult_commutativity(a in small_int(), b in small_int()) {
        // a * b = b * a
        let lhs = eval_str(&format!("{} * {}", a, b)).unwrap();
        let rhs = eval_str(&format!("{} * {}", b, a)).unwrap();
        prop_assert_eq!(lhs, rhs);
    }

    #[test]
    fn prop_mult_associativity(a in -100i32..100, b in -100i32..100, c in -100i32..100) {
        // (a * b) * c = a * (b * c)
        let lhs = eval_str(&format!("({} * {}) * {}", a, b, c)).unwrap();
        let rhs = eval_str(&format!("{} * ({} * {})", a, b, c)).unwrap();
        prop_assert_eq!(lhs, rhs);
    }

    #[test]
    fn prop_mult_identity(a in small_int()) {
        // a * 1 = a
        let result = eval_str(&format!("{} * 1", a)).unwrap();
        prop_assert_eq!(result, Value::int(a.into()));
    }

    #[test]
    fn prop_mult_zero(a in small_int()) {
        // a * 0 = 0
        let result = eval_str(&format!("{} * 0", a)).unwrap();
        prop_assert_eq!(result, Value::int(0));
    }

    #[test]
    fn prop_distributive(a in -50i32..50, b in -50i32..50, c in -50i32..50) {
        // a * (b + c) = a * b + a * c
        let lhs = eval_str(&format!("{} * ({} + {})", a, b, c)).unwrap();
        let rhs = eval_str(&format!("{} * {} + {} * {}", a, b, a, c)).unwrap();
        prop_assert_eq!(lhs, rhs);
    }

    // --- Subtraction properties ---

    #[test]
    fn prop_sub_definition(a in small_int(), b in small_int()) {
        // a - b = a + (-b)
        let lhs = eval_str(&format!("{} - {}", a, b)).unwrap();
        let rhs = eval_str(&format!("{} + ({})", a, -b)).unwrap();
        prop_assert_eq!(lhs, rhs);
    }

    #[test]
    fn prop_sub_self(a in small_int()) {
        // a - a = 0
        let result = eval_str(&format!("{} - {}", a, a)).unwrap();
        prop_assert_eq!(result, Value::int(0));
    }

    // --- Negation properties ---

    #[test]
    fn prop_negation_involution(a in small_int()) {
        // -(-a) = a
        let result = eval_str(&format!("-({})", -a)).unwrap();
        prop_assert_eq!(result, Value::int(a.into()));
    }

    // --- Comparison properties ---

    #[test]
    fn prop_less_than_irreflexive(a in small_int()) {
        // ~(a < a)
        let result = eval_str(&format!("{} < {}", a, a)).unwrap();
        prop_assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn prop_less_equal_reflexive(a in small_int()) {
        // a <= a
        let result = eval_str(&format!("{} <= {}", a, a)).unwrap();
        prop_assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn prop_equality_reflexive(a in small_int()) {
        // a = a
        let result = eval_str(&format!("{} = {}", a, a)).unwrap();
        prop_assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn prop_inequality_irreflexive(a in small_int()) {
        // ~(a /= a)
        let result = eval_str(&format!("{} /= {}", a, a)).unwrap();
        prop_assert_eq!(result, Value::Bool(false));
    }

    // --- Division/modulo properties (TLA+ uses Euclidean) ---

    #[test]
    fn prop_div_mod_identity(a in small_int(), b in positive_int()) {
        // a = (a \div b) * b + (a % b)
        let div_result = eval_str(&format!(r#"({}) \div {}"#, a, b)).unwrap();
        let mod_result = eval_str(&format!("({}) % {}", a, b)).unwrap();

        // Use to_bigint() to handle both SmallInt and Int variants
        let div_val = div_result.to_bigint().expect("Expected integer for div");
        let mod_val = mod_result.to_bigint().expect("Expected integer for mod");

        let reconstructed: BigInt = &div_val * BigInt::from(b) + &mod_val;
        prop_assert_eq!(reconstructed, BigInt::from(a));
    }

    #[test]
    fn prop_mod_range(a in small_int(), b in positive_int()) {
        // 0 <= (a % b) < b  (Euclidean modulo is always non-negative)
        let mod_result = eval_str(&format!("({}) % {}", a, b)).unwrap();

        // Use to_bigint() to handle both SmallInt and Int variants
        let m = mod_result.to_bigint().expect("Expected integer");
        prop_assert!(m >= BigInt::from(0));
        prop_assert!(m < BigInt::from(b));
    }
}

// ============================================================================
// Quantifier property tests
// ============================================================================

// Non-parameterized quantifier tests (outside proptest! block)
#[test]
fn test_forall_empty() {
    // \A x \in {} : FALSE = TRUE (vacuously true)
    // Note: Using parentheses around the body to avoid parser issues
    let result = eval_str(r#"\A x \in {} : (FALSE)"#).unwrap();
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn test_exists_empty() {
    // \E x \in {} : TRUE = FALSE (no witnesses)
    let result = eval_str(r#"\E x \in {} : (TRUE)"#).unwrap();
    assert_eq!(result, Value::Bool(false));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn prop_forall_singleton(v in -10i32..10) {
        // \A x \in {v} : x = v
        let result = eval_str(&format!(r#"\A x \in {{{}}} : x = {}"#, v, v)).unwrap();
        prop_assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn prop_exists_singleton(v in -10i32..10) {
        // \E x \in {v} : x = v
        let result = eval_str(&format!(r#"\E x \in {{{}}} : x = {}"#, v, v)).unwrap();
        prop_assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn prop_forall_exists_duality(s in small_int_set()) {
        // ~(\A x \in S : x > 0) = \E x \in S : ~(x > 0)
        let s_str = format!("{{{}}}", s.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let lhs = eval_str(&format!(r#"~(\A x \in {} : x > 0)"#, s_str)).unwrap();
        let rhs = eval_str(&format!(r#"\E x \in {} : ~(x > 0)"#, s_str)).unwrap();
        prop_assert_eq!(lhs, rhs);
    }

    #[test]
    fn prop_exists_forall_duality(s in small_int_set()) {
        // ~(\E x \in S : x > 0) = \A x \in S : ~(x > 0)
        let s_str = format!("{{{}}}", s.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let lhs = eval_str(&format!(r#"~(\E x \in {} : x > 0)"#, s_str)).unwrap();
        let rhs = eval_str(&format!(r#"\A x \in {} : ~(x > 0)"#, s_str)).unwrap();
        prop_assert_eq!(lhs, rhs);
    }

    #[test]
    fn prop_forall_and_split(s in small_int_set()) {
        // \A x \in S : (P(x) /\ Q(x)) = (\A x \in S : P(x)) /\ (\A x \in S : Q(x))
        let s_str = format!("{{{}}}", s.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let lhs = eval_str(&format!(r#"\A x \in {} : (x >= 0 /\ x <= 10)"#, s_str)).unwrap();
        let rhs = eval_str(&format!(r#"(\A x \in {} : x >= 0) /\ (\A x \in {} : x <= 10)"#, s_str, s_str)).unwrap();
        prop_assert_eq!(lhs, rhs);
    }

    #[test]
    fn prop_exists_or_split(s in small_int_set()) {
        // \E x \in S : (P(x) \/ Q(x)) = (\E x \in S : P(x)) \/ (\E x \in S : Q(x))
        let s_str = format!("{{{}}}", s.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let lhs = eval_str(&format!(r#"\E x \in {} : (x < 0 \/ x > 5)"#, s_str)).unwrap();
        let rhs = eval_str(&format!(r#"(\E x \in {} : x < 0) \/ (\E x \in {} : x > 5)"#, s_str, s_str)).unwrap();
        prop_assert_eq!(lhs, rhs);
    }
}

// ============================================================================
// Interval property tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn prop_interval_membership(lo in -10i32..10, hi in -10i32..20, x in -15i32..25) {
        // x \in lo..hi <=> lo <= x /\ x <= hi
        if lo <= hi {
            let lhs = eval_str(&format!(r#"{} \in {}..{}"#, x, lo, hi)).unwrap();
            let expected = lo <= x && x <= hi;
            prop_assert_eq!(lhs, Value::Bool(expected));
        }
    }

    #[test]
    fn prop_interval_cardinality(lo in -10i32..10, hi in -10i32..20) {
        // Cardinality(lo..hi) = Max(0, hi - lo + 1)
        let result = eval_str(&format!("Cardinality({}..{})", lo, hi)).unwrap();
        let expected = if hi >= lo { (hi - lo + 1) as i64 } else { 0 };
        prop_assert_eq!(result, Value::SmallInt(expected));
    }

    #[test]
    fn prop_interval_bounds(lo in -10i32..10, hi in -10i32..20) {
        // lo \in lo..hi if lo <= hi
        if lo <= hi {
            let result = eval_str(&format!(r#"{} \in {}..{}"#, lo, lo, hi)).unwrap();
            prop_assert_eq!(result, Value::Bool(true));

            let result = eval_str(&format!(r#"{} \in {}..{}"#, hi, lo, hi)).unwrap();
            prop_assert_eq!(result, Value::Bool(true));
        }
    }

    #[test]
    fn prop_interval_outside_bounds(lo in -10i32..10, hi in -10i32..20) {
        // (lo - 1) \notin lo..hi
        if lo <= hi {
            let result = eval_str(&format!(r#"{} \in {}..{}"#, lo - 1, lo, hi)).unwrap();
            prop_assert_eq!(result, Value::Bool(false));

            let result = eval_str(&format!(r#"{} \in {}..{}"#, hi + 1, lo, hi)).unwrap();
            prop_assert_eq!(result, Value::Bool(false));
        }
    }
}

// ============================================================================
// Function and sequence property tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    #[test]
    fn prop_sequence_len(elems in prop::collection::vec(-10i32..10, 0..8)) {
        // Len(<<e1, e2, ...>>) = n
        let seq_str = format!("<<{}>>", elems.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let result = eval_str(&format!("Len({})", seq_str)).unwrap();
        prop_assert_eq!(result, Value::SmallInt(elems.len() as i64));
    }

    #[test]
    fn prop_sequence_head(elems in prop::collection::vec(-10i32..10, 1..8)) {
        // Head(<<e1, e2, ...>>) = e1
        let seq_str = format!("<<{}>>", elems.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let result = eval_str(&format!("Head({})", seq_str)).unwrap();
        prop_assert_eq!(result, Value::int(elems[0].into()));
    }

    #[test]
    fn prop_sequence_index(elems in prop::collection::vec(-10i32..10, 1..8), idx in 1usize..8) {
        // s[i] returns the i-th element (1-indexed)
        if idx <= elems.len() {
            let seq_str = format!("<<{}>>", elems.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
            let result = eval_str(&format!("{}[{}]", seq_str, idx)).unwrap();
            prop_assert_eq!(result, Value::int(elems[idx - 1].into()));
        }
    }

    #[test]
    fn prop_append_len(elems in prop::collection::vec(-10i32..10, 0..6), v in -10i32..10) {
        // Len(Append(s, v)) = Len(s) + 1
        let seq_str = format!("<<{}>>", elems.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let result = eval_str(&format!("Len(Append({}, {}))", seq_str, v)).unwrap();
        prop_assert_eq!(result, Value::SmallInt((elems.len() + 1) as i64));
    }

    #[test]
    fn prop_domain_function(keys in prop::collection::vec(-5i32..5, 1..5)) {
        // DOMAIN [x \in S |-> e] = S
        let unique_keys: std::collections::HashSet<_> = keys.iter().copied().collect();
        let s_str = format!("{{{}}}", keys.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let result = eval_str(&format!(r#"DOMAIN [x \in {} |-> x * 2]"#, s_str)).unwrap();
        let expected = int_set(&unique_keys.into_iter().collect::<Vec<_>>());
        prop_assert_eq!(result, expected);
    }

    // ============================================================================
    // Cartesian Product (TupleSet) Tests
    // ============================================================================

    #[test]
    fn prop_cartesian_membership(
        s1 in prop::collection::vec(-5i32..5, 1..4),
        s2 in prop::collection::vec(-5i32..5, 1..4),
        a in -5i32..5,
        b in -5i32..5
    ) {
        // <<a, b>> \in S \X T <==> a \in S /\ b \in T
        let set1: std::collections::HashSet<_> = s1.iter().copied().collect();
        let set2: std::collections::HashSet<_> = s2.iter().copied().collect();

        let s1_str = format!("{{{}}}", s1.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let s2_str = format!("{{{}}}", s2.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));

        let result = eval_str(&format!("<<{}, {}>> \\in {} \\X {}", a, b, s1_str, s2_str)).unwrap();
        let expected = set1.contains(&a) && set2.contains(&b);
        prop_assert_eq!(result, Value::Bool(expected));
    }

    #[test]
    fn prop_cartesian_cardinality(
        s1 in prop::collection::vec(-5i32..5, 1..4),
        s2 in prop::collection::vec(-5i32..5, 1..4)
    ) {
        // Cardinality(S \X T) = Cardinality(S) * Cardinality(T)
        let set1: std::collections::HashSet<_> = s1.iter().copied().collect();
        let set2: std::collections::HashSet<_> = s2.iter().copied().collect();

        let s1_str = format!("{{{}}}", s1.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let s2_str = format!("{{{}}}", s2.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));

        let result = eval_str(&format!("Cardinality({} \\X {})", s1_str, s2_str)).unwrap();
        let expected = BigInt::from(set1.len() * set2.len());
        prop_assert_eq!(result, Value::big_int(expected));
    }

    #[test]
    fn prop_cartesian_empty_left(s in prop::collection::vec(-5i32..5, 1..4)) {
        // {} \X S = {}
        let s_str = format!("{{{}}}", s.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let _result = eval_str(&format!("{{}} \\X {}", s_str)).unwrap();
        // The result should be an empty set
        let card = eval_str(&format!("Cardinality({{}} \\X {})", s_str)).unwrap();
        prop_assert_eq!(card, Value::SmallInt(0));
    }

    #[test]
    fn prop_cartesian_empty_right(s in prop::collection::vec(-5i32..5, 1..4)) {
        // S \X {} = {}
        let s_str = format!("{{{}}}", s.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let _result = eval_str(&format!("{} \\X {{}}", s_str)).unwrap();
        // The result should be an empty set
        let card = eval_str(&format!("Cardinality({} \\X {{}})", s_str)).unwrap();
        prop_assert_eq!(card, Value::SmallInt(0));
    }

    #[test]
    fn prop_cartesian_singleton(a in -5i32..5, b in -5i32..5) {
        // {a} \X {b} = {<<a, b>>}
        let _result = eval_str(&format!("{{{}}} \\X {{{}}}", a, b)).unwrap();
        // Check that the tuple is in the result
        let contains = eval_str(&format!("<<{}, {}>> \\in {{{}}} \\X {{{}}}", a, b, a, b)).unwrap();
        prop_assert_eq!(contains, Value::Bool(true));
        // Check cardinality is 1
        let card = eval_str(&format!("Cardinality({{{}}} \\X {{{}}})", a, b)).unwrap();
        prop_assert_eq!(card, Value::SmallInt(1));
    }

    #[test]
    fn prop_cartesian_triple(
        s1 in prop::collection::vec(-3i32..3, 1..3),
        s2 in prop::collection::vec(-3i32..3, 1..3),
        s3 in prop::collection::vec(-3i32..3, 1..3)
    ) {
        // Cardinality(S1 \X S2 \X S3) = Cardinality(S1) * Cardinality(S2) * Cardinality(S3)
        let set1: std::collections::HashSet<_> = s1.iter().copied().collect();
        let set2: std::collections::HashSet<_> = s2.iter().copied().collect();
        let set3: std::collections::HashSet<_> = s3.iter().copied().collect();

        let s1_str = format!("{{{}}}", s1.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let s2_str = format!("{{{}}}", s2.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
        let s3_str = format!("{{{}}}", s3.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));

        let result = eval_str(&format!("Cardinality({} \\X {} \\X {})", s1_str, s2_str, s3_str)).unwrap();
        let expected = BigInt::from(set1.len() * set2.len() * set3.len());
        prop_assert_eq!(result, Value::big_int(expected));
    }

    #[test]
    fn prop_cartesian_with_interval(low in 0i32..3, high in 3i32..6) {
        // S \X (low..high) works with lazy intervals
        let result = eval_str(&format!("Cardinality({{1, 2}} \\X ({}..{}))", low, high)).unwrap();
        let interval_size = if high >= low { (high - low + 1) as usize } else { 0 };
        let expected = BigInt::from(2 * interval_size);
        prop_assert_eq!(result, Value::big_int(expected));
    }
}

// ============================================================================
// State Fingerprinting Property Tests
// ============================================================================

use tla_check::State;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    // --- Determinism: Same state always produces same fingerprint ---

    #[test]
    fn prop_fingerprint_determinism(x in -1000i64..1000, y in -1000i64..1000) {
        // Building the same state twice should give the same fingerprint
        let s1 = State::from_pairs([("x", Value::int(x)), ("y", Value::int(y))]);
        let s2 = State::from_pairs([("x", Value::int(x)), ("y", Value::int(y))]);
        prop_assert_eq!(s1.fingerprint(), s2.fingerprint());
    }

    #[test]
    fn prop_fingerprint_order_independence(x in -100i64..100, y in -100i64..100) {
        // Variable insertion order should not affect fingerprint
        // (OrdMap sorts by key, so insertion order shouldn't matter)
        let s1 = State::from_pairs([("x", Value::int(x)), ("y", Value::int(y))]);
        let s2 = State::from_pairs([("y", Value::int(y)), ("x", Value::int(x))]);
        prop_assert_eq!(s1.fingerprint(), s2.fingerprint());
    }

    // --- Sensitivity: Different states produce different fingerprints ---

    #[test]
    fn prop_fingerprint_value_sensitivity(x in -100i64..100, y in -100i64..100) {
        // Different values should (almost always) produce different fingerprints
        prop_assume!(x != y); // Only test when x != y
        let s1 = State::from_pairs([("x", Value::int(x))]);
        let s2 = State::from_pairs([("x", Value::int(y))]);
        // While collisions are possible, they should be very rare for small integers
        prop_assert_ne!(s1.fingerprint(), s2.fingerprint(),
            "Collision detected for x={}, y={}", x, y);
    }

    #[test]
    fn prop_fingerprint_name_sensitivity(x in -100i64..100) {
        // Different variable names should produce different fingerprints
        let s1 = State::from_pairs([("x", Value::int(x))]);
        let s2 = State::from_pairs([("y", Value::int(x))]);
        prop_assert_ne!(s1.fingerprint(), s2.fingerprint());
    }

    #[test]
    fn prop_fingerprint_multi_var_sensitivity(
        x1 in -50i64..50, y1 in -50i64..50,
        x2 in -50i64..50, y2 in -50i64..50
    ) {
        // States with different (x,y) pairs should have different fingerprints
        prop_assume!(x1 != x2 || y1 != y2);
        let s1 = State::from_pairs([("x", Value::int(x1)), ("y", Value::int(y1))]);
        let s2 = State::from_pairs([("x", Value::int(x2)), ("y", Value::int(y2))]);
        prop_assert_ne!(s1.fingerprint(), s2.fingerprint());
    }

    // --- Type sensitivity: Fingerprints distinguish different types ---

    #[test]
    fn prop_fingerprint_type_sensitivity_bool_vs_int(v in prop::bool::ANY) {
        // Bool and Int should have different fingerprints even for 0/1 vs false/true
        let b = v as i64;
        let s1 = State::from_pairs([("x", Value::Bool(v))]);
        let s2 = State::from_pairs([("x", Value::int(b))]);
        prop_assert_ne!(s1.fingerprint(), s2.fingerprint());
    }

    #[test]
    fn prop_fingerprint_set_vs_seq(a in -10i64..10, b in -10i64..10) {
        // Set {a, b} and Sequence <<a, b>> should have different fingerprints
        let s1 = State::from_pairs([("x", Value::set([Value::int(a), Value::int(b)]))]);
        let s2 = State::from_pairs([("x", Value::seq([Value::int(a), Value::int(b)]))]);
        prop_assert_ne!(s1.fingerprint(), s2.fingerprint());
    }

    // --- Set element order independence ---

    #[test]
    fn prop_fingerprint_set_order_independence(
        vals in prop::collection::vec(-10i64..10, 2..5)
    ) {
        // Set elements should be fingerprinted in canonical order
        // regardless of how the set was constructed
        let set1: im::OrdSet<Value> = vals.iter().map(|&v| Value::int(v)).collect();
        let set2: im::OrdSet<Value> = vals.iter().rev().map(|&v| Value::int(v)).collect();
        let s1 = State::from_pairs([("x", Value::Set(SortedSet::from_ord_set(&set1)))]);
        let s2 = State::from_pairs([("x", Value::Set(SortedSet::from_ord_set(&set2)))]);
        prop_assert_eq!(s1.fingerprint(), s2.fingerprint());
    }

    // --- Complex nested structure tests ---

    #[test]
    fn prop_fingerprint_nested_set(
        vals in prop::collection::vec(-5i64..5, 1..4)
    ) {
        // Nested sets should fingerprint consistently
        let inner_set: im::OrdSet<Value> = vals.iter().map(|&v| Value::int(v)).collect();
        let outer_set: im::OrdSet<Value> = vec![Value::Set(SortedSet::from_ord_set(&inner_set))].into_iter().collect();

        let s1 = State::from_pairs([("x", Value::Set(SortedSet::from_ord_set(&outer_set)))]);
        let s2 = State::from_pairs([("x", Value::Set(SortedSet::from_ord_set(&outer_set)))]);
        prop_assert_eq!(s1.fingerprint(), s2.fingerprint());
    }

    #[test]
    fn prop_fingerprint_func_vs_record(key in "[a-z]{1,3}", val in -10i64..10) {
        // In TLA+, records are functions with string domains.
        // A record [k |-> v] is semantically equivalent to a function with domain {k}.
        // Therefore, they should have the SAME fingerprint.
        use tla_check::FuncValue;
        use tla_check::value::RecordBuilder;
        use im::OrdMap;
        use std::sync::Arc;

        let key_arc: Arc<str> = key.clone().into();

        let mut func_map = OrdMap::new();
        func_map.insert(Value::string(key_arc.clone()), Value::int(val));
        let func = FuncValue::new(
            vec![Value::string(key_arc.clone())].into_iter().collect(),
            func_map,
        );

        let mut rec_builder = RecordBuilder::new();
        rec_builder.insert(key_arc, Value::int(val));

        let s1 = State::from_pairs([("x", Value::Func(func))]);
        let s2 = State::from_pairs([("x", Value::Record(rec_builder.build()))]);

        // Records are functions with string domains - same fingerprint
        prop_assert_eq!(s1.fingerprint(), s2.fingerprint());
    }

    // --- Interval fingerprinting ---

    #[test]
    fn prop_fingerprint_interval_consistency(lo in -10i64..10, hi in -10i64..20) {
        // Same interval should always produce the same fingerprint
        let iv1 = tla_check::IntervalValue::new(lo.into(), hi.into());
        let iv2 = tla_check::IntervalValue::new(lo.into(), hi.into());
        let s1 = State::from_pairs([("x", Value::Interval(iv1))]);
        let s2 = State::from_pairs([("x", Value::Interval(iv2))]);
        prop_assert_eq!(s1.fingerprint(), s2.fingerprint());
    }

    #[test]
    fn prop_fingerprint_interval_vs_set(lo in 0i64..5, hi in 5i64..10) {
        // IntervalValue and equivalent Set should have the SAME fingerprint
        // (extensional equivalence)
        let iv = tla_check::IntervalValue::new(lo.into(), hi.into());
        let set: im::OrdSet<Value> = (lo..=hi).map(Value::int).collect();
        let s1 = State::from_pairs([("x", Value::Interval(iv))]);
        let s2 = State::from_pairs([("x", Value::Set(SortedSet::from_ord_set(&set)))]);
        prop_assert_eq!(s1.fingerprint(), s2.fingerprint());
    }
}

// ============================================================================
// Bags module tests (non-property-based unit tests)
// ============================================================================

/// Helper function to evaluate a TLA+ expression with Bags module
fn eval_bags_str(src: &str) -> Result<Value, String> {
    let module_src = format!(
        "---- MODULE Test ----\nEXTENDS Integers, Bags\n\nOp == {}\n\n====",
        src
    );
    let tree = parse_to_syntax_tree(&module_src);
    let lower_result = lower(FileId(0), &tree);
    let module = match lower_result.module {
        Some(m) => m,
        None => return Err(format!("Parse error: {:?}", lower_result.errors)),
    };

    for unit in &module.units {
        if let tla_core::ast::Unit::Operator(def) = &unit.node {
            if def.name.node == "Op" {
                let ctx = tla_check::eval::EvalCtx::new();
                return tla_check::eval::eval(&ctx, &def.body).map_err(|e| format!("{:?}", e));
            }
        }
    }
    Err("Op not found".to_string())
}

#[test]
fn test_empty_bag() {
    let result = eval_bags_str("EmptyBag").unwrap();
    if let Value::Func(f) = result {
        assert!(f.domain_is_empty());
        assert!(f.domain_is_empty());
    } else {
        panic!("Expected Func value");
    }
}

#[test]
fn test_set_to_bag() {
    let result = eval_bags_str("SetToBag({1, 2, 3})").unwrap();
    if let Value::Func(f) = result {
        assert_eq!(f.domain_len(), 3);
        // Each element should have count 1
        for v in f.mapping_values() {
            assert_eq!(v, &Value::int(1));
        }
    } else {
        panic!("Expected Func value, got {:?}", result);
    }
}

#[test]
fn test_bag_to_set() {
    let result = eval_bags_str("BagToSet(SetToBag({1, 2, 3}))").unwrap();
    assert_eq!(result, int_set(&[1, 2, 3]));
}

#[test]
fn test_is_a_bag() {
    // EmptyBag is a bag
    let result = eval_bags_str("IsABag(EmptyBag)").unwrap();
    assert_eq!(result, Value::Bool(true));

    // SetToBag result is a bag
    let result = eval_bags_str("IsABag(SetToBag({1, 2}))").unwrap();
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn test_bag_cardinality() {
    // Empty bag has cardinality 0
    let result = eval_bags_str("BagCardinality(EmptyBag)").unwrap();
    assert_eq!(result, Value::int(0));

    // SetToBag({1, 2, 3}) has cardinality 3
    let result = eval_bags_str("BagCardinality(SetToBag({1, 2, 3}))").unwrap();
    assert_eq!(result, Value::int(3));
}

#[test]
fn test_copies_in() {
    // CopiesIn of element not in bag is 0
    let result = eval_bags_str("CopiesIn(5, EmptyBag)").unwrap();
    assert_eq!(result, Value::int(0));

    // CopiesIn of element in SetToBag is 1
    let result = eval_bags_str("CopiesIn(2, SetToBag({1, 2, 3}))").unwrap();
    assert_eq!(result, Value::int(1));
}

#[test]
fn test_bag_in() {
    // Element not in empty bag
    let result = eval_bags_str("BagIn(1, EmptyBag)").unwrap();
    assert_eq!(result, Value::Bool(false));

    // Element in SetToBag
    let result = eval_bags_str("BagIn(2, SetToBag({1, 2, 3}))").unwrap();
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn test_bag_union() {
    let result = eval_bags_str("BagUnion({SetToBag({1, 2}), SetToBag({2, 3})})").unwrap();
    if let Value::Func(f) = result {
        assert_eq!(f.mapping_get(&Value::int(1)), Some(&Value::int(1)));
        assert_eq!(f.mapping_get(&Value::int(2)), Some(&Value::int(2)));
        assert_eq!(f.mapping_get(&Value::int(3)), Some(&Value::int(1)));
    } else {
        panic!("Expected Func value, got {:?}", result);
    }
}

#[test]
fn test_bag_of_all_identity() {
    let expected = eval_bags_str("SetToBag({1, 2, 3})").unwrap();
    let result = eval_bags_str("LET F(x) == x IN BagOfAll(F, SetToBag({1, 2, 3}))").unwrap();
    assert_eq!(result, expected);
}

#[test]
fn test_bag_of_all_merges_counts() {
    let result =
        eval_bags_str("BagOfAll(LAMBDA x : 0, [e \\in {1, 2} |-> IF e = 1 THEN 2 ELSE 1])")
            .unwrap();
    if let Value::Func(f) = result {
        assert_eq!(f.domain_len(), 1);
        assert_eq!(f.mapping_get(&Value::int(0)), Some(&Value::int(3)));
    } else {
        panic!("Expected Func value, got {:?}", result);
    }
}

#[test]
fn test_sub_bag_set_to_bag() {
    let result = eval_bags_str(
        "SubBag(SetToBag({1, 2})) = {EmptyBag, SetToBag({1}), SetToBag({2}), SetToBag({1, 2})}",
    )
    .unwrap();
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn test_sub_bag_counts() {
    let result = eval_bags_str("SubBag([e \\in {1} |-> 2])").unwrap();
    if let Value::Set(s) = result {
        assert_eq!(s.len(), 3);
    } else {
        panic!("Expected Set value, got {:?}", result);
    }
}

// ============================================================================
// TLC function operators (:> and @@) tests
// ============================================================================

/// Helper to evaluate with TLC module
fn eval_tlc_str(src: &str) -> Result<Value, String> {
    let module_src = format!(
        "---- MODULE Test ----\nEXTENDS TLC\n\nOp == {}\n\n====",
        src
    );
    let tree = parse_to_syntax_tree(&module_src);
    let lower_result = lower(FileId(0), &tree);
    let module = match lower_result.module {
        Some(m) => m,
        None => return Err(format!("Parse error: {:?}", lower_result.errors)),
    };

    for unit in &module.units {
        if let tla_core::ast::Unit::Operator(def) = &unit.node {
            if def.name.node == "Op" {
                let ctx = tla_check::eval::EvalCtx::new();
                return tla_check::eval::eval(&ctx, &def.body).map_err(|e| format!("{:?}", e));
            }
        }
    }
    Err("Op not found".to_string())
}

#[test]
fn test_make_fcn_single_element() {
    // d :> e creates [d |-> e]
    let result = eval_tlc_str("1 :> \"a\"").unwrap();
    if let Value::Func(f) = result {
        assert_eq!(f.domain_len(), 1);
        assert!(f.domain_contains(&Value::int(1)));
        assert_eq!(
            f.mapping_get(&Value::int(1)),
            Some(&Value::String("a".into()))
        );
    } else {
        panic!("Expected Func value, got {:?}", result);
    }
}

#[test]
fn test_make_fcn_domain_application() {
    // (d :> e)[d] = e
    let result = eval_tlc_str("(1 :> \"a\")[1]").unwrap();
    assert_eq!(result, Value::String("a".into()));
}

#[test]
fn test_make_fcn_domain() {
    // DOMAIN (d :> e) = {d}
    let result = eval_tlc_str("DOMAIN (1 :> \"a\")").unwrap();
    assert_eq!(result, int_set(&[1]));
}

#[test]
fn test_combine_fcn_disjoint() {
    // f @@ g with disjoint domains
    let result = eval_tlc_str("(1 :> \"a\") @@ (2 :> \"b\")").unwrap();
    if let Value::Func(f) = result {
        assert_eq!(f.domain_len(), 2);
        assert_eq!(
            f.mapping_get(&Value::int(1)),
            Some(&Value::String("a".into()))
        );
        assert_eq!(
            f.mapping_get(&Value::int(2)),
            Some(&Value::String("b".into()))
        );
    } else {
        panic!("Expected Func value, got {:?}", result);
    }
}

#[test]
fn test_combine_fcn_priority() {
    // f @@ g with overlapping domains: f takes priority
    let result = eval_tlc_str("(1 :> \"first\") @@ (1 :> \"second\")").unwrap();
    if let Value::Func(f) = result {
        assert_eq!(f.domain_len(), 1);
        assert_eq!(
            f.mapping_get(&Value::int(1)),
            Some(&Value::String("first".into()))
        );
    } else {
        panic!("Expected Func value, got {:?}", result);
    }
}

#[test]
fn test_combine_fcn_chain() {
    // (f @@ g) @@ h
    let result = eval_tlc_str("((1 :> \"a\") @@ (2 :> \"b\")) @@ (3 :> \"c\")").unwrap();
    if let Value::Func(f) = result {
        assert_eq!(f.domain_len(), 3);
        assert_eq!(
            f.mapping_get(&Value::int(1)),
            Some(&Value::String("a".into()))
        );
        assert_eq!(
            f.mapping_get(&Value::int(2)),
            Some(&Value::String("b".into()))
        );
        assert_eq!(
            f.mapping_get(&Value::int(3)),
            Some(&Value::String("c".into()))
        );
    } else {
        panic!("Expected Func value, got {:?}", result);
    }
}

#[test]
fn test_combine_fcn_application() {
    // (f @@ g)[x]
    let result = eval_tlc_str("((1 :> \"a\") @@ (2 :> \"b\"))[2]").unwrap();
    assert_eq!(result, Value::String("b".into()));
}

#[test]
fn test_java_time() {
    // JavaTime returns a positive integer (seconds since epoch)
    let result = eval_tlc_str("JavaTime").unwrap();
    // Use to_bigint() to handle both SmallInt and Int variants
    let n = result.to_bigint().expect("Expected integer value");
    assert!(n > BigInt::from(0), "JavaTime should return positive value");
    // Should be less than 2^31 (due to MSB zeroing)
    assert!(
        n < BigInt::from(1i64 << 31),
        "JavaTime should have MSB zeroed"
    );
}

#[test]
fn test_tlc_get_returns_zero() {
    // TLCGet always returns 0 in model checking mode
    let result = eval_tlc_str("TLCGet(1)").unwrap();
    assert_eq!(result, Value::int(0));
}

#[test]
fn test_tlc_set_returns_true() {
    // TLCSet always returns TRUE
    let result = eval_tlc_str("TLCSet(1, 42)").unwrap();
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn test_random_element() {
    // RandomElement returns an element from the set
    let result = eval_tlc_str("RandomElement({1, 2, 3}) \\in {1, 2, 3}").unwrap();
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn test_tlc_eval_identity() {
    // TLCEval returns its argument unchanged
    let result = eval_tlc_str("TLCEval(42)").unwrap();
    assert_eq!(result, Value::int(42));

    let result = eval_tlc_str("TLCEval({1, 2, 3})").unwrap();
    assert_eq!(result, int_set(&[1, 2, 3]));
}

#[test]
fn test_any_returns_any_set() {
    let result = eval_tlc_str("Any").unwrap();
    assert!(matches!(result, Value::AnySet));
}

#[test]
fn test_any_membership() {
    let result = eval_tlc_str("42 \\in Any").unwrap();
    assert_eq!(result, Value::Bool(true));

    let result = eval_tlc_str("\"hello\" \\in Any").unwrap();
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn test_subseteq_with_infinite_sets() {
    // Enumerable left side, non-enumerable right side.
    let result = eval_tlc_str("{1, 2} \\subseteq Any").unwrap();
    assert_eq!(result, Value::Bool(true));

    let result = eval_tlc_str("{0, 1, 2} \\subseteq Nat").unwrap();
    assert_eq!(result, Value::Bool(true));

    let result = eval_tlc_str("{-1} \\subseteq Nat").unwrap();
    assert_eq!(result, Value::Bool(false));

    // Non-enumerable left side should be an error.
    assert!(eval_tlc_str("Any \\subseteq Any").is_err());
    assert!(eval_tlc_str("Nat \\subseteq Nat").is_err());
}

// ============================================================================
// SequencesExt operator tests
// ============================================================================

fn eval_seqext_str(expr: &str) -> Result<Value, String> {
    let module_src = format!(
        "---- MODULE Test ----\nEXTENDS Sequences, SequencesExt\n\nOp == {}\n\n====",
        expr
    );
    let tree = parse_to_syntax_tree(&module_src);
    let lower_result = lower(FileId(0), &tree);
    let module = match lower_result.module {
        Some(m) => m,
        None => return Err(format!("Parse error: {:?}", lower_result.errors)),
    };

    for unit in &module.units {
        if let tla_core::ast::Unit::Operator(def) = &unit.node {
            if def.name.node == "Op" {
                let ctx = tla_check::eval::EvalCtx::new();
                return tla_check::eval::eval(&ctx, &def.body).map_err(|e| format!("{:?}", e));
            }
        }
    }
    Err("Op not found".to_string())
}

#[test]
fn test_to_set() {
    // ToSet converts sequence to set
    let result = eval_seqext_str("ToSet(<<1, 2, 3, 2, 1>>)").unwrap();
    assert_eq!(result, int_set(&[1, 2, 3]));

    // Empty sequence -> empty set
    let result = eval_seqext_str("ToSet(<<>>)").unwrap();
    assert_eq!(result, int_set(&[]));
}

#[test]
fn test_cons() {
    // Cons prepends element to sequence
    let result = eval_seqext_str("Cons(0, <<1, 2, 3>>)").unwrap();
    assert_eq!(
        result,
        Value::seq([Value::int(0), Value::int(1), Value::int(2), Value::int(3)])
    );

    // Cons to empty sequence
    let result = eval_seqext_str("Cons(42, <<>>)").unwrap();
    assert_eq!(result, Value::seq([Value::int(42)]));
}

#[test]
fn test_contains() {
    // Contains checks if element is in sequence
    let result = eval_seqext_str("Contains(<<1, 2, 3>>, 2)").unwrap();
    assert_eq!(result, Value::Bool(true));

    let result = eval_seqext_str("Contains(<<1, 2, 3>>, 5)").unwrap();
    assert_eq!(result, Value::Bool(false));

    let result = eval_seqext_str("Contains(<<>>, 1)").unwrap();
    assert_eq!(result, Value::Bool(false));
}

#[test]
fn test_is_prefix() {
    // IsPrefix checks prefix relationship
    let result = eval_seqext_str("IsPrefix(<<1, 2>>, <<1, 2, 3>>)").unwrap();
    assert_eq!(result, Value::Bool(true));

    let result = eval_seqext_str("IsPrefix(<<1, 3>>, <<1, 2, 3>>)").unwrap();
    assert_eq!(result, Value::Bool(false));

    // Empty sequence is prefix of any sequence
    let result = eval_seqext_str("IsPrefix(<<>>, <<1, 2, 3>>)").unwrap();
    assert_eq!(result, Value::Bool(true));

    // Sequence is prefix of itself
    let result = eval_seqext_str("IsPrefix(<<1, 2>>, <<1, 2>>)").unwrap();
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn test_is_suffix() {
    // IsSuffix checks suffix relationship
    let result = eval_seqext_str("IsSuffix(<<2, 3>>, <<1, 2, 3>>)").unwrap();
    assert_eq!(result, Value::Bool(true));

    let result = eval_seqext_str("IsSuffix(<<1, 3>>, <<1, 2, 3>>)").unwrap();
    assert_eq!(result, Value::Bool(false));

    // Empty sequence is suffix of any sequence
    let result = eval_seqext_str("IsSuffix(<<>>, <<1, 2, 3>>)").unwrap();
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn test_indices() {
    // Indices returns the set {1..Len(s)}
    let result = eval_seqext_str("Indices(<<1, 2, 3>>)").unwrap();
    // Interval 1..3
    assert_eq!(result, int_set(&[1, 2, 3]));

    let result = eval_seqext_str("Indices(<<>>)").unwrap();
    // Empty interval 1..0
    assert_eq!(result, int_set(&[]));
}

#[test]
fn test_insert_at() {
    // InsertAt inserts element at position
    let result = eval_seqext_str("InsertAt(<<1, 2, 3>>, 2, 99)").unwrap();
    assert_eq!(
        result,
        Value::seq([Value::int(1), Value::int(99), Value::int(2), Value::int(3)])
    );

    // Insert at beginning
    let result = eval_seqext_str("InsertAt(<<1, 2, 3>>, 1, 99)").unwrap();
    assert_eq!(
        result,
        Value::seq([Value::int(99), Value::int(1), Value::int(2), Value::int(3)])
    );

    // Insert at end
    let result = eval_seqext_str("InsertAt(<<1, 2, 3>>, 4, 99)").unwrap();
    assert_eq!(
        result,
        Value::seq([Value::int(1), Value::int(2), Value::int(3), Value::int(99)])
    );
}

#[test]
fn test_remove_at() {
    // RemoveAt removes element at position
    let result = eval_seqext_str("RemoveAt(<<1, 2, 3>>, 2)").unwrap();
    assert_eq!(result, Value::seq([Value::int(1), Value::int(3)]));

    let result = eval_seqext_str("RemoveAt(<<1, 2, 3>>, 1)").unwrap();
    assert_eq!(result, Value::seq([Value::int(2), Value::int(3)]));

    let result = eval_seqext_str("RemoveAt(<<1, 2, 3>>, 3)").unwrap();
    assert_eq!(result, Value::seq([Value::int(1), Value::int(2)]));
}

#[test]
fn test_replace_at() {
    // ReplaceAt replaces element at position
    let result = eval_seqext_str("ReplaceAt(<<1, 2, 3>>, 2, 99)").unwrap();
    assert_eq!(
        result,
        Value::seq([Value::int(1), Value::int(99), Value::int(3)])
    );
}

#[test]
fn test_remove() {
    // Remove removes first occurrence of element
    let result = eval_seqext_str("Remove(<<1, 2, 3, 2>>, 2)").unwrap();
    assert_eq!(
        result,
        Value::seq([Value::int(1), Value::int(3), Value::int(2)])
    );

    // Remove non-existent element - sequence unchanged
    let result = eval_seqext_str("Remove(<<1, 2, 3>>, 99)").unwrap();
    assert_eq!(
        result,
        Value::seq([Value::int(1), Value::int(2), Value::int(3)])
    );
}

#[test]
fn test_flatten_seq() {
    // FlattenSeq flattens nested sequences
    let result = eval_seqext_str("FlattenSeq(<< <<1, 2>>, <<3>>, <<4, 5>> >>)").unwrap();
    assert_eq!(
        result,
        Value::seq([
            Value::int(1),
            Value::int(2),
            Value::int(3),
            Value::int(4),
            Value::int(5)
        ])
    );

    // Empty sequences
    let result = eval_seqext_str("FlattenSeq(<< <<>>, <<1>>, <<>> >>)").unwrap();
    assert_eq!(result, Value::seq([Value::int(1)]));
}

#[test]
fn test_zip() {
    // Zip combines two sequences into sequence of pairs
    let result = eval_seqext_str("Zip(<<1, 2>>, <<3, 4>>)").unwrap();
    assert_eq!(
        result,
        Value::Seq(
            vec![
                Value::Tuple(vec![Value::int(1), Value::int(3)].into()),
                Value::Tuple(vec![Value::int(2), Value::int(4)].into()),
            ]
            .into()
        )
    );

    // Different lengths - takes minimum
    let result = eval_seqext_str("Zip(<<1, 2, 3>>, <<4>>)").unwrap();
    assert_eq!(
        result,
        Value::Seq(vec![Value::Tuple(vec![Value::int(1), Value::int(4)].into())].into())
    );
}

#[test]
fn test_fold_left() {
    // FoldLeft folds from left to right
    let module_src = r#"---- MODULE Test ----
EXTENDS Sequences, SequencesExt

Add(a, b) == a + b
Op == FoldLeft(Add, 0, <<1, 2, 3, 4>>)

===="#;
    let tree = parse_to_syntax_tree(module_src);
    let lower_result = lower(FileId(0), &tree);
    let module = lower_result.module.unwrap();

    let mut ctx = tla_check::eval::EvalCtx::new();
    ctx.load_module(&module);
    let result = ctx.eval_op("Op").unwrap();
    assert_eq!(result, Value::int(10)); // 0+1+2+3+4 = 10
}

#[test]
fn test_fold_right() {
    // FoldRight folds from right to left
    // Using subtraction to verify order matters
    let module_src = r#"---- MODULE Test ----
EXTENDS Sequences, SequencesExt

Sub(a, b) == a - b
Op == FoldRight(Sub, <<1, 2, 3>>, 0)

===="#;
    let tree = parse_to_syntax_tree(module_src);
    let lower_result = lower(FileId(0), &tree);
    let module = lower_result.module.unwrap();

    let mut ctx = tla_check::eval::EvalCtx::new();
    ctx.load_module(&module);
    let result = ctx.eval_op("Op").unwrap();
    // FoldRight(Sub, <<1, 2, 3>>, 0) = 1 - (2 - (3 - 0)) = 1 - (2 - 3) = 1 - (-1) = 2
    assert_eq!(result, Value::int(2));
}

// ============================================================================
// TLCExt operator tests
// ============================================================================

fn eval_tlcext_str(src: &str) -> Result<Value, String> {
    let module_src = format!(
        "---- MODULE Test ----\nEXTENDS TLC, TLCExt\n\nOp == {}\n\n====",
        src
    );
    let tree = parse_to_syntax_tree(&module_src);
    let lower_result = lower(FileId(0), &tree);
    let module = match lower_result.module {
        Some(m) => m,
        None => return Err(format!("Parse error: {:?}", lower_result.errors)),
    };

    for unit in &module.units {
        if let tla_core::ast::Unit::Operator(def) = &unit.node {
            if def.name.node == "Op" {
                let ctx = tla_check::eval::EvalCtx::new();
                return tla_check::eval::eval(&ctx, &def.body).map_err(|e| format!("{:?}", e));
            }
        }
    }
    Err("Op not found".to_string())
}

#[test]
fn test_assert_eq_equal() {
    // AssertEq returns TRUE when values are equal
    let result = eval_tlcext_str("AssertEq(1, 1)").unwrap();
    assert_eq!(result, Value::Bool(true));

    let result = eval_tlcext_str("AssertEq({1, 2}, {2, 1})").unwrap();
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn test_assert_eq_unequal() {
    // AssertEq returns FALSE when values are not equal
    let result = eval_tlcext_str("AssertEq(1, 2)").unwrap();
    assert_eq!(result, Value::Bool(false));
}

#[test]
fn test_tlc_defer() {
    // TLCDefer evaluates and returns the expression
    let result = eval_tlcext_str("TLCDefer(1 + 2)").unwrap();
    assert_eq!(result, Value::int(3));
}

#[test]
fn test_pick_successor() {
    // PickSuccessor returns TRUE (stub behavior)
    let result = eval_tlcext_str("PickSuccessor(TRUE)").unwrap();
    assert_eq!(result, Value::Bool(true));

    let result = eval_tlcext_str("PickSuccessor(FALSE)").unwrap();
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn test_tlc_noop() {
    // TLCNoOp returns its argument unchanged
    let result = eval_tlcext_str("TLCNoOp(42)").unwrap();
    assert_eq!(result, Value::int(42));

    let result = eval_tlcext_str("TLCNoOp({1, 2, 3})").unwrap();
    assert_eq!(result, int_set(&[1, 2, 3]));
}

#[test]
fn test_tlc_cache() {
    // TLCCache returns the expression (stub behavior)
    let result = eval_tlcext_str("TLCCache(1 + 2, 3)").unwrap();
    assert_eq!(result, Value::int(3));
}

#[test]
fn test_tlc_get_or_default() {
    // TLCGetOrDefault returns default value when key not set
    let result = eval_tlcext_str("TLCGetOrDefault(1, 42)").unwrap();
    assert_eq!(result, Value::int(42));

    // Works with different types
    let result = eval_tlcext_str("TLCGetOrDefault(0, TRUE)").unwrap();
    assert_eq!(result, Value::Bool(true));

    let result = eval_tlcext_str("TLCGetOrDefault(99, {1, 2, 3})").unwrap();
    assert_eq!(result, int_set(&[1, 2, 3]));
}

#[test]
fn test_tlc_get_and_set() {
    // TLCGetAndSet returns the default value (old value) since no registers are set
    // TLCGetAndSet(key, Op, val, defaultVal) - returns oldVal before setting
    let module_src = r#"---- MODULE Test ----
EXTENDS TLC, TLCExt

Add(a, b) == a + b
Op == TLCGetAndSet(1, Add, 5, 10)

===="#;
    let tree = parse_to_syntax_tree(module_src);
    let lower_result = lower(FileId(0), &tree);
    let module = lower_result.module.unwrap();

    let mut ctx = tla_check::eval::EvalCtx::new();
    ctx.load_module(&module);
    let result = ctx.eval_op("Op").unwrap();
    // Returns the default value (since no register was set)
    assert_eq!(result, Value::int(10));
}

#[test]
fn test_tlcfp_integer() {
    // TLCFP returns a fingerprint for an integer
    let result = eval_tlcext_str("TLCFP(42)").unwrap();
    // Should be an integer (lower 32 bits of fingerprint)
    // Use is_int() to handle both SmallInt and Int variants
    assert!(result.is_int());

    // Same value should produce same fingerprint
    let result1 = eval_tlcext_str("TLCFP(42)").unwrap();
    let result2 = eval_tlcext_str("TLCFP(42)").unwrap();
    assert_eq!(result1, result2);
}

#[test]
fn test_tlcfp_different_values() {
    // Different values should (very likely) produce different fingerprints
    let result1 = eval_tlcext_str("TLCFP(1)").unwrap();
    let result2 = eval_tlcext_str("TLCFP(2)").unwrap();
    assert_ne!(result1, result2);

    // Different types should produce different fingerprints
    let result_int = eval_tlcext_str("TLCFP(1)").unwrap();
    let result_str = eval_tlcext_str(r#"TLCFP("1")"#).unwrap();
    assert_ne!(result_int, result_str);
}

#[test]
fn test_tlcfp_set() {
    // TLCFP works with sets
    let result = eval_tlcext_str("TLCFP({1, 2, 3})").unwrap();
    // Use is_int() to handle both SmallInt and Int variants
    assert!(result.is_int());

    // Same set (different order) should produce same fingerprint
    let result1 = eval_tlcext_str("TLCFP({1, 2, 3})").unwrap();
    let result2 = eval_tlcext_str("TLCFP({3, 2, 1})").unwrap();
    assert_eq!(result1, result2);
}

// ============================================================================
// FiniteSetsExt operator tests
// ============================================================================

fn eval_fsext_str(src: &str) -> Result<Value, String> {
    let module_src = format!(
        "---- MODULE Test ----\nEXTENDS FiniteSets, FiniteSetsExt\n\nOp == {}\n\n====",
        src
    );
    let tree = parse_to_syntax_tree(&module_src);
    let lower_result = lower(FileId(0), &tree);
    let module = match lower_result.module {
        Some(m) => m,
        None => return Err(format!("Parse error: {:?}", lower_result.errors)),
    };

    for unit in &module.units {
        if let tla_core::ast::Unit::Operator(def) = &unit.node {
            if def.name.node == "Op" {
                let ctx = tla_check::eval::EvalCtx::new();
                return tla_check::eval::eval(&ctx, &def.body).map_err(|e| format!("{:?}", e));
            }
        }
    }
    Err("Op not found".to_string())
}

#[test]
fn test_quantify() {
    // Quantify counts elements satisfying predicate
    let module_src = r#"---- MODULE Test ----
EXTENDS FiniteSets, FiniteSetsExt

IsEven(x) == x % 2 = 0
Op == Quantify({1, 2, 3, 4, 5, 6}, IsEven)

===="#;
    let tree = parse_to_syntax_tree(module_src);
    let lower_result = lower(FileId(0), &tree);
    let module = lower_result.module.unwrap();

    let mut ctx = tla_check::eval::EvalCtx::new();
    ctx.load_module(&module);
    let result = ctx.eval_op("Op").unwrap();
    assert_eq!(result, Value::int(3)); // 2, 4, 6 are even
}

#[test]
fn test_ksubsets() {
    // Ksubsets returns a lazy KSubsetValue (or Set for empty cases)
    let result = eval_fsext_str("Ksubsets({1, 2, 3}, 2)").unwrap();
    // Convert lazy value to set for comparison
    let s = result.to_ord_set().expect("Should be convertible to set");
    assert_eq!(s.len(), 3); // C(3,2) = 3
                            // Check that each element is a 2-element set
    for elem in s.iter() {
        if let Value::Set(inner) = elem {
            assert_eq!(inner.len(), 2);
        } else {
            panic!("Expected Set value in Ksubsets result, got {:?}", elem);
        }
    }

    // k > n returns KSubset with empty enumeration
    let result = eval_fsext_str("Ksubsets({1, 2}, 5)").unwrap();
    let s = result.to_ord_set().expect("Should be convertible to set");
    assert!(s.is_empty());

    // k = 0 returns KSubset containing just empty set
    let result = eval_fsext_str("Ksubsets({1, 2, 3}, 0)").unwrap();
    let s = result.to_ord_set().expect("Should be convertible to set");
    assert_eq!(s.len(), 1);
    // The single element should be the empty set
    let elem = s.iter().next().unwrap();
    assert_eq!(elem, &Value::set(vec![]));
}

#[test]
fn test_sym_diff() {
    // SymDiff is symmetric difference
    let result = eval_fsext_str("SymDiff({1, 2, 3}, {2, 3, 4})").unwrap();
    assert_eq!(result, int_set(&[1, 4])); // {1} ∪ {4}

    // SymDiff of identical sets is empty
    let result = eval_fsext_str("SymDiff({1, 2, 3}, {1, 2, 3})").unwrap();
    assert_eq!(result, int_set(&[]));

    // SymDiff of disjoint sets is their union
    let result = eval_fsext_str("SymDiff({1, 2}, {3, 4})").unwrap();
    assert_eq!(result, int_set(&[1, 2, 3, 4]));
}

#[test]
fn test_flatten() {
    // Flatten is union of set of sets
    let result = eval_fsext_str("Flatten({{1, 2}, {2, 3}, {4}})").unwrap();
    assert_eq!(result, int_set(&[1, 2, 3, 4]));

    // Flatten of empty set is empty
    let result = eval_fsext_str("Flatten({})").unwrap();
    assert_eq!(result, int_set(&[]));
}

#[test]
fn test_choose() {
    // Choose returns an element from the set
    let result = eval_fsext_str("Choose({1, 2, 3}) \\in {1, 2, 3}").unwrap();
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn test_sum() {
    // Sum adds all elements
    let result = eval_fsext_str("Sum({1, 2, 3, 4})").unwrap();
    assert_eq!(result, Value::int(10));

    // Sum of empty set is 0
    let result = eval_fsext_str("Sum({})").unwrap();
    assert_eq!(result, Value::int(0));
}

#[test]
fn test_product() {
    // Product multiplies all elements
    let result = eval_fsext_str("Product({1, 2, 3, 4})").unwrap();
    assert_eq!(result, Value::int(24));

    // Product of empty set is 1
    let result = eval_fsext_str("Product({})").unwrap();
    assert_eq!(result, Value::int(1));
}

// ============================================================================
// SetToSortSeq test
// ============================================================================

#[test]
fn test_set_to_sort_seq() {
    // SetToSortSeq converts set to sorted sequence
    let module_src = r#"---- MODULE Test ----
EXTENDS Sequences, SequencesExt

LessThan(a, b) == a < b
Op == SetToSortSeq({3, 1, 4, 1, 5}, LessThan)

===="#;
    let tree = parse_to_syntax_tree(module_src);
    let lower_result = lower(FileId(0), &tree);
    let module = lower_result.module.unwrap();

    let mut ctx = tla_check::eval::EvalCtx::new();
    ctx.load_module(&module);
    let result = ctx.eval_op("Op").unwrap();
    // Set deduplicates, so {3, 1, 4, 1, 5} = {1, 3, 4, 5}, sorted ascending
    assert_eq!(
        result,
        Value::seq([Value::int(1), Value::int(3), Value::int(4), Value::int(5)])
    );
}

// ============================================================================
// ReduceSet test
// ============================================================================

#[test]
fn test_reduceset() {
    // ReduceSet(op, S, base) - like FoldSet but different arg order
    let module_src = r#"---- MODULE Test ----
EXTENDS FiniteSets, FiniteSetsExt

Add(a, b) == a + b
Op == ReduceSet(Add, {1, 2, 3}, 0)

===="#;
    let tree = parse_to_syntax_tree(module_src);
    let lower_result = lower(FileId(0), &tree);
    let module = lower_result.module.unwrap();

    let mut ctx = tla_check::eval::EvalCtx::new();
    ctx.load_module(&module);
    let result = ctx.eval_op("Op").unwrap();
    assert_eq!(result, Value::int(6));
}

#[test]
fn test_reduceset_empty() {
    // ReduceSet on empty set returns base
    let module_src = r#"---- MODULE Test ----
EXTENDS FiniteSets, FiniteSetsExt

Add(a, b) == a + b
Op == ReduceSet(Add, {}, 100)

===="#;
    let tree = parse_to_syntax_tree(module_src);
    let lower_result = lower(FileId(0), &tree);
    let module = lower_result.module.unwrap();

    let mut ctx = tla_check::eval::EvalCtx::new();
    ctx.load_module(&module);
    let result = ctx.eval_op("Op").unwrap();
    assert_eq!(result, Value::int(100));
}

// ============================================================================
// Mean test
// ============================================================================

#[test]
fn test_mean() {
    // Mean of a set of integers
    let result = eval_fsext_str("Mean({1, 2, 3, 4, 5})").unwrap();
    assert_eq!(result, Value::int(3)); // (1+2+3+4+5)/5 = 15/5 = 3

    // Mean with integer division (floor)
    let result = eval_fsext_str("Mean({1, 2})").unwrap();
    assert_eq!(result, Value::int(1)); // (1+2)/2 = 3/2 = 1 (floor)
}

// ============================================================================
// AssertError test
// ============================================================================

#[test]
fn test_assert_error() {
    // AssertError returns TRUE when condition is true
    let result = eval_tlcext_str(r#"AssertError("should not fail", TRUE)"#).unwrap();
    assert_eq!(result, Value::Bool(true));

    // AssertError returns FALSE when condition is false
    let result = eval_tlcext_str(r#"AssertError("expected failure", FALSE)"#).unwrap();
    assert_eq!(result, Value::Bool(false));
}

// ============================================================================
// STRING module tests
// ============================================================================

fn eval_strings_str(src: &str) -> Result<Value, String> {
    let module_src = format!(
        "---- MODULE Test ----\nEXTENDS Strings\n\nOp == {}\n\n====",
        src
    );
    let tree = parse_to_syntax_tree(&module_src);
    let lower_result = lower(FileId(0), &tree);
    let module = match lower_result.module {
        Some(m) => m,
        None => return Err(format!("Parse error: {:?}", lower_result.errors)),
    };

    let mut ctx = tla_check::eval::EvalCtx::new();
    ctx.load_module(&module);
    ctx.eval_op("Op").map_err(|e| format!("{:?}", e))
}

#[test]
fn test_string_membership() {
    // A string is a member of STRING
    let result = eval_strings_str(r#""hello" \in STRING"#).unwrap();
    assert_eq!(result, Value::Bool(true));

    // An integer is not a member of STRING
    let result = eval_strings_str(r#"42 \in STRING"#).unwrap();
    assert_eq!(result, Value::Bool(false));

    // TRUE is not a member of STRING
    let result = eval_strings_str(r#"TRUE \in STRING"#).unwrap();
    assert_eq!(result, Value::Bool(false));
}

#[test]
fn test_string_not_in() {
    // An integer is not in STRING
    let result = eval_strings_str(r#"42 \notin STRING"#).unwrap();
    assert_eq!(result, Value::Bool(true));

    // A string is in STRING
    let result = eval_strings_str(r#""test" \notin STRING"#).unwrap();
    assert_eq!(result, Value::Bool(false));
}

#[test]
fn test_string_set_equality() {
    // Ensure STRING comparisons don't panic (STRING is non-enumerable).
    let result = eval_strings_str("STRING = {}").unwrap();
    assert_eq!(result, Value::Bool(false));

    let result = eval_strings_str("{} = STRING").unwrap();
    assert_eq!(result, Value::Bool(false));
}

#[test]
fn test_string_set_subseteq() {
    let result = eval_strings_str(r#"{"a", "b"} \subseteq STRING"#).unwrap();
    assert_eq!(result, Value::Bool(true));

    let result = eval_strings_str(r#"{"a", 1} \subseteq STRING"#).unwrap();
    assert_eq!(result, Value::Bool(false));

    // Non-enumerable left side should be an error.
    assert!(eval_strings_str("STRING \\subseteq STRING").is_err());
}

// ============================================================================
// BagsExt advanced operators tests
// ============================================================================

#[test]
fn test_sum_bag() {
    // SumBag([1 |-> 2, 3 |-> 1]) = 1*2 + 3*1 = 5
    let result = eval_bags_str("SumBag([e \\in {1, 3} |-> IF e = 1 THEN 2 ELSE 1])").unwrap();
    assert_eq!(result, Value::int(5));
}

#[test]
fn test_sum_bag_empty() {
    // SumBag(EmptyBag) = 0
    let result = eval_bags_str("SumBag(EmptyBag)").unwrap();
    assert_eq!(result, Value::int(0));
}

#[test]
#[ignore = "Passes in release mode (cargo test --release) but stack overflows in debug due to larger stack frames"]
fn test_product_bag() {
    // ProductBag([2 |-> 3]) = 2^3 = 8
    // Use BagCup to add counts: [2 |-> 1] (+) [2 |-> 1] (+) [2 |-> 1] = [2 |-> 3]
    // Note: Deep nesting of BagCup causes stack overflow in debug builds.
    // Use test_product_bag_with_literal for debug-mode testing.
    let result =
        eval_bags_str("ProductBag(BagCup(BagCup(SetToBag({2}), SetToBag({2})), SetToBag({2})))")
            .unwrap();
    // This is [2 |-> 3], so 2*2*2 = 8
    assert_eq!(result, Value::int(8));
}

#[test]
fn test_product_bag_empty() {
    // ProductBag(EmptyBag) = 1
    let result = eval_bags_str("ProductBag(EmptyBag)").unwrap();
    assert_eq!(result, Value::int(1));
}

// ============================================================================
// Functions module tests
// ============================================================================

/// Helper to evaluate with Functions module
fn eval_functions_str(src: &str) -> Result<Value, String> {
    let module_src = format!(
        "---- MODULE Test ----\nEXTENDS Functions\n\nOp == {}\n\n====",
        src
    );
    let tree = parse_to_syntax_tree(&module_src);
    let lower_result = lower(FileId(0), &tree);
    let module = match lower_result.module {
        Some(m) => m,
        None => return Err(format!("Parse error: {:?}", lower_result.errors)),
    };

    let mut ctx = tla_check::eval::EvalCtx::new();
    ctx.load_module(&module);
    ctx.eval_op("Op").map_err(|e| format!("{:?}", e))
}

#[test]
fn test_restrict_function() {
    // Restrict([x \in {1, 2, 3} |-> x * 2], {1, 2}) should give [x \in {1, 2} |-> x * 2]
    let result = eval_functions_str("Restrict([x \\in {1, 2, 3} |-> x * 2], {1, 2})").unwrap();
    if let Value::Func(f) = result {
        assert_eq!(f.domain_len(), 2);
        assert!(f.domain_contains(&Value::int(1)));
        assert!(f.domain_contains(&Value::int(2)));
        assert!(!f.domain_contains(&Value::int(3)));
        assert_eq!(f.mapping_get(&Value::int(1)), Some(&Value::int(2)));
        assert_eq!(f.mapping_get(&Value::int(2)), Some(&Value::int(4)));
    } else {
        panic!("Expected Func value, got {:?}", result);
    }
}

#[test]
fn test_restrict_empty() {
    // Restrict(f, {}) = <<>> (empty function)
    let result = eval_functions_str("Restrict([x \\in {1, 2} |-> x], {})").unwrap();
    if let Value::Func(f) = result {
        assert_eq!(f.domain_len(), 0);
    } else {
        panic!("Expected Func value, got {:?}", result);
    }
}

#[test]
fn test_is_injective_true() {
    // [1 |-> "a", 2 |-> "b"] is injective (different inputs map to different outputs)
    let result =
        eval_functions_str(r#"IsInjective([x \in {1, 2} |-> IF x = 1 THEN "a" ELSE "b"])"#)
            .unwrap();
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn test_is_injective_false() {
    // [1 |-> "a", 2 |-> "a"] is NOT injective (different inputs map to same output)
    let result = eval_functions_str(r#"IsInjective([x \in {1, 2} |-> "a"])"#).unwrap();
    assert_eq!(result, Value::Bool(false));
}

#[test]
fn test_is_surjective_true() {
    // f = [x \in {1, 2} |-> x] is surjective from {1, 2} to {1, 2}
    let result = eval_functions_str("IsSurjective([x \\in {1, 2} |-> x], {1, 2}, {1, 2})").unwrap();
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn test_is_surjective_false() {
    // f = [x \in {1, 2} |-> 1] is NOT surjective from {1, 2} to {1, 2} (2 never hit)
    let result = eval_functions_str("IsSurjective([x \\in {1, 2} |-> 1], {1, 2}, {1, 2})").unwrap();
    assert_eq!(result, Value::Bool(false));
}

#[test]
fn test_is_bijection_true() {
    // Identity function is a bijection
    let result = eval_functions_str("IsBijection([x \\in {1, 2} |-> x], {1, 2}, {1, 2})").unwrap();
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn test_is_bijection_false_not_injective() {
    // Constant function is not a bijection (not injective)
    let result = eval_functions_str("IsBijection([x \\in {1, 2} |-> 1], {1, 2}, {1, 2})").unwrap();
    assert_eq!(result, Value::Bool(false));
}

#[test]
fn test_inverse_identity() {
    // Inverse of identity function is identity
    let result = eval_functions_str("Inverse([x \\in {1, 2} |-> x], {1, 2}, {1, 2})").unwrap();
    if let Value::Func(f) = result {
        assert_eq!(f.domain_len(), 2);
        assert_eq!(f.mapping_get(&Value::int(1)), Some(&Value::int(1)));
        assert_eq!(f.mapping_get(&Value::int(2)), Some(&Value::int(2)));
    } else {
        panic!("Expected Func value, got {:?}", result);
    }
}

#[test]
fn test_inverse_swap() {
    // Inverse of [1 |-> 2, 2 |-> 1] should be [2 |-> 1, 1 |-> 2]
    let result =
        eval_functions_str("Inverse([x \\in {1, 2} |-> IF x = 1 THEN 2 ELSE 1], {1, 2}, {1, 2})")
            .unwrap();
    if let Value::Func(f) = result {
        assert_eq!(f.mapping_get(&Value::int(1)), Some(&Value::int(2)));
        assert_eq!(f.mapping_get(&Value::int(2)), Some(&Value::int(1)));
    } else {
        panic!("Expected Func value, got {:?}", result);
    }
}

#[test]
fn test_range_function() {
    // Range([x \in {1, 2} |-> x + 10]) = {11, 12}
    let result = eval_functions_str("Range([x \\in {1, 2} |-> x + 10])").unwrap();
    assert_eq!(result, int_set(&[11, 12]));
}

// New Functions module operators: RestrictDomain, RestrictValues, IsRestriction, Pointwise, AntiFunction

#[test]
fn test_restrict_domain() {
    // RestrictDomain(f, P) restricts function to domain elements satisfying P
    let module_src = r#"---- MODULE Test ----
EXTENDS Functions

f == [x \in {1, 2, 3, 4} |-> x * 2]
IsEven(x) == x % 2 = 0
Op == RestrictDomain(f, LAMBDA x: IsEven(x))

===="#;
    let tree = parse_to_syntax_tree(module_src);
    let lower_result = lower(FileId(0), &tree);
    let module = lower_result.module.unwrap();
    let mut ctx = tla_check::eval::EvalCtx::new();
    ctx.load_module(&module);
    let result = ctx.eval_op("Op").unwrap();
    if let Value::Func(f) = result {
        assert_eq!(f.domain_len(), 2);
        assert!(f.domain_contains(&Value::int(2)));
        assert!(f.domain_contains(&Value::int(4)));
        assert!(!f.domain_contains(&Value::int(1)));
        assert!(!f.domain_contains(&Value::int(3)));
    } else {
        panic!("Expected Func, got {:?}", result);
    }
}

#[test]
fn test_restrict_values() {
    // RestrictValues(f, P) restricts function to keys whose values satisfy P
    let module_src = r#"---- MODULE Test ----
EXTENDS Functions

f == [x \in {1, 2, 3, 4} |-> x * 2]
IsGt5(y) == y > 5
Op == RestrictValues(f, LAMBDA y: IsGt5(y))

===="#;
    let tree = parse_to_syntax_tree(module_src);
    let lower_result = lower(FileId(0), &tree);
    let module = lower_result.module.unwrap();
    let mut ctx = tla_check::eval::EvalCtx::new();
    ctx.load_module(&module);
    let result = ctx.eval_op("Op").unwrap();
    if let Value::Func(f) = result {
        // f[3] = 6, f[4] = 8 are > 5
        assert_eq!(f.domain_len(), 2);
        assert!(f.domain_contains(&Value::int(3)));
        assert!(f.domain_contains(&Value::int(4)));
    } else {
        panic!("Expected Func, got {:?}", result);
    }
}

#[test]
fn test_is_restriction_true() {
    // IsRestriction([x \in {1} |-> x], [x \in {1, 2} |-> x]) = TRUE
    let result =
        eval_functions_str("IsRestriction([x \\in {1} |-> x], [x \\in {1, 2} |-> x])").unwrap();
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn test_is_restriction_false_domain() {
    // IsRestriction fails if domain not a subset
    let result =
        eval_functions_str("IsRestriction([x \\in {1, 3} |-> x], [x \\in {1, 2} |-> x])").unwrap();
    assert_eq!(result, Value::Bool(false));
}

#[test]
fn test_is_restriction_false_values() {
    // IsRestriction fails if values differ
    let result =
        eval_functions_str("IsRestriction([x \\in {1} |-> x + 1], [x \\in {1, 2} |-> x])").unwrap();
    assert_eq!(result, Value::Bool(false));
}

#[test]
fn test_pointwise() {
    // Pointwise(Op, f, g) combines functions pointwise
    let module_src = r#"---- MODULE Test ----
EXTENDS Functions

Add(a, b) == a + b
f == [x \in {1, 2} |-> x]
g == [x \in {1, 2} |-> x * 10]
Op == Pointwise(Add, f, g)

===="#;
    let tree = parse_to_syntax_tree(module_src);
    let lower_result = lower(FileId(0), &tree);
    let module = lower_result.module.unwrap();
    let mut ctx = tla_check::eval::EvalCtx::new();
    ctx.load_module(&module);
    let result = ctx.eval_op("Op").unwrap();
    if let Value::Func(f) = result {
        // f[1] = 1 + 10 = 11, f[2] = 2 + 20 = 22
        assert_eq!(f.mapping_get(&Value::int(1)), Some(&Value::int(11)));
        assert_eq!(f.mapping_get(&Value::int(2)), Some(&Value::int(22)));
    } else {
        panic!("Expected Func, got {:?}", result);
    }
}

#[test]
fn test_anti_function() {
    // AntiFunction reverses key-value pairs
    let result = eval_functions_str("AntiFunction([x \\in {1, 2} |-> x * 10])").unwrap();
    if let Value::Func(f) = result {
        // {10 -> 1, 20 -> 2}
        assert_eq!(f.mapping_get(&Value::int(10)), Some(&Value::int(1)));
        assert_eq!(f.mapping_get(&Value::int(20)), Some(&Value::int(2)));
    } else {
        panic!("Expected Func, got {:?}", result);
    }
}

// ============================================================================
// FiniteSetsExt - MapThenSumSet, Choices, ChooseUnique
// ============================================================================

/// Helper to evaluate with FiniteSetsExt module
fn eval_finitesetsext_str(src: &str) -> Result<Value, String> {
    let module_src = format!(
        "---- MODULE Test ----\nEXTENDS FiniteSetsExt\n\nOp == {}\n\n====",
        src
    );
    let tree = parse_to_syntax_tree(&module_src);
    let lower_result = lower(FileId(0), &tree);
    let module = match lower_result.module {
        Some(m) => m,
        None => return Err(format!("Parse error: {:?}", lower_result.errors)),
    };

    let mut ctx = tla_check::eval::EvalCtx::new();
    ctx.load_module(&module);
    ctx.eval_op("Op").map_err(|e| format!("{:?}", e))
}

#[test]
fn test_map_then_sum_set() {
    // MapThenSumSet(Op, S) maps unary operator over S and sums results
    let module_src = r#"---- MODULE Test ----
EXTENDS FiniteSetsExt

Double(x) == x * 2
Op == MapThenSumSet(Double, {1, 2, 3})

===="#;
    let tree = parse_to_syntax_tree(module_src);
    let lower_result = lower(FileId(0), &tree);
    let module = lower_result.module.unwrap();
    let mut ctx = tla_check::eval::EvalCtx::new();
    ctx.load_module(&module);
    let result = ctx.eval_op("Op").unwrap();
    // Double(1) + Double(2) + Double(3) = 2 + 4 + 6 = 12
    assert_eq!(result, Value::int(12));
}

#[test]
fn test_map_then_sum_set_empty() {
    // MapThenSumSet on empty set = 0
    let module_src = r#"---- MODULE Test ----
EXTENDS FiniteSetsExt

Id(x) == x
Op == MapThenSumSet(Id, {})

===="#;
    let tree = parse_to_syntax_tree(module_src);
    let lower_result = lower(FileId(0), &tree);
    let module = lower_result.module.unwrap();
    let mut ctx = tla_check::eval::EvalCtx::new();
    ctx.load_module(&module);
    let result = ctx.eval_op("Op").unwrap();
    assert_eq!(result, Value::int(0));
}

#[test]
fn test_choices_basic() {
    // Choices({{1}, {2}}) has 1 choice function: [S1 -> 1, S2 -> 2]
    let result = eval_finitesetsext_str("Choices({{1}, {2}})").unwrap();
    if let Value::Set(s) = result {
        assert_eq!(s.len(), 1);
    } else {
        panic!("Expected Set, got {:?}", result);
    }
}

#[test]
fn test_choices_multiple() {
    // Choices({{1, 2}, {3}}) has 2 choice functions
    let result = eval_finitesetsext_str("Choices({{1, 2}, {3}})").unwrap();
    if let Value::Set(s) = result {
        assert_eq!(s.len(), 2);
    } else {
        panic!("Expected Set, got {:?}", result);
    }
}

#[test]
fn test_choices_empty_set() {
    // Choices({}) = one empty function
    let result = eval_finitesetsext_str("Choices({})").unwrap();
    if let Value::Set(s) = result {
        assert_eq!(s.len(), 1);
    } else {
        panic!("Expected Set, got {:?}", result);
    }
}

#[test]
fn test_choices_contains_empty() {
    // Choices({{}, {1}}) = {} because one set is empty
    let result = eval_finitesetsext_str("Choices({{}, {1}})").unwrap();
    if let Value::Set(s) = result {
        assert_eq!(s.len(), 0);
    } else {
        panic!("Expected Set, got {:?}", result);
    }
}

#[test]
fn test_choose_unique() {
    // ChooseUnique({1, 2, 3}, LAMBDA x: x = 2) = 2
    let result = eval_finitesetsext_str("ChooseUnique({1, 2, 3}, LAMBDA x: x = 2)").unwrap();
    assert_eq!(result, Value::int(2));
}

// ============================================================================
// SequencesExt - Snoc, IsStrictPrefix, IsStrictSuffix, Prefixes, Suffixes
// ============================================================================

#[test]
fn test_snoc() {
    // Snoc(<<1, 2>>, 3) = <<1, 2, 3>>
    let result = eval_str("Snoc(<<1, 2>>, 3)").unwrap();
    assert_eq!(
        result,
        Value::Seq(vec![Value::int(1), Value::int(2), Value::int(3)].into())
    );
}

#[test]
fn test_snoc_empty() {
    // Snoc(<<>>, 1) = <<1>>
    let result = eval_str("Snoc(<<>>, 1)").unwrap();
    assert_eq!(result, Value::Seq(vec![Value::int(1)].into()));
}

#[test]
fn test_is_strict_prefix_true() {
    // <<1, 2>> is strict prefix of <<1, 2, 3>>
    let result = eval_str("IsStrictPrefix(<<1, 2>>, <<1, 2, 3>>)").unwrap();
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn test_is_strict_prefix_false_equal() {
    // <<1, 2>> is NOT strict prefix of <<1, 2>> (equal sequences)
    let result = eval_str("IsStrictPrefix(<<1, 2>>, <<1, 2>>)").unwrap();
    assert_eq!(result, Value::Bool(false));
}

#[test]
fn test_is_strict_prefix_false_not_prefix() {
    // <<2, 1>> is NOT strict prefix of <<1, 2, 3>>
    let result = eval_str("IsStrictPrefix(<<2, 1>>, <<1, 2, 3>>)").unwrap();
    assert_eq!(result, Value::Bool(false));
}

#[test]
fn test_is_strict_suffix_true() {
    // <<2, 3>> is strict suffix of <<1, 2, 3>>
    let result = eval_str("IsStrictSuffix(<<2, 3>>, <<1, 2, 3>>)").unwrap();
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn test_is_strict_suffix_false_equal() {
    // <<1, 2>> is NOT strict suffix of <<1, 2>>
    let result = eval_str("IsStrictSuffix(<<1, 2>>, <<1, 2>>)").unwrap();
    assert_eq!(result, Value::Bool(false));
}

#[test]
fn test_prefixes() {
    // Prefixes(<<1, 2>>) = { <<>>, <<1>>, <<1, 2>> }
    let result = eval_str("Prefixes(<<1, 2>>)").unwrap();
    if let Value::Set(set) = result {
        assert_eq!(set.len(), 3);
        assert!(set.contains(&Value::Seq(vec![].into())));
        assert!(set.contains(&Value::Seq(vec![Value::int(1)].into())));
        assert!(set.contains(&Value::Seq(vec![Value::int(1), Value::int(2)].into())));
    } else {
        panic!("Expected Set, got {:?}", result);
    }
}

#[test]
fn test_prefixes_empty() {
    // Prefixes(<<>>) = { <<>> }
    let result = eval_str("Prefixes(<<>>)").unwrap();
    if let Value::Set(set) = result {
        assert_eq!(set.len(), 1);
        assert!(set.contains(&Value::Seq(vec![].into())));
    } else {
        panic!("Expected Set, got {:?}", result);
    }
}

#[test]
fn test_suffixes() {
    // Suffixes(<<1, 2>>) = { <<>>, <<2>>, <<1, 2>> }
    let result = eval_str("Suffixes(<<1, 2>>)").unwrap();
    if let Value::Set(set) = result {
        assert_eq!(set.len(), 3);
        assert!(set.contains(&Value::Seq(vec![].into())));
        assert!(set.contains(&Value::Seq(vec![Value::int(2)].into())));
        assert!(set.contains(&Value::Seq(vec![Value::int(1), Value::int(2)].into())));
    } else {
        panic!("Expected Set, got {:?}", result);
    }
}

#[test]
fn test_suffixes_empty() {
    // Suffixes(<<>>) = { <<>> }
    let result = eval_str("Suffixes(<<>>)").unwrap();
    if let Value::Set(set) = result {
        assert_eq!(set.len(), 1);
        assert!(set.contains(&Value::Seq(vec![].into())));
    } else {
        panic!("Expected Set, got {:?}", result);
    }
}

#[test]
fn test_tuple_of() {
    // TupleOf({1, 2}, 2) = { <<1,1>>, <<1,2>>, <<2,1>>, <<2,2>> }
    let result = eval_str("TupleOf({1, 2}, 2)").unwrap();
    if let Value::Set(set) = result {
        assert_eq!(set.len(), 4);
    } else {
        panic!("Expected Set, got {:?}", result);
    }
}

#[test]
fn test_tuple_of_zero() {
    // TupleOf({1, 2}, 0) = { <<>> }
    let result = eval_str("TupleOf({1, 2}, 0)").unwrap();
    if let Value::Set(set) = result {
        assert_eq!(set.len(), 1);
        assert!(set.contains(&Value::Seq(vec![].into())));
    } else {
        panic!("Expected Set, got {:?}", result);
    }
}

#[test]
fn test_bounded_seq() {
    // BoundedSeq({1}, 2) = { <<>>, <<1>>, <<1, 1>> }
    let result = eval_str("BoundedSeq({1}, 2)").unwrap();
    if let Value::Set(set) = result {
        assert_eq!(set.len(), 3);
        assert!(set.contains(&Value::Seq(vec![].into())));
        assert!(set.contains(&Value::Seq(vec![Value::int(1)].into())));
        assert!(set.contains(&Value::Seq(vec![Value::int(1), Value::int(1)].into())));
    } else {
        panic!("Expected Set, got {:?}", result);
    }
}

#[test]
fn test_seq_of_alias() {
    // SeqOf is an alias for BoundedSeq
    let result = eval_str("SeqOf({1}, 2)").unwrap();
    if let Value::Set(set) = result {
        assert_eq!(set.len(), 3);
    } else {
        panic!("Expected Set, got {:?}", result);
    }
}

// ============================================================================
// Functions - Injection, Surjection, Bijection, Exists*
// ============================================================================

#[test]
fn test_injection_set() {
    // Injection({1, 2}, {a, b, c}) should have 6 elements (3 * 2 = 6 injective functions)
    let result = eval_functions_str(r#"Cardinality(Injection({1, 2}, {"a", "b", "c"}))"#).unwrap();
    assert_eq!(result, Value::int(6));
}

#[test]
fn test_injection_too_large_source() {
    // Injection({1, 2, 3}, {a, b}) should be empty (no injection possible)
    let result = eval_functions_str(r#"Injection({1, 2, 3}, {"a", "b"})"#).unwrap();
    assert_eq!(result, Value::Set(SortedSet::new()));
}

#[test]
fn test_bijection_set() {
    // Bijection({1, 2}, {a, b}) should have 2 elements (2! = 2)
    let result = eval_functions_str(r#"Cardinality(Bijection({1, 2}, {"a", "b"}))"#).unwrap();
    assert_eq!(result, Value::int(2));
}

#[test]
fn test_bijection_different_cardinality() {
    // Bijection({1, 2}, {a, b, c}) should be empty (cardinalities differ)
    let result = eval_functions_str(r#"Bijection({1, 2}, {"a", "b", "c"})"#).unwrap();
    assert_eq!(result, Value::Set(SortedSet::new()));
}

#[test]
fn test_surjection_set() {
    // Surjection({1, 2}, {a}) should have 1 element (both map to "a")
    let result = eval_functions_str(r#"Cardinality(Surjection({1, 2}, {"a"}))"#).unwrap();
    assert_eq!(result, Value::int(1));
}

#[test]
fn test_surjection_too_small_source() {
    // Surjection({1}, {a, b}) should be empty (no surjection possible)
    let result = eval_functions_str(r#"Surjection({1}, {"a", "b"})"#).unwrap();
    assert_eq!(result, Value::Set(SortedSet::new()));
}

#[test]
fn test_exists_injection_true() {
    // ExistsInjection({1, 2}, {a, b, c}) = TRUE
    let result = eval_functions_str(r#"ExistsInjection({1, 2}, {"a", "b", "c"})"#).unwrap();
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn test_exists_injection_false() {
    // ExistsInjection({1, 2, 3}, {a, b}) = FALSE
    let result = eval_functions_str(r#"ExistsInjection({1, 2, 3}, {"a", "b"})"#).unwrap();
    assert_eq!(result, Value::Bool(false));
}

#[test]
fn test_exists_surjection_true() {
    // ExistsSurjection({1, 2, 3}, {a, b}) = TRUE
    let result = eval_functions_str(r#"ExistsSurjection({1, 2, 3}, {"a", "b"})"#).unwrap();
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn test_exists_surjection_false() {
    // ExistsSurjection({1}, {a, b}) = FALSE
    let result = eval_functions_str(r#"ExistsSurjection({1}, {"a", "b"})"#).unwrap();
    assert_eq!(result, Value::Bool(false));
}

#[test]
fn test_exists_bijection_true() {
    // ExistsBijection({1, 2}, {a, b}) = TRUE
    let result = eval_functions_str(r#"ExistsBijection({1, 2}, {"a", "b"})"#).unwrap();
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn test_exists_bijection_false() {
    // ExistsBijection({1, 2, 3}, {a, b}) = FALSE
    let result = eval_functions_str(r#"ExistsBijection({1, 2, 3}, {"a", "b"})"#).unwrap();
    assert_eq!(result, Value::Bool(false));
}

// ============================================================================
// TransitiveClosure module tests
// ============================================================================

#[test]
fn test_transitive_closure_simple_chain() {
    // A chain 1->2->3 should produce closure with 1->2, 2->3, 1->3
    let result = eval_str(r#"TransitiveClosure({<<1, 2>>, <<2, 3>>})"#).unwrap();
    let expected = eval_str(r#"{<<1, 2>>, <<2, 3>>, <<1, 3>>}"#).unwrap();
    assert_eq!(result, expected);
}

#[test]
fn test_transitive_closure_simple_two_edges() {
    // Two edges 1->2, 1->3 (no transitivity possible)
    let result = eval_str(r#"TransitiveClosure({<<1, 2>>, <<1, 3>>})"#).unwrap();
    let expected = eval_str(r#"{<<1, 2>>, <<1, 3>>}"#).unwrap();
    assert_eq!(result, expected);
}

#[test]
fn test_transitive_closure_cycle() {
    // A cycle 1->2->1 should produce full connectivity
    let result = eval_str(r#"TransitiveClosure({<<1, 2>>, <<2, 1>>})"#).unwrap();
    let expected = eval_str(r#"{<<1, 2>>, <<2, 1>>, <<1, 1>>, <<2, 2>>}"#).unwrap();
    assert_eq!(result, expected);
}

#[test]
fn test_transitive_closure_empty() {
    // Empty relation should produce empty closure
    let result = eval_str(r#"TransitiveClosure({})"#).unwrap();
    assert_eq!(result, Value::Set(SortedSet::new()));
}

#[test]
fn test_transitive_closure_single_edge() {
    // Single edge should stay the same
    let result = eval_str(r#"TransitiveClosure({<<1, 2>>})"#).unwrap();
    let expected = eval_str(r#"{<<1, 2>>}"#).unwrap();
    assert_eq!(result, expected);
}

#[test]
fn test_transitive_closure_longer_chain() {
    // Chain 1->2->3->4 should produce all reachable pairs
    let result = eval_str(r#"TransitiveClosure({<<1, 2>>, <<2, 3>>, <<3, 4>>})"#).unwrap();
    let expected =
        eval_str(r#"{<<1, 2>>, <<1, 3>>, <<1, 4>>, <<2, 3>>, <<2, 4>>, <<3, 4>>}"#).unwrap();
    assert_eq!(result, expected);
}

#[test]
fn test_warshall_alias() {
    // Warshall should be an alias for TransitiveClosure
    let tc = eval_str(r#"TransitiveClosure({<<1, 2>>, <<2, 3>>})"#).unwrap();
    let warshall = eval_str(r#"Warshall({<<1, 2>>, <<2, 3>>})"#).unwrap();
    assert_eq!(tc, warshall);
}

#[test]
fn test_connected_nodes_simple() {
    // ConnectedNodes extracts all nodes from the relation
    let result = eval_str(r#"ConnectedNodes({<<1, 2>>, <<2, 3>>})"#).unwrap();
    let expected = eval_str(r#"{1, 2, 3}"#).unwrap();
    assert_eq!(result, expected);
}

#[test]
fn test_connected_nodes_empty() {
    // Empty relation should have no connected nodes
    let result = eval_str(r#"ConnectedNodes({})"#).unwrap();
    assert_eq!(result, Value::Set(SortedSet::new()));
}

#[test]
fn test_connected_nodes_with_strings() {
    // ConnectedNodes should work with string values
    let result = eval_str(r#"ConnectedNodes({<<"a", "b">>, <<"b", "c">>})"#).unwrap();
    let expected = eval_str(r#"{"a", "b", "c"}"#).unwrap();
    assert_eq!(result, expected);
}

#[test]
fn test_reflexive_transitive_closure_simple() {
    // ReflexiveTransitiveClosure adds reflexive edges
    let result = eval_str(r#"ReflexiveTransitiveClosure({<<1, 2>>}, {1, 2})"#).unwrap();
    let expected = eval_str(r#"{<<1, 1>>, <<1, 2>>, <<2, 2>>}"#).unwrap();
    assert_eq!(result, expected);
}

#[test]
fn test_reflexive_transitive_closure_chain() {
    // Chain with reflexive closure
    let result =
        eval_str(r#"ReflexiveTransitiveClosure({<<1, 2>>, <<2, 3>>}, {1, 2, 3})"#).unwrap();
    let expected =
        eval_str(r#"{<<1, 1>>, <<1, 2>>, <<1, 3>>, <<2, 2>>, <<2, 3>>, <<3, 3>>}"#).unwrap();
    assert_eq!(result, expected);
}

#[test]
fn test_reflexive_transitive_closure_empty_relation() {
    // Empty relation with domain should only have reflexive edges
    let result = eval_str(r#"ReflexiveTransitiveClosure({}, {1, 2})"#).unwrap();
    let expected = eval_str(r#"{<<1, 1>>, <<2, 2>>}"#).unwrap();
    assert_eq!(result, expected);
}

// ============================================================================
// Randomization module tests
// ============================================================================

#[test]
fn test_random_subset_basic() {
    // RandomSubset(2, {1, 2, 3, 4}) returns a 2-element subset
    let result = eval_str(r#"RandomSubset(2, {1, 2, 3, 4})"#).unwrap();
    let set = result.as_set().expect("Expected set");
    assert_eq!(set.len(), 2);
}

#[test]
fn test_random_subset_full_set() {
    // RandomSubset(4, {1, 2, 3, 4}) returns the full set
    let result = eval_str(r#"RandomSubset(4, {1, 2, 3, 4})"#).unwrap();
    let expected = eval_str(r#"{1, 2, 3, 4}"#).unwrap();
    assert_eq!(result, expected);
}

#[test]
fn test_random_subset_empty() {
    // RandomSubset(0, {1, 2, 3}) returns empty set
    let result = eval_str(r#"RandomSubset(0, {1, 2, 3})"#).unwrap();
    assert_eq!(result, Value::Set(SortedSet::new()));
}

#[test]
fn test_random_subset_singleton() {
    // RandomSubset(1, {42}) returns {42}
    let result = eval_str(r#"RandomSubset(1, {42})"#).unwrap();
    let expected = eval_str(r#"{42}"#).unwrap();
    assert_eq!(result, expected);
}

#[test]
fn test_random_set_of_subsets_basic() {
    // RandomSetOfSubsets(3, 2, {1, 2, 3, 4}) returns a set of subsets
    let result = eval_str(r#"RandomSetOfSubsets(3, 2, {1, 2, 3, 4})"#).unwrap();
    let set = result.as_set().expect("Expected set of sets");
    // Should have at most 3 subsets (may have fewer due to dedup)
    assert!(set.len() <= 3);
    // Each should be a subset of {1, 2, 3, 4}
    for subset in set.iter() {
        let sub = subset.as_set().expect("Expected inner set");
        assert!(sub.len() <= 4);
    }
}

#[test]
fn test_random_set_of_subsets_empty_base() {
    // RandomSetOfSubsets with empty set returns {{}}
    let result = eval_str(r#"RandomSetOfSubsets(3, 0, {})"#).unwrap();
    let set = result.as_set().expect("Expected set");
    // Should contain only the empty set
    assert_eq!(set.len(), 1);
}

#[test]
fn test_random_subset_set_basic() {
    // RandomSubsetSet(3, "0.5", {1, 2, 3, 4}) returns a set of subsets
    let result = eval_str(r#"RandomSubsetSet(3, "0.5", {1, 2, 3, 4})"#).unwrap();
    let set = result.as_set().expect("Expected set of sets");
    assert!(set.len() <= 3);
}

#[test]
fn test_random_subset_set_zero_prob() {
    // RandomSubsetSet with 0.0 probability returns set of empty sets
    let result = eval_str(r#"RandomSubsetSet(2, "0.0", {1, 2, 3})"#).unwrap();
    let set = result.as_set().expect("Expected set");
    // Each subset should be empty
    for subset in set.iter() {
        let sub = subset.as_set().expect("Expected inner set");
        assert!(sub.is_empty());
    }
}

// ============================================================================
// Json module tests
// ============================================================================

#[test]
fn test_to_json_boolean() {
    let result = eval_str(r#"ToJson(TRUE)"#).unwrap();
    assert_eq!(result.as_string().unwrap(), "true");

    let result = eval_str(r#"ToJson(FALSE)"#).unwrap();
    assert_eq!(result.as_string().unwrap(), "false");
}

#[test]
fn test_to_json_integer() {
    let result = eval_str(r#"ToJson(42)"#).unwrap();
    assert_eq!(result.as_string().unwrap(), "42");

    let result = eval_str(r#"ToJson(-123)"#).unwrap();
    assert_eq!(result.as_string().unwrap(), "-123");
}

#[test]
fn test_to_json_string() {
    let result = eval_str(r#"ToJson("hello")"#).unwrap();
    assert_eq!(result.as_string().unwrap(), "\"hello\"");
}

#[test]
fn test_to_json_sequence() {
    let result = eval_str(r#"ToJson(<<1, 2, 3>>)"#).unwrap();
    assert_eq!(result.as_string().unwrap(), "[1,2,3]");
}

#[test]
fn test_to_json_set() {
    let result = eval_str(r#"ToJson({1, 2, 3})"#).unwrap();
    // Sets are converted to arrays
    let json = result.as_string().unwrap();
    assert!(json.starts_with('['));
    assert!(json.ends_with(']'));
}

#[test]
fn test_to_json_record() {
    let result = eval_str(r#"ToJson([a |-> 1, b |-> 2])"#).unwrap();
    let json = result.as_string().unwrap();
    assert!(json.starts_with('{'));
    assert!(json.ends_with('}'));
    assert!(json.contains("\"a\":1") || json.contains("\"a\": 1"));
}

#[test]
fn test_to_json_nested() {
    let result = eval_str(r#"ToJson(<<[a |-> 1], [a |-> 2]>>)"#).unwrap();
    let json = result.as_string().unwrap();
    // Should be a JSON array of objects
    assert!(json.starts_with('['));
    assert!(json.contains("\"a\""));
}

#[test]
fn test_to_json_array_sequence() {
    let result = eval_str(r#"ToJsonArray(<<1, 2, 3>>)"#).unwrap();
    assert_eq!(result.as_string().unwrap(), "[1,2,3]");
}

#[test]
fn test_to_json_array_set() {
    let result = eval_str(r#"ToJsonArray({1, 2, 3})"#).unwrap();
    let json = result.as_string().unwrap();
    assert!(json.starts_with('['));
    assert!(json.ends_with(']'));
}

#[test]
fn test_to_json_object_record() {
    let result = eval_str(r#"ToJsonObject([x |-> 10, y |-> 20])"#).unwrap();
    let json = result.as_string().unwrap();
    assert!(json.starts_with('{'));
    assert!(json.ends_with('}'));
}

#[test]
fn test_json_serialize_returns_true() {
    // JsonSerialize should return TRUE (stub implementation)
    let result = eval_str(r#"JsonSerialize("test.json", <<1, 2, 3>>)"#).unwrap();
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn test_json_deserialize_returns_empty() {
    // JsonDeserialize returns empty record (stub implementation)
    let result = eval_str(r#"JsonDeserialize("test.json")"#).unwrap();
    let rec = result.as_record().expect("Expected record");
    assert!(rec.is_empty());
}

#[test]
fn test_ndjson_serialize_returns_true() {
    // ndJsonSerialize should return TRUE (stub implementation)
    let result = eval_str(r#"ndJsonSerialize("test.ndjson", <<1, 2, 3>>)"#).unwrap();
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn test_ndjson_deserialize_returns_empty() {
    // ndJsonDeserialize returns empty sequence (stub implementation)
    let result = eval_str(r#"ndJsonDeserialize("test.ndjson")"#).unwrap();
    let seq = result.as_seq().expect("Expected sequence");
    assert!(seq.is_empty());
}

// ============================================================================
// Reals module tests
// ============================================================================
// TLA+ Reals module: TLC doesn't actually support real numbers.
// In TLA2, we implement Real as a superset of Int (Int ⊆ Real).
// The Infinity constant is defined but errors on arithmetic operations.

#[test]
fn test_reals_membership_positive() {
    // EXTENDS Reals required for Real constant
    let result = eval_with_extends(r#"5 \in Real"#, &["Reals"]).unwrap();
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn test_reals_membership_negative() {
    // Negative integers are also in Real
    let result = eval_with_extends(r#"-3 \in Real"#, &["Reals"]).unwrap();
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn test_reals_membership_zero() {
    // Zero is in Real
    let result = eval_with_extends(r#"0 \in Real"#, &["Reals"]).unwrap();
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn test_reals_non_membership_string() {
    // Strings are not in Real
    let result = eval_with_extends(r#""hello" \in Real"#, &["Reals"]).unwrap();
    assert_eq!(result, Value::Bool(false));
}

#[test]
fn test_reals_non_membership_bool() {
    // Booleans are not in Real
    let result = eval_with_extends(r#"TRUE \in Real"#, &["Reals"]).unwrap();
    assert_eq!(result, Value::Bool(false));
}

#[test]
fn test_reals_non_membership_set() {
    // Sets are not in Real
    let result = eval_with_extends(r#"{1, 2} \in Real"#, &["Reals"]).unwrap();
    assert_eq!(result, Value::Bool(false));
}

#[test]
fn test_reals_notin() {
    // \notin works for Real
    let result = eval_with_extends(r#""hello" \notin Real"#, &["Reals"]).unwrap();
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn test_reals_int_notin() {
    // Integers are NOT in the "not in Real" category
    let result = eval_with_extends(r#"42 \notin Real"#, &["Reals"]).unwrap();
    assert_eq!(result, Value::Bool(false));
}

#[test]
fn test_reals_subseteq_from_finite_set() {
    // {1, 2, 3} \subseteq Real is TRUE (since Int ⊆ Real)
    let result = eval_with_extends(r#"{1, 2, 3} \subseteq Real"#, &["Reals"]).unwrap();
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn test_reals_subseteq_mixed_types() {
    // {"hello", 1} \subseteq Real is FALSE (string not in Real)
    let result = eval_with_extends(r#"{"hello"} \subseteq Real"#, &["Reals"]).unwrap();
    assert_eq!(result, Value::Bool(false));
}

#[test]
fn test_infinity_constant_exists() {
    // Infinity constant should be defined (as a ModelValue)
    let result = eval_with_extends(r#"Infinity"#, &["Reals"]).unwrap();
    // Infinity is a ModelValue representing an abstract concept
    if let Value::ModelValue(name) = result {
        assert_eq!(name.as_ref(), "Infinity");
    } else {
        panic!("Expected Infinity to be a ModelValue");
    }
}

#[test]
fn test_real_function_domain() {
    // Functions with Real domain should work
    let result = eval_with_extends(r#"LET f[n \in Real] == n * 2 IN f[5]"#, &["Reals"]).unwrap();
    assert_eq!(result, Value::int(10));
}

/// Helper function to evaluate TLA+ with specific EXTENDS
fn eval_with_extends(src: &str, extends: &[&str]) -> Result<Value, String> {
    let extends_list = extends.join(", ");
    let module_src = format!(
        "---- MODULE Test ----\nEXTENDS {}\n\nOp == {}\n\n====",
        extends_list, src
    );
    let tree = parse_to_syntax_tree(&module_src);
    let lower_result = lower(FileId(0), &tree);
    let module = match lower_result.module {
        Some(m) => m,
        None => return Err(format!("Parse error: {:?}", lower_result.errors)),
    };

    for unit in &module.units {
        if let tla_core::ast::Unit::Operator(def) = &unit.node {
            if def.name.node == "Op" {
                let ctx = tla_check::eval::EvalCtx::new();
                return tla_check::eval::eval(&ctx, &def.body).map_err(|e| format!("{:?}", e));
            }
        }
    }
    Err("Op not found".to_string())
}

// ============================================================================
// TLCExt operator tests - TLCEvalDefinition, Trace, CounterExample, ToTrace
// ============================================================================

#[test]
fn test_tlc_eval_definition() {
    // TLCEvalDefinition evaluates a zero-arity definition by name
    let module_src = r#"---- MODULE Test ----
EXTENDS TLC, TLCExt

MyConst == 42
Op == TLCEvalDefinition("MyConst")

===="#;
    let tree = parse_to_syntax_tree(module_src);
    let lower_result = lower(FileId(0), &tree);
    let module = lower_result.module.unwrap();

    let mut ctx = tla_check::eval::EvalCtx::new();
    ctx.load_module(&module);
    let result = ctx.eval_op("Op").unwrap();
    assert_eq!(result, Value::int(42));
}

#[test]
fn test_tlc_eval_definition_complex() {
    // TLCEvalDefinition with a more complex expression
    let module_src = r#"---- MODULE Test ----
EXTENDS TLC, TLCExt

MySet == {1, 2, 3}
Op == TLCEvalDefinition("MySet")

===="#;
    let tree = parse_to_syntax_tree(module_src);
    let lower_result = lower(FileId(0), &tree);
    let module = lower_result.module.unwrap();

    let mut ctx = tla_check::eval::EvalCtx::new();
    ctx.load_module(&module);
    let result = ctx.eval_op("Op").unwrap();
    assert_eq!(result, int_set(&[1, 2, 3]));
}

#[test]
fn test_trace_stub() {
    // Trace returns empty tuple (stub implementation)
    let result = eval_tlcext_str("Trace").unwrap();
    assert_eq!(result, Value::Tuple(vec![].into()));
}

#[test]
fn test_counter_example_stub() {
    // CounterExample returns empty record (stub implementation)
    let result = eval_tlcext_str("CounterExample").unwrap();
    assert_eq!(result, Value::Record(tla_check::value::RecordValue::new()));
}

#[test]
fn test_to_trace_stub() {
    // ToTrace returns empty tuple from CounterExample (stub implementation)
    let result = eval_tlcext_str("ToTrace(CounterExample)").unwrap();
    assert_eq!(result, Value::Tuple(vec![].into()));
}

// ============================================================================
// SequencesExt - ReplaceAll, Interleave, SubSeqs
// ============================================================================

#[test]
fn test_replace_all_basic() {
    // ReplaceAll(s, old, new) replaces all occurrences
    let result = eval_seqext_str("ReplaceAll(<<1, 2, 1, 3, 1>>, 1, 9)").unwrap();
    assert_eq!(
        result,
        Value::Seq(
            vec![
                Value::int(9),
                Value::int(2),
                Value::int(9),
                Value::int(3),
                Value::int(9)
            ]
            .into()
        )
    );
}

#[test]
fn test_replace_all_no_match() {
    // ReplaceAll with no matching elements leaves sequence unchanged
    let result = eval_seqext_str("ReplaceAll(<<1, 2, 3>>, 5, 9)").unwrap();
    assert_eq!(
        result,
        Value::Seq(vec![Value::int(1), Value::int(2), Value::int(3)].into())
    );
}

#[test]
fn test_replace_all_empty_seq() {
    // ReplaceAll on empty sequence returns empty sequence
    let result = eval_seqext_str("ReplaceAll(<<>>, 1, 2)").unwrap();
    assert_eq!(result, Value::Seq(vec![].into()));
}

#[test]
fn test_interleave_equal_length() {
    // Interleave(s, t) interleaves two sequences of equal length
    let result = eval_seqext_str("Interleave(<<1, 2, 3>>, <<4, 5, 6>>)").unwrap();
    assert_eq!(
        result,
        Value::Seq(
            vec![
                Value::int(1),
                Value::int(4),
                Value::int(2),
                Value::int(5),
                Value::int(3),
                Value::int(6)
            ]
            .into()
        )
    );
}

#[test]
fn test_interleave_first_longer() {
    // Interleave with first sequence longer - remaining elements appended
    let result = eval_seqext_str("Interleave(<<1, 2, 3, 4>>, <<5, 6>>)").unwrap();
    assert_eq!(
        result,
        Value::Seq(
            vec![
                Value::int(1),
                Value::int(5),
                Value::int(2),
                Value::int(6),
                Value::int(3),
                Value::int(4)
            ]
            .into()
        )
    );
}

#[test]
fn test_interleave_second_longer() {
    // Interleave with second sequence longer - remaining elements appended
    let result = eval_seqext_str("Interleave(<<1, 2>>, <<5, 6, 7, 8>>)").unwrap();
    assert_eq!(
        result,
        Value::Seq(
            vec![
                Value::int(1),
                Value::int(5),
                Value::int(2),
                Value::int(6),
                Value::int(7),
                Value::int(8)
            ]
            .into()
        )
    );
}

#[test]
fn test_interleave_empty() {
    // Interleave with both empty sequences
    let result = eval_seqext_str("Interleave(<<>>, <<>>)").unwrap();
    assert_eq!(result, Value::Seq(vec![].into()));
}

#[test]
fn test_interleave_one_empty() {
    // Interleave with one empty sequence
    let result = eval_seqext_str("Interleave(<<1, 2>>, <<>>)").unwrap();
    assert_eq!(
        result,
        Value::Seq(vec![Value::int(1), Value::int(2)].into())
    );
}

#[test]
fn test_subseqs_basic() {
    // SubSeqs(s) returns all contiguous subsequences including empty
    let result = eval_seqext_str("SubSeqs(<<1, 2>>)").unwrap();
    let expected = int_set_of_seqs(&[vec![], vec![1], vec![2], vec![1, 2]]);
    assert_eq!(result, expected);
}

#[test]
fn test_subseqs_singleton() {
    // SubSeqs of singleton
    let result = eval_seqext_str("SubSeqs(<<1>>)").unwrap();
    let expected = int_set_of_seqs(&[vec![], vec![1]]);
    assert_eq!(result, expected);
}

#[test]
fn test_subseqs_empty() {
    // SubSeqs of empty sequence is just the empty sequence
    let result = eval_seqext_str("SubSeqs(<<>>)").unwrap();
    let expected = int_set_of_seqs(&[vec![]]);
    assert_eq!(result, expected);
}

#[test]
fn test_subseqs_three() {
    // SubSeqs of length 3 sequence
    let result = eval_seqext_str("SubSeqs(<<1, 2, 3>>)").unwrap();
    // Count: empty + 3 singletons + 2 pairs + 1 triple = 7
    let expected = int_set_of_seqs(&[
        vec![],
        vec![1],
        vec![2],
        vec![3],
        vec![1, 2],
        vec![2, 3],
        vec![1, 2, 3],
    ]);
    assert_eq!(result, expected);
}

/// Helper to create a set of integer sequences
fn int_set_of_seqs(seqs: &[Vec<i64>]) -> Value {
    let mut set = im::OrdSet::new();
    for seq in seqs {
        let values: Vec<Value> = seq.iter().map(|&n| Value::int(n)).collect();
        set.insert(Value::Seq(values.into()));
    }
    Value::Set(SortedSet::from_ord_set(&set))
}

// ============================================================================
// SetToSeqs tests (all permutations)
// ============================================================================

#[test]
fn test_set_to_seqs_basic() {
    // SetToSeqs({1, 2}) returns both permutations
    let result = eval_seqext_str("SetToSeqs({1, 2})").unwrap();
    let expected = int_set_of_seqs(&[vec![1, 2], vec![2, 1]]);
    assert_eq!(result, expected);
}

#[test]
fn test_set_to_seqs_three_elements() {
    // SetToSeqs({1, 2, 3}) returns all 6 permutations (3! = 6)
    let result = eval_seqext_str("SetToSeqs({1, 2, 3})").unwrap();
    let expected = int_set_of_seqs(&[
        vec![1, 2, 3],
        vec![1, 3, 2],
        vec![2, 1, 3],
        vec![2, 3, 1],
        vec![3, 1, 2],
        vec![3, 2, 1],
    ]);
    assert_eq!(result, expected);
}

#[test]
fn test_set_to_seqs_singleton() {
    // SetToSeqs({1}) returns single permutation
    let result = eval_seqext_str("SetToSeqs({1})").unwrap();
    let expected = int_set_of_seqs(&[vec![1]]);
    assert_eq!(result, expected);
}

#[test]
fn test_set_to_seqs_empty() {
    // SetToSeqs({}) returns set with empty sequence
    let result = eval_seqext_str("SetToSeqs({})").unwrap();
    let expected = int_set_of_seqs(&[vec![]]);
    assert_eq!(result, expected);
}

// ============================================================================
// AllSubSeqs tests (non-contiguous subsequences)
// ============================================================================

#[test]
fn test_all_subseqs_basic() {
    // AllSubSeqs(<<1, 2>>) returns all 2^2 = 4 subsequences
    let result = eval_seqext_str("AllSubSeqs(<<1, 2>>)").unwrap();
    let expected = int_set_of_seqs(&[vec![], vec![1], vec![2], vec![1, 2]]);
    assert_eq!(result, expected);
}

#[test]
fn test_all_subseqs_three() {
    // AllSubSeqs(<<1, 2, 3>>) returns all 2^3 = 8 subsequences
    let result = eval_seqext_str("AllSubSeqs(<<1, 2, 3>>)").unwrap();
    let expected = int_set_of_seqs(&[
        vec![],
        vec![1],
        vec![2],
        vec![3],
        vec![1, 2],
        vec![1, 3],
        vec![2, 3],
        vec![1, 2, 3],
    ]);
    assert_eq!(result, expected);
}

#[test]
fn test_all_subseqs_empty() {
    // AllSubSeqs(<<>>) returns just empty sequence
    let result = eval_seqext_str("AllSubSeqs(<<>>)").unwrap();
    let expected = int_set_of_seqs(&[vec![]]);
    assert_eq!(result, expected);
}

#[test]
fn test_all_subseqs_singleton() {
    // AllSubSeqs(<<1>>) returns empty and singleton
    let result = eval_seqext_str("AllSubSeqs(<<1>>)").unwrap();
    let expected = int_set_of_seqs(&[vec![], vec![1]]);
    assert_eq!(result, expected);
}

// ============================================================================
// LongestCommonPrefix tests
// ============================================================================

#[test]
fn test_longest_common_prefix_basic() {
    // LongestCommonPrefix of two sequences with common prefix
    let result = eval_seqext_str("LongestCommonPrefix({<<1, 2, 3>>, <<1, 2, 4>>})").unwrap();
    assert_eq!(
        result,
        Value::Seq(vec![Value::int(1), Value::int(2)].into())
    );
}

#[test]
fn test_longest_common_prefix_no_common() {
    // LongestCommonPrefix with no common prefix
    let result = eval_seqext_str("LongestCommonPrefix({<<1, 2>>, <<3, 4>>})").unwrap();
    assert_eq!(result, Value::Seq(vec![].into()));
}

#[test]
fn test_longest_common_prefix_identical() {
    // LongestCommonPrefix of identical sequences
    let result = eval_seqext_str("LongestCommonPrefix({<<1, 2, 3>>, <<1, 2, 3>>})").unwrap();
    assert_eq!(
        result,
        Value::Seq(vec![Value::int(1), Value::int(2), Value::int(3)].into())
    );
}

#[test]
fn test_longest_common_prefix_one_prefix_of_other() {
    // LongestCommonPrefix where one is prefix of other
    let result = eval_seqext_str("LongestCommonPrefix({<<1, 2>>, <<1, 2, 3>>})").unwrap();
    assert_eq!(
        result,
        Value::Seq(vec![Value::int(1), Value::int(2)].into())
    );
}

#[test]
fn test_longest_common_prefix_single() {
    // LongestCommonPrefix of single sequence
    let result = eval_seqext_str("LongestCommonPrefix({<<1, 2, 3>>})").unwrap();
    assert_eq!(
        result,
        Value::Seq(vec![Value::int(1), Value::int(2), Value::int(3)].into())
    );
}

#[test]
fn test_longest_common_prefix_empty_set() {
    // LongestCommonPrefix of empty set
    let result = eval_seqext_str("LongestCommonPrefix({})").unwrap();
    assert_eq!(result, Value::Seq(vec![].into()));
}

// ============================================================================
// CommonPrefixes tests
// ============================================================================

#[test]
fn test_common_prefixes_basic() {
    // CommonPrefixes of two sequences with common prefix
    let result = eval_seqext_str("CommonPrefixes({<<1, 2, 3>>, <<1, 2, 4>>})").unwrap();
    let expected = int_set_of_seqs(&[vec![], vec![1], vec![1, 2]]);
    assert_eq!(result, expected);
}

#[test]
fn test_common_prefixes_no_common() {
    // CommonPrefixes with no common prefix (only empty)
    let result = eval_seqext_str("CommonPrefixes({<<1, 2>>, <<3, 4>>})").unwrap();
    let expected = int_set_of_seqs(&[vec![]]);
    assert_eq!(result, expected);
}

#[test]
fn test_common_prefixes_identical() {
    // CommonPrefixes of identical sequences
    let result = eval_seqext_str("CommonPrefixes({<<1, 2>>, <<1, 2>>})").unwrap();
    let expected = int_set_of_seqs(&[vec![], vec![1], vec![1, 2]]);
    assert_eq!(result, expected);
}

#[test]
fn test_common_prefixes_empty_set() {
    // CommonPrefixes of empty set
    let result = eval_seqext_str("CommonPrefixes({})").unwrap();
    let expected = int_set_of_seqs(&[vec![]]);
    assert_eq!(result, expected);
}

// ============================================================================
// FoldLeftDomain tests
// ============================================================================

#[test]
fn test_fold_left_domain_sum_indices() {
    // Use FoldLeftDomain to sum elements weighted by their indices
    // Sum(i * s[i]) over sequence <<10, 20, 30>>
    // = 1*10 + 2*20 + 3*30 = 10 + 40 + 90 = 140
    let module_src = r#"---- MODULE Test ----
EXTENDS Sequences, SequencesExt

WeightedAdd(acc, elem, idx) == acc + (idx * elem)
Result == FoldLeftDomain(WeightedAdd, 0, <<10, 20, 30>>)

===="#;
    let tree = parse_to_syntax_tree(module_src);
    let lower_result = lower(FileId(0), &tree);
    let module = lower_result.module.unwrap();

    let mut ctx = tla_check::eval::EvalCtx::new();
    ctx.load_module(&module);
    let result = ctx.eval_op("Result").unwrap();
    assert_eq!(result, Value::int(140));
}

#[test]
fn test_fold_left_domain_build_pairs() {
    // Use FoldLeftDomain to collect (index, element) pairs into a set
    let module_src = r#"---- MODULE Test ----
EXTENDS Sequences, SequencesExt

AddPair(acc, elem, idx) == acc \union { <<idx, elem>> }
Result == FoldLeftDomain(AddPair, {}, <<10, 20>>)

===="#;
    let tree = parse_to_syntax_tree(module_src);
    let lower_result = lower(FileId(0), &tree);
    let module = lower_result.module.unwrap();

    let mut ctx = tla_check::eval::EvalCtx::new();
    ctx.load_module(&module);
    let result = ctx.eval_op("Result").unwrap();

    // Expected: { <<1, 10>>, <<2, 20>> }
    let mut expected = im::OrdSet::new();
    expected.insert(Value::Tuple(vec![Value::int(1), Value::int(10)].into()));
    expected.insert(Value::Tuple(vec![Value::int(2), Value::int(20)].into()));
    assert_eq!(result, Value::Set(SortedSet::from_ord_set(&expected)));
}

#[test]
fn test_fold_left_domain_empty() {
    // FoldLeftDomain on empty sequence returns base
    let module_src = r#"---- MODULE Test ----
EXTENDS Sequences, SequencesExt

Op(acc, elem, idx) == acc + elem
Result == FoldLeftDomain(Op, 42, <<>>)

===="#;
    let tree = parse_to_syntax_tree(module_src);
    let lower_result = lower(FileId(0), &tree);
    let module = lower_result.module.unwrap();

    let mut ctx = tla_check::eval::EvalCtx::new();
    ctx.load_module(&module);
    let result = ctx.eval_op("Result").unwrap();
    assert_eq!(result, Value::int(42));
}

// ============================================================================
// FoldRightDomain tests
// ============================================================================

#[test]
fn test_fold_right_domain_build_list() {
    // Use FoldRightDomain to prepend (idx, elem) pairs
    // Processing <<10, 20>> from right:
    // idx=2: Op(20, <<>>, 2) = <<2, 20>> \o <<>> = <<2, 20>>
    // idx=1: Op(10, <<2, 20>>, 1) = <<1, 10>> \o <<2, 20>> = <<1, 10, 2, 20>>
    let module_src = r#"---- MODULE Test ----
EXTENDS Sequences, SequencesExt

Prepend(elem, acc, idx) == <<idx, elem>> \o acc
Result == FoldRightDomain(Prepend, <<10, 20>>, <<>>)

===="#;
    let tree = parse_to_syntax_tree(module_src);
    let lower_result = lower(FileId(0), &tree);
    let module = lower_result.module.unwrap();

    let mut ctx = tla_check::eval::EvalCtx::new();
    ctx.load_module(&module);
    let result = ctx.eval_op("Result").unwrap();
    assert_eq!(
        result,
        Value::Seq(vec![Value::int(1), Value::int(10), Value::int(2), Value::int(20)].into())
    );
}

#[test]
fn test_fold_right_domain_empty() {
    // FoldRightDomain on empty sequence returns base
    let module_src = r#"---- MODULE Test ----
EXTENDS Sequences, SequencesExt

Op(elem, acc, idx) == acc + elem
Result == FoldRightDomain(Op, <<>>, 99)

===="#;
    let tree = parse_to_syntax_tree(module_src);
    let lower_result = lower(FileId(0), &tree);
    let module = lower_result.module.unwrap();

    let mut ctx = tla_check::eval::EvalCtx::new();
    ctx.load_module(&module);
    let result = ctx.eval_op("Result").unwrap();
    assert_eq!(result, Value::int(99));
}

// Additional ProductBag tests (debug-mode alternatives to test_product_bag)
#[test]
fn test_product_bag_with_literal() {
    // ProductBag with function literal (simpler than BagCup, works in debug mode)
    // [2 |-> 3] means element 2 appears 3 times, product = 2*2*2 = 8
    let result = eval_bags_str("ProductBag([x \\in {2} |-> 3])").unwrap();
    assert_eq!(result, Value::int(8));
}

#[test]
fn test_product_bag_double_nested_bagcup() {
    // ProductBag with double-nested BagCup (works in debug mode, unlike triple nesting)
    let result = eval_bags_str("ProductBag(BagCup(SetToBag({2}), SetToBag({2})))").unwrap();
    // [2 |-> 2], so 2*2 = 4
    assert_eq!(result, Value::int(4));
}
