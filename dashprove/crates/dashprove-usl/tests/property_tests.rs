//! Property-based tests for USL parser and type checker
//!
//! Uses proptest to generate random valid USL inputs and verify:
//! - Parsing never panics on any input
//! - Valid AST structures can be serialized/deserialized
//! - Type checking is deterministic
//! - Property name extraction is always non-empty for valid properties

use dashprove_usl::{
    parse, typecheck, Expr, Field, Invariant, Property, SemanticProperty, Spec, Temporal,
    TemporalExpr, Theorem, Type, TypeDef,
};
use proptest::prelude::*;

// ============================================================================
// Generators for USL AST types
// ============================================================================

/// Generate valid identifiers
fn identifier_strategy() -> impl Strategy<Value = String> {
    "[a-z][a-z0-9_]{0,10}".prop_map(|s| s.to_string())
}

/// Generate simple types (non-recursive)
fn simple_type_strategy() -> impl Strategy<Value = Type> {
    prop_oneof![
        Just(Type::Named("Bool".to_string())),
        Just(Type::Named("Int".to_string())),
        Just(Type::Named("String".to_string())),
        Just(Type::Unit),
    ]
}

/// Generate types up to depth 2
fn type_strategy() -> impl Strategy<Value = Type> {
    prop_oneof![
        simple_type_strategy(),
        simple_type_strategy().prop_map(|t| Type::Set(Box::new(t))),
        simple_type_strategy().prop_map(|t| Type::List(Box::new(t))),
        simple_type_strategy().prop_map(|t| Type::Result(Box::new(t))),
        (simple_type_strategy(), simple_type_strategy())
            .prop_map(|(k, v)| Type::Map(Box::new(k), Box::new(v))),
    ]
}

/// Generate simple expressions (non-recursive base cases)
fn leaf_expr_strategy() -> impl Strategy<Value = Expr> {
    prop_oneof![
        Just(Expr::Bool(true)),
        Just(Expr::Bool(false)),
        any::<i64>().prop_map(Expr::Int),
        identifier_strategy().prop_map(Expr::Var),
    ]
}

/// Generate expressions up to depth 2
fn expr_strategy() -> impl Strategy<Value = Expr> {
    leaf_expr_strategy().prop_recursive(2, 8, 4, |inner| {
        prop_oneof![
            inner.clone().prop_map(|e| Expr::Not(Box::new(e))),
            (inner.clone(), inner.clone()).prop_map(|(a, b)| Expr::And(Box::new(a), Box::new(b))),
            (inner.clone(), inner.clone()).prop_map(|(a, b)| Expr::Or(Box::new(a), Box::new(b))),
            (inner.clone(), inner.clone())
                .prop_map(|(a, b)| Expr::Implies(Box::new(a), Box::new(b))),
            (identifier_strategy(), simple_type_strategy(), inner.clone()).prop_map(
                |(var, ty, body)| Expr::ForAll {
                    var,
                    ty: Some(ty),
                    body: Box::new(body),
                }
            ),
            (identifier_strategy(), simple_type_strategy(), inner).prop_map(|(var, ty, body)| {
                Expr::Exists {
                    var,
                    ty: Some(ty),
                    body: Box::new(body),
                }
            }),
        ]
    })
}

/// Generate temporal expressions
fn temporal_expr_strategy() -> impl Strategy<Value = TemporalExpr> {
    leaf_expr_strategy()
        .prop_map(TemporalExpr::Atom)
        .prop_recursive(2, 6, 2, |inner| {
            prop_oneof![
                inner
                    .clone()
                    .prop_map(|e| TemporalExpr::Always(Box::new(e))),
                inner
                    .clone()
                    .prop_map(|e| TemporalExpr::Eventually(Box::new(e))),
                (inner.clone(), inner)
                    .prop_map(|(a, b)| TemporalExpr::LeadsTo(Box::new(a), Box::new(b))),
            ]
        })
}

/// Generate theorem properties
fn theorem_strategy() -> impl Strategy<Value = Theorem> {
    (identifier_strategy(), expr_strategy()).prop_map(|(name, body)| Theorem { name, body })
}

/// Generate invariant properties
fn invariant_strategy() -> impl Strategy<Value = Invariant> {
    (identifier_strategy(), expr_strategy()).prop_map(|(name, body)| Invariant { name, body })
}

/// Generate temporal properties
fn temporal_property_strategy() -> impl Strategy<Value = Temporal> {
    (identifier_strategy(), temporal_expr_strategy()).prop_map(|(name, body)| Temporal {
        name,
        body,
        fairness: vec![],
    })
}

/// Generate semantic properties
fn semantic_property_strategy() -> impl Strategy<Value = SemanticProperty> {
    (identifier_strategy(), expr_strategy())
        .prop_map(|(name, body)| SemanticProperty { name, body })
}

/// Generate property variants
fn property_strategy() -> impl Strategy<Value = Property> {
    prop_oneof![
        theorem_strategy().prop_map(Property::Theorem),
        invariant_strategy().prop_map(Property::Invariant),
        temporal_property_strategy().prop_map(Property::Temporal),
        semantic_property_strategy().prop_map(Property::Semantic),
    ]
}

/// Generate field definitions
fn field_strategy() -> impl Strategy<Value = Field> {
    (identifier_strategy(), type_strategy()).prop_map(|(name, ty)| Field { name, ty })
}

/// Generate type definitions
fn typedef_strategy() -> impl Strategy<Value = TypeDef> {
    (
        identifier_strategy()
            .prop_map(|s| s.chars().next().unwrap().to_uppercase().to_string() + &s[1..]),
        prop::collection::vec(field_strategy(), 0..4),
    )
        .prop_map(|(name, fields)| TypeDef { name, fields })
}

/// Generate complete specs
fn spec_strategy() -> impl Strategy<Value = Spec> {
    (
        prop::collection::vec(typedef_strategy(), 0..3),
        prop::collection::vec(property_strategy(), 0..5),
    )
        .prop_map(|(types, properties)| Spec { types, properties })
}

// ============================================================================
// Property tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Property: AST serialization round-trips through JSON
    #[test]
    fn spec_serialization_roundtrip(spec in spec_strategy()) {
        let json = serde_json::to_string(&spec).expect("serialize failed");
        let roundtrip: Spec = serde_json::from_str(&json).expect("deserialize failed");
        prop_assert_eq!(spec, roundtrip);
    }

    /// Property: Property::name() always returns non-empty string
    #[test]
    fn property_name_nonempty(prop in property_strategy()) {
        let name = prop.name();
        prop_assert!(!name.is_empty(), "Property name was empty");
    }

    /// Property: Type serialization round-trips
    #[test]
    fn type_serialization_roundtrip(ty in type_strategy()) {
        let json = serde_json::to_string(&ty).expect("serialize failed");
        let roundtrip: Type = serde_json::from_str(&json).expect("deserialize failed");
        prop_assert_eq!(ty, roundtrip);
    }

    /// Property: Expr serialization round-trips
    #[test]
    fn expr_serialization_roundtrip(expr in expr_strategy()) {
        let json = serde_json::to_string(&expr).expect("serialize failed");
        let roundtrip: Expr = serde_json::from_str(&json).expect("deserialize failed");
        prop_assert_eq!(expr, roundtrip);
    }

    /// Property: TemporalExpr serialization round-trips
    #[test]
    fn temporal_expr_serialization_roundtrip(expr in temporal_expr_strategy()) {
        let json = serde_json::to_string(&expr).expect("serialize failed");
        let roundtrip: TemporalExpr = serde_json::from_str(&json).expect("deserialize failed");
        prop_assert_eq!(expr, roundtrip);
    }

    /// Property: Empty spec is valid
    #[test]
    fn empty_spec_valid(_dummy in 0..1u8) {
        let spec = Spec::default();
        prop_assert!(spec.types.is_empty());
        prop_assert!(spec.properties.is_empty());
    }

    /// Property: Spec clone equals original
    #[test]
    fn spec_clone_equals(spec in spec_strategy()) {
        let cloned = spec.clone();
        prop_assert_eq!(spec, cloned);
    }
}

// ============================================================================
// Parser fuzzing tests (don't panic on arbitrary input)
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Property: Parser never panics on arbitrary UTF-8 input
    #[test]
    fn parser_no_panic_on_arbitrary_input(input in ".*") {
        // Should return Err, not panic
        let _ = parse(&input);
    }

    /// Property: Parser never panics on structured noise
    #[test]
    fn parser_no_panic_on_structured_noise(
        keyword in prop_oneof![
            Just("theorem"),
            Just("invariant"),
            Just("temporal"),
            Just("contract"),
            Just("type"),
            Just("forall"),
            Just("exists"),
        ],
        name in "[a-z]{1,5}",
        body in ".*",
    ) {
        let input = format!("{keyword} {name} {{ {body} }}");
        let _ = parse(&input);
    }
}

// ============================================================================
// Type checker property tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Property: Type checking is deterministic (same input = same result)
    #[test]
    fn typecheck_deterministic(spec in spec_strategy()) {
        let result1 = typecheck(spec.clone());
        let result2 = typecheck(spec);

        match (&result1, &result2) {
            (Ok(t1), Ok(t2)) => {
                prop_assert_eq!(t1.spec.types.len(), t2.spec.types.len());
                prop_assert_eq!(t1.spec.properties.len(), t2.spec.properties.len());
            }
            (Err(_), Err(_)) => {
                // Both failed, that's deterministic
            }
            _ => {
                prop_assert!(false, "Type checking gave different results");
            }
        }
    }
}

// ============================================================================
// Specific grammar tests via parsing
// ============================================================================

#[test]
fn parse_simple_theorem() {
    let input = "theorem test { true }";
    let spec = parse(input).expect("parse failed");
    assert_eq!(spec.properties.len(), 1);
    assert_eq!(spec.properties[0].name(), "test");
}

#[test]
fn parse_simple_invariant() {
    let input = "invariant safety { true }";
    let spec = parse(input).expect("parse failed");
    assert_eq!(spec.properties.len(), 1);
    assert_eq!(spec.properties[0].name(), "safety");
}

#[test]
fn parse_temporal_always() {
    let input = "temporal liveness { always(true) }";
    let spec = parse(input).expect("parse failed");
    assert_eq!(spec.properties.len(), 1);
    assert_eq!(spec.properties[0].name(), "liveness");
}

#[test]
fn parse_multiple_properties() {
    let input = r#"
        theorem t1 { true }
        theorem t2 { false }
        invariant i1 { true }
    "#;
    let spec = parse(input).expect("parse failed");
    assert_eq!(spec.properties.len(), 3);
}

#[test]
fn parse_with_types() {
    let input = r#"
        type State = {
            count: Int,
            active: Bool
        }

        invariant valid { true }
    "#;
    let spec = parse(input).expect("parse failed");
    assert_eq!(spec.types.len(), 1);
    assert_eq!(spec.types[0].name, "State");
    assert_eq!(spec.types[0].fields.len(), 2);
}

#[test]
fn parse_quantifiers() {
    let input = r#"
        theorem universal { forall x: Bool . x or not x }
        theorem existential { exists y: Int . y == 0 }
    "#;
    let spec = parse(input).expect("parse failed");
    assert_eq!(spec.properties.len(), 2);
}

#[test]
fn parse_nested_quantifiers() {
    let input = "theorem nested { forall x: Bool . forall y: Bool . x or y or not x }";
    let spec = parse(input).expect("parse failed");
    assert_eq!(spec.properties.len(), 1);
}

#[test]
fn typecheck_valid_spec() {
    let input = "theorem test { forall x: Bool . x or not x }";
    let spec = parse(input).expect("parse failed");
    let typed = typecheck(spec).expect("typecheck failed");
    assert_eq!(typed.spec.properties.len(), 1);
}

#[test]
fn empty_spec_parses() {
    let input = "";
    let spec = parse(input).expect("parse failed");
    assert!(spec.types.is_empty());
    assert!(spec.properties.is_empty());
}

#[test]
fn whitespace_only_parses() {
    let input = "   \n\t\n   ";
    let spec = parse(input).expect("parse failed");
    assert!(spec.types.is_empty());
    assert!(spec.properties.is_empty());
}
