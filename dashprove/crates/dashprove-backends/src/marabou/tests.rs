//! Unit tests for Marabou backend

use super::*;
use crate::counterexample::CounterexampleValue;
use crate::traits::{PropertyType, VerificationStatus};
use dashprove_usl::ast::{
    BinaryOp, ComparisonOp, Expr, Field, Invariant, Property, Spec, Theorem, Type, TypeDef,
};
use dashprove_usl::typecheck::TypedSpec;
use std::time::Duration;

fn make_typed_spec(spec: Spec) -> TypedSpec {
    TypedSpec {
        spec,
        type_info: std::collections::HashMap::new(),
    }
}

// ============================================================================
// Config tests
// ============================================================================

#[test]
fn default_config() {
    let config = config::MarabouConfig::default();
    assert!(config.marabou_path.is_none());
    assert_eq!(config.timeout, Duration::from_secs(300));
    assert!(!config.split_and_conquer);
}

// ============================================================================
// Output parsing tests
// ============================================================================

#[test]
fn parse_sat_output() {
    let status = parsing::parse_output("sat\n", "");
    assert!(matches!(status, VerificationStatus::Disproven));
}

#[test]
fn parse_unsat_output() {
    let status = parsing::parse_output("unsat\n", "");
    assert!(matches!(status, VerificationStatus::Proven));
}

#[test]
fn parse_timeout_output() {
    let status = parsing::parse_output("", "TIMEOUT exceeded\n");
    assert!(matches!(status, VerificationStatus::Unknown { .. }));
}

// ============================================================================
// Counterexample parsing tests
// ============================================================================

#[test]
fn parse_counterexample_no_sat() {
    // No SAT in output - should return None
    let ce = parsing::parse_counterexample("unsat\n");
    assert!(ce.is_none());
}

#[test]
fn parse_counterexample_simple_equals() {
    let output = "sat\nx0 = 0.5\nx1 = -0.25\ny0 = 1.0\n";
    let ce = parsing::parse_counterexample(output).unwrap();

    assert_eq!(ce.witness.len(), 3);
    assert!(matches!(
        ce.witness.get("x0"),
        Some(CounterexampleValue::Float { value }) if (*value - 0.5).abs() < 1e-10
    ));
    assert!(matches!(
        ce.witness.get("x1"),
        Some(CounterexampleValue::Float { value }) if (*value - (-0.25)).abs() < 1e-10
    ));
    assert!(matches!(
        ce.witness.get("y0"),
        Some(CounterexampleValue::Float { value }) if (*value - 1.0).abs() < 1e-10
    ));
}

#[test]
fn parse_counterexample_colon_separator() {
    let output = "sat\nx0 : 0.5\nx1 : -0.25\n";
    let ce = parsing::parse_counterexample(output).unwrap();

    assert_eq!(ce.witness.len(), 2);
    assert!(ce.witness.contains_key("x0"));
    assert!(ce.witness.contains_key("x1"));
}

#[test]
fn parse_counterexample_labeled_inputs() {
    let output = "sat\nInput 0 = 0.5\nInput 1 = -0.25\nOutput 0 = 1.0\n";
    let ce = parsing::parse_counterexample(output).unwrap();

    assert_eq!(ce.witness.len(), 3);
    assert!(matches!(
        ce.witness.get("x0"),
        Some(CounterexampleValue::Float { value }) if (*value - 0.5).abs() < 1e-10
    ));
    assert!(matches!(
        ce.witness.get("x1"),
        Some(CounterexampleValue::Float { value }) if (*value - (-0.25)).abs() < 1e-10
    ));
    assert!(matches!(
        ce.witness.get("y0"),
        Some(CounterexampleValue::Float { value }) if (*value - 1.0).abs() < 1e-10
    ));
}

#[test]
fn parse_counterexample_labeled_bracket_format() {
    let output = "sat\nInput[0] = 0.5\nInput[1] = -0.25\nOutput[0] = 1.0\n";
    let ce = parsing::parse_counterexample(output).unwrap();

    assert_eq!(ce.witness.len(), 3);
    assert!(ce.witness.contains_key("x0"));
    assert!(ce.witness.contains_key("x1"));
    assert!(ce.witness.contains_key("y0"));
}

#[test]
fn parse_counterexample_numeric_lines() {
    let output = "sat\n0.5\n-0.25\n1.0\n";
    let ce = parsing::parse_counterexample(output).unwrap();

    // Should create x0, x1, x2 for sequential values
    assert_eq!(ce.witness.len(), 3);
    assert!(matches!(
        ce.witness.get("x0"),
        Some(CounterexampleValue::Float { value }) if (*value - 0.5).abs() < 1e-10
    ));
    assert!(matches!(
        ce.witness.get("x1"),
        Some(CounterexampleValue::Float { value }) if (*value - (-0.25)).abs() < 1e-10
    ));
    assert!(matches!(
        ce.witness.get("x2"),
        Some(CounterexampleValue::Float { value }) if (*value - 1.0).abs() < 1e-10
    ));
}

#[test]
fn parse_counterexample_special_values() {
    // Test infinity
    let output = "sat\nx0 = inf\nx1 = -inf\n";
    let ce = parsing::parse_counterexample(output).unwrap();
    assert!(matches!(
        ce.witness.get("x0"),
        Some(CounterexampleValue::Float { value }) if value.is_infinite() && *value > 0.0
    ));
    assert!(matches!(
        ce.witness.get("x1"),
        Some(CounterexampleValue::Float { value }) if value.is_infinite() && *value < 0.0
    ));
}

#[test]
fn parse_counterexample_scientific_notation() {
    let output = "sat\nx0 = 1.5e-10\nx1 = -2.3e5\n";
    let ce = parsing::parse_counterexample(output).unwrap();

    assert!(matches!(
        ce.witness.get("x0"),
        Some(CounterexampleValue::Float { value }) if (*value - 1.5e-10).abs() < 1e-20
    ));
    assert!(matches!(
        ce.witness.get("x1"),
        Some(CounterexampleValue::Float { value }) if (*value - (-2.3e5)).abs() < 1.0
    ));
}

#[test]
fn parse_counterexample_preserves_raw() {
    let output = "sat\nx0 = 0.5\n";
    let ce = parsing::parse_counterexample(output).unwrap();

    assert!(ce.raw.is_some());
    assert_eq!(ce.raw.as_ref().unwrap(), output);
}

#[test]
fn parse_counterexample_empty_sat_gives_raw() {
    // SAT but no parseable values - should return raw fallback
    let output = "sat\nsome unknown output format\n";
    let ce = parsing::parse_counterexample(output).unwrap();

    // Should have raw but possibly no structured witness (or unknown values)
    assert!(ce.raw.is_some());
}

#[test]
fn parse_value_integer() {
    let value = parsing::parse_value("42").unwrap();
    // Note: integers get parsed as floats first since they're valid floats
    assert!(matches!(value, CounterexampleValue::Float { value } if (value - 42.0).abs() < 1e-10));
}

#[test]
fn parse_value_float() {
    let value = parsing::parse_value("1.23456").unwrap();
    assert!(
        matches!(value, CounterexampleValue::Float { value } if (value - 1.23456).abs() < 1e-10)
    );
}

#[test]
fn parse_value_unknown() {
    let value = parsing::parse_value("some_string").unwrap();
    assert!(matches!(value, CounterexampleValue::Unknown(s) if s == "some_string"));
}

#[test]
fn parse_labeled_variable_input() {
    let (name, value) = parsing::parse_labeled_variable("Input 5 = 0.123").unwrap();
    assert_eq!(name, "x5");
    assert!(matches!(value, CounterexampleValue::Float { value } if (value - 0.123).abs() < 1e-10));
}

#[test]
fn parse_labeled_variable_output() {
    let (name, value) = parsing::parse_labeled_variable("Output 3 = -0.456").unwrap();
    assert_eq!(name, "y3");
    assert!(
        matches!(value, CounterexampleValue::Float { value } if (value - (-0.456)).abs() < 1e-10)
    );
}

#[test]
fn parse_labeled_variable_invalid() {
    assert!(parsing::parse_labeled_variable("x0 = 0.5").is_none());
    assert!(parsing::parse_labeled_variable("Invalid 0 = 0.5").is_none());
}

// ============================================================================
// VNNLIB compilation tests
// ============================================================================

#[test]
fn vnnlib_identifier_conversion() {
    // Input patterns
    assert_eq!(vnnlib::to_vnnlib_ident("x0"), "X_0");
    assert_eq!(vnnlib::to_vnnlib_ident("x1"), "X_1");
    assert_eq!(vnnlib::to_vnnlib_ident("input0"), "X_0");
    assert_eq!(vnnlib::to_vnnlib_ident("input_5"), "X_5");
    assert_eq!(vnnlib::to_vnnlib_ident("in3"), "X_3");

    // Output patterns
    assert_eq!(vnnlib::to_vnnlib_ident("y0"), "Y_0");
    assert_eq!(vnnlib::to_vnnlib_ident("output0"), "Y_0");
    assert_eq!(vnnlib::to_vnnlib_ident("output_2"), "Y_2");
    assert_eq!(vnnlib::to_vnnlib_ident("out1"), "Y_1");

    // Non-indexed names stay as-is
    assert_eq!(vnnlib::to_vnnlib_ident("state"), "state");
    assert_eq!(vnnlib::to_vnnlib_ident("counter"), "counter");
}

#[test]
fn extract_index_from_name() {
    assert_eq!(vnnlib::extract_index("x0"), Some(0));
    assert_eq!(vnnlib::extract_index("x123"), Some(123));
    assert_eq!(vnnlib::extract_index("input_5"), Some(5));
    assert_eq!(vnnlib::extract_index("y42"), Some(42));
    assert_eq!(vnnlib::extract_index("output"), None);
    assert_eq!(vnnlib::extract_index("abc"), None);
}

#[test]
fn vnnlib_basic_generation() {
    let spec = make_typed_spec(Spec {
        types: vec![],
        properties: vec![Property::Theorem(Theorem {
            name: "test".to_string(),
            body: Expr::Bool(true),
        })],
    });

    let vnnlib_str = vnnlib::generate_vnnlib(&spec).unwrap();
    assert!(vnnlib_str.contains("; Generated by DashProve"));
    assert!(vnnlib_str.contains("; Property: test"));
    // Should have at least one input and output
    assert!(vnnlib_str.contains("(declare-const X_0 Real)"));
    assert!(vnnlib_str.contains("(declare-const Y_0 Real)"));
}

#[test]
fn vnnlib_extracts_input_bounds() {
    // Property: x0 >= 0 and x0 <= 1
    let spec = make_typed_spec(Spec {
        types: vec![],
        properties: vec![Property::Theorem(Theorem {
            name: "bounded_input".to_string(),
            body: Expr::And(
                Box::new(Expr::Compare(
                    Box::new(Expr::Var("x0".to_string())),
                    ComparisonOp::Ge,
                    Box::new(Expr::Float(0.0)),
                )),
                Box::new(Expr::Compare(
                    Box::new(Expr::Var("x0".to_string())),
                    ComparisonOp::Le,
                    Box::new(Expr::Float(1.0)),
                )),
            ),
        })],
    });

    let vnnlib_str = vnnlib::generate_vnnlib(&spec).unwrap();
    assert!(vnnlib_str.contains("(declare-const X_0 Real)"));
    // Lower bound
    assert!(vnnlib_str.contains("(assert (>= X_0 0") || vnnlib_str.contains("(assert (>= X_0 0.0"));
    // Upper bound
    assert!(vnnlib_str.contains("(assert (<= X_0 1") || vnnlib_str.contains("(assert (<= X_0 1.0"));
}

#[test]
fn vnnlib_extracts_output_constraints() {
    // Property: y0 > 0.5
    let spec = make_typed_spec(Spec {
        types: vec![],
        properties: vec![Property::Theorem(Theorem {
            name: "output_positive".to_string(),
            body: Expr::Compare(
                Box::new(Expr::Var("y0".to_string())),
                ComparisonOp::Gt,
                Box::new(Expr::Float(0.5)),
            ),
        })],
    });

    let vnnlib_str = vnnlib::generate_vnnlib(&spec).unwrap();
    assert!(vnnlib_str.contains("(declare-const Y_0 Real)"));
    // Should have output constraint
    assert!(vnnlib_str.contains("; Output constraints"));
}

#[test]
fn vnnlib_multiple_inputs_outputs() {
    // Property: x0 >= 0 and x1 >= 0 and y0 > 0 and y1 > 0
    let spec = make_typed_spec(Spec {
        types: vec![],
        properties: vec![Property::Theorem(Theorem {
            name: "multi_io".to_string(),
            body: Expr::And(
                Box::new(Expr::And(
                    Box::new(Expr::Compare(
                        Box::new(Expr::Var("x0".to_string())),
                        ComparisonOp::Ge,
                        Box::new(Expr::Float(0.0)),
                    )),
                    Box::new(Expr::Compare(
                        Box::new(Expr::Var("x1".to_string())),
                        ComparisonOp::Ge,
                        Box::new(Expr::Float(0.0)),
                    )),
                )),
                Box::new(Expr::And(
                    Box::new(Expr::Compare(
                        Box::new(Expr::Var("y0".to_string())),
                        ComparisonOp::Gt,
                        Box::new(Expr::Float(0.0)),
                    )),
                    Box::new(Expr::Compare(
                        Box::new(Expr::Var("y1".to_string())),
                        ComparisonOp::Gt,
                        Box::new(Expr::Float(0.0)),
                    )),
                )),
            ),
        })],
    });

    let vnnlib_str = vnnlib::generate_vnnlib(&spec).unwrap();
    assert!(vnnlib_str.contains("(declare-const X_0 Real)"));
    assert!(vnnlib_str.contains("(declare-const X_1 Real)"));
    assert!(vnnlib_str.contains("(declare-const Y_0 Real)"));
    assert!(vnnlib_str.contains("(declare-const Y_1 Real)"));
}

#[test]
fn vnnlib_type_fields_extraction() {
    let spec = make_typed_spec(Spec {
        types: vec![TypeDef {
            name: "NeuralInput".to_string(),
            fields: vec![
                Field {
                    name: "x0".to_string(),
                    ty: Type::Named("Float".to_string()),
                },
                Field {
                    name: "x1".to_string(),
                    ty: Type::Named("Float".to_string()),
                },
            ],
        }],
        properties: vec![Property::Theorem(Theorem {
            name: "test".to_string(),
            body: Expr::Bool(true),
        })],
    });

    let vnnlib_str = vnnlib::generate_vnnlib(&spec).unwrap();
    // Type fields with 'x' prefix should contribute inputs
    assert!(vnnlib_str.contains("(declare-const X_0 Real)"));
    assert!(vnnlib_str.contains("(declare-const X_1 Real)"));
}

#[test]
fn compile_smt_comparison_operators() {
    // Test each comparison operator
    let x = Expr::Var("x0".to_string());
    let one = Expr::Float(1.0);

    assert_eq!(
        vnnlib::compile_comparison_to_smt(&x, ComparisonOp::Eq, &one),
        "(= X_0 1)"
    );
    assert_eq!(
        vnnlib::compile_comparison_to_smt(&x, ComparisonOp::Ne, &one),
        "(distinct X_0 1)"
    );
    assert_eq!(
        vnnlib::compile_comparison_to_smt(&x, ComparisonOp::Lt, &one),
        "(< X_0 1)"
    );
    assert_eq!(
        vnnlib::compile_comparison_to_smt(&x, ComparisonOp::Le, &one),
        "(<= X_0 1)"
    );
    assert_eq!(
        vnnlib::compile_comparison_to_smt(&x, ComparisonOp::Gt, &one),
        "(> X_0 1)"
    );
    assert_eq!(
        vnnlib::compile_comparison_to_smt(&x, ComparisonOp::Ge, &one),
        "(>= X_0 1)"
    );
}

#[test]
fn compile_smt_arithmetic() {
    let x = Expr::Var("x0".to_string());
    let y = Expr::Var("y0".to_string());

    // Addition
    let add = Expr::Binary(Box::new(x.clone()), BinaryOp::Add, Box::new(y.clone()));
    assert_eq!(vnnlib::compile_expr_to_smt(&add), "(+ X_0 Y_0)");

    // Subtraction
    let sub = Expr::Binary(Box::new(x.clone()), BinaryOp::Sub, Box::new(y.clone()));
    assert_eq!(vnnlib::compile_expr_to_smt(&sub), "(- X_0 Y_0)");

    // Multiplication
    let mul = Expr::Binary(Box::new(x.clone()), BinaryOp::Mul, Box::new(y.clone()));
    assert_eq!(vnnlib::compile_expr_to_smt(&mul), "(* X_0 Y_0)");

    // Division
    let div = Expr::Binary(Box::new(x.clone()), BinaryOp::Div, Box::new(y.clone()));
    assert_eq!(vnnlib::compile_expr_to_smt(&div), "(/ X_0 Y_0)");
}

#[test]
fn compile_smt_logical_operators() {
    let a = Expr::Bool(true);
    let b = Expr::Bool(false);

    // And
    let and_expr = Expr::And(Box::new(a.clone()), Box::new(b.clone()));
    assert_eq!(vnnlib::compile_expr_to_smt(&and_expr), "(and true false)");

    // Or
    let or_expr = Expr::Or(Box::new(a.clone()), Box::new(b.clone()));
    assert_eq!(vnnlib::compile_expr_to_smt(&or_expr), "(or true false)");

    // Not
    let not_expr = Expr::Not(Box::new(a.clone()));
    assert_eq!(vnnlib::compile_expr_to_smt(&not_expr), "(not true)");

    // Implies
    let implies_expr = Expr::Implies(Box::new(a.clone()), Box::new(b.clone()));
    assert_eq!(
        vnnlib::compile_expr_to_smt(&implies_expr),
        "(=> true false)"
    );
}

#[test]
fn compile_smt_literals() {
    // Positive int
    assert_eq!(vnnlib::compile_expr_to_smt(&Expr::Int(42)), "42.0");

    // Negative int
    assert_eq!(vnnlib::compile_expr_to_smt(&Expr::Int(-5)), "(- 5)");

    // Positive float
    assert_eq!(vnnlib::compile_expr_to_smt(&Expr::Float(3.25)), "3.25");

    // Negative float
    assert_eq!(vnnlib::compile_expr_to_smt(&Expr::Float(-2.5)), "(- 2.5)");

    // Booleans
    assert_eq!(vnnlib::compile_expr_to_smt(&Expr::Bool(true)), "true");
    assert_eq!(vnnlib::compile_expr_to_smt(&Expr::Bool(false)), "false");
}

#[test]
fn compile_smt_quantifiers() {
    // ForAll
    let forall = Expr::ForAll {
        var: "x".to_string(),
        ty: None,
        body: Box::new(Expr::Compare(
            Box::new(Expr::Var("x".to_string())),
            ComparisonOp::Ge,
            Box::new(Expr::Float(0.0)),
        )),
    };
    let smt = vnnlib::compile_expr_to_smt(&forall);
    assert!(smt.contains("(forall ((x Real))"));

    // Exists
    let exists = Expr::Exists {
        var: "y".to_string(),
        ty: None,
        body: Box::new(Expr::Compare(
            Box::new(Expr::Var("y".to_string())),
            ComparisonOp::Gt,
            Box::new(Expr::Float(0.0)),
        )),
    };
    let smt = vnnlib::compile_expr_to_smt(&exists);
    assert!(smt.contains("(exists ((y Real))"));
}

#[test]
fn consolidate_bounds_combines_correctly() {
    // Multiple bounds for same variable should be consolidated
    let bounds = vec![
        (0, 0.0, f64::INFINITY),     // x0 >= 0
        (0, f64::NEG_INFINITY, 1.0), // x0 <= 1
        (1, -1.0, f64::INFINITY),    // x1 >= -1
        (1, f64::NEG_INFINITY, 2.0), // x1 <= 2
    ];

    let consolidated = vnnlib::consolidate_bounds(&bounds);

    assert_eq!(consolidated.len(), 2);
    assert_eq!(consolidated.get(&0), Some(&(0.0, 1.0)));
    assert_eq!(consolidated.get(&1), Some(&(-1.0, 2.0)));
}

#[test]
fn vnnlib_invariant_property() {
    let spec = make_typed_spec(Spec {
        types: vec![],
        properties: vec![Property::Invariant(Invariant {
            name: "robustness".to_string(),
            body: Expr::Compare(
                Box::new(Expr::Var("x0".to_string())),
                ComparisonOp::Ge,
                Box::new(Expr::Float(-0.1)),
            ),
        })],
    });

    let vnnlib_str = vnnlib::generate_vnnlib(&spec).unwrap();
    assert!(vnnlib_str.contains("; Property: robustness"));
    assert!(vnnlib_str.contains("(declare-const X_0 Real)"));
}

#[test]
fn vnnlib_handles_empty_spec() {
    let spec = make_typed_spec(Spec {
        types: vec![],
        properties: vec![],
    });

    let vnnlib_str = vnnlib::generate_vnnlib(&spec).unwrap();
    // Should still generate valid VNNLIB with default input/output
    assert!(vnnlib_str.contains("; Generated by DashProve"));
    assert!(vnnlib_str.contains("(declare-const X_0 Real)"));
    assert!(vnnlib_str.contains("(declare-const Y_0 Real)"));
}

#[test]
fn vnnlib_field_access_extraction() {
    // Property with field access: net.x0
    let spec = make_typed_spec(Spec {
        types: vec![],
        properties: vec![Property::Theorem(Theorem {
            name: "field_test".to_string(),
            body: Expr::Compare(
                Box::new(Expr::FieldAccess(
                    Box::new(Expr::Var("net".to_string())),
                    "x0".to_string(),
                )),
                ComparisonOp::Ge,
                Box::new(Expr::Float(0.0)),
            ),
        })],
    });

    let vnnlib_str = vnnlib::generate_vnnlib(&spec).unwrap();
    // Should extract x0 from field access
    assert!(vnnlib_str.contains("(declare-const X_0 Real)"));
}

#[test]
fn vnnlib_binary_expression_extraction() {
    // Property: x0 + x1 > 0
    let spec = make_typed_spec(Spec {
        types: vec![],
        properties: vec![Property::Theorem(Theorem {
            name: "sum_positive".to_string(),
            body: Expr::Compare(
                Box::new(Expr::Binary(
                    Box::new(Expr::Var("x0".to_string())),
                    BinaryOp::Add,
                    Box::new(Expr::Var("x1".to_string())),
                )),
                ComparisonOp::Gt,
                Box::new(Expr::Float(0.0)),
            ),
        })],
    });

    let vnnlib_str = vnnlib::generate_vnnlib(&spec).unwrap();
    assert!(vnnlib_str.contains("(declare-const X_0 Real)"));
    assert!(vnnlib_str.contains("(declare-const X_1 Real)"));
}

#[test]
fn vnnlib_negation_expression() {
    // Property: -x0 < 1
    let spec = make_typed_spec(Spec {
        types: vec![],
        properties: vec![Property::Theorem(Theorem {
            name: "neg_test".to_string(),
            body: Expr::Compare(
                Box::new(Expr::Neg(Box::new(Expr::Var("x0".to_string())))),
                ComparisonOp::Lt,
                Box::new(Expr::Float(1.0)),
            ),
        })],
    });

    let vnnlib_str = vnnlib::generate_vnnlib(&spec).unwrap();
    assert!(vnnlib_str.contains("(declare-const X_0 Real)"));
}

#[test]
fn vnnlib_function_application() {
    // Property: abs(x0) < 1
    let spec = make_typed_spec(Spec {
        types: vec![],
        properties: vec![Property::Theorem(Theorem {
            name: "abs_test".to_string(),
            body: Expr::Compare(
                Box::new(Expr::App(
                    "abs".to_string(),
                    vec![Expr::Var("x0".to_string())],
                )),
                ComparisonOp::Lt,
                Box::new(Expr::Float(1.0)),
            ),
        })],
    });

    let vnnlib_str = vnnlib::generate_vnnlib(&spec).unwrap();
    assert!(vnnlib_str.contains("(declare-const X_0 Real)"));
}

#[test]
fn compile_smt_function_application() {
    // Function with no args
    let f0 = Expr::App("foo".to_string(), vec![]);
    assert_eq!(vnnlib::compile_expr_to_smt(&f0), "foo");

    // Function with args
    let f1 = Expr::App("abs".to_string(), vec![Expr::Var("x0".to_string())]);
    assert_eq!(vnnlib::compile_expr_to_smt(&f1), "(abs X_0)");

    // Function with multiple args
    let f2 = Expr::App(
        "max".to_string(),
        vec![Expr::Var("x0".to_string()), Expr::Var("x1".to_string())],
    );
    assert_eq!(vnnlib::compile_expr_to_smt(&f2), "(max X_0 X_1)");
}

// ============================================================================
// Backend trait tests
// ============================================================================

#[test]
fn backend_id_is_marabou() {
    use crate::traits::VerificationBackend;
    let backend = MarabouBackend::new();
    assert_eq!(backend.id(), crate::traits::BackendId::Marabou);
}

#[test]
fn backend_supports_neural_properties() {
    use crate::traits::VerificationBackend;
    let backend = MarabouBackend::new();
    let supported = backend.supports();
    assert!(supported.contains(&PropertyType::NeuralRobustness));
    assert!(supported.contains(&PropertyType::NeuralReachability));
}

// ============================================================================
// Model Path Extraction Tests
// ============================================================================

#[test]
fn extract_model_path_from_string_literal() {
    let spec = make_typed_spec(Spec {
        types: vec![],
        properties: vec![Property::Theorem(Theorem {
            name: "test".to_string(),
            body: Expr::String("models/mnist.onnx".to_string()),
        })],
    });
    let path = model::extract_model_path(&spec);
    assert_eq!(path, Some("models/mnist.onnx".to_string()));
}

#[test]
fn extract_model_path_from_comparison() {
    let spec = make_typed_spec(Spec {
        types: vec![],
        properties: vec![Property::Theorem(Theorem {
            name: "test".to_string(),
            body: Expr::Compare(
                Box::new(Expr::Var("model".to_string())),
                ComparisonOp::Eq,
                Box::new(Expr::String("network.onnx".to_string())),
            ),
        })],
    });
    let path = model::extract_model_path(&spec);
    assert_eq!(path, Some("network.onnx".to_string()));
}

#[test]
fn extract_model_path_from_app() {
    let spec = make_typed_spec(Spec {
        types: vec![],
        properties: vec![Property::Theorem(Theorem {
            name: "test".to_string(),
            body: Expr::App(
                "load_model".to_string(),
                vec![Expr::String("classifier.onnx".to_string())],
            ),
        })],
    });
    let path = model::extract_model_path(&spec);
    assert_eq!(path, Some("classifier.onnx".to_string()));
}

#[test]
fn extract_model_path_nested_in_and() {
    let spec = make_typed_spec(Spec {
        types: vec![],
        properties: vec![Property::Theorem(Theorem {
            name: "test".to_string(),
            body: Expr::And(
                Box::new(Expr::Bool(true)),
                Box::new(Expr::String("deep/model.onnx".to_string())),
            ),
        })],
    });
    let path = model::extract_model_path(&spec);
    assert_eq!(path, Some("deep/model.onnx".to_string()));
}

#[test]
fn extract_model_path_none_when_absent() {
    let spec = make_typed_spec(Spec {
        types: vec![],
        properties: vec![Property::Theorem(Theorem {
            name: "test".to_string(),
            body: Expr::Bool(true),
        })],
    });
    let path = model::extract_model_path(&spec);
    assert!(path.is_none());
}

#[test]
fn extract_model_path_pb_format() {
    let spec = make_typed_spec(Spec {
        types: vec![],
        properties: vec![Property::Theorem(Theorem {
            name: "test".to_string(),
            body: Expr::String("tensorflow/model.pb".to_string()),
        })],
    });
    let path = model::extract_model_path(&spec);
    assert_eq!(path, Some("tensorflow/model.pb".to_string()));
}

#[test]
fn extract_model_path_from_invariant() {
    let spec = make_typed_spec(Spec {
        types: vec![],
        properties: vec![Property::Invariant(Invariant {
            name: "test".to_string(),
            body: Expr::String("verifier/test.onnx".to_string()),
        })],
    });
    let path = model::extract_model_path(&spec);
    assert_eq!(path, Some("verifier/test.onnx".to_string()));
}

#[test]
fn extract_model_path_from_nested_expression() {
    let spec = make_typed_spec(Spec {
        types: vec![],
        properties: vec![Property::Theorem(Theorem {
            name: "test".to_string(),
            body: Expr::And(
                Box::new(Expr::Compare(
                    Box::new(Expr::Var("x0".to_string())),
                    ComparisonOp::Ge,
                    Box::new(Expr::Float(0.0)),
                )),
                Box::new(Expr::Or(
                    Box::new(Expr::Bool(false)),
                    Box::new(Expr::String("nested/path.onnx".to_string())),
                )),
            ),
        })],
    });
    let path = model::extract_model_path(&spec);
    assert_eq!(path, Some("nested/path.onnx".to_string()));
}
