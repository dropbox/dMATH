//! Tests for alpha-beta-CROWN backend

use super::*;
use crate::counterexample::CounterexampleValue;
use crate::traits::VerificationStatus;
use dashprove_usl::ast::{
    BinaryOp, ComparisonOp, Expr, Field, Invariant, Property, Spec, Theorem, Type, TypeDef,
};
use dashprove_usl::typecheck::TypedSpec;
use std::collections::HashMap;
use std::time::Duration;

fn make_typed_spec(spec: Spec) -> TypedSpec {
    TypedSpec {
        spec,
        type_info: HashMap::new(),
    }
}

#[test]
fn default_config() {
    let config = AbCrownConfig::default();
    assert!(config.use_gpu);
    assert_eq!(config.timeout, Duration::from_secs(300));
}

#[test]
fn parse_verified_output() {
    let status = parsing::parse_output("Property verified\n", "");
    assert!(matches!(status, VerificationStatus::Proven));
}

#[test]
fn parse_falsified_output() {
    let status = parsing::parse_output("Property falsified - counterexample found\n", "");
    assert!(matches!(status, VerificationStatus::Disproven));
}

#[test]
fn parse_unsat_output() {
    let status = parsing::parse_output("unsat\n", "");
    assert!(matches!(status, VerificationStatus::Proven));
}

#[test]
fn vnnlib_identifier_conversion() {
    assert_eq!(vnnlib::to_vnnlib_ident("x0"), "X_0");
    assert_eq!(vnnlib::to_vnnlib_ident("input_3"), "X_3");
    assert_eq!(vnnlib::to_vnnlib_ident("y1"), "Y_1");
    assert_eq!(vnnlib::to_vnnlib_ident("output5"), "Y_5");
    assert_eq!(vnnlib::to_vnnlib_ident("state"), "state");
}

#[test]
fn extract_index_from_name() {
    assert_eq!(vnnlib::extract_index("x0"), Some(0));
    assert_eq!(vnnlib::extract_index("input_12"), Some(12));
    assert_eq!(vnnlib::extract_index("output7"), Some(7));
    assert_eq!(vnnlib::extract_index("noindex"), None);
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
    assert!(vnnlib_str.contains("; Property: test"));
    assert!(vnnlib_str.contains("(declare-const X_0 Real)"));
    assert!(vnnlib_str.contains("(declare-const Y_0 Real)"));
}

#[test]
fn vnnlib_extracts_input_bounds() {
    let spec = make_typed_spec(Spec {
        types: vec![],
        properties: vec![Property::Theorem(Theorem {
            name: "bounded".to_string(),
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
    assert!(vnnlib_str.contains("(assert (>= X_0 0"));
    assert!(vnnlib_str.contains("(assert (<= X_0 1"));
}

#[test]
fn vnnlib_extracts_output_constraints() {
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
    assert!(vnnlib_str.contains("; Output constraints"));
}

#[test]
fn vnnlib_type_field_inference() {
    let spec = make_typed_spec(Spec {
        types: vec![TypeDef {
            name: "Inputs".to_string(),
            fields: vec![
                Field {
                    name: "x0".to_string(),
                    ty: Type::Named("Float".to_string()),
                },
                Field {
                    name: "output".to_string(),
                    ty: Type::Named("Float".to_string()),
                },
            ],
        }],
        properties: vec![Property::Invariant(Invariant {
            name: "always".to_string(),
            body: Expr::Bool(true),
        })],
    });

    let vnnlib_str = vnnlib::generate_vnnlib(&spec).unwrap();
    assert!(vnnlib_str.contains("(declare-const X_0 Real)"));
    assert!(vnnlib_str.contains("(declare-const Y_1 Real)"));
}

#[test]
fn compile_smt_arithmetic() {
    let x = Expr::Var("x0".to_string());
    let y = Expr::Var("y0".to_string());

    let add = Expr::Binary(Box::new(x.clone()), BinaryOp::Add, Box::new(y.clone()));
    assert_eq!(vnnlib::compile_expr_to_smt(&add), "(+ X_0 Y_0)");

    let cmp = Expr::Compare(
        Box::new(x.clone()),
        ComparisonOp::Ge,
        Box::new(Expr::Float(0.0)),
    );
    assert_eq!(vnnlib::compile_expr_to_smt(&cmp), "(>= X_0 0)");
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

// ============================================================================
// Counterexample Parsing Tests
// ============================================================================

#[test]
fn parse_counterexample_no_sat() {
    // No SAT in output - should return None
    let ce = parsing::parse_counterexample("unsat\n");
    assert!(ce.is_none());
}

#[test]
fn parse_counterexample_array_format() {
    let output = "sat\nadv_example: [0.1, 0.2, 0.3]\n";
    let ce = parsing::parse_counterexample(output).unwrap();

    assert_eq!(ce.witness.len(), 3);
    assert!(matches!(
        ce.witness.get("x0"),
        Some(CounterexampleValue::Float { value }) if (*value - 0.1).abs() < 1e-10
    ));
    assert!(matches!(
        ce.witness.get("x1"),
        Some(CounterexampleValue::Float { value }) if (*value - 0.2).abs() < 1e-10
    ));
    assert!(matches!(
        ce.witness.get("x2"),
        Some(CounterexampleValue::Float { value }) if (*value - 0.3).abs() < 1e-10
    ));
}

#[test]
fn parse_counterexample_nested_array() {
    let output = "sat\ninput: [[0.5, 0.6]]\n";
    let ce = parsing::parse_counterexample(output).unwrap();

    assert_eq!(ce.witness.len(), 2);
    assert!(matches!(
        ce.witness.get("x0"),
        Some(CounterexampleValue::Float { value }) if (*value - 0.5).abs() < 1e-10
    ));
    assert!(matches!(
        ce.witness.get("x1"),
        Some(CounterexampleValue::Float { value }) if (*value - 0.6).abs() < 1e-10
    ));
}

#[test]
fn parse_counterexample_variable_assignments() {
    let output = "sat\nX_0 = 0.5\nX_1 = -0.25\nY_0 = 1.0\n";
    let ce = parsing::parse_counterexample(output).unwrap();

    assert_eq!(ce.witness.len(), 3);
    assert!(ce.witness.contains_key("x0"));
    assert!(ce.witness.contains_key("x1"));
    assert!(ce.witness.contains_key("y0"));
}

#[test]
fn parse_counterexample_mixed_case_vars() {
    let output = "sat\nInput_0 = 0.1\nOutput_0 = 0.9\n";
    let ce = parsing::parse_counterexample(output).unwrap();

    assert_eq!(ce.witness.len(), 2);
    assert!(ce.witness.contains_key("x0"));
    assert!(ce.witness.contains_key("y0"));
}

#[test]
fn parse_counterexample_standalone_array() {
    let output = "sat\n[0.7, 0.8, 0.9]\n";
    let ce = parsing::parse_counterexample(output).unwrap();

    assert_eq!(ce.witness.len(), 3);
    assert!(matches!(
        ce.witness.get("x0"),
        Some(CounterexampleValue::Float { value }) if (*value - 0.7).abs() < 1e-10
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
fn parse_counterexample_fallback_to_raw() {
    // SAT but no parseable values
    let output = "sat\nsome unparseable format\n";
    let ce = parsing::parse_counterexample(output).unwrap();

    assert!(ce.raw.is_some());
    assert!(ce.witness.is_empty());
}

#[test]
fn parse_counterexample_negative_values() {
    let output = "sat\nadv_example: [-0.5, 0.3, -0.1]\n";
    let ce = parsing::parse_counterexample(output).unwrap();

    assert_eq!(ce.witness.len(), 3);
    assert!(matches!(
        ce.witness.get("x0"),
        Some(CounterexampleValue::Float { value }) if (*value - (-0.5)).abs() < 1e-10
    ));
}

#[test]
fn parse_counterexample_scientific_notation() {
    let output = "sat\ncounter: [1.5e-10, -2.3e5]\n";
    let ce = parsing::parse_counterexample(output).unwrap();

    assert_eq!(ce.witness.len(), 2);
    assert!(matches!(
        ce.witness.get("x0"),
        Some(CounterexampleValue::Float { value }) if *value < 1e-9
    ));
}

// ============================================================================
// Helper Function Tests
// ============================================================================

#[test]
fn parse_array_values_simple() {
    let values = parsing::parse_array_values("[1.0, 2.0, 3.0]").unwrap();
    assert_eq!(values.len(), 3);
    assert!((values[0] - 1.0).abs() < 1e-10);
    assert!((values[1] - 2.0).abs() < 1e-10);
    assert!((values[2] - 3.0).abs() < 1e-10);
}

#[test]
fn parse_array_values_nested() {
    let values = parsing::parse_array_values("[[1.0, 2.0]]").unwrap();
    assert_eq!(values.len(), 2);
}

#[test]
fn parse_array_values_empty() {
    let values = parsing::parse_array_values("[]");
    assert!(values.is_none());
}

#[test]
fn normalize_var_name_x_underscore() {
    let name = parsing::normalize_var_name("X_0");
    assert_eq!(name, "x0");
}

#[test]
fn normalize_var_name_input() {
    let name = parsing::normalize_var_name("Input_5");
    assert_eq!(name, "x5");
}

#[test]
fn normalize_var_name_output() {
    let name = parsing::normalize_var_name("Output_3");
    assert_eq!(name, "y3");
}

#[test]
fn normalize_var_name_y() {
    let name = parsing::normalize_var_name("Y_1");
    assert_eq!(name, "y1");
}

#[test]
fn extract_index_static_works() {
    assert_eq!(parsing::extract_index_static("x0"), Some(0));
    assert_eq!(parsing::extract_index_static("input_12"), Some(12));
    assert_eq!(parsing::extract_index_static("noindex"), None);
}
