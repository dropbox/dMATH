//! Tests for ERAN backend
//!
//! Unit tests for configuration, USL analysis, parsing, and verification.

use super::*;
use crate::counterexample::CounterexampleValue;
use crate::traits::{BackendId, PropertyType};
use dashprove_usl::ast::{
    BinaryOp, ComparisonOp, Expr, Field, Invariant, Property, Spec, Theorem, Type, TypeDef,
};
use dashprove_usl::typecheck::TypedSpec;
use std::collections::HashSet;

fn make_typed_spec(properties: Vec<Property>, types: Vec<TypeDef>) -> TypedSpec {
    TypedSpec {
        spec: Spec { properties, types },
        type_info: std::collections::HashMap::new(),
    }
}

#[test]
fn default_config() {
    let config = EranConfig::default();
    assert!(matches!(config.domain, EranDomain::DeepPoly));
    assert_eq!(config.epsilon, 0.01);
}

#[test]
fn parse_certified_output() {
    let (status, pct) = parsing::parse_output("certified: 100.0%\n", "");
    assert!(matches!(status, VerificationStatus::Proven));
    assert_eq!(pct, Some(100.0));
}

#[test]
fn parse_partial_output() {
    let (status, pct) = parsing::parse_output("certified: 85.5%\n", "");
    assert!(matches!(status, VerificationStatus::Partial { .. }));
    assert_eq!(pct, Some(85.5));
}

// USL compilation tests

#[test]
fn extract_index_from_names() {
    assert_eq!(usl_analysis::extract_index("x0"), Some(0));
    assert_eq!(usl_analysis::extract_index("x1"), Some(1));
    assert_eq!(usl_analysis::extract_index("input_2"), Some(2));
    assert_eq!(usl_analysis::extract_index("output3"), Some(3));
    assert_eq!(usl_analysis::extract_index("y10"), Some(10));
    assert_eq!(usl_analysis::extract_index("abc"), None);
}

#[test]
fn extract_epsilon_from_comparison() {
    // epsilon <= 0.01
    let expr = Expr::Compare(
        Box::new(Expr::Var("epsilon".to_string())),
        ComparisonOp::Le,
        Box::new(Expr::Float(0.01)),
    );
    assert_eq!(usl_analysis::extract_epsilon(&expr), Some(0.01));

    // eps <= 0.001
    let expr = Expr::Compare(
        Box::new(Expr::Var("eps".to_string())),
        ComparisonOp::Le,
        Box::new(Expr::Float(0.001)),
    );
    assert_eq!(usl_analysis::extract_epsilon(&expr), Some(0.001));
}

#[test]
fn extract_epsilon_from_reversed_comparison() {
    // 0.05 >= epsilon
    let expr = Expr::Compare(
        Box::new(Expr::Float(0.05)),
        ComparisonOp::Ge,
        Box::new(Expr::Var("epsilon".to_string())),
    );
    assert_eq!(usl_analysis::extract_epsilon(&expr), Some(0.05));
}

#[test]
fn extract_epsilon_from_abs_pattern() {
    // abs(x - x0) <= 0.03
    let expr = Expr::Compare(
        Box::new(Expr::App(
            "abs".to_string(),
            vec![Expr::Binary(
                Box::new(Expr::Var("x".to_string())),
                BinaryOp::Sub,
                Box::new(Expr::Var("x0".to_string())),
            )],
        )),
        ComparisonOp::Le,
        Box::new(Expr::Float(0.03)),
    );
    assert_eq!(usl_analysis::extract_epsilon(&expr), Some(0.03));
}

#[test]
fn extract_epsilon_from_robustness_function() {
    // robustness(x, 0.02)
    let expr = Expr::App(
        "robustness".to_string(),
        vec![Expr::Var("x".to_string()), Expr::Float(0.02)],
    );
    assert_eq!(usl_analysis::extract_epsilon(&expr), Some(0.02));
}

#[test]
fn extract_epsilon_nested_in_and() {
    // true and epsilon <= 0.04
    let expr = Expr::And(
        Box::new(Expr::Bool(true)),
        Box::new(Expr::Compare(
            Box::new(Expr::Var("epsilon".to_string())),
            ComparisonOp::Le,
            Box::new(Expr::Float(0.04)),
        )),
    );
    assert_eq!(usl_analysis::extract_epsilon(&expr), Some(0.04));
}

#[test]
fn extract_neural_dimensions_from_vars() {
    let mut inputs = HashSet::new();
    let mut outputs = HashSet::new();

    // x0 and y0
    let expr = Expr::And(
        Box::new(Expr::Var("x0".to_string())),
        Box::new(Expr::Var("y0".to_string())),
    );
    usl_analysis::extract_neural_dimensions(&expr, &mut inputs, &mut outputs);

    assert!(inputs.contains(&0));
    assert!(outputs.contains(&0));
}

#[test]
fn extract_neural_dimensions_multiple() {
    let mut inputs = HashSet::new();
    let mut outputs = HashSet::new();

    // x0 >= 0 and x1 <= 1 and y0 > 0.5
    let expr = Expr::And(
        Box::new(Expr::And(
            Box::new(Expr::Compare(
                Box::new(Expr::Var("x0".to_string())),
                ComparisonOp::Ge,
                Box::new(Expr::Float(0.0)),
            )),
            Box::new(Expr::Compare(
                Box::new(Expr::Var("x1".to_string())),
                ComparisonOp::Le,
                Box::new(Expr::Float(1.0)),
            )),
        )),
        Box::new(Expr::Compare(
            Box::new(Expr::Var("y0".to_string())),
            ComparisonOp::Gt,
            Box::new(Expr::Float(0.5)),
        )),
    );
    usl_analysis::extract_neural_dimensions(&expr, &mut inputs, &mut outputs);

    assert!(inputs.contains(&0));
    assert!(inputs.contains(&1));
    assert!(outputs.contains(&0));
    assert_eq!(inputs.len(), 2);
    assert_eq!(outputs.len(), 1);
}

#[test]
fn extract_input_bounds_basic() {
    // x0 >= 0.1 and x0 <= 0.9
    let expr = Expr::And(
        Box::new(Expr::Compare(
            Box::new(Expr::Var("x0".to_string())),
            ComparisonOp::Ge,
            Box::new(Expr::Float(0.1)),
        )),
        Box::new(Expr::Compare(
            Box::new(Expr::Var("x0".to_string())),
            ComparisonOp::Le,
            Box::new(Expr::Float(0.9)),
        )),
    );

    let bounds = usl_analysis::extract_input_bounds(&expr);
    assert_eq!(bounds.len(), 1);
    assert_eq!(bounds[0], (0, 0.1, 0.9));
}

#[test]
fn extract_input_bounds_reversed() {
    // 0.2 <= x0 and 0.8 >= x0
    let expr = Expr::And(
        Box::new(Expr::Compare(
            Box::new(Expr::Float(0.2)),
            ComparisonOp::Le,
            Box::new(Expr::Var("x0".to_string())),
        )),
        Box::new(Expr::Compare(
            Box::new(Expr::Float(0.8)),
            ComparisonOp::Ge,
            Box::new(Expr::Var("x0".to_string())),
        )),
    );

    let bounds = usl_analysis::extract_input_bounds(&expr);
    assert_eq!(bounds.len(), 1);
    assert_eq!(bounds[0], (0, 0.2, 0.8));
}

#[test]
fn extract_input_bounds_multiple_inputs() {
    // x0 >= 0 and x0 <= 1 and x1 >= 0 and x1 <= 1
    let expr = Expr::And(
        Box::new(Expr::And(
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
        )),
        Box::new(Expr::And(
            Box::new(Expr::Compare(
                Box::new(Expr::Var("x1".to_string())),
                ComparisonOp::Ge,
                Box::new(Expr::Float(0.0)),
            )),
            Box::new(Expr::Compare(
                Box::new(Expr::Var("x1".to_string())),
                ComparisonOp::Le,
                Box::new(Expr::Float(1.0)),
            )),
        )),
    );

    let bounds = usl_analysis::extract_input_bounds(&expr);
    assert_eq!(bounds.len(), 2);
    assert!(bounds
        .iter()
        .any(|(idx, l, u)| *idx == 0 && *l == 0.0 && *u == 1.0));
    assert!(bounds
        .iter()
        .any(|(idx, l, u)| *idx == 1 && *l == 0.0 && *u == 1.0));
}

#[test]
fn consolidate_bounds_multiple_constraints() {
    // Same variable constrained multiple times
    let bounds = vec![
        (0, 0.1, f64::INFINITY),     // x0 >= 0.1
        (0, f64::NEG_INFINITY, 0.9), // x0 <= 0.9
        (0, 0.2, f64::INFINITY),     // x0 >= 0.2 (tighter)
        (0, f64::NEG_INFINITY, 0.8), // x0 <= 0.8 (tighter)
    ];

    let consolidated = usl_analysis::consolidate_bounds(&bounds);
    assert_eq!(consolidated.len(), 1);
    assert_eq!(consolidated[0], (0, 0.2, 0.8)); // Takes tightest bounds
}

#[test]
fn generate_zonotope_spec_basic() {
    // Property with bounds: x0 >= 0.2 and x0 <= 0.8
    let expr = Expr::And(
        Box::new(Expr::Compare(
            Box::new(Expr::Var("x0".to_string())),
            ComparisonOp::Ge,
            Box::new(Expr::Float(0.2)),
        )),
        Box::new(Expr::Compare(
            Box::new(Expr::Var("x0".to_string())),
            ComparisonOp::Le,
            Box::new(Expr::Float(0.8)),
        )),
    );

    let spec = make_typed_spec(
        vec![Property::Invariant(Invariant {
            name: "test".to_string(),
            body: expr,
        })],
        vec![],
    );

    let zonotope_spec = zonotope::generate_zonotope_spec(&spec);
    assert!(zonotope_spec.contains("dim,center,epsilon"));
    // Center should be 0.5, radius should be 0.3
    assert!(zonotope_spec.contains("0,0.5,0.3"));
}

#[test]
fn generate_zonotope_spec_from_types() {
    let spec = make_typed_spec(
        vec![Property::Invariant(Invariant {
            name: "test".to_string(),
            body: Expr::Bool(true),
        })],
        vec![TypeDef {
            name: "NeuralInput".to_string(),
            fields: vec![
                Field {
                    name: "x0".to_string(),
                    ty: Type::Named("Real".to_string()),
                },
                Field {
                    name: "x1".to_string(),
                    ty: Type::Named("Real".to_string()),
                },
            ],
        }],
    );

    let zonotope_spec = zonotope::generate_zonotope_spec(&spec);
    assert!(zonotope_spec.contains("dim,center,epsilon"));
    // Default bounds [0, 1] -> center 0.5, radius 0.5
    assert!(zonotope_spec.contains("0,0.5,0.5"));
    assert!(zonotope_spec.contains("1,0.5,0.5"));
}

#[test]
fn select_domain_small_epsilon() {
    // epsilon <= 0.0005 (very small)
    let expr = Expr::Compare(
        Box::new(Expr::Var("epsilon".to_string())),
        ComparisonOp::Le,
        Box::new(Expr::Float(0.0005)),
    );

    let spec = make_typed_spec(
        vec![Property::Invariant(Invariant {
            name: "test".to_string(),
            body: expr,
        })],
        vec![],
    );

    let domain = zonotope::select_domain_for_property(&spec, EranDomain::DeepPoly);
    assert!(matches!(domain, EranDomain::DeepZ));
}

#[test]
fn select_domain_medium_epsilon() {
    // epsilon <= 0.05 (medium)
    let expr = Expr::Compare(
        Box::new(Expr::Var("epsilon".to_string())),
        ComparisonOp::Le,
        Box::new(Expr::Float(0.05)),
    );

    let spec = make_typed_spec(
        vec![Property::Invariant(Invariant {
            name: "test".to_string(),
            body: expr,
        })],
        vec![],
    );

    let domain = zonotope::select_domain_for_property(&spec, EranDomain::DeepZ);
    assert!(matches!(domain, EranDomain::DeepPoly));
}

#[test]
fn select_domain_large_epsilon() {
    // epsilon <= 0.3 (large)
    let expr = Expr::Compare(
        Box::new(Expr::Var("epsilon".to_string())),
        ComparisonOp::Le,
        Box::new(Expr::Float(0.3)),
    );

    let spec = make_typed_spec(
        vec![Property::Invariant(Invariant {
            name: "test".to_string(),
            body: expr,
        })],
        vec![],
    );

    let domain = zonotope::select_domain_for_property(&spec, EranDomain::DeepZ);
    assert!(matches!(domain, EranDomain::RefinePoly));
}

#[test]
fn extract_epsilon_from_spec_test() {
    let expr = Expr::Compare(
        Box::new(Expr::Var("epsilon".to_string())),
        ComparisonOp::Le,
        Box::new(Expr::Float(0.025)),
    );

    let spec = make_typed_spec(
        vec![Property::Theorem(Theorem {
            name: "robustness".to_string(),
            body: expr,
        })],
        vec![],
    );

    let epsilon = usl_analysis::extract_epsilon_from_spec(&spec, 0.01);
    assert_eq!(epsilon, 0.025);
}

#[test]
fn extract_epsilon_from_spec_default() {
    // No epsilon in spec - should return config default
    let spec = make_typed_spec(
        vec![Property::Invariant(Invariant {
            name: "test".to_string(),
            body: Expr::Bool(true),
        })],
        vec![],
    );

    let epsilon = usl_analysis::extract_epsilon_from_spec(&spec, 0.01);
    assert_eq!(epsilon, 0.01); // Default config value
}

#[test]
fn needs_zonotope_spec_with_bounds() {
    let expr = Expr::Compare(
        Box::new(Expr::Var("x0".to_string())),
        ComparisonOp::Le,
        Box::new(Expr::Float(0.5)),
    );

    let spec = make_typed_spec(
        vec![Property::Invariant(Invariant {
            name: "test".to_string(),
            body: expr,
        })],
        vec![],
    );

    assert!(usl_analysis::needs_zonotope_spec(&spec));
}

#[test]
fn needs_zonotope_spec_without_bounds() {
    // Only epsilon, no input bounds
    let expr = Expr::Compare(
        Box::new(Expr::Var("epsilon".to_string())),
        ComparisonOp::Le,
        Box::new(Expr::Float(0.01)),
    );

    let spec = make_typed_spec(
        vec![Property::Invariant(Invariant {
            name: "test".to_string(),
            body: expr,
        })],
        vec![],
    );

    assert!(!usl_analysis::needs_zonotope_spec(&spec));
}

#[test]
fn extract_numeric_value() {
    assert_eq!(
        usl_analysis::extract_numeric_value(&Expr::Float(1.5)),
        Some(1.5)
    );
    assert_eq!(
        usl_analysis::extract_numeric_value(&Expr::Int(42)),
        Some(42.0)
    );
    assert_eq!(
        usl_analysis::extract_numeric_value(&Expr::Neg(Box::new(Expr::Float(3.15)))),
        Some(-3.15)
    );
    assert_eq!(
        usl_analysis::extract_numeric_value(&Expr::Var("x".to_string())),
        None
    );
}

#[test]
fn backend_id_and_supports() {
    let backend = EranBackend::new();
    assert_eq!(backend.id(), BackendId::Eran);
    assert!(backend.supports().contains(&PropertyType::NeuralRobustness));
}

#[test]
fn domain_as_str() {
    assert_eq!(EranDomain::DeepZ.as_str(), "deepzono");
    assert_eq!(EranDomain::DeepPoly.as_str(), "deeppoly");
    assert_eq!(EranDomain::RefinePoly.as_str(), "refinepoly");
    assert_eq!(EranDomain::GpuPoly.as_str(), "gpupoly");
}

#[test]
fn extract_field_access_dimensions() {
    let mut inputs = HashSet::new();
    let mut outputs = HashSet::new();

    // network.x0 and network.y0
    let expr = Expr::And(
        Box::new(Expr::FieldAccess(
            Box::new(Expr::Var("network".to_string())),
            "x0".to_string(),
        )),
        Box::new(Expr::FieldAccess(
            Box::new(Expr::Var("network".to_string())),
            "y0".to_string(),
        )),
    );
    usl_analysis::extract_neural_dimensions(&expr, &mut inputs, &mut outputs);

    assert!(inputs.contains(&0));
    assert!(outputs.contains(&0));
}

#[test]
fn extract_epsilon_from_forall() {
    // forall x: Input . epsilon <= 0.02
    let expr = Expr::ForAll {
        var: "x".to_string(),
        ty: Some(Type::Named("Input".to_string())),
        body: Box::new(Expr::Compare(
            Box::new(Expr::Var("epsilon".to_string())),
            ComparisonOp::Le,
            Box::new(Expr::Float(0.02)),
        )),
    };

    assert_eq!(usl_analysis::extract_epsilon(&expr), Some(0.02));
}

#[test]
fn extract_input_bounds_from_input_named_vars() {
    // input_0 >= 0.0 and input_0 <= 1.0
    let expr = Expr::And(
        Box::new(Expr::Compare(
            Box::new(Expr::Var("input_0".to_string())),
            ComparisonOp::Ge,
            Box::new(Expr::Float(0.0)),
        )),
        Box::new(Expr::Compare(
            Box::new(Expr::Var("input_0".to_string())),
            ComparisonOp::Le,
            Box::new(Expr::Float(1.0)),
        )),
    );

    let bounds = usl_analysis::extract_input_bounds(&expr);
    assert_eq!(bounds.len(), 1);
    assert_eq!(bounds[0], (0, 0.0, 1.0));
}

// ============================================================================
// Model Path Extraction Tests
// ============================================================================

#[test]
fn extract_model_path_from_string_literal() {
    let spec = make_typed_spec(
        vec![Property::Theorem(Theorem {
            name: "test".to_string(),
            body: Expr::String("models/mnist.onnx".to_string()),
        })],
        vec![],
    );
    let path = model::extract_model_path(&spec);
    assert_eq!(path, Some("models/mnist.onnx".to_string()));
}

#[test]
fn extract_model_path_from_comparison() {
    let spec = make_typed_spec(
        vec![Property::Theorem(Theorem {
            name: "test".to_string(),
            body: Expr::Compare(
                Box::new(Expr::Var("model".to_string())),
                ComparisonOp::Eq,
                Box::new(Expr::String("network.onnx".to_string())),
            ),
        })],
        vec![],
    );
    let path = model::extract_model_path(&spec);
    assert_eq!(path, Some("network.onnx".to_string()));
}

#[test]
fn extract_model_path_from_app() {
    let spec = make_typed_spec(
        vec![Property::Theorem(Theorem {
            name: "test".to_string(),
            body: Expr::App(
                "load_model".to_string(),
                vec![Expr::String("classifier.onnx".to_string())],
            ),
        })],
        vec![],
    );
    let path = model::extract_model_path(&spec);
    assert_eq!(path, Some("classifier.onnx".to_string()));
}

#[test]
fn extract_model_path_nested_in_and() {
    let spec = make_typed_spec(
        vec![Property::Theorem(Theorem {
            name: "test".to_string(),
            body: Expr::And(
                Box::new(Expr::Bool(true)),
                Box::new(Expr::String("deep/model.onnx".to_string())),
            ),
        })],
        vec![],
    );
    let path = model::extract_model_path(&spec);
    assert_eq!(path, Some("deep/model.onnx".to_string()));
}

#[test]
fn extract_model_path_none_when_absent() {
    let spec = make_typed_spec(
        vec![Property::Theorem(Theorem {
            name: "test".to_string(),
            body: Expr::Bool(true),
        })],
        vec![],
    );
    let path = model::extract_model_path(&spec);
    assert!(path.is_none());
}

#[test]
fn extract_model_path_pb_format() {
    let spec = make_typed_spec(
        vec![Property::Theorem(Theorem {
            name: "test".to_string(),
            body: Expr::String("tensorflow/model.pb".to_string()),
        })],
        vec![],
    );
    let path = model::extract_model_path(&spec);
    assert_eq!(path, Some("tensorflow/model.pb".to_string()));
}

// ============================================================================
// Counterexample Parsing Tests
// ============================================================================

#[test]
fn parse_counterexample_certified_returns_none() {
    // Certified = safe, no counterexample
    let ce = parsing::parse_counterexample("certified: 100%\n");
    assert!(ce.is_none());
}

#[test]
fn parse_counterexample_verified_returns_none() {
    // Verified = safe, no counterexample
    let ce = parsing::parse_counterexample("verified safe\n");
    assert!(ce.is_none());
}

#[test]
fn parse_counterexample_unsafe_array_format() {
    let output = "not certified\nadversarial: [0.1, 0.2, 0.3]\n";
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
    let output = "unsafe\ninput: [[0.5, 0.6]]\n";
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
    let output = "fail\nx0 = 0.5\nx1 = -0.25\ny0 = 1.0\n";
    let ce = parsing::parse_counterexample(output).unwrap();

    assert_eq!(ce.witness.len(), 3);
    assert!(ce.witness.contains_key("x0"));
    assert!(ce.witness.contains_key("x1"));
    assert!(ce.witness.contains_key("y0"));
}

#[test]
fn parse_counterexample_input_bracket_format() {
    let output = "not verified\ninput[0] = 0.5\ninput[1] = -0.25\noutput[0] = 1.0\n";
    let ce = parsing::parse_counterexample(output).unwrap();

    assert_eq!(ce.witness.len(), 3);
    assert!(ce.witness.contains_key("x0"));
    assert!(ce.witness.contains_key("x1"));
    assert!(ce.witness.contains_key("y0"));
}

#[test]
fn parse_counterexample_standalone_array() {
    let output = "unsafe\n[0.7, 0.8, 0.9]\n";
    let ce = parsing::parse_counterexample(output).unwrap();

    assert_eq!(ce.witness.len(), 3);
    assert!(matches!(
        ce.witness.get("x0"),
        Some(CounterexampleValue::Float { value }) if (*value - 0.7).abs() < 1e-10
    ));
}

#[test]
fn parse_counterexample_preserves_raw() {
    let output = "fail\nx0 = 0.5\n";
    let ce = parsing::parse_counterexample(output).unwrap();

    assert!(ce.raw.is_some());
    assert_eq!(ce.raw.as_ref().unwrap(), output);
}

#[test]
fn parse_counterexample_fallback_to_raw() {
    // Fail but no parseable values
    let output = "unsafe\nsome unparseable format\n";
    let ce = parsing::parse_counterexample(output).unwrap();

    assert!(ce.raw.is_some());
    assert!(ce.witness.is_empty());
}

#[test]
fn parse_counterexample_negative_values() {
    let output = "not certified\nadversarial: [-0.5, 0.3, -0.1]\n";
    let ce = parsing::parse_counterexample(output).unwrap();

    assert_eq!(ce.witness.len(), 3);
    assert!(matches!(
        ce.witness.get("x0"),
        Some(CounterexampleValue::Float { value }) if (*value - (-0.5)).abs() < 1e-10
    ));
}

#[test]
fn parse_counterexample_scientific_notation() {
    let output = "fail\ncounter: [1.5e-10, -2.3e5]\n";
    let ce = parsing::parse_counterexample(output).unwrap();

    assert_eq!(ce.witness.len(), 2);
    assert!(matches!(
        ce.witness.get("x0"),
        Some(CounterexampleValue::Float { value }) if *value < 1e-9
    ));
}

#[test]
fn parse_array_values_simple() {
    let values = parsing::parse_array_values("[1.0, 2.0, 3.0]").unwrap();
    assert_eq!(values, vec![1.0, 2.0, 3.0]);
}

#[test]
fn parse_array_values_nested() {
    let values = parsing::parse_array_values("[[1.0, 2.0]]").unwrap();
    assert_eq!(values, vec![1.0, 2.0]);
}

#[test]
fn parse_array_values_invalid() {
    assert!(parsing::parse_array_values("not an array").is_none());
    assert!(parsing::parse_array_values("[no, numbers]").is_none());
}

#[test]
fn normalize_var_name_input_formats() {
    assert_eq!(parsing::normalize_var_name("x0"), "x0");
    assert_eq!(parsing::normalize_var_name("input[0]"), "x0");
    assert_eq!(parsing::normalize_var_name("Input_5"), "x5");
    assert_eq!(parsing::normalize_var_name("output[0]"), "y0");
    assert_eq!(parsing::normalize_var_name("Output_3"), "y3");
    assert_eq!(parsing::normalize_var_name("y2"), "y2");
}
