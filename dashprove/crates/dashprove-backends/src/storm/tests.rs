//! Tests for Storm backend

use super::*;
use crate::traits::{PropertyType, VerificationStatus};
use std::path::PathBuf;
use std::time::Duration;

#[test]
fn parse_true_result() {
    let status = parsing::parse_output("Result: true\n", "");
    assert!(matches!(status, VerificationStatus::Proven));
}

#[test]
fn parse_false_result() {
    let status = parsing::parse_output("Result: false\n", "");
    assert!(matches!(status, VerificationStatus::Disproven));
}

#[test]
fn parse_probability_result() {
    let status = parsing::parse_output("Result: 0.95\n", "");
    match status {
        VerificationStatus::Partial {
            verified_percentage,
        } => {
            assert!(
                (verified_percentage - 95.0).abs() < 0.1,
                "Expected 95.0, got {}",
                verified_percentage
            );
        }
        _ => panic!("Expected Partial status, got {:?}", status),
    }
}

#[test]
fn parse_probability_result_zero() {
    let status = parsing::parse_output("Result: 0.0\n", "");
    match status {
        VerificationStatus::Partial {
            verified_percentage,
        } => {
            assert!(
                verified_percentage.abs() < 0.1,
                "Expected 0.0, got {}",
                verified_percentage
            );
        }
        _ => panic!("Expected Partial status, got {:?}", status),
    }
}

#[test]
fn parse_probability_result_one() {
    let status = parsing::parse_output("Result: 1.0\n", "");
    match status {
        VerificationStatus::Partial {
            verified_percentage,
        } => {
            assert!(
                (verified_percentage - 100.0).abs() < 0.1,
                "Expected 100.0, got {}",
                verified_percentage
            );
        }
        _ => panic!("Expected Partial status, got {:?}", status),
    }
}

#[test]
fn parse_probability_result_small() {
    let status = parsing::parse_output("Result: 0.0001\n", "");
    match status {
        VerificationStatus::Partial {
            verified_percentage,
        } => {
            assert!(
                (verified_percentage - 0.01).abs() < 0.001,
                "Expected 0.01, got {}",
                verified_percentage
            );
        }
        _ => panic!("Expected Partial status, got {:?}", status),
    }
}

#[test]
fn parse_unknown_output() {
    let status = parsing::parse_output("some random text\nno result here\n", "");
    assert!(
        matches!(status, VerificationStatus::Unknown { .. }),
        "Expected Unknown status for unparseable output"
    );
}

#[test]
fn parse_empty_output() {
    let status = parsing::parse_output("", "");
    assert!(
        matches!(status, VerificationStatus::Unknown { .. }),
        "Expected Unknown status for empty output"
    );
}

#[test]
fn parse_result_with_prefix_text() {
    let stdout = "Storm 1.8.0\nLoading model...\nChecking property...\nResult: true\n";
    let status = parsing::parse_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Proven));
}

#[test]
fn parse_result_with_suffix_text() {
    let stdout = "Result: false\nTime: 0.123s\nMemory: 45MB\n";
    let status = parsing::parse_output(stdout, "");
    assert!(matches!(status, VerificationStatus::Disproven));
}

#[test]
fn parse_result_with_leading_spaces() {
    let status = parsing::parse_output("  Result: 0.75  \n", "");
    match status {
        VerificationStatus::Partial {
            verified_percentage,
        } => {
            assert!(
                (verified_percentage - 75.0).abs() < 0.1,
                "Expected 75.0, got {}",
                verified_percentage
            );
        }
        _ => panic!("Expected Partial status, got {:?}", status),
    }
}

#[test]
fn parse_invalid_probability() {
    // Non-numeric value after "Result:"
    let status = parsing::parse_output("Result: undefined\n", "");
    assert!(
        matches!(status, VerificationStatus::Unknown { .. }),
        "Expected Unknown status for invalid probability"
    );
}

#[test]
fn parse_stderr_ignored() {
    // stdout has result, stderr has errors - should parse stdout
    let status = parsing::parse_output("Result: true\n", "Warning: some error\n");
    assert!(matches!(status, VerificationStatus::Proven));
}

#[test]
fn backend_id() {
    let backend = StormBackend::new();
    assert!(matches!(backend.id(), BackendId::Storm));
}

#[test]
fn backend_supports_probabilistic() {
    let backend = StormBackend::new();
    let supported = backend.supports();
    assert!(
        supported.contains(&PropertyType::Probabilistic),
        "Storm should support Probabilistic property type"
    );
    assert_eq!(
        supported.len(),
        1,
        "Storm should only support Probabilistic"
    );
}

#[test]
fn default_config() {
    let config = StormConfig::default();
    assert!(config.storm_path.is_none());
    assert_eq!(config.timeout, Duration::from_secs(300));
    assert!((config.precision - 1e-6).abs() < 1e-10);
}

#[test]
fn custom_config() {
    let config = StormConfig {
        storm_path: Some(PathBuf::from("/custom/storm")),
        timeout: Duration::from_secs(60),
        precision: 1e-8,
    };
    let backend = StormBackend::with_config(config);
    assert_eq!(
        backend.config.storm_path,
        Some(PathBuf::from("/custom/storm"))
    );
    assert_eq!(backend.config.timeout, Duration::from_secs(60));
}

#[test]
fn generate_prism_contains_expected_elements() {
    // Create a minimal TypedSpec for testing
    let spec = dashprove_usl::Spec {
        types: vec![],
        properties: vec![],
    };
    let typed_spec = dashprove_usl::typecheck(spec).unwrap();

    let prism_output = prism::generate_prism(&typed_spec);

    assert!(
        prism_output.contains("dtmc"),
        "PRISM should specify model type"
    );
    assert!(
        prism_output.contains("module"),
        "PRISM should contain module"
    );
    assert!(
        prism_output.contains("endmodule"),
        "PRISM should close module"
    );
    assert!(
        prism_output.contains("Generated by DashProve"),
        "PRISM should have DashProve header"
    );
}

#[test]
fn generate_prism_has_transitions() {
    let spec = dashprove_usl::Spec {
        types: vec![],
        properties: vec![],
    };
    let typed_spec = dashprove_usl::typecheck(spec).unwrap();

    let prism_output = prism::generate_prism(&typed_spec);

    // Check for transition syntax ([] guard -> prob:update)
    assert!(
        prism_output.contains("->"),
        "PRISM should contain transition arrow"
    );
    assert!(
        prism_output.contains("[]"),
        "PRISM should contain guard brackets"
    );
    assert!(
        prism_output.contains("0.9:") || prism_output.contains("1:"),
        "PRISM should contain probability syntax"
    );
}

#[test]
fn parse_multiline_result_output() {
    let stdout = r#"
Storm 1.8.0

Model type: DTMC (sparse)
States: 11
Transitions: 11

Checking property 1:
    P=? [F s=10]

Result: 0.9999999999999999
"#;
    let status = parsing::parse_output(stdout, "");
    match status {
        VerificationStatus::Partial {
            verified_percentage,
        } => {
            assert!(
                (verified_percentage - 100.0).abs() < 0.01,
                "Expected ~100.0, got {}",
                verified_percentage
            );
        }
        _ => panic!("Expected Partial status, got {:?}", status),
    }
}

// ========== USL to PRISM Compilation Tests ==========

#[test]
fn test_to_prism_ident() {
    assert_eq!(util::to_prism_ident("state"), "state");
    assert_eq!(util::to_prism_ident("myVar"), "myVar");
    assert_eq!(util::to_prism_ident("123var"), "_123var");
    assert_eq!(util::to_prism_ident("var-name"), "var_name");
    assert_eq!(util::to_prism_ident("module"), "module_var");
    assert_eq!(util::to_prism_ident("true"), "true_var");
}

#[test]
fn test_generate_prism_with_types() {
    use dashprove_usl::ast::{Field, Type, TypeDef};

    let spec = dashprove_usl::Spec {
        types: vec![TypeDef {
            name: "System".to_string(),
            fields: vec![
                Field {
                    name: "count".to_string(),
                    ty: Type::Named("Int".to_string()),
                },
                Field {
                    name: "active".to_string(),
                    ty: Type::Named("Bool".to_string()),
                },
            ],
        }],
        properties: vec![],
    };
    let typed_spec = dashprove_usl::typecheck(spec).unwrap();

    let prism_output = prism::generate_prism(&typed_spec);

    // Should include state variables from type fields
    assert!(
        prism_output.contains("count") || prism_output.contains("active"),
        "PRISM should include state variables from types: {}",
        prism_output
    );
}

#[test]
fn test_generate_prism_reachability_property() {
    use dashprove_usl::ast::{ComparisonOp, Expr, Probabilistic, Property};

    let spec = dashprove_usl::Spec {
        types: vec![],
        properties: vec![Property::Probabilistic(Probabilistic {
            name: "reach_goal".to_string(),
            condition: Expr::App("goal".to_string(), vec![]),
            comparison: ComparisonOp::Ge,
            bound: 0.99,
        })],
    };
    let typed_spec = dashprove_usl::typecheck(spec).unwrap();

    let prism_output = prism::generate_prism(&typed_spec);
    let pctl_output = pctl::generate_pctl_property(&typed_spec);

    // Should have label for goal
    assert!(
        prism_output.contains("label") && prism_output.contains("goal"),
        "PRISM should have goal label: {}",
        prism_output
    );

    // PCTL should reference the bound
    assert!(
        pctl_output.contains(">=0.99") || pctl_output.contains(">= 0.99"),
        "PCTL should have probability bound: {}",
        pctl_output
    );
}

#[test]
fn test_generate_prism_response_time() {
    use dashprove_usl::ast::{ComparisonOp, Expr, Probabilistic, Property};

    // probability(response_time < 100) >= 0.95
    let spec = dashprove_usl::Spec {
        types: vec![],
        properties: vec![Property::Probabilistic(Probabilistic {
            name: "response_bound".to_string(),
            condition: Expr::Compare(
                Box::new(Expr::Var("response_time".to_string())),
                ComparisonOp::Lt,
                Box::new(Expr::Int(100)),
            ),
            comparison: ComparisonOp::Ge,
            bound: 0.95,
        })],
    };
    let typed_spec = dashprove_usl::typecheck(spec).unwrap();

    let prism_output = prism::generate_prism(&typed_spec);

    // Should have state variable for response_time
    assert!(
        prism_output.contains("response_time"),
        "PRISM should have response_time state variable: {}",
        prism_output
    );
    // Should have reward structure for time
    assert!(
        prism_output.contains("rewards") || prism_output.contains("time"),
        "PRISM could include reward structure for time-based properties"
    );
}

#[test]
fn test_compile_pctl_eventually() {
    use dashprove_usl::ast::Expr;

    // eventually(done)
    let expr = Expr::App(
        "eventually".to_string(),
        vec![Expr::Var("done".to_string())],
    );

    let pctl_output = pctl::compile_pctl_condition(&expr);

    assert!(
        pctl_output.contains("F"),
        "Eventually should compile to F (finally): {}",
        pctl_output
    );
}

#[test]
fn test_compile_pctl_always() {
    use dashprove_usl::ast::Expr;

    // always(safe)
    let expr = Expr::App("always".to_string(), vec![Expr::Var("safe".to_string())]);

    let pctl_output = pctl::compile_pctl_condition(&expr);

    assert!(
        pctl_output.contains("G"),
        "Always should compile to G (globally): {}",
        pctl_output
    );
}

#[test]
fn test_compile_pctl_until() {
    use dashprove_usl::ast::Expr;

    // until(waiting, served)
    let expr = Expr::App(
        "until".to_string(),
        vec![
            Expr::Var("waiting".to_string()),
            Expr::Var("served".to_string()),
        ],
    );

    let pctl_output = pctl::compile_pctl_condition(&expr);

    assert!(
        pctl_output.contains("U"),
        "Until should compile to U: {}",
        pctl_output
    );
    assert!(
        pctl_output.contains("waiting"),
        "Until should include left operand: {}",
        pctl_output
    );
    assert!(
        pctl_output.contains("served"),
        "Until should include right operand: {}",
        pctl_output
    );
}

#[test]
fn test_compile_pctl_comparison() {
    use dashprove_usl::ast::{ComparisonOp, Expr};

    // count >= 5
    let expr = Expr::Compare(
        Box::new(Expr::Var("count".to_string())),
        ComparisonOp::Ge,
        Box::new(Expr::Int(5)),
    );

    let pctl_output = pctl::compile_pctl_condition(&expr);

    assert!(
        pctl_output.contains("count"),
        "Comparison should include variable: {}",
        pctl_output
    );
    assert!(
        pctl_output.contains(">="),
        "Comparison should include operator: {}",
        pctl_output
    );
    assert!(
        pctl_output.contains("5"),
        "Comparison should include bound: {}",
        pctl_output
    );
}

#[test]
fn test_compile_pctl_and() {
    use dashprove_usl::ast::Expr;

    // a and b
    let expr = Expr::And(
        Box::new(Expr::Var("a".to_string())),
        Box::new(Expr::Var("b".to_string())),
    );

    let pctl_output = pctl::compile_pctl_condition(&expr);

    assert!(
        pctl_output.contains("&"),
        "And should compile to &: {}",
        pctl_output
    );
    assert!(
        pctl_output.contains("a"),
        "And should include left operand: {}",
        pctl_output
    );
    assert!(
        pctl_output.contains("b"),
        "And should include right operand: {}",
        pctl_output
    );
}

#[test]
fn test_compile_pctl_or() {
    use dashprove_usl::ast::Expr;

    // a or b
    let expr = Expr::Or(
        Box::new(Expr::Var("a".to_string())),
        Box::new(Expr::Var("b".to_string())),
    );

    let pctl_output = pctl::compile_pctl_condition(&expr);

    assert!(
        pctl_output.contains("|"),
        "Or should compile to |: {}",
        pctl_output
    );
}

#[test]
fn test_compile_pctl_not() {
    use dashprove_usl::ast::Expr;

    // not error
    let expr = Expr::Not(Box::new(Expr::Var("error".to_string())));

    let pctl_output = pctl::compile_pctl_condition(&expr);

    assert!(
        pctl_output.contains("!"),
        "Not should compile to !: {}",
        pctl_output
    );
    assert!(
        pctl_output.contains("error"),
        "Not should include operand: {}",
        pctl_output
    );
}

#[test]
fn test_compile_pctl_implies() {
    use dashprove_usl::ast::Expr;

    // request implies response
    let expr = Expr::Implies(
        Box::new(Expr::Var("request".to_string())),
        Box::new(Expr::Var("response".to_string())),
    );

    let pctl_output = pctl::compile_pctl_condition(&expr);

    // implies translates to !P | Q
    assert!(
        pctl_output.contains("!") && pctl_output.contains("|"),
        "Implies should compile to !P | Q: {}",
        pctl_output
    );
}

#[test]
fn test_generate_mdp_from_quantifiers() {
    use dashprove_usl::ast::{ComparisonOp, Expr, Probabilistic, Property};

    // Property with existential quantifier → MDP
    let spec = dashprove_usl::Spec {
        types: vec![],
        properties: vec![Property::Probabilistic(Probabilistic {
            name: "mdp_property".to_string(),
            condition: Expr::Exists {
                var: "action".to_string(),
                ty: None,
                body: Box::new(Expr::App("success".to_string(), vec![])),
            },
            comparison: ComparisonOp::Ge,
            bound: 0.9,
        })],
    };
    let typed_spec = dashprove_usl::typecheck(spec).unwrap();

    let prism_output = prism::generate_prism(&typed_spec);

    assert!(
        prism_output.contains("mdp"),
        "Existential quantifier should trigger MDP model type: {}",
        prism_output
    );
}

#[test]
fn test_generate_dtmc_without_quantifiers() {
    use dashprove_usl::ast::{ComparisonOp, Expr, Probabilistic, Property};

    // Simple property without quantifiers → DTMC
    let spec = dashprove_usl::Spec {
        types: vec![],
        properties: vec![Property::Probabilistic(Probabilistic {
            name: "dtmc_property".to_string(),
            condition: Expr::App("done".to_string(), vec![]),
            comparison: ComparisonOp::Ge,
            bound: 0.99,
        })],
    };
    let typed_spec = dashprove_usl::typecheck(spec).unwrap();

    let prism_output = prism::generate_prism(&typed_spec);

    assert!(
        prism_output.contains("dtmc"),
        "Simple property should generate DTMC model: {}",
        prism_output
    );
}

#[test]
fn test_model_without_probabilistic_properties() {
    let spec = dashprove_usl::Spec {
        types: vec![],
        properties: vec![],
    };
    let typed_spec = dashprove_usl::typecheck(spec).unwrap();

    let prism_output = prism::generate_prism(&typed_spec);
    let pctl_output = pctl::generate_pctl_property(&typed_spec);

    // Should still generate valid PRISM model with defaults
    assert!(
        prism_output.contains("dtmc"),
        "Should default to DTMC: {}",
        prism_output
    );
    assert!(
        prism_output.contains("module main"),
        "Should have main module: {}",
        prism_output
    );
    assert!(
        prism_output.contains("state"),
        "Should have default state variable: {}",
        prism_output
    );

    // Should have default property
    assert!(
        pctl_output.contains("P=?") || pctl_output.contains("F"),
        "Should have default reachability property: {}",
        pctl_output
    );
}

#[test]
fn test_extract_state_bounds_from_comparison() {
    use dashprove_usl::ast::{ComparisonOp, Expr, Probabilistic, Property};

    // probability(count < 100) >= 0.8
    let spec = dashprove_usl::Spec {
        types: vec![],
        properties: vec![Property::Probabilistic(Probabilistic {
            name: "bounded_count".to_string(),
            condition: Expr::Compare(
                Box::new(Expr::Var("count".to_string())),
                ComparisonOp::Lt,
                Box::new(Expr::Int(100)),
            ),
            comparison: ComparisonOp::Ge,
            bound: 0.8,
        })],
    };
    let typed_spec = dashprove_usl::typecheck(spec).unwrap();

    let prism_output = prism::generate_prism(&typed_spec);

    // Should extract bound from comparison and include in state variable range
    assert!(
        prism_output.contains("count"),
        "Should have count state variable: {}",
        prism_output
    );
    // The max should be at least 100
    assert!(
        prism_output.contains("100")
            || prism_output.contains("..100")
            || prism_output.contains("MAX_STATE"),
        "Should have appropriate range for count"
    );
}

#[test]
fn test_compile_pctl_field_access() {
    use dashprove_usl::ast::Expr;

    // system.counter
    let expr = Expr::FieldAccess(
        Box::new(Expr::Var("system".to_string())),
        "counter".to_string(),
    );

    let pctl_output = pctl::compile_pctl_condition(&expr);

    assert!(
        pctl_output.contains("counter"),
        "Field access should extract field name: {}",
        pctl_output
    );
}

#[test]
fn test_compile_pctl_arithmetic() {
    use dashprove_usl::ast::{BinaryOp, Expr};

    // x + 1
    let expr = Expr::Binary(
        Box::new(Expr::Var("x".to_string())),
        BinaryOp::Add,
        Box::new(Expr::Int(1)),
    );

    let pctl_output = pctl::compile_pctl_condition(&expr);

    assert!(
        pctl_output.contains("+"),
        "Addition should compile to +: {}",
        pctl_output
    );
    assert!(
        pctl_output.contains("x"),
        "Should include variable: {}",
        pctl_output
    );
    assert!(
        pctl_output.contains("1"),
        "Should include constant: {}",
        pctl_output
    );
}

#[test]
fn test_generate_pctl_multiple_properties() {
    use dashprove_usl::ast::{ComparisonOp, Expr, Probabilistic, Property};

    let spec = dashprove_usl::Spec {
        types: vec![],
        properties: vec![
            Property::Probabilistic(Probabilistic {
                name: "prop1".to_string(),
                condition: Expr::App("goal1".to_string(), vec![]),
                comparison: ComparisonOp::Ge,
                bound: 0.9,
            }),
            Property::Probabilistic(Probabilistic {
                name: "prop2".to_string(),
                condition: Expr::App("goal2".to_string(), vec![]),
                comparison: ComparisonOp::Le,
                bound: 0.1,
            }),
        ],
    };
    let typed_spec = dashprove_usl::typecheck(spec).unwrap();

    let pctl_output = pctl::generate_pctl_property(&typed_spec);

    // Should combine multiple properties
    assert!(
        pctl_output.contains(">=0.9") || pctl_output.contains(">= 0.9"),
        "Should have first property bound: {}",
        pctl_output
    );
    assert!(
        pctl_output.contains("<=0.1") || pctl_output.contains("<= 0.1"),
        "Should have second property bound: {}",
        pctl_output
    );
    assert!(
        pctl_output.contains("&"),
        "Should combine with &: {}",
        pctl_output
    );
}
