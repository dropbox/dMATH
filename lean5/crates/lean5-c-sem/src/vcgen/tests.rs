//! Tests for verification condition generation
//!
//! This module contains unit tests for the VCGen implementation.

use super::*;
use crate::expr::Initializer;
use crate::expr::UnaryOp;
use crate::spec::{FuncSpec, Location};
use crate::stmt::{CaseLabel, FuncDef, FuncParam, StorageClass, VarDecl};
use crate::types::CType;

#[test]
fn test_wp_skip() {
    let mut vcgen = VCGen::new();
    let postcond = Spec::ge(Spec::var("x"), Spec::int(0));
    let wp = vcgen.wp_stmt(&CStmt::Empty, &postcond, None);
    assert_eq!(wp, postcond);
}

#[test]
fn test_wp_assignment() {
    let mut vcgen = VCGen::new();
    // wp(x = 1, x >= 0) = 1 >= 0 = true
    let stmt = CStmt::Expr(CExpr::assign(CExpr::var("x"), CExpr::int(1)));
    let postcond = Spec::ge(Spec::var("x"), Spec::int(0));
    let wp = vcgen.wp_stmt(&stmt, &postcond, None);

    // After substitution: 1 >= 0
    match wp {
        Spec::BinOp {
            op: BinOp::Ge,
            left,
            right,
        } => {
            // Check it substituted x with 1
            assert!(matches!(*left, Spec::Int(1)));
            assert!(matches!(*right, Spec::Int(0)));
        }
        _ => panic!("Expected BinOp Ge, got {wp:?}"),
    }
}

#[test]
fn test_wp_if() {
    let mut vcgen = VCGen::new();
    let stmt = CStmt::if_stmt(
        CExpr::binop(BinOp::Lt, CExpr::var("x"), CExpr::int(0)),
        CStmt::Expr(CExpr::assign(CExpr::var("y"), CExpr::int(1))),
    );
    let postcond = Spec::ge(Spec::var("y"), Spec::int(0));
    let wp = vcgen.wp_stmt(&stmt, &postcond, None);

    // Should be: (x < 0 → y[1/y] >= 0) ∧ (x >= 0 → y >= 0)
    assert!(matches!(wp, Spec::And(_)));
}

#[test]
fn test_gen_function_abs() {
    let mut vcgen = VCGen::new();

    // abs function: if (n < 0) return -n; else return n;
    let func = FuncDef {
        name: "abs".into(),
        return_type: CType::int(),
        params: vec![FuncParam {
            name: "n".into(),
            ty: CType::int(),
        }],
        body: Box::new(CStmt::if_else(
            CExpr::binop(BinOp::Lt, CExpr::var("n"), CExpr::int(0)),
            CStmt::return_stmt(Some(CExpr::unary(
                crate::expr::UnaryOp::Neg,
                CExpr::var("n"),
            ))),
            CStmt::return_stmt(Some(CExpr::var("n"))),
        )),
        variadic: false,
        storage: StorageClass::Auto,
    };

    let spec = FuncSpec {
        requires: vec![Spec::True],
        ensures: vec![Spec::ge(Spec::result(), Spec::int(0))],
        ..Default::default()
    };

    let vcs = vcgen.gen_function(&func, &spec);

    // Should generate at least one VC for the function contract
    assert!(!vcs.is_empty());
    assert!(vcs.iter().any(|vc| vc.kind == VCKind::Postcondition));
}

#[test]
fn test_division_vc() {
    let mut vcgen = VCGen::new();
    let expr = CExpr::div(CExpr::var("a"), CExpr::var("b"));
    let postcond = Spec::True;
    vcgen.wp_expr(&expr, &postcond);

    let vcs = vcgen.get_vcs();
    assert!(vcs.iter().any(|vc| vc.kind == VCKind::NoUB));
}

#[test]
fn test_pointer_deref_vc() {
    let mut vcgen = VCGen::new();
    let expr = CExpr::UnaryOp {
        op: crate::expr::UnaryOp::Deref,
        operand: Box::new(CExpr::var("p")),
    };
    let postcond = Spec::True;
    vcgen.wp_expr(&expr, &postcond);

    let vcs = vcgen.get_vcs();
    assert!(vcs.iter().any(|vc| vc.kind == VCKind::MemorySafety));
}

#[test]
fn test_loop_invariant_vc() {
    let mut vcgen = VCGen::new();

    let loop_spec = LoopSpec {
        invariant: vec![Spec::ge(Spec::var("i"), Spec::int(0))],
        variant: Some(Spec::var("n")),
        ..Default::default()
    };

    let while_stmt = CStmt::While {
        cond: CExpr::binop(BinOp::Lt, CExpr::var("i"), CExpr::var("n")),
        body: Box::new(CStmt::Expr(CExpr::BinOp {
            op: BinOp::AddAssign,
            left: Box::new(CExpr::var("i")),
            right: Box::new(CExpr::int(1)),
        })),
    };

    let postcond = Spec::True;
    vcgen.wp_stmt(&while_stmt, &postcond, Some(&loop_spec));

    let vcs = vcgen.get_vcs();

    // Should have loop preservation and variant VCs
    assert!(vcs
        .iter()
        .any(|vc| vc.kind == VCKind::LoopInvariantPreserved));
    assert!(vcs.iter().any(|vc| vc.kind == VCKind::LoopVariantDecreases));
    assert!(vcs
        .iter()
        .any(|vc| vc.kind == VCKind::LoopVariantNonNegative));
}

#[test]
fn test_wp_while_auto_inference() {
    // Test that wp_while automatically infers invariants when no spec provided
    let mut vcgen = VCGen::new();

    // for (i = 0; i < n; i++) - without explicit loop spec
    let while_stmt = CStmt::While {
        cond: CExpr::binop(BinOp::Lt, CExpr::var("i"), CExpr::var("n")),
        body: Box::new(CStmt::Expr(CExpr::UnaryOp {
            op: crate::expr::UnaryOp::PostInc,
            operand: Box::new(CExpr::var("i")),
        })),
    };

    let postcond = Spec::ge(Spec::var("i"), Spec::int(0));
    let wp = vcgen.wp_stmt(&while_stmt, &postcond, None); // No loop spec!

    let vcs = vcgen.get_vcs();

    // Should have generated VCs with inferred invariant
    // (Not just a warning VC)
    let has_inferred = vcs.iter().any(|vc| {
        vc.kind == VCKind::LoopInvariantPreserved
            || (vc.kind == VCKind::LoopInvariantEntry && !vc.description.contains("annotation"))
    });

    assert!(
        has_inferred,
        "Should have generated VCs with inferred invariant. Got: {:?}",
        vcs.iter().map(|vc| &vc.description).collect::<Vec<_>>()
    );

    // The WP should be the inferred invariant, not the postcondition
    assert!(
        !matches!(wp, Spec::BinOp { op: BinOp::Ge, .. }),
        "WP should be inferred invariant, not postcondition"
    );
}

#[test]
fn test_wp_while_search_inference() {
    // Test automatic inference for search loop pattern
    let mut vcgen = VCGen::new();

    // while (!found && i < n) { if (arr[i] == target) found = 1; i++; }
    let while_stmt = CStmt::While {
        cond: CExpr::BinOp {
            op: BinOp::LogAnd,
            left: Box::new(CExpr::UnaryOp {
                op: crate::expr::UnaryOp::LogNot,
                operand: Box::new(CExpr::var("found")),
            }),
            right: Box::new(CExpr::binop(BinOp::Lt, CExpr::var("i"), CExpr::var("n"))),
        },
        body: Box::new(CStmt::Block(vec![
            CStmt::If {
                cond: CExpr::BinOp {
                    op: BinOp::Eq,
                    left: Box::new(CExpr::Index {
                        array: Box::new(CExpr::var("arr")),
                        index: Box::new(CExpr::var("i")),
                    }),
                    right: Box::new(CExpr::var("target")),
                },
                then_stmt: Box::new(CStmt::Expr(CExpr::assign(
                    CExpr::var("found"),
                    CExpr::int(1),
                ))),
                else_stmt: None,
            },
            CStmt::Expr(CExpr::UnaryOp {
                op: crate::expr::UnaryOp::PostInc,
                operand: Box::new(CExpr::var("i")),
            }),
        ])),
    };

    let postcond = Spec::True;
    vcgen.wp_stmt(&while_stmt, &postcond, None); // No loop spec!

    let vcs = vcgen.get_vcs();

    // Should have generated VCs with inferred invariant
    let has_preserved = vcs
        .iter()
        .any(|vc| vc.kind == VCKind::LoopInvariantPreserved);
    assert!(
        has_preserved,
        "Should have generated preservation VC for search loop. Got: {:?}",
        vcs.iter().map(|vc| &vc.description).collect::<Vec<_>>()
    );
}

#[test]
fn test_substitution() {
    let vcgen = VCGen::new();

    // x + 1 >= 0, substitute 5 for x
    let spec = Spec::ge(
        Spec::binop(BinOp::Add, Spec::var("x"), Spec::int(1)),
        Spec::int(0),
    );
    let result = vcgen.substitute(&spec, "x", &CExpr::int(5));

    // Should get 5 + 1 >= 0
    match result {
        Spec::BinOp {
            op: BinOp::Ge,
            left,
            ..
        } => match *left {
            Spec::BinOp {
                left: l, right: r, ..
            } => {
                assert!(matches!(*l, Spec::Int(5)));
                assert!(matches!(*r, Spec::Int(1)));
            }
            _ => panic!("Expected binop"),
        },
        _ => panic!("Expected ge"),
    }
}

#[test]
fn test_result_substitution() {
    let vcgen = VCGen::new();

    // \result >= 0, substitute n for \result
    let spec = Spec::ge(Spec::result(), Spec::int(0));
    let result = vcgen.substitute_result(&spec, &CExpr::var("n"));

    // Should get n >= 0
    match result {
        Spec::BinOp {
            op: BinOp::Ge,
            left,
            right,
        } => {
            assert!(matches!(*left, Spec::Var(ref s) if s == "n"));
            assert!(matches!(*right, Spec::Int(0)));
        }
        _ => panic!("Expected ge"),
    }
}

#[test]
fn test_vc_to_lean5() {
    let vc = VC {
        description: "Test".into(),
        obligation: Spec::ge(Spec::var("x"), Spec::int(0)),
        location: None,
        kind: VCKind::Assertion,
    };

    let lean_expr = vc_to_lean5(&vc);

    // Should produce a Lean5 expression (App for GE.ge)
    assert!(matches!(lean_expr, lean5_kernel::Expr::App(_, _)));
}

#[test]
fn test_block_wp() {
    let mut vcgen = VCGen::new();

    // { x = 1; y = x + 1; }
    // wp(block, y >= 0) = 1 + 1 >= 0
    let block = CStmt::Block(vec![
        CStmt::Expr(CExpr::assign(CExpr::var("x"), CExpr::int(1))),
        CStmt::Expr(CExpr::assign(
            CExpr::var("y"),
            CExpr::add(CExpr::var("x"), CExpr::int(1)),
        )),
    ]);
    let postcond = Spec::ge(Spec::var("y"), Spec::int(0));
    let _wp = vcgen.wp_stmt(&block, &postcond, None);

    // The WP should substitute through the block
    // Final result after both substitutions should reference the literal values
}

#[test]
fn test_decl_with_init() {
    let mut vcgen = VCGen::new();

    let decl = CStmt::Decl(VarDecl {
        name: "x".into(),
        ty: CType::int(),
        storage: StorageClass::Auto,
        init: Some(Initializer::Expr(CExpr::int(42))),
    });

    let postcond = Spec::ge(Spec::var("x"), Spec::int(0));
    let wp = vcgen.wp_stmt(&decl, &postcond, None);

    // wp(int x = 42, x >= 0) = 42 >= 0
    match wp {
        Spec::BinOp {
            op: BinOp::Ge,
            left,
            ..
        } => {
            assert!(matches!(*left, Spec::Int(42)));
        }
        _ => panic!("Expected ge"),
    }
}

#[test]
fn test_invariant_inference_counter_loop() {
    let mut inference = InvariantInference::new();
    let context = InferenceContext::new();

    // for (i = 0; i < n; i++)
    let cond = CExpr::binop(BinOp::Lt, CExpr::var("i"), CExpr::var("n"));
    let body = CStmt::Expr(CExpr::UnaryOp {
        op: crate::expr::UnaryOp::PostInc,
        operand: Box::new(CExpr::var("i")),
    });

    let invariants = inference.infer_while_invariant(&cond, &body, &context);

    // Should infer: 0 <= i <= n
    assert!(
        !invariants.is_empty(),
        "Should infer at least one invariant"
    );

    // Check we got a bound invariant
    let has_lower_bound = invariants
        .iter()
        .any(|inv| matches!(inv, Spec::BinOp { op: BinOp::Ge, .. } | Spec::And(_)));
    assert!(has_lower_bound, "Should infer lower bound invariant");
}

#[test]
fn test_invariant_inference_decrementing_loop() {
    let mut inference = InvariantInference::new();
    let context = InferenceContext::new();

    // while (i > 0) i--
    let cond = CExpr::binop(BinOp::Gt, CExpr::var("i"), CExpr::int(0));
    let body = CStmt::Expr(CExpr::UnaryOp {
        op: crate::expr::UnaryOp::PostDec,
        operand: Box::new(CExpr::var("i")),
    });

    let invariants = inference.infer_while_invariant(&cond, &body, &context);

    // Should infer: i >= 0 (for decreasing vars)
    assert!(
        !invariants.is_empty(),
        "Should infer invariant for decreasing loop"
    );
}

#[test]
fn test_invariant_inference_with_context() {
    let mut inference = InvariantInference::new();
    let mut context = InferenceContext::new();

    // Set initial value
    context.set_initial("i", Spec::int(0));

    let cond = CExpr::binop(BinOp::Lt, CExpr::var("i"), CExpr::int(10));
    let body = CStmt::Expr(CExpr::BinOp {
        op: BinOp::AddAssign,
        left: Box::new(CExpr::var("i")),
        right: Box::new(CExpr::int(1)),
    });

    let invariants = inference.infer_while_invariant(&cond, &body, &context);

    // Should infer invariant using initial value
    assert!(!invariants.is_empty());
}

#[test]
fn test_inference_context() {
    let mut ctx = InferenceContext::new();
    ctx.set_initial("x", Spec::int(5));
    ctx.preconditions
        .push(Spec::ge(Spec::var("n"), Spec::int(0)));

    assert!(ctx.initial_values.contains_key("x"));
    assert_eq!(ctx.preconditions.len(), 1);
}

#[test]
fn test_accumulator_pattern_add_assign() {
    let inference = InvariantInference::new();
    let mut context = InferenceContext::new();
    context.set_initial("sum", Spec::int(0));

    // sum += arr[i]
    let body = CStmt::Expr(CExpr::BinOp {
        op: BinOp::AddAssign,
        left: Box::new(CExpr::var("sum")),
        right: Box::new(CExpr::Index {
            array: Box::new(CExpr::var("arr")),
            index: Box::new(CExpr::var("i")),
        }),
    });

    let result = inference.detect_sum_pattern(&body, &context);
    assert!(result.is_some(), "Should detect sum accumulator pattern");

    // Should infer sum >= initial (0)
    let inv = result.unwrap();
    match inv {
        Spec::BinOp {
            op: BinOp::Ge,
            left,
            right,
        } => {
            assert!(matches!(*left, Spec::Var { .. }));
            assert!(matches!(*right, Spec::Int(0)));
        }
        _ => panic!("Expected ge invariant, got {inv:?}"),
    }
}

#[test]
fn test_accumulator_pattern_mul_assign() {
    let inference = InvariantInference::new();
    let mut context = InferenceContext::new();
    context.set_initial("product", Spec::int(1));

    // product *= arr[i]
    let body = CStmt::Expr(CExpr::BinOp {
        op: BinOp::MulAssign,
        left: Box::new(CExpr::var("product")),
        right: Box::new(CExpr::var("x")),
    });

    let result = inference.detect_sum_pattern(&body, &context);
    assert!(
        result.is_some(),
        "Should detect product accumulator pattern"
    );
}

#[test]
fn test_accumulator_pattern_sum_form() {
    let inference = InvariantInference::new();
    let context = InferenceContext::new();

    // sum = sum + x (alternative form)
    let body = CStmt::Expr(CExpr::BinOp {
        op: BinOp::Assign,
        left: Box::new(CExpr::var("sum")),
        right: Box::new(CExpr::BinOp {
            op: BinOp::Add,
            left: Box::new(CExpr::var("sum")),
            right: Box::new(CExpr::var("x")),
        }),
    });

    let result = inference.detect_sum_pattern(&body, &context);
    assert!(result.is_some(), "Should detect sum = sum + x pattern");
}

#[test]
fn test_search_pattern() {
    let inference = InvariantInference::new();

    // while (!found && i < n)
    let cond = CExpr::BinOp {
        op: BinOp::LogAnd,
        left: Box::new(CExpr::UnaryOp {
            op: crate::expr::UnaryOp::LogNot,
            operand: Box::new(CExpr::var("found")),
        }),
        right: Box::new(CExpr::BinOp {
            op: BinOp::Lt,
            left: Box::new(CExpr::var("i")),
            right: Box::new(CExpr::var("n")),
        }),
    };

    // if (arr[i] == target) found = 1
    let body = CStmt::If {
        cond: CExpr::BinOp {
            op: BinOp::Eq,
            left: Box::new(CExpr::Index {
                array: Box::new(CExpr::var("arr")),
                index: Box::new(CExpr::var("i")),
            }),
            right: Box::new(CExpr::var("target")),
        },
        then_stmt: Box::new(CStmt::Expr(CExpr::BinOp {
            op: BinOp::Assign,
            left: Box::new(CExpr::var("found")),
            right: Box::new(CExpr::int(1)),
        })),
        else_stmt: None,
    };

    let result = inference.detect_search_pattern(&cond, &body);
    assert!(result.is_some(), "Should detect search loop pattern");

    // Should infer bounds on found and i
    let inv = result.unwrap();
    assert!(
        matches!(inv, Spec::And(_)),
        "Should be conjunction of invariants"
    );
}

#[test]
fn test_bitwise_accumulator() {
    let inference = InvariantInference::new();
    let context = InferenceContext::new();

    // flags |= new_flag
    let body = CStmt::Expr(CExpr::BinOp {
        op: BinOp::BitOrAssign,
        left: Box::new(CExpr::var("flags")),
        right: Box::new(CExpr::var("new_flag")),
    });

    let result = inference.detect_sum_pattern(&body, &context);
    assert!(
        result.is_some(),
        "Should detect bitwise or accumulator pattern"
    );
}

#[test]
fn test_min_pattern_detection() {
    let inference = InvariantInference::new();
    let mut context = InferenceContext::new();
    context.set_initial("min", Spec::var("init_value"));

    // if (arr[i] < min) min = arr[i];
    // This is a min accumulator pattern
    let arr_i = CExpr::index(CExpr::var("arr"), CExpr::var("i"));

    let body = CStmt::If {
        cond: CExpr::binop(BinOp::Lt, arr_i.clone(), CExpr::var("min")),
        then_stmt: Box::new(CStmt::Expr(CExpr::assign(CExpr::var("min"), arr_i))),
        else_stmt: None,
    };

    let result = inference.detect_sum_pattern(&body, &context);
    assert!(result.is_some(), "Should detect min pattern invariant");

    // Min pattern: min <= init_value (monotonically decreasing)
    let inv = result.unwrap();
    // Check that it's a <= comparison (BinOp with Le)
    assert!(
        matches!(inv, Spec::BinOp { op: BinOp::Le, .. }),
        "Min invariant should be <= comparison, got {inv:?}"
    );
}

#[test]
fn test_max_pattern_detection() {
    let inference = InvariantInference::new();
    let mut context = InferenceContext::new();
    context.set_initial("max", Spec::var("init_value"));

    // if (arr[i] > max) max = arr[i];
    // This is a max accumulator pattern
    let arr_i = CExpr::index(CExpr::var("arr"), CExpr::var("i"));

    let body = CStmt::If {
        cond: CExpr::binop(BinOp::Gt, arr_i.clone(), CExpr::var("max")),
        then_stmt: Box::new(CStmt::Expr(CExpr::assign(CExpr::var("max"), arr_i))),
        else_stmt: None,
    };

    let result = inference.detect_sum_pattern(&body, &context);
    assert!(result.is_some(), "Should detect max pattern invariant");

    // Max pattern: max >= init_value (monotonically increasing)
    let inv = result.unwrap();
    assert!(
        matches!(inv, Spec::BinOp { op: BinOp::Ge, .. }),
        "Max invariant should be >= comparison, got {inv:?}"
    );
}

#[test]
fn test_min_pattern_in_block() {
    let inference = InvariantInference::new();
    let context = InferenceContext::new();

    // Block containing: if (value < min) { min = value; }
    let body = CStmt::Block(vec![CStmt::If {
        cond: CExpr::binop(BinOp::Lt, CExpr::var("value"), CExpr::var("min")),
        then_stmt: Box::new(CStmt::Block(vec![CStmt::Expr(CExpr::assign(
            CExpr::var("min"),
            CExpr::var("value"),
        ))])),
        else_stmt: None,
    }]);

    // detect_sum_pattern internally calls find_accumulator_pattern
    // which should find the min pattern in the nested if
    let _result = inference.detect_sum_pattern(&body, &context);
    // Without initial value, min pattern returns None
    // So we test that it at least doesn't crash
    // With initial value it would return Some
    // Let's test with initial value
    let mut context_with_init = InferenceContext::new();
    context_with_init.set_initial("min", Spec::int(100));
    let result2 = inference.detect_sum_pattern(&body, &context_with_init);
    assert!(
        result2.is_some(),
        "Should detect min pattern with initial value"
    );
}

#[test]
fn test_exprs_equal() {
    let inference = InvariantInference::new();

    // Test simple variable equality
    assert!(inference.exprs_equal(&CExpr::var("x"), &CExpr::var("x")));
    assert!(!inference.exprs_equal(&CExpr::var("x"), &CExpr::var("y")));

    // Test array index equality
    let arr_i = CExpr::index(CExpr::var("arr"), CExpr::var("i"));
    let arr_i2 = CExpr::index(CExpr::var("arr"), CExpr::var("i"));
    let arr_j = CExpr::index(CExpr::var("arr"), CExpr::var("j"));

    assert!(inference.exprs_equal(&arr_i, &arr_i2));
    assert!(!inference.exprs_equal(&arr_i, &arr_j));
}

// ========================================================================
// Ghost Variable Tests
// ========================================================================

#[test]
fn test_ghost_variable_iteration_count() {
    let ghost = GhostVariable::iteration_count("i");
    assert_eq!(ghost.name, "__ghost_iter_i");
    assert_eq!(ghost.kind, GhostKind::IterationCount);
    assert_eq!(ghost.related_var, Some("i".to_string()));
    assert!(matches!(ghost.initial_value, Some(Spec::Int(0))));
}

#[test]
fn test_ghost_variable_original_value() {
    let ghost = GhostVariable::original_value("x", Some(Spec::int(42)));
    assert_eq!(ghost.name, "__ghost_old_x");
    assert_eq!(ghost.kind, GhostKind::OriginalValue);
    assert_eq!(ghost.related_var, Some("x".to_string()));
    assert!(matches!(ghost.initial_value, Some(Spec::Int(42))));
}

#[test]
fn test_ghost_variable_partial_result() {
    let ghost = GhostVariable::partial_result("sum", Spec::int(0));
    assert_eq!(ghost.name, "__ghost_partial_sum");
    assert_eq!(ghost.kind, GhostKind::PartialResult);
    assert_eq!(ghost.related_var, Some("sum".to_string()));
}

#[test]
fn test_context_generate_loop_ghosts() {
    let mut ctx = InferenceContext::new();
    ctx.set_initial("sum", Spec::int(0));
    ctx.set_initial("count", Spec::int(0));

    ctx.generate_loop_ghosts(Some("i"), &["sum".to_string(), "count".to_string()]);

    // Should have 3 ghosts: iteration counter + 2 original values
    assert_eq!(ctx.ghost_vars.len(), 3);

    // Check iteration counter ghost
    assert!(ctx
        .ghost_vars
        .iter()
        .any(|g| g.name == "__ghost_iter_i" && g.kind == GhostKind::IterationCount));

    // Check original value ghosts
    assert!(ctx
        .ghost_vars
        .iter()
        .any(|g| g.name == "__ghost_old_sum" && g.kind == GhostKind::OriginalValue));
    assert!(ctx
        .ghost_vars
        .iter()
        .any(|g| g.name == "__ghost_old_count" && g.kind == GhostKind::OriginalValue));
}

#[test]
fn test_context_ghost_invariants() {
    let mut ctx = InferenceContext::new();
    ctx.set_initial("x", Spec::int(10));

    // Add a ghost for original value
    ctx.add_ghost(GhostVariable::original_value("x", Some(Spec::int(10))));

    // Add a ghost for iteration count
    ctx.add_ghost(GhostVariable::iteration_count("i"));

    // Add a ghost for partial result
    ctx.add_ghost(GhostVariable::partial_result("sum", Spec::int(0)));

    let invariants = ctx.ghost_invariants();

    // Should generate invariants for each ghost
    // - Original value: __ghost_old_x == 10
    // - Iteration count: __ghost_iter_i >= 0
    // - Partial result: __ghost_partial_sum >= 0
    assert_eq!(invariants.len(), 3);

    // Check that we have a >= 0 invariant for iteration count
    assert!(invariants.iter().any(|inv| {
        matches!(inv, Spec::BinOp { op: BinOp::Ge, left, right }
            if matches!(left.as_ref(), Spec::Var(name) if name == "__ghost_iter_i")
            && matches!(right.as_ref(), Spec::Int(0)))
    }));
}

#[test]
fn test_ghost_custom() {
    let ghost = GhostVariable::custom("my_ghost");
    assert_eq!(ghost.name, "my_ghost");
    assert_eq!(ghost.kind, GhostKind::Custom);
    assert_eq!(ghost.related_var, None);
    assert_eq!(ghost.initial_value, None);
}

// ========================================================================
// Counting Pattern Tests
// ========================================================================

#[test]
fn test_detect_counting_pattern_postinc() {
    // if (arr[i] > 0) { count++; }
    let inference = InvariantInference::new();

    let cond = CExpr::BinOp {
        op: BinOp::Gt,
        left: Box::new(CExpr::Index {
            array: Box::new(CExpr::Var("arr".to_string())),
            index: Box::new(CExpr::Var("i".to_string())),
        }),
        right: Box::new(CExpr::IntLit(0)),
    };

    let then_body = CStmt::Expr(CExpr::UnaryOp {
        op: UnaryOp::PostInc,
        operand: Box::new(CExpr::Var("count".to_string())),
    });

    let result = inference.detect_counting_pattern(&cond, &then_body);
    assert!(result.is_some(), "Should detect postinc counting pattern");

    let info = result.unwrap();
    assert_eq!(info.accumulator, "count");
    assert_eq!(info.op, AccumulatorOp::Count);
}

#[test]
fn test_detect_counting_pattern_preinc() {
    // if (x == target) { ++count; }
    let inference = InvariantInference::new();

    let cond = CExpr::BinOp {
        op: BinOp::Eq,
        left: Box::new(CExpr::Var("x".to_string())),
        right: Box::new(CExpr::Var("target".to_string())),
    };

    let then_body = CStmt::Expr(CExpr::UnaryOp {
        op: UnaryOp::PreInc,
        operand: Box::new(CExpr::Var("count".to_string())),
    });

    let result = inference.detect_counting_pattern(&cond, &then_body);
    assert!(result.is_some(), "Should detect preinc counting pattern");

    let info = result.unwrap();
    assert_eq!(info.accumulator, "count");
    assert_eq!(info.op, AccumulatorOp::Count);
}

#[test]
fn test_detect_counting_pattern_add_assign() {
    // if (condition) { count += 1; }
    let inference = InvariantInference::new();

    let cond = CExpr::BinOp {
        op: BinOp::Lt,
        left: Box::new(CExpr::Var("value".to_string())),
        right: Box::new(CExpr::IntLit(10)),
    };

    let then_body = CStmt::Expr(CExpr::BinOp {
        op: BinOp::AddAssign,
        left: Box::new(CExpr::Var("count".to_string())),
        right: Box::new(CExpr::IntLit(1)),
    });

    let result = inference.detect_counting_pattern(&cond, &then_body);
    assert!(result.is_some(), "Should detect count += 1 pattern");

    let info = result.unwrap();
    assert_eq!(info.accumulator, "count");
    assert_eq!(info.op, AccumulatorOp::Count);
}

#[test]
fn test_detect_counting_pattern_assign_add() {
    // if (condition) { count = count + 1; }
    let inference = InvariantInference::new();

    let cond = CExpr::BinOp {
        op: BinOp::Ne,
        left: Box::new(CExpr::Var("x".to_string())),
        right: Box::new(CExpr::IntLit(0)),
    };

    let then_body = CStmt::Expr(CExpr::BinOp {
        op: BinOp::Assign,
        left: Box::new(CExpr::Var("count".to_string())),
        right: Box::new(CExpr::BinOp {
            op: BinOp::Add,
            left: Box::new(CExpr::Var("count".to_string())),
            right: Box::new(CExpr::IntLit(1)),
        }),
    });

    let result = inference.detect_counting_pattern(&cond, &then_body);
    assert!(result.is_some(), "Should detect count = count + 1 pattern");

    let info = result.unwrap();
    assert_eq!(info.accumulator, "count");
    assert_eq!(info.op, AccumulatorOp::Count);
}

#[test]
fn test_detect_counting_pattern_in_block() {
    // if (condition) { count++; }  (wrapped in block)
    let inference = InvariantInference::new();

    let cond = CExpr::BinOp {
        op: BinOp::Gt,
        left: Box::new(CExpr::Var("x".to_string())),
        right: Box::new(CExpr::IntLit(0)),
    };

    let then_body = CStmt::Block(vec![CStmt::Expr(CExpr::UnaryOp {
        op: UnaryOp::PostInc,
        operand: Box::new(CExpr::Var("n".to_string())),
    })]);

    let result = inference.detect_counting_pattern(&cond, &then_body);
    assert!(result.is_some(), "Should detect count++ in block");

    let info = result.unwrap();
    assert_eq!(info.accumulator, "n");
    assert_eq!(info.op, AccumulatorOp::Count);
}

#[test]
fn test_count_invariant_inference() {
    // Test that Count operation generates correct invariant
    let inference = InvariantInference::new();
    let mut context = InferenceContext::new();
    context.set_initial("count", Spec::int(0));

    // if (condition) { count++; }
    let body = CStmt::If {
        cond: CExpr::BinOp {
            op: BinOp::Gt,
            left: Box::new(CExpr::Var("x".to_string())),
            right: Box::new(CExpr::IntLit(0)),
        },
        then_stmt: Box::new(CStmt::Expr(CExpr::UnaryOp {
            op: UnaryOp::PostInc,
            operand: Box::new(CExpr::Var("count".to_string())),
        })),
        else_stmt: None,
    };

    let result = inference.detect_sum_pattern(&body, &context);
    assert!(result.is_some(), "Should infer invariant for count pattern");

    // Invariant should be count >= 0 (initial value)
    let inv = result.unwrap();
    // Check it's a >= comparison
    if let Spec::BinOp { op, left, right } = inv {
        assert_eq!(op, BinOp::Ge);
        // Left should be count var
        assert!(matches!(left.as_ref(), Spec::Var(name) if name == "count"));
        // Right should be 0
        assert!(matches!(right.as_ref(), Spec::Int(0)));
    } else {
        panic!("Expected BinOp invariant");
    }
}

#[test]
fn test_is_increment_expr_not_increment() {
    // count += 2 is NOT a simple increment (it adds 2, not 1)
    let inference = InvariantInference::new();

    let expr = CExpr::BinOp {
        op: BinOp::AddAssign,
        left: Box::new(CExpr::Var("count".to_string())),
        right: Box::new(CExpr::IntLit(2)),
    };

    let result = inference.is_increment_expr(&expr);
    assert!(result.is_none(), "count += 2 is not a simple increment");
}

// ========================================================================
// Ghost Variable Integration Tests
// ========================================================================

#[test]
fn test_extract_loop_variable() {
    // i < n
    let cond = CExpr::BinOp {
        op: BinOp::Lt,
        left: Box::new(CExpr::Var("i".to_string())),
        right: Box::new(CExpr::Var("n".to_string())),
    };
    let result = VCGen::extract_loop_variable(&cond);
    assert_eq!(result, Some("i".to_string()));

    // i != 0
    let cond2 = CExpr::BinOp {
        op: BinOp::Ne,
        left: Box::new(CExpr::Var("i".to_string())),
        right: Box::new(CExpr::IntLit(0)),
    };
    let result2 = VCGen::extract_loop_variable(&cond2);
    assert_eq!(result2, Some("i".to_string()));

    // !found
    let cond3 = CExpr::UnaryOp {
        op: UnaryOp::LogNot,
        operand: Box::new(CExpr::Var("found".to_string())),
    };
    let result3 = VCGen::extract_loop_variable(&cond3);
    assert_eq!(result3, Some("found".to_string()));
}

#[test]
fn test_extract_modified_variables() {
    // i++ (PostInc)
    let body = CStmt::Block(vec![
        CStmt::Expr(CExpr::UnaryOp {
            op: UnaryOp::PostInc,
            operand: Box::new(CExpr::Var("i".to_string())),
        }),
        CStmt::Expr(CExpr::BinOp {
            op: BinOp::AddAssign,
            left: Box::new(CExpr::Var("sum".to_string())),
            right: Box::new(CExpr::Index {
                array: Box::new(CExpr::Var("arr".to_string())),
                index: Box::new(CExpr::Var("i".to_string())),
            }),
        }),
    ]);

    let vars = VCGen::extract_modified_variables(&body);
    assert!(vars.contains(&"i".to_string()));
    assert!(vars.contains(&"sum".to_string()));
    assert_eq!(vars.len(), 2);
}

#[test]
fn test_ghost_integration_in_inference() {
    // Test that ghost invariants are included when context has ghosts
    let mut inference = InvariantInference::new();
    let mut context = InferenceContext::new();

    // Add a ghost variable
    context.add_ghost(GhostVariable::iteration_count("i"));

    // Create a simple loop: while (i < n) { i++; }
    let cond = CExpr::BinOp {
        op: BinOp::Lt,
        left: Box::new(CExpr::Var("i".to_string())),
        right: Box::new(CExpr::Var("n".to_string())),
    };
    let body = CStmt::Expr(CExpr::UnaryOp {
        op: UnaryOp::PostInc,
        operand: Box::new(CExpr::Var("i".to_string())),
    });

    let invariants = inference.infer_while_invariant(&cond, &body, &context);

    // Should include ghost invariant (iteration count >= 0)
    let has_ghost_inv = invariants.iter().any(|inv| {
        matches!(inv, Spec::BinOp { op: BinOp::Ge, left, right }
            if matches!(left.as_ref(), Spec::Var(name) if name == "__ghost_iter_i")
            && matches!(right.as_ref(), Spec::Int(0)))
    });
    assert!(
        has_ghost_inv,
        "Should include ghost iteration count invariant"
    );
}

#[test]
fn test_ghost_array_element() {
    let ghost = GhostVariable::array_element("arr", Spec::var("i"), Some(Spec::int(42)));
    assert_eq!(ghost.name, "__ghost_arr_arr_i");
    assert_eq!(ghost.kind, GhostKind::ArrayElement);
    assert_eq!(ghost.array_name, Some("arr".to_string()));
    assert!(matches!(&ghost.array_index, Some(Spec::Var(v)) if v == "i"));
    assert!(matches!(&ghost.initial_value, Some(Spec::Int(42))));
}

#[test]
fn test_ghost_array_element_with_constant_index() {
    let ghost = GhostVariable::array_element("data", Spec::int(5), None);
    assert_eq!(ghost.name, "__ghost_arr_data_5");
    assert_eq!(ghost.kind, GhostKind::ArrayElement);
    assert_eq!(ghost.array_name, Some("data".to_string()));
    assert!(matches!(&ghost.array_index, Some(Spec::Int(5))));
}

#[test]
fn test_context_add_array_ghost() {
    let mut ctx = InferenceContext::new();

    ctx.add_array_ghost("arr", Spec::var("i"), Some(Spec::int(10)));

    assert_eq!(ctx.ghost_vars.len(), 1);
    let ghost = &ctx.ghost_vars[0];
    assert_eq!(ghost.kind, GhostKind::ArrayElement);
    assert_eq!(ghost.array_name, Some("arr".to_string()));
}

#[test]
fn test_context_generate_array_frame_ghosts() {
    let mut ctx = InferenceContext::new();

    // Simulate accesses: arr[i] read, arr[j] written, arr[k] read
    let accesses = vec![
        ("arr".to_string(), Spec::var("i"), false), // read
        ("arr".to_string(), Spec::var("j"), true),  // write
        ("arr".to_string(), Spec::var("k"), false), // read
    ];

    ctx.generate_array_frame_ghosts(&accesses);

    // Should generate frame ghosts for read-only accesses (i and k)
    // j is written so no frame ghost for it
    let frame_ghosts: Vec<_> = ctx
        .ghost_vars
        .iter()
        .filter(|g| g.kind == GhostKind::ArrayElement)
        .collect();

    assert_eq!(
        frame_ghosts.len(),
        2,
        "Should have 2 frame ghosts for read-only indices"
    );

    // Check that we have ghosts for i and k but not j
    let has_i = frame_ghosts
        .iter()
        .any(|g| matches!(&g.array_index, Some(Spec::Var(v)) if v == "i"));
    let has_k = frame_ghosts
        .iter()
        .any(|g| matches!(&g.array_index, Some(Spec::Var(v)) if v == "k"));
    let has_j = frame_ghosts
        .iter()
        .any(|g| matches!(&g.array_index, Some(Spec::Var(v)) if v == "j"));

    assert!(has_i, "Should have frame ghost for read-only arr[i]");
    assert!(has_k, "Should have frame ghost for read-only arr[k]");
    assert!(!has_j, "Should NOT have frame ghost for written arr[j]");
}

#[test]
fn test_ghost_invariants_for_array_element() {
    let mut ctx = InferenceContext::new();

    // Add an array element ghost with known initial value
    ctx.add_array_ghost("data", Spec::var("idx"), Some(Spec::int(100)));

    let invariants = ctx.ghost_invariants();

    // Should generate invariants:
    // 1. __ghost_arr_data_idx == 100 (ghost equals initial)
    // 2. data[idx] == __ghost_arr_data_idx (frame condition)
    assert!(
        invariants.len() >= 2,
        "Should generate at least 2 invariants for array element ghost"
    );

    // Check for the frame condition invariant
    let has_frame_invariant = invariants.iter().any(|inv| {
        if let Spec::BinOp {
            op: BinOp::Eq,
            left,
            right,
        } = inv
        {
            // Check if left is data[idx]
            matches!(
                left.as_ref(),
                Spec::Index { base, index }
                    if matches!(base.as_ref(), Spec::Var(b) if b == "data")
                    && matches!(index.as_ref(), Spec::Var(i) if i == "idx")
            ) && matches!(right.as_ref(), Spec::Var(g) if g == "__ghost_arr_data_idx")
        } else {
            false
        }
    });

    assert!(has_frame_invariant, "Should have frame condition invariant");
}

#[test]
fn test_extract_array_accesses() {
    // Test extraction from: arr[i] = arr[j] + 1;
    let body = CStmt::Expr(CExpr::assign(
        CExpr::index(CExpr::var("arr"), CExpr::var("i")),
        CExpr::binop(
            BinOp::Add,
            CExpr::index(CExpr::var("arr"), CExpr::var("j")),
            CExpr::int(1),
        ),
    ));

    let accesses = VCGen::extract_array_accesses(&body);

    assert_eq!(accesses.len(), 2, "Should find 2 array accesses");

    // arr[i] is written (is_write = true)
    let write_access = accesses
        .iter()
        .find(|(_, idx, is_write)| matches!(idx, CExpr::Var(v) if v == "i") && *is_write);
    assert!(write_access.is_some(), "Should find write to arr[i]");

    // arr[j] is read (is_write = false)
    let read_access = accesses
        .iter()
        .find(|(_, idx, is_write)| matches!(idx, CExpr::Var(v) if v == "j") && !*is_write);
    assert!(read_access.is_some(), "Should find read from arr[j]");
}

#[test]
fn test_cexpr_to_spec() {
    // Test variable
    let var_spec = VCGen::cexpr_to_spec(&CExpr::var("x"));
    assert!(matches!(var_spec, Some(Spec::Var(v)) if v == "x"));

    // Test integer literal
    let int_spec = VCGen::cexpr_to_spec(&CExpr::int(42));
    assert!(matches!(int_spec, Some(Spec::Int(42))));

    // Test binary operation
    let add_expr = CExpr::binop(BinOp::Add, CExpr::var("a"), CExpr::int(1));
    let add_spec = VCGen::cexpr_to_spec(&add_expr);
    assert!(matches!(add_spec, Some(Spec::BinOp { op: BinOp::Add, .. })));

    // Test array index
    let idx_expr = CExpr::index(CExpr::var("arr"), CExpr::var("i"));
    let idx_spec = VCGen::cexpr_to_spec(&idx_expr);
    assert!(matches!(idx_spec, Some(Spec::Index { .. })));
}

#[test]
fn test_switch_wp_simple() {
    // Test WP for a simple switch statement:
    // switch (x) {
    //     case 1: y = 10; break;
    //     case 2: y = 20; break;
    //     default: y = 0;
    // }
    // postcond: y >= 0

    let mut vcgen = VCGen::new();

    let switch_body = CStmt::Block(vec![
        CStmt::Case {
            label: CaseLabel::Case(CExpr::int(1)),
            stmt: Box::new(CStmt::Block(vec![
                CStmt::Expr(CExpr::assign(CExpr::var("y"), CExpr::int(10))),
                CStmt::Break,
            ])),
        },
        CStmt::Case {
            label: CaseLabel::Case(CExpr::int(2)),
            stmt: Box::new(CStmt::Block(vec![
                CStmt::Expr(CExpr::assign(CExpr::var("y"), CExpr::int(20))),
                CStmt::Break,
            ])),
        },
        CStmt::Case {
            label: CaseLabel::Default,
            stmt: Box::new(CStmt::Expr(CExpr::assign(CExpr::var("y"), CExpr::int(0)))),
        },
    ]);

    let switch_stmt = CStmt::Switch {
        cond: CExpr::var("x"),
        body: Box::new(switch_body),
    };

    let postcond = Spec::binop(BinOp::Ge, Spec::var("y"), Spec::int(0));
    let wp = vcgen.wp_stmt(&switch_stmt, &postcond, None);

    // WP should be a conjunction of implications
    assert!(
        matches!(wp, Spec::And(_)),
        "Switch WP should be a conjunction"
    );
}

#[test]
fn test_switch_extract_cases() {
    // Test that extract_switch_cases correctly extracts cases

    let vcgen = VCGen::new();

    let switch_body = CStmt::Block(vec![
        CStmt::Case {
            label: CaseLabel::Case(CExpr::int(1)),
            stmt: Box::new(CStmt::Expr(CExpr::var("a"))),
        },
        CStmt::Break,
        CStmt::Case {
            label: CaseLabel::Case(CExpr::int(2)),
            stmt: Box::new(CStmt::Expr(CExpr::var("b"))),
        },
        CStmt::Break,
        CStmt::Case {
            label: CaseLabel::Default,
            stmt: Box::new(CStmt::Expr(CExpr::var("c"))),
        },
    ]);

    let cases = vcgen.extract_switch_cases(&switch_body);

    assert_eq!(cases.len(), 3, "Should extract 3 cases");

    // Check case labels
    assert!(matches!(&cases[0].0, CaseLabel::Case(CExpr::IntLit(1))));
    assert!(matches!(&cases[1].0, CaseLabel::Case(CExpr::IntLit(2))));
    assert!(matches!(&cases[2].0, CaseLabel::Default));
}

#[test]
fn test_switch_wp_no_default() {
    // Test switch without default case

    let mut vcgen = VCGen::new();

    let switch_body = CStmt::Block(vec![
        CStmt::Case {
            label: CaseLabel::Case(CExpr::int(1)),
            stmt: Box::new(CStmt::Block(vec![
                CStmt::Expr(CExpr::assign(CExpr::var("y"), CExpr::int(10))),
                CStmt::Break,
            ])),
        },
        CStmt::Case {
            label: CaseLabel::Case(CExpr::int(2)),
            stmt: Box::new(CStmt::Block(vec![
                CStmt::Expr(CExpr::assign(CExpr::var("y"), CExpr::int(20))),
                CStmt::Break,
            ])),
        },
    ]);

    let switch_stmt = CStmt::Switch {
        cond: CExpr::var("x"),
        body: Box::new(switch_body),
    };

    let postcond = Spec::True;
    let wp = vcgen.wp_stmt(&switch_stmt, &postcond, None);

    // WP should still be valid without default
    assert!(
        matches!(wp, Spec::And(_)),
        "Switch WP should handle missing default"
    );
}

#[test]
fn test_switch_wp_empty() {
    // Test empty switch (no cases)

    let mut vcgen = VCGen::new();

    let switch_stmt = CStmt::Switch {
        cond: CExpr::var("x"),
        body: Box::new(CStmt::Block(vec![])),
    };

    let postcond = Spec::var("P");
    let wp = vcgen.wp_stmt(&switch_stmt, &postcond, None);

    // Empty switch should just return postcond
    assert!(
        matches!(wp, Spec::Var(v) if v == "P"),
        "Empty switch should return postcond"
    );
}

// ========== Assigns Clause Tests ==========

#[test]
fn test_collect_modified_locations_assignment() {
    // Test that x = 5 collects modification to x
    let vcgen = VCGen::new();

    let stmt = CStmt::Expr(CExpr::BinOp {
        op: BinOp::Assign,
        left: Box::new(CExpr::Var("x".to_string())),
        right: Box::new(CExpr::IntLit(5)),
    });

    let modified = vcgen.collect_modified_locations(&stmt);
    assert_eq!(modified.len(), 1);
    assert!(matches!(&modified[0].location, Location::Deref(Spec::Var(n)) if n == "x"));
}

#[test]
fn test_collect_modified_locations_increment() {
    // Test that x++ collects modification to x
    let vcgen = VCGen::new();

    let stmt = CStmt::Expr(CExpr::UnaryOp {
        op: UnaryOp::PostInc,
        operand: Box::new(CExpr::Var("x".to_string())),
    });

    let modified = vcgen.collect_modified_locations(&stmt);
    assert_eq!(modified.len(), 1);
    assert!(matches!(&modified[0].location, Location::Deref(Spec::Var(n)) if n == "x"));
}

#[test]
fn test_collect_modified_locations_array_assignment() {
    // Test that arr[i] = v collects modification to arr[i]
    let vcgen = VCGen::new();

    let stmt = CStmt::Expr(CExpr::BinOp {
        op: BinOp::Assign,
        left: Box::new(CExpr::Index {
            array: Box::new(CExpr::Var("arr".to_string())),
            index: Box::new(CExpr::Var("i".to_string())),
        }),
        right: Box::new(CExpr::IntLit(42)),
    });

    let modified = vcgen.collect_modified_locations(&stmt);
    assert_eq!(modified.len(), 1);
    // arr[i] becomes Location::Deref(arr + i)
    assert!(matches!(&modified[0].location, Location::Deref(_)));
}

#[test]
fn test_collect_modified_locations_block() {
    // Test that modifications in block are all collected
    let vcgen = VCGen::new();

    let stmt = CStmt::Block(vec![
        CStmt::Expr(CExpr::BinOp {
            op: BinOp::Assign,
            left: Box::new(CExpr::Var("x".to_string())),
            right: Box::new(CExpr::IntLit(1)),
        }),
        CStmt::Expr(CExpr::BinOp {
            op: BinOp::Assign,
            left: Box::new(CExpr::Var("y".to_string())),
            right: Box::new(CExpr::IntLit(2)),
        }),
    ]);

    let modified = vcgen.collect_modified_locations(&stmt);
    assert_eq!(modified.len(), 2);
}

#[test]
fn test_check_assigns_covered() {
    // When modification is covered by assigns, no VCs are generated
    let mut vcgen = VCGen::new();

    let assigns = vec![Location::Deref(Spec::var("x"))];
    let modified = vec![ModifiedLocation {
        location: Location::Deref(Spec::var("x")),
        description: "assignment to 'x'".to_string(),
        source_line: None,
    }];

    let vcs = vcgen.check_assigns(&assigns, &modified);
    assert!(
        vcs.is_empty(),
        "No VCs should be generated when modification is covered"
    );
}

#[test]
fn test_check_assigns_not_covered() {
    // When modification is NOT covered by assigns, VC is generated
    let mut vcgen = VCGen::new();

    let assigns = vec![Location::Deref(Spec::var("x"))];
    let modified = vec![ModifiedLocation {
        location: Location::Deref(Spec::var("y")),
        description: "assignment to 'y'".to_string(),
        source_line: None,
    }];

    let vcs = vcgen.check_assigns(&assigns, &modified);
    assert_eq!(
        vcs.len(),
        1,
        "One VC should be generated for uncovered modification"
    );
    assert_eq!(vcs[0].kind, VCKind::AssignsClause);
}

#[test]
fn test_check_assigns_everything_allows_all() {
    // assigns \everything allows all modifications
    let mut vcgen = VCGen::new();

    let assigns = vec![Location::Everything];
    let modified = vec![
        ModifiedLocation {
            location: Location::Deref(Spec::var("x")),
            description: "assignment to 'x'".to_string(),
            source_line: None,
        },
        ModifiedLocation {
            location: Location::Deref(Spec::var("y")),
            description: "assignment to 'y'".to_string(),
            source_line: None,
        },
    ];

    let vcs = vcgen.check_assigns(&assigns, &modified);
    assert!(
        vcs.is_empty(),
        "assigns \\everything should allow all modifications"
    );
}

#[test]
fn test_check_assigns_empty_allows_nothing() {
    // Empty assigns clause allows nothing
    let mut vcgen = VCGen::new();

    let assigns: Vec<Location> = vec![];
    let modified = vec![ModifiedLocation {
        location: Location::Deref(Spec::var("x")),
        description: "assignment to 'x'".to_string(),
        source_line: None,
    }];

    let vcs = vcgen.check_assigns(&assigns, &modified);
    assert_eq!(vcs.len(), 1);
    // The obligation should be False (cannot prove location in empty assigns)
    assert!(matches!(vcs[0].obligation, Spec::False));
}

#[test]
fn test_check_assigns_multiple_locations() {
    // Multiple allowed locations
    let mut vcgen = VCGen::new();

    let assigns = vec![
        Location::Deref(Spec::var("x")),
        Location::Deref(Spec::var("y")),
    ];
    let modified = vec![
        ModifiedLocation {
            location: Location::Deref(Spec::var("x")),
            description: "assignment to 'x'".to_string(),
            source_line: None,
        },
        ModifiedLocation {
            location: Location::Deref(Spec::var("y")),
            description: "assignment to 'y'".to_string(),
            source_line: None,
        },
        ModifiedLocation {
            location: Location::Deref(Spec::var("z")),
            description: "assignment to 'z'".to_string(),
            source_line: None,
        },
    ];

    let vcs = vcgen.check_assigns(&assigns, &modified);
    // x and y are covered, z is not
    assert_eq!(vcs.len(), 1);
    assert!(vcs[0].description.contains("'z'"));
}

#[test]
fn test_check_assigns_pointer_offset_within_range() {
    // *(base + k) should be covered when k is inside the allowed range
    let mut vcgen = VCGen::new();

    let assigns = vec![Location::Range {
        base: Spec::var("arr"),
        lo: Spec::int(0),
        hi: Spec::int(10),
    }];
    let modified = vec![ModifiedLocation {
        location: Location::Deref(Spec::binop(BinOp::Add, Spec::var("arr"), Spec::int(3))),
        description: "assignment to arr[3]".to_string(),
        source_line: None,
    }];

    let vcs = vcgen.check_assigns(&assigns, &modified);
    assert!(
        vcs.is_empty(),
        "Pointer offset inside assigns range should be treated as covered"
    );
}

#[test]
fn test_check_assigns_pointer_offset_out_of_range() {
    // Pointer arithmetic outside the assigns range should trigger a VC
    let mut vcgen = VCGen::new();

    let assigns = vec![Location::Range {
        base: Spec::var("arr"),
        lo: Spec::int(0),
        hi: Spec::int(5),
    }];
    let modified = vec![ModifiedLocation {
        location: Location::Deref(Spec::binop(BinOp::Add, Spec::var("arr"), Spec::int(12))),
        description: "assignment to arr[12]".to_string(),
        source_line: None,
    }];

    let vcs = vcgen.check_assigns(&assigns, &modified);
    assert_eq!(
        vcs.len(),
        1,
        "Out-of-range pointer offset should produce an assigns VC"
    );
}

#[test]
fn test_check_assigns_range_subset_covered() {
    // A modified subrange entirely within assigns range should be treated as covered
    let mut vcgen = VCGen::new();

    let assigns = vec![Location::Range {
        base: Spec::var("arr"),
        lo: Spec::int(0),
        hi: Spec::int(10),
    }];
    let modified = vec![ModifiedLocation {
        location: Location::Range {
            base: Spec::var("arr"),
            lo: Spec::int(2),
            hi: Spec::int(4),
        },
        description: "write to arr[2..4]".to_string(),
        source_line: None,
    }];

    let vcs = vcgen.check_assigns(&assigns, &modified);
    assert!(
        vcs.is_empty(),
        "Subrange inside assigns bounds should not produce VCs"
    );
}

#[test]
fn test_location_subset_deref_to_range() {
    // Test that *p subset of p[0..n] generates correct VC
    let vcgen = VCGen::new();

    let loc = Location::Deref(Spec::var("ptr"));
    let range = Location::Range {
        base: Spec::var("arr"),
        lo: Spec::int(0),
        hi: Spec::var("n"),
    };

    let vc = vcgen.location_subset_of(&loc, &range);
    // Should be: 0 <= (ptr - arr) < n
    assert!(matches!(vc, Spec::And(_)));
}

#[test]
fn test_collect_local_variables() {
    // Test collecting local variables from a function body
    let vcgen = VCGen::new();

    let stmt = CStmt::Block(vec![
        CStmt::Decl(VarDecl {
            name: "x".to_string(),
            ty: crate::types::CType::int(),
            storage: StorageClass::Auto,
            init: None,
        }),
        CStmt::Decl(VarDecl {
            name: "y".to_string(),
            ty: crate::types::CType::int(),
            storage: StorageClass::Auto,
            init: Some(Initializer::Expr(CExpr::IntLit(0))),
        }),
    ]);

    let locals = vcgen.collect_local_variables(&stmt);
    assert_eq!(locals.len(), 2);
    assert!(locals.contains(&"x".to_string()));
    assert!(locals.contains(&"y".to_string()));
}

#[test]
fn test_collect_local_variables_nested() {
    // Test collecting locals from nested blocks and control flow
    let vcgen = VCGen::new();

    let stmt = CStmt::Block(vec![
        CStmt::Decl(VarDecl {
            name: "outer".to_string(),
            ty: crate::types::CType::int(),
            storage: StorageClass::Auto,
            init: None,
        }),
        CStmt::If {
            cond: CExpr::IntLit(1),
            then_stmt: Box::new(CStmt::Decl(VarDecl {
                name: "inner".to_string(),
                ty: crate::types::CType::int(),
                storage: StorageClass::Auto,
                init: None,
            })),
            else_stmt: None,
        },
    ]);

    let locals = vcgen.collect_local_variables(&stmt);
    assert_eq!(locals.len(), 2);
    assert!(locals.contains(&"outer".to_string()));
    assert!(locals.contains(&"inner".to_string()));
}

#[test]
fn test_filter_non_locals() {
    // Test filtering out local variables from modified locations
    let vcgen = VCGen::new();

    let modified = vec![
        ModifiedLocation {
            location: Location::Deref(Spec::var("local_var")),
            description: "local variable".to_string(),
            source_line: None,
        },
        ModifiedLocation {
            location: Location::Deref(Spec::var("global_var")),
            description: "global variable".to_string(),
            source_line: None,
        },
    ];

    let locals = vec!["local_var".to_string()];
    let filtered = vcgen.filter_non_locals(modified, &locals);

    assert_eq!(filtered.len(), 1);
    assert!(filtered[0].description.contains("global"));
}

#[test]
fn test_gen_function_assigns_with_locals() {
    // Integration test: gen_function should not flag local variables as assigns violations

    let mut vcgen = VCGen::new();

    // Function: void foo(int n) { int x = 0; x = n; *ptr = 1; }
    // with assigns { *ptr }
    let func = FuncDef {
        name: "foo".to_string(),
        return_type: crate::types::CType::Void,
        params: vec![FuncParam::new("n", crate::types::CType::int())],
        variadic: false,
        storage: StorageClass::Auto,
        body: Box::new(CStmt::Block(vec![
            CStmt::Decl(VarDecl {
                name: "x".to_string(),
                ty: crate::types::CType::int(),
                storage: StorageClass::Auto,
                init: Some(Initializer::Expr(CExpr::IntLit(0))),
            }),
            // x = n (assignment to local - should NOT generate VC)
            CStmt::Expr(CExpr::BinOp {
                op: BinOp::Assign,
                left: Box::new(CExpr::Var("x".to_string())),
                right: Box::new(CExpr::Var("n".to_string())),
            }),
            // *ptr = 1 (assignment through pointer - covered by assigns)
            CStmt::Expr(CExpr::BinOp {
                op: BinOp::Assign,
                left: Box::new(CExpr::UnaryOp {
                    op: UnaryOp::Deref,
                    operand: Box::new(CExpr::Var("ptr".to_string())),
                }),
                right: Box::new(CExpr::IntLit(1)),
            }),
        ])),
    };

    let spec = FuncSpec {
        params: vec![],
        requires: vec![],
        ensures: vec![],
        assigns: vec![Location::Deref(Spec::var("ptr"))],
        reads: vec![],
        terminates: None,
        behaviors: vec![],
        complete: vec![],
        disjoint: vec![],
    };

    let vcs = vcgen.gen_function(&func, &spec);

    // Should have 1 VC for postcondition, 0 for assigns violations
    // The local variable 'x' and parameter 'n' should be excluded
    let assigns_vcs: Vec<_> = vcs
        .iter()
        .filter(|vc| vc.kind == VCKind::AssignsClause)
        .collect();
    assert!(
        assigns_vcs.is_empty(),
        "No assigns VCs should be generated for local variables"
    );
}

#[test]
fn test_gen_function_assigns_violation() {
    // Integration test: gen_function should flag non-local modifications not in assigns

    let mut vcgen = VCGen::new();

    // Function that modifies *ptr but assigns only allows *other
    let func = FuncDef {
        name: "bar".to_string(),
        return_type: crate::types::CType::Void,
        params: vec![FuncParam::new("n", crate::types::CType::int())],
        variadic: false,
        storage: StorageClass::Auto,
        body: Box::new(CStmt::Expr(CExpr::BinOp {
            op: BinOp::Assign,
            left: Box::new(CExpr::UnaryOp {
                op: UnaryOp::Deref,
                operand: Box::new(CExpr::Var("ptr".to_string())),
            }),
            right: Box::new(CExpr::IntLit(1)),
        })),
    };

    let spec = FuncSpec {
        params: vec![],
        requires: vec![],
        ensures: vec![],
        assigns: vec![Location::Deref(Spec::var("other"))], // Only allows *other
        reads: vec![],
        terminates: None,
        behaviors: vec![],
        complete: vec![],
        disjoint: vec![],
    };

    let vcs = vcgen.gen_function(&func, &spec);

    // Should have assigns VCs since *ptr is not in assigns clause
    let assigns_vcs: Vec<_> = vcs
        .iter()
        .filter(|vc| vc.kind == VCKind::AssignsClause)
        .collect();
    assert_eq!(
        assigns_vcs.len(),
        1,
        "Should generate 1 assigns VC for *ptr violation"
    );
}

#[test]
fn test_interprocedural_assigns_callee_modifies() {
    // Test that calling a function with an assigns clause propagates modifications
    let mut vcgen = VCGen::new();

    // Register callee spec: helper modifies *global_ptr
    let callee_spec = FuncSpec {
        requires: vec![],
        ensures: vec![],
        assigns: vec![Location::Deref(Spec::var("global_ptr"))],
        ..Default::default()
    };
    vcgen.register_func_spec("helper", callee_spec);

    // Caller function that calls helper()
    let caller = FuncDef {
        name: "caller".into(),
        return_type: CType::void(),
        params: vec![],
        body: Box::new(CStmt::Expr(CExpr::call(CExpr::var("helper"), vec![]))),
        variadic: false,
        storage: StorageClass::Auto,
    };

    // Caller only allows modifying *local_ptr (not *global_ptr)
    let caller_spec = FuncSpec {
        requires: vec![],
        ensures: vec![],
        assigns: vec![Location::Deref(Spec::var("local_ptr"))],
        ..Default::default()
    };

    let vcs = vcgen.gen_function(&caller, &caller_spec);

    // Should detect that helper() modifies *global_ptr which is not in caller's assigns
    let assigns_vcs: Vec<_> = vcs
        .iter()
        .filter(|vc| vc.kind == VCKind::AssignsClause)
        .collect();
    assert_eq!(
        assigns_vcs.len(),
        1,
        "Should detect callee's assigns clause violation"
    );
}

#[test]
fn test_interprocedural_assigns_callee_allowed() {
    // Test that callee's assigns are allowed when covered by caller's assigns
    let mut vcgen = VCGen::new();

    // Register callee spec: helper modifies *ptr
    let callee_spec = FuncSpec {
        requires: vec![],
        ensures: vec![],
        assigns: vec![Location::Deref(Spec::var("ptr"))],
        ..Default::default()
    };
    vcgen.register_func_spec("helper", callee_spec);

    // Caller function that calls helper()
    let caller = FuncDef {
        name: "caller".into(),
        return_type: CType::void(),
        params: vec![],
        body: Box::new(CStmt::Expr(CExpr::call(CExpr::var("helper"), vec![]))),
        variadic: false,
        storage: StorageClass::Auto,
    };

    // Caller also allows modifying *ptr - should be compatible
    let caller_spec = FuncSpec {
        requires: vec![],
        ensures: vec![],
        assigns: vec![Location::Deref(Spec::var("ptr"))],
        ..Default::default()
    };

    let vcs = vcgen.gen_function(&caller, &caller_spec);

    // Should NOT generate assigns violation since callee's *ptr matches caller's *ptr
    let assigns_vcs: Vec<_> = vcs
        .iter()
        .filter(|vc| vc.kind == VCKind::AssignsClause)
        .collect();
    assert_eq!(
        assigns_vcs.len(),
        0,
        "Callee's assigns should be covered by caller's assigns"
    );
}

#[test]
fn test_interprocedural_postcondition_assumption() {
    // Test that callee's postcondition is assumed for the call result
    let mut vcgen = VCGen::new();

    // Register callee spec: get_positive() ensures \result > 0
    let callee_spec = FuncSpec {
        requires: vec![],
        ensures: vec![Spec::gt(Spec::result(), Spec::int(0))],
        assigns: vec![],
        ..Default::default()
    };
    vcgen.register_func_spec("get_positive", callee_spec);

    // Just test wp_expr directly
    let call_expr = CExpr::call(CExpr::var("get_positive"), vec![]);
    let postcond = Spec::True;
    let wp = vcgen.wp_expr(&call_expr, &postcond);

    // WP should include the implication from callee's postcondition
    // The result should be: (get_positive() > 0) → True
    match wp {
        Spec::Implies(antecedent, consequent) => {
            // Antecedent should be the callee's postcondition (with \result substituted)
            assert!(matches!(*antecedent, Spec::BinOp { op: BinOp::Gt, .. }));
            assert!(matches!(*consequent, Spec::True));
        }
        _ => panic!("Expected Implies, got {wp:?}"),
    }
}

#[test]
fn test_interprocedural_precondition_vc() {
    // Test that callee's precondition generates a VC
    let mut vcgen = VCGen::new();

    // Register callee spec: divide(a, b) requires b != 0
    let callee_spec = FuncSpec {
        requires: vec![Spec::ne(Spec::var("b"), Spec::int(0))],
        ensures: vec![],
        assigns: vec![],
        ..Default::default()
    };
    vcgen.register_func_spec("divide", callee_spec);

    // Call divide(x, y)
    let call_expr = CExpr::call(CExpr::var("divide"), vec![CExpr::var("x"), CExpr::var("y")]);
    let postcond = Spec::True;
    vcgen.wp_expr(&call_expr, &postcond);

    let vcs = vcgen.get_vcs();

    // Should generate a precondition VC for divide's requires clause
    let precond_vcs: Vec<_> = vcs
        .iter()
        .filter(|vc| vc.kind == VCKind::Precondition)
        .collect();
    assert_eq!(
        precond_vcs.len(),
        1,
        "Should generate VC for callee's precondition"
    );
}

#[test]
fn test_interprocedural_assigns_nested_calls() {
    // Test interprocedural assigns with nested function calls
    let mut vcgen = VCGen::new();

    // inner() modifies *a
    let inner_spec = FuncSpec {
        requires: vec![],
        ensures: vec![],
        assigns: vec![Location::Deref(Spec::var("a"))],
        ..Default::default()
    };
    vcgen.register_func_spec("inner", inner_spec);

    // outer() calls inner() and also modifies *b
    // But we model outer() as only allowing *b in its own assigns
    let outer_spec = FuncSpec {
        requires: vec![],
        ensures: vec![],
        assigns: vec![Location::Deref(Spec::var("b"))],
        ..Default::default()
    };
    vcgen.register_func_spec("outer", outer_spec);

    // top() calls outer()
    let top = FuncDef {
        name: "top".into(),
        return_type: CType::void(),
        params: vec![],
        body: Box::new(CStmt::Expr(CExpr::call(CExpr::var("outer"), vec![]))),
        variadic: false,
        storage: StorageClass::Auto,
    };

    // top() only allows modifying *c
    let top_spec = FuncSpec {
        requires: vec![],
        ensures: vec![],
        assigns: vec![Location::Deref(Spec::var("c"))],
        ..Default::default()
    };

    let vcs = vcgen.gen_function(&top, &top_spec);

    // Should detect that outer() modifies *b which is not in top's assigns
    // Note: We only see direct callee's assigns, not transitive (inner's *a)
    let assigns_vcs: Vec<_> = vcs
        .iter()
        .filter(|vc| vc.kind == VCKind::AssignsClause)
        .collect();
    assert_eq!(
        assigns_vcs.len(),
        1,
        "Should detect outer's assigns violation"
    );
}

#[test]
fn test_param_substitution_in_precondition() {
    // Test that formal parameters are substituted with actual arguments in preconditions
    let mut vcgen = VCGen::new();

    // Register callee spec: divide(a, b) requires b != 0
    // params: ["a", "b"] maps formal to actual
    let callee_spec = FuncSpec {
        params: vec!["a".into(), "b".into()],
        requires: vec![Spec::ne(Spec::var("b"), Spec::int(0))],
        ensures: vec![],
        assigns: vec![],
        ..Default::default()
    };
    vcgen.register_func_spec("divide", callee_spec);

    // Call divide(x, y) - should substitute b -> y in the precondition
    let call_expr = CExpr::call(CExpr::var("divide"), vec![CExpr::var("x"), CExpr::var("y")]);
    let postcond = Spec::True;
    vcgen.wp_expr(&call_expr, &postcond);

    let vcs = vcgen.get_vcs();

    // Should generate a precondition VC with 'y != 0' (not 'b != 0')
    let precond_vcs: Vec<_> = vcs
        .iter()
        .filter(|vc| vc.kind == VCKind::Precondition)
        .collect();
    assert_eq!(
        precond_vcs.len(),
        1,
        "Should generate VC for callee's precondition"
    );

    // The VC should contain 'y', not 'b'
    let vc_obligation = &precond_vcs[0].obligation;
    match vc_obligation {
        Spec::BinOp {
            op: BinOp::Ne,
            left,
            right,
        } => {
            assert!(
                matches!(left.as_ref(), Spec::Var(name) if name == "y"),
                "Expected var 'y' in left side, got {left:?}"
            );
            assert!(
                matches!(right.as_ref(), Spec::Int(0)),
                "Expected 0 in right side, got {right:?}"
            );
        }
        _ => panic!("Expected BinOp Ne, got {vc_obligation:?}"),
    }
}

#[test]
fn test_param_substitution_in_postcondition() {
    // Test that params are substituted in postconditions too
    let mut vcgen = VCGen::new();

    // Register callee spec: max(a, b) ensures \result >= a && \result >= b
    let callee_spec = FuncSpec {
        params: vec!["a".into(), "b".into()],
        requires: vec![],
        ensures: vec![
            Spec::ge(Spec::result(), Spec::var("a")),
            Spec::ge(Spec::result(), Spec::var("b")),
        ],
        assigns: vec![],
        ..Default::default()
    };
    vcgen.register_func_spec("max", callee_spec);

    // Call max(x, y)
    let call_expr = CExpr::call(CExpr::var("max"), vec![CExpr::var("x"), CExpr::var("y")]);
    let postcond = Spec::True;
    let wp = vcgen.wp_expr(&call_expr, &postcond);

    // WP should be: (max(x,y) >= x && max(x,y) >= y) => True
    // The postcondition should have a and b substituted with x and y
    match wp {
        Spec::Implies(antecedent, _) => {
            // Antecedent should be the And of two Ge conditions with x and y
            match antecedent.as_ref() {
                Spec::And(conditions) => {
                    assert_eq!(conditions.len(), 2, "Expected 2 postconditions");
                    // Check that substitution occurred (should contain x and y, not a and b)
                    let spec_str = format!("{conditions:?}");
                    assert!(
                        spec_str.contains("\"x\"") || spec_str.contains("Var(\"x\")"),
                        "Expected 'x' in postcondition, got {conditions:?}"
                    );
                    assert!(
                        spec_str.contains("\"y\"") || spec_str.contains("Var(\"y\")"),
                        "Expected 'y' in postcondition, got {conditions:?}"
                    );
                    assert!(
                        !spec_str.contains("\"a\"") || spec_str.contains("\"max\""),
                        "Should not contain 'a' as standalone var in postcondition"
                    );
                }
                _ => panic!("Expected And, got {antecedent:?}"),
            }
        }
        _ => panic!("Expected Implies, got {wp:?}"),
    }
}

#[test]
fn test_param_substitution_in_assigns() {
    // Test that formal parameters are substituted in assigns clauses during interprocedural analysis
    let mut vcgen = VCGen::new();

    // Register callee spec: write_ptr(p) assigns *p
    let callee_spec = FuncSpec {
        params: vec!["p".into()],
        requires: vec![],
        ensures: vec![],
        assigns: vec![Location::Deref(Spec::var("p"))],
        ..Default::default()
    };
    vcgen.register_func_spec("write_ptr", callee_spec);

    // Caller function that calls write_ptr(global_ptr)
    let caller = FuncDef {
        name: "caller".into(),
        return_type: CType::void(),
        params: vec![],
        body: Box::new(CStmt::Expr(CExpr::call(
            CExpr::var("write_ptr"),
            vec![CExpr::var("global_ptr")],
        ))),
        variadic: false,
        storage: StorageClass::Auto,
    };

    // Caller allows modifying *global_ptr - should NOT generate violation
    let caller_spec = FuncSpec {
        requires: vec![],
        ensures: vec![],
        assigns: vec![Location::Deref(Spec::var("global_ptr"))],
        ..Default::default()
    };

    let vcs = vcgen.gen_function(&caller, &caller_spec);

    // Should NOT detect violation since *p gets substituted to *global_ptr
    // which is allowed by caller's assigns
    let assigns_vcs: Vec<_> = vcs
        .iter()
        .filter(|vc| vc.kind == VCKind::AssignsClause)
        .collect();
    assert_eq!(
        assigns_vcs.len(),
        0,
        "After param substitution, *p should become *global_ptr which is allowed"
    );
}

#[test]
fn test_param_substitution_assigns_violation() {
    // Test that param substitution correctly detects violations
    let mut vcgen = VCGen::new();

    // Register callee spec: write_ptr(p) assigns *p
    let callee_spec = FuncSpec {
        params: vec!["p".into()],
        requires: vec![],
        ensures: vec![],
        assigns: vec![Location::Deref(Spec::var("p"))],
        ..Default::default()
    };
    vcgen.register_func_spec("write_ptr", callee_spec);

    // Caller function that calls write_ptr(other_ptr)
    let caller = FuncDef {
        name: "caller".into(),
        return_type: CType::void(),
        params: vec![],
        body: Box::new(CStmt::Expr(CExpr::call(
            CExpr::var("write_ptr"),
            vec![CExpr::var("other_ptr")],
        ))),
        variadic: false,
        storage: StorageClass::Auto,
    };

    // Caller only allows modifying *my_ptr - should generate violation for *other_ptr
    let caller_spec = FuncSpec {
        requires: vec![],
        ensures: vec![],
        assigns: vec![Location::Deref(Spec::var("my_ptr"))],
        ..Default::default()
    };

    let vcs = vcgen.gen_function(&caller, &caller_spec);

    // Should detect violation since *p gets substituted to *other_ptr
    // which is NOT allowed by caller's assigns (only *my_ptr is allowed)
    let assigns_vcs: Vec<_> = vcs
        .iter()
        .filter(|vc| vc.kind == VCKind::AssignsClause)
        .collect();
    assert_eq!(
        assigns_vcs.len(),
        1,
        "After param substitution, *p should become *other_ptr which violates assigns"
    );
}

#[test]
fn test_param_substitution_multiple_params() {
    // Test substitution with multiple parameters
    let mut vcgen = VCGen::new();

    // Register callee spec: swap(p, q) assigns *p, *q; requires valid(p) && valid(q)
    let callee_spec = FuncSpec {
        params: vec!["p".into(), "q".into()],
        requires: vec![Spec::valid(Spec::var("p")), Spec::valid(Spec::var("q"))],
        ensures: vec![],
        assigns: vec![
            Location::Deref(Spec::var("p")),
            Location::Deref(Spec::var("q")),
        ],
        ..Default::default()
    };
    vcgen.register_func_spec("swap", callee_spec);

    // Call swap(x_ptr, y_ptr)
    let call_expr = CExpr::call(
        CExpr::var("swap"),
        vec![CExpr::var("x_ptr"), CExpr::var("y_ptr")],
    );
    let postcond = Spec::True;
    vcgen.wp_expr(&call_expr, &postcond);

    let vcs = vcgen.get_vcs();

    // Should generate 2 precondition VCs with substituted params
    let precond_vcs: Vec<_> = vcs
        .iter()
        .filter(|vc| vc.kind == VCKind::Precondition)
        .collect();
    assert_eq!(precond_vcs.len(), 2, "Should generate 2 precondition VCs");

    // Check that both VCs have the correct substitutions
    let vc_strs: Vec<String> = precond_vcs
        .iter()
        .map(|vc| format!("{:?}", vc.obligation))
        .collect();
    let combined = vc_strs.join(" ");
    assert!(
        combined.contains("x_ptr"),
        "Should contain x_ptr after substitution"
    );
    assert!(
        combined.contains("y_ptr"),
        "Should contain y_ptr after substitution"
    );
    assert!(
        !combined.contains("\"p\"") || combined.contains("swap"),
        "Should not contain standalone 'p' var after substitution"
    );
}

#[test]
fn test_old_resolution_in_postcondition() {
    // Test that \old(param) in callee postcondition becomes actual arg at call site
    // Callee: ensures \result == \old(x) + 1 (result is old value of x plus 1)
    // Call: increment(y)
    // After substitution: \result == \old(y) + 1
    // After old resolution: \result == y + 1 (y's value at call site)
    let mut vcgen = VCGen::new();

    // Register callee spec: increment(x) ensures \result == \old(x) + 1
    let callee_spec = FuncSpec {
        params: vec!["x".into()],
        requires: vec![],
        ensures: vec![Spec::eq(
            Spec::result(),
            Spec::binop(BinOp::Add, Spec::old(Spec::var("x")), Spec::int(1)),
        )],
        assigns: vec![],
        ..Default::default()
    };
    vcgen.register_func_spec("increment", callee_spec);

    // Call increment(y)
    let call_expr = CExpr::call(CExpr::var("increment"), vec![CExpr::var("y")]);
    let postcond = Spec::True;
    let wp = vcgen.wp_expr(&call_expr, &postcond);

    // WP should be: (increment(y) == y + 1) => True
    // Note: \old(y) should be resolved to just y
    match wp {
        Spec::Implies(antecedent, _) => {
            let spec_str = format!("{antecedent:?}");
            // Should contain "y" for the actual argument
            assert!(
                spec_str.contains("\"y\""),
                "Should contain 'y' in postcondition, got: {spec_str}"
            );
            // Should NOT contain Old wrapper (old should be resolved)
            assert!(
                !spec_str.contains("Old("),
                "Should not contain Old() after resolution, got: {spec_str}"
            );
        }
        _ => panic!("Expected Implies, got {wp:?}"),
    }
}

#[test]
fn test_old_resolution_nested() {
    // Test nested \old expressions: \old(\old(x)) should become x
    let mut vcgen = VCGen::new();

    // Register callee with nested old (unusual but valid)
    let callee_spec = FuncSpec {
        params: vec!["x".into()],
        requires: vec![],
        ensures: vec![Spec::eq(
            Spec::result(),
            Spec::old(Spec::old(Spec::var("x"))), // \old(\old(x))
        )],
        assigns: vec![],
        ..Default::default()
    };
    vcgen.register_func_spec("nested_old", callee_spec);

    let call_expr = CExpr::call(CExpr::var("nested_old"), vec![CExpr::var("z")]);
    let wp = vcgen.wp_expr(&call_expr, &Spec::True);

    match wp {
        Spec::Implies(antecedent, _) => {
            let spec_str = format!("{antecedent:?}");
            // Should have resolved all Old wrappers
            assert!(
                !spec_str.contains("Old("),
                "All Old() wrappers should be resolved, got: {spec_str}"
            );
            assert!(
                spec_str.contains("\"z\""),
                "Should contain 'z' after resolution, got: {spec_str}"
            );
        }
        _ => panic!("Expected Implies, got {wp:?}"),
    }
}

#[test]
fn test_old_resolution_in_quantifier() {
    // Test \old inside a quantifier: \forall int i; arr[i] == \old(arr[i])
    let mut vcgen = VCGen::new();

    // Register callee: identity(arr, n) ensures \forall int i; 0 <= i < n ==> arr[i] == \old(arr[i])
    let callee_spec = FuncSpec {
        params: vec!["arr".into(), "n".into()],
        requires: vec![],
        ensures: vec![Spec::forall(
            "i",
            CType::int(),
            Spec::implies(
                Spec::and(vec![
                    Spec::ge(Spec::var("i"), Spec::int(0)),
                    Spec::lt(Spec::var("i"), Spec::var("n")),
                ]),
                Spec::eq(
                    Spec::Index {
                        base: Box::new(Spec::var("arr")),
                        index: Box::new(Spec::var("i")),
                    },
                    Spec::old(Spec::Index {
                        base: Box::new(Spec::var("arr")),
                        index: Box::new(Spec::var("i")),
                    }),
                ),
            ),
        )],
        assigns: vec![],
        ..Default::default()
    };
    vcgen.register_func_spec("identity", callee_spec);

    let call_expr = CExpr::call(
        CExpr::var("identity"),
        vec![CExpr::var("my_arr"), CExpr::var("len")],
    );
    let wp = vcgen.wp_expr(&call_expr, &Spec::True);

    match wp {
        Spec::Implies(antecedent, _) => {
            let spec_str = format!("{antecedent:?}");
            // Old should be resolved inside the quantifier
            assert!(
                !spec_str.contains("Old("),
                "Old() should be resolved inside quantifier, got: {spec_str}"
            );
            // Should have my_arr and len substituted
            assert!(
                spec_str.contains("my_arr"),
                "Should contain 'my_arr' after substitution, got: {spec_str}"
            );
            assert!(
                spec_str.contains("len"),
                "Should contain 'len' after substitution, got: {spec_str}"
            );
        }
        _ => panic!("Expected Implies, got {wp:?}"),
    }
}

#[test]
fn test_old_resolution_preserves_structure() {
    // Test that old resolution preserves the structure of complex Spec expressions
    // (Note: Spec::Expr containing CExpr doesn't support param substitution currently)
    let mut vcgen = VCGen::new();

    // Use Spec-level constructs for proper substitution
    // Register: swap_values(a, b) ensures result_a == \old(b_val) && result_b == \old(a_val)
    // Using Spec::Var for values that can be substituted
    let callee_spec = FuncSpec {
        params: vec!["a_val".into(), "b_val".into()],
        requires: vec![],
        ensures: vec![
            // result_a == \old(b_val)
            Spec::eq(Spec::var("result_a"), Spec::old(Spec::var("b_val"))),
            // result_b == \old(a_val)
            Spec::eq(Spec::var("result_b"), Spec::old(Spec::var("a_val"))),
        ],
        assigns: vec![],
        ..Default::default()
    };
    vcgen.register_func_spec("swap_values", callee_spec);

    // Call swap_values(x, y) - should substitute a_val->x, b_val->y
    let call_expr = CExpr::call(
        CExpr::var("swap_values"),
        vec![CExpr::var("x"), CExpr::var("y")],
    );
    let wp = vcgen.wp_expr(&call_expr, &Spec::True);

    match wp {
        Spec::Implies(antecedent, _) => {
            let spec_str = format!("{antecedent:?}");
            // Old should be resolved
            assert!(
                !spec_str.contains("Old("),
                "Old() should be resolved, got: {spec_str}"
            );
            // Should have x and y substituted for a_val and b_val
            assert!(
                spec_str.contains("\"x\""),
                "Should contain 'x' after substitution, got: {spec_str}"
            );
            assert!(
                spec_str.contains("\"y\""),
                "Should contain 'y' after substitution, got: {spec_str}"
            );
            // Should NOT contain original params
            assert!(
                !spec_str.contains("a_val"),
                "Should not contain 'a_val', got: {spec_str}"
            );
            assert!(
                !spec_str.contains("b_val"),
                "Should not contain 'b_val', got: {spec_str}"
            );
        }
        _ => panic!("Expected Implies, got {wp:?}"),
    }
}

#[test]
fn test_subst_expr_simple_var() {
    // Test substituting a variable in a C expression
    let vcgen = VCGen::new();

    // CExpr: x + 1, substitute y for x
    let expr = CExpr::binop(BinOp::Add, CExpr::var("x"), CExpr::int(1));
    let result = vcgen.subst_expr(&expr, "x", &Spec::var("y"));

    // Should get y + 1
    match result {
        CExpr::BinOp {
            op: BinOp::Add,
            left,
            right,
        } => {
            assert!(matches!(*left, CExpr::Var(ref s) if s == "y"));
            assert!(matches!(*right, CExpr::IntLit(1)));
        }
        _ => panic!("Expected BinOp Add, got {result:?}"),
    }
}

#[test]
fn test_subst_expr_int_replacement() {
    // Test substituting with an integer spec
    let vcgen = VCGen::new();

    // CExpr: x * 2, substitute 5 for x
    let expr = CExpr::binop(BinOp::Mul, CExpr::var("x"), CExpr::int(2));
    let result = vcgen.subst_expr(&expr, "x", &Spec::int(5));

    // Should get 5 * 2
    match result {
        CExpr::BinOp {
            op: BinOp::Mul,
            left,
            right,
        } => {
            assert!(matches!(*left, CExpr::IntLit(5)));
            assert!(matches!(*right, CExpr::IntLit(2)));
        }
        _ => panic!("Expected BinOp Mul, got {result:?}"),
    }
}

#[test]
fn test_subst_expr_nested() {
    // Test substituting in nested expressions
    let vcgen = VCGen::new();

    // CExpr: (x + 1) * (x - 1), substitute y for x
    let expr = CExpr::binop(
        BinOp::Mul,
        CExpr::binop(BinOp::Add, CExpr::var("x"), CExpr::int(1)),
        CExpr::binop(BinOp::Sub, CExpr::var("x"), CExpr::int(1)),
    );
    let result = vcgen.subst_expr(&expr, "x", &Spec::var("y"));

    // Should have substituted all occurrences
    match result {
        CExpr::BinOp {
            op: BinOp::Mul,
            left,
            right,
        } => match (*left, *right) {
            (CExpr::BinOp { left: l1, .. }, CExpr::BinOp { left: l2, .. }) => {
                assert!(matches!(*l1, CExpr::Var(ref s) if s == "y"));
                assert!(matches!(*l2, CExpr::Var(ref s) if s == "y"));
            }
            _ => panic!("Expected nested BinOps"),
        },
        _ => panic!("Expected BinOp Mul"),
    }
}

#[test]
fn test_subst_expr_unary() {
    // Test substituting in unary expressions
    let vcgen = VCGen::new();

    // CExpr: -x, substitute 10 for x
    let expr = CExpr::unary(UnaryOp::Neg, CExpr::var("x"));
    let result = vcgen.subst_expr(&expr, "x", &Spec::int(10));

    // Should get -10
    match result {
        CExpr::UnaryOp {
            op: UnaryOp::Neg,
            operand,
        } => {
            assert!(matches!(*operand, CExpr::IntLit(10)));
        }
        _ => panic!("Expected UnaryOp Neg, got {result:?}"),
    }
}

#[test]
fn test_subst_expr_conditional() {
    // Test substituting in conditional (ternary) expressions
    let vcgen = VCGen::new();

    // CExpr: x > 0 ? x : -x, substitute y for x
    let expr = CExpr::Conditional {
        cond: Box::new(CExpr::binop(BinOp::Gt, CExpr::var("x"), CExpr::int(0))),
        then_expr: Box::new(CExpr::var("x")),
        else_expr: Box::new(CExpr::unary(UnaryOp::Neg, CExpr::var("x"))),
    };
    let result = vcgen.subst_expr(&expr, "x", &Spec::var("y"));

    // All x should be replaced with y
    match result {
        CExpr::Conditional {
            cond,
            then_expr,
            else_expr,
        } => {
            match *cond {
                CExpr::BinOp { left, .. } => {
                    assert!(matches!(*left, CExpr::Var(ref s) if s == "y"));
                }
                _ => panic!("Expected BinOp in cond"),
            }
            assert!(matches!(*then_expr, CExpr::Var(ref s) if s == "y"));
            match *else_expr {
                CExpr::UnaryOp { operand, .. } => {
                    assert!(matches!(*operand, CExpr::Var(ref s) if s == "y"));
                }
                _ => panic!("Expected UnaryOp in else"),
            }
        }
        _ => panic!("Expected Conditional"),
    }
}

#[test]
fn test_subst_expr_index() {
    // Test substituting in array index expressions
    let vcgen = VCGen::new();

    // CExpr: arr[i], substitute j for i
    let expr = CExpr::index(CExpr::var("arr"), CExpr::var("i"));
    let result = vcgen.subst_expr(&expr, "i", &Spec::var("j"));

    match result {
        CExpr::Index { array, index } => {
            assert!(matches!(*array, CExpr::Var(ref s) if s == "arr"));
            assert!(matches!(*index, CExpr::Var(ref s) if s == "j"));
        }
        _ => panic!("Expected Index"),
    }
}

#[test]
fn test_subst_expr_call() {
    // Test substituting in function call arguments
    let vcgen = VCGen::new();

    // CExpr: foo(x, y), substitute z for x
    let expr = CExpr::call(CExpr::var("foo"), vec![CExpr::var("x"), CExpr::var("y")]);
    let result = vcgen.subst_expr(&expr, "x", &Spec::var("z"));

    match result {
        CExpr::Call { func, args } => {
            assert!(matches!(*func, CExpr::Var(ref s) if s == "foo"));
            assert_eq!(args.len(), 2);
            assert!(matches!(&args[0], CExpr::Var(ref s) if s == "z"));
            assert!(matches!(&args[1], CExpr::Var(ref s) if s == "y"));
        }
        _ => panic!("Expected Call"),
    }
}

#[test]
fn test_subst_expr_member() {
    // Test substituting in member access
    let vcgen = VCGen::new();

    // CExpr: obj.field, substitute s for obj
    let expr = CExpr::Member {
        object: Box::new(CExpr::var("obj")),
        field: "field".into(),
    };
    let result = vcgen.subst_expr(&expr, "obj", &Spec::var("s"));

    match result {
        CExpr::Member { object, field } => {
            assert!(matches!(*object, CExpr::Var(ref s) if s == "s"));
            assert_eq!(field, "field");
        }
        _ => panic!("Expected Member"),
    }
}

#[test]
fn test_subst_expr_spec_with_binop() {
    // Test substituting with a Spec that has a BinOp
    let vcgen = VCGen::new();

    // CExpr: x * 2, substitute (a + b) for x
    let expr = CExpr::binop(BinOp::Mul, CExpr::var("x"), CExpr::int(2));
    let replacement = Spec::binop(BinOp::Add, Spec::var("a"), Spec::var("b"));
    let result = vcgen.subst_expr(&expr, "x", &replacement);

    // Should get (a + b) * 2
    match result {
        CExpr::BinOp {
            op: BinOp::Mul,
            left,
            right,
        } => {
            match *left {
                CExpr::BinOp {
                    op: BinOp::Add,
                    left: l,
                    right: r,
                } => {
                    assert!(matches!(*l, CExpr::Var(ref s) if s == "a"));
                    assert!(matches!(*r, CExpr::Var(ref s) if s == "b"));
                }
                _ => panic!("Expected inner BinOp Add"),
            }
            assert!(matches!(*right, CExpr::IntLit(2)));
        }
        _ => panic!("Expected BinOp Mul"),
    }
}

#[test]
fn test_spec_to_cexpr_simple() {
    // Test spec_to_cexpr for simple specs
    let vcgen = VCGen::new();

    // Spec::Int
    assert!(matches!(
        vcgen.spec_to_cexpr(&Spec::int(42)),
        Some(CExpr::IntLit(42))
    ));

    // Spec::Var
    match vcgen.spec_to_cexpr(&Spec::var("x")) {
        Some(CExpr::Var(name)) => assert_eq!(name, "x"),
        _ => panic!("Expected Var"),
    }

    // Spec::Null
    assert!(matches!(
        vcgen.spec_to_cexpr(&Spec::Null),
        Some(CExpr::IntLit(0))
    ));
}

#[test]
fn test_spec_to_cexpr_complex() {
    // Test spec_to_cexpr for complex specs
    let vcgen = VCGen::new();

    // Spec::BinOp
    let binop_spec = Spec::binop(BinOp::Add, Spec::var("x"), Spec::int(1));
    match vcgen.spec_to_cexpr(&binop_spec) {
        Some(CExpr::BinOp {
            op: BinOp::Add,
            left,
            right,
        }) => {
            assert!(matches!(*left, CExpr::Var(ref s) if s == "x"));
            assert!(matches!(*right, CExpr::IntLit(1)));
        }
        _ => panic!("Expected BinOp"),
    }

    // Spec::UnaryOp
    let unary_spec = Spec::UnaryOp {
        op: UnaryOp::Neg,
        operand: Box::new(Spec::var("y")),
    };
    match vcgen.spec_to_cexpr(&unary_spec) {
        Some(CExpr::UnaryOp {
            op: UnaryOp::Neg,
            operand,
        }) => {
            assert!(matches!(*operand, CExpr::Var(ref s) if s == "y"));
        }
        _ => panic!("Expected UnaryOp"),
    }

    // Spec::Index
    let index_spec = Spec::Index {
        base: Box::new(Spec::var("arr")),
        index: Box::new(Spec::int(0)),
    };
    match vcgen.spec_to_cexpr(&index_spec) {
        Some(CExpr::Index { array, index }) => {
            assert!(matches!(*array, CExpr::Var(ref s) if s == "arr"));
            assert!(matches!(*index, CExpr::IntLit(0)));
        }
        _ => panic!("Expected Index"),
    }
}

#[test]
fn test_spec_to_cexpr_non_convertible() {
    // Test spec_to_cexpr returns None for non-convertible specs
    let vcgen = VCGen::new();

    // Logic-only constructs should return None
    assert!(vcgen.spec_to_cexpr(&Spec::True).is_none());
    assert!(vcgen.spec_to_cexpr(&Spec::False).is_none());
    assert!(vcgen.spec_to_cexpr(&Spec::Result).is_none());
    assert!(vcgen.spec_to_cexpr(&Spec::old(Spec::var("x"))).is_none());
    assert!(vcgen
        .spec_to_cexpr(&Spec::forall("x", CType::int(), Spec::True))
        .is_none());
    assert!(vcgen
        .spec_to_cexpr(&Spec::exists("x", CType::int(), Spec::True))
        .is_none());
    assert!(vcgen
        .spec_to_cexpr(&Spec::implies(Spec::True, Spec::True))
        .is_none());
    assert!(vcgen.spec_to_cexpr(&Spec::and(vec![Spec::True])).is_none());
    assert!(vcgen.spec_to_cexpr(&Spec::or(vec![Spec::True])).is_none());
    assert!(vcgen.spec_to_cexpr(&Spec::not(Spec::True)).is_none());
    assert!(vcgen.spec_to_cexpr(&Spec::valid(Spec::var("p"))).is_none());
}

#[test]
fn test_subst_expr_in_spec_expr() {
    // Test that subst_var correctly handles Spec::Expr with our new implementation
    let vcgen = VCGen::new();

    // Create a Spec::Expr containing a CExpr with variable x
    let cexpr = CExpr::binop(BinOp::Add, CExpr::var("x"), CExpr::int(1));
    let spec = Spec::Expr(cexpr);

    // Substitute y for x
    let result = vcgen.subst_var(&spec, "x", &Spec::var("y"));

    // The result should have y substituted
    match result {
        Spec::Expr(CExpr::BinOp {
            op: BinOp::Add,
            left,
            right,
        }) => {
            assert!(matches!(*left, CExpr::Var(ref s) if s == "y"));
            assert!(matches!(*right, CExpr::IntLit(1)));
        }
        _ => panic!("Expected Spec::Expr with BinOp, got {result:?}"),
    }
}
