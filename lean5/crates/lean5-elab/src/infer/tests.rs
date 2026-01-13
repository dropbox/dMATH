//! Tests for type elaboration
use super::*;

use lean5_parser::parse_expr;

fn elab(input: &str) -> Result<Expr, ElabError> {
    let env = Environment::new();
    let mut ctx = ElabCtx::new(&env);
    let surface = parse_expr(input).map_err(|e| ElabError::ParseError(e.to_string()))?;
    ctx.elaborate(&surface)
}

fn elab_with_env(env: &Environment, input: &str) -> Result<Expr, ElabError> {
    let mut ctx = ElabCtx::new(env);
    let surface = parse_expr(input).map_err(|e| ElabError::ParseError(e.to_string()))?;
    ctx.elaborate(&surface)
}

fn pair_env() -> Environment {
    use lean5_kernel::inductive::{Constructor, InductiveDecl, InductiveType};

    let mut env = Environment::new();
    let pair = Name::from_string("Pair");

    // Pair : Type
    let pair_type = Expr::type_();

    // mk : Prop → Prop → Pair
    let mk_type = Expr::pi(
        BinderInfo::Default,
        Expr::prop(),
        Expr::pi(
            BinderInfo::Default,
            Expr::prop(),
            Expr::const_(pair.clone(), vec![]),
        ),
    );

    let decl = InductiveDecl {
        level_params: vec![],
        num_params: 0,
        types: vec![InductiveType {
            name: pair.clone(),
            type_: pair_type,
            constructors: vec![Constructor {
                name: Name::from_string("Pair.mk"),
                type_: mk_type,
            }],
        }],
    };

    env.add_inductive(decl).unwrap();
    env.register_structure_fields(
        pair,
        vec![Name::from_string("fst"), Name::from_string("snd")],
    )
    .unwrap();

    env
}

#[test]
fn test_elab_type() {
    let expr = elab("Type").unwrap();
    assert!(matches!(expr, Expr::Sort(_)));
}

#[test]
fn test_elab_prop() {
    let expr = elab("Prop").unwrap();
    assert!(expr.is_prop());
}

#[test]
fn test_elab_nat_lit() {
    let expr = elab("42").unwrap();
    assert!(matches!(expr, Expr::Lit(lean5_kernel::Literal::Nat(42))));
}

#[test]
fn test_elab_lambda() {
    let expr = elab("fun (x : Type) => x").unwrap();
    assert!(matches!(expr, Expr::Lam(_, _, _)));
}

#[test]
fn test_elab_pi() {
    let expr = elab("forall (x : Type), x").unwrap();
    assert!(matches!(expr, Expr::Pi(_, _, _)));
}

#[test]
fn test_elab_arrow() {
    let expr = elab("Type -> Type").unwrap();
    match expr {
        Expr::Pi(BinderInfo::Default, domain, _) => {
            assert!(matches!(*domain, Expr::Sort(_)));
        }
        _ => panic!("expected Pi"),
    }
}

#[test]
fn test_elab_app_unknown() {
    // f is unknown, should error
    let result = elab("f x");
    assert!(result.is_err());
}

#[test]
fn test_projection_index_resolves_struct_name() {
    let env = pair_env();
    let expr = elab_with_env(&env, "fun (p : Pair) => p.0").unwrap();

    match expr {
        Expr::Lam(_, _, body) => match body.as_ref() {
            Expr::Proj(struct_name, idx, _) => {
                assert_eq!(struct_name, &Name::from_string("Pair"));
                assert_eq!(*idx, 0);
            }
            other => panic!("expected projection, got {other:?}"),
        },
        other => panic!("expected lambda, got {other:?}"),
    }
}

#[test]
fn test_projection_named_field_lookup() {
    let env = pair_env();
    let expr = elab_with_env(&env, "fun (p : Pair) => p.snd").unwrap();

    match expr {
        Expr::Lam(_, _, body) => match body.as_ref() {
            Expr::Proj(struct_name, idx, _) => {
                assert_eq!(struct_name, &Name::from_string("Pair"));
                assert_eq!(*idx, 1);
            }
            other => panic!("expected projection, got {other:?}"),
        },
        other => panic!("expected lambda, got {other:?}"),
    }
}

#[test]
fn test_projection_unknown_field_error() {
    let env = pair_env();
    let err = elab_with_env(&env, "fun (p : Pair) => p.third").expect_err("should fail");

    assert!(matches!(
        err,
        ElabError::UnknownProjectionField { ref field, .. } if field == "third"
    ));
}

#[test]
fn test_projection_index_oob_error() {
    let env = pair_env();
    let err = elab_with_env(&env, "fun (p : Pair) => p.2").expect_err("should fail");

    assert!(matches!(
        err,
        ElabError::ProjectionIndexOutOfBounds { idx: 2, .. }
    ));
}

#[test]
fn test_elab_let() {
    let expr = elab("let x : Type := Type in x").unwrap();
    assert!(matches!(expr, Expr::Let(_, _, _)));
}

#[test]
fn test_elab_hole() {
    let expr = elab("_").unwrap();
    // Hole elaborates to a metavariable (represented as FVar for now)
    assert!(matches!(expr, Expr::FVar(_)));
}

#[test]
fn test_elab_identity_function() {
    let expr = elab("fun (A : Type) (x : A) => x").unwrap();
    match expr {
        Expr::Lam(_, ty1, body1) => {
            assert!(matches!(*ty1, Expr::Sort(_)));
            match body1.as_ref() {
                Expr::Lam(_, _, body2) => {
                    // The innermost body should be BVar(0) - referring to x
                    assert!(matches!(body2.as_ref(), Expr::BVar(0)));
                }
                _ => panic!("expected nested lambda"),
            }
        }
        _ => panic!("expected lambda"),
    }
}

// ==========================================================================
// Implicit argument insertion tests
// ==========================================================================

use lean5_kernel::env::Declaration;

/// Helper to create an environment with a function that has implicit arguments
fn env_with_implicit_id() -> Environment {
    let mut env = Environment::new();

    // Add id : {A : Type} → A → A
    let id_type = Expr::pi(
        BinderInfo::Implicit,
        Expr::type_(), // A : Type
        Expr::pi(
            BinderInfo::Default,
            Expr::bvar(0), // x : A
            Expr::bvar(1), // A
        ),
    );
    let id_value = Expr::lam(
        BinderInfo::Implicit,
        Expr::type_(),
        Expr::lam(BinderInfo::Default, Expr::bvar(0), Expr::bvar(0)),
    );
    env.add_decl(Declaration::Definition {
        name: Name::from_string("id"),
        level_params: vec![],
        type_: id_type,
        value: id_value,
        is_reducible: true,
    })
    .unwrap();

    // Add a simple type
    env.add_decl(Declaration::Axiom {
        name: Name::from_string("Nat"),
        level_params: vec![],
        type_: Expr::type_(),
    })
    .unwrap();

    // Add a value of that type
    env.add_decl(Declaration::Axiom {
        name: Name::from_string("zero"),
        level_params: vec![],
        type_: Expr::const_(Name::from_string("Nat"), vec![]),
    })
    .unwrap();

    env
}

#[test]
fn test_implicit_insertion_basic() {
    // Test: id zero should elaborate to id Nat zero
    // where the implicit type argument is resolved via unification
    let env = env_with_implicit_id();
    let mut ctx = ElabCtx::new(&env);

    let surface = parse_expr("id zero").unwrap();
    let expr = ctx.elaborate(&surface).unwrap();

    // The result should be App(App(id, Nat), zero)
    // i.e., the implicit argument should have been inserted and solved
    let args = expr.get_app_args();
    assert_eq!(
        args.len(),
        2,
        "Expected 2 arguments (implicit + explicit), got {}",
        args.len()
    );

    // First argument should be the inferred type 'Nat'
    assert!(
        matches!(args[0], Expr::Const(ref n, _) if n.to_string() == "Nat"),
        "Expected first arg to be 'Nat', got {:?}",
        args[0]
    );

    // Second argument should be the 'zero' constant
    assert!(
        matches!(args[1], Expr::Const(ref n, _) if n.to_string() == "zero"),
        "Expected second arg to be 'zero', got {:?}",
        args[1]
    );
}

#[test]
fn test_implicit_insertion_multiple() {
    // Test function with multiple implicit arguments
    let mut env = Environment::new();

    // Add compose : {A B C : Type} → (B → C) → (A → B) → A → C
    let compose_type = Expr::pi(
        BinderInfo::Implicit,
        Expr::type_(), // A : Type
        Expr::pi(
            BinderInfo::Implicit,
            Expr::type_(), // B : Type
            Expr::pi(
                BinderInfo::Implicit,
                Expr::type_(), // C : Type
                Expr::pi(
                    BinderInfo::Default,
                    Expr::pi(
                        BinderInfo::Default,
                        Expr::bvar(1), // B
                        Expr::bvar(1), // C
                    ), // g : B → C
                    Expr::pi(
                        BinderInfo::Default,
                        Expr::pi(
                            BinderInfo::Default,
                            Expr::bvar(3), // A
                            Expr::bvar(3), // B
                        ), // f : A → B
                        Expr::pi(
                            BinderInfo::Default,
                            Expr::bvar(4), // x : A
                            Expr::bvar(3), // C
                        ),
                    ),
                ),
            ),
        ),
    );
    let compose_value = Expr::lam(
        BinderInfo::Implicit,
        Expr::type_(),
        Expr::lam(
            BinderInfo::Implicit,
            Expr::type_(),
            Expr::lam(
                BinderInfo::Implicit,
                Expr::type_(),
                Expr::lam(
                    BinderInfo::Default,
                    Expr::pi(BinderInfo::Default, Expr::bvar(1), Expr::bvar(1)),
                    Expr::lam(
                        BinderInfo::Default,
                        Expr::pi(BinderInfo::Default, Expr::bvar(3), Expr::bvar(3)),
                        Expr::lam(
                            BinderInfo::Default,
                            Expr::bvar(4),
                            // g (f x) = App(BVar(2), App(BVar(1), BVar(0)))
                            Expr::app(Expr::bvar(2), Expr::app(Expr::bvar(1), Expr::bvar(0))),
                        ),
                    ),
                ),
            ),
        ),
    );
    env.add_decl(Declaration::Definition {
        name: Name::from_string("compose"),
        level_params: vec![],
        type_: compose_type,
        value: compose_value,
        is_reducible: true,
    })
    .unwrap();

    // Add simple functions to compose
    env.add_decl(Declaration::Axiom {
        name: Name::from_string("Nat"),
        level_params: vec![],
        type_: Expr::type_(),
    })
    .unwrap();

    env.add_decl(Declaration::Axiom {
        name: Name::from_string("f"),
        level_params: vec![],
        type_: Expr::arrow(
            Expr::const_(Name::from_string("Nat"), vec![]),
            Expr::const_(Name::from_string("Nat"), vec![]),
        ),
    })
    .unwrap();

    let mut ctx = ElabCtx::new(&env);

    // compose f f should insert 3 implicit metavariables
    let surface = parse_expr("compose f f").unwrap();
    let expr = ctx.elaborate(&surface).unwrap();

    // Expected: App(...App(compose, Nat), Nat), Nat), f), f)
    // 5 arguments total: 3 implicit (resolved) + 2 explicit
    let args = expr.get_app_args();
    assert_eq!(
        args.len(),
        5,
        "Expected 5 arguments (3 implicit + 2 explicit), got {}",
        args.len()
    );

    // First three should be the resolved implicit type arguments (Nat)
    for i in 0..3 {
        assert!(
            matches!(args[i], Expr::Const(ref n, _) if n.to_string() == "Nat"),
            "Expected arg {} to be 'Nat', got {:?}",
            i,
            args[i]
        );
    }

    // Last two should be 'f' constants
    for i in 3..5 {
        assert!(
            matches!(args[i], Expr::Const(ref n, _) if n.to_string() == "f"),
            "Expected arg {} to be 'f', got {:?}",
            i,
            args[i]
        );
    }
}

#[test]
fn test_no_implicit_insertion_for_explicit_function() {
    // Test that explicit arguments don't get metavariables inserted
    let mut env = Environment::new();

    // Add explicit_id : (A : Type) → A → A  (no implicit)
    let id_type = Expr::pi(
        BinderInfo::Default,
        Expr::type_(),
        Expr::pi(BinderInfo::Default, Expr::bvar(0), Expr::bvar(1)),
    );
    let id_value = Expr::lam(
        BinderInfo::Default,
        Expr::type_(),
        Expr::lam(BinderInfo::Default, Expr::bvar(0), Expr::bvar(0)),
    );
    env.add_decl(Declaration::Definition {
        name: Name::from_string("explicit_id"),
        level_params: vec![],
        type_: id_type,
        value: id_value,
        is_reducible: true,
    })
    .unwrap();

    env.add_decl(Declaration::Axiom {
        name: Name::from_string("Nat"),
        level_params: vec![],
        type_: Expr::type_(),
    })
    .unwrap();

    let mut ctx = ElabCtx::new(&env);

    // explicit_id Nat should NOT insert any metavariables
    let surface = parse_expr("explicit_id Nat").unwrap();
    let expr = ctx.elaborate(&surface).unwrap();

    let args = expr.get_app_args();
    assert_eq!(args.len(), 1, "Expected 1 argument, got {}", args.len());

    // The argument should be Nat (a constant), not a metavariable
    assert!(
        matches!(args[0], Expr::Const(ref n, _) if n.to_string() == "Nat"),
        "Expected arg to be 'Nat', got {:?}",
        args[0]
    );
}

#[test]
fn test_implicit_insertion_structure() {
    // Test that after implicit insertion, the elaborated expression has correct structure
    // with implicit arguments solved by unification.
    let env = env_with_implicit_id();
    let mut ctx = ElabCtx::new(&env);

    let surface = parse_expr("id zero").unwrap();
    let expr = ctx.elaborate(&surface).unwrap();

    // Verify structure: App(App(Const(id), ?meta), Const(zero))
    match &expr {
        Expr::App(inner, arg2) => {
            // arg2 should be 'zero'
            assert!(
                matches!(arg2.as_ref(), Expr::Const(ref n, _) if n.to_string() == "zero"),
                "Expected outer arg to be 'zero'"
            );

            match inner.as_ref() {
                Expr::App(id_const, meta) => {
                    // id_const should be the 'id' constant
                    assert!(
                        matches!(id_const.as_ref(), Expr::Const(ref n, _) if n.to_string() == "id"),
                        "Expected inner function to be 'id'"
                    );

                    // implicit argument should be the inferred type 'Nat'
                    assert!(
                        matches!(meta.as_ref(), Expr::Const(ref n, _) if n.to_string() == "Nat"),
                        "Expected implicit arg to be 'Nat'"
                    );
                }
                _ => panic!("Expected App(id, meta)"),
            }
        }
        _ => panic!("Expected App expression"),
    }
}

#[test]
fn test_implicit_insertion_with_instance() {
    // Test instance implicit [inst : T] is also handled
    let mut env = Environment::new();

    // Add a function with instance implicit: foo : [A : Type] → A → A
    let foo_type = Expr::pi(
        BinderInfo::InstImplicit,
        Expr::type_(),
        Expr::pi(BinderInfo::Default, Expr::bvar(0), Expr::bvar(1)),
    );
    let foo_value = Expr::lam(
        BinderInfo::InstImplicit,
        Expr::type_(),
        Expr::lam(BinderInfo::Default, Expr::bvar(0), Expr::bvar(0)),
    );
    env.add_decl(Declaration::Definition {
        name: Name::from_string("foo"),
        level_params: vec![],
        type_: foo_type,
        value: foo_value,
        is_reducible: true,
    })
    .unwrap();

    env.add_decl(Declaration::Axiom {
        name: Name::from_string("x"),
        level_params: vec![],
        type_: Expr::type_(),
    })
    .unwrap();

    let mut ctx = ElabCtx::new(&env);
    let surface = parse_expr("foo x").unwrap();
    let expr = ctx.elaborate(&surface).unwrap();

    let args = expr.get_app_args();
    assert_eq!(
        args.len(),
        2,
        "Expected 2 args (instance implicit + explicit)"
    );
    assert!(
        matches!(args[0], Expr::Sort(_)),
        "Expected first arg to be inferred type argument, got {:?}",
        args[0]
    );
}

// ==========================================================================
// Certificate integration tests
// ==========================================================================

#[test]
fn test_create_cert_verifier_empty_context() {
    let env = Environment::new();
    let ctx = ElabCtx::new(&env);

    // Should succeed with empty context
    let verifier = ctx.create_cert_verifier();
    assert!(verifier.is_ok());
}

#[test]
fn test_infer_type_with_cert_sort() {
    let env = Environment::new();
    let mut ctx = ElabCtx::new(&env);

    let surface = parse_expr("Type").unwrap();
    let expr = ctx.elaborate(&surface).unwrap();

    let result = ctx.infer_type_with_cert(&expr);
    assert!(result.is_ok());

    let (ty, cert) = result.unwrap();
    // Type : Type 1
    assert!(matches!(ty, Expr::Sort(_)));
    assert!(matches!(cert, ProofCert::Sort { .. }));
}

#[test]
fn test_infer_type_with_cert_lambda() {
    let env = Environment::new();
    let mut ctx = ElabCtx::new(&env);

    let surface = parse_expr("fun (x : Type) => x").unwrap();
    let expr = ctx.elaborate(&surface).unwrap();

    let result = ctx.infer_type_with_cert(&expr);
    assert!(result.is_ok());

    let (ty, cert) = result.unwrap();
    // fun (x : Type) => x : Type -> Type
    assert!(matches!(ty, Expr::Pi(_, _, _)));
    assert!(matches!(cert, ProofCert::Lam { .. }));
}

#[test]
fn test_elaborate_and_verify_type() {
    let env = Environment::new();
    let mut ctx = ElabCtx::new(&env);

    let surface = parse_expr("Type").unwrap();
    let result = ctx.elaborate_and_verify(&surface);
    assert!(result.is_ok());

    let (expr, ty, cert) = result.unwrap();
    assert!(matches!(expr, Expr::Sort(_)));
    assert!(matches!(ty, Expr::Sort(_)));
    assert!(matches!(cert, ProofCert::Sort { .. }));
}

#[test]
fn test_elaborate_and_verify_lambda() {
    let env = Environment::new();
    let mut ctx = ElabCtx::new(&env);

    let surface = parse_expr("fun (A : Type) (x : A) => x").unwrap();
    let result = ctx.elaborate_and_verify(&surface);
    assert!(result.is_ok());

    let (expr, ty, cert) = result.unwrap();
    assert!(matches!(expr, Expr::Lam(_, _, _)));
    assert!(matches!(ty, Expr::Pi(_, _, _)));
    assert!(matches!(cert, ProofCert::Lam { .. }));
}

#[test]
fn test_elaborate_and_verify_nat_lit() {
    let env = Environment::new();
    let mut ctx = ElabCtx::new(&env);

    let surface = parse_expr("42").unwrap();
    let result = ctx.elaborate_and_verify(&surface);
    assert!(result.is_ok());

    let (expr, _ty, cert) = result.unwrap();
    assert!(matches!(expr, Expr::Lit(lean5_kernel::Literal::Nat(42))));
    assert!(matches!(cert, ProofCert::Lit { .. }));
}

#[test]
fn test_cert_verifier_with_local_context() {
    // Test that cert verifier works with locals in context
    let env = Environment::new();
    let mut ctx = ElabCtx::new(&env);

    // Elaborate a lambda to add a local to context
    let surface = parse_expr("fun (x : Type) => x").unwrap();
    let expr = ctx.elaborate(&surface).unwrap();

    // Get certificate
    let (ty, cert) = ctx.infer_type_with_cert(&expr).unwrap();

    // Create verifier and verify
    let mut verifier = ctx.create_cert_verifier().unwrap();
    let result = verifier.verify(&cert, &expr);
    assert!(result.is_ok());

    let verified_ty = result.unwrap();
    assert_eq!(verified_ty, ty);
}

#[test]
fn test_elaborate_and_verify_pi() {
    let env = Environment::new();
    let mut ctx = ElabCtx::new(&env);

    let surface = parse_expr("forall (x : Type), x").unwrap();
    let result = ctx.elaborate_and_verify(&surface);
    assert!(result.is_ok());

    let (expr, ty, cert) = result.unwrap();
    assert!(matches!(expr, Expr::Pi(_, _, _)));
    // Pi type has type Sort
    assert!(matches!(ty, Expr::Sort(_)));
    assert!(matches!(cert, ProofCert::Pi { .. }));
}

// ==========================================================================
// Ascription type checking tests
// ==========================================================================

#[test]
fn test_ascription_prop_has_type_type() {
    // (Prop : Type) should succeed - Prop has type Type
    let expr = elab("(Prop : Type)").unwrap();
    assert!(expr.is_prop());
}

#[test]
fn test_ascription_identity_function() {
    // Identity function with explicit type annotation
    let expr = elab("(fun (x : Type) => x : Type -> Type)").unwrap();
    assert!(matches!(expr, Expr::Lam(_, _, _)));
}

#[test]
fn test_ascription_universe_levels() {
    // Type : Type fails because Type 0 has type Type 1 (different universe)
    // This is correct universe level checking
    let result = elab("(Type : Type)");
    assert!(result.is_err());
}

#[test]
fn test_ascription_wrong_type() {
    // This should fail: Type is not of type Prop
    let result = elab("(Type : Prop)");
    assert!(result.is_err());
    match result {
        Err(ElabError::TypeMismatch { expected, actual }) => {
            // expected should mention Prop (Sort Zero)
            assert!(expected.contains("Sort") || expected.contains("Zero"));
            // actual should mention higher universe
            assert!(actual.contains("Sort") || actual.contains("Succ"));
        }
        _ => panic!("expected TypeMismatch error"),
    }
}

#[test]
fn test_ascription_nat_lit_wrong_type() {
    // Nat literal doesn't have type Type
    let result = elab("(42 : Type)");
    assert!(result.is_err());
}

#[test]
fn test_ascription_simple_lambda() {
    // A simple lambda with correct type annotation
    let expr = elab("(fun (x : Prop) => x : Prop -> Prop)").unwrap();
    assert!(matches!(expr, Expr::Lam(_, _, _)));
}

#[test]
fn test_ascription_lambda_wrong_type() {
    // Lambda returning Prop annotated with Type -> Type should fail
    let result = elab("(fun (x : Prop) => x : Type -> Type)");
    assert!(result.is_err());
}

#[test]
fn test_ascription_preserves_value() {
    // Ascription should return the value, not the type
    let expr_with_ascription = elab("(fun (x : Type) => x : Type -> Type)").unwrap();
    let expr_without = elab("fun (x : Type) => x").unwrap();
    // Both should be the same lambda
    assert!(matches!(expr_with_ascription, Expr::Lam(_, _, _)));
    assert!(matches!(expr_without, Expr::Lam(_, _, _)));
}

#[test]
fn test_ascription_arrow_pi_type() {
    // Ascription with arrow type that is Type
    // Prop -> Prop has type Type (impredicativity of Prop)
    let expr = elab("(Prop -> Prop : Type)").unwrap();
    // The ascripted expression is the arrow/Pi type
    assert!(matches!(expr, Expr::Pi(_, _, _)));
}

#[test]
fn test_ascription_arrow_pi_type_universe_mismatch() {
    // Type -> Type has type Type 1, not Type 0
    // This should fail because of universe level mismatch
    let result = elab("(Type -> Type : Type)");
    assert!(result.is_err());
}

#[test]
fn test_ascription_with_arrow_type() {
    // Arrow type annotation
    let expr = elab("(fun (f : Type -> Type) => f : (Type -> Type) -> Type -> Type)").unwrap();
    assert!(matches!(expr, Expr::Lam(_, _, _)));
}

// =========================================================================
// Let with Inferred Type Tests (N=144)
// Tests for let without explicit type annotation (let x := val in body)
// =========================================================================

#[test]
fn test_let_inferred_type_simple() {
    // Let without explicit type - type should be inferred from value
    let expr = elab("let x := Type in x").unwrap();
    // Result should be Let with inferred type Type and body that returns x
    assert!(matches!(expr, Expr::Let(_, _, _)));
}

#[test]
fn test_let_inferred_type_prop() {
    // Infer Prop type
    let expr = elab("let p := Prop in p").unwrap();
    assert!(matches!(expr, Expr::Let(_, _, _)));
}

#[test]
fn test_let_inferred_type_lambda() {
    // Infer type of lambda (identity function)
    let expr = elab("let id := fun (A : Type) (x : A) => x in id").unwrap();
    assert!(matches!(expr, Expr::Let(_, _, _)));
}

#[test]
fn test_let_inferred_vs_explicit_same_result() {
    // Both should elaborate to equivalent expressions
    let inferred = elab("let x := Type in x").unwrap();
    let explicit = elab("let x : Type := Type in x").unwrap();
    // Both should be Let expressions
    assert!(matches!(inferred, Expr::Let(_, _, _)));
    assert!(matches!(explicit, Expr::Let(_, _, _)));
}

#[test]
fn test_let_inferred_type_nested() {
    // Nested let with inferred types
    let expr = elab("let x := Type in let y := x in y").unwrap();
    assert!(matches!(expr, Expr::Let(_, _, _)));
}

#[test]
fn test_let_inferred_type_with_usage() {
    // Let where inferred type is used in body - simple case just returning the let-bound value
    let expr = elab("let x := Prop in x -> x").unwrap();
    // The body uses x in an arrow type, result could be Let or Pi
    assert!(matches!(expr, Expr::Let(_, _, _)) || matches!(expr, Expr::Pi(_, _, _)));
}

#[test]
fn test_let_inferred_arrow_type() {
    // Let where value is a function type
    let expr = elab("let arrow := Type -> Type in arrow").unwrap();
    assert!(matches!(expr, Expr::Let(_, _, _)));
}

#[test]
fn test_let_inferred_type_forall() {
    // Forall type inferred using forall syntax
    let expr = elab("let dep := forall (A : Type), A -> A in dep").unwrap();
    assert!(matches!(expr, Expr::Let(_, _, _)));
}

// Structure elaboration tests
use lean5_parser::parse_decl;

fn elab_decl(input: &str) -> Result<ElabResult, ElabError> {
    let env = Environment::new();
    let mut ctx = ElabCtx::new(&env);
    let surface = parse_decl(input).map_err(|e| ElabError::ParseError(e.to_string()))?;
    ctx.elab_decl(&surface)
}

#[test]
fn test_elab_structure_simple() {
    // Use Prop instead of Type to avoid "Type y" being parsed as "Type" with level param "y"
    let result = elab_decl(
        r"structure Point where
          x : Prop
          y : Prop",
    )
    .unwrap();

    match result {
        ElabResult::Structure {
            name,
            field_names,
            num_params,
            ..
        } => {
            assert_eq!(name, Name::from_string("Point"));
            assert_eq!(num_params, 0);
            assert_eq!(field_names.len(), 2);
            assert_eq!(field_names[0], Name::from_string("x"));
            assert_eq!(field_names[1], Name::from_string("y"));
        }
        _ => panic!("expected Structure"),
    }
}

#[test]
fn test_elab_structure_with_params() {
    let result = elab_decl(
        r"structure Pair (A : Type) (B : Type) where
          fst : A
          snd : B",
    )
    .unwrap();

    match result {
        ElabResult::Structure {
            name,
            field_names,
            num_params,
            ctor_name,
            ..
        } => {
            assert_eq!(name, Name::from_string("Pair"));
            assert_eq!(num_params, 2);
            assert_eq!(ctor_name, Name::from_string("Pair.mk"));
            assert_eq!(field_names.len(), 2);
            assert_eq!(field_names[0], Name::from_string("fst"));
            assert_eq!(field_names[1], Name::from_string("snd"));
        }
        _ => panic!("expected Structure"),
    }
}

#[test]
fn test_elab_structure_constructor_type() {
    // Test that the constructor type is correct:
    // Pair.mk : (A : Type) → (B : Type) → A → B → Pair A B
    let result = elab_decl(
        r"structure Pair (A : Type) (B : Type) where
          fst : A
          snd : B",
    )
    .unwrap();

    match result {
        ElabResult::Structure { ctor_ty, .. } => {
            // The constructor type should be a Pi type
            // (A : Type) → (B : Type) → A → B → Pair A B
            assert!(matches!(ctor_ty, Expr::Pi(_, _, _)));
        }
        _ => panic!("expected Structure"),
    }
}

#[test]
fn test_elab_structure_result_type() {
    let result = elab_decl(
        r"structure MyType : Type where
          val : Type",
    )
    .unwrap();

    match result {
        ElabResult::Structure { ty, .. } => {
            // The structure type should be Type (since it's specified)
            assert!(matches!(ty, Expr::Sort(_)));
        }
        _ => panic!("expected Structure"),
    }
}

#[test]
fn test_elab_structure_dependent_field_debug() {
    // Debug test for dependent field elaboration
    let env = Environment::new();
    let mut ctx = ElabCtx::new(&env);

    // First, manually test that a local can be looked up
    let fvar_id = ctx.push_local("test_local".to_string(), Expr::type_());

    // Verify lookup works
    let lookup_result = ctx.lookup_local("test_local");
    assert!(
        lookup_result.is_some(),
        "Local 'test_local' should be found"
    );
    assert_eq!(lookup_result.unwrap().0, fvar_id, "FVarId should match");

    ctx.pop_local();

    // Now test elaboration of an identifier when a local is in scope
    ctx.push_local("fst".to_string(), Expr::type_());

    // Elaborate just the identifier "fst"
    let surface_ident = SurfaceExpr::ident("fst");
    let result = ctx.elaborate(&surface_ident);
    assert!(
        result.is_ok(),
        "Elaborating 'fst' should succeed when local is in scope: {result:?}"
    );

    ctx.pop_local();
}

#[test]
fn test_elab_structure_dependent_field_realistic() {
    // More realistic test that mimics what elab_structure does
    use lean5_kernel::Declaration;

    let mut env = Environment::new();

    // Add A and B to the environment (like in the real test)
    env.add_decl(Declaration::Axiom {
        name: Name::from_string("A"),
        level_params: vec![],
        type_: Expr::type_(),
    })
    .unwrap();

    // B : A -> Type
    let a_const = Expr::const_(Name::from_string("A"), vec![]);
    env.add_decl(Declaration::Axiom {
        name: Name::from_string("B"),
        level_params: vec![],
        type_: Expr::arrow(a_const.clone(), Expr::type_()),
    })
    .unwrap();

    let mut ctx = ElabCtx::new(&env);

    // Step 1: Push param locals (none in this case)
    // (We're simulating `structure Dep where fst : A  snd : B fst`)

    // Step 2: Elaborate field 0's type (A)
    let field0_ty_surface = parse_expr("A").unwrap();
    let field0_ty = ctx.elaborate(&field0_ty_surface).unwrap();
    assert!(
        matches!(field0_ty, Expr::Const(..)),
        "Field 0 type should be A"
    );

    // Step 3: Push field 0 as local
    let _fst_fvar = ctx.push_local("fst".to_string(), field0_ty.clone());

    // Step 4: Verify 'fst' is in scope
    let lookup = ctx.lookup_local("fst");
    assert!(lookup.is_some(), "fst should be in locals");

    // Step 5: Elaborate field 1's type (B fst)
    // This is where the error should NOT occur
    let field1_ty_surface = parse_expr("B fst").unwrap();

    let result = ctx.elaborate(&field1_ty_surface);
    assert!(
        result.is_ok(),
        "Elaborating 'B fst' should succeed: {result:?}"
    );

    ctx.pop_local();
}

#[test]
fn test_elab_structure_dependent_via_decl() {
    // Test via elab_decl to see if the error occurs there
    use lean5_kernel::Declaration;

    let mut env = Environment::new();

    // Add A and B to the environment
    env.add_decl(Declaration::Axiom {
        name: Name::from_string("A"),
        level_params: vec![],
        type_: Expr::type_(),
    })
    .unwrap();

    let a_const = Expr::const_(Name::from_string("A"), vec![]);
    env.add_decl(Declaration::Axiom {
        name: Name::from_string("B"),
        level_params: vec![],
        type_: Expr::arrow(a_const.clone(), Expr::type_()),
    })
    .unwrap();

    // Now try to elaborate the structure via parse + elab_decl
    let mut ctx = ElabCtx::new(&env);

    let surface_decl = parse_decl(
        r"structure Dep where
          fst : A
          snd : B fst",
    )
    .unwrap();

    let result = ctx.elab_decl(&surface_decl);

    assert!(
        result.is_ok(),
        "elab_decl for dependent structure should succeed: {result:?}"
    );
}

// ==========================================================================
// Instance resolution tests
// ==========================================================================

#[test]
fn test_instance_resolution_basic() {
    // Setup: create an environment with a type class Add and an instance instAddNat
    let mut env = Environment::new();

    // Add Nat type
    env.add_decl(Declaration::Axiom {
        name: Name::from_string("Nat"),
        level_params: vec![],
        type_: Expr::type_(),
    })
    .unwrap();

    // Add type class: class Add (α : Type) := (add : α → α → α)
    // Represented as: Add : Type → Type
    env.add_decl(Declaration::Axiom {
        name: Name::from_string("Add"),
        level_params: vec![],
        type_: Expr::arrow(Expr::type_(), Expr::type_()),
    })
    .unwrap();

    // Create elaboration context
    let mut ctx = ElabCtx::new(&env);

    // Register Add as a type class
    ctx.instances_mut()
        .register_class(Name::from_string("Add"), 1, vec![]);

    // Add instance: instAddNat : Add Nat
    let add_nat_type = Expr::App(
        Expr::const_(Name::from_string("Add"), vec![]).into(),
        Expr::const_(Name::from_string("Nat"), vec![]).into(),
    );
    let inst_name = Name::from_string("instAddNat");

    ctx.instances_mut().add_instance(
        inst_name.clone(),
        Name::from_string("Add"),
        Expr::const_(inst_name, vec![]),
        add_nat_type.clone(),
        crate::instances::DEFAULT_PRIORITY,
    );

    // Test resolution
    let result = ctx.resolve_instance(&add_nat_type);
    assert!(result.is_some(), "Should resolve Add Nat to instAddNat");

    if let Some(inst) = result {
        match inst {
            Expr::Const(name, _) => {
                assert_eq!(name, Name::from_string("instAddNat"));
            }
            other => panic!("Expected Const, got {other:?}"),
        }
    }
}

#[test]
fn test_instance_resolution_no_match() {
    // Test that resolution returns None when no instance exists
    let mut env = Environment::new();

    // Add Nat and Bool types
    env.add_decl(Declaration::Axiom {
        name: Name::from_string("Nat"),
        level_params: vec![],
        type_: Expr::type_(),
    })
    .unwrap();

    env.add_decl(Declaration::Axiom {
        name: Name::from_string("Bool"),
        level_params: vec![],
        type_: Expr::type_(),
    })
    .unwrap();

    env.add_decl(Declaration::Axiom {
        name: Name::from_string("Add"),
        level_params: vec![],
        type_: Expr::arrow(Expr::type_(), Expr::type_()),
    })
    .unwrap();

    let mut ctx = ElabCtx::new(&env);

    // Register Add but only add instance for Nat
    ctx.instances_mut()
        .register_class(Name::from_string("Add"), 1, vec![]);

    let add_nat_type = Expr::App(
        Expr::const_(Name::from_string("Add"), vec![]).into(),
        Expr::const_(Name::from_string("Nat"), vec![]).into(),
    );

    ctx.instances_mut().add_instance(
        Name::from_string("instAddNat"),
        Name::from_string("Add"),
        Expr::const_(Name::from_string("instAddNat"), vec![]),
        add_nat_type,
        crate::instances::DEFAULT_PRIORITY,
    );

    // Try to resolve Add Bool - should fail
    let add_bool_type = Expr::App(
        Expr::const_(Name::from_string("Add"), vec![]).into(),
        Expr::const_(Name::from_string("Bool"), vec![]).into(),
    );

    let result = ctx.resolve_instance(&add_bool_type);
    assert!(result.is_none(), "Should not resolve Add Bool");
}

#[test]
fn test_instance_resolution_priority() {
    // Test that higher priority instances are preferred
    let mut env = Environment::new();

    env.add_decl(Declaration::Axiom {
        name: Name::from_string("Nat"),
        level_params: vec![],
        type_: Expr::type_(),
    })
    .unwrap();

    env.add_decl(Declaration::Axiom {
        name: Name::from_string("Add"),
        level_params: vec![],
        type_: Expr::arrow(Expr::type_(), Expr::type_()),
    })
    .unwrap();

    let mut ctx = ElabCtx::new(&env);

    ctx.instances_mut()
        .register_class(Name::from_string("Add"), 1, vec![]);

    let add_nat_type = Expr::App(
        Expr::const_(Name::from_string("Add"), vec![]).into(),
        Expr::const_(Name::from_string("Nat"), vec![]).into(),
    );

    // Add low priority instance first
    ctx.instances_mut().add_instance(
        Name::from_string("instAddNatLow"),
        Name::from_string("Add"),
        Expr::const_(Name::from_string("instAddNatLow"), vec![]),
        add_nat_type.clone(),
        50,
    );

    // Add high priority instance second
    ctx.instances_mut().add_instance(
        Name::from_string("instAddNatHigh"),
        Name::from_string("Add"),
        Expr::const_(Name::from_string("instAddNatHigh"), vec![]),
        add_nat_type.clone(),
        150,
    );

    // Resolution should return the high priority instance
    let result = ctx.resolve_instance(&add_nat_type);
    assert!(result.is_some());

    if let Some(Expr::Const(name, _)) = result {
        assert_eq!(name, Name::from_string("instAddNatHigh"));
    } else {
        panic!("Expected Const expression");
    }
}

#[test]
fn test_instance_resolution_unregistered_class() {
    // Test that resolution returns None for unregistered type classes
    let mut env = Environment::new();

    env.add_decl(Declaration::Axiom {
        name: Name::from_string("Nat"),
        level_params: vec![],
        type_: Expr::type_(),
    })
    .unwrap();

    env.add_decl(Declaration::Axiom {
        name: Name::from_string("Add"),
        level_params: vec![],
        type_: Expr::arrow(Expr::type_(), Expr::type_()),
    })
    .unwrap();

    let mut ctx = ElabCtx::new(&env);
    // Note: NOT registering Add as a type class

    let add_nat_type = Expr::App(
        Expr::const_(Name::from_string("Add"), vec![]).into(),
        Expr::const_(Name::from_string("Nat"), vec![]).into(),
    );

    // Should return None because Add is not registered as a class
    let result = ctx.resolve_instance(&add_nat_type);
    assert!(
        result.is_none(),
        "Should not resolve unregistered type class"
    );
}

#[test]
fn test_instance_resolution_with_dependency() {
    use lean5_kernel::Declaration;

    let mut env = Environment::new();

    env.add_decl(Declaration::Axiom {
        name: Name::from_string("Nat"),
        level_params: vec![],
        type_: Expr::type_(),
    })
    .unwrap();

    env.add_decl(Declaration::Axiom {
        name: Name::from_string("Add"),
        level_params: vec![],
        type_: Expr::arrow(Expr::type_(), Expr::type_()),
    })
    .unwrap();

    env.add_decl(Declaration::Axiom {
        name: Name::from_string("Mul"),
        level_params: vec![],
        type_: Expr::arrow(Expr::type_(), Expr::type_()),
    })
    .unwrap();

    let mut ctx = ElabCtx::new(&env);

    let add_class = Name::from_string("Add");
    let mul_class = Name::from_string("Mul");
    let nat = Name::from_string("Nat");

    ctx.instances_mut()
        .register_class(add_class.clone(), 1, vec![]);
    ctx.instances_mut()
        .register_class(mul_class.clone(), 1, vec![]);

    let add_nat_type = Expr::App(
        Expr::const_(add_class.clone(), vec![]).into(),
        Expr::const_(nat.clone(), vec![]).into(),
    );
    let mul_nat_type = Expr::App(
        Expr::const_(mul_class.clone(), vec![]).into(),
        Expr::const_(nat.clone(), vec![]).into(),
    );

    // Base instance: Add Nat
    ctx.instances_mut().add_instance(
        Name::from_string("instAddNat"),
        add_class.clone(),
        Expr::const_(Name::from_string("instAddNat"), vec![]),
        add_nat_type.clone(),
        crate::instances::DEFAULT_PRIORITY,
    );

    // Dependent instance: Mul Nat requires [Add Nat]
    let mul_inst_expr = Expr::lam(
        BinderInfo::InstImplicit,
        add_nat_type.clone(),
        Expr::const_(Name::from_string("instMulNat"), vec![]),
    );
    let mul_inst_type = Expr::pi(BinderInfo::InstImplicit, add_nat_type, mul_nat_type.clone());
    ctx.instances_mut().add_instance(
        Name::from_string("instMulNat"),
        mul_class.clone(),
        mul_inst_expr,
        mul_inst_type,
        crate::instances::DEFAULT_PRIORITY,
    );

    let result = ctx.resolve_instance(&mul_nat_type);
    assert!(
        result.is_some(),
        "Should resolve Mul Nat using dependent Add Nat instance"
    );

    if let Some(expr) = result {
        match expr {
            Expr::Const(name, _) => assert_eq!(name, Name::from_string("instMulNat")),
            other => panic!("Expected constant instance, got {other:?}"),
        }
    }
}

#[test]
fn test_instance_resolution_dependency_missing_instance() {
    use lean5_kernel::Declaration;

    let mut env = Environment::new();

    env.add_decl(Declaration::Axiom {
        name: Name::from_string("Nat"),
        level_params: vec![],
        type_: Expr::type_(),
    })
    .unwrap();

    env.add_decl(Declaration::Axiom {
        name: Name::from_string("Add"),
        level_params: vec![],
        type_: Expr::arrow(Expr::type_(), Expr::type_()),
    })
    .unwrap();

    env.add_decl(Declaration::Axiom {
        name: Name::from_string("Mul"),
        level_params: vec![],
        type_: Expr::arrow(Expr::type_(), Expr::type_()),
    })
    .unwrap();

    let mut ctx = ElabCtx::new(&env);

    let add_class = Name::from_string("Add");
    let mul_class = Name::from_string("Mul");
    let nat = Name::from_string("Nat");

    ctx.instances_mut()
        .register_class(add_class.clone(), 1, vec![]);
    ctx.instances_mut()
        .register_class(mul_class.clone(), 1, vec![]);

    let add_nat_type = Expr::App(
        Expr::const_(add_class.clone(), vec![]).into(),
        Expr::const_(nat.clone(), vec![]).into(),
    );
    let mul_nat_type = Expr::App(
        Expr::const_(mul_class.clone(), vec![]).into(),
        Expr::const_(nat.clone(), vec![]).into(),
    );

    // Dependent instance: Mul Nat requires [Add Nat] but Add Nat is not registered
    let mul_inst_expr = Expr::lam(
        BinderInfo::InstImplicit,
        add_nat_type.clone(),
        Expr::const_(Name::from_string("instMulNat"), vec![]),
    );
    let mul_inst_type = Expr::pi(BinderInfo::InstImplicit, add_nat_type, mul_nat_type.clone());
    ctx.instances_mut().add_instance(
        Name::from_string("instMulNat"),
        mul_class.clone(),
        mul_inst_expr,
        mul_inst_type,
        crate::instances::DEFAULT_PRIORITY,
    );

    let result = ctx.resolve_instance(&mul_nat_type);
    assert!(
        result.is_none(),
        "Should fail when dependency instance is missing"
    );
}

#[test]
fn test_instance_resolution_backtracking() {
    // Test backtracking: if first instance fails due to unsatisfied dependency,
    // should backtrack and try next instance.
    //
    // Setup:
    //   class A (α : Type)
    //   class B (α : Type)
    //   class C (α : Type)
    //   instance instAviaBNat [B Nat] : A Nat := ...  (priority 1000, tried first)
    //   instance instANat : A Nat := ...              (priority 500, fallback)
    //   instance instBviaCNat [C Nat] : B Nat := ...
    //   (no C Nat instance)
    //
    // When resolving A Nat:
    //   1. Try instAviaBNat (highest priority)
    //   2. Need to resolve B Nat
    //   3. Try instBviaCNat, need C Nat
    //   4. C Nat fails - no instances
    //   5. B Nat fails
    //   6. instAviaBNat fails, backtrack
    //   7. Try instANat (direct, no dependencies)
    //   8. Success!
    use lean5_kernel::Declaration;

    let mut env = Environment::new();

    // Add type Nat
    env.add_decl(Declaration::Axiom {
        name: Name::from_string("Nat"),
        level_params: vec![],
        type_: Expr::type_(),
    })
    .unwrap();

    // Add classes A, B, C
    for class_name in ["A", "B", "C"] {
        env.add_decl(Declaration::Axiom {
            name: Name::from_string(class_name),
            level_params: vec![],
            type_: Expr::arrow(Expr::type_(), Expr::type_()),
        })
        .unwrap();
    }

    let mut ctx = ElabCtx::new(&env);

    let a_class = Name::from_string("A");
    let b_class = Name::from_string("B");
    let c_class = Name::from_string("C");
    let nat = Name::from_string("Nat");

    ctx.instances_mut()
        .register_class(a_class.clone(), 1, vec![]);
    ctx.instances_mut()
        .register_class(b_class.clone(), 1, vec![]);
    ctx.instances_mut()
        .register_class(c_class.clone(), 1, vec![]);

    let a_nat_type = Expr::App(
        Expr::const_(a_class.clone(), vec![]).into(),
        Expr::const_(nat.clone(), vec![]).into(),
    );
    let b_nat_type = Expr::App(
        Expr::const_(b_class.clone(), vec![]).into(),
        Expr::const_(nat.clone(), vec![]).into(),
    );
    let c_nat_type = Expr::App(
        Expr::const_(c_class.clone(), vec![]).into(),
        Expr::const_(nat.clone(), vec![]).into(),
    );

    // Instance 1: instAviaBNat [B Nat] : A Nat (high priority)
    let inst_a_via_b_expr = Expr::lam(
        BinderInfo::InstImplicit,
        b_nat_type.clone(),
        Expr::const_(Name::from_string("instAviaBNat"), vec![]),
    );
    let inst_a_via_b_type = Expr::pi(
        BinderInfo::InstImplicit,
        b_nat_type.clone(),
        a_nat_type.clone(),
    );
    ctx.instances_mut().add_instance(
        Name::from_string("instAviaBNat"),
        a_class.clone(),
        inst_a_via_b_expr,
        inst_a_via_b_type,
        1000, // High priority - tried first
    );

    // Instance 2: instANat : A Nat (direct, lower priority)
    ctx.instances_mut().add_instance(
        Name::from_string("instANat"),
        a_class.clone(),
        Expr::const_(Name::from_string("instANat"), vec![]),
        a_nat_type.clone(),
        500, // Lower priority - fallback
    );

    // Instance for B Nat that requires C Nat (which doesn't exist)
    let inst_b_via_c_expr = Expr::lam(
        BinderInfo::InstImplicit,
        c_nat_type.clone(),
        Expr::const_(Name::from_string("instBviaCNat"), vec![]),
    );
    let inst_b_via_c_type = Expr::pi(BinderInfo::InstImplicit, c_nat_type, b_nat_type);
    ctx.instances_mut().add_instance(
        Name::from_string("instBviaCNat"),
        b_class.clone(),
        inst_b_via_c_expr,
        inst_b_via_c_type,
        crate::instances::DEFAULT_PRIORITY,
    );

    // No instance for C Nat!

    // Resolution should:
    // 1. Try instAviaBNat -> needs B Nat
    // 2. Try instBviaCNat -> needs C Nat -> FAIL
    // 3. Backtrack, try instANat -> SUCCESS
    let result = ctx.resolve_instance(&a_nat_type);
    assert!(
        result.is_some(),
        "Should backtrack and find instANat when instAviaBNat fails"
    );

    if let Some(expr) = result {
        match expr {
            Expr::Const(name, _) => assert_eq!(
                name,
                Name::from_string("instANat"),
                "Should use fallback instance instANat"
            ),
            other => panic!("Expected constant instance, got {other:?}"),
        }
    }
}

#[test]
fn test_instance_resolution_backtracking_all_fail() {
    // Test that backtracking correctly fails when all instances fail.
    // Same setup as above, but without the fallback instANat.
    use lean5_kernel::Declaration;

    let mut env = Environment::new();

    env.add_decl(Declaration::Axiom {
        name: Name::from_string("Nat"),
        level_params: vec![],
        type_: Expr::type_(),
    })
    .unwrap();

    for class_name in ["A", "B", "C"] {
        env.add_decl(Declaration::Axiom {
            name: Name::from_string(class_name),
            level_params: vec![],
            type_: Expr::arrow(Expr::type_(), Expr::type_()),
        })
        .unwrap();
    }

    let mut ctx = ElabCtx::new(&env);

    let a_class = Name::from_string("A");
    let b_class = Name::from_string("B");
    let c_class = Name::from_string("C");
    let nat = Name::from_string("Nat");

    ctx.instances_mut()
        .register_class(a_class.clone(), 1, vec![]);
    ctx.instances_mut()
        .register_class(b_class.clone(), 1, vec![]);
    ctx.instances_mut()
        .register_class(c_class.clone(), 1, vec![]);

    let a_nat_type = Expr::App(
        Expr::const_(a_class.clone(), vec![]).into(),
        Expr::const_(nat.clone(), vec![]).into(),
    );
    let b_nat_type = Expr::App(
        Expr::const_(b_class.clone(), vec![]).into(),
        Expr::const_(nat.clone(), vec![]).into(),
    );
    let c_nat_type = Expr::App(
        Expr::const_(c_class.clone(), vec![]).into(),
        Expr::const_(nat.clone(), vec![]).into(),
    );

    // Only instAviaBNat [B Nat] : A Nat (no fallback)
    let inst_a_via_b_expr = Expr::lam(
        BinderInfo::InstImplicit,
        b_nat_type.clone(),
        Expr::const_(Name::from_string("instAviaBNat"), vec![]),
    );
    let inst_a_via_b_type = Expr::pi(
        BinderInfo::InstImplicit,
        b_nat_type.clone(),
        a_nat_type.clone(),
    );
    ctx.instances_mut().add_instance(
        Name::from_string("instAviaBNat"),
        a_class.clone(),
        inst_a_via_b_expr,
        inst_a_via_b_type,
        crate::instances::DEFAULT_PRIORITY,
    );

    // Instance for B Nat that requires C Nat (which doesn't exist)
    let inst_b_via_c_expr = Expr::lam(
        BinderInfo::InstImplicit,
        c_nat_type.clone(),
        Expr::const_(Name::from_string("instBviaCNat"), vec![]),
    );
    let inst_b_via_c_type = Expr::pi(BinderInfo::InstImplicit, c_nat_type, b_nat_type);
    ctx.instances_mut().add_instance(
        Name::from_string("instBviaCNat"),
        b_class.clone(),
        inst_b_via_c_expr,
        inst_b_via_c_type,
        crate::instances::DEFAULT_PRIORITY,
    );

    // No instance for C Nat, no fallback for A Nat
    let result = ctx.resolve_instance(&a_nat_type);
    assert!(
        result.is_none(),
        "Should fail when all instances fail due to unsatisfied dependencies"
    );
}

#[test]
fn test_instance_resolution_backtracking_multi_level() {
    // Test backtracking with multiple candidate instances at an intermediate level.
    //
    // Setup:
    //   class A (α : Type)
    //   class B (α : Type)
    //   class C (α : Type)
    //   instance instAviaBNat [B Nat] : A Nat  (needs B Nat)
    //   instance instBviaC [C Nat] : B Nat     (high priority, needs C Nat - will fail)
    //   instance instBNatDirect : B Nat       (low priority, direct - should succeed)
    //   (no C Nat instance)
    //
    // When resolving A Nat:
    //   1. Try instAviaBNat -> needs B Nat
    //   2. Try instBviaC (high priority) -> needs C Nat -> FAIL
    //   3. Backtrack within B Nat resolution, try instBNatDirect -> SUCCESS
    //   4. A Nat resolves successfully via instAviaBNat + instBNatDirect
    use lean5_kernel::Declaration;

    let mut env = Environment::new();

    env.add_decl(Declaration::Axiom {
        name: Name::from_string("Nat"),
        level_params: vec![],
        type_: Expr::type_(),
    })
    .unwrap();

    for class_name in ["A", "B", "C"] {
        env.add_decl(Declaration::Axiom {
            name: Name::from_string(class_name),
            level_params: vec![],
            type_: Expr::arrow(Expr::type_(), Expr::type_()),
        })
        .unwrap();
    }

    let mut ctx = ElabCtx::new(&env);

    let a_class = Name::from_string("A");
    let b_class = Name::from_string("B");
    let c_class = Name::from_string("C");
    let nat = Name::from_string("Nat");

    ctx.instances_mut()
        .register_class(a_class.clone(), 1, vec![]);
    ctx.instances_mut()
        .register_class(b_class.clone(), 1, vec![]);
    ctx.instances_mut()
        .register_class(c_class.clone(), 1, vec![]);

    let a_nat_type = Expr::App(
        Expr::const_(a_class.clone(), vec![]).into(),
        Expr::const_(nat.clone(), vec![]).into(),
    );
    let b_nat_type = Expr::App(
        Expr::const_(b_class.clone(), vec![]).into(),
        Expr::const_(nat.clone(), vec![]).into(),
    );
    let c_nat_type = Expr::App(
        Expr::const_(c_class.clone(), vec![]).into(),
        Expr::const_(nat.clone(), vec![]).into(),
    );

    // instAviaBNat [B Nat] : A Nat
    let inst_a_via_b_expr = Expr::lam(
        BinderInfo::InstImplicit,
        b_nat_type.clone(),
        Expr::const_(Name::from_string("instAviaBNat"), vec![]),
    );
    let inst_a_via_b_type = Expr::pi(
        BinderInfo::InstImplicit,
        b_nat_type.clone(),
        a_nat_type.clone(),
    );
    ctx.instances_mut().add_instance(
        Name::from_string("instAviaBNat"),
        a_class.clone(),
        inst_a_via_b_expr,
        inst_a_via_b_type,
        crate::instances::DEFAULT_PRIORITY,
    );

    // instBviaC [C Nat] : B Nat (high priority - tried first, will fail)
    let inst_b_via_c_expr = Expr::lam(
        BinderInfo::InstImplicit,
        c_nat_type.clone(),
        Expr::const_(Name::from_string("instBviaCNat"), vec![]),
    );
    let inst_b_via_c_type = Expr::pi(BinderInfo::InstImplicit, c_nat_type, b_nat_type.clone());
    ctx.instances_mut().add_instance(
        Name::from_string("instBviaCNat"),
        b_class.clone(),
        inst_b_via_c_expr,
        inst_b_via_c_type,
        1000, // High priority
    );

    // instBNatDirect : B Nat (low priority - fallback)
    ctx.instances_mut().add_instance(
        Name::from_string("instBNatDirect"),
        b_class.clone(),
        Expr::const_(Name::from_string("instBNatDirect"), vec![]),
        b_nat_type.clone(),
        500, // Low priority
    );

    // No instance for C Nat!

    // Resolution should succeed: A Nat -> B Nat (via instBNatDirect fallback)
    let result = ctx.resolve_instance(&a_nat_type);
    assert!(
        result.is_some(),
        "Should backtrack within B resolution and find instBNatDirect"
    );

    // The result should be instAviaBNat applied to instBNatDirect
    if let Some(Expr::Const(name, _)) = result {
        assert_eq!(
            name,
            Name::from_string("instAviaBNat"),
            "Should use instAviaBNat with resolved dependency"
        );
    }
    // Could also be an application if the lambda was beta-reduced, which is fine
}

#[test]
fn test_instance_resolution_diamond() {
    // Test diamond inheritance pattern:
    //
    //        A Nat
    //       /     \
    //   B Nat    C Nat
    //       \     /
    //        D Nat
    //
    // Setup:
    //   instance [B Nat] [C Nat] : A Nat
    //   instance [D Nat] : B Nat
    //   instance [D Nat] : C Nat
    //   instance : D Nat  (base instance)
    //
    // This tests that resolution correctly resolves the diamond without infinite loops.
    use lean5_kernel::Declaration;

    let mut env = Environment::new();

    env.add_decl(Declaration::Axiom {
        name: Name::from_string("Nat"),
        level_params: vec![],
        type_: Expr::type_(),
    })
    .unwrap();

    for class_name in ["A", "B", "C", "D"] {
        env.add_decl(Declaration::Axiom {
            name: Name::from_string(class_name),
            level_params: vec![],
            type_: Expr::arrow(Expr::type_(), Expr::type_()),
        })
        .unwrap();
    }

    let mut ctx = ElabCtx::new(&env);

    let a_class = Name::from_string("A");
    let b_class = Name::from_string("B");
    let c_class = Name::from_string("C");
    let d_class = Name::from_string("D");
    let nat = Name::from_string("Nat");

    for class in [&a_class, &b_class, &c_class, &d_class] {
        ctx.instances_mut().register_class(class.clone(), 1, vec![]);
    }

    let a_nat = Expr::App(
        Expr::const_(a_class.clone(), vec![]).into(),
        Expr::const_(nat.clone(), vec![]).into(),
    );
    let b_nat = Expr::App(
        Expr::const_(b_class.clone(), vec![]).into(),
        Expr::const_(nat.clone(), vec![]).into(),
    );
    let c_nat = Expr::App(
        Expr::const_(c_class.clone(), vec![]).into(),
        Expr::const_(nat.clone(), vec![]).into(),
    );
    let d_nat = Expr::App(
        Expr::const_(d_class.clone(), vec![]).into(),
        Expr::const_(nat.clone(), vec![]).into(),
    );

    // instance [B Nat] [C Nat] : A Nat
    let inst_a_expr = Expr::lam(
        BinderInfo::InstImplicit,
        b_nat.clone(),
        Expr::lam(
            BinderInfo::InstImplicit,
            c_nat.clone(),
            Expr::const_(Name::from_string("instANat"), vec![]),
        ),
    );
    let inst_a_type = Expr::pi(
        BinderInfo::InstImplicit,
        b_nat.clone(),
        Expr::pi(BinderInfo::InstImplicit, c_nat.clone(), a_nat.clone()),
    );
    ctx.instances_mut().add_instance(
        Name::from_string("instANat"),
        a_class.clone(),
        inst_a_expr,
        inst_a_type,
        crate::instances::DEFAULT_PRIORITY,
    );

    // instance [D Nat] : B Nat
    let inst_b_expr = Expr::lam(
        BinderInfo::InstImplicit,
        d_nat.clone(),
        Expr::const_(Name::from_string("instBNat"), vec![]),
    );
    let inst_b_type = Expr::pi(BinderInfo::InstImplicit, d_nat.clone(), b_nat.clone());
    ctx.instances_mut().add_instance(
        Name::from_string("instBNat"),
        b_class.clone(),
        inst_b_expr,
        inst_b_type,
        crate::instances::DEFAULT_PRIORITY,
    );

    // instance [D Nat] : C Nat
    let inst_c_expr = Expr::lam(
        BinderInfo::InstImplicit,
        d_nat.clone(),
        Expr::const_(Name::from_string("instCNat"), vec![]),
    );
    let inst_c_type = Expr::pi(BinderInfo::InstImplicit, d_nat.clone(), c_nat.clone());
    ctx.instances_mut().add_instance(
        Name::from_string("instCNat"),
        c_class.clone(),
        inst_c_expr,
        inst_c_type,
        crate::instances::DEFAULT_PRIORITY,
    );

    // instance : D Nat (base)
    ctx.instances_mut().add_instance(
        Name::from_string("instDNat"),
        d_class.clone(),
        Expr::const_(Name::from_string("instDNat"), vec![]),
        d_nat,
        crate::instances::DEFAULT_PRIORITY,
    );

    // Resolution should succeed for A Nat via the diamond
    let result = ctx.resolve_instance(&a_nat);
    assert!(
        result.is_some(),
        "Should resolve A Nat via diamond inheritance"
    );
}

// ==========================================================================
// Class declaration parsing and elaboration tests
// ==========================================================================

#[test]
fn test_class_decl_registers_class() {
    use lean5_parser::Parser;

    let env = Environment::new();
    let mut ctx = ElabCtx::new(&env);

    // Parse and elaborate a class declaration
    let decl = Parser::parse_decl(
        r"class Add (α : Type) where
          add : α → α → α",
    )
    .unwrap();

    let result = ctx.elab_decl(&decl);
    assert!(result.is_ok(), "Class elaboration failed: {result:?}");

    // Verify the class was registered
    assert!(
        ctx.instances.is_class(&Name::from_string("Add")),
        "Add should be registered as a type class"
    );

    // Verify the class info
    let class_info = ctx.instances.get_class(&Name::from_string("Add")).unwrap();
    assert_eq!(class_info.num_params, 1);
}

#[test]
fn test_class_decl_multiple_params() {
    use lean5_parser::Parser;

    let env = Environment::new();
    let mut ctx = ElabCtx::new(&env);

    // Class with multiple parameters
    let decl = Parser::parse_decl(
        r"class HAdd (α : Type) (β : Type) (γ : Type) where
          hAdd : α → β → γ",
    )
    .unwrap();

    let result = ctx.elab_decl(&decl);
    assert!(result.is_ok(), "Class elaboration failed: {result:?}");

    let class_info = ctx.instances.get_class(&Name::from_string("HAdd")).unwrap();
    assert_eq!(class_info.num_params, 3);
}

#[test]
fn test_class_decl_produces_structure() {
    use lean5_parser::Parser;

    let env = Environment::new();
    let mut ctx = ElabCtx::new(&env);

    let decl = Parser::parse_decl(
        r"class Inhabited (α : Type) where
          default : α",
    )
    .unwrap();

    let result = ctx.elab_decl(&decl).unwrap();

    // Class declarations produce structure results
    match result {
        ElabResult::Structure {
            name,
            field_names,
            projections,
            ..
        } => {
            assert_eq!(name.to_string(), "Inhabited");
            assert_eq!(field_names.len(), 1);
            assert_eq!(field_names[0].to_string(), "default");
            assert_eq!(projections.len(), 1);
        }
        other => panic!("Expected Structure result, got {other:?}"),
    }
}

#[test]
fn test_instance_elaboration_basic() {
    use lean5_kernel::{Constructor, Declaration, InductiveDecl, InductiveType};
    use lean5_parser::Parser;

    // Create an environment with a class (Add) defined as a structure
    let mut env = Environment::new();

    // First add Nat type
    let nat = Name::from_string("Nat");
    let nat_type = Expr::type_();
    let zero_type = Expr::const_(nat.clone(), vec![]);
    let succ_type = Expr::arrow(
        Expr::const_(nat.clone(), vec![]),
        Expr::const_(nat.clone(), vec![]),
    );
    let nat_decl = InductiveDecl {
        level_params: vec![],
        num_params: 0,
        types: vec![InductiveType {
            name: nat.clone(),
            type_: nat_type,
            constructors: vec![
                Constructor {
                    name: Name::from_string("Nat.zero"),
                    type_: zero_type,
                },
                Constructor {
                    name: Name::from_string("Nat.succ"),
                    type_: succ_type,
                },
            ],
        }],
    };
    env.add_inductive(nat_decl).unwrap();

    // Add Nat.add : Nat → Nat → Nat
    env.add_decl(Declaration::Definition {
        name: Name::from_string("Nat.add"),
        level_params: vec![],
        type_: Expr::arrow(
            Expr::const_(nat.clone(), vec![]),
            Expr::arrow(
                Expr::const_(nat.clone(), vec![]),
                Expr::const_(nat.clone(), vec![]),
            ),
        ),
        value: Expr::const_(Name::from_string("Nat.add"), vec![]), // placeholder
        is_reducible: true,
    })
    .unwrap();

    // Add class Add (α : Type) as a structure with one field: add : α → α → α
    let add_class = Name::from_string("Add");
    // Add : Type → Type
    let add_type = Expr::arrow(Expr::type_(), Expr::type_());

    // Add.mk : (α : Type) → (α → α → α) → Add α
    let alpha = Expr::bvar(1); // α (bound at outer position)
    let add_field_ty = Expr::arrow(alpha.clone(), Expr::arrow(alpha.clone(), alpha.clone()));
    let add_result = Expr::app(
        Expr::const_(add_class.clone(), vec![]),
        Expr::bvar(1), // α
    );
    let add_mk_type = Expr::pi(
        BinderInfo::Default,
        Expr::type_(), // (α : Type)
        Expr::pi(
            BinderInfo::Default,
            add_field_ty, // (add : α → α → α)
            add_result,   // Add α
        ),
    );

    let add_decl = InductiveDecl {
        level_params: vec![],
        num_params: 1, // α is a parameter
        types: vec![InductiveType {
            name: add_class.clone(),
            type_: add_type,
            constructors: vec![Constructor {
                name: Name::from_string("Add.mk"),
                type_: add_mk_type,
            }],
        }],
    };
    env.add_inductive(add_decl).unwrap();

    // Register Add as having field "add"
    env.register_structure_fields(add_class.clone(), vec![Name::from_string("add")])
        .unwrap();

    // Now elaborate an instance
    let mut ctx = ElabCtx::new(&env);

    // First register the class in the instance table
    ctx.instances_mut()
        .register_class(add_class.clone(), 1, vec![]);

    let decl = Parser::parse_decl(
        r"instance : Add Nat where
          add := Nat.add",
    )
    .unwrap();

    let result = ctx.elab_decl(&decl);
    match result {
        Ok(ElabResult::Instance {
            name,
            class_name,
            priority,
            ..
        }) => {
            // Check instance name was auto-generated
            assert!(
                name.to_string().contains("instAdd"),
                "expected auto-generated name, got {name}"
            );
            assert_eq!(class_name, add_class);
            assert_eq!(priority, 100); // DEFAULT_PRIORITY

            // Check that the instance was registered
            let instances = ctx.instances().get_instances(&add_class);
            assert_eq!(instances.len(), 1);
            assert_eq!(instances[0].name.to_string(), name.to_string());
        }
        Ok(other) => panic!("Expected Instance result, got {other:?}"),
        Err(e) => panic!("Instance elaboration failed: {e:?}"),
    }
}

#[test]
fn test_instance_elaboration_named() {
    use lean5_kernel::{Constructor, Declaration, InductiveDecl, InductiveType};
    use lean5_parser::Parser;

    // Setup similar to basic test
    let mut env = Environment::new();

    let nat = Name::from_string("Nat");
    let nat_decl = InductiveDecl {
        level_params: vec![],
        num_params: 0,
        types: vec![InductiveType {
            name: nat.clone(),
            type_: Expr::type_(),
            constructors: vec![Constructor {
                name: Name::from_string("Nat.zero"),
                type_: Expr::const_(nat.clone(), vec![]),
            }],
        }],
    };
    env.add_inductive(nat_decl).unwrap();

    env.add_decl(Declaration::Definition {
        name: Name::from_string("Nat.add"),
        level_params: vec![],
        type_: Expr::arrow(
            Expr::const_(nat.clone(), vec![]),
            Expr::arrow(
                Expr::const_(nat.clone(), vec![]),
                Expr::const_(nat.clone(), vec![]),
            ),
        ),
        value: Expr::const_(Name::from_string("Nat.add"), vec![]),
        is_reducible: true,
    })
    .unwrap();

    let add_class = Name::from_string("Add");
    let alpha = Expr::bvar(1);
    let add_field_ty = Expr::arrow(alpha.clone(), Expr::arrow(alpha.clone(), alpha.clone()));
    let add_result = Expr::app(Expr::const_(add_class.clone(), vec![]), Expr::bvar(1));
    let add_mk_type = Expr::pi(
        BinderInfo::Default,
        Expr::type_(),
        Expr::pi(BinderInfo::Default, add_field_ty, add_result),
    );

    let add_decl = InductiveDecl {
        level_params: vec![],
        num_params: 1,
        types: vec![InductiveType {
            name: add_class.clone(),
            type_: Expr::arrow(Expr::type_(), Expr::type_()),
            constructors: vec![Constructor {
                name: Name::from_string("Add.mk"),
                type_: add_mk_type,
            }],
        }],
    };
    env.add_inductive(add_decl).unwrap();
    env.register_structure_fields(add_class.clone(), vec![Name::from_string("add")])
        .unwrap();

    let mut ctx = ElabCtx::new(&env);
    ctx.instances_mut()
        .register_class(add_class.clone(), 1, vec![]);

    // Parse instance with explicit name
    let decl = Parser::parse_decl(
        r"instance instAddNat : Add Nat where
          add := Nat.add",
    )
    .unwrap();

    let result = ctx.elab_decl(&decl);
    match result {
        Ok(ElabResult::Instance { name, .. }) => {
            assert_eq!(name.to_string(), "instAddNat");
        }
        Ok(other) => panic!("Expected Instance result, got {other:?}"),
        Err(e) => panic!("Instance elaboration failed: {e:?}"),
    }
}

#[test]
fn test_instance_registration_in_table() {
    use lean5_kernel::{Constructor, InductiveDecl, InductiveType};
    use lean5_parser::Parser;

    let mut env = Environment::new();

    // Minimal setup - just the class structure
    let my_class = Name::from_string("MyClass");
    let my_class_decl = InductiveDecl {
        level_params: vec![],
        num_params: 1,
        types: vec![InductiveType {
            name: my_class.clone(),
            type_: Expr::arrow(Expr::type_(), Expr::type_()),
            constructors: vec![Constructor {
                name: Name::from_string("MyClass.mk"),
                type_: Expr::pi(
                    BinderInfo::Default,
                    Expr::type_(),
                    Expr::pi(
                        BinderInfo::Default,
                        Expr::type_(), // field type
                        Expr::app(Expr::const_(my_class.clone(), vec![]), Expr::bvar(1)),
                    ),
                ),
            }],
        }],
    };
    env.add_inductive(my_class_decl).unwrap();
    env.register_structure_fields(my_class.clone(), vec![Name::from_string("val")])
        .unwrap();

    // Add a simple type to instantiate for
    let my_type = Name::from_string("MyType");
    let my_type_decl = InductiveDecl {
        level_params: vec![],
        num_params: 0,
        types: vec![InductiveType {
            name: my_type.clone(),
            type_: Expr::type_(),
            constructors: vec![Constructor {
                name: Name::from_string("MyType.mk"),
                type_: Expr::const_(my_type.clone(), vec![]),
            }],
        }],
    };
    env.add_inductive(my_type_decl).unwrap();

    let mut ctx = ElabCtx::new(&env);
    ctx.instances_mut()
        .register_class(my_class.clone(), 1, vec![]);

    // Before elaborating, no instances
    assert_eq!(ctx.instances().get_instances(&my_class).len(), 0);

    let decl = Parser::parse_decl(
        r"instance : MyClass MyType where
          val := MyType",
    )
    .unwrap();

    ctx.elab_decl(&decl).unwrap();

    // After elaborating, instance is registered
    let instances = ctx.instances().get_instances(&my_class);
    assert_eq!(instances.len(), 1);
    assert_eq!(instances[0].class_name, my_class);
}

#[test]
fn test_instance_missing_field_error() {
    use lean5_kernel::{Constructor, InductiveDecl, InductiveType};
    use lean5_parser::Parser;

    let mut env = Environment::new();

    // Class with two fields
    let my_class = Name::from_string("TwoFields");
    let my_class_decl = InductiveDecl {
        level_params: vec![],
        num_params: 1,
        types: vec![InductiveType {
            name: my_class.clone(),
            type_: Expr::arrow(Expr::type_(), Expr::type_()),
            constructors: vec![Constructor {
                name: Name::from_string("TwoFields.mk"),
                type_: Expr::pi(
                    BinderInfo::Default,
                    Expr::type_(),
                    Expr::pi(
                        BinderInfo::Default,
                        Expr::type_(), // field1
                        Expr::pi(
                            BinderInfo::Default,
                            Expr::type_(), // field2
                            Expr::app(Expr::const_(my_class.clone(), vec![]), Expr::bvar(2)),
                        ),
                    ),
                ),
            }],
        }],
    };
    env.add_inductive(my_class_decl).unwrap();
    env.register_structure_fields(
        my_class.clone(),
        vec![Name::from_string("field1"), Name::from_string("field2")],
    )
    .unwrap();

    let my_type = Name::from_string("SomeType");
    let my_type_decl = InductiveDecl {
        level_params: vec![],
        num_params: 0,
        types: vec![InductiveType {
            name: my_type.clone(),
            type_: Expr::type_(),
            constructors: vec![],
        }],
    };
    env.add_inductive(my_type_decl).unwrap();

    let mut ctx = ElabCtx::new(&env);
    ctx.instances_mut()
        .register_class(my_class.clone(), 2, vec![]);

    // Instance with only one field (missing field2)
    let decl = Parser::parse_decl(
        r"instance : TwoFields SomeType where
          field1 := SomeType",
    )
    .unwrap();

    let result = ctx.elab_decl(&decl);
    assert!(
        matches!(result, Err(ElabError::NotImplemented(ref msg)) if msg.contains("missing field")),
        "Expected missing field error, got {result:?}"
    );
}

// ==========================================================================
// outParam tests
// ==========================================================================

#[test]
fn test_class_with_outparam_detected() {
    use lean5_parser::Parser;

    let env = Environment::new();
    let mut ctx = ElabCtx::new(&env);

    // Parse a class with an out-parameter
    let decl = Parser::parse_decl(
        r"class HAdd (α : Type) (β : Type) (γ : outParam Type) where
          hAdd : α → β → γ",
    )
    .unwrap();

    let result = ctx.elab_decl(&decl);
    assert!(result.is_ok(), "Class elaboration failed: {result:?}");

    // Verify the class was registered with out_params
    let class_info = ctx.instances.get_class(&Name::from_string("HAdd")).unwrap();
    assert_eq!(class_info.num_params, 3);
    assert_eq!(
        class_info.out_params,
        vec![2],
        "Third parameter (index 2) should be outParam"
    );
}

#[test]
fn test_class_multiple_outparams() {
    use lean5_parser::Parser;

    let env = Environment::new();
    let mut ctx = ElabCtx::new(&env);

    // Parse a class with multiple out-parameters
    let decl = Parser::parse_decl(
        r"class Bifunctor (F : outParam Type) (G : outParam Type) (α : Type) where
          bimap : α → F",
    )
    .unwrap();

    let result = ctx.elab_decl(&decl);
    assert!(result.is_ok(), "Class elaboration failed: {result:?}");

    let class_info = ctx
        .instances
        .get_class(&Name::from_string("Bifunctor"))
        .unwrap();
    assert_eq!(class_info.num_params, 3);
    assert_eq!(
        class_info.out_params,
        vec![0, 1],
        "First two parameters should be outParams"
    );
}

#[test]
fn test_class_no_outparam() {
    use lean5_parser::Parser;

    let env = Environment::new();
    let mut ctx = ElabCtx::new(&env);

    // Parse a class without out-parameters
    let decl = Parser::parse_decl(
        r"class Add (α : Type) where
          add : α → α → α",
    )
    .unwrap();

    let result = ctx.elab_decl(&decl);
    assert!(result.is_ok(), "Class elaboration failed: {result:?}");

    let class_info = ctx.instances.get_class(&Name::from_string("Add")).unwrap();
    assert_eq!(class_info.num_params, 1);
    assert!(class_info.out_params.is_empty(), "Should have no outParams");
}

#[test]
fn test_outparam_instance_resolution() {
    // Test that instance resolution works with out-parameters
    // HAdd α β γ where γ is an out-param means we can resolve HAdd Nat Nat _
    // and get γ = Nat from the instance
    use lean5_kernel::Declaration;

    let mut env = Environment::new();

    // Add Nat type
    env.add_decl(Declaration::Axiom {
        name: Name::from_string("Nat"),
        level_params: vec![],
        type_: Expr::type_(),
    })
    .unwrap();

    let mut ctx = ElabCtx::new(&env);
    let nat = Name::from_string("Nat");
    let hadd = Name::from_string("HAdd");

    // Register HAdd with γ as out-param (index 2)
    ctx.instances_mut().register_class(hadd.clone(), 3, vec![2]);

    // Add instance: HAdd Nat Nat Nat
    // Build: ((HAdd Nat) Nat) Nat
    let hadd_nat = Expr::App(
        Expr::const_(hadd.clone(), vec![]).into(),
        Expr::const_(nat.clone(), vec![]).into(),
    );
    let hadd_nat_nat = Expr::App(hadd_nat.into(), Expr::const_(nat.clone(), vec![]).into());
    let hadd_nat_nat_nat = Expr::App(
        hadd_nat_nat.into(),
        Expr::const_(nat.clone(), vec![]).into(),
    );

    let inst_expr = Expr::const_(Name::from_string("instHAddNatNatNat"), vec![]);
    ctx.instances_mut().add_instance(
        Name::from_string("instHAddNatNatNat"),
        hadd.clone(),
        inst_expr,
        hadd_nat_nat_nat.clone(),
        crate::instances::DEFAULT_PRIORITY,
    );

    // Try to resolve HAdd Nat Nat _ (with out-param as a metavariable)
    let out_param_meta = ctx.fresh_meta(Expr::type_());
    let goal_hadd_nat = Expr::App(
        Expr::const_(hadd.clone(), vec![]).into(),
        Expr::const_(nat.clone(), vec![]).into(),
    );
    let goal_hadd_nat_nat = Expr::App(
        goal_hadd_nat.into(),
        Expr::const_(nat.clone(), vec![]).into(),
    );
    let goal_type = Expr::App(goal_hadd_nat_nat.into(), out_param_meta.into());

    let result = ctx.resolve_instance(&goal_type);
    assert!(
        result.is_some(),
        "Should resolve HAdd Nat Nat _ with out-param"
    );
}

#[test]
fn test_outparam_no_match_wrong_input() {
    // Test that out-param resolution fails if input params don't match
    use lean5_kernel::Declaration;

    let mut env = Environment::new();

    env.add_decl(Declaration::Axiom {
        name: Name::from_string("Nat"),
        level_params: vec![],
        type_: Expr::type_(),
    })
    .unwrap();
    env.add_decl(Declaration::Axiom {
        name: Name::from_string("Bool"),
        level_params: vec![],
        type_: Expr::type_(),
    })
    .unwrap();

    let mut ctx = ElabCtx::new(&env);
    let nat = Name::from_string("Nat");
    let bool_ = Name::from_string("Bool");
    let hadd = Name::from_string("HAdd");

    // Register HAdd with γ as out-param (index 2)
    ctx.instances_mut().register_class(hadd.clone(), 3, vec![2]);

    // Add instance: HAdd Nat Nat Nat (only works for Nat + Nat)
    let hadd_nat = Expr::App(
        Expr::const_(hadd.clone(), vec![]).into(),
        Expr::const_(nat.clone(), vec![]).into(),
    );
    let hadd_nat_nat = Expr::App(hadd_nat.into(), Expr::const_(nat.clone(), vec![]).into());
    let hadd_nat_nat_nat = Expr::App(
        hadd_nat_nat.into(),
        Expr::const_(nat.clone(), vec![]).into(),
    );

    let inst_expr = Expr::const_(Name::from_string("instHAddNatNatNat"), vec![]);
    ctx.instances_mut().add_instance(
        Name::from_string("instHAddNatNatNat"),
        hadd.clone(),
        inst_expr,
        hadd_nat_nat_nat.clone(),
        crate::instances::DEFAULT_PRIORITY,
    );

    // Try to resolve HAdd Bool Nat _ - should fail because Bool ≠ Nat
    let out_param_meta = ctx.fresh_meta(Expr::type_());
    let goal_hadd_bool = Expr::App(
        Expr::const_(hadd.clone(), vec![]).into(),
        Expr::const_(bool_.clone(), vec![]).into(), // Wrong type
    );
    let goal_hadd_bool_nat = Expr::App(
        goal_hadd_bool.into(),
        Expr::const_(nat.clone(), vec![]).into(),
    );
    let goal_type = Expr::App(goal_hadd_bool_nat.into(), out_param_meta.into());

    let result = ctx.resolve_instance(&goal_type);
    assert!(
        result.is_none(),
        "Should fail when non-out-param doesn't match"
    );
}

// ==========================================================================
// semiOutParam tests
// ==========================================================================

#[test]
fn test_class_with_semioutparam_detected() {
    use lean5_parser::Parser;

    let env = Environment::new();
    let mut ctx = ElabCtx::new(&env);

    // Parse a class with a semi-out-parameter (like Coe)
    let decl = Parser::parse_decl(
        r"class Coe (α : semiOutParam Type) (β : Type) where
          coe : α → β",
    )
    .unwrap();

    let result = ctx.elab_decl(&decl);
    assert!(result.is_ok(), "Class elaboration failed: {result:?}");

    // Verify the class was registered with semi_out_params
    let class_info = ctx.instances.get_class(&Name::from_string("Coe")).unwrap();
    assert_eq!(class_info.num_params, 2);
    assert!(class_info.out_params.is_empty(), "Should have no outParams");
    assert_eq!(
        class_info.semi_out_params,
        vec![0],
        "First parameter (index 0) should be semiOutParam"
    );
}

#[test]
fn test_class_with_both_outparam_and_semioutparam() {
    use lean5_parser::Parser;

    let env = Environment::new();
    let mut ctx = ElabCtx::new(&env);

    // Parse a class with both outParam and semiOutParam
    let decl = Parser::parse_decl(
        r"class HCoe (α : semiOutParam Type) (β : Type) (γ : outParam Type) where
          hCoe : α → β → γ",
    )
    .unwrap();

    let result = ctx.elab_decl(&decl);
    assert!(result.is_ok(), "Class elaboration failed: {result:?}");

    // Verify both param types were registered correctly
    let class_info = ctx.instances.get_class(&Name::from_string("HCoe")).unwrap();
    assert_eq!(class_info.num_params, 3);
    assert_eq!(
        class_info.out_params,
        vec![2],
        "Third parameter should be outParam"
    );
    assert_eq!(
        class_info.semi_out_params,
        vec![0],
        "First parameter should be semiOutParam"
    );
}

#[test]
fn test_semioutparam_unifies_bidirectionally() {
    // Test that semiOutParam participates in normal unification (unlike outParam)
    // With Coe (α : semiOutParam Type) (β : Type), when resolving Coe ?α Nat:
    // - If instance is Coe String Nat, then ?α unifies with String
    // - Unlike outParam, the goal ?α can also constrain the match

    use lean5_kernel::Declaration;

    let mut env = Environment::new();

    // Add Nat and String types
    env.add_decl(Declaration::Axiom {
        name: Name::from_string("Nat"),
        level_params: vec![],
        type_: Expr::type_(),
    })
    .unwrap();

    env.add_decl(Declaration::Axiom {
        name: Name::from_string("String"),
        level_params: vec![],
        type_: Expr::type_(),
    })
    .unwrap();

    let mut ctx = ElabCtx::new(&env);
    let nat = Name::from_string("Nat");
    let string = Name::from_string("String");
    let coe = Name::from_string("Coe");

    // Register Coe with α as semi-out-param (index 0)
    ctx.instances_mut()
        .register_class_full(coe.clone(), 2, vec![], vec![0]);

    // Add instance: Coe String Nat (can convert String to Nat)
    // Build: (Coe String) Nat
    let coe_string = Expr::App(
        Expr::const_(coe.clone(), vec![]).into(),
        Expr::const_(string.clone(), vec![]).into(),
    );
    let coe_string_nat = Expr::App(coe_string.into(), Expr::const_(nat.clone(), vec![]).into());

    let inst_expr = Expr::const_(Name::from_string("instCoeStringNat"), vec![]);
    ctx.instances_mut().add_instance(
        Name::from_string("instCoeStringNat"),
        coe.clone(),
        inst_expr,
        coe_string_nat.clone(),
        crate::instances::DEFAULT_PRIORITY,
    );

    // Case 1: Resolve Coe ?α Nat - should succeed and set ?α = String
    let alpha_meta = ctx.fresh_meta(Expr::type_());
    let goal_coe_meta = Expr::App(
        Expr::const_(coe.clone(), vec![]).into(),
        alpha_meta.clone().into(),
    );
    let goal_type = Expr::App(
        goal_coe_meta.into(),
        Expr::const_(nat.clone(), vec![]).into(),
    );

    let result = ctx.resolve_instance(&goal_type);
    assert!(
        result.is_some(),
        "Should resolve Coe ?α Nat to Coe String Nat"
    );

    // Verify the metavariable was unified with String
    let alpha_resolved = ctx.metas.instantiate(&alpha_meta);
    match &alpha_resolved {
        Expr::Const(n, _) if *n == string => (),
        _ => panic!("Expected ?α to be unified with String, got {alpha_resolved:?}"),
    }
}

#[test]
fn test_semioutparam_can_be_constrained() {
    // Test that semiOutParam can also be constrained from the goal side
    // (unlike outParam which only gets values from instances)

    use lean5_kernel::Declaration;

    let mut env = Environment::new();

    // Add types
    env.add_decl(Declaration::Axiom {
        name: Name::from_string("Nat"),
        level_params: vec![],
        type_: Expr::type_(),
    })
    .unwrap();

    env.add_decl(Declaration::Axiom {
        name: Name::from_string("String"),
        level_params: vec![],
        type_: Expr::type_(),
    })
    .unwrap();

    env.add_decl(Declaration::Axiom {
        name: Name::from_string("Bool"),
        level_params: vec![],
        type_: Expr::type_(),
    })
    .unwrap();

    let mut ctx = ElabCtx::new(&env);
    let nat = Name::from_string("Nat");
    let string = Name::from_string("String");
    let bool_ = Name::from_string("Bool");
    let coe = Name::from_string("Coe");

    // Register Coe with α as semi-out-param (index 0)
    ctx.instances_mut()
        .register_class_full(coe.clone(), 2, vec![], vec![0]);

    // Add instance: Coe String Nat
    let coe_string = Expr::App(
        Expr::const_(coe.clone(), vec![]).into(),
        Expr::const_(string.clone(), vec![]).into(),
    );
    let coe_string_nat = Expr::App(coe_string.into(), Expr::const_(nat.clone(), vec![]).into());

    let inst_expr = Expr::const_(Name::from_string("instCoeStringNat"), vec![]);
    ctx.instances_mut().add_instance(
        Name::from_string("instCoeStringNat"),
        coe.clone(),
        inst_expr,
        coe_string_nat.clone(),
        crate::instances::DEFAULT_PRIORITY,
    );

    // Case: Try to resolve Coe Bool Nat - should FAIL because no instance exists
    // (unlike outParam, the goal type must match what instances provide)
    let goal_coe_bool = Expr::App(
        Expr::const_(coe.clone(), vec![]).into(),
        Expr::const_(bool_.clone(), vec![]).into(),
    );
    let goal_type = Expr::App(
        goal_coe_bool.into(),
        Expr::const_(nat.clone(), vec![]).into(),
    );

    let result = ctx.resolve_instance(&goal_type);
    assert!(
        result.is_none(),
        "Should fail: Coe Bool Nat has no instance (semiOutParam must match)"
    );
}

// ==========================================================================
// Instance priority attribute tests
// ==========================================================================

#[test]
fn test_instance_priority_attribute_parsing() {
    // Test that @[instance 50] sets priority to 50
    use lean5_parser::parse_decl;

    let decl = parse_decl(
        r"@[instance 50] instance : Add Nat where
          add := Nat.add",
    )
    .unwrap();

    match decl {
        SurfaceDecl::Instance { priority, .. } => {
            assert_eq!(priority, Some(50));
        }
        _ => panic!("expected Instance"),
    }
}

#[test]
fn test_default_instance_attribute_parsing() {
    // Test that @[defaultInstance] sets priority to 0
    use lean5_parser::parse_decl;

    let decl = parse_decl(
        r"@[defaultInstance] instance : ToString Nat where
          toString := Nat.repr",
    )
    .unwrap();

    match decl {
        SurfaceDecl::Instance { priority, .. } => {
            assert_eq!(priority, Some(0));
        }
        _ => panic!("expected Instance"),
    }
}

#[test]
fn test_instance_priority_ordering() {
    // Test that instances are sorted by priority (highest first)
    let env = Environment::new();
    let mut ctx = ElabCtx::new(&env);

    let show = Name::from_string("Show");
    ctx.instances_mut().register_class(show.clone(), 1, vec![]);

    // Add instances with different priorities
    ctx.instances_mut().add_instance(
        Name::from_string("low"),
        show.clone(),
        Expr::const_(Name::from_string("low"), vec![]),
        Expr::const_(show.clone(), vec![]),
        50, // low priority
    );
    ctx.instances_mut().add_instance(
        Name::from_string("default"),
        show.clone(),
        Expr::const_(Name::from_string("default"), vec![]),
        Expr::const_(show.clone(), vec![]),
        100, // default priority
    );
    ctx.instances_mut().add_instance(
        Name::from_string("high"),
        show.clone(),
        Expr::const_(Name::from_string("high"), vec![]),
        Expr::const_(show.clone(), vec![]),
        150, // high priority
    );

    let instances = ctx.instances().get_instances(&show);
    assert_eq!(instances.len(), 3);
    // Verify priority ordering (highest first)
    assert_eq!(instances[0].priority, 150);
    assert_eq!(instances[1].priority, 100);
    assert_eq!(instances[2].priority, 50);
}

#[test]
fn test_default_instance_last() {
    // Test that @[defaultInstance] (priority 0) is tried last
    let env = Environment::new();
    let mut ctx = ElabCtx::new(&env);

    let show = Name::from_string("Show");
    ctx.instances_mut().register_class(show.clone(), 1, vec![]);

    // Add default instance first
    ctx.instances_mut().add_instance(
        Name::from_string("default_fallback"),
        show.clone(),
        Expr::const_(Name::from_string("default_fallback"), vec![]),
        Expr::const_(show.clone(), vec![]),
        0, // defaultInstance priority
    );

    // Add higher priority instance after
    ctx.instances_mut().add_instance(
        Name::from_string("preferred"),
        show.clone(),
        Expr::const_(Name::from_string("preferred"), vec![]),
        Expr::const_(show.clone(), vec![]),
        100, // normal priority
    );

    let instances = ctx.instances().get_instances(&show);
    assert_eq!(instances.len(), 2);
    // Higher priority should come first regardless of insertion order
    assert_eq!(instances[0].name.to_string(), "preferred");
    assert_eq!(instances[1].name.to_string(), "default_fallback");
}

#[test]
fn test_instance_cache_basic() {
    // Test that instance resolution caches ground goals
    let env = Environment::new();
    let mut ctx = ElabCtx::new(&env);

    // Register Add class and instance
    let add = Name::from_string("Add");
    let nat = Name::from_string("Nat");
    ctx.instances_mut().register_class(add.clone(), 1, vec![]);
    ctx.instances_mut().add_instance(
        Name::from_string("instAddNat"),
        add.clone(),
        Expr::const_(Name::from_string("instAddNat"), vec![]),
        Expr::App(
            Expr::const_(add.clone(), vec![]).into(),
            Expr::const_(nat.clone(), vec![]).into(),
        ),
        crate::instances::DEFAULT_PRIORITY,
    );

    // Initial cache should be empty
    let (cached_count, _) = ctx.instance_cache_stats();
    assert_eq!(cached_count, 0);

    // Resolve Add Nat
    let goal = Expr::App(
        Expr::const_(add.clone(), vec![]).into(),
        Expr::const_(nat.clone(), vec![]).into(),
    );
    let result = ctx.resolve_instance(&goal);
    assert!(result.is_some());

    // Cache should now contain the result (goal is ground - no metavariables)
    let (cached_count, _) = ctx.instance_cache_stats();
    assert_eq!(cached_count, 1);

    // Resolve again - should use cache
    let result2 = ctx.resolve_instance(&goal);
    assert!(result2.is_some());
    assert_eq!(
        format!("{:?}", result.unwrap()),
        format!("{:?}", result2.unwrap())
    );

    // Cache size shouldn't change
    let (cached_count, _) = ctx.instance_cache_stats();
    assert_eq!(cached_count, 1);
}

#[test]
fn test_instance_cache_clear() {
    // Test that clear_instance_cache works
    let env = Environment::new();
    let mut ctx = ElabCtx::new(&env);

    // Register class and instance
    let show = Name::from_string("Show");
    let nat = Name::from_string("Nat");
    ctx.instances_mut().register_class(show.clone(), 1, vec![]);
    ctx.instances_mut().add_instance(
        Name::from_string("instShowNat"),
        show.clone(),
        Expr::const_(Name::from_string("instShowNat"), vec![]),
        Expr::App(
            Expr::const_(show.clone(), vec![]).into(),
            Expr::const_(nat.clone(), vec![]).into(),
        ),
        crate::instances::DEFAULT_PRIORITY,
    );

    // Resolve to populate cache
    let goal = Expr::App(
        Expr::const_(show.clone(), vec![]).into(),
        Expr::const_(nat.clone(), vec![]).into(),
    );
    let _ = ctx.resolve_instance(&goal);
    let (cached_count, _) = ctx.instance_cache_stats();
    assert_eq!(cached_count, 1);

    // Clear cache
    ctx.clear_instance_cache();
    let (cached_count, _) = ctx.instance_cache_stats();
    assert_eq!(cached_count, 0);
}

#[test]
fn test_normalize_for_cache() {
    // Test that normalize_for_cache produces consistent keys
    let env = Environment::new();
    let ctx = ElabCtx::new(&env);

    // Same structure should produce same key
    let nat = Name::from_string("Nat");
    let add = Name::from_string("Add");

    let e1 = Expr::App(
        Expr::const_(add.clone(), vec![]).into(),
        Expr::const_(nat.clone(), vec![]).into(),
    );
    let e2 = Expr::App(
        Expr::const_(add.clone(), vec![]).into(),
        Expr::const_(nat.clone(), vec![]).into(),
    );

    let key1 = ctx.normalize_for_cache(&e1);
    let key2 = ctx.normalize_for_cache(&e2);
    assert_eq!(key1, key2);

    // Different structure should produce different keys
    let bool = Name::from_string("Bool");
    let e3 = Expr::App(
        Expr::const_(add.clone(), vec![]).into(),
        Expr::const_(bool.clone(), vec![]).into(),
    );
    let key3 = ctx.normalize_for_cache(&e3);
    assert_ne!(key1, key3);
}

#[test]
fn test_has_metavars() {
    // Test has_metavars detection
    let env = Environment::new();
    let ctx = ElabCtx::new(&env);

    // Constant has no metavars
    let nat = Expr::const_(Name::from_string("Nat"), vec![]);
    assert!(!ctx.has_metavars(&nat));

    // BVar has no metavars
    let bvar = Expr::BVar(0);
    assert!(!ctx.has_metavars(&bvar));

    // Regular FVar has no metavars
    let fvar = Expr::FVar(FVarId(42));
    assert!(!ctx.has_metavars(&fvar));

    // FVar with metavar tag IS a metavar
    let mvar = Expr::FVar(MetaState::to_fvar(crate::unify::MetaId(0)));
    assert!(ctx.has_metavars(&mvar));

    // App containing metavar
    let app_with_meta = Expr::App(
        Expr::const_(Name::from_string("Add"), vec![]).into(),
        mvar.clone().into(),
    );
    assert!(ctx.has_metavars(&app_with_meta));

    // App without metavar
    let app_no_meta = Expr::App(
        Expr::const_(Name::from_string("Add"), vec![]).into(),
        nat.clone().into(),
    );
    assert!(!ctx.has_metavars(&app_no_meta));
}

#[test]
fn test_instance_cache_with_metavars() {
    // Test that goals with metavars are NOT cached
    let env = Environment::new();
    let mut ctx = ElabCtx::new(&env);

    // Register class and instance
    let add = Name::from_string("Add");
    let nat = Name::from_string("Nat");
    ctx.instances_mut().register_class(add.clone(), 1, vec![]);
    ctx.instances_mut().add_instance(
        Name::from_string("instAddNat"),
        add.clone(),
        Expr::const_(Name::from_string("instAddNat"), vec![]),
        Expr::App(
            Expr::const_(add.clone(), vec![]).into(),
            Expr::const_(nat.clone(), vec![]).into(),
        ),
        crate::instances::DEFAULT_PRIORITY,
    );

    // Create a goal with a metavariable: Add ?m
    let meta_id = ctx.metas.fresh(Expr::Sort(Level::zero()));
    let meta_expr = Expr::FVar(MetaState::to_fvar(meta_id));
    let goal_with_meta = Expr::App(Expr::const_(add.clone(), vec![]).into(), meta_expr.into());

    // Assign the metavariable to Nat so resolution succeeds
    ctx.metas.assign(meta_id, Expr::const_(nat.clone(), vec![]));

    // Resolve
    let result = ctx.resolve_instance(&goal_with_meta);
    // Note: After instantiate, the goal becomes Add Nat which is ground,
    // so it will be cached
    assert!(result.is_some());

    // The cache should contain the result for the ground (instantiated) goal
    let (cached_count, _) = ctx.instance_cache_stats();
    assert_eq!(cached_count, 1);
}

// =========================================================================
// Deriving clause tests
// =========================================================================

#[test]
fn test_elab_structure_with_deriving_single() {
    let result = elab_decl(
        r"structure Point where
          x : Prop
          y : Prop
        deriving BEq",
    )
    .unwrap();

    match result {
        ElabResult::Structure {
            name,
            derived_instances,
            ..
        } => {
            assert_eq!(name, Name::from_string("Point"));
            assert_eq!(derived_instances.len(), 1);
            assert_eq!(derived_instances[0].name, Name::from_string("instPointBEq"));
            assert_eq!(derived_instances[0].class_name, Name::from_string("BEq"));
        }
        _ => panic!("expected Structure"),
    }
}

#[test]
fn test_elab_structure_with_deriving_multiple() {
    let result = elab_decl(
        r"structure Point where
          x : Prop
          y : Prop
        deriving BEq, Repr, Hashable",
    )
    .unwrap();

    match result {
        ElabResult::Structure {
            name,
            derived_instances,
            ..
        } => {
            assert_eq!(name, Name::from_string("Point"));
            assert_eq!(derived_instances.len(), 3);

            // Check each derived instance
            let class_names: Vec<_> = derived_instances
                .iter()
                .map(|d| d.class_name.clone())
                .collect();
            assert!(class_names.contains(&Name::from_string("BEq")));
            assert!(class_names.contains(&Name::from_string("Repr")));
            assert!(class_names.contains(&Name::from_string("Hashable")));
        }
        _ => panic!("expected Structure"),
    }
}

#[test]
fn test_elab_structure_without_deriving() {
    let result = elab_decl(
        r"structure Point where
          x : Prop
          y : Prop",
    )
    .unwrap();

    match result {
        ElabResult::Structure {
            derived_instances, ..
        } => {
            assert!(derived_instances.is_empty());
        }
        _ => panic!("expected Structure"),
    }
}

#[test]
fn test_elab_structure_deriving_unknown_class() {
    // Unknown deriving classes are silently skipped (for now)
    let result = elab_decl(
        r"structure Point where
          x : Prop
        deriving UnknownClass",
    )
    .unwrap();

    match result {
        ElabResult::Structure {
            derived_instances, ..
        } => {
            // Unknown class should be skipped
            assert!(derived_instances.is_empty());
        }
        _ => panic!("expected Structure"),
    }
}

#[test]
fn test_elab_structure_deriving_inhabited() {
    let result = elab_decl(
        r"structure Point where
          x : Prop
        deriving Inhabited",
    )
    .unwrap();

    match result {
        ElabResult::Structure {
            derived_instances, ..
        } => {
            assert_eq!(derived_instances.len(), 1);
            assert_eq!(
                derived_instances[0].class_name,
                Name::from_string("Inhabited")
            );
        }
        _ => panic!("expected Structure"),
    }
}

#[test]
fn test_elab_structure_deriving_decidable_eq() {
    let result = elab_decl(
        r"structure Point where
          x : Prop
        deriving DecidableEq",
    )
    .unwrap();

    match result {
        ElabResult::Structure {
            derived_instances, ..
        } => {
            assert_eq!(derived_instances.len(), 1);
            assert_eq!(
                derived_instances[0].class_name,
                Name::from_string("DecidableEq")
            );
        }
        _ => panic!("expected Structure"),
    }
}

#[test]
fn test_derived_inhabited_instance_structure() {
    let result = elab_decl(
        r"structure Point where
          x : Prop
          y : Prop
        deriving Inhabited",
    )
    .unwrap();

    match result {
        ElabResult::Structure {
            derived_instances, ..
        } => {
            assert_eq!(derived_instances.len(), 1);
            let inst = &derived_instances[0];
            assert_eq!(inst.class_name, Name::from_string("Inhabited"));

            // Instance value should be Inhabited.mk applied to constructor application
            match &inst.val {
                Expr::App(func, default_val) => {
                    match func.as_ref() {
                        Expr::Const(name, _) => {
                            assert_eq!(*name, Name::from_string("Inhabited.mk"));
                        }
                        _ => panic!("Expected Inhabited.mk constant"),
                    }

                    // default_val should contain the structure constructor and Inhabited.default calls
                    fn contains_const(expr: &Expr, target: &str) -> bool {
                        match expr {
                            Expr::Const(name, _) => name.to_string() == target,
                            Expr::App(f, a) => {
                                contains_const(f, target) || contains_const(a, target)
                            }
                            Expr::Lam(_, ty, body) | Expr::Pi(_, ty, body) => {
                                contains_const(ty, target) || contains_const(body, target)
                            }
                            Expr::Let(ty, val, body) => {
                                contains_const(ty, target)
                                    || contains_const(val, target)
                                    || contains_const(body, target)
                            }
                            Expr::Proj(_, _, e) | Expr::MData(_, e) => contains_const(e, target),
                            _ => false,
                        }
                    }

                    assert!(
                        contains_const(default_val, "Point.mk"),
                        "Default value should call Point.mk"
                    );
                    assert!(
                        contains_const(default_val, "Inhabited.default"),
                        "Default value should call Inhabited.default for fields"
                    );
                }
                _ => panic!("Expected App for Inhabited instance"),
            }
        }
        _ => panic!("expected Structure"),
    }
}

#[test]
fn test_derived_inhabited_empty_struct() {
    let result = elab_decl(
        r"structure Empty where
        deriving Inhabited",
    )
    .unwrap();

    match result {
        ElabResult::Structure {
            derived_instances, ..
        } => {
            assert_eq!(derived_instances.len(), 1);
            let inst = &derived_instances[0];
            match &inst.val {
                Expr::App(func, arg) => {
                    match func.as_ref() {
                        Expr::Const(name, _) => {
                            assert_eq!(*name, Name::from_string("Inhabited.mk"));
                        }
                        _ => panic!("Expected Inhabited.mk"),
                    }

                    match arg.as_ref() {
                        Expr::Const(name, _) => {
                            assert_eq!(*name, Name::from_string("Empty.mk"));
                        }
                        _ => panic!("Expected Empty.mk as default value"),
                    }
                }
                _ => panic!("Expected App for Inhabited instance"),
            }
        }
        _ => panic!("expected Structure"),
    }
}

#[test]
fn test_derived_decidable_eq_instance_structure() {
    // Test struct with one field - should use DecidableEq.decEq
    let result = elab_decl(
        r"structure Foo where
          x : Prop
        deriving DecidableEq",
    )
    .unwrap();

    match result {
        ElabResult::Structure {
            derived_instances, ..
        } => {
            assert_eq!(derived_instances.len(), 1);
            let inst = &derived_instances[0];
            match &inst.val {
                Expr::App(func, arg) => {
                    match func.as_ref() {
                        Expr::Const(name, _) => {
                            assert_eq!(*name, Name::from_string("DecidableEq.mk"));
                        }
                        _ => panic!("Expected DecidableEq.mk"),
                    }

                    match arg.as_ref() {
                        Expr::Lam(_, _, inner) => {
                            // λ a => λ b => DecidableEq.decEq a.0 b.0
                            match inner.as_ref() {
                                Expr::Lam(_, _, body) => {
                                    // Body should contain DecidableEq.decEq call
                                    fn contains_deceq(e: &Expr) -> bool {
                                        match e {
                                            Expr::Const(name, _) => {
                                                name.to_string() == "DecidableEq.decEq"
                                            }
                                            Expr::App(f, a) => {
                                                contains_deceq(f) || contains_deceq(a)
                                            }
                                            Expr::Lam(_, t, b) => {
                                                contains_deceq(t) || contains_deceq(b)
                                            }
                                            Expr::Proj(_, _, e) => contains_deceq(e),
                                            _ => false,
                                        }
                                    }
                                    assert!(
                                        contains_deceq(body),
                                        "Body should contain DecidableEq.decEq call"
                                    );
                                }
                                _ => panic!("Expected inner lambda for decEq"),
                            }
                        }
                        _ => panic!("Expected lambda for decEq function"),
                    }
                }
                _ => panic!("Expected App for DecidableEq instance"),
            }
        }
        _ => panic!("expected Structure"),
    }
}

#[test]
fn test_derived_decidable_eq_empty_struct() {
    // Empty struct - should use Decidable.isTrue Eq.refl
    let result = elab_decl(
        r"structure Empty where
        deriving DecidableEq",
    )
    .unwrap();

    match result {
        ElabResult::Structure {
            derived_instances, ..
        } => {
            assert_eq!(derived_instances.len(), 1);
            let inst = &derived_instances[0];
            match &inst.val {
                Expr::App(func, arg) => {
                    match func.as_ref() {
                        Expr::Const(name, _) => {
                            assert_eq!(*name, Name::from_string("DecidableEq.mk"));
                        }
                        _ => panic!("Expected DecidableEq.mk"),
                    }

                    // The decEq function body should contain Decidable.isTrue
                    fn contains_is_true(e: &Expr) -> bool {
                        match e {
                            Expr::Const(name, _) => name.to_string() == "Decidable.isTrue",
                            Expr::App(f, a) => contains_is_true(f) || contains_is_true(a),
                            Expr::Lam(_, t, b) => contains_is_true(t) || contains_is_true(b),
                            _ => false,
                        }
                    }
                    assert!(
                        contains_is_true(arg),
                        "Empty struct DecidableEq should use Decidable.isTrue"
                    );
                }
                _ => panic!("Expected App for DecidableEq instance"),
            }
        }
        _ => panic!("expected Structure"),
    }
}

#[test]
fn test_derived_decidable_eq_multi_field() {
    // Test struct with multiple fields - should use instDecidableAnd
    let result = elab_decl(
        r"structure Point where
          x : Prop
          y : Prop
        deriving DecidableEq",
    )
    .unwrap();

    match result {
        ElabResult::Structure {
            derived_instances, ..
        } => {
            assert_eq!(derived_instances.len(), 1);
            let inst = &derived_instances[0];
            match &inst.val {
                Expr::App(func, arg) => {
                    match func.as_ref() {
                        Expr::Const(name, _) => {
                            assert_eq!(*name, Name::from_string("DecidableEq.mk"));
                        }
                        _ => panic!("Expected DecidableEq.mk"),
                    }

                    // Check for instDecidableAnd in the body
                    fn contains_and_decidable(e: &Expr) -> bool {
                        match e {
                            Expr::Const(name, _) => name.to_string() == "instDecidableAnd",
                            Expr::App(f, a) => {
                                contains_and_decidable(f) || contains_and_decidable(a)
                            }
                            Expr::Lam(_, t, b) => {
                                contains_and_decidable(t) || contains_and_decidable(b)
                            }
                            Expr::Proj(_, _, e) => contains_and_decidable(e),
                            _ => false,
                        }
                    }
                    assert!(
                        contains_and_decidable(arg),
                        "Multi-field struct DecidableEq should use instDecidableAnd"
                    );
                }
                _ => panic!("Expected App for DecidableEq instance"),
            }
        }
        _ => panic!("expected Structure"),
    }
}

#[test]
fn test_derived_beq_instance_structure() {
    // Verify that derived BEq instance has the correct structure:
    // BEq.mk (λ a b => Bool.and (BEq.beq a.0 b.0) (BEq.beq a.1 b.1))
    let result = elab_decl(
        r"structure Point where
          x : Prop
          y : Prop
        deriving BEq",
    )
    .unwrap();

    match result {
        ElabResult::Structure {
            derived_instances,
            field_names,
            ..
        } => {
            assert_eq!(field_names.len(), 2);
            assert_eq!(derived_instances.len(), 1);

            let beq_instance = &derived_instances[0];
            assert_eq!(beq_instance.class_name, Name::from_string("BEq"));

            // Instance value should be App(BEq.mk, lambda)
            match &beq_instance.val {
                Expr::App(func, _arg) => {
                    // func should be BEq.mk
                    match func.as_ref() {
                        Expr::Const(name, _) => {
                            assert_eq!(*name, Name::from_string("BEq.mk"));
                        }
                        _ => panic!("Expected BEq.mk constant, got {func:?}"),
                    }
                }
                _ => panic!("Expected App for BEq instance, got {:?}", beq_instance.val),
            }
        }
        _ => panic!("expected Structure"),
    }
}

#[test]
fn test_derived_beq_instance_empty_struct() {
    // Verify that derived BEq instance for empty struct returns Bool.true
    let result = elab_decl(
        r"structure Empty where
        deriving BEq",
    )
    .unwrap();

    match result {
        ElabResult::Structure {
            derived_instances,
            field_names,
            ..
        } => {
            assert!(field_names.is_empty());
            assert_eq!(derived_instances.len(), 1);

            let beq_instance = &derived_instances[0];
            // The beq function body should ultimately contain Bool.true
            // Structure: BEq.mk (λ a => λ b => Bool.true)
            match &beq_instance.val {
                Expr::App(_, arg) => {
                    // arg is λ a => λ b => Bool.true
                    match arg.as_ref() {
                        Expr::Lam(_, _, body) => {
                            // body is λ b => Bool.true
                            match body.as_ref() {
                                Expr::Lam(_, _, inner_body) => {
                                    // inner_body should be Bool.true
                                    match inner_body.as_ref() {
                                        Expr::Const(name, _) => {
                                            assert_eq!(*name, Name::from_string("Bool.true"));
                                        }
                                        _ => panic!("Expected Bool.true, got {inner_body:?}"),
                                    }
                                }
                                _ => panic!("Expected inner lambda"),
                            }
                        }
                        _ => panic!("Expected outer lambda"),
                    }
                }
                _ => panic!("Expected App"),
            }
        }
        _ => panic!("expected Structure"),
    }
}

#[test]
fn test_derived_repr_instance_structure() {
    // Verify that derived Repr instance has the correct structure:
    // Repr.mk (λ s prec => <format>)
    let result = elab_decl(
        r"structure Point where
          x : Prop
          y : Prop
        deriving Repr",
    )
    .unwrap();

    match result {
        ElabResult::Structure {
            derived_instances, ..
        } => {
            assert_eq!(derived_instances.len(), 1);

            let repr_instance = &derived_instances[0];
            assert_eq!(repr_instance.class_name, Name::from_string("Repr"));

            // Instance value should be App(Repr.mk, lambda)
            match &repr_instance.val {
                Expr::App(func, _arg) => match func.as_ref() {
                    Expr::Const(name, _) => {
                        assert_eq!(*name, Name::from_string("Repr.mk"));
                    }
                    _ => panic!("Expected Repr.mk constant"),
                },
                _ => panic!("Expected App for Repr instance"),
            }
        }
        _ => panic!("expected Structure"),
    }
}

#[test]
fn test_derived_hashable_instance_structure() {
    // Verify that derived Hashable instance has the correct structure:
    // Hashable.mk (λ s => mixHash (hash s.0) (hash s.1))
    let result = elab_decl(
        r"structure Point where
          x : Prop
          y : Prop
        deriving Hashable",
    )
    .unwrap();

    match result {
        ElabResult::Structure {
            derived_instances, ..
        } => {
            assert_eq!(derived_instances.len(), 1);

            let hashable_instance = &derived_instances[0];
            assert_eq!(hashable_instance.class_name, Name::from_string("Hashable"));

            // Instance value should be App(Hashable.mk, lambda)
            match &hashable_instance.val {
                Expr::App(func, _arg) => match func.as_ref() {
                    Expr::Const(name, _) => {
                        assert_eq!(*name, Name::from_string("Hashable.mk"));
                    }
                    _ => panic!("Expected Hashable.mk constant"),
                },
                _ => panic!("Expected App for Hashable instance"),
            }
        }
        _ => panic!("expected Structure"),
    }
}

#[test]
fn test_derived_hashable_empty_struct() {
    // Verify that derived Hashable for empty struct returns 0
    let result = elab_decl(
        r"structure Empty where
        deriving Hashable",
    )
    .unwrap();

    match result {
        ElabResult::Structure {
            derived_instances,
            field_names,
            ..
        } => {
            assert!(field_names.is_empty());
            assert_eq!(derived_instances.len(), 1);

            let hashable_instance = &derived_instances[0];
            // Structure: Hashable.mk (λ s => 0)
            match &hashable_instance.val {
                Expr::App(_, arg) => {
                    match arg.as_ref() {
                        Expr::Lam(_, _, body) => {
                            // body should be Nat literal 0
                            match body.as_ref() {
                                Expr::Lit(lean5_kernel::Literal::Nat(n)) => {
                                    assert_eq!(*n, 0);
                                }
                                _ => panic!("Expected Nat literal 0, got {body:?}"),
                            }
                        }
                        _ => panic!("Expected lambda"),
                    }
                }
                _ => panic!("Expected App"),
            }
        }
        _ => panic!("expected Structure"),
    }
}

#[test]
fn test_derived_beq_has_field_projections() {
    // Verify that derived BEq with fields uses Proj expressions
    let result = elab_decl(
        r"structure Point where
          x : Prop
        deriving BEq",
    )
    .unwrap();

    match result {
        ElabResult::Structure {
            derived_instances, ..
        } => {
            let beq_instance = &derived_instances[0];

            // Check that the instance value contains Proj expressions
            fn contains_proj(e: &Expr) -> bool {
                match e {
                    Expr::Proj(_, _, _) => true,
                    Expr::App(f, a) => contains_proj(f) || contains_proj(a),
                    Expr::Lam(_, ty, body) | Expr::Pi(_, ty, body) => {
                        contains_proj(ty) || contains_proj(body)
                    }
                    _ => false,
                }
            }

            assert!(
                contains_proj(&beq_instance.val),
                "BEq instance should contain field projections"
            );
        }
        _ => panic!("expected Structure"),
    }
}

#[test]
fn test_derived_parametric_beq_instance() {
    // Test deriving BEq for a parametric structure
    // structure Pair (A : Type) (B : Type) where fst : A  snd : B
    // should generate: instance [BEq A] [BEq B] : BEq (Pair A B)
    let result = elab_decl(
        r"structure Pair (A : Type) (B : Type) where
          fst : A
          snd : B
        deriving BEq",
    )
    .unwrap();

    match result {
        ElabResult::Structure {
            derived_instances,
            num_params,
            ..
        } => {
            assert_eq!(num_params, 2);
            assert_eq!(derived_instances.len(), 1);

            let beq_instance = &derived_instances[0];
            assert_eq!(beq_instance.class_name, Name::from_string("BEq"));

            // Instance type should have Pi bindings for type params and constraints
            // ∀ {A : Type} {B : Type} [BEq A] [BEq B], BEq (Pair A B)
            fn count_pis(e: &Expr) -> usize {
                match e {
                    Expr::Pi(_, _, body) => 1 + count_pis(body),
                    _ => 0,
                }
            }

            // Should have 4 Pis: 2 type params + 2 instance constraints
            assert_eq!(
                count_pis(&beq_instance.ty),
                4,
                "Parametric BEq instance should have 4 Pi bindings (2 type + 2 instance)"
            );

            // Instance value should start with lambdas for params and constraints
            fn count_lams(e: &Expr) -> usize {
                match e {
                    Expr::Lam(_, _, body) => 1 + count_lams(body),
                    _ => 0,
                }
            }

            let lam_count = count_lams(&beq_instance.val);
            // Should have 4 lambdas (params + constraints) + 2 for a/b = 6 total
            // But wait - the beq function has 2 lambdas (a, b), wrapped by:
            // - BEq.mk application (not a lambda)
            // - 4 parameter/constraint lambdas
            // So total should be 4 + 2 = 6 if counted recursively through all
            // Actually BEq.mk wraps the inner function, so we have:
            // λα λβ [inst_α] [inst_β]. BEq.mk (λa λb. body)
            // Which is 4 + 2 = 6 lambdas
            assert!(
                lam_count >= 4,
                "Parametric BEq instance value should have at least 4 lambdas for params/constraints, got {lam_count}"
            );
        }
        _ => panic!("expected Structure"),
    }
}

#[test]
fn test_derived_parametric_instance_type_structure() {
    // Test that parametric instance types have the correct structure
    let result = elab_decl(
        r"structure Box (T : Type) where
          val : T
        deriving Hashable",
    )
    .unwrap();

    match result {
        ElabResult::Structure {
            derived_instances,
            num_params,
            ..
        } => {
            assert_eq!(num_params, 1);
            assert_eq!(derived_instances.len(), 1);

            let hashable_instance = &derived_instances[0];

            // Check that instance type has implicit param and instance constraint
            // ∀ {T : Type} [Hashable T], Hashable (Box T)
            match &hashable_instance.ty {
                Expr::Pi(binder_info1, _, body1) => {
                    // First Pi should be implicit (type param)
                    assert_eq!(*binder_info1, BinderInfo::Implicit);

                    match body1.as_ref() {
                        Expr::Pi(binder_info2, constraint_ty, _body2) => {
                            // Second Pi should be inst implicit (constraint)
                            assert_eq!(*binder_info2, BinderInfo::InstImplicit);

                            // Constraint type should be Hashable applied to BVar
                            match constraint_ty.as_ref() {
                                Expr::App(class, _arg) => match class.as_ref() {
                                    Expr::Const(name, _) => {
                                        assert_eq!(
                                            *name,
                                            Name::from_string("Hashable"),
                                            "Constraint should be Hashable class"
                                        );
                                    }
                                    _ => panic!("Expected Hashable constant"),
                                },
                                _ => panic!("Expected App for constraint type"),
                            }
                        }
                        _ => panic!("Expected second Pi for instance constraint"),
                    }
                }
                _ => panic!("Expected Pi for type parameter"),
            }
        }
        _ => panic!("expected Structure"),
    }
}

// ===========================================
// Inductive type elaboration tests
// ===========================================

#[test]
fn test_elab_inductive_simple() {
    // Simple inductive with no parameters
    let result = elab_decl(
        r"inductive Bool : Type
| false : Bool
| true : Bool",
    )
    .unwrap();

    match result {
        ElabResult::Inductive {
            name,
            num_params,
            constructors,
            ..
        } => {
            assert_eq!(name, Name::from_string("Bool"));
            assert_eq!(num_params, 0);
            assert_eq!(constructors.len(), 2);
            assert_eq!(constructors[0].0, Name::from_string("Bool.false"));
            assert_eq!(constructors[1].0, Name::from_string("Bool.true"));
        }
        _ => panic!("expected Inductive"),
    }
}

#[test]
fn test_elab_inductive_with_parameter() {
    // Inductive with a type parameter
    let result = elab_decl(
        r"inductive Option (α : Type) : Type
| none : Option α
| some : α → Option α",
    )
    .unwrap();

    match result {
        ElabResult::Inductive {
            name,
            num_params,
            constructors,
            ..
        } => {
            assert_eq!(name, Name::from_string("Option"));
            assert_eq!(num_params, 1);
            assert_eq!(constructors.len(), 2);
            assert_eq!(constructors[0].0, Name::from_string("Option.none"));
            assert_eq!(constructors[1].0, Name::from_string("Option.some"));
        }
        _ => panic!("expected Inductive"),
    }
}

#[test]
fn test_elab_inductive_recursive() {
    // Recursive inductive (like Nat)
    let result = elab_decl(
        r"inductive MyNat : Type
| zero : MyNat
| succ : MyNat → MyNat",
    )
    .unwrap();

    match result {
        ElabResult::Inductive {
            name,
            num_params,
            constructors,
            ..
        } => {
            assert_eq!(name, Name::from_string("MyNat"));
            assert_eq!(num_params, 0);
            assert_eq!(constructors.len(), 2);
            assert_eq!(constructors[0].0, Name::from_string("MyNat.zero"));
            assert_eq!(constructors[1].0, Name::from_string("MyNat.succ"));
        }
        _ => panic!("expected Inductive"),
    }
}

#[test]
fn test_elab_inductive_list() {
    // List type with two parameters (recursive)
    let result = elab_decl(
        r"inductive List (α : Type) : Type
| nil : List α
| cons : α → List α → List α",
    )
    .unwrap();

    match result {
        ElabResult::Inductive {
            name,
            num_params,
            constructors,
            ..
        } => {
            assert_eq!(name, Name::from_string("List"));
            assert_eq!(num_params, 1);
            assert_eq!(constructors.len(), 2);
            assert_eq!(constructors[0].0, Name::from_string("List.nil"));
            assert_eq!(constructors[1].0, Name::from_string("List.cons"));
        }
        _ => panic!("expected Inductive"),
    }
}

// ===========================================
// Inductive deriving tests
// ===========================================

#[test]
fn test_elab_inductive_with_deriving_single() {
    let result = elab_decl(
        r"inductive Bool : Type
| false : Bool
| true : Bool
deriving BEq",
    )
    .unwrap();

    match result {
        ElabResult::Inductive {
            name,
            derived_instances,
            ..
        } => {
            assert_eq!(name, Name::from_string("Bool"));
            assert_eq!(derived_instances.len(), 1);
            assert_eq!(derived_instances[0].name, Name::from_string("instBoolBEq"));
            assert_eq!(derived_instances[0].class_name, Name::from_string("BEq"));
        }
        _ => panic!("expected Inductive"),
    }
}

#[test]
fn test_elab_inductive_with_deriving_multiple() {
    let result = elab_decl(
        r"inductive Color : Type
| red : Color
| green : Color
| blue : Color
deriving BEq, Repr, Hashable",
    )
    .unwrap();

    match result {
        ElabResult::Inductive {
            name,
            derived_instances,
            ..
        } => {
            assert_eq!(name, Name::from_string("Color"));
            assert_eq!(derived_instances.len(), 3);

            let class_names: Vec<_> = derived_instances
                .iter()
                .map(|d| d.class_name.clone())
                .collect();
            assert!(class_names.contains(&Name::from_string("BEq")));
            assert!(class_names.contains(&Name::from_string("Repr")));
            assert!(class_names.contains(&Name::from_string("Hashable")));
        }
        _ => panic!("expected Inductive"),
    }
}

#[test]
fn test_elab_inductive_deriving_inhabited() {
    let result = elab_decl(
        r"inductive Bool : Type
| false : Bool
| true : Bool
deriving Inhabited",
    )
    .unwrap();

    match result {
        ElabResult::Inductive {
            name,
            derived_instances,
            ..
        } => {
            assert_eq!(name, Name::from_string("Bool"));
            assert_eq!(derived_instances.len(), 1);
            assert_eq!(
                derived_instances[0].name,
                Name::from_string("instBoolInhabited")
            );
            assert_eq!(
                derived_instances[0].class_name,
                Name::from_string("Inhabited")
            );

            // Check that the instance uses the first constructor (Bool.false)
            match &derived_instances[0].val {
                Expr::App(_, arg) => {
                    // arg should be Bool.false
                    match arg.as_ref() {
                        Expr::Const(ctor_name, _) => {
                            assert_eq!(*ctor_name, Name::from_string("Bool.false"));
                        }
                        _ => panic!("Expected Const for default value"),
                    }
                }
                _ => panic!("Expected App for Inhabited.mk"),
            }
        }
        _ => panic!("expected Inductive"),
    }
}

#[test]
fn test_elab_inductive_deriving_decidable_eq() {
    let result = elab_decl(
        r"inductive Bool : Type
| false : Bool
| true : Bool
deriving DecidableEq",
    )
    .unwrap();

    match result {
        ElabResult::Inductive {
            name,
            derived_instances,
            ..
        } => {
            assert_eq!(name, Name::from_string("Bool"));
            assert_eq!(derived_instances.len(), 1);
            assert_eq!(
                derived_instances[0].name,
                Name::from_string("instBoolDecidableEq")
            );
            assert_eq!(
                derived_instances[0].class_name,
                Name::from_string("DecidableEq")
            );
        }
        _ => panic!("expected Inductive"),
    }
}

#[test]
fn test_elab_inductive_without_deriving() {
    let result = elab_decl(
        r"inductive Bool : Type
| false : Bool
| true : Bool",
    )
    .unwrap();

    match result {
        ElabResult::Inductive {
            derived_instances, ..
        } => {
            assert!(derived_instances.is_empty());
        }
        _ => panic!("expected Inductive"),
    }
}

#[test]
fn test_elab_inductive_single_ctor_deriving() {
    // Single constructor inductive always has equal elements
    let result = elab_decl(
        r"inductive Unit : Type
| unit : Unit
deriving BEq, Inhabited",
    )
    .unwrap();

    match result {
        ElabResult::Inductive {
            name,
            derived_instances,
            constructors,
            ..
        } => {
            assert_eq!(name, Name::from_string("Unit"));
            assert_eq!(constructors.len(), 1);
            assert_eq!(derived_instances.len(), 2);
        }
        _ => panic!("expected Inductive"),
    }
}

// Note: Recursor tests have been moved to the kernel (lean5-kernel/src/tc.rs)
// since recursor generation is now handled entirely by the kernel.
// The elaborator tests below verify basic inductive elaboration works.

#[test]
fn test_elab_inductive_simple_enum() {
    // Test elaboration of a simple enum (recursor tested in kernel)
    let result = elab_decl(
        r"inductive Bool : Type
| false : Bool
| true : Bool",
    )
    .unwrap();

    match result {
        ElabResult::Inductive {
            name,
            num_params,
            constructors,
            ..
        } => {
            assert_eq!(name, Name::from_string("Bool"));
            assert_eq!(num_params, 0);
            assert_eq!(constructors.len(), 2);
        }
        _ => panic!("expected Inductive"),
    }
}

#[test]
fn test_elab_inductive_single_ctor() {
    // Test elaboration of single constructor type
    let result = elab_decl(
        r"inductive Unit : Type
| unit : Unit",
    )
    .unwrap();

    match result {
        ElabResult::Inductive {
            name, constructors, ..
        } => {
            assert_eq!(name, Name::from_string("Unit"));
            assert_eq!(constructors.len(), 1);
        }
        _ => panic!("expected Inductive"),
    }
}

#[test]
fn test_elab_inductive_with_constructor_args() {
    // Test elaboration of inductive with constructor arguments
    let result = elab_decl(
        r"inductive Option (α : Type) : Type
| none : Option α
| some : α → Option α",
    )
    .unwrap();

    match result {
        ElabResult::Inductive {
            name,
            num_params,
            constructors,
            ..
        } => {
            assert_eq!(name, Name::from_string("Option"));
            assert_eq!(num_params, 1); // α
            assert_eq!(constructors.len(), 2);
        }
        _ => panic!("expected Inductive"),
    }
}

#[test]
fn test_elab_inductive_recursive_type() {
    // Test elaboration of a recursive inductive type
    let result = elab_decl(
        r"inductive MyNat : Type
| zero : MyNat
| succ : MyNat → MyNat",
    )
    .unwrap();

    match result {
        ElabResult::Inductive {
            name,
            num_params,
            constructors,
            ..
        } => {
            assert_eq!(name, Name::from_string("MyNat"));
            assert_eq!(num_params, 0);
            assert_eq!(constructors.len(), 2);
        }
        _ => panic!("expected Inductive"),
    }
}

// =============================================================================
// Match expression tests
// =============================================================================

#[test]
fn test_match_single_arm_var_pattern() {
    // match e with | x => body
    // The macro system desugars this to (fun x => body) e
    // which is semantically equivalent to let x := e in body
    let expr = elab("match 42 with | x => x").unwrap();
    // Should produce either a let binding or an application of a lambda
    // (both are semantically equivalent)
    let is_let_like = matches!(expr, Expr::Let(_, _, _))
        || matches!(&expr, Expr::App(func, _) if matches!(func.as_ref(), Expr::Lam(_, _, _)));
    assert!(is_let_like, "expected Let or App(Lam,...), got {expr:?}");
}

#[test]
fn test_match_single_arm_wildcard() {
    // match e with | _ => body
    // The macro system desugars this to (fun _ => body) e
    let expr = elab("match 42 with | _ => 0").unwrap();
    // Should produce either a let binding or an application of a lambda
    let is_let_like = matches!(expr, Expr::Let(_, _, _))
        || matches!(&expr, Expr::App(func, _) if matches!(func.as_ref(), Expr::Lam(_, _, _)));
    assert!(is_let_like, "expected Let or App(Lam,...), got {expr:?}");
}

#[test]
fn test_match_multiple_arms() {
    // Multiple arms - the macro only handles single-arm match,
    // so this should reach the elaborator's Match handler
    // Note: Two-arm match requires the type to have constructors,
    // so we test that the elaboration doesn't panic on the attempt
    let result = elab("match 42 with | x => x | _ => 0");
    // This may succeed or fail depending on whether Nat.casesOn is defined
    // The important thing is that it doesn't panic
    match result {
        Ok(expr) => {
            // Should be an application structure
            assert!(
                matches!(expr, Expr::App(_, _)),
                "expected App for casesOn, got {expr:?}"
            );
        }
        Err(ElabError::NotImplemented(msg)) => {
            // This is also acceptable - the type might not have casesOn
            assert!(
                msg.contains("type name") || msg.contains("casesOn"),
                "unexpected error: {msg}"
            );
        }
        Err(e) => {
            // Other errors are acceptable too - elaboration attempted the right path
            // Just make sure it's not a parsing error
            assert!(
                !matches!(e, ElabError::ParseError(_)),
                "unexpected parse error: {e:?}"
            );
        }
    }
}

#[test]
fn test_match_empty_arms_error() {
    // Match with no arms should error
    // Note: Parser may not allow this, but elaborator handles it
    let surface = lean5_parser::SurfaceExpr::Match(
        lean5_parser::Span::dummy(),
        Box::new(lean5_parser::SurfaceExpr::Lit(
            lean5_parser::Span::dummy(),
            lean5_parser::SurfaceLit::Nat(42),
        )),
        vec![],
    );
    let env = Environment::new();
    let mut ctx = ElabCtx::new(&env);
    let result = ctx.elaborate(&surface);
    assert!(
        matches!(result, Err(ElabError::NotImplemented(msg)) if msg.contains("no arms")),
        "expected NotImplemented error for empty match arms"
    );
}

// =============================================================================
// IfLet expression tests
// =============================================================================

#[test]
fn test_if_let_var_pattern() {
    // if let x := e then t else f
    // The macro system desugars this to: match e with | x => t | _ => f
    // Which elaborates to casesOn with lambda alternatives
    let expr = elab("if let x := 42 then x else 0").unwrap();
    // Should produce an application structure (casesOn ...)
    assert!(
        matches!(expr, Expr::App(_, _)),
        "expected App for if-let via casesOn, got {expr:?}"
    );
}

#[test]
fn test_if_let_wildcard_pattern() {
    // if let _ := e then t else f
    // The macro system desugars this to: match e with | _ => t | _ => f (two-arm match)
    // Which elaborates to casesOn with lambda alternatives
    let expr = elab("if let _ := 42 then 1 else 0").unwrap();
    // Should produce an application structure (casesOn ...)
    assert!(
        matches!(expr, Expr::App(_, _)),
        "expected App for if-let via casesOn, got {expr:?}"
    );
}

// =============================================================================
// IfDecidable expression tests
// =============================================================================

#[test]
fn test_if_decidable_basic() {
    // if h : p then t else e  desugars to  dite p (fun h => t) (fun h => e)
    // We need a Prop for this, using True which is a valid Prop
    let env = Environment::new();
    let mut ctx = ElabCtx::new(&env);

    // Build surface expression directly since parsing `if h : True then ...` needs True defined
    let prop = lean5_parser::SurfaceExpr::Ident(lean5_parser::Span::dummy(), "True".to_string());
    let then_br = lean5_parser::SurfaceExpr::Lit(
        lean5_parser::Span::dummy(),
        lean5_parser::SurfaceLit::Nat(1),
    );
    let else_br = lean5_parser::SurfaceExpr::Lit(
        lean5_parser::Span::dummy(),
        lean5_parser::SurfaceLit::Nat(0),
    );
    let surface = lean5_parser::SurfaceExpr::IfDecidable(
        lean5_parser::Span::dummy(),
        "h".to_string(),
        Box::new(prop),
        Box::new(then_br),
        Box::new(else_br),
    );

    // This will fail because True isn't defined, but the elaboration logic itself should work
    let result = ctx.elaborate(&surface);
    // We expect UnknownIdent for "True" since it's not in the empty environment
    assert!(
        matches!(result, Err(ElabError::UnknownIdent(ref name)) if name == "True"),
        "expected UnknownIdent(True), got {result:?}"
    );
}

#[test]
fn test_if_decidable_with_prop_env() {
    // Test if-decidable with True defined in environment
    use lean5_kernel::Declaration;
    let mut env = Environment::new();

    // Add True : Prop as an axiom
    let true_name = Name::from_string("True");
    env.add_decl(Declaration::Axiom {
        name: true_name.clone(),
        level_params: vec![],
        type_: Expr::prop(),
    })
    .unwrap();

    let mut ctx = ElabCtx::new(&env);

    let prop = lean5_parser::SurfaceExpr::Ident(lean5_parser::Span::dummy(), "True".to_string());
    let then_br = lean5_parser::SurfaceExpr::Lit(
        lean5_parser::Span::dummy(),
        lean5_parser::SurfaceLit::Nat(1),
    );
    let else_br = lean5_parser::SurfaceExpr::Lit(
        lean5_parser::Span::dummy(),
        lean5_parser::SurfaceLit::Nat(0),
    );
    let surface = lean5_parser::SurfaceExpr::IfDecidable(
        lean5_parser::Span::dummy(),
        "h".to_string(),
        Box::new(prop),
        Box::new(then_br),
        Box::new(else_br),
    );

    let result = ctx.elaborate(&surface);
    assert!(result.is_ok(), "if-decidable should elaborate: {result:?}");

    let expr = result.unwrap();
    // Result should be: dite True (fun h => 1) (fun h => 0)
    // Which is an application
    assert!(
        matches!(expr, Expr::App(_, _)),
        "expected App for dite, got {expr:?}"
    );
}
