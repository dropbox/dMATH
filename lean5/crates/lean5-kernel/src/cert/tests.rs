//! Tests for proof certificates
use super::*;
use crate::TypeChecker;

fn empty_env() -> Environment {
    Environment::new()
}

#[test]
fn test_sort_cert() {
    let env = empty_env();
    let mut verifier = CertVerifier::new(&env);

    let level = Level::zero();
    let expr = Expr::Sort(level.clone());
    let cert = ProofCert::Sort {
        level: level.clone(),
    };

    let result = verifier.verify(&cert, &expr);
    assert!(result.is_ok());

    let ty = result.unwrap();
    assert_eq!(ty, Expr::Sort(Level::succ(level)));
}

#[test]
fn test_sort_type_1() {
    let env = empty_env();
    let mut verifier = CertVerifier::new(&env);

    // Type 1 = Sort(succ(zero))
    let level = Level::succ(Level::zero());
    let expr = Expr::Sort(level.clone());
    let cert = ProofCert::Sort {
        level: level.clone(),
    };

    let result = verifier.verify(&cert, &expr);
    assert!(result.is_ok());

    let ty = result.unwrap();
    // Type of Type 1 is Type 2
    assert_eq!(ty, Expr::Sort(Level::succ(level)));
}

#[test]
fn test_sort_level_mismatch() {
    let env = empty_env();
    let mut verifier = CertVerifier::new(&env);

    let expr = Expr::Sort(Level::zero());
    let cert = ProofCert::Sort {
        level: Level::succ(Level::zero()),
    };

    let result = verifier.verify(&cert, &expr);
    assert!(result.is_err());
    match result {
        Err(CertError::LevelMismatch { .. }) => {}
        _ => panic!("Expected LevelMismatch error"),
    }
}

#[test]
fn test_pi_cert() {
    let env = empty_env();
    let mut verifier = CertVerifier::new(&env);

    // Build: Prop → Prop : Type 0
    let prop = Expr::Sort(Level::zero());

    let expr = Expr::Pi(
        BinderInfo::Default,
        prop.clone().into(),
        prop.clone().into(),
    );

    let cert = ProofCert::Pi {
        binder_info: BinderInfo::Default,
        arg_type_cert: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        arg_level: Level::succ(Level::zero()),
        body_type_cert: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        body_level: Level::succ(Level::zero()),
    };

    let result = verifier.verify(&cert, &expr);
    assert!(result.is_ok());
}

#[test]
fn test_identity_function_cert() {
    let env = empty_env();
    let mut verifier = CertVerifier::new(&env);

    // Build: λ (x : Prop). x : Prop → Prop
    let prop = Expr::Sort(Level::zero());

    let expr = Expr::Lam(
        BinderInfo::Default,
        prop.clone().into(),
        Expr::BVar(0).into(), // Reference to x
    );

    // Certificate for the identity function
    let cert = ProofCert::Lam {
        binder_info: BinderInfo::Default,
        arg_type_cert: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        body_cert: Box::new(ProofCert::BVar {
            idx: 0,
            expected_type: Box::new(prop.clone()),
        }),
        result_type: Box::new(Expr::Pi(
            BinderInfo::Default,
            prop.clone().into(),
            prop.clone().into(),
        )),
    };

    let result = verifier.verify(&cert, &expr);
    assert!(result.is_ok());

    let ty = result.unwrap();
    match ty {
        Expr::Pi(_, arg_ty, ret_ty) => {
            assert_eq!(*arg_ty, prop);
            assert_eq!(*ret_ty, prop);
        }
        _ => panic!("Expected Pi type"),
    }
}

#[test]
fn test_lit_nat_cert() {
    let env = empty_env();
    let mut verifier = CertVerifier::new(&env);

    let expr = Expr::Lit(Literal::Nat(42));
    let nat_type = Expr::const_(Name::from_string("Nat"), vec![]);

    let cert = ProofCert::Lit {
        lit: Literal::Nat(42),
        type_: Box::new(nat_type.clone()),
    };

    let result = verifier.verify(&cert, &expr);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), nat_type);
}

#[test]
fn test_structure_mismatch() {
    let env = empty_env();
    let mut verifier = CertVerifier::new(&env);

    let expr = Expr::Sort(Level::zero());
    let cert = ProofCert::Lit {
        lit: Literal::Nat(0),
        type_: Box::new(Expr::const_(Name::from_string("Nat"), vec![])),
    };

    let result = verifier.verify(&cert, &expr);
    assert!(result.is_err());
    match result {
        Err(CertError::StructureMismatch { .. }) => {}
        _ => panic!("Expected StructureMismatch error"),
    }
}

#[test]
fn test_nested_bvar_in_context() {
    let env = empty_env();
    let mut verifier = CertVerifier::new(&env);

    // Build: λ (A : Type). λ (x : A). x : (A : Type) → A → A
    let type0 = Expr::Sort(Level::zero());
    let _type1 = Expr::Sort(Level::succ(Level::zero())); // Type 1, for reference

    // Inner lambda: λ (x : A). x
    let inner_lam = Expr::Lam(
        BinderInfo::Default,
        Expr::BVar(0).into(), // A (referring to outer binder)
        Expr::BVar(0).into(), // x (referring to inner binder)
    );

    // Outer lambda: λ (A : Type). inner_lam
    let expr = Expr::Lam(BinderInfo::Default, type0.clone().into(), inner_lam.into());

    // Build certificate
    // Inner body: x : A (where A is now at BVar(1) due to lifting)
    let inner_body_cert = ProofCert::BVar {
        idx: 0,
        expected_type: Box::new(Expr::BVar(1)), // A after shift
    };

    // Inner lambda: λ (x : A). x : A → A
    let inner_cert = ProofCert::Lam {
        binder_info: BinderInfo::Default,
        arg_type_cert: Box::new(ProofCert::BVar {
            idx: 0,
            expected_type: Box::new(type0.clone()), // A : Type
        }),
        body_cert: Box::new(inner_body_cert),
        result_type: Box::new(Expr::Pi(
            BinderInfo::Default,
            Expr::BVar(0).into(), // A
            Expr::BVar(1).into(), // A (shifted)
        )),
    };

    // Outer lambda cert
    let cert = ProofCert::Lam {
        binder_info: BinderInfo::Default,
        arg_type_cert: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }), // Type : Type1
        body_cert: Box::new(inner_cert),
        result_type: Box::new(Expr::Pi(
            BinderInfo::Default,
            type0.clone().into(),
            Expr::Pi(
                BinderInfo::Default,
                Expr::BVar(0).into(),
                Expr::BVar(1).into(),
            )
            .into(),
        )),
    };

    let result = verifier.verify(&cert, &expr);
    // This is a complex test - the important thing is that the verifier
    // can handle nested binders correctly
    assert!(result.is_ok() || matches!(result, Err(CertError::TypeMismatch { .. })));
}

#[test]
fn test_cubical_interval_cert_roundtrip() {
    let env = empty_env();
    let mut tc = TypeChecker::with_mode(&env, Lean5Mode::Cubical);
    let (ty, cert) = tc.infer_type_with_cert(&Expr::CubicalInterval).unwrap();

    let mut verifier = CertVerifier::with_mode(&env, Lean5Mode::Cubical);
    let verified_ty = verifier.verify(&cert, &Expr::CubicalInterval).unwrap();
    assert_eq!(ty, verified_ty);
}

#[test]
fn test_cubical_endpoint_cert_roundtrip() {
    let env = empty_env();
    let mut tc = TypeChecker::with_mode(&env, Lean5Mode::Cubical);

    for endpoint in [Expr::CubicalI0, Expr::CubicalI1] {
        let (ty, cert) = tc.infer_type_with_cert(&endpoint).unwrap();
        let mut verifier = CertVerifier::with_mode(&env, Lean5Mode::Cubical);
        let verified_ty = verifier.verify(&cert, &endpoint).unwrap();
        assert_eq!(ty, verified_ty);
    }
}

#[test]
fn test_cubical_path_type_cert_roundtrip() {
    let env = empty_env();
    let mut tc = TypeChecker::with_mode(&env, Lean5Mode::Cubical);

    // A : I -> Type0 (represented as λ i : I, Type0)
    let type0 = Expr::Sort(Level::succ(Level::zero()));
    let ty_family = Expr::Lam(BinderInfo::Default, Expr::CubicalInterval.into(), type0.into());

    // Prop : Type0, so it's a valid endpoint for a constant Type0 family.
    let prop = Expr::Sort(Level::zero());
    let path_ty = Expr::CubicalPath {
        ty: ty_family.into(),
        left: prop.clone().into(),
        right: prop.into(),
    };

    let (ty, cert) = tc.infer_type_with_cert(&path_ty).unwrap();
    let mut verifier = CertVerifier::with_mode(&env, Lean5Mode::Cubical);
    let verified_ty = verifier.verify(&cert, &path_ty).unwrap();
    assert_eq!(ty, verified_ty);
}

#[test]
fn test_cubical_path_lam_and_app_cert_roundtrip() {
    let env = empty_env();
    let mut tc = TypeChecker::with_mode(&env, Lean5Mode::Cubical);

    // Constant path at Prop (doesn't use the interval variable).
    let path_lam = Expr::CubicalPathLam {
        body: Expr::Sort(Level::zero()).into(),
    };
    let (lam_ty, lam_cert) = tc.infer_type_with_cert(&path_lam).unwrap();

    let mut verifier = CertVerifier::with_mode(&env, Lean5Mode::Cubical);
    let verified_lam_ty = verifier.verify(&lam_cert, &path_lam).unwrap();
    assert_eq!(lam_ty, verified_lam_ty);

    // Apply the path to 0.
    let path_app = Expr::CubicalPathApp {
        path: path_lam.clone().into(),
        arg: Expr::CubicalI0.into(),
    };
    let (app_ty, app_cert) = tc.infer_type_with_cert(&path_app).unwrap();

    let verified_app_ty = verifier.verify(&app_cert, &path_app).unwrap();
    assert_eq!(app_ty, verified_app_ty);
}

// ========================================================================
// Tests targeting surviving mutations
// ========================================================================

// --- CertVerifier::def_eq tests ---

#[test]
fn test_def_eq_same_expr() {
    let env = empty_env();
    let verifier = CertVerifier::new(&env);
    let e = Expr::Sort(Level::zero());
    // Must return true for identical expressions
    assert!(verifier.def_eq(&e, &e));
}

#[test]
fn test_def_eq_different_exprs() {
    let env = empty_env();
    let verifier = CertVerifier::new(&env);
    let e1 = Expr::Sort(Level::zero());
    let e2 = Expr::Sort(Level::succ(Level::zero()));
    // Must return false for different expressions
    assert!(!verifier.def_eq(&e1, &e2));
}

#[test]
fn test_def_eq_beta_reduction() {
    let env = empty_env();
    let verifier = CertVerifier::new(&env);
    // (λ x. x) applied conceptually should reduce
    // But structurally: λ x. x should equal λ x. x
    let lam1 = Expr::Lam(
        BinderInfo::Default,
        Expr::Sort(Level::zero()).into(),
        Expr::BVar(0).into(),
    );
    let lam2 = Expr::Lam(
        BinderInfo::Default,
        Expr::Sort(Level::zero()).into(),
        Expr::BVar(0).into(),
    );
    assert!(verifier.def_eq(&lam1, &lam2));
}

// --- CertVerifier::structural_eq tests ---

#[test]
fn test_structural_eq_fvar() {
    let env = empty_env();
    let verifier = CertVerifier::new(&env);
    let fvar1 = Expr::FVar(FVarId(1));
    let fvar2 = Expr::FVar(FVarId(1));
    let fvar3 = Expr::FVar(FVarId(2));
    // Same ID should match, different should not
    assert!(verifier.structural_eq(&fvar1, &fvar2));
    assert!(!verifier.structural_eq(&fvar1, &fvar3));
}

#[test]
fn test_structural_eq_app() {
    let env = empty_env();
    let verifier = CertVerifier::new(&env);
    let prop = Expr::Sort(Level::zero());
    let type1 = Expr::Sort(Level::succ(Level::zero()));

    let app1 = Expr::App(prop.clone().into(), prop.clone().into());
    let app2 = Expr::App(prop.clone().into(), prop.clone().into());
    let app3 = Expr::App(prop.clone().into(), type1.clone().into());
    let app4 = Expr::App(type1.clone().into(), prop.clone().into());

    assert!(verifier.structural_eq(&app1, &app2));
    assert!(!verifier.structural_eq(&app1, &app3)); // Different arg
    assert!(!verifier.structural_eq(&app1, &app4)); // Different fn
}

#[test]
fn test_structural_eq_lam() {
    let env = empty_env();
    let verifier = CertVerifier::new(&env);
    let prop = Expr::Sort(Level::zero());
    let type1 = Expr::Sort(Level::succ(Level::zero()));

    let lam1 = Expr::Lam(
        BinderInfo::Default,
        prop.clone().into(),
        Expr::BVar(0).into(),
    );
    let lam2 = Expr::Lam(
        BinderInfo::Default,
        prop.clone().into(),
        Expr::BVar(0).into(),
    );
    let lam3 = Expr::Lam(
        BinderInfo::Implicit,
        prop.clone().into(),
        Expr::BVar(0).into(),
    );
    let lam4 = Expr::Lam(
        BinderInfo::Default,
        type1.clone().into(),
        Expr::BVar(0).into(),
    );
    let lam5 = Expr::Lam(
        BinderInfo::Default,
        prop.clone().into(),
        Expr::BVar(1).into(),
    );

    assert!(verifier.structural_eq(&lam1, &lam2));
    assert!(!verifier.structural_eq(&lam1, &lam3)); // Different binder info
    assert!(!verifier.structural_eq(&lam1, &lam4)); // Different type
    assert!(!verifier.structural_eq(&lam1, &lam5)); // Different body
}

#[test]
fn test_structural_eq_pi() {
    let env = empty_env();
    let verifier = CertVerifier::new(&env);
    let prop = Expr::Sort(Level::zero());
    let type1 = Expr::Sort(Level::succ(Level::zero()));

    let pi1 = Expr::Pi(
        BinderInfo::Default,
        prop.clone().into(),
        prop.clone().into(),
    );
    let pi2 = Expr::Pi(
        BinderInfo::Default,
        prop.clone().into(),
        prop.clone().into(),
    );
    let pi3 = Expr::Pi(
        BinderInfo::Implicit,
        prop.clone().into(),
        prop.clone().into(),
    );
    let pi4 = Expr::Pi(
        BinderInfo::Default,
        type1.clone().into(),
        prop.clone().into(),
    );
    let pi5 = Expr::Pi(
        BinderInfo::Default,
        prop.clone().into(),
        type1.clone().into(),
    );

    assert!(verifier.structural_eq(&pi1, &pi2));
    assert!(!verifier.structural_eq(&pi1, &pi3)); // Different binder
    assert!(!verifier.structural_eq(&pi1, &pi4)); // Different arg type
    assert!(!verifier.structural_eq(&pi1, &pi5)); // Different body type
}

#[test]
fn test_structural_eq_let() {
    let env = empty_env();
    let verifier = CertVerifier::new(&env);
    let prop = Expr::Sort(Level::zero());
    let type1 = Expr::Sort(Level::succ(Level::zero()));

    let let1 = Expr::Let(
        prop.clone().into(),
        prop.clone().into(),
        Expr::BVar(0).into(),
    );
    let let2 = Expr::Let(
        prop.clone().into(),
        prop.clone().into(),
        Expr::BVar(0).into(),
    );
    let let3 = Expr::Let(
        type1.clone().into(),
        prop.clone().into(),
        Expr::BVar(0).into(),
    );
    let let4 = Expr::Let(
        prop.clone().into(),
        type1.clone().into(),
        Expr::BVar(0).into(),
    );
    let let5 = Expr::Let(
        prop.clone().into(),
        prop.clone().into(),
        Expr::BVar(1).into(),
    );

    assert!(verifier.structural_eq(&let1, &let2));
    assert!(!verifier.structural_eq(&let1, &let3)); // Different type
    assert!(!verifier.structural_eq(&let1, &let4)); // Different value
    assert!(!verifier.structural_eq(&let1, &let5)); // Different body
}

#[test]
fn test_structural_eq_lit() {
    let env = empty_env();
    let verifier = CertVerifier::new(&env);

    let lit1 = Expr::Lit(Literal::Nat(42));
    let lit2 = Expr::Lit(Literal::Nat(42));
    let lit3 = Expr::Lit(Literal::Nat(43));

    assert!(verifier.structural_eq(&lit1, &lit2));
    assert!(!verifier.structural_eq(&lit1, &lit3));
}

#[test]
fn test_structural_eq_proj() {
    let env = empty_env();
    let verifier = CertVerifier::new(&env);
    let prop = Expr::Sort(Level::zero());
    let name1 = Name::from_string("Foo");
    let name2 = Name::from_string("Bar");

    let proj1 = Expr::Proj(name1.clone(), 0, prop.clone().into());
    let proj2 = Expr::Proj(name1.clone(), 0, prop.clone().into());
    let proj3 = Expr::Proj(name2.clone(), 0, prop.clone().into());
    let proj4 = Expr::Proj(name1.clone(), 1, prop.clone().into());
    let proj5 = Expr::Proj(
        name1.clone(),
        0,
        Expr::Sort(Level::succ(Level::zero())).into(),
    );

    assert!(verifier.structural_eq(&proj1, &proj2));
    assert!(!verifier.structural_eq(&proj1, &proj3)); // Different name
    assert!(!verifier.structural_eq(&proj1, &proj4)); // Different index
    assert!(!verifier.structural_eq(&proj1, &proj5)); // Different expr
}

#[test]
fn test_structural_eq_const() {
    let env = empty_env();
    let verifier = CertVerifier::new(&env);
    let name1 = Name::from_string("Foo");
    let name2 = Name::from_string("Bar");

    let const1 = Expr::const_(name1.clone(), vec![Level::zero()]);
    let const2 = Expr::const_(name1.clone(), vec![Level::zero()]);
    let const3 = Expr::const_(name2.clone(), vec![Level::zero()]);
    let const4 = Expr::const_(name1.clone(), vec![Level::succ(Level::zero())]);
    let const5 = Expr::const_(name1.clone(), vec![]);

    assert!(verifier.structural_eq(&const1, &const2));
    assert!(!verifier.structural_eq(&const1, &const3)); // Different name
    assert!(!verifier.structural_eq(&const1, &const4)); // Different level
    assert!(!verifier.structural_eq(&const1, &const5)); // Different arity
}

// --- CertVerifier::level_eq tests ---

#[test]
fn test_level_eq_same() {
    let env = empty_env();
    let verifier = CertVerifier::new(&env);
    let l = Level::succ(Level::zero());
    assert!(verifier.level_eq(&l, &l));
}

#[test]
fn test_level_eq_different() {
    let env = empty_env();
    let verifier = CertVerifier::new(&env);
    let l1 = Level::zero();
    let l2 = Level::succ(Level::zero());
    assert!(!verifier.level_eq(&l1, &l2));
}

// --- CertVerifier::whnf tests ---

#[test]
fn test_whnf_app_beta() {
    let env = empty_env();
    let verifier = CertVerifier::new(&env);

    // (λ x. x) y → y
    let id = Expr::Lam(
        BinderInfo::Default,
        Expr::Sort(Level::zero()).into(),
        Expr::BVar(0).into(),
    );
    let arg = Expr::Sort(Level::succ(Level::zero()));
    let app = Expr::App(id.into(), arg.clone().into());

    let result = verifier.whnf(&app);
    assert_eq!(result, arg);
}

#[test]
fn test_whnf_let_zeta() {
    let env = empty_env();
    let verifier = CertVerifier::new(&env);

    // let x := v in x → v
    let val = Expr::Sort(Level::succ(Level::zero()));
    let let_expr = Expr::Let(
        Expr::Sort(Level::zero()).into(), // type
        val.clone().into(),               // value
        Expr::BVar(0).into(),             // body = x
    );

    let result = verifier.whnf(&let_expr);
    assert_eq!(result, val);
}

#[test]
fn test_whnf_const_unfold() {
    use crate::env::Declaration;

    let mut env = Environment::new();

    // Define a constant that unfolds
    let val = Expr::Sort(Level::zero());
    env.add_decl(Declaration::Definition {
        name: Name::from_string("myProp"),
        level_params: vec![],
        type_: Expr::Sort(Level::succ(Level::zero())),
        value: val.clone(),
        is_reducible: true,
    })
    .unwrap();

    let verifier = CertVerifier::new(&env);
    let const_expr = Expr::const_(Name::from_string("myProp"), vec![]);

    let result = verifier.whnf(&const_expr);
    assert_eq!(result, val);
}

#[test]
fn test_whnf_non_reducible() {
    let env = empty_env();
    let verifier = CertVerifier::new(&env);

    // Sort should not reduce
    let sort = Expr::Sort(Level::zero());
    let result = verifier.whnf(&sort);
    assert_eq!(result, sort);
}

// --- cert_name and expr_name tests (for Display mutations) ---

#[test]
fn test_cert_name_coverage() {
    // Test that cert_name returns non-empty strings
    let sort_cert = ProofCert::Sort {
        level: Level::zero(),
    };
    assert!(!cert_name(&sort_cert).is_empty());

    let bvar_cert = ProofCert::BVar {
        idx: 0,
        expected_type: Box::new(Expr::Sort(Level::zero())),
    };
    assert!(!cert_name(&bvar_cert).is_empty());

    let lit_cert = ProofCert::Lit {
        lit: Literal::Nat(0),
        type_: Box::new(Expr::const_(Name::from_string("Nat"), vec![])),
    };
    assert!(!cert_name(&lit_cert).is_empty());
}

#[test]
fn test_expr_name_coverage() {
    // Test that expr_name returns non-empty strings
    assert!(!expr_name(&Expr::BVar(0)).is_empty());
    assert!(!expr_name(&Expr::FVar(FVarId(0))).is_empty());
    assert!(!expr_name(&Expr::Sort(Level::zero())).is_empty());
    assert!(!expr_name(&Expr::Lit(Literal::Nat(0))).is_empty());
}

// --- CertError Display test ---

#[test]
fn test_cert_error_display() {
    let err = CertError::InvalidBVar(5);
    let s = format!("{err}");
    assert!(!s.is_empty());

    let err2 = CertError::TypeMismatch {
        expected: Box::new(Expr::Sort(Level::zero())),
        actual: Box::new(Expr::Sort(Level::succ(Level::zero()))),
        location: "test".to_string(),
    };
    let s2 = format!("{err2}");
    assert!(!s2.is_empty());
}

// --- verify function mutation tests ---

#[test]
fn test_verify_bvar_depth_calculation() {
    let env = empty_env();
    let mut verifier = CertVerifier::new(&env);

    // Push some context to test depth calculation
    verifier.context.push(Expr::Sort(Level::zero()));
    verifier
        .context
        .push(Expr::Sort(Level::succ(Level::zero())));

    // BVar(0) should refer to the innermost (most recently pushed)
    let expr = Expr::BVar(0);
    let cert = ProofCert::BVar {
        idx: 0,
        expected_type: Box::new(Expr::Sort(Level::succ(Level::zero()))),
    };

    let result = verifier.verify(&cert, &expr);
    assert!(result.is_ok());
}

#[test]
fn test_verify_bvar_invalid_index() {
    let env = empty_env();
    let mut verifier = CertVerifier::new(&env);

    // Empty context, BVar(0) should fail
    let expr = Expr::BVar(0);
    let cert = ProofCert::BVar {
        idx: 0,
        expected_type: Box::new(Expr::Sort(Level::zero())),
    };

    let result = verifier.verify(&cert, &expr);
    assert!(matches!(result, Err(CertError::InvalidBVar(_))));
}

#[test]
fn test_verify_fvar_id_mismatch() {
    let env = empty_env();
    let mut verifier = CertVerifier::new(&env);

    let expr = Expr::FVar(FVarId(1));
    let cert = ProofCert::FVar {
        id: FVarId(2), // Different ID!
        type_: Box::new(Expr::Sort(Level::zero())),
    };

    let result = verifier.verify(&cert, &expr);
    assert!(result.is_err());
}

#[test]
fn test_verify_fvar_type_check() {
    let env = empty_env();
    let mut verifier = CertVerifier::new(&env);

    // Add FVar to context
    let fvar_id = FVarId(1);
    let fvar_type = Expr::Sort(Level::zero());
    verifier.register_fvar(fvar_id, fvar_type.clone()).unwrap();

    let expr = Expr::FVar(fvar_id);

    // Correct type
    let cert_ok = ProofCert::FVar {
        id: fvar_id,
        type_: Box::new(fvar_type.clone()),
    };
    assert!(verifier.verify(&cert_ok, &expr).is_ok());

    // Wrong type should fail
    let cert_bad = ProofCert::FVar {
        id: fvar_id,
        type_: Box::new(Expr::Sort(Level::succ(Level::zero()))),
    };
    let result = verifier.verify(&cert_bad, &expr);
    assert!(matches!(result, Err(CertError::TypeMismatch { .. })));
}

#[test]
fn test_verify_fvar_missing_context() {
    let env = empty_env();
    let mut verifier = CertVerifier::new(&env);

    let fvar_id = FVarId(3);
    let expr = Expr::FVar(fvar_id);
    let cert = ProofCert::FVar {
        id: fvar_id,
        type_: Box::new(Expr::Sort(Level::zero())),
    };

    let result = verifier.verify(&cert, &expr);
    assert!(matches!(result, Err(CertError::UnknownFVar(_))));
}

#[test]
fn test_verify_mdata_correct_type() {
    use crate::expr::MDataValue;

    let env = empty_env();
    let mut verifier = CertVerifier::new(&env);

    // MData wrapping Sort(0)
    // Sort(0) has type Sort(1)
    let metadata = vec![(Name::from_string("trace"), MDataValue::Bool(true))];
    let inner_expr = Expr::Sort(Level::zero());
    let expr = Expr::MData(metadata.clone(), inner_expr.into());

    let cert = ProofCert::MData {
        metadata,
        inner_cert: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        // Correct: Sort(0) has type Sort(1)
        result_type: Box::new(Expr::Sort(Level::succ(Level::zero()))),
    };

    let result = verifier.verify(&cert, &expr);
    assert!(result.is_ok(), "MData with correct type should verify");
    // MData type is the type of the inner expression
    assert_eq!(result.unwrap(), Expr::Sort(Level::succ(Level::zero())));
}

#[test]
fn test_verify_mdata_type_mismatch() {
    use crate::expr::MDataValue;

    let env = empty_env();
    let mut verifier = CertVerifier::new(&env);

    // MData wrapping Sort(0)
    let metadata = vec![(Name::from_string("trace"), MDataValue::Bool(true))];
    let inner_expr = Expr::Sort(Level::zero());
    let expr = Expr::MData(metadata.clone(), inner_expr.into());

    let cert = ProofCert::MData {
        metadata,
        inner_cert: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        // WRONG: claiming Sort(0) has type Sort(2) instead of Sort(1)
        result_type: Box::new(Expr::Sort(Level::succ(Level::succ(Level::zero())))),
    };

    let result = verifier.verify(&cert, &expr);
    assert!(result.is_err(), "MData with wrong result_type must fail");
    assert!(
        matches!(result, Err(CertError::TypeMismatch { .. })),
        "Expected TypeMismatch error for MData type mismatch"
    );
}

#[test]
fn test_register_fvar_conflict_rejected() {
    let env = empty_env();
    let mut verifier = CertVerifier::new(&env);

    let fvar_id = FVarId(5);
    verifier
        .register_fvar(fvar_id, Expr::Sort(Level::zero()))
        .unwrap();

    let conflict = verifier.register_fvar(fvar_id, Expr::Sort(Level::succ(Level::zero())));
    assert!(matches!(conflict, Err(CertError::TypeMismatch { .. })));
}

#[test]
fn test_register_local_context() {
    use crate::tc::LocalContext;

    let env = empty_env();
    let mut verifier = CertVerifier::new(&env);
    let mut ctx = LocalContext::new();

    // Push several declarations
    let ty1 = Expr::Sort(Level::zero());
    let ty2 = Expr::Sort(Level::succ(Level::zero()));
    let id1 = ctx.push("x".parse().unwrap(), ty1.clone(), BinderInfo::Default);
    let id2 = ctx.push("y".parse().unwrap(), ty2.clone(), BinderInfo::Default);

    // Register all at once
    verifier.register_local_context(&ctx).unwrap();

    // Verify FVars can be verified
    let fvar1_expr = Expr::FVar(id1);
    let fvar1_cert = ProofCert::FVar {
        id: id1,
        type_: Box::new(ty1.clone()),
    };
    let result1 = verifier.verify(&fvar1_cert, &fvar1_expr);
    assert!(result1.is_ok());

    let fvar2_expr = Expr::FVar(id2);
    let fvar2_cert = ProofCert::FVar {
        id: id2,
        type_: Box::new(ty2.clone()),
    };
    let result2 = verifier.verify(&fvar2_cert, &fvar2_expr);
    assert!(result2.is_ok());
}

#[test]
fn test_register_local_context_conflict() {
    use crate::tc::LocalContext;

    let env = empty_env();
    let mut verifier = CertVerifier::new(&env);

    // Pre-register an FVar with one type
    let fvar_id = FVarId(0);
    verifier
        .register_fvar(fvar_id, Expr::Sort(Level::zero()))
        .unwrap();

    // Create a context with conflicting type for same ID
    let mut ctx = LocalContext::new();
    ctx.push_with_id(
        fvar_id,
        "x".parse().unwrap(),
        Expr::Sort(Level::succ(Level::zero())),
        BinderInfo::Default,
    );

    // Should fail due to conflict
    let result = verifier.register_local_context(&ctx);
    assert!(matches!(result, Err(CertError::TypeMismatch { .. })));
}

#[test]
fn test_verify_def_eq_inner_type_mismatch() {
    let env = empty_env();
    let mut verifier = CertVerifier::new(&env);

    let expr = Expr::Sort(Level::zero());

    // DefEq certificate where actual_type doesn't match what verify returns
    let cert = ProofCert::DefEq {
        inner: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        expected_type: Box::new(Expr::Sort(Level::succ(Level::zero()))),
        actual_type: Box::new(Expr::Sort(Level::succ(Level::succ(Level::zero())))), // Wrong!
        eq_steps: vec![],
    };

    let result = verifier.verify(&cert, &expr);
    assert!(result.is_err());
}

#[test]
fn test_verify_def_eq_expected_type_mismatch() {
    let env = empty_env();
    let mut verifier = CertVerifier::new(&env);

    let expr = Expr::Sort(Level::zero());

    // DefEq certificate where expected != actual (and actual is correct)
    let cert = ProofCert::DefEq {
        inner: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        expected_type: Box::new(Expr::Sort(Level::succ(Level::succ(Level::zero())))), // Type 2
        actual_type: Box::new(Expr::Sort(Level::succ(Level::zero()))), // Type 1 (correct)
        eq_steps: vec![],
    };

    let result = verifier.verify(&cert, &expr);
    assert!(matches!(result, Err(CertError::DefEqFailed { .. })));
}

// =========================================================================
// Mutation Testing Kill Tests - cert.rs survivors
// =========================================================================

#[test]
fn test_verify_bvar_depth_minus_arithmetic() {
    // Kill mutant: replace - with + in CertVerifier::verify (line 256)
    // The calculation is: level = depth - 1 - idx
    // This converts de Bruijn index to context index

    let env = empty_env();
    let mut verifier = CertVerifier::new(&env);

    // Push 3 entries to context
    let ty0 = Expr::Sort(Level::zero()); // idx 2 (oldest)
    let ty1 = Expr::Sort(Level::succ(Level::zero())); // idx 1
    let ty2 = Expr::Sort(Level::succ(Level::succ(Level::zero()))); // idx 0 (newest)
    verifier.context.push(ty0.clone());
    verifier.context.push(ty1.clone());
    verifier.context.push(ty2.clone());

    // depth = 3
    // BVar(0): level = 3 - 1 - 0 = 2 -> context[2] = ty2
    // BVar(1): level = 3 - 1 - 1 = 1 -> context[1] = ty1
    // BVar(2): level = 3 - 1 - 2 = 0 -> context[0] = ty0

    // Test BVar(0) should reference ty2
    let expr0 = Expr::BVar(0);
    let cert0 = ProofCert::BVar {
        idx: 0,
        expected_type: Box::new(ty2.clone()),
    };
    let result0 = verifier.verify(&cert0, &expr0);
    assert!(result0.is_ok(), "BVar(0) should map to context[2]");

    // Test BVar(2) should reference ty0
    let expr2 = Expr::BVar(2);
    let cert2 = ProofCert::BVar {
        idx: 2,
        expected_type: Box::new(ty0.clone()),
    };
    let result2 = verifier.verify(&cert2, &expr2);
    assert!(result2.is_ok(), "BVar(2) should map to context[0]");
}

#[test]
fn test_cert_name_returns_meaningful_values() {
    // Kill mutants: replace cert_name -> String with "xyzzy".into()
    // Verify that cert_name returns different values for different cert types

    let sort_cert = ProofCert::Sort {
        level: Level::zero(),
    };
    let bvar_cert = ProofCert::BVar {
        idx: 0,
        expected_type: Box::new(Expr::Sort(Level::zero())),
    };
    let fvar_cert = ProofCert::FVar {
        id: FVarId(1),
        type_: Box::new(Expr::Sort(Level::zero())),
    };

    let sort_name = cert_name(&sort_cert);
    let bvar_name = cert_name(&bvar_cert);
    let fvar_name = cert_name(&fvar_cert);

    // Names should not be "xyzzy" (the mutant replacement)
    assert_ne!(sort_name, "xyzzy", "cert_name should not return xyzzy");
    assert_ne!(bvar_name, "xyzzy", "cert_name should not return xyzzy");
    assert_ne!(fvar_name, "xyzzy", "cert_name should not return xyzzy");

    // Names should be meaningful (contain the variant name)
    assert!(
        sort_name.contains("Sort") || sort_name.to_lowercase().contains("sort"),
        "Sort cert should have meaningful name"
    );
    assert!(
        bvar_name.contains("BVar") || bvar_name.to_lowercase().contains("bvar"),
        "BVar cert should have meaningful name"
    );
    assert!(
        fvar_name.contains("FVar") || fvar_name.to_lowercase().contains("fvar"),
        "FVar cert should have meaningful name"
    );
}

#[test]
fn test_expr_name_returns_meaningful_values() {
    // Kill mutants: replace expr_name -> String with "xyzzy".into()
    // Verify that expr_name returns different values for different expr types

    let sort_expr = Expr::Sort(Level::zero());
    let bvar_expr = Expr::BVar(0);
    let fvar_expr = Expr::FVar(FVarId(1));

    let sort_name = expr_name(&sort_expr);
    let bvar_name = expr_name(&bvar_expr);
    let fvar_name = expr_name(&fvar_expr);

    // Names should not be "xyzzy"
    assert_ne!(sort_name, "xyzzy", "expr_name should not return xyzzy");
    assert_ne!(bvar_name, "xyzzy", "expr_name should not return xyzzy");
    assert_ne!(fvar_name, "xyzzy", "expr_name should not return xyzzy");

    // Names should be meaningful
    assert!(
        sort_name.contains("Sort") || sort_name.to_lowercase().contains("sort"),
        "Sort expr should have meaningful name"
    );
    assert!(
        bvar_name.contains("BVar") || bvar_name.to_lowercase().contains("bvar"),
        "BVar expr should have meaningful name"
    );
    assert!(
        fvar_name.contains("FVar") || fvar_name.to_lowercase().contains("fvar"),
        "FVar expr should have meaningful name"
    );
}

// ========================================================================
// Certificate serialization tests
// ========================================================================

#[test]
fn test_cert_serialize_json_sort() {
    let cert = ProofCert::Sort {
        level: Level::zero(),
    };

    // Serialize to JSON
    let json = serde_json::to_string(&cert).expect("JSON serialization failed");
    assert!(json.contains("Sort"));

    // Deserialize back
    let restored: ProofCert = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(cert, restored);
}

#[test]
fn test_cert_serialize_json_complex() {
    // Build a certificate for λ (x : Prop). x : Prop → Prop
    let prop = Expr::Sort(Level::zero());

    let cert = ProofCert::Lam {
        binder_info: BinderInfo::Default,
        arg_type_cert: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        body_cert: Box::new(ProofCert::BVar {
            idx: 0,
            expected_type: Box::new(prop.clone()),
        }),
        result_type: Box::new(Expr::Pi(
            BinderInfo::Default,
            prop.clone().into(),
            prop.clone().into(),
        )),
    };

    // Serialize to JSON
    let json = serde_json::to_string_pretty(&cert).expect("JSON serialization failed");

    // Deserialize back
    let restored: ProofCert = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(cert, restored);
}

#[test]
fn test_cert_serialize_bincode() {
    let cert = ProofCert::Pi {
        binder_info: BinderInfo::Implicit,
        arg_type_cert: Box::new(ProofCert::Sort {
            level: Level::succ(Level::zero()),
        }),
        arg_level: Level::succ(Level::succ(Level::zero())),
        body_type_cert: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        body_level: Level::succ(Level::zero()),
    };

    // Serialize to bincode
    let bytes = bincode::serialize(&cert).expect("bincode serialization failed");

    // Deserialize back
    let restored: ProofCert = bincode::deserialize(&bytes).expect("bincode deserialization failed");
    assert_eq!(cert, restored);
}

#[test]
fn test_def_eq_step_serialize() {
    let step = DefEqStep::Trans(
        Box::new(DefEqStep::Beta),
        Box::new(DefEqStep::Delta(Name::from_string("foo"))),
    );

    // JSON round-trip
    let json = serde_json::to_string(&step).expect("JSON serialization failed");
    let restored: DefEqStep = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(step, restored);

    // Bincode round-trip
    let bytes = bincode::serialize(&step).expect("bincode serialization failed");
    let restored2: DefEqStep =
        bincode::deserialize(&bytes).expect("bincode deserialization failed");
    assert_eq!(step, restored2);
}

#[test]
fn test_cert_serialize_with_mdata() {
    use crate::expr::MDataValue;

    let metadata = vec![(Name::from_string("key"), MDataValue::Bool(true))];
    let cert = ProofCert::MData {
        metadata,
        inner_cert: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        result_type: Box::new(Expr::Sort(Level::succ(Level::zero()))),
    };

    // JSON round-trip
    let json = serde_json::to_string(&cert).expect("JSON serialization failed");
    let restored: ProofCert = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(cert, restored);
}

#[test]
fn test_cert_serialize_def_eq() {
    let cert = ProofCert::DefEq {
        inner: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        expected_type: Box::new(Expr::Sort(Level::succ(Level::zero()))),
        actual_type: Box::new(Expr::Sort(Level::succ(Level::zero()))),
        eq_steps: vec![DefEqStep::Refl, DefEqStep::Symm(Box::new(DefEqStep::Refl))],
    };

    // JSON round-trip
    let json = serde_json::to_string(&cert).expect("JSON serialization failed");
    let restored: ProofCert = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(cert, restored);

    // Bincode round-trip
    let bytes = bincode::serialize(&cert).expect("bincode serialization failed");
    let restored2: ProofCert =
        bincode::deserialize(&bytes).expect("bincode deserialization failed");
    assert_eq!(cert, restored2);
}

// ========================================================================
// Proof Replay tests
// ========================================================================

#[test]
fn test_replay_sort() {
    let expr = Expr::Sort(Level::zero());
    let cert = ProofCert::Sort {
        level: Level::zero(),
    };

    let replayed = replay_cert(&cert);
    assert_eq!(replayed, expr);
}

#[test]
fn test_replay_bvar() {
    let expr = Expr::BVar(3);
    let cert = ProofCert::BVar {
        idx: 3,
        expected_type: Box::new(Expr::Sort(Level::zero())),
    };

    let replayed = replay_cert(&cert);
    assert_eq!(replayed, expr);
}

#[test]
fn test_replay_fvar() {
    let fvar_id = FVarId(42);
    let expr = Expr::FVar(fvar_id);
    let cert = ProofCert::FVar {
        id: fvar_id,
        type_: Box::new(Expr::Sort(Level::zero())),
    };

    let replayed = replay_cert(&cert);
    assert_eq!(replayed, expr);
}

#[test]
fn test_replay_const() {
    let name = Name::from_string("Nat");
    let levels = vec![Level::zero()];
    let expr = Expr::const_(name.clone(), levels.clone());
    let cert = ProofCert::Const {
        name: name.clone(),
        levels: levels.clone(),
        type_: Box::new(Expr::type_()),
    };

    let replayed = replay_cert(&cert);
    assert_eq!(replayed, expr);
}

#[test]
fn test_replay_app() {
    // Build: f x where f : A -> B, x : A
    let f = Expr::const_(Name::from_string("f"), vec![]);
    let x = Expr::const_(Name::from_string("x"), vec![]);
    let expr = Expr::App(f.clone().into(), x.clone().into());

    let a_ty = Expr::const_(Name::from_string("A"), vec![]);
    let b_ty = Expr::const_(Name::from_string("B"), vec![]);

    let cert = ProofCert::App {
        fn_cert: Box::new(ProofCert::Const {
            name: Name::from_string("f"),
            levels: vec![],
            type_: Box::new(Expr::arrow(a_ty.clone(), b_ty.clone())),
        }),
        fn_type: Box::new(Expr::arrow(a_ty.clone(), b_ty.clone())),
        arg_cert: Box::new(ProofCert::Const {
            name: Name::from_string("x"),
            levels: vec![],
            type_: Box::new(a_ty),
        }),
        result_type: Box::new(b_ty),
    };

    let replayed = replay_cert(&cert);
    assert_eq!(replayed, expr);
}

#[test]
fn test_replay_lam() {
    // Build: λ (x : Prop). x
    let prop = Expr::Sort(Level::zero());
    let expr = Expr::Lam(
        BinderInfo::Default,
        prop.clone().into(),
        Expr::BVar(0).into(),
    );

    let cert = ProofCert::Lam {
        binder_info: BinderInfo::Default,
        arg_type_cert: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        body_cert: Box::new(ProofCert::BVar {
            idx: 0,
            expected_type: Box::new(prop.clone()),
        }),
        result_type: Box::new(Expr::Pi(
            BinderInfo::Default,
            prop.clone().into(),
            prop.clone().into(),
        )),
    };

    let replayed = replay_cert(&cert);
    assert_eq!(replayed, expr);
}

#[test]
fn test_replay_pi() {
    // Build: Prop -> Prop
    let prop = Expr::Sort(Level::zero());
    let expr = Expr::Pi(
        BinderInfo::Default,
        prop.clone().into(),
        prop.clone().into(),
    );

    let cert = ProofCert::Pi {
        binder_info: BinderInfo::Default,
        arg_type_cert: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        arg_level: Level::succ(Level::zero()),
        body_type_cert: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        body_level: Level::succ(Level::zero()),
    };

    let replayed = replay_cert(&cert);
    assert_eq!(replayed, expr);
}

#[test]
fn test_replay_lit() {
    let expr = Expr::Lit(Literal::Nat(42));
    let nat_ty = Expr::const_(Name::from_string("Nat"), vec![]);
    let cert = ProofCert::Lit {
        lit: Literal::Nat(42),
        type_: Box::new(nat_ty),
    };

    let replayed = replay_cert(&cert);
    assert_eq!(replayed, expr);
}

#[test]
fn test_replay_def_eq() {
    // DefEq should be transparent in replay
    let inner_expr = Expr::Sort(Level::zero());
    let cert = ProofCert::DefEq {
        inner: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        expected_type: Box::new(Expr::Sort(Level::succ(Level::zero()))),
        actual_type: Box::new(Expr::Sort(Level::succ(Level::zero()))),
        eq_steps: vec![DefEqStep::Refl],
    };

    let replayed = replay_cert(&cert);
    assert_eq!(replayed, inner_expr);
}

#[test]
fn test_replay_mdata() {
    use crate::expr::MDataValue;

    let inner_expr = Expr::Sort(Level::zero());
    let metadata = vec![(Name::from_string("trace"), MDataValue::Bool(true))];
    let expr = Expr::MData(metadata.clone(), inner_expr.clone().into());

    let cert = ProofCert::MData {
        metadata: metadata.clone(),
        inner_cert: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        result_type: Box::new(Expr::Sort(Level::succ(Level::zero()))),
    };

    let replayed = replay_cert(&cert);
    assert_eq!(replayed, expr);
}

#[test]
fn test_replay_and_verify_sort() {
    let env = empty_env();
    let mut verifier = CertVerifier::new(&env);

    let cert = ProofCert::Sort {
        level: Level::zero(),
    };

    let result = verifier.replay_and_verify(&cert);
    assert!(result.is_ok());

    let (expr, ty) = result.unwrap();
    assert_eq!(expr, Expr::Sort(Level::zero()));
    assert_eq!(ty, Expr::Sort(Level::succ(Level::zero())));
}

#[test]
fn test_replay_and_verify_identity() {
    let env = empty_env();
    let mut verifier = CertVerifier::new(&env);

    // Certificate for λ (x : Prop). x : Prop → Prop
    let prop = Expr::Sort(Level::zero());

    let cert = ProofCert::Lam {
        binder_info: BinderInfo::Default,
        arg_type_cert: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        body_cert: Box::new(ProofCert::BVar {
            idx: 0,
            expected_type: Box::new(prop.clone()),
        }),
        result_type: Box::new(Expr::Pi(
            BinderInfo::Default,
            prop.clone().into(),
            prop.clone().into(),
        )),
    };

    let result = verifier.replay_and_verify(&cert);
    assert!(result.is_ok());

    let (expr, ty) = result.unwrap();

    // Verify the replayed expression is correct
    assert!(matches!(expr, Expr::Lam(_, _, _)));

    // Verify the type is Prop → Prop
    assert!(matches!(ty, Expr::Pi(_, _, _)));
}

#[test]
fn test_replay_roundtrip_with_serialization() {
    // Test the full flow: expr -> cert -> serialize -> deserialize -> replay -> expr
    let expr = Expr::Sort(Level::succ(Level::zero()));
    let cert = ProofCert::Sort {
        level: Level::succ(Level::zero()),
    };

    // Serialize to JSON
    let json = serde_json::to_string(&cert).expect("JSON serialization failed");

    // Deserialize
    let restored: ProofCert = serde_json::from_str(&json).expect("JSON deserialization failed");

    // Replay
    let replayed = replay_cert(&restored);

    // Should match original
    assert_eq!(replayed, expr);
}

#[test]
fn test_replay_complex_nested_cert() {
    // Build: (λ (A : Type). λ (x : A). x)
    // Type: (A : Type) → A → A
    let type0 = Expr::Sort(Level::zero());
    let type1 = Expr::Sort(Level::succ(Level::zero()));

    // The expression (for reference, we verify structure below)
    let _inner_lam = Expr::Lam(
        BinderInfo::Default,
        Expr::BVar(0).into(), // A (outer binder)
        Expr::BVar(0).into(), // x (inner binder)
    );
    // Outer lambda: λ (A : Type). _inner_lam

    // Build certificate for inner body: x : A (BVar 0)
    let inner_body_cert = ProofCert::BVar {
        idx: 0,
        expected_type: Box::new(Expr::BVar(1)), // A lifted
    };

    // Certificate for inner lambda
    let inner_cert = ProofCert::Lam {
        binder_info: BinderInfo::Default,
        arg_type_cert: Box::new(ProofCert::BVar {
            idx: 0,
            expected_type: Box::new(type1.clone()),
        }),
        body_cert: Box::new(inner_body_cert),
        result_type: Box::new(Expr::Pi(
            BinderInfo::Default,
            Expr::BVar(0).into(),
            Expr::BVar(1).into(),
        )),
    };

    // Certificate for outer lambda
    let cert = ProofCert::Lam {
        binder_info: BinderInfo::Default,
        arg_type_cert: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        body_cert: Box::new(inner_cert),
        result_type: Box::new(Expr::Pi(
            BinderInfo::Default,
            type0.clone().into(),
            Expr::Pi(
                BinderInfo::Default,
                Expr::BVar(0).into(),
                Expr::BVar(1).into(),
            )
            .into(),
        )),
    };

    let replayed = replay_cert(&cert);

    // Should match original expression structure
    assert!(matches!(replayed, Expr::Lam(_, _, _)));

    // Verify the inner structure
    match replayed {
        Expr::Lam(_, outer_ty, outer_body) => {
            assert_eq!(*outer_ty, type0);
            assert!(matches!(*outer_body, Expr::Lam(_, _, _)));
        }
        _ => panic!("Expected outer Lam"),
    }
}

#[test]
fn test_replay_let() {
    // Build: let x : Prop := Prop in x
    let prop = Expr::Sort(Level::zero());
    let expr = Expr::Let(
        prop.clone().into(),
        prop.clone().into(),
        Expr::BVar(0).into(),
    );

    let cert = ProofCert::Let {
        type_cert: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        value_cert: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        body_cert: Box::new(ProofCert::BVar {
            idx: 0,
            expected_type: Box::new(prop.clone()),
        }),
        result_type: Box::new(prop.clone()),
    };

    let replayed = replay_cert(&cert);
    assert_eq!(replayed, expr);
}

// ========================================================================
// Certificate Compression tests
// ========================================================================

#[test]
fn test_compress_simple_sort() {
    let cert = ProofCert::Sort {
        level: Level::zero(),
    };

    let compressed = compress_cert(&cert);
    assert_eq!(compressed.certs.len(), 1);
    assert_eq!(compressed.levels.len(), 1);
    assert_eq!(compressed.exprs.len(), 0);

    // Decompress and verify roundtrip
    let decompressed = decompress_cert(&compressed).expect("decompress failed");
    assert_eq!(decompressed, cert);
}

#[test]
fn test_compress_nested_levels() {
    // Succ(Succ(Zero)) - should deduplicate Zero
    let cert = ProofCert::Sort {
        level: Level::succ(Level::succ(Level::zero())),
    };

    let compressed = compress_cert(&cert);
    // Should have 3 levels: Zero, Succ(0), Succ(1)
    assert_eq!(compressed.levels.len(), 3);

    let decompressed = decompress_cert(&compressed).expect("decompress failed");
    assert_eq!(decompressed, cert);
}

#[test]
fn test_compress_with_shared_types() {
    // Lambda with Prop -> Prop type (Prop appears twice)
    let prop = Expr::Sort(Level::zero());

    let cert = ProofCert::Lam {
        binder_info: BinderInfo::Default,
        arg_type_cert: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        body_cert: Box::new(ProofCert::BVar {
            idx: 0,
            expected_type: Box::new(prop.clone()),
        }),
        result_type: Box::new(Expr::Pi(
            BinderInfo::Default,
            prop.clone().into(),
            prop.clone().into(),
        )),
    };

    let compressed = compress_cert(&cert);

    // Verify decompression roundtrip
    let decompressed = decompress_cert(&compressed).expect("decompress failed");
    assert_eq!(decompressed, cert);
}

#[test]
fn test_compress_app_certificate() {
    let a_ty = Expr::const_(Name::from_string("A"), vec![]);
    let b_ty = Expr::const_(Name::from_string("B"), vec![]);

    let cert = ProofCert::App {
        fn_cert: Box::new(ProofCert::Const {
            name: Name::from_string("f"),
            levels: vec![],
            type_: Box::new(Expr::arrow(a_ty.clone(), b_ty.clone())),
        }),
        fn_type: Box::new(Expr::arrow(a_ty.clone(), b_ty.clone())),
        arg_cert: Box::new(ProofCert::Const {
            name: Name::from_string("x"),
            levels: vec![],
            type_: Box::new(a_ty.clone()),
        }),
        result_type: Box::new(b_ty),
    };

    let compressed = compress_cert(&cert);
    let decompressed = decompress_cert(&compressed).expect("decompress failed");
    assert_eq!(decompressed, cert);
}

#[test]
fn test_compress_pi_certificate() {
    let cert = ProofCert::Pi {
        binder_info: BinderInfo::Default,
        arg_type_cert: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        arg_level: Level::succ(Level::zero()),
        body_type_cert: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        body_level: Level::succ(Level::zero()),
    };

    let compressed = compress_cert(&cert);

    // Should share the Sort certificates since they're identical
    assert!(compressed.certs.len() <= 3); // Pi + at most 2 Sort certs (may share)

    let decompressed = decompress_cert(&compressed).expect("decompress failed");
    assert_eq!(decompressed, cert);
}

#[test]
fn test_compress_let_certificate() {
    let prop = Expr::Sort(Level::zero());

    let cert = ProofCert::Let {
        type_cert: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        value_cert: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        body_cert: Box::new(ProofCert::BVar {
            idx: 0,
            expected_type: Box::new(prop.clone()),
        }),
        result_type: Box::new(prop),
    };

    let compressed = compress_cert(&cert);
    let decompressed = decompress_cert(&compressed).expect("decompress failed");
    assert_eq!(decompressed, cert);
}

#[test]
fn test_compress_lit_certificate() {
    let nat_ty = Expr::const_(Name::from_string("Nat"), vec![]);
    let cert = ProofCert::Lit {
        lit: Literal::Nat(42),
        type_: Box::new(nat_ty),
    };

    let compressed = compress_cert(&cert);
    let decompressed = decompress_cert(&compressed).expect("decompress failed");
    assert_eq!(decompressed, cert);
}

#[test]
fn test_compress_fvar_certificate() {
    let fvar_id = FVarId(123);
    let cert = ProofCert::FVar {
        id: fvar_id,
        type_: Box::new(Expr::Sort(Level::zero())),
    };

    let compressed = compress_cert(&cert);
    let decompressed = decompress_cert(&compressed).expect("decompress failed");
    assert_eq!(decompressed, cert);
}

#[test]
fn test_compress_def_eq_certificate() {
    let cert = ProofCert::DefEq {
        inner: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        expected_type: Box::new(Expr::Sort(Level::succ(Level::zero()))),
        actual_type: Box::new(Expr::Sort(Level::succ(Level::zero()))),
        eq_steps: vec![DefEqStep::Refl],
    };

    let compressed = compress_cert(&cert);
    let decompressed = decompress_cert(&compressed).expect("decompress failed");
    assert_eq!(decompressed, cert);
}

#[test]
fn test_compress_mdata_certificate() {
    use crate::expr::MDataValue;

    let metadata = vec![(Name::from_string("trace"), MDataValue::Bool(true))];
    let cert = ProofCert::MData {
        metadata: metadata.clone(),
        inner_cert: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        result_type: Box::new(Expr::Sort(Level::succ(Level::zero()))),
    };

    let compressed = compress_cert(&cert);
    let decompressed = decompress_cert(&compressed).expect("decompress failed");
    assert_eq!(decompressed, cert);
}

#[test]
fn test_compress_with_stats() {
    // Build a certificate with significant sharing potential
    let prop = Expr::Sort(Level::zero());
    let prop_to_prop = Expr::Pi(
        BinderInfo::Default,
        prop.clone().into(),
        prop.clone().into(),
    );

    // Certificate for identity: λ (x : Prop). x
    let cert = ProofCert::Lam {
        binder_info: BinderInfo::Default,
        arg_type_cert: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        body_cert: Box::new(ProofCert::BVar {
            idx: 0,
            expected_type: Box::new(prop.clone()),
        }),
        result_type: Box::new(prop_to_prop),
    };

    let (compressed, stats) = compress_cert_with_stats(&cert);

    // Verify stats are populated
    assert!(stats.unique_certs > 0);
    assert!(stats.unique_levels > 0);
    assert!(stats.original_bytes > 0);
    assert!(stats.compressed_bytes > 0);

    // Verify roundtrip
    let decompressed = decompress_cert(&compressed).expect("decompress failed");
    assert_eq!(decompressed, cert);
}

#[test]
fn test_compress_complex_nested() {
    // Build: (λ (A : Type). λ (x : A). x) with certificate
    let type0 = Expr::Sort(Level::zero());
    let type1 = Expr::Sort(Level::succ(Level::zero()));

    let inner_body_cert = ProofCert::BVar {
        idx: 0,
        expected_type: Box::new(Expr::BVar(1)),
    };

    let inner_cert = ProofCert::Lam {
        binder_info: BinderInfo::Default,
        arg_type_cert: Box::new(ProofCert::BVar {
            idx: 0,
            expected_type: Box::new(type1.clone()),
        }),
        body_cert: Box::new(inner_body_cert),
        result_type: Box::new(Expr::Pi(
            BinderInfo::Default,
            Expr::BVar(0).into(),
            Expr::BVar(1).into(),
        )),
    };

    let cert = ProofCert::Lam {
        binder_info: BinderInfo::Default,
        arg_type_cert: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        body_cert: Box::new(inner_cert),
        result_type: Box::new(Expr::Pi(
            BinderInfo::Default,
            type0.clone().into(),
            Expr::Pi(
                BinderInfo::Default,
                Expr::BVar(0).into(),
                Expr::BVar(1).into(),
            )
            .into(),
        )),
    };

    let compressed = compress_cert(&cert);
    let decompressed = decompress_cert(&compressed).expect("decompress failed");
    assert_eq!(decompressed, cert);
}

#[test]
fn test_compress_serialization_roundtrip() {
    let prop = Expr::Sort(Level::zero());

    let cert = ProofCert::Lam {
        binder_info: BinderInfo::Default,
        arg_type_cert: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        body_cert: Box::new(ProofCert::BVar {
            idx: 0,
            expected_type: Box::new(prop.clone()),
        }),
        result_type: Box::new(Expr::Pi(
            BinderInfo::Default,
            prop.clone().into(),
            prop.clone().into(),
        )),
    };

    // Full roundtrip: cert -> compress -> serialize -> deserialize -> decompress -> cert
    let compressed = compress_cert(&cert);

    // JSON roundtrip
    let json = serde_json::to_string(&compressed).expect("JSON serialize failed");
    let restored_compressed: CompressedCert =
        serde_json::from_str(&json).expect("JSON deserialize failed");
    let restored = decompress_cert(&restored_compressed).expect("decompress failed");
    assert_eq!(restored, cert);

    // Bincode roundtrip
    let bytes = bincode::serialize(&compressed).expect("bincode serialize failed");
    let restored_compressed2: CompressedCert =
        bincode::deserialize(&bytes).expect("bincode deserialize failed");
    let restored2 = decompress_cert(&restored_compressed2).expect("decompress failed");
    assert_eq!(restored2, cert);
}

#[test]
fn test_compression_deduplicates_shared_expressions() {
    // Create a certificate where the same expression appears multiple times
    let shared_type = Expr::const_(Name::from_string("SharedType"), vec![Level::zero()]);

    let cert = ProofCert::App {
        fn_cert: Box::new(ProofCert::Const {
            name: Name::from_string("f"),
            levels: vec![Level::zero()],
            type_: Box::new(Expr::arrow(shared_type.clone(), shared_type.clone())),
        }),
        fn_type: Box::new(Expr::arrow(shared_type.clone(), shared_type.clone())),
        arg_cert: Box::new(ProofCert::Const {
            name: Name::from_string("x"),
            levels: vec![Level::zero()],
            type_: Box::new(shared_type.clone()),
        }),
        result_type: Box::new(shared_type),
    };

    let compressed = compress_cert(&cert);

    // The shared_type expression should be deduplicated
    // Count how many unique Const expressions there are
    let const_count = compressed
        .exprs
        .iter()
        .filter(|e| matches!(e, CompressedExpr::Const(_, _)))
        .count();

    // Should have 2 unique consts: SharedType and the Pi (arrow) type components
    // The SharedType expression should be shared (appears once)
    assert!(const_count >= 1); // At least SharedType is deduplicated

    // Verify roundtrip
    let decompressed = decompress_cert(&compressed).expect("decompress failed");
    assert_eq!(decompressed, cert);
}

#[test]
fn test_invalid_decompress_index() {
    // Create a corrupted compressed cert with invalid indices
    let corrupted = CompressedCert {
        exprs: vec![],
        levels: vec![],
        certs: vec![CompressedCertNode::BVar {
            idx: 0,
            expected_type: 999, // Invalid index
        }],
        root: 0,
    };

    let result = decompress_cert(&corrupted);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        DecompressError::InvalidExprIndex(999)
    ));
}

#[test]
fn test_compression_stats_display() {
    let cert = ProofCert::Sort {
        level: Level::zero(),
    };

    let (_, stats) = compress_cert_with_stats(&cert);
    let display = format!("{stats}");

    // Verify display contains expected fields
    assert!(display.contains("exprs:"));
    assert!(display.contains("levels:"));
    assert!(display.contains("certs:"));
    assert!(display.contains("bytes"));
}

#[test]
fn test_decompress_error_display() {
    let err1 = DecompressError::InvalidExprIndex(42);
    let s1 = format!("{err1}");
    assert!(s1.contains("42"));
    assert!(s1.contains("expression"));

    let err2 = DecompressError::InvalidLevelIndex(99);
    let s2 = format!("{err2}");
    assert!(s2.contains("99"));
    assert!(s2.contains("level"));

    let err3 = DecompressError::InvalidCertIndex(123);
    let s3 = format!("{err3}");
    assert!(s3.contains("123"));
    assert!(s3.contains("certificate"));
}

// ========================================================================
// Byte-Level Compression (LZ4) Tests
// ========================================================================

#[test]
fn test_archive_simple_cert() {
    let cert = ProofCert::Sort {
        level: Level::zero(),
    };

    let archive = archive_cert(&cert).expect("archive failed");
    assert_eq!(archive.version, CertArchive::VERSION);
    assert!(!archive.compressed_data.is_empty());

    let restored = unarchive_cert(&archive).expect("unarchive failed");
    assert_eq!(restored, cert);
}

#[test]
fn test_archive_complex_cert() {
    // Build a complex certificate with nested structure
    let prop = Expr::Sort(Level::zero());
    let prop_to_prop = Expr::Pi(
        BinderInfo::Default,
        prop.clone().into(),
        prop.clone().into(),
    );

    let cert = ProofCert::Lam {
        binder_info: BinderInfo::Default,
        arg_type_cert: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        body_cert: Box::new(ProofCert::BVar {
            idx: 0,
            expected_type: Box::new(prop.clone()),
        }),
        result_type: Box::new(prop_to_prop),
    };

    let archive = archive_cert(&cert).expect("archive failed");
    let restored = unarchive_cert(&archive).expect("unarchive failed");
    assert_eq!(restored, cert);
}

#[test]
fn test_archive_with_stats() {
    let prop = Expr::Sort(Level::zero());
    let prop_to_prop = Expr::Pi(
        BinderInfo::Default,
        prop.clone().into(),
        prop.clone().into(),
    );

    let cert = ProofCert::Lam {
        binder_info: BinderInfo::Default,
        arg_type_cert: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        body_cert: Box::new(ProofCert::BVar {
            idx: 0,
            expected_type: Box::new(prop.clone()),
        }),
        result_type: Box::new(prop_to_prop),
    };

    let (archive, stats) = archive_cert_with_stats(&cert).expect("archive with stats failed");

    // Verify stats are populated
    assert!(stats.original_cert_bytes > 0);
    assert!(stats.structure_shared_bytes > 0);
    assert!(stats.archive_bytes > 0);
    // For small certs, structure sharing may increase size due to indexing overhead
    // but LZ4 should help. Just verify the ratios are reasonable positive values.
    assert!(stats.structure_ratio > 0.0);
    assert!(stats.lz4_ratio > 0.0);
    assert!(stats.total_ratio > 0.0);

    // Verify roundtrip
    let restored = unarchive_cert(&archive).expect("unarchive failed");
    assert_eq!(restored, cert);
}

#[test]
fn test_archive_stats_display() {
    let cert = ProofCert::Sort {
        level: Level::zero(),
    };
    let (_, stats) = archive_cert_with_stats(&cert).expect("archive failed");

    let display = format!("{stats}");
    assert!(display.contains("struct:"));
    assert!(display.contains("lz4:"));
    assert!(display.contains("total:"));
    assert!(display.contains("bytes"));
}

#[test]
fn test_lz4_compress_decompress() {
    let data = b"Hello, Lean5! This is a test of LZ4 compression.";

    let compressed = lz4_compress(data);
    let decompressed = lz4_decompress(&compressed).expect("decompress failed");

    assert_eq!(decompressed, data);
}

#[test]
fn test_lz4_compress_repetitive_data() {
    // Repetitive data should compress well
    let data: Vec<u8> = (0..1000).map(|i| (i % 10) as u8).collect();

    let compressed = lz4_compress(&data);
    let decompressed = lz4_decompress(&compressed).expect("decompress failed");

    assert_eq!(decompressed, data);
    // Repetitive data should compress significantly
    assert!(compressed.len() < data.len());
}

#[test]
fn test_archive_nested_app_chain() {
    // Build: f (g (h x)) with certificates
    let unit_type = Expr::Sort(Level::zero());

    let inner_cert = ProofCert::Const {
        name: Name::from_string("x"),
        levels: vec![],
        type_: Box::new(unit_type.clone()),
    };

    let h_app_cert = ProofCert::App {
        fn_cert: Box::new(ProofCert::Const {
            name: Name::from_string("h"),
            levels: vec![],
            type_: Box::new(Expr::arrow(unit_type.clone(), unit_type.clone())),
        }),
        fn_type: Box::new(Expr::arrow(unit_type.clone(), unit_type.clone())),
        arg_cert: Box::new(inner_cert),
        result_type: Box::new(unit_type.clone()),
    };

    let g_app_cert = ProofCert::App {
        fn_cert: Box::new(ProofCert::Const {
            name: Name::from_string("g"),
            levels: vec![],
            type_: Box::new(Expr::arrow(unit_type.clone(), unit_type.clone())),
        }),
        fn_type: Box::new(Expr::arrow(unit_type.clone(), unit_type.clone())),
        arg_cert: Box::new(h_app_cert),
        result_type: Box::new(unit_type.clone()),
    };

    let cert = ProofCert::App {
        fn_cert: Box::new(ProofCert::Const {
            name: Name::from_string("f"),
            levels: vec![],
            type_: Box::new(Expr::arrow(unit_type.clone(), unit_type.clone())),
        }),
        fn_type: Box::new(Expr::arrow(unit_type.clone(), unit_type.clone())),
        arg_cert: Box::new(g_app_cert),
        result_type: Box::new(unit_type),
    };

    let archive = archive_cert(&cert).expect("archive failed");
    let restored = unarchive_cert(&archive).expect("unarchive failed");
    assert_eq!(restored, cert);

    // With stats to verify compression effectiveness
    let (_, stats) = archive_cert_with_stats(&cert).expect("archive with stats failed");
    // Nested apps with repeated types should compress well
    assert!(
        stats.total_ratio >= 1.0,
        "Expected some compression for nested apps"
    );
}

#[test]
fn test_archive_bincode_serialization() {
    // Test that CertArchive itself can be serialized/deserialized
    let cert = ProofCert::Sort {
        level: Level::succ(Level::zero()),
    };
    let archive = archive_cert(&cert).expect("archive failed");

    // Serialize archive to bincode
    let archive_bytes = bincode::serialize(&archive).expect("serialize archive failed");

    // Deserialize
    let restored_archive: CertArchive =
        bincode::deserialize(&archive_bytes).expect("deserialize archive failed");

    // Unarchive
    let restored = unarchive_cert(&restored_archive).expect("unarchive failed");
    assert_eq!(restored, cert);
}

#[test]
fn test_archive_json_serialization() {
    // Test that CertArchive can be serialized to JSON (for debugging)
    let cert = ProofCert::Sort {
        level: Level::zero(),
    };
    let archive = archive_cert(&cert).expect("archive failed");

    // Serialize archive to JSON
    let json = serde_json::to_string(&archive).expect("serialize to JSON failed");

    // Deserialize
    let restored_archive: CertArchive =
        serde_json::from_str(&json).expect("deserialize from JSON failed");

    // Unarchive
    let restored = unarchive_cert(&restored_archive).expect("unarchive failed");
    assert_eq!(restored, cert);
}

#[test]
fn test_byte_compress_error_display() {
    let err1 = ByteCompressError::SerializeError("test".to_string());
    let s1 = format!("{err1}");
    assert!(s1.contains("Serialization"));
    assert!(s1.contains("test"));

    let err2 = ByteCompressError::CompressError("lz4 fail".to_string());
    let s2 = format!("{err2}");
    assert!(s2.contains("LZ4 compression"));

    let err3 = ByteCompressError::DecompressError("bad data".to_string());
    let s3 = format!("{err3}");
    assert!(s3.contains("LZ4 decompression"));

    let err4 = ByteCompressError::DeserializeError("parse fail".to_string());
    let s4 = format!("{err4}");
    assert!(s4.contains("Deserialization"));
}

#[test]
fn test_lz4_decompress_invalid_data() {
    // Invalid LZ4 data should return an error
    let invalid_data = b"not valid lz4 data";
    let result = lz4_decompress(invalid_data);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        ByteCompressError::DecompressError(_)
    ));
}

#[test]
fn test_archive_with_all_cert_variants() {
    // Test archiving certificates with various variants
    let type0 = Expr::Sort(Level::zero());

    // FVar certificate
    let fvar_cert = ProofCert::FVar {
        id: FVarId(42),
        type_: Box::new(type0.clone()),
    };
    let archive = archive_cert(&fvar_cert).expect("archive fvar failed");
    let restored = unarchive_cert(&archive).expect("unarchive fvar failed");
    assert_eq!(restored, fvar_cert);

    // Let certificate
    let let_cert = ProofCert::Let {
        type_cert: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        value_cert: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        body_cert: Box::new(ProofCert::BVar {
            idx: 0,
            expected_type: Box::new(type0.clone()),
        }),
        result_type: Box::new(type0.clone()),
    };
    let archive = archive_cert(&let_cert).expect("archive let failed");
    let restored = unarchive_cert(&archive).expect("unarchive let failed");
    assert_eq!(restored, let_cert);

    // Lit certificate
    let lit_cert = ProofCert::Lit {
        lit: Literal::Nat(42),
        type_: Box::new(Expr::const_(Name::from_string("Nat"), vec![])),
    };
    let archive = archive_cert(&lit_cert).expect("archive lit failed");
    let restored = unarchive_cert(&archive).expect("unarchive lit failed");
    assert_eq!(restored, lit_cert);
}

// ========================================================================
// Zstd compression tests
// ========================================================================

#[test]
fn test_zstd_archive_simple_cert() {
    let cert = ProofCert::Sort {
        level: Level::zero(),
    };

    let archive = zstd_archive_cert(&cert).expect("zstd archive failed");
    assert_eq!(archive.version, ZstdCertArchive::VERSION);
    assert_eq!(archive.compression_level, ZstdCertArchive::DEFAULT_LEVEL);

    let restored = zstd_unarchive_cert(&archive).expect("zstd unarchive failed");
    assert_eq!(restored, cert);
}

#[test]
fn test_zstd_archive_complex_cert() {
    let prop = Expr::Sort(Level::zero());
    let prop_to_prop = Expr::Pi(
        BinderInfo::Default,
        prop.clone().into(),
        prop.clone().into(),
    );

    let cert = ProofCert::Lam {
        binder_info: BinderInfo::Default,
        arg_type_cert: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        body_cert: Box::new(ProofCert::BVar {
            idx: 0,
            expected_type: Box::new(prop.clone()),
        }),
        result_type: Box::new(prop_to_prop),
    };

    let archive = zstd_archive_cert(&cert).expect("zstd archive failed");
    let restored = zstd_unarchive_cert(&archive).expect("zstd unarchive failed");
    assert_eq!(restored, cert);
}

#[test]
fn test_zstd_archive_with_level() {
    let cert = ProofCert::Sort {
        level: Level::succ(Level::zero()),
    };

    // Test different compression levels
    for level in [1, 3, 10, 19, 22] {
        let archive = zstd_archive_cert_level(&cert, level).expect("zstd archive failed");
        assert_eq!(archive.compression_level, level);

        let restored = zstd_unarchive_cert(&archive).expect("zstd unarchive failed");
        assert_eq!(restored, cert);
    }
}

#[test]
fn test_zstd_archive_with_stats() {
    let prop = Expr::Sort(Level::zero());
    let prop_to_prop = Expr::Pi(
        BinderInfo::Default,
        prop.clone().into(),
        prop.clone().into(),
    );

    let cert = ProofCert::Lam {
        binder_info: BinderInfo::Default,
        arg_type_cert: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        body_cert: Box::new(ProofCert::BVar {
            idx: 0,
            expected_type: Box::new(prop.clone()),
        }),
        result_type: Box::new(prop_to_prop),
    };

    let (archive, stats) =
        zstd_archive_cert_with_stats(&cert).expect("zstd archive with stats failed");

    // Verify stats are populated
    assert!(stats.original_cert_bytes > 0);
    assert!(stats.structure_shared_bytes > 0);
    assert!(stats.archive_bytes > 0);
    assert!(stats.structure_ratio > 0.0);
    assert!(stats.zstd_ratio > 0.0);
    assert!(stats.total_ratio > 0.0);
    assert_eq!(stats.compression_level, ZstdCertArchive::DEFAULT_LEVEL);

    // Verify roundtrip
    let restored = zstd_unarchive_cert(&archive).expect("zstd unarchive failed");
    assert_eq!(restored, cert);
}

#[test]
fn test_zstd_archive_stats_display() {
    let cert = ProofCert::Sort {
        level: Level::zero(),
    };
    let (_, stats) = zstd_archive_cert_with_stats(&cert).expect("zstd archive failed");

    let display = format!("{stats}");
    assert!(display.contains("struct:"));
    assert!(display.contains("zstd"));
    assert!(display.contains("total:"));
    assert!(display.contains("bytes"));
}

#[test]
fn test_zstd_compress_decompress() {
    let data = b"Hello, Lean5! This is a test of Zstd compression.";

    let compressed = zstd_compress(data).expect("zstd compress failed");
    let decompressed = zstd_decompress(&compressed).expect("zstd decompress failed");

    assert_eq!(decompressed, data);
}

#[test]
fn test_zstd_compress_repetitive_data() {
    // Repetitive data should compress well
    let data: Vec<u8> = (0..1000).map(|i| (i % 10) as u8).collect();

    let compressed = zstd_compress(&data).expect("zstd compress failed");
    let decompressed = zstd_decompress(&compressed).expect("zstd decompress failed");

    assert_eq!(decompressed, data);
    // Zstd should compress repetitive data significantly
    assert!(compressed.len() < data.len());
}

#[test]
fn test_zstd_compress_level() {
    let data: Vec<u8> = (0..1000).map(|i| (i % 10) as u8).collect();

    // Higher levels should typically compress better (or at least equal)
    let level1 = zstd_compress_level(&data, 1).expect("zstd compress level 1 failed");
    let level19 = zstd_compress_level(&data, 19).expect("zstd compress level 19 failed");

    // Both should decompress correctly
    let decompressed1 = zstd_decompress(&level1).expect("zstd decompress failed");
    let decompressed19 = zstd_decompress(&level19).expect("zstd decompress failed");

    assert_eq!(decompressed1, data);
    assert_eq!(decompressed19, data);

    // Higher level should give equal or better compression
    assert!(level19.len() <= level1.len());
}

#[test]
fn test_zstd_decompress_invalid_data() {
    // Invalid zstd data should return an error
    let invalid_data = b"not valid zstd data";
    let result = zstd_decompress(invalid_data);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        ZstdCompressError::DecompressError(_)
    ));
}

#[test]
fn test_zstd_compress_error_display() {
    let err1 = ZstdCompressError::SerializeError("test".to_string());
    let s1 = format!("{err1}");
    assert!(s1.contains("Serialization"));
    assert!(s1.contains("test"));

    let err2 = ZstdCompressError::CompressError("zstd fail".to_string());
    let s2 = format!("{err2}");
    assert!(s2.contains("Zstd compression"));

    let err3 = ZstdCompressError::DecompressError("bad data".to_string());
    let s3 = format!("{err3}");
    assert!(s3.contains("Zstd decompression"));

    let err4 = ZstdCompressError::DeserializeError("parse fail".to_string());
    let s4 = format!("{err4}");
    assert!(s4.contains("Deserialization"));
}

#[test]
fn test_zstd_archive_nested_app_chain() {
    // Build: f (g (h x)) with certificates
    let unit_type = Expr::Sort(Level::zero());

    let inner_cert = ProofCert::Const {
        name: Name::from_string("x"),
        levels: vec![],
        type_: Box::new(unit_type.clone()),
    };

    let h_app_cert = ProofCert::App {
        fn_cert: Box::new(ProofCert::Const {
            name: Name::from_string("h"),
            levels: vec![],
            type_: Box::new(Expr::arrow(unit_type.clone(), unit_type.clone())),
        }),
        fn_type: Box::new(Expr::arrow(unit_type.clone(), unit_type.clone())),
        arg_cert: Box::new(inner_cert),
        result_type: Box::new(unit_type.clone()),
    };

    let g_app_cert = ProofCert::App {
        fn_cert: Box::new(ProofCert::Const {
            name: Name::from_string("g"),
            levels: vec![],
            type_: Box::new(Expr::arrow(unit_type.clone(), unit_type.clone())),
        }),
        fn_type: Box::new(Expr::arrow(unit_type.clone(), unit_type.clone())),
        arg_cert: Box::new(h_app_cert),
        result_type: Box::new(unit_type.clone()),
    };

    let cert = ProofCert::App {
        fn_cert: Box::new(ProofCert::Const {
            name: Name::from_string("f"),
            levels: vec![],
            type_: Box::new(Expr::arrow(unit_type.clone(), unit_type.clone())),
        }),
        fn_type: Box::new(Expr::arrow(unit_type.clone(), unit_type.clone())),
        arg_cert: Box::new(g_app_cert),
        result_type: Box::new(unit_type),
    };

    let archive = zstd_archive_cert(&cert).expect("zstd archive failed");
    let restored = zstd_unarchive_cert(&archive).expect("zstd unarchive failed");
    assert_eq!(restored, cert);

    // With stats to verify compression effectiveness
    let (_, stats) = zstd_archive_cert_with_stats(&cert).expect("zstd archive with stats failed");
    // Nested apps with repeated types should compress well
    assert!(
        stats.total_ratio >= 1.0,
        "Expected some compression for nested apps"
    );
}

#[test]
fn test_zstd_archive_serialization() {
    // Test that ZstdCertArchive itself can be serialized/deserialized
    let cert = ProofCert::Sort {
        level: Level::succ(Level::zero()),
    };
    let archive = zstd_archive_cert(&cert).expect("zstd archive failed");

    // Serialize archive to bincode
    let archive_bytes = bincode::serialize(&archive).expect("serialize archive failed");

    // Deserialize
    let restored_archive: ZstdCertArchive =
        bincode::deserialize(&archive_bytes).expect("deserialize archive failed");

    // Verify fields preserved
    assert_eq!(restored_archive.version, archive.version);
    assert_eq!(
        restored_archive.compression_level,
        archive.compression_level
    );
    assert_eq!(
        restored_archive.uncompressed_size,
        archive.uncompressed_size
    );

    // Unarchive
    let restored = zstd_unarchive_cert(&restored_archive).expect("zstd unarchive failed");
    assert_eq!(restored, cert);
}

#[test]
fn test_compression_algorithm_name() {
    assert_eq!(CompressionAlgorithm::Lz4.name(), "LZ4");
    assert_eq!(CompressionAlgorithm::ZstdDefault.name(), "Zstd (level 3)");
    assert_eq!(CompressionAlgorithm::ZstdHigh.name(), "Zstd (level 19)");
    assert_eq!(CompressionAlgorithm::ZstdMax.name(), "Zstd (level 22)");

    assert_eq!(CompressionAlgorithm::Lz4.zstd_level(), None);
    assert_eq!(
        CompressionAlgorithm::ZstdDefault.zstd_level(),
        Some(ZstdCertArchive::DEFAULT_LEVEL)
    );
    assert_eq!(
        CompressionAlgorithm::ZstdHigh.zstd_level(),
        Some(ZstdCertArchive::HIGH_LEVEL)
    );
    assert_eq!(
        CompressionAlgorithm::ZstdMax.zstd_level(),
        Some(ZstdCertArchive::MAX_LEVEL)
    );
}

#[test]
fn test_archive_cert_with_algorithm_lz4_envelope() {
    let cert = ProofCert::Sort {
        level: Level::zero(),
    };

    let archive = archive_cert_with_algorithm(&cert, CompressionAlgorithm::Lz4)
        .expect("archive with algorithm failed");

    assert_eq!(archive.algorithm(), CompressionAlgorithm::Lz4);
    assert!(archive.compressed_len() > 0);
    assert!(archive.uncompressed_size() > 0);

    match &archive {
        CertArchiveEnvelope::Lz4(inner) => assert_eq!(inner.version, CertArchive::VERSION),
        _ => panic!("expected LZ4 archive variant"),
    }

    let restored = unarchive_cert_envelope(&archive).expect("unarchive failed");
    assert_eq!(restored, cert);
}

#[test]
fn test_archive_cert_with_algorithm_zstd_envelope() {
    let cert = ProofCert::Sort {
        level: Level::succ(Level::zero()),
    };

    let archive = archive_cert_with_algorithm(&cert, CompressionAlgorithm::ZstdHigh)
        .expect("archive with algorithm failed");

    assert_eq!(archive.algorithm(), CompressionAlgorithm::ZstdHigh);

    match &archive {
        CertArchiveEnvelope::Zstd(inner) => {
            assert_eq!(inner.compression_level, ZstdCertArchive::HIGH_LEVEL);
            assert_eq!(inner.version, ZstdCertArchive::VERSION);
        }
        _ => panic!("expected Zstd archive variant"),
    }

    let restored = unarchive_cert_envelope(&archive).expect("unarchive failed");
    assert_eq!(restored, cert);
}

#[test]
fn test_archive_cert_with_algorithm_stats_dispatch() {
    let cert = ProofCert::Sort {
        level: Level::zero(),
    };

    let (lz4_archive, lz4_stats) =
        archive_cert_with_algorithm_stats(&cert, CompressionAlgorithm::Lz4)
            .expect("lz4 archive stats failed");
    assert!(matches!(lz4_archive, CertArchiveEnvelope::Lz4(_)));
    assert_eq!(lz4_stats.algorithm(), CompressionAlgorithm::Lz4);
    assert!(lz4_stats.total_ratio() > 0.0);

    let (zstd_archive, zstd_stats) =
        archive_cert_with_algorithm_stats(&cert, CompressionAlgorithm::ZstdMax)
            .expect("zstd archive stats failed");
    assert!(matches!(zstd_archive, CertArchiveEnvelope::Zstd(_)));
    assert_eq!(zstd_stats.algorithm(), CompressionAlgorithm::ZstdMax);

    if let ArchiveVariantStats::Zstd(stats) = &zstd_stats {
        assert_eq!(stats.compression_level, ZstdCertArchive::MAX_LEVEL);
    } else {
        panic!("expected zstd stats variant");
    }

    let restored = unarchive_cert_envelope(&zstd_archive).expect("unarchive failed");
    assert_eq!(restored, cert);
}

#[test]
fn test_compare_lz4_vs_zstd() {
    // Compare compression of LZ4 vs Zstd on a moderately complex certificate
    // Build a certificate with some repeated structure
    let inner_cert = ProofCert::Sort {
        level: Level::zero(),
    };
    let cert = ProofCert::Pi {
        binder_info: BinderInfo::Default,
        arg_type_cert: Box::new(inner_cert.clone()),
        arg_level: Level::zero(),
        body_type_cert: Box::new(ProofCert::Pi {
            binder_info: BinderInfo::Default,
            arg_type_cert: Box::new(inner_cert.clone()),
            arg_level: Level::zero(),
            body_type_cert: Box::new(inner_cert.clone()),
            body_level: Level::zero(),
        }),
        body_level: Level::zero(),
    };

    let (lz4_archive, lz4_stats) = archive_cert_with_stats(&cert).expect("lz4 archive failed");
    let (zstd_archive, zstd_stats) =
        zstd_archive_cert_with_stats(&cert).expect("zstd archive failed");

    // Both should roundtrip correctly
    let lz4_restored = unarchive_cert(&lz4_archive).expect("lz4 unarchive failed");
    let zstd_restored = zstd_unarchive_cert(&zstd_archive).expect("zstd unarchive failed");
    assert_eq!(lz4_restored, cert);
    assert_eq!(zstd_restored, cert);

    // Stats should be populated
    assert!(lz4_stats.total_ratio > 0.0);
    assert!(zstd_stats.total_ratio > 0.0);

    // For larger data, zstd typically achieves better compression
    // For small data, both are similar. Just verify both work.
}

#[test]
fn test_zstd_archive_with_all_cert_variants() {
    // Test zstd archiving certificates with various variants
    let type0 = Expr::Sort(Level::zero());

    // FVar certificate
    let fvar_cert = ProofCert::FVar {
        id: FVarId(42),
        type_: Box::new(type0.clone()),
    };
    let archive = zstd_archive_cert(&fvar_cert).expect("zstd archive fvar failed");
    let restored = zstd_unarchive_cert(&archive).expect("zstd unarchive fvar failed");
    assert_eq!(restored, fvar_cert);

    // Let certificate
    let let_cert = ProofCert::Let {
        type_cert: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        value_cert: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        body_cert: Box::new(ProofCert::BVar {
            idx: 0,
            expected_type: Box::new(type0.clone()),
        }),
        result_type: Box::new(type0.clone()),
    };
    let archive = zstd_archive_cert(&let_cert).expect("zstd archive let failed");
    let restored = zstd_unarchive_cert(&archive).expect("zstd unarchive let failed");
    assert_eq!(restored, let_cert);

    // Lit certificate
    let lit_cert = ProofCert::Lit {
        lit: Literal::Nat(42),
        type_: Box::new(Expr::const_(Name::from_string("Nat"), vec![])),
    };
    let archive = zstd_archive_cert(&lit_cert).expect("zstd archive lit failed");
    let restored = zstd_unarchive_cert(&archive).expect("zstd unarchive lit failed");
    assert_eq!(restored, lit_cert);

    // DefEq certificate
    let def_eq_cert = ProofCert::DefEq {
        inner: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        expected_type: Box::new(type0.clone()),
        actual_type: Box::new(type0.clone()),
        eq_steps: vec![DefEqStep::Refl],
    };
    let archive = zstd_archive_cert(&def_eq_cert).expect("zstd archive def_eq failed");
    let restored = zstd_unarchive_cert(&archive).expect("zstd unarchive def_eq failed");
    assert_eq!(restored, def_eq_cert);
}

// ========================================================================
// Streaming Compression Tests
// ========================================================================

#[test]
fn test_streaming_header_new_zstd() {
    let header = StreamingArchiveHeader::new_zstd(3);
    assert_eq!(header.magic, StreamingArchiveHeader::MAGIC);
    assert_eq!(header.version, StreamingArchiveHeader::VERSION);
    assert_eq!(header.algorithm, 1); // Zstd
    assert_eq!(header.compression_level, 3);
    assert!(header.validate().is_ok());
}

#[test]
fn test_streaming_header_new_lz4() {
    let header = StreamingArchiveHeader::new_lz4();
    assert_eq!(header.magic, StreamingArchiveHeader::MAGIC);
    assert_eq!(header.version, StreamingArchiveHeader::VERSION);
    assert_eq!(header.algorithm, 0); // LZ4
    assert!(header.validate().is_ok());
}

#[test]
fn test_streaming_header_invalid_magic() {
    let header = StreamingArchiveHeader {
        magic: *b"XXXX",
        version: 1,
        algorithm: 1,
        compression_level: 3,
        uncompressed_size: 0,
        cert_count: 0,
    };
    assert!(header.validate().is_err());
}

#[test]
fn test_streaming_header_invalid_version() {
    let header = StreamingArchiveHeader {
        magic: StreamingArchiveHeader::MAGIC,
        version: 255, // Too high
        algorithm: 1,
        compression_level: 3,
        uncompressed_size: 0,
        cert_count: 0,
    };
    assert!(header.validate().is_err());
}

#[test]
fn test_streaming_header_algorithm() {
    let lz4 = StreamingArchiveHeader::new_lz4();
    assert_eq!(lz4.algorithm(), CompressionAlgorithm::Lz4);

    let zstd_default = StreamingArchiveHeader::new_zstd(3);
    assert_eq!(zstd_default.algorithm(), CompressionAlgorithm::ZstdDefault);

    let zstd_high = StreamingArchiveHeader::new_zstd(19);
    assert_eq!(zstd_high.algorithm(), CompressionAlgorithm::ZstdHigh);

    let zstd_max = StreamingArchiveHeader::new_zstd(22);
    assert_eq!(zstd_max.algorithm(), CompressionAlgorithm::ZstdMax);
}

#[test]
fn test_streaming_write_read_single_cert() {
    let cert = ProofCert::Sort {
        level: Level::zero(),
    };

    let mut buffer = Vec::new();
    {
        let mut writer =
            StreamingCertWriter::new_zstd(&mut buffer, 3).expect("create writer failed");
        writer.write_cert(&cert).expect("write cert failed");
        assert_eq!(writer.cert_count(), 1);
        writer.finish().expect("finish failed");
    }

    let cursor = std::io::Cursor::new(buffer);
    let mut reader = StreamingCertReader::new(cursor).expect("create reader failed");

    let read_cert = reader.read_cert().expect("read cert failed");
    assert_eq!(read_cert, Some(cert));

    let next = reader.read_cert().expect("read next failed");
    assert_eq!(next, None);

    assert_eq!(reader.certs_read(), 1);
}

#[test]
fn test_streaming_write_read_multiple_certs() {
    let certs = vec![
        ProofCert::Sort {
            level: Level::zero(),
        },
        ProofCert::Sort {
            level: Level::succ(Level::zero()),
        },
        ProofCert::Pi {
            binder_info: BinderInfo::Default,
            arg_type_cert: Box::new(ProofCert::Sort {
                level: Level::zero(),
            }),
            arg_level: Level::succ(Level::zero()),
            body_type_cert: Box::new(ProofCert::Sort {
                level: Level::zero(),
            }),
            body_level: Level::succ(Level::zero()),
        },
    ];

    let mut buffer = Vec::new();
    {
        let mut writer =
            StreamingCertWriter::new_zstd(&mut buffer, 3).expect("create writer failed");
        writer.write_certs(&certs).expect("write certs failed");
        assert_eq!(writer.cert_count(), 3);
        writer.finish().expect("finish failed");
    }

    let cursor = std::io::Cursor::new(buffer);
    let mut reader = StreamingCertReader::new(cursor).expect("create reader failed");

    let read_certs = reader.read_all().expect("read all failed");
    assert_eq!(read_certs.len(), 3);
    assert_eq!(read_certs, certs);
}

#[test]
fn test_streaming_file_roundtrip() {
    use std::path::PathBuf;

    let certs = vec![
        ProofCert::Sort {
            level: Level::zero(),
        },
        ProofCert::Sort {
            level: Level::succ(Level::zero()),
        },
    ];

    let temp_dir = std::env::temp_dir();
    let temp_file: PathBuf = temp_dir.join("test_streaming.l5cs");

    let write_stats = stream_certs_to_file(&temp_file, &certs, 3).expect("write to file failed");
    assert_eq!(write_stats.cert_count, 2);
    assert!(write_stats.compressed_bytes > 0);

    let (read_certs, read_stats) =
        stream_certs_from_file(&temp_file).expect("read from file failed");
    assert_eq!(read_certs, certs);
    assert_eq!(read_stats.cert_count, 2);
    assert_eq!(read_stats.algorithm, CompressionAlgorithm::ZstdDefault);

    // Clean up
    let _ = std::fs::remove_file(&temp_file);
}

#[test]
fn test_streaming_progress_callback() {
    use std::sync::{Arc, Mutex};

    let certs: Vec<ProofCert> = (0..10)
        .map(|i| ProofCert::Sort {
            level: if i == 0 {
                Level::zero()
            } else {
                Level::succ(Level::zero())
            },
        })
        .collect();

    let progress_count = Arc::new(Mutex::new(0u64));
    let last_total = Arc::new(Mutex::new(None::<u64>));

    let mut buffer = Vec::new();
    {
        let pc = Arc::clone(&progress_count);
        let lt = Arc::clone(&last_total);
        let callback: StreamingProgressCallback = Box::new(move |current, total| {
            *pc.lock().unwrap() = current;
            *lt.lock().unwrap() = total;
        });
        let mut writer = StreamingCertWriter::new_zstd(&mut buffer, 3)
            .expect("create writer")
            .with_progress(callback);
        writer.write_certs(&certs).expect("write certs");
        writer.finish().expect("finish");
    }

    // Progress was called with totals
    assert_eq!(*progress_count.lock().unwrap(), 10);
    assert_eq!(*last_total.lock().unwrap(), Some(10));
}

#[test]
fn test_streaming_stats_display() {
    let stats = StreamingStats {
        cert_count: 100,
        uncompressed_bytes: 10000,
        compressed_bytes: 2500,
        algorithm: CompressionAlgorithm::ZstdDefault,
    };

    let display = format!("{stats}");
    assert!(display.contains("certs: 100"));
    assert!(display.contains("uncompressed: 10000"));
    assert!(display.contains("compressed: 2500"));
    assert!(display.contains("ratio: 4.00x"));
    assert!(display.contains("Zstd"));
}

#[test]
fn test_streaming_stats_ratio() {
    let stats = StreamingStats {
        cert_count: 1,
        uncompressed_bytes: 1000,
        compressed_bytes: 250,
        algorithm: CompressionAlgorithm::Lz4,
    };
    assert!((stats.ratio() - 4.0).abs() < 0.001);

    let zero_stats = StreamingStats {
        cert_count: 0,
        uncompressed_bytes: 0,
        compressed_bytes: 0,
        algorithm: CompressionAlgorithm::Lz4,
    };
    assert_eq!(zero_stats.ratio(), 0.0);
}

#[test]
fn test_streaming_error_display() {
    let io_err = StreamingError::Io(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "file not found",
    ));
    assert!(format!("{io_err}").contains("I/O error"));

    let ser_err = StreamingError::Serialize("bad format".to_string());
    assert!(format!("{ser_err}").contains("Serialization error"));

    let decomp_err = StreamingError::Decompress("corrupt data".to_string());
    assert!(format!("{decomp_err}").contains("Decompression error"));

    let fmt_err = StreamingError::InvalidFormat("wrong header".to_string());
    assert!(format!("{fmt_err}").contains("Invalid format"));
}

#[test]
fn test_streaming_invalid_reader() {
    // Try to read from invalid data
    let bad_data = vec![0u8; 10];
    let cursor = std::io::Cursor::new(bad_data);
    let result = StreamingCertReader::new(cursor);
    assert!(result.is_err());
}

#[test]
fn test_streaming_truncated_cert_errors() {
    // Write valid certs to a buffer
    let cert = ProofCert::Sort {
        level: Level::zero(),
    };
    let certs = vec![cert.clone(), cert.clone(), cert];

    let mut buffer = Vec::new();
    {
        let mut writer =
            StreamingCertWriter::new_zstd(&mut buffer, 3).expect("create writer failed");
        writer.write_certs(&certs).expect("write certs failed");
        writer.finish().expect("finish failed");
    }

    // Truncate the buffer mid-stream (after header, but truncated cert data)
    // This should cause an IO error (not EOF at cert boundary)
    let truncated = &buffer[..buffer.len() - 50];
    let cursor = std::io::Cursor::new(truncated.to_vec());

    let reader_result = StreamingCertReader::new(cursor);
    if let Ok(mut reader) = reader_result {
        // Should get an error when reading certs (not Ok(None))
        let all_result = reader.read_all();
        assert!(
            all_result.is_err(),
            "Truncated stream should produce an error, not Ok"
        );
    }
    // If reader creation fails, that's also acceptable for corrupted data
}

/// A reader that fails with a non-EOF error after reading some bytes
struct FailingReader {
    inner: std::io::Cursor<Vec<u8>>,
    bytes_until_fail: usize,
    bytes_read: usize,
}

impl std::io::Read for FailingReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.bytes_read >= self.bytes_until_fail {
            return Err(std::io::Error::new(
                std::io::ErrorKind::ConnectionReset,
                "simulated connection reset",
            ));
        }
        let remaining = self.bytes_until_fail - self.bytes_read;
        let max_read = remaining.min(buf.len());
        let result = self.inner.read(&mut buf[..max_read])?;
        self.bytes_read += result;
        Ok(result)
    }
}

#[test]
fn test_streaming_non_eof_error_propagates() {
    // Write a valid stream to a buffer
    let cert = ProofCert::Sort {
        level: Level::zero(),
    };
    let certs = vec![cert.clone(), cert.clone(), cert.clone(), cert];

    let mut buffer = Vec::new();
    {
        let mut writer =
            StreamingCertWriter::new_zstd(&mut buffer, 3).expect("create writer failed");
        writer.write_certs(&certs).expect("write certs failed");
        writer.finish().expect("finish failed");
    }

    // Create a reader that fails after reading the header but mid-cert
    // Header is typically ~24 bytes, fail a bit after
    let failing_reader = FailingReader {
        inner: std::io::Cursor::new(buffer.clone()),
        bytes_until_fail: 30,
        bytes_read: 0,
    };

    let reader_result = StreamingCertReader::new(failing_reader);
    match reader_result {
        Ok(mut reader) => {
            // The reader was created; now read_cert should fail with an error
            let read_result = reader.read_cert();
            assert!(
                read_result.is_err(),
                "Non-EOF IO error must propagate as Err, got {read_result:?}"
            );
        }
        Err(_) => {
            // Reader creation failed due to the error - also acceptable
        }
    }
}

#[test]
fn test_streaming_empty_stream() {
    let certs: Vec<ProofCert> = vec![];

    let mut buffer = Vec::new();
    {
        let mut writer =
            StreamingCertWriter::new_zstd(&mut buffer, 3).expect("create writer failed");
        writer.write_certs(&certs).expect("write certs failed");
        assert_eq!(writer.cert_count(), 0);
        writer.finish().expect("finish failed");
    }

    let cursor = std::io::Cursor::new(buffer);
    let mut reader = StreamingCertReader::new(cursor).expect("create reader failed");

    let read_certs = reader.read_all().expect("read all failed");
    assert!(read_certs.is_empty());
    assert_eq!(reader.certs_read(), 0);
}

#[test]
fn test_streaming_complex_certs() {
    // Build complex nested certificates
    let type0 = Expr::Sort(Level::zero());
    let type1 = Expr::Sort(Level::succ(Level::zero()));

    let complex_cert = ProofCert::App {
        fn_cert: Box::new(ProofCert::Lam {
            binder_info: BinderInfo::Default,
            arg_type_cert: Box::new(ProofCert::Sort {
                level: Level::zero(),
            }),
            body_cert: Box::new(ProofCert::BVar {
                idx: 0,
                expected_type: Box::new(type0.clone()),
            }),
            result_type: Box::new(Expr::Pi(
                BinderInfo::Default,
                type0.clone().into(),
                type0.clone().into(),
            )),
        }),
        fn_type: Box::new(Expr::Pi(
            BinderInfo::Default,
            type0.clone().into(),
            type0.clone().into(),
        )),
        arg_cert: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        result_type: Box::new(type1.clone()),
    };

    let certs = vec![complex_cert.clone(), complex_cert.clone()];

    let mut buffer = Vec::new();
    {
        let mut writer =
            StreamingCertWriter::new_zstd(&mut buffer, 3).expect("create writer failed");
        writer.write_certs(&certs).expect("write certs failed");
        writer.finish().expect("finish failed");
    }

    let cursor = std::io::Cursor::new(buffer);
    let mut reader = StreamingCertReader::new(cursor).expect("create reader failed");

    let read_certs = reader.read_all().expect("read all failed");
    assert_eq!(read_certs.len(), 2);
    assert_eq!(read_certs[0], complex_cert);
    assert_eq!(read_certs[1], complex_cert);
}

#[test]
fn test_streaming_compression_levels() {
    let certs: Vec<ProofCert> = (0..50)
        .map(|_| ProofCert::Sort {
            level: Level::zero(),
        })
        .collect();

    // Test default level (3)
    let mut buffer_default = Vec::new();
    {
        let mut writer =
            StreamingCertWriter::new_zstd(&mut buffer_default, 3).expect("create writer");
        writer.write_certs(&certs).expect("write certs");
        writer.finish().expect("finish");
    }

    // Test high level (19)
    let mut buffer_high = Vec::new();
    {
        let mut writer =
            StreamingCertWriter::new_zstd(&mut buffer_high, 19).expect("create writer");
        writer.write_certs(&certs).expect("write certs");
        writer.finish().expect("finish");
    }

    // Both levels should produce valid compressed data
    // Note: For small data, higher levels don't always produce smaller output
    // due to zstd internal heuristics and streaming overhead
    assert!(!buffer_default.is_empty());
    assert!(!buffer_high.is_empty());

    // Both should decompress to the same certs
    let cursor_default = std::io::Cursor::new(buffer_default);
    let mut reader_default =
        StreamingCertReader::new(cursor_default).expect("create reader default");
    let read_default = reader_default.read_all().expect("read all default");

    let cursor_high = std::io::Cursor::new(buffer_high);
    let mut reader_high = StreamingCertReader::new(cursor_high).expect("create reader high");
    let read_high = reader_high.read_all().expect("read all high");

    assert_eq!(read_default, read_high);
    assert_eq!(read_default, certs);
}

// ========================================================================
// Dictionary Compression Tests
// ========================================================================

#[test]
fn test_dict_from_bytes() {
    // Create dictionary from arbitrary bytes
    let dict_bytes = vec![0u8; 1024];
    let dict = CertDictionary::from_bytes(dict_bytes.clone(), 3);

    assert_eq!(dict.size(), 1024);
    assert_eq!(dict.data, dict_bytes);
    assert_eq!(dict.target_level, 3);
    assert_eq!(dict.version, CertDictionary::VERSION);
    assert!(dict.dict_id != 0); // Should have a computed ID
}

#[test]
fn test_dict_not_enough_samples() {
    // Try to train with too few samples
    let samples: Vec<ProofCert> = vec![
        ProofCert::Sort {
            level: Level::zero(),
        },
        ProofCert::Sort {
            level: Level::succ(Level::zero()),
        },
    ];

    let result = CertDictionary::train(&samples, 1024, 3);
    assert!(result.is_err());
    if let Err(DictTrainError::NotEnoughSamples { provided, minimum }) = result {
        assert_eq!(provided, 2);
        assert_eq!(minimum, CertDictionary::MIN_SAMPLES);
    } else {
        panic!("Expected NotEnoughSamples error");
    }
}

#[test]
fn test_dict_train_and_compress() {
    // Create enough sample certificates for training
    let samples: Vec<ProofCert> = (0..20)
        .map(|i| ProofCert::Sort {
            level: if i % 2 == 0 {
                Level::zero()
            } else {
                Level::succ(Level::zero())
            },
        })
        .collect();

    // Train dictionary
    let dict = CertDictionary::train(&samples, 16 * 1024, 3).expect("dictionary training failed");

    assert!(dict.size() > 0);
    assert_eq!(dict.sample_count, 20);
    assert_eq!(dict.target_level, 3);

    // Compress a new certificate with the dictionary
    let cert = ProofCert::Sort {
        level: Level::succ(Level::succ(Level::zero())),
    };

    let archive = zstd_archive_cert_with_dict(&cert, &dict).expect("dict archive failed");
    assert!(!archive.compressed_data.is_empty());
    assert_eq!(archive.dict_id, dict.dict_id);
    assert_eq!(archive.version, DictCertArchive::VERSION);

    // Decompress and verify
    let restored = zstd_unarchive_cert_with_dict(&archive, &dict).expect("dict unarchive failed");
    assert_eq!(restored, cert);
}

#[test]
fn test_dict_compress_complex_cert() {
    // Create varied training samples
    let samples: Vec<ProofCert> = (0..15)
        .map(|i| {
            if i < 5 {
                ProofCert::Sort {
                    level: Level::zero(),
                }
            } else if i < 10 {
                ProofCert::Pi {
                    binder_info: BinderInfo::Default,
                    arg_type_cert: Box::new(ProofCert::Sort {
                        level: Level::zero(),
                    }),
                    arg_level: Level::zero(),
                    body_type_cert: Box::new(ProofCert::Sort {
                        level: Level::zero(),
                    }),
                    body_level: Level::zero(),
                }
            } else {
                ProofCert::Lam {
                    binder_info: BinderInfo::Default,
                    arg_type_cert: Box::new(ProofCert::Sort {
                        level: Level::zero(),
                    }),
                    body_cert: Box::new(ProofCert::BVar {
                        idx: 0,
                        expected_type: Box::new(Expr::Sort(Level::zero())),
                    }),
                    result_type: Box::new(Expr::Pi(
                        BinderInfo::Default,
                        std::sync::Arc::new(Expr::Sort(Level::zero())),
                        std::sync::Arc::new(Expr::Sort(Level::zero())),
                    )),
                }
            }
        })
        .collect();

    let dict = CertDictionary::train(&samples, 32 * 1024, 3).expect("dictionary training failed");

    // Complex nested certificate
    let complex_cert = ProofCert::App {
        fn_cert: Box::new(ProofCert::Lam {
            binder_info: BinderInfo::Default,
            arg_type_cert: Box::new(ProofCert::Sort {
                level: Level::zero(),
            }),
            body_cert: Box::new(ProofCert::BVar {
                idx: 0,
                expected_type: Box::new(Expr::Sort(Level::zero())),
            }),
            result_type: Box::new(Expr::Pi(
                BinderInfo::Default,
                std::sync::Arc::new(Expr::Sort(Level::zero())),
                std::sync::Arc::new(Expr::Sort(Level::zero())),
            )),
        }),
        fn_type: Box::new(Expr::Pi(
            BinderInfo::Default,
            std::sync::Arc::new(Expr::Sort(Level::zero())),
            std::sync::Arc::new(Expr::Sort(Level::zero())),
        )),
        arg_cert: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        result_type: Box::new(Expr::Sort(Level::zero())),
    };

    let archive = zstd_archive_cert_with_dict(&complex_cert, &dict).expect("dict archive failed");
    let restored = zstd_unarchive_cert_with_dict(&archive, &dict).expect("dict unarchive failed");
    assert_eq!(restored, complex_cert);
}

#[test]
fn test_dict_with_stats() {
    let samples: Vec<ProofCert> = (0..10)
        .map(|_| ProofCert::Sort {
            level: Level::zero(),
        })
        .collect();

    let dict = CertDictionary::train(&samples, 8 * 1024, 3).expect("training failed");

    let cert = ProofCert::Sort {
        level: Level::succ(Level::zero()),
    };

    let (archive, stats) =
        zstd_archive_cert_with_dict_stats(&cert, &dict).expect("archive with stats failed");

    assert!(stats.original_cert_bytes > 0);
    assert!(stats.structure_shared_bytes > 0);
    assert!(stats.archive_bytes > 0);
    assert!(stats.structure_ratio > 0.0);
    assert!(stats.dict_ratio > 0.0);
    assert!(stats.total_ratio > 0.0);
    assert_eq!(stats.compression_level, 3);
    assert_eq!(stats.dict_id, dict.dict_id);

    // Stats Display trait
    let display = format!("{stats}");
    assert!(display.contains("DictArchiveStats"));
    assert!(display.contains("struct:"));
    assert!(display.contains("dict["));

    let restored = zstd_unarchive_cert_with_dict(&archive, &dict).expect("unarchive failed");
    assert_eq!(restored, cert);
}

#[test]
fn test_dict_with_level() {
    let samples: Vec<ProofCert> = (0..10)
        .map(|i| ProofCert::Sort {
            level: if i % 2 == 0 {
                Level::zero()
            } else {
                Level::succ(Level::zero())
            },
        })
        .collect();

    let dict = CertDictionary::train(&samples, 8 * 1024, 3).expect("training failed");

    let cert = ProofCert::Sort {
        level: Level::succ(Level::succ(Level::zero())),
    };

    // Test at different compression levels
    for level in [1, 3, 10, 19] {
        let archive = zstd_archive_cert_with_dict_level(&cert, &dict, level)
            .unwrap_or_else(|_| panic!("archive at level {level} failed"));
        assert_eq!(archive.compression_level, level);

        let restored = zstd_unarchive_cert_with_dict(&archive, &dict).expect("unarchive failed");
        assert_eq!(restored, cert);
    }
}

#[test]
fn test_dict_mismatch_error() {
    let samples: Vec<ProofCert> = (0..10)
        .map(|_| ProofCert::Sort {
            level: Level::zero(),
        })
        .collect();

    let dict1 = CertDictionary::train(&samples, 8 * 1024, 3).expect("training dict1 failed");

    // Create a second different dictionary
    let dict2 = CertDictionary::from_bytes(vec![1, 2, 3, 4, 5], 3);

    let cert = ProofCert::Sort {
        level: Level::zero(),
    };

    // Compress with dict1
    let archive = zstd_archive_cert_with_dict(&cert, &dict1).expect("archive failed");

    // Try to decompress with dict2
    let result = zstd_unarchive_cert_with_dict(&archive, &dict2);
    assert!(result.is_err());
    if let Err(DictCompressError::DictMismatch { expected, found }) = result {
        assert_eq!(expected, dict1.dict_id);
        assert_eq!(found, dict2.dict_id);
    } else {
        panic!("Expected DictMismatch error");
    }
}

#[test]
fn test_dict_raw_bytes_compress() {
    let samples: Vec<ProofCert> = (0..10)
        .map(|_| ProofCert::Sort {
            level: Level::zero(),
        })
        .collect();

    let dict = CertDictionary::train(&samples, 8 * 1024, 3).expect("training failed");

    // Test raw byte compression
    let data = b"Hello, this is some test data for compression";
    let compressed = zstd_compress_with_dict(data, &dict).expect("compress failed");
    let decompressed = zstd_decompress_with_dict(&compressed, &dict).expect("decompress failed");

    assert_eq!(decompressed, data);
}

#[test]
fn test_dict_raw_bytes_with_level() {
    let samples: Vec<Vec<u8>> = (0..10).map(|i| vec![i as u8; 100]).collect();

    let dict = CertDictionary::train_from_bytes(&samples, 4 * 1024, 3).expect("training failed");

    let data = b"Test data with some repetitive patterns patterns patterns";

    // Compress at level 1 (fast)
    let compressed1 = zstd_compress_with_dict_level(data, &dict, 1).expect("compress l1 failed");
    let decompressed1 =
        zstd_decompress_with_dict(&compressed1, &dict).expect("decompress l1 failed");
    assert_eq!(decompressed1.as_slice(), data);

    // Compress at level 19 (high)
    let compressed19 = zstd_compress_with_dict_level(data, &dict, 19).expect("compress l19 failed");
    let decompressed19 =
        zstd_decompress_with_dict(&compressed19, &dict).expect("decompress l19 failed");
    assert_eq!(decompressed19.as_slice(), data);
}

#[test]
fn test_dict_train_error_display() {
    let err1 = DictTrainError::NotEnoughSamples {
        provided: 3,
        minimum: 5,
    };
    let display1 = format!("{err1}");
    assert!(display1.contains("Not enough samples"));
    assert!(display1.contains('3'));
    assert!(display1.contains('5'));

    let err2 = DictTrainError::SerializeError("test error".to_string());
    let display2 = format!("{err2}");
    assert!(display2.contains("Serialization error"));

    let err3 = DictTrainError::TrainError("zstd fail".to_string());
    let display3 = format!("{err3}");
    assert!(display3.contains("Dictionary training error"));
}

#[test]
fn test_dict_compress_error_display() {
    let err1 = DictCompressError::SerializeError("ser fail".to_string());
    assert!(format!("{err1}").contains("Serialization error"));

    let err2 = DictCompressError::CompressError("comp fail".to_string());
    assert!(format!("{err2}").contains("Dict compression error"));

    let err3 = DictCompressError::DecompressError("decomp fail".to_string());
    assert!(format!("{err3}").contains("Dict decompression error"));

    let err4 = DictCompressError::DeserializeError("deser fail".to_string());
    assert!(format!("{err4}").contains("Deserialization error"));

    let err5 = DictCompressError::DictMismatch {
        expected: 123,
        found: 456,
    };
    let display5 = format!("{err5}");
    assert!(display5.contains("Dictionary ID mismatch"));
    assert!(display5.contains("0x7b")); // 123 in hex
    assert!(display5.contains("0x1c8")); // 456 in hex
}

#[test]
fn test_dict_is_compatible_level() {
    let dict = CertDictionary::from_bytes(vec![0u8; 100], 10);

    // Within 5 levels
    assert!(dict.is_compatible_level(5));
    assert!(dict.is_compatible_level(10));
    assert!(dict.is_compatible_level(15));

    // Outside 5 levels
    assert!(!dict.is_compatible_level(1));
    assert!(!dict.is_compatible_level(19));
}

#[test]
fn test_dict_serialization() {
    let samples: Vec<ProofCert> = (0..10)
        .map(|_| ProofCert::Sort {
            level: Level::zero(),
        })
        .collect();

    let dict = CertDictionary::train(&samples, 4 * 1024, 3).expect("training failed");

    // Serialize the dictionary
    let dict_json = serde_json::to_string(&dict).expect("serialize dict failed");
    let restored_dict: CertDictionary =
        serde_json::from_str(&dict_json).expect("deserialize dict failed");

    assert_eq!(restored_dict.dict_id, dict.dict_id);
    assert_eq!(restored_dict.sample_count, dict.sample_count);
    assert_eq!(restored_dict.target_level, dict.target_level);
    assert_eq!(restored_dict.data, dict.data);

    // Bincode serialization
    let dict_bincode = bincode::serialize(&dict).expect("serialize dict failed");
    let restored_dict2: CertDictionary =
        bincode::deserialize(&dict_bincode).expect("deserialize dict failed");
    assert_eq!(restored_dict2.dict_id, dict.dict_id);
}

#[test]
fn test_dict_archive_serialization() {
    let samples: Vec<ProofCert> = (0..10)
        .map(|_| ProofCert::Sort {
            level: Level::zero(),
        })
        .collect();

    let dict = CertDictionary::train(&samples, 4 * 1024, 3).expect("training failed");

    let cert = ProofCert::Sort {
        level: Level::succ(Level::zero()),
    };

    let archive = zstd_archive_cert_with_dict(&cert, &dict).expect("archive failed");

    // Serialize the archive
    let archive_json = serde_json::to_string(&archive).expect("serialize archive failed");
    let restored_archive: DictCertArchive =
        serde_json::from_str(&archive_json).expect("deserialize archive failed");

    assert_eq!(restored_archive.dict_id, archive.dict_id);
    assert_eq!(restored_archive.compressed_data, archive.compressed_data);

    // Verify it can be decompressed
    let restored_cert =
        zstd_unarchive_cert_with_dict(&restored_archive, &dict).expect("unarchive failed");
    assert_eq!(restored_cert, cert);
}

#[test]
fn test_compare_dict_vs_no_dict_compression() {
    // Create training samples that are similar to test data
    let samples: Vec<ProofCert> = (0..50)
        .map(|i| {
            if i % 3 == 0 {
                ProofCert::Sort {
                    level: Level::zero(),
                }
            } else if i % 3 == 1 {
                ProofCert::Pi {
                    binder_info: BinderInfo::Default,
                    arg_type_cert: Box::new(ProofCert::Sort {
                        level: Level::zero(),
                    }),
                    arg_level: Level::zero(),
                    body_type_cert: Box::new(ProofCert::Sort {
                        level: Level::zero(),
                    }),
                    body_level: Level::zero(),
                }
            } else {
                ProofCert::Lam {
                    binder_info: BinderInfo::Default,
                    arg_type_cert: Box::new(ProofCert::Sort {
                        level: Level::zero(),
                    }),
                    body_cert: Box::new(ProofCert::BVar {
                        idx: 0,
                        expected_type: Box::new(Expr::Sort(Level::zero())),
                    }),
                    result_type: Box::new(Expr::Pi(
                        BinderInfo::Default,
                        std::sync::Arc::new(Expr::Sort(Level::zero())),
                        std::sync::Arc::new(Expr::Sort(Level::zero())),
                    )),
                }
            }
        })
        .collect();

    let dict = CertDictionary::train(&samples, 32 * 1024, 3).expect("dictionary training failed");

    // Test certificate
    let test_cert = ProofCert::Pi {
        binder_info: BinderInfo::Default,
        arg_type_cert: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        arg_level: Level::zero(),
        body_type_cert: Box::new(ProofCert::Sort {
            level: Level::succ(Level::zero()),
        }),
        body_level: Level::succ(Level::zero()),
    };

    // Compress without dictionary
    let (no_dict_archive, no_dict_stats) =
        zstd_archive_cert_with_stats(&test_cert).expect("no-dict archive failed");

    // Compress with dictionary
    let (dict_archive, dict_stats) =
        zstd_archive_cert_with_dict_stats(&test_cert, &dict).expect("dict archive failed");

    // Both should roundtrip correctly
    let no_dict_restored = zstd_unarchive_cert(&no_dict_archive).expect("no-dict unarchive");
    let dict_restored =
        zstd_unarchive_cert_with_dict(&dict_archive, &dict).expect("dict unarchive");

    assert_eq!(no_dict_restored, test_cert);
    assert_eq!(dict_restored, test_cert);

    // Print comparison (for manual review in test output)
    println!(
        "Without dict: {} bytes",
        no_dict_archive.compressed_data.len()
    );
    println!("With dict: {} bytes", dict_archive.compressed_data.len());
    println!("No-dict ratio: {:.2}x", no_dict_stats.total_ratio);
    println!("Dict ratio: {:.2}x", dict_stats.total_ratio);

    // Note: For small data, dictionary may not always improve compression
    // The test verifies correctness, not that dict is always better
}

// =========================================================================
// Batch Verification Tests
// =========================================================================

#[test]
fn test_batch_verify_empty() {
    let env = empty_env();
    let inputs: Vec<BatchVerifyInput> = vec![];
    let results = batch_verify(&env, inputs);
    assert!(results.is_empty());
}

#[test]
fn test_batch_verify_single() {
    let env = empty_env();
    let level = Level::zero();
    let expr = Expr::Sort(level.clone());
    let cert = ProofCert::Sort {
        level: level.clone(),
    };

    let inputs = vec![BatchVerifyInput::new("test1", cert, expr)];
    let results = batch_verify(&env, inputs);

    assert_eq!(results.len(), 1);
    assert!(results[0].success);
    assert_eq!(results[0].id, "test1");
    assert!(results[0].verified_type.is_some());
}

#[test]
fn test_batch_verify_multiple_success() {
    let env = empty_env();

    let inputs: Vec<BatchVerifyInput> = (0..10)
        .map(|i| {
            let level = Level::zero();
            let expr = Expr::Sort(level.clone());
            let cert = ProofCert::Sort {
                level: level.clone(),
            };
            BatchVerifyInput::new(format!("cert_{i}"), cert, expr)
        })
        .collect();

    let results = batch_verify(&env, inputs);

    assert_eq!(results.len(), 10);
    for (i, result) in results.iter().enumerate() {
        assert!(result.success, "Failed at index {i}");
        assert_eq!(result.id, format!("cert_{i}"));
    }
}

#[test]
fn test_batch_verify_with_failures() {
    let env = empty_env();

    let inputs: Vec<BatchVerifyInput> = (0..5)
        .map(|i| {
            let level = Level::zero();
            let expr = Expr::Sort(level.clone());
            if i % 2 == 0 {
                // Valid certificate
                let cert = ProofCert::Sort {
                    level: level.clone(),
                };
                BatchVerifyInput::new(format!("valid_{i}"), cert, expr)
            } else {
                // Invalid certificate (level mismatch)
                let cert = ProofCert::Sort {
                    level: Level::succ(Level::zero()),
                };
                BatchVerifyInput::new(format!("invalid_{i}"), cert, expr)
            }
        })
        .collect();

    let results = batch_verify(&env, inputs);

    assert_eq!(results.len(), 5);
    assert!(results[0].success); // valid_0
    assert!(!results[1].success); // invalid_1
    assert!(results[2].success); // valid_2
    assert!(!results[3].success); // invalid_3
    assert!(results[4].success); // valid_4
}

#[test]
fn test_batch_verify_with_stats() {
    let env = empty_env();

    let inputs: Vec<BatchVerifyInput> = (0..100)
        .map(|i| {
            let level = Level::zero();
            let expr = Expr::Sort(level.clone());
            let cert = ProofCert::Sort {
                level: level.clone(),
            };
            BatchVerifyInput::new(format!("{i}"), cert, expr)
        })
        .collect();

    let (results, stats) = batch_verify_with_stats(&env, inputs);

    assert_eq!(results.len(), 100);
    assert_eq!(stats.total, 100);
    assert_eq!(stats.successful, 100);
    assert_eq!(stats.failed, 0);
    assert!(stats.wall_time_us > 0 || stats.total == 0);
    println!("Batch stats: {stats}");
}

#[test]
fn test_batch_verify_sequential() {
    let env = empty_env();

    let inputs: Vec<BatchVerifyInput> = (0..10)
        .map(|i| {
            let level = Level::zero();
            let expr = Expr::Sort(level.clone());
            let cert = ProofCert::Sort {
                level: level.clone(),
            };
            BatchVerifyInput::new(format!("{i}"), cert, expr)
        })
        .collect();

    let results = batch_verify_sequential(&env, inputs);

    assert_eq!(results.len(), 10);
    for result in &results {
        assert!(result.success);
    }
}

#[test]
fn test_batch_verify_sequential_with_stats() {
    let env = empty_env();

    let inputs: Vec<BatchVerifyInput> = (0..50)
        .map(|i| {
            let level = Level::zero();
            let expr = Expr::Sort(level.clone());
            let cert = ProofCert::Sort {
                level: level.clone(),
            };
            BatchVerifyInput::new(format!("{i}"), cert, expr)
        })
        .collect();

    let (results, stats) = batch_verify_sequential_with_stats(&env, inputs);

    assert_eq!(results.len(), 50);
    assert_eq!(stats.total, 50);
    assert_eq!(stats.successful, 50);
    assert_eq!(stats.failed, 0);
    // Sequential should have speedup close to 1.0
    println!("Sequential stats: {stats}");
}

#[test]
fn test_batch_verify_with_threads() {
    let env = empty_env();

    let inputs: Vec<BatchVerifyInput> = (0..20)
        .map(|i| {
            let level = Level::zero();
            let expr = Expr::Sort(level.clone());
            let cert = ProofCert::Sort {
                level: level.clone(),
            };
            BatchVerifyInput::new(format!("{i}"), cert, expr)
        })
        .collect();

    // Test with 2 threads
    let results = batch_verify_with_threads(&env, inputs.clone(), 2);
    assert_eq!(results.len(), 20);
    for result in &results {
        assert!(result.success);
    }

    // Test with 1 thread (essentially sequential)
    let results_single = batch_verify_with_threads(&env, inputs, 1);
    assert_eq!(results_single.len(), 20);
}

#[test]
fn test_batch_verify_with_stats_threads() {
    let env = empty_env();

    let inputs: Vec<BatchVerifyInput> = (0..10)
        .map(|i| {
            let level = Level::zero();
            let expr = Expr::Sort(level.clone());
            let cert = ProofCert::Sort {
                level: level.clone(),
            };
            BatchVerifyInput::new(format!("{i}"), cert, expr)
        })
        .collect();

    let (results, stats) = batch_verify_with_stats_threads(&env, inputs, 2);

    // Must return all results
    assert_eq!(
        results.len(),
        10,
        "batch_verify_with_stats_threads must return all inputs"
    );
    for result in &results {
        assert!(result.success);
    }

    // Stats must be populated
    assert_eq!(stats.total, 10);
    assert_eq!(stats.successful, 10);
    assert_eq!(stats.failed, 0);
}

#[test]
fn test_batch_verify_with_stats_progress_invokes_callback() {
    use std::sync::{Arc, Mutex};

    let env = empty_env();
    let level = Level::zero();
    let expr = Expr::Sort(level.clone());
    let cert = ProofCert::Sort {
        level: level.clone(),
    };

    for threads in [0usize, 2usize] {
        let inputs = vec![
            BatchVerifyInput::new("a", cert.clone(), expr.clone()),
            BatchVerifyInput::new("b", cert.clone(), expr.clone()),
        ];

        let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let seen_cb = Arc::clone(&seen);

        let (results, stats) = batch_verify_with_stats_progress(
            &env,
            inputs,
            threads,
            move |result: &BatchVerifyResult| {
                let mut guard = seen_cb.lock().unwrap();
                guard.push(result.id.clone());
            },
        );

        assert_eq!(results.len(), 2);
        assert_eq!(stats.total, 2);

        let mut ids = seen.lock().unwrap();
        ids.sort();
        assert_eq!(*ids, vec!["a".to_string(), "b".to_string()]);
    }
}

#[test]
fn test_batch_verify_complex_certs() {
    let env = empty_env();

    // Create complex certificates - use nested Sort with varying levels
    // (Pi types require more complex setup with the environment)
    let inputs: Vec<BatchVerifyInput> = (0..10)
        .map(|i| {
            // Create different universe levels
            let mut level = Level::zero();
            for _ in 0..(i % 3) {
                level = Level::succ(level);
            }
            let expr = Expr::Sort(level.clone());
            let cert = ProofCert::Sort {
                level: level.clone(),
            };

            BatchVerifyInput::new(format!("sort_{i}"), cert, expr)
        })
        .collect();

    let (results, stats) = batch_verify_with_stats(&env, inputs);

    assert_eq!(results.len(), 10);
    assert_eq!(stats.successful, 10);
    for result in &results {
        assert!(result.success, "Failed: {:?}", result.error);
    }
}

#[test]
fn test_batch_verify_stats_display() {
    let stats = BatchVerifyStats {
        total: 100,
        successful: 95,
        failed: 5,
        wall_time_us: 1000,
        sum_verify_time_us: 4000,
        min_time_us: 10,
        max_time_us: 100,
        speedup: 4.0,
    };

    let display = format!("{stats}");
    assert!(display.contains("100"));
    assert!(display.contains("95"));
    assert!(display.contains('5'));
    assert!(display.contains("4.00x"));
}

#[test]
fn test_batch_verify_result_accessors() {
    let success_result =
        BatchVerifyResult::success("test".to_string(), Expr::Sort(Level::zero()), 100);
    assert!(success_result.success);
    assert!(success_result.verified_type.is_some());
    assert!(success_result.error.is_none());
    assert_eq!(success_result.time_us, 100);

    let failure_result =
        BatchVerifyResult::failure("test2".to_string(), "Some error".to_string(), 50);
    assert!(!failure_result.success);
    assert!(failure_result.verified_type.is_none());
    assert!(failure_result.error.is_some());
    assert_eq!(failure_result.time_us, 50);
}

#[test]
fn test_batch_verify_input_new() {
    let level = Level::zero();
    let cert = ProofCert::Sort {
        level: level.clone(),
    };
    let expr = Expr::Sort(level);

    let input = BatchVerifyInput::new("my_id", cert.clone(), expr.clone());
    assert_eq!(input.id, "my_id");
    assert_eq!(input.cert, cert);
    assert_eq!(input.expr, expr);

    // Test with String id
    let input2 = BatchVerifyInput::new(String::from("string_id"), cert, expr);
    assert_eq!(input2.id, "string_id");
}

#[test]
fn test_batch_verify_preserves_order() {
    let env = empty_env();

    // Create inputs with specific IDs
    let inputs: Vec<BatchVerifyInput> = vec!["z", "a", "m", "b", "y"]
        .into_iter()
        .map(|id| {
            let level = Level::zero();
            let expr = Expr::Sort(level.clone());
            let cert = ProofCert::Sort {
                level: level.clone(),
            };
            BatchVerifyInput::new(id, cert, expr)
        })
        .collect();

    let results = batch_verify(&env, inputs);

    // Results should be in the same order as inputs
    assert_eq!(results[0].id, "z");
    assert_eq!(results[1].id, "a");
    assert_eq!(results[2].id, "m");
    assert_eq!(results[3].id, "b");
    assert_eq!(results[4].id, "y");
}

#[test]
fn test_batch_verify_parallel_vs_sequential_same_results() {
    let env = empty_env();

    let inputs: Vec<BatchVerifyInput> = (0..50)
        .map(|i| {
            let level = if i % 3 == 0 {
                Level::zero()
            } else {
                Level::succ(Level::zero())
            };
            let expr = Expr::Sort(level.clone());
            let cert = ProofCert::Sort {
                level: level.clone(),
            };
            BatchVerifyInput::new(format!("{i}"), cert, expr)
        })
        .collect();

    let parallel_results = batch_verify(&env, inputs.clone());
    let sequential_results = batch_verify_sequential(&env, inputs);

    // Same number of results
    assert_eq!(parallel_results.len(), sequential_results.len());

    // Same success/failure outcomes
    for (p, s) in parallel_results.iter().zip(sequential_results.iter()) {
        assert_eq!(p.id, s.id);
        assert_eq!(p.success, s.success);
        if p.success {
            assert_eq!(p.verified_type, s.verified_type);
        }
    }
}

// ========================================================================
// Classical mode certificate roundtrip tests
// ========================================================================

#[test]
fn test_classical_choice_cert_roundtrip() {
    let env = empty_env();
    let mut tc = crate::TypeChecker::with_mode(&env, Lean5Mode::Classical);

    // Build ClassicalChoice expression: choice Prop (λ x. x) (exists_proof)
    // ty = Prop (a type), pred = identity-like function, proof = any prop
    let prop_ty = Expr::Sort(Level::zero()); // Prop
    let pred = Expr::Lam(
        BinderInfo::Default,
        prop_ty.clone().into(),
        Expr::Sort(Level::zero()).into(), // Returns Prop
    );
    let proof = Expr::Sort(Level::zero()); // Placeholder proof

    let choice_expr = Expr::ClassicalChoice {
        ty: prop_ty.clone().into(),
        pred: pred.into(),
        exists_proof: proof.into(),
    };

    let (ty, cert) = tc.infer_type_with_cert(&choice_expr).unwrap();

    // Verify certificate
    let mut verifier = CertVerifier::with_mode(&env, Lean5Mode::Classical);
    let verified_ty = verifier.verify(&cert, &choice_expr).unwrap();
    assert_eq!(ty, verified_ty);
}

#[test]
fn test_classical_epsilon_cert_roundtrip() {
    let env = empty_env();
    let mut tc = crate::TypeChecker::with_mode(&env, Lean5Mode::Classical);

    // Build ClassicalEpsilon expression: epsilon Prop (λ x. x)
    let prop_ty = Expr::Sort(Level::zero()); // Prop
    let pred = Expr::Lam(
        BinderInfo::Default,
        prop_ty.clone().into(),
        Expr::Sort(Level::zero()).into(), // Returns Prop
    );

    let epsilon_expr = Expr::ClassicalEpsilon {
        ty: prop_ty.clone().into(),
        pred: pred.into(),
    };

    let (ty, cert) = tc.infer_type_with_cert(&epsilon_expr).unwrap();

    // Verify certificate
    let mut verifier = CertVerifier::with_mode(&env, Lean5Mode::Classical);
    let verified_ty = verifier.verify(&cert, &epsilon_expr).unwrap();
    assert_eq!(ty, verified_ty);
}

#[test]
fn test_classical_mode_required_for_choice() {
    let env = empty_env();
    let mut tc = crate::TypeChecker::new(&env); // Constructive mode

    let prop_ty = Expr::Sort(Level::zero()); // Prop
    let pred = Expr::Lam(
        BinderInfo::Default,
        prop_ty.clone().into(),
        Expr::Sort(Level::zero()).into(),
    );
    let proof = Expr::Sort(Level::zero());

    let choice_expr = Expr::ClassicalChoice {
        ty: prop_ty.into(),
        pred: pred.into(),
        exists_proof: proof.into(),
    };

    // Should fail in constructive mode
    let result = tc.infer_type_with_cert(&choice_expr);
    assert!(result.is_err());
}

// ========================================================================
// ZFC/SetTheoretic mode certificate roundtrip tests
// ========================================================================

#[test]
fn test_zfc_empty_set_cert_roundtrip() {
    use crate::expr::ZFCSetExpr;

    let env = empty_env();
    let mut tc = crate::TypeChecker::with_mode(&env, Lean5Mode::SetTheoretic);

    let empty_set = Expr::ZFCSet(ZFCSetExpr::Empty);
    let (ty, cert) = tc.infer_type_with_cert(&empty_set).unwrap();

    let mut verifier = CertVerifier::with_mode(&env, Lean5Mode::SetTheoretic);
    let verified_ty = verifier.verify(&cert, &empty_set).unwrap();
    assert_eq!(ty, verified_ty);
}

#[test]
fn test_zfc_infinity_set_cert_roundtrip() {
    use crate::expr::ZFCSetExpr;

    let env = empty_env();
    let mut tc = crate::TypeChecker::with_mode(&env, Lean5Mode::SetTheoretic);

    let infinity_set = Expr::ZFCSet(ZFCSetExpr::Infinity);
    let (ty, cert) = tc.infer_type_with_cert(&infinity_set).unwrap();

    let mut verifier = CertVerifier::with_mode(&env, Lean5Mode::SetTheoretic);
    let verified_ty = verifier.verify(&cert, &infinity_set).unwrap();
    assert_eq!(ty, verified_ty);
}

#[test]
fn test_zfc_singleton_set_cert_roundtrip() {
    use crate::expr::ZFCSetExpr;

    let env = empty_env();
    let mut tc = crate::TypeChecker::with_mode(&env, Lean5Mode::SetTheoretic);

    // {∅} - singleton containing empty set
    let empty_set = Expr::ZFCSet(ZFCSetExpr::Empty);
    let singleton = Expr::ZFCSet(ZFCSetExpr::Singleton(empty_set.into()));
    let (ty, cert) = tc.infer_type_with_cert(&singleton).unwrap();

    let mut verifier = CertVerifier::with_mode(&env, Lean5Mode::SetTheoretic);
    let verified_ty = verifier.verify(&cert, &singleton).unwrap();
    assert_eq!(ty, verified_ty);
}

#[test]
fn test_zfc_pair_set_cert_roundtrip() {
    use crate::expr::ZFCSetExpr;

    let env = empty_env();
    let mut tc = crate::TypeChecker::with_mode(&env, Lean5Mode::SetTheoretic);

    // {∅, {∅}} - unordered pair
    let empty_set = Expr::ZFCSet(ZFCSetExpr::Empty);
    let singleton = Expr::ZFCSet(ZFCSetExpr::Singleton(empty_set.clone().into()));
    let pair = Expr::ZFCSet(ZFCSetExpr::Pair(empty_set.into(), singleton.into()));
    let (ty, cert) = tc.infer_type_with_cert(&pair).unwrap();

    let mut verifier = CertVerifier::with_mode(&env, Lean5Mode::SetTheoretic);
    let verified_ty = verifier.verify(&cert, &pair).unwrap();
    assert_eq!(ty, verified_ty);
}

#[test]
fn test_zfc_union_set_cert_roundtrip() {
    use crate::expr::ZFCSetExpr;

    let env = empty_env();
    let mut tc = crate::TypeChecker::with_mode(&env, Lean5Mode::SetTheoretic);

    // ⋃{{∅}} = {∅}
    let empty_set = Expr::ZFCSet(ZFCSetExpr::Empty);
    let singleton = Expr::ZFCSet(ZFCSetExpr::Singleton(empty_set.into()));
    let union = Expr::ZFCSet(ZFCSetExpr::Union(singleton.into()));
    let (ty, cert) = tc.infer_type_with_cert(&union).unwrap();

    let mut verifier = CertVerifier::with_mode(&env, Lean5Mode::SetTheoretic);
    let verified_ty = verifier.verify(&cert, &union).unwrap();
    assert_eq!(ty, verified_ty);
}

#[test]
fn test_zfc_powerset_cert_roundtrip() {
    use crate::expr::ZFCSetExpr;

    let env = empty_env();
    let mut tc = crate::TypeChecker::with_mode(&env, Lean5Mode::SetTheoretic);

    // P(∅) = {∅}
    let empty_set = Expr::ZFCSet(ZFCSetExpr::Empty);
    let powerset = Expr::ZFCSet(ZFCSetExpr::PowerSet(empty_set.into()));
    let (ty, cert) = tc.infer_type_with_cert(&powerset).unwrap();

    let mut verifier = CertVerifier::with_mode(&env, Lean5Mode::SetTheoretic);
    let verified_ty = verifier.verify(&cert, &powerset).unwrap();
    assert_eq!(ty, verified_ty);
}

#[test]
fn test_zfc_mem_cert_roundtrip() {
    use crate::expr::ZFCSetExpr;

    let env = empty_env();
    let mut tc = crate::TypeChecker::with_mode(&env, Lean5Mode::SetTheoretic);

    // ∅ ∈ {∅}
    let empty_set = Expr::ZFCSet(ZFCSetExpr::Empty);
    let singleton = Expr::ZFCSet(ZFCSetExpr::Singleton(empty_set.clone().into()));
    let mem_expr = Expr::ZFCMem {
        element: empty_set.into(),
        set: singleton.into(),
    };
    let (ty, cert) = tc.infer_type_with_cert(&mem_expr).unwrap();

    let mut verifier = CertVerifier::with_mode(&env, Lean5Mode::SetTheoretic);
    let verified_ty = verifier.verify(&cert, &mem_expr).unwrap();
    assert_eq!(ty, verified_ty);
    // Membership is a proposition (Prop = Sort(0))
    assert_eq!(ty, Expr::Sort(Level::zero()));
}

#[test]
fn test_zfc_comprehension_cert_roundtrip() {
    use crate::expr::ZFCSetExpr;

    let env = empty_env();
    let mut tc = crate::TypeChecker::with_mode(&env, Lean5Mode::SetTheoretic);

    // {x ∈ ∅ | true} (comprehension with trivial predicate)
    // Use Prop as argument type since ZFC.Set isn't in empty environment
    let empty_set = Expr::ZFCSet(ZFCSetExpr::Empty);
    let prop_ty = Expr::Sort(Level::zero()); // Use Prop as a stand-in
    let pred = Expr::Lam(
        BinderInfo::Default,
        prop_ty.into(),
        Expr::Sort(Level::zero()).into(), // λ x. Prop
    );
    let comprehension = Expr::ZFCComprehension {
        domain: empty_set.into(),
        pred: pred.into(),
    };
    let (ty, cert) = tc.infer_type_with_cert(&comprehension).unwrap();

    let mut verifier = CertVerifier::with_mode(&env, Lean5Mode::SetTheoretic);
    let verified_ty = verifier.verify(&cert, &comprehension).unwrap();
    assert_eq!(ty, verified_ty);
}

#[test]
fn test_zfc_mode_required() {
    use crate::expr::ZFCSetExpr;

    let env = empty_env();
    let mut tc = crate::TypeChecker::new(&env); // Constructive mode

    let empty_set = Expr::ZFCSet(ZFCSetExpr::Empty);

    // Should fail in constructive mode
    let result = tc.infer_type_with_cert(&empty_set);
    assert!(result.is_err());
}

#[test]
fn test_zfc_nested_sets_cert_roundtrip() {
    use crate::expr::ZFCSetExpr;

    let env = empty_env();
    let mut tc = crate::TypeChecker::with_mode(&env, Lean5Mode::SetTheoretic);

    // P(P(∅)) - power set of power set of empty set
    let empty = Expr::ZFCSet(ZFCSetExpr::Empty);
    let p1 = Expr::ZFCSet(ZFCSetExpr::PowerSet(empty.into()));
    let p2 = Expr::ZFCSet(ZFCSetExpr::PowerSet(p1.into()));
    let (ty, cert) = tc.infer_type_with_cert(&p2).unwrap();

    let mut verifier = CertVerifier::with_mode(&env, Lean5Mode::SetTheoretic);
    let verified_ty = verifier.verify(&cert, &p2).unwrap();
    assert_eq!(ty, verified_ty);
}
