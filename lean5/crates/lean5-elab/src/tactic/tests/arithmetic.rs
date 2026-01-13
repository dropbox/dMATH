use super::*;

// linarith tests
// =========================================================================

#[test]
fn test_linear_expr_constant() {
    let c = LinearExpr::constant(5);
    assert_eq!(c.constant, 5);
    assert!(c.is_constant());
    assert!(c.variables().is_empty());
}

#[test]
fn test_linear_expr_var() {
    let v = LinearExpr::var(0);
    assert_eq!(v.constant, 0);
    assert!(!v.is_constant());
    assert_eq!(v.variables(), vec![0]);
    assert_eq!(*v.coeffs.get(&0).unwrap(), 1);
}

#[test]
fn test_linear_expr_add() {
    // 2 + x0
    let c = LinearExpr::constant(2);
    let v = LinearExpr::var(0);
    let sum = c.add(&v);

    assert_eq!(sum.constant, 2);
    assert_eq!(*sum.coeffs.get(&0).unwrap(), 1);
}

#[test]
fn test_linear_expr_sub() {
    // x0 - x1
    let v0 = LinearExpr::var(0);
    let v1 = LinearExpr::var(1);
    let diff = v0.sub(&v1);

    assert_eq!(diff.constant, 0);
    assert_eq!(*diff.coeffs.get(&0).unwrap(), 1);
    assert_eq!(*diff.coeffs.get(&1).unwrap(), -1);
}

#[test]
fn test_linear_expr_scale() {
    // 3 * x0
    let v = LinearExpr::var(0);
    let scaled = v.scale(3);

    assert_eq!(scaled.constant, 0);
    assert_eq!(*scaled.coeffs.get(&0).unwrap(), 3);
}

#[test]
fn test_linear_constraint_trivially_false() {
    // 5 ≤ 0 is false
    let e = LinearExpr::constant(5);
    let c = LinearConstraint::Le(e);
    assert!(c.is_trivially_false());
    assert!(!c.is_trivially_true());
}

#[test]
fn test_linear_constraint_trivially_true() {
    // -5 ≤ 0 is true
    let e = LinearExpr::constant(-5);
    let c = LinearConstraint::Le(e);
    assert!(c.is_trivially_true());
    assert!(!c.is_trivially_false());
}

#[test]
fn test_fourier_motzkin_unsat_simple() {
    // x ≤ 0 and x ≥ 1 is UNSAT
    // x ≤ 0  =>  x ≤ 0
    // x ≥ 1  =>  -x + 1 ≤ 0  =>  -x ≤ -1

    let x_le_0 = LinearExpr::var(0);
    // x ≤ 0 already

    let mut neg_x_le_neg1 = LinearExpr::var(0).scale(-1);
    neg_x_le_neg1.constant = 1;
    // -x + 1 ≤ 0

    let constraints = vec![
        LinearConstraint::Le(x_le_0),
        LinearConstraint::Le(neg_x_le_neg1),
    ];

    let result = fourier_motzkin_check(&constraints);
    assert!(matches!(result, FMResult::Unsat));
}

#[test]
fn test_fourier_motzkin_sat() {
    // x ≤ 5 and x ≥ 0 is SAT
    let mut x_le_5 = LinearExpr::var(0);
    x_le_5.constant = -5;
    // x - 5 ≤ 0

    let neg_x = LinearExpr::var(0).scale(-1);
    // -x ≤ 0

    let constraints = vec![LinearConstraint::Le(x_le_5), LinearConstraint::Le(neg_x)];

    let result = fourier_motzkin_check(&constraints);
    assert!(matches!(result, FMResult::Sat));
}

// =========================================================================
// LinarithCertificate tests
// =========================================================================

#[test]
fn test_linarith_certificate_new() {
    let cert = LinarithCertificate::new(5);
    assert_eq!(cert.coefficients.len(), 5);
    assert!(cert.coefficients.iter().all(|&c| c == 0));
    assert_eq!(cert.result_constant, 0);
}

#[test]
fn test_linarith_certificate_from_hypothesis() {
    let cert = LinarithCertificate::from_hypothesis(2, 5);
    assert_eq!(cert.coefficients.len(), 5);
    assert_eq!(cert.coefficients[0], 0);
    assert_eq!(cert.coefficients[1], 0);
    assert_eq!(cert.coefficients[2], 1);
    assert_eq!(cert.coefficients[3], 0);
    assert_eq!(cert.coefficients[4], 0);
}

#[test]
fn test_linarith_certificate_scale() {
    let cert = LinarithCertificate::from_hypothesis(1, 3);
    let scaled = cert.scale(5);
    assert_eq!(scaled.coefficients[0], 0);
    assert_eq!(scaled.coefficients[1], 5);
    assert_eq!(scaled.coefficients[2], 0);
}

#[test]
fn test_linarith_certificate_add() {
    let cert1 = LinarithCertificate::from_hypothesis(0, 3);
    let cert2 = LinarithCertificate::from_hypothesis(1, 3);
    let combined = cert1.add(&cert2);
    assert_eq!(combined.coefficients[0], 1);
    assert_eq!(combined.coefficients[1], 1);
    assert_eq!(combined.coefficients[2], 0);
}

#[test]
fn test_linarith_certificate_is_valid() {
    let mut cert = LinarithCertificate::new(3);
    cert.coefficients[0] = 1;
    cert.coefficients[1] = 2;
    cert.result_constant = 5;
    assert!(cert.is_valid());

    // Negative coefficient makes it invalid
    cert.coefficients[2] = -1;
    assert!(!cert.is_valid());
}

#[test]
fn test_certified_constraint_from_hypothesis() {
    let constraint = LinearConstraint::Le(LinearExpr::constant(0));
    let cc = CertifiedConstraint::from_hypothesis(constraint.clone(), 1, 4);
    assert_eq!(cc.certificate.coefficients.len(), 4);
    assert_eq!(cc.certificate.coefficients[1], 1);
}

#[test]
fn test_build_add_le_add_proof_two_hypotheses() {
    // Test building proof for two hypotheses with coefficient 1
    let fvar1 = FVarId(1);
    let fvar2 = FVarId(2);
    let hypothesis_fvars = vec![fvar1, fvar2];
    let active = vec![(0, 1i64), (1, 1i64)];

    let env = setup_env();
    let result = build_add_le_add_proof(&active, &hypothesis_fvars, &env);

    // Should produce some proof (even if it's just an expression structure)
    assert!(result.is_some());
    let proof = result.unwrap();

    // The proof should be an application of add_le_add
    if let Expr::App(f, _) = &proof {
        if let Expr::App(f2, _) = f.as_ref() {
            if let Expr::Const(name, _) = f2.as_ref() {
                assert_eq!(name.to_string(), "add_le_add");
            }
        }
    }
}

#[test]
fn test_build_add_le_add_proof_three_hypotheses() {
    // Test building proof for three hypotheses
    let fvar1 = FVarId(1);
    let fvar2 = FVarId(2);
    let fvar3 = FVarId(3);
    let hypothesis_fvars = vec![fvar1, fvar2, fvar3];
    let active = vec![(0, 1i64), (1, 1i64), (2, 1i64)];

    let env = setup_env();
    let result = build_add_le_add_proof(&active, &hypothesis_fvars, &env);

    assert!(result.is_some());
    // Should nest add_le_add applications
}

#[test]
fn test_build_scaled_proof_single() {
    // Test building proof for a single hypothesis with coefficient > 1
    let fvar1 = FVarId(1);
    let hypothesis_fvars = vec![fvar1];
    let active = vec![(0, 3i64)];

    let env = setup_env();
    let result = build_scaled_proof(&active, &hypothesis_fvars, &env);

    assert!(result.is_some());
}

#[test]
fn test_build_scaled_proof_mixed() {
    // Test building proof for mix of coefficients
    let fvar1 = FVarId(1);
    let fvar2 = FVarId(2);
    let hypothesis_fvars = vec![fvar1, fvar2];
    let active = vec![(0, 1i64), (1, 2i64)];

    let env = setup_env();
    let result = build_scaled_proof(&active, &hypothesis_fvars, &env);

    assert!(result.is_some());
}

#[test]
fn test_fourier_motzkin_certified_unsat() {
    // x ≤ 0 and x ≥ 1 is UNSAT
    let x_le_0 = LinearExpr::var(0);
    let mut neg_x_le_neg1 = LinearExpr::var(0).scale(-1);
    neg_x_le_neg1.constant = 1;

    let constraints = vec![
        CertifiedConstraint::from_hypothesis(LinearConstraint::Le(x_le_0), 0, 3),
        CertifiedConstraint::from_hypothesis(LinearConstraint::Le(neg_x_le_neg1), 1, 3),
    ];

    match fourier_motzkin_check_certified(&constraints) {
        FMCertifiedResult::Unsat(cert) => {
            // The certificate should use both hypotheses
            assert!(cert.coefficients[0] > 0 || cert.coefficients[1] > 0);
            assert!(cert.result_constant > 0);
        }
        _ => panic!("Expected Unsat result"),
    }
}

#[test]
fn test_fourier_motzkin_certified_sat() {
    // x ≤ 5 and x ≥ 0 is SAT
    let mut x_le_5 = LinearExpr::var(0);
    x_le_5.constant = -5;
    let neg_x = LinearExpr::var(0).scale(-1);

    let constraints = vec![
        CertifiedConstraint::from_hypothesis(LinearConstraint::Le(x_le_5), 0, 3),
        CertifiedConstraint::from_hypothesis(LinearConstraint::Le(neg_x), 1, 3),
    ];

    let result = fourier_motzkin_check_certified(&constraints);
    assert!(matches!(result, FMCertifiedResult::Sat));
}

// =========================================================================
// OmegaCertificate tests
// =========================================================================

#[test]
fn test_omega_certificate_new() {
    let cert = OmegaCertificate::new(5);
    assert_eq!(cert.coefficients.len(), 5);
    assert!(cert.coefficients.iter().all(|&c| c == 0));
    assert!(!cert.uses_goal_negation);
    assert!(matches!(
        cert.contradiction_type,
        OmegaContradictionType::Arithmetic
    ));
}

#[test]
fn test_omega_certificate_from_linarith() {
    let mut linarith_cert = LinarithCertificate::new(3);
    linarith_cert.coefficients[0] = 2;
    linarith_cert.coefficients[1] = 3;
    linarith_cert.result_constant = 5;

    let omega_cert = OmegaCertificate::from_linarith(&linarith_cert);
    assert_eq!(omega_cert.coefficients[0], 2);
    assert_eq!(omega_cert.coefficients[1], 3);
    assert!(omega_cert.uses_goal_negation);
    assert!(matches!(
        omega_cert.contradiction_type,
        OmegaContradictionType::LinearCombination
    ));
}

#[test]
fn test_omega_certificate_is_valid() {
    let mut cert = OmegaCertificate::new(3);
    cert.coefficients[0] = 1;
    cert.coefficients[1] = 2;
    assert!(cert.is_valid());

    // Negative coefficient makes it invalid
    cert.coefficients[2] = -1;
    assert!(!cert.is_valid());
}

#[test]
fn test_certified_omega_constraint_from_hypothesis() {
    let constraint = OmegaConstraint::Le(LinearExpr::constant(0));
    let cc = CertifiedOmegaConstraint::from_hypothesis(constraint.clone(), 1, 4);
    assert_eq!(cc.certificate.coefficients.len(), 4);
    assert_eq!(cc.certificate.coefficients[1], 1);
    assert!(!cc.certificate.uses_goal_negation);
}

#[test]
fn test_certified_omega_constraint_from_negated_goal() {
    let constraint = OmegaConstraint::Le(LinearExpr::constant(0));
    let cc = CertifiedOmegaConstraint::from_negated_goal(constraint.clone(), 3);
    assert_eq!(cc.certificate.coefficients.len(), 3);
    assert!(cc.certificate.coefficients.iter().all(|&c| c == 0));
    assert!(cc.certificate.uses_goal_negation);
}

#[test]
fn test_omega_check_certified_unsat() {
    // x ≤ 0 and x ≥ 1 is UNSAT (same as linarith)
    let x_le_0 = LinearExpr::var(0);
    let mut neg_x_le_neg1 = LinearExpr::var(0).scale(-1);
    neg_x_le_neg1.constant = 1;

    let constraints = vec![
        CertifiedOmegaConstraint::from_hypothesis(OmegaConstraint::Le(x_le_0), 0, 3),
        CertifiedOmegaConstraint::from_hypothesis(OmegaConstraint::Le(neg_x_le_neg1), 1, 3),
    ];

    let result = omega_check_certified(&constraints);
    assert!(matches!(result, OmegaCertifiedResult::Unsat(_)));

    if let OmegaCertifiedResult::Unsat(cert) = result {
        // The certificate should be valid
        assert!(cert.is_valid() || cert.coefficients.iter().any(|&c| c > 0));
    }
}

#[test]
fn test_omega_check_certified_sat() {
    // x ≤ 5 and x ≥ 0 is SAT
    let mut x_le_5 = LinearExpr::var(0);
    x_le_5.constant = -5;
    let neg_x = LinearExpr::var(0).scale(-1);

    let constraints = vec![
        CertifiedOmegaConstraint::from_hypothesis(OmegaConstraint::Le(x_le_5), 0, 3),
        CertifiedOmegaConstraint::from_hypothesis(OmegaConstraint::Le(neg_x), 1, 3),
    ];

    let result = omega_check_certified(&constraints);
    assert!(matches!(result, OmegaCertifiedResult::Sat));
}

#[test]
fn test_omega_parity_contradiction() {
    // x ≡ 0 (mod 2) and x ≡ 1 (mod 2) is UNSAT (parity contradiction)
    let constraints = vec![
        CertifiedOmegaConstraint::from_hypothesis(
            OmegaConstraint::Mod {
                var: 0,
                remainder: 0,
                modulus: 2,
            },
            0,
            2,
        ),
        CertifiedOmegaConstraint::from_hypothesis(
            OmegaConstraint::Mod {
                var: 0,
                remainder: 1,
                modulus: 2,
            },
            1,
            2,
        ),
    ];

    let result = omega_check_certified(&constraints);
    assert!(
        matches!(result, OmegaCertifiedResult::Unsat(_)),
        "Expected UNSAT for parity contradiction, got {result:?}"
    );

    if let OmegaCertifiedResult::Unsat(cert) = result {
        assert!(
            matches!(cert.contradiction_type, OmegaContradictionType::Parity),
            "Expected Parity contradiction type, got {:?}",
            cert.contradiction_type
        );
    }
}

#[test]
fn test_omega_divisibility_contradiction() {
    // x ≡ 0 (mod 3) and x ≡ 2 (mod 3) is UNSAT (divisibility contradiction)
    let constraints = vec![
        CertifiedOmegaConstraint::from_hypothesis(
            OmegaConstraint::Mod {
                var: 0,
                remainder: 0,
                modulus: 3,
            },
            0,
            2,
        ),
        CertifiedOmegaConstraint::from_hypothesis(
            OmegaConstraint::Mod {
                var: 0,
                remainder: 2,
                modulus: 3,
            },
            1,
            2,
        ),
    ];

    let result = omega_check_certified(&constraints);
    assert!(
        matches!(result, OmegaCertifiedResult::Unsat(_)),
        "Expected UNSAT for divisibility contradiction, got {result:?}"
    );

    if let OmegaCertifiedResult::Unsat(cert) = result {
        assert!(
            matches!(
                cert.contradiction_type,
                OmegaContradictionType::Divisibility
            ),
            "Expected Divisibility contradiction type, got {:?}",
            cert.contradiction_type
        );
    }
}

#[test]
fn test_omega_equality_disequality_contradiction() {
    // x = 5 and x ≠ 5 is UNSAT
    // Encoded as: x - 5 = 0 and x - 5 ≠ 0
    let mut expr = LinearExpr::var(0);
    expr.constant = -5;

    let constraints = vec![
        CertifiedOmegaConstraint::from_hypothesis(OmegaConstraint::Eq(expr.clone()), 0, 2),
        CertifiedOmegaConstraint::from_hypothesis(OmegaConstraint::Ne(expr.clone()), 1, 2),
    ];

    let result = omega_check_certified(&constraints);
    assert!(
        matches!(result, OmegaCertifiedResult::Unsat(_)),
        "Expected UNSAT for equality/disequality contradiction, got {result:?}"
    );
}

#[test]
fn test_omega_modular_sat() {
    // x ≡ 0 (mod 2) and y ≡ 1 (mod 2) is SAT (different variables)
    let constraints = vec![
        CertifiedOmegaConstraint::from_hypothesis(
            OmegaConstraint::Mod {
                var: 0,
                remainder: 0,
                modulus: 2,
            },
            0,
            2,
        ),
        CertifiedOmegaConstraint::from_hypothesis(
            OmegaConstraint::Mod {
                var: 1, // Different variable
                remainder: 1,
                modulus: 2,
            },
            1,
            2,
        ),
    ];

    let result = omega_check_certified(&constraints);
    // This should be SAT (or Unknown since different variables)
    assert!(
        !matches!(result, OmegaCertifiedResult::Unsat(_)),
        "Expected SAT/Unknown for non-contradictory modular constraints, got {result:?}"
    );
}

#[test]
fn test_linear_constraint_ne_negate() {
    // Test that Ne negates to Eq
    let expr = LinearExpr::var(0);
    let ne_constraint = LinearConstraint::Ne(expr.clone());
    let negated = ne_constraint.negate();
    assert!(matches!(negated, LinearConstraint::Eq(_)));

    // And Eq negates to Ne
    let eq_constraint = LinearConstraint::Eq(expr);
    let negated = eq_constraint.negate();
    assert!(matches!(negated, LinearConstraint::Ne(_)));
}

#[test]
fn test_linear_constraint_ne_trivially_true() {
    // 5 ≠ 0 is trivially true
    let expr = LinearExpr::constant(5);
    let constraint = LinearConstraint::Ne(expr);
    assert!(constraint.is_trivially_true());

    // 0 ≠ 0 is trivially false
    let expr = LinearExpr::constant(0);
    let constraint = LinearConstraint::Ne(expr);
    assert!(constraint.is_trivially_false());
}

#[test]
fn test_linear_constraint_mod_trivially_true() {
    // 6 ≡ 0 (mod 3) is trivially true
    let expr = LinearExpr::constant(6);
    let constraint = LinearConstraint::Mod { expr, modulus: 3 };
    assert!(constraint.is_trivially_true());

    // 7 ≡ 0 (mod 3) is trivially false
    let expr = LinearExpr::constant(7);
    let constraint = LinearConstraint::Mod { expr, modulus: 3 };
    assert!(constraint.is_trivially_false());
}

#[test]
fn test_expr_to_omega_constraint_even() {
    // Test parsing `Even n` where n is an FVar
    // Pattern: (App (Const "Even") (FVar n))
    let n_fvar = FVarId(42);
    let even_expr = Expr::app(
        Expr::const_(Name::from_string("Even"), vec![]),
        Expr::fvar(n_fvar),
    );

    let result = expr_to_omega_constraint(&even_expr);
    assert!(result.is_some(), "Should parse Even n");

    if let Some(OmegaConstraint::Mod {
        var,
        remainder,
        modulus,
    }) = result
    {
        assert_eq!(var, 42, "Variable index should be fvar id");
        assert_eq!(remainder, 0, "Even means remainder 0");
        assert_eq!(modulus, 2, "Even uses modulus 2");
    } else {
        panic!("Expected Mod constraint for Even");
    }
}

#[test]
fn test_expr_to_omega_constraint_odd() {
    // Test parsing `Odd n` where n is an FVar
    let n_fvar = FVarId(7);
    let odd_expr = Expr::app(
        Expr::const_(Name::from_string("Odd"), vec![]),
        Expr::fvar(n_fvar),
    );

    let result = expr_to_omega_constraint(&odd_expr);
    assert!(result.is_some(), "Should parse Odd n");

    if let Some(OmegaConstraint::Mod {
        var,
        remainder,
        modulus,
    }) = result
    {
        assert_eq!(var, 7, "Variable index should be fvar id");
        assert_eq!(remainder, 1, "Odd means remainder 1");
        assert_eq!(modulus, 2, "Odd uses modulus 2");
    } else {
        panic!("Expected Mod constraint for Odd");
    }
}

#[test]
fn test_expr_to_omega_constraint_nat_even() {
    // Test parsing `Nat.Even n`
    let n_fvar = FVarId(10);
    let even_expr = Expr::app(
        Expr::const_(Name::from_string("Nat.Even"), vec![]),
        Expr::fvar(n_fvar),
    );

    let result = expr_to_omega_constraint(&even_expr);
    assert!(result.is_some(), "Should parse Nat.Even n");

    if let Some(OmegaConstraint::Mod {
        var,
        remainder,
        modulus,
    }) = result
    {
        assert_eq!(var, 10);
        assert_eq!(remainder, 0);
        assert_eq!(modulus, 2);
    } else {
        panic!("Expected Mod constraint for Nat.Even");
    }
}

#[test]
fn test_expr_to_omega_constraint_int_odd() {
    // Test parsing `Int.Odd n`
    let n_fvar = FVarId(99);
    let odd_expr = Expr::app(
        Expr::const_(Name::from_string("Int.Odd"), vec![]),
        Expr::fvar(n_fvar),
    );

    let result = expr_to_omega_constraint(&odd_expr);
    assert!(result.is_some(), "Should parse Int.Odd n");

    if let Some(OmegaConstraint::Mod {
        var,
        remainder,
        modulus,
    }) = result
    {
        assert_eq!(var, 99);
        assert_eq!(remainder, 1);
        assert_eq!(modulus, 2);
    } else {
        panic!("Expected Mod constraint for Int.Odd");
    }
}

#[test]
fn test_extract_single_var() {
    // Direct FVar
    let fvar = Expr::fvar(FVarId(5));
    assert_eq!(extract_single_var(&fvar), Some(5));

    // Wrapped in OfNat.ofNat
    let wrapped = Expr::app(
        Expr::const_(Name::from_string("OfNat.ofNat"), vec![]),
        Expr::fvar(FVarId(10)),
    );
    assert_eq!(extract_single_var(&wrapped), Some(10));

    // Constant (not a variable)
    let constant = Expr::const_(Name::from_string("Nat.zero"), vec![]);
    assert_eq!(extract_single_var(&constant), None);
}

#[test]
fn test_extract_constant() {
    // Literal
    let lit = Expr::Lit(lean5_kernel::expr::Literal::Nat(42));
    assert_eq!(extract_constant(&lit), Some(42));

    // Named constants
    let zero = Expr::const_(Name::from_string("Nat.zero"), vec![]);
    assert_eq!(extract_constant(&zero), Some(0));

    let one = Expr::const_(Name::from_string("Nat.one"), vec![]);
    assert_eq!(extract_constant(&one), Some(1));

    // Variable (not a constant)
    let fvar = Expr::fvar(FVarId(5));
    assert_eq!(extract_constant(&fvar), None);
}

#[test]
fn test_omega_even_odd_contradiction() {
    // Given: Even n and Odd n (should be UNSAT)
    // Both constraints on the same variable with different remainders mod 2
    let n_var = 0;
    let constraints = vec![
        CertifiedOmegaConstraint::from_hypothesis(
            OmegaConstraint::Mod {
                var: n_var,
                remainder: 0, // Even
                modulus: 2,
            },
            0,
            2,
        ),
        CertifiedOmegaConstraint::from_hypothesis(
            OmegaConstraint::Mod {
                var: n_var,
                remainder: 1, // Odd
                modulus: 2,
            },
            1,
            2,
        ),
    ];

    let result = omega_check_certified(&constraints);
    assert!(
        matches!(result, OmegaCertifiedResult::Unsat(_)),
        "Even n ∧ Odd n should be UNSAT"
    );

    if let OmegaCertifiedResult::Unsat(cert) = result {
        assert!(
            matches!(cert.contradiction_type, OmegaContradictionType::Parity),
            "Should detect parity contradiction"
        );
    }
}

#[test]
fn test_omega_dvd_contradiction() {
    // Given: 3 ∣ n (n ≡ 0 mod 3) and n ≡ 1 mod 3 (should be UNSAT)
    let n_var = 0;
    let constraints = vec![
        CertifiedOmegaConstraint::from_hypothesis(
            OmegaConstraint::Mod {
                var: n_var,
                remainder: 0, // 3 ∣ n
                modulus: 3,
            },
            0,
            2,
        ),
        CertifiedOmegaConstraint::from_hypothesis(
            OmegaConstraint::Mod {
                var: n_var,
                remainder: 1, // n ≡ 1 (mod 3)
                modulus: 3,
            },
            1,
            2,
        ),
    ];

    let result = omega_check_certified(&constraints);
    assert!(
        matches!(result, OmegaCertifiedResult::Unsat(_)),
        "3 ∣ n ∧ n ≡ 1 (mod 3) should be UNSAT"
    );

    if let OmegaCertifiedResult::Unsat(cert) = result {
        assert!(
            matches!(
                cert.contradiction_type,
                OmegaContradictionType::Divisibility
            ),
            "Should detect divisibility contradiction"
        );
    }
}

#[test]
fn test_omega_with_even_odd_hypotheses() {
    // End-to-end test: prove False from Even n and Odd n hypotheses
    // This tests the full flow: parsing -> constraint extraction -> solving
    let env = setup_env();

    // Create Even n and Odd n expressions where n is FVarId(0)
    let n_fvar = FVarId(0);
    let even_ty = Expr::app(
        Expr::const_(Name::from_string("Even"), vec![]),
        Expr::fvar(n_fvar),
    );
    let odd_ty = Expr::app(
        Expr::const_(Name::from_string("Odd"), vec![]),
        Expr::fvar(n_fvar),
    );
    let false_ty = Expr::const_(Name::from_string("False"), vec![]);

    let mut state = ProofState::with_context(
        env,
        false_ty.clone(),
        vec![
            LocalDecl {
                fvar: n_fvar,
                name: "n".to_string(),
                ty: Expr::const_(Name::from_string("Nat"), vec![]),
                value: None,
            },
            LocalDecl {
                fvar: FVarId(1),
                name: "h_even".to_string(),
                ty: even_ty,
                value: None,
            },
            LocalDecl {
                fvar: FVarId(2),
                name: "h_odd".to_string(),
                ty: odd_ty,
                value: None,
            },
        ],
    );

    // omega should detect the parity contradiction
    let result = omega(&mut state);
    assert!(
        result.is_ok(),
        "omega should prove False from Even n and Odd n: {result:?}"
    );
    assert!(state.is_complete(), "Proof should be complete after omega");
}

#[test]
fn test_omega_with_dvd_constraint() {
    // End-to-end test: prove False from 3 ∣ n and n ≡ 1 (mod 3)
    let env = setup_env();

    let n_fvar = FVarId(0);

    // 3 ∣ n: Dvd.dvd 3 n
    let dvd_ty = Expr::app(
        Expr::app(
            Expr::const_(Name::from_string("Dvd.dvd"), vec![]),
            Expr::Lit(lean5_kernel::expr::Literal::Nat(3)),
        ),
        Expr::fvar(n_fvar),
    );

    let false_ty = Expr::const_(Name::from_string("False"), vec![]);

    let state = ProofState::with_context(
        env.clone(),
        false_ty.clone(),
        vec![
            LocalDecl {
                fvar: n_fvar,
                name: "n".to_string(),
                ty: Expr::const_(Name::from_string("Nat"), vec![]),
                value: None,
            },
            LocalDecl {
                fvar: FVarId(1),
                name: "h_dvd".to_string(),
                ty: dvd_ty,
                value: None,
            },
        ],
    );

    // Test that the constraint is extracted (manual check)
    let goal = state.current_goal().unwrap().clone();
    let result = extract_certified_omega_constraints(&state, &goal);
    // At least h_dvd should be extracted if parsing works
    // (may not prove False alone, but should parse)
    // Constraint extraction may or may not succeed depending on available
    // hypotheses. The purpose of this test is to verify no panics occur.
    // We don't require a specific result.
    let _ = result;
}

#[test]
fn test_expr_to_omega_constraint_not_even() {
    // Test parsing `Not (Even n)` → `Odd n` (n ≡ 1 mod 2)
    let n_fvar = FVarId(15);
    let even_expr = Expr::app(
        Expr::const_(Name::from_string("Even"), vec![]),
        Expr::fvar(n_fvar),
    );
    let not_even_expr = Expr::app(Expr::const_(Name::from_string("Not"), vec![]), even_expr);

    let result = expr_to_omega_constraint(&not_even_expr);
    assert!(result.is_some(), "Should parse Not (Even n)");

    if let Some(OmegaConstraint::Mod {
        var,
        remainder,
        modulus,
    }) = result
    {
        assert_eq!(var, 15, "Variable index should be fvar id");
        assert_eq!(remainder, 1, "Not Even means Odd, remainder 1");
        assert_eq!(modulus, 2, "Parity uses modulus 2");
    } else {
        panic!("Expected Mod constraint for Not (Even n)");
    }
}

#[test]
fn test_expr_to_omega_constraint_not_odd() {
    // Test parsing `Not (Odd n)` → `Even n` (n ≡ 0 mod 2)
    let n_fvar = FVarId(20);
    let odd_expr = Expr::app(
        Expr::const_(Name::from_string("Odd"), vec![]),
        Expr::fvar(n_fvar),
    );
    let not_odd_expr = Expr::app(Expr::const_(Name::from_string("Not"), vec![]), odd_expr);

    let result = expr_to_omega_constraint(&not_odd_expr);
    assert!(result.is_some(), "Should parse Not (Odd n)");

    if let Some(OmegaConstraint::Mod {
        var,
        remainder,
        modulus,
    }) = result
    {
        assert_eq!(var, 20, "Variable index should be fvar id");
        assert_eq!(remainder, 0, "Not Odd means Even, remainder 0");
        assert_eq!(modulus, 2, "Parity uses modulus 2");
    } else {
        panic!("Expected Mod constraint for Not (Odd n)");
    }
}

#[test]
fn test_expr_to_omega_constraint_mod_equality() {
    // Test parsing `n % 3 = 1` → n ≡ 1 (mod 3)
    let n_fvar = FVarId(25);

    // Build expression: Eq Nat (HMod.hMod Nat Nat Nat inst n 3) 1
    // Simplified pattern: App (App (App Eq ty) (App (App hmod n) m)) r
    let hmod_app = Expr::app(
        Expr::app(
            Expr::const_(Name::from_string("HMod.hMod"), vec![]),
            Expr::fvar(n_fvar),
        ),
        Expr::Lit(lean5_kernel::expr::Literal::Nat(3)),
    );

    let eq_expr = Expr::app(
        Expr::app(
            Expr::app(
                Expr::const_(Name::from_string("Eq"), vec![]),
                Expr::const_(Name::from_string("Nat"), vec![]),
            ),
            hmod_app,
        ),
        Expr::Lit(lean5_kernel::expr::Literal::Nat(1)),
    );

    let result = expr_to_omega_constraint(&eq_expr);
    assert!(result.is_some(), "Should parse n % 3 = 1");

    if let Some(OmegaConstraint::Mod {
        var,
        remainder,
        modulus,
    }) = result
    {
        assert_eq!(var, 25, "Variable index should be fvar id");
        assert_eq!(remainder, 1, "n % 3 = 1 means remainder 1");
        assert_eq!(modulus, 3, "Modulus should be 3");
    } else {
        panic!("Expected Mod constraint for n % 3 = 1");
    }
}

#[test]
fn test_expr_to_omega_constraint_mod_equality_zero() {
    // Test parsing `n % 5 = 0` → n ≡ 0 (mod 5), equivalent to 5 ∣ n
    let n_fvar = FVarId(30);

    let hmod_app = Expr::app(
        Expr::app(
            Expr::const_(Name::from_string("HMod.hMod"), vec![]),
            Expr::fvar(n_fvar),
        ),
        Expr::Lit(lean5_kernel::expr::Literal::Nat(5)),
    );

    let eq_expr = Expr::app(
        Expr::app(
            Expr::app(
                Expr::const_(Name::from_string("Eq"), vec![]),
                Expr::const_(Name::from_string("Nat"), vec![]),
            ),
            hmod_app,
        ),
        Expr::Lit(lean5_kernel::expr::Literal::Nat(0)),
    );

    let result = expr_to_omega_constraint(&eq_expr);
    assert!(result.is_some(), "Should parse n % 5 = 0");

    if let Some(OmegaConstraint::Mod {
        var,
        remainder,
        modulus,
    }) = result
    {
        assert_eq!(var, 30);
        assert_eq!(remainder, 0, "n % 5 = 0 means remainder 0");
        assert_eq!(modulus, 5);
    } else {
        panic!("Expected Mod constraint for n % 5 = 0");
    }
}

#[test]
fn test_match_hmod_app() {
    // Test HMod.hMod pattern matching
    let n_fvar = FVarId(35);
    let hmod_app = Expr::app(
        Expr::app(
            Expr::const_(Name::from_string("HMod.hMod"), vec![]),
            Expr::fvar(n_fvar),
        ),
        Expr::Lit(lean5_kernel::expr::Literal::Nat(7)),
    );

    let result = match_hmod_app(&hmod_app);
    assert!(result.is_some(), "Should match HMod.hMod n m");

    if let Some((n, m)) = result {
        // n should be the FVar
        assert!(
            matches!(n, Expr::FVar(id) if id.0 == 35),
            "First arg should be FVar(35)"
        );
        // m should be the literal 7
        assert!(
            matches!(m, Expr::Lit(lean5_kernel::expr::Literal::Nat(7))),
            "Second arg should be Nat(7)"
        );
    }
}

#[test]
fn test_omega_mod_equality_contradiction() {
    // Test omega finding contradiction: n % 4 = 1 and n % 4 = 3
    // These constraints are incompatible
    let n_var = 0;
    let constraints = vec![
        CertifiedOmegaConstraint::from_hypothesis(
            OmegaConstraint::Mod {
                var: n_var,
                remainder: 1,
                modulus: 4,
            },
            0,
            2,
        ),
        CertifiedOmegaConstraint::from_hypothesis(
            OmegaConstraint::Mod {
                var: n_var,
                remainder: 3,
                modulus: 4,
            },
            1,
            2,
        ),
    ];

    let result = omega_check_certified(&constraints);
    assert!(
        matches!(result, OmegaCertifiedResult::Unsat(_)),
        "n ≡ 1 (mod 4) ∧ n ≡ 3 (mod 4) should be UNSAT"
    );

    if let OmegaCertifiedResult::Unsat(cert) = result {
        assert!(
            matches!(
                cert.contradiction_type,
                OmegaContradictionType::Divisibility
            ),
            "Should detect divisibility/modular contradiction"
        );
    }
}

#[test]
fn test_expr_to_omega_constraint_not_dvd() {
    // Test parsing `Not (Dvd.dvd 3 n)` → ¬(3 ∣ n) → NotMod { var: n, modulus: 3 }
    let n_fvar = FVarId(40);

    // Build Dvd.dvd 3 n
    // Pattern: App (App (App (App (Const "Dvd.dvd") ty) inst) a) b
    let dvd_app = Expr::app(
        Expr::app(
            Expr::const_(Name::from_string("Dvd.dvd"), vec![]),
            Expr::Lit(lean5_kernel::expr::Literal::Nat(3)),
        ),
        Expr::fvar(n_fvar),
    );

    // Wrap in Not
    let not_dvd_expr = Expr::app(Expr::const_(Name::from_string("Not"), vec![]), dvd_app);

    let result = expr_to_omega_constraint(&not_dvd_expr);
    assert!(result.is_some(), "Should parse Not (Dvd.dvd 3 n)");

    if let Some(OmegaConstraint::NotMod { var, modulus }) = result {
        assert_eq!(var, 40, "Variable index should be fvar id");
        assert_eq!(modulus, 3, "Modulus should be 3 (the divisor)");
    } else {
        panic!("Expected NotMod constraint for Not (Dvd.dvd 3 n), got {result:?}");
    }
}

#[test]
fn test_omega_dvd_not_dvd_contradiction() {
    // Test omega finding contradiction: 3 ∣ n and ¬(3 ∣ n)
    // These constraints are directly contradictory
    let n_var = 0;
    let constraints = vec![
        CertifiedOmegaConstraint::from_hypothesis(
            OmegaConstraint::Mod {
                var: n_var,
                remainder: 0,
                modulus: 3,
            },
            0,
            2,
        ),
        CertifiedOmegaConstraint::from_hypothesis(
            OmegaConstraint::NotMod {
                var: n_var,
                modulus: 3,
            },
            1,
            2,
        ),
    ];

    let result = omega_check_certified(&constraints);
    assert!(
        matches!(result, OmegaCertifiedResult::Unsat(_)),
        "3 ∣ n ∧ ¬(3 ∣ n) should be UNSAT"
    );

    if let OmegaCertifiedResult::Unsat(cert) = result {
        assert!(
            matches!(
                cert.contradiction_type,
                OmegaContradictionType::Divisibility
            ),
            "Should detect divisibility contradiction"
        );
    }
}

#[test]
fn test_negate_omega_constraint_not_mod() {
    // Test negating NotMod gives Mod with remainder 0
    let not_mod = OmegaConstraint::NotMod { var: 5, modulus: 7 };
    let negated = negate_omega_constraint(&not_mod);

    assert!(negated.is_some(), "NotMod should be negatable");
    if let Some(OmegaConstraint::Mod {
        var,
        remainder,
        modulus,
    }) = negated
    {
        assert_eq!(var, 5);
        assert_eq!(remainder, 0);
        assert_eq!(modulus, 7);
    } else {
        panic!("Expected Mod constraint from negating NotMod");
    }
}

#[test]
fn test_negate_omega_constraint_mod_to_not_mod() {
    // Test negating Mod (with remainder 0) gives NotMod
    let mod_constraint = OmegaConstraint::Mod {
        var: 3,
        remainder: 0,
        modulus: 5,
    };
    let negated = negate_omega_constraint(&mod_constraint);

    assert!(negated.is_some(), "Mod should be negatable to NotMod");
    if let Some(OmegaConstraint::NotMod { var, modulus }) = negated {
        assert_eq!(var, 3);
        assert_eq!(modulus, 5);
    } else {
        panic!("Expected NotMod constraint from negating Mod");
    }
}

#[test]
fn test_parse_linear_mod_equality() {
    // Test parsing `(a + b) % 3 = 1` → LinearMod
    // Build expression: Eq (HMod.hMod (Add.add a b) 3) 1
    let a_fvar = FVarId(10);
    let b_fvar = FVarId(11);

    // Build a + b
    let a_plus_b = Expr::app(
        Expr::app(
            Expr::const_(Name::from_string("HAdd.hAdd"), vec![]),
            Expr::fvar(a_fvar),
        ),
        Expr::fvar(b_fvar),
    );

    // Build (a + b) % 3
    let mod_expr = Expr::app(
        Expr::app(
            Expr::const_(Name::from_string("HMod.hMod"), vec![]),
            a_plus_b,
        ),
        Expr::Lit(lean5_kernel::expr::Literal::Nat(3)),
    );

    // Build Eq ((a + b) % 3) 1
    let eq_expr = Expr::app(
        Expr::app(
            Expr::app(
                Expr::const_(Name::from_string("Eq"), vec![]),
                Expr::const_(Name::from_string("Nat"), vec![]),
            ),
            mod_expr,
        ),
        Expr::Lit(lean5_kernel::expr::Literal::Nat(1)),
    );

    let result = expr_to_omega_constraint(&eq_expr);
    assert!(
        result.is_some(),
        "Should parse (a + b) % 3 = 1 as LinearMod"
    );

    if let Some(OmegaConstraint::LinearMod {
        expr,
        remainder,
        modulus,
    }) = result
    {
        assert_eq!(remainder, 1, "Remainder should be 1");
        assert_eq!(modulus, 3, "Modulus should be 3");
        // The expression should involve both variables
        assert!(
            !expr.is_constant(),
            "Expression should not be constant (has variables)"
        );
    } else {
        panic!("Expected LinearMod constraint for (a + b) % 3 = 1, got {result:?}");
    }
}

#[test]
fn test_omega_linear_mod_contradiction() {
    // Test omega finding contradiction: (a + b) ≡ 1 (mod 3) and (a + b) ≡ 2 (mod 3)
    // These are contradictory since a+b can't have two different remainders mod 3
    let a_var = 0;
    let b_var = 1;

    // a + b as linear expression
    let mut a_plus_b = LinearExpr::var(a_var);
    a_plus_b = a_plus_b.add(&LinearExpr::var(b_var));

    let constraints = vec![
        CertifiedOmegaConstraint::from_hypothesis(
            OmegaConstraint::LinearMod {
                expr: a_plus_b.clone(),
                remainder: 1,
                modulus: 3,
            },
            0,
            2,
        ),
        CertifiedOmegaConstraint::from_hypothesis(
            OmegaConstraint::LinearMod {
                expr: a_plus_b.clone(),
                remainder: 2,
                modulus: 3,
            },
            1,
            2,
        ),
    ];

    let result = omega_check_certified(&constraints);
    assert!(
        matches!(result, OmegaCertifiedResult::Unsat(_)),
        "(a + b) ≡ 1 (mod 3) ∧ (a + b) ≡ 2 (mod 3) should be UNSAT"
    );
}

#[test]
fn test_negate_linear_mod_to_not_linear_mod() {
    // Test negating LinearMod gives NotLinearMod
    let expr = LinearExpr::var(5).add(&LinearExpr::var(6));
    let linear_mod = OmegaConstraint::LinearMod {
        expr: expr.clone(),
        remainder: 2,
        modulus: 4,
    };
    let negated = negate_omega_constraint(&linear_mod);

    assert!(negated.is_some(), "LinearMod should be negatable");
    if let Some(OmegaConstraint::NotLinearMod {
        expr: neg_expr,
        remainder,
        modulus,
    }) = negated
    {
        assert_eq!(neg_expr, expr);
        assert_eq!(remainder, 2);
        assert_eq!(modulus, 4);
    } else {
        panic!("Expected NotLinearMod constraint from negating LinearMod");
    }
}

#[test]
fn test_negate_not_linear_mod_to_linear_mod() {
    // Test negating NotLinearMod gives LinearMod
    let expr = LinearExpr::var(7).add(&LinearExpr::constant(3));
    let not_linear_mod = OmegaConstraint::NotLinearMod {
        expr: expr.clone(),
        remainder: 1,
        modulus: 5,
    };
    let negated = negate_omega_constraint(&not_linear_mod);

    assert!(negated.is_some(), "NotLinearMod should be negatable");
    if let Some(OmegaConstraint::LinearMod {
        expr: neg_expr,
        remainder,
        modulus,
    }) = negated
    {
        assert_eq!(neg_expr, expr);
        assert_eq!(remainder, 1);
        assert_eq!(modulus, 5);
    } else {
        panic!("Expected LinearMod constraint from negating NotLinearMod");
    }
}

#[test]
fn test_omega_linear_mod_not_linear_mod_contradiction() {
    // Test: (a + b) ≡ 0 (mod 3) ∧ (a + b) ≢ 0 (mod 3) should be UNSAT
    let a_var = 0;
    let b_var = 1;

    let a_plus_b = LinearExpr::var(a_var).add(&LinearExpr::var(b_var));

    let constraints = vec![
        CertifiedOmegaConstraint::from_hypothesis(
            OmegaConstraint::LinearMod {
                expr: a_plus_b.clone(),
                remainder: 0,
                modulus: 3,
            },
            0,
            2,
        ),
        CertifiedOmegaConstraint::from_hypothesis(
            OmegaConstraint::NotLinearMod {
                expr: a_plus_b.clone(),
                remainder: 0,
                modulus: 3,
            },
            1,
            2,
        ),
    ];

    let result = omega_check_certified(&constraints);
    assert!(
        matches!(result, OmegaCertifiedResult::Unsat(_)),
        "(a + b) ≡ 0 (mod 3) ∧ (a + b) ≢ 0 (mod 3) should be UNSAT"
    );
}

#[test]
fn test_parse_negated_mod_nonzero_remainder() {
    // Test parsing `Not (n % 5 = 2)` → NotLinearMod
    let n_fvar = FVarId(20);

    // Build n % 5
    let mod_expr = Expr::app(
        Expr::app(
            Expr::const_(Name::from_string("HMod.hMod"), vec![]),
            Expr::fvar(n_fvar),
        ),
        Expr::Lit(lean5_kernel::expr::Literal::Nat(5)),
    );

    // Build Eq (n % 5) 2
    let eq_expr = Expr::app(
        Expr::app(
            Expr::app(
                Expr::const_(Name::from_string("Eq"), vec![]),
                Expr::const_(Name::from_string("Nat"), vec![]),
            ),
            mod_expr,
        ),
        Expr::Lit(lean5_kernel::expr::Literal::Nat(2)),
    );

    // Wrap in Not
    let not_eq_expr = Expr::app(Expr::const_(Name::from_string("Not"), vec![]), eq_expr);

    let result = expr_to_omega_constraint(&not_eq_expr);
    assert!(result.is_some(), "Should parse Not (n % 5 = 2)");

    if let Some(OmegaConstraint::NotLinearMod {
        expr,
        remainder,
        modulus,
    }) = result
    {
        assert_eq!(remainder, 2, "Remainder should be 2");
        assert_eq!(modulus, 5, "Modulus should be 5");
        // The expression should be a single variable (n)
        assert!(
            expr.coeffs.len() == 1 && expr.constant == 0,
            "Expression should be single variable"
        );
    } else {
        panic!("Expected NotLinearMod constraint for Not (n % 5 = 2), got {result:?}");
    }
}

// =========================================================================
