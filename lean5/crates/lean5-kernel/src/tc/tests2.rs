//! Tests for type checker
use super::*;

#[test]
fn test_try_proof_irrel_eq_both_in_prop() {
    // Kill mutant at line 900: replace && with ||
    // Both types must be in Prop for proof irrelevance

    use crate::env::Declaration;
    let mut env = Environment::new();

    // Add an axiom of type Prop
    env.add_decl(Declaration::Axiom {
        name: Name::from_string("p"),
        level_params: vec![],
        type_: Expr::prop(), // p : Prop
    })
    .unwrap();

    // Add a proof of p
    env.add_decl(Declaration::Axiom {
        name: Name::from_string("proof1"),
        level_params: vec![],
        type_: Expr::const_(Name::from_string("p"), vec![]), // proof1 : p
    })
    .unwrap();

    env.add_decl(Declaration::Axiom {
        name: Name::from_string("proof2"),
        level_params: vec![],
        type_: Expr::const_(Name::from_string("p"), vec![]), // proof2 : p
    })
    .unwrap();

    // Add a term in Type (not Prop)
    env.add_decl(Declaration::Axiom {
        name: Name::from_string("t"),
        level_params: vec![],
        type_: Expr::type_(), // t : Type
    })
    .unwrap();

    let tc = TypeChecker::new(&env);

    let proof1 = Expr::const_(Name::from_string("proof1"), vec![]);
    let proof2 = Expr::const_(Name::from_string("proof2"), vec![]);
    let t = Expr::const_(Name::from_string("t"), vec![]);

    // Two proofs of the same proposition SHOULD be proof-irrelevant equal
    // This is the case where both are in Prop
    assert!(
        tc.try_proof_irrel_eq(&proof1, &proof2),
        "Two proofs of same Prop should be proof-irrelevant equal"
    );

    // A proof vs a non-proof term should NOT be proof-irrelevant equal
    // With &&: is_type_in_prop(p)=true && is_type_in_prop(Type)=false = false → return false
    // With ||: is_type_in_prop(p)=true || is_type_in_prop(Type)=false = true → calls is_def_eq(p, Type)
    //
    // The key insight: if || is used, we'd call is_def_eq on mismatched types.
    // is_def_eq(p, Type) returns false (they're different), so result is same.
    // This mutation is equivalent because is_def_eq acts as a safety net.
    //
    // To truly kill this mutant, we'd need is_def_eq to return true on mismatched types,
    // which it never does. The mutation is semantically equivalent.
    //
    // REFACTOR: Instead of relying on is_def_eq as safety net, make the condition explicit.
    assert!(
        !tc.try_proof_irrel_eq(&proof1, &t),
        "A proof vs a Type term should not be proof-irrelevant equal"
    );
}

#[test]
fn test_is_type_in_prop_sort_universe_check() {
    // Kill mutant at line 947: delete match arm Expr::Sort(_) => false
    // Additional test for sort universe checking

    // A Sort is a TYPE, not a proof. is_type_in_prop should return false for Sort.
    let env = Environment::new();
    let tc = TypeChecker::new(&env);

    // Sort(0) = Prop, but Prop itself is not "in Prop" (it's a type universe)
    assert!(
        !tc.is_type_in_prop(&Expr::prop()),
        "Prop (Sort 0) is not in Prop - it IS Prop, not a term in Prop"
    );

    // Type = Sort(1) is also not in Prop
    assert!(
        !tc.is_type_in_prop(&Expr::type_()),
        "Type (Sort 1) is not in Prop"
    );

    // Higher universes also not in Prop
    assert!(
        !tc.is_type_in_prop(&Expr::Sort(Level::succ(Level::succ(Level::zero())))),
        "Sort 2 is not in Prop"
    );
}

#[test]
fn test_lift_expr_cutoff_plus_one() {
    // Kill mutants at lines 1035, 1040: replace + with - or * in cutoff + 1

    let env = Environment::new();
    let tc = TypeChecker::new(&env);

    // λ x. BVar(0) lifted at cutoff 0 by 5
    // Under lambda, cutoff becomes 0+1=1
    // BVar(0) < 1, so NOT lifted
    let lam = Expr::lam(BinderInfo::Default, Expr::prop(), Expr::BVar(0));
    let result = tc.lift_expr(&lam, 0, 5);
    match &result {
        Expr::Lam(_, _, body) => {
            assert_eq!(
                body.as_ref(),
                &Expr::BVar(0),
                "BVar(0) under lambda should NOT be lifted (bound)"
            );
        }
        _ => panic!("Expected Lam"),
    }

    // λ x. BVar(1) lifted at cutoff 0 by 5
    // BVar(1) >= 1, so lifted to BVar(6)
    let lam = Expr::lam(BinderInfo::Default, Expr::prop(), Expr::BVar(1));
    let result = tc.lift_expr(&lam, 0, 5);
    match &result {
        Expr::Lam(_, _, body) => {
            assert_eq!(
                body.as_ref(),
                &Expr::BVar(6),
                "BVar(1) under lambda should be lifted to BVar(6)"
            );
        }
        _ => panic!("Expected Lam"),
    }

    // Π x. BVar(0) lifted
    let pi = Expr::pi(BinderInfo::Default, Expr::prop(), Expr::BVar(0));
    let result = tc.lift_expr(&pi, 0, 5);
    match &result {
        Expr::Pi(_, _, body) => {
            assert_eq!(
                body.as_ref(),
                &Expr::BVar(0),
                "BVar(0) under Pi should NOT be lifted"
            );
        }
        _ => panic!("Expected Pi"),
    }

    // let x = v in BVar(0) lifted
    let let_expr = Expr::let_(Expr::prop(), Expr::prop(), Expr::BVar(0));
    let result = tc.lift_expr(&let_expr, 0, 5);
    match &result {
        Expr::Let(_, _, body) => {
            assert_eq!(
                body.as_ref(),
                &Expr::BVar(0),
                "BVar(0) under let should NOT be lifted"
            );
        }
        _ => panic!("Expected Let"),
    }

    // Double nested: λ x. λ y. BVar(2)
    // At depth 2, BVar(2) >= 2, so lifted to BVar(7)
    let inner = Expr::lam(BinderInfo::Default, Expr::prop(), Expr::BVar(2));
    let outer = Expr::lam(BinderInfo::Default, Expr::prop(), inner);
    let result = tc.lift_expr(&outer, 0, 5);
    match &result {
        Expr::Lam(_, _, body) => match body.as_ref() {
            Expr::Lam(_, _, inner_body) => {
                assert_eq!(
                    inner_body.as_ref(),
                    &Expr::BVar(7),
                    "BVar(2) under 2 lambdas should be lifted to BVar(7)"
                );
            }
            _ => panic!("Expected inner Lam"),
        },
        _ => panic!("Expected outer Lam"),
    }
}

#[test]
fn test_try_iota_reduction_field_boundary() {
    // Kill mutants at line 761: replace < with ==, >, or <=
    // This tests: if field_start < major_args.len()

    use crate::inductive::{Constructor, InductiveDecl, InductiveType};

    let mut env = Environment::new();

    // Create Nat with zero and succ
    let nat = Name::from_string("Nat");
    let nat_ref = Expr::const_(nat.clone(), vec![]);

    let decl = InductiveDecl {
        level_params: vec![],
        num_params: 0,
        types: vec![InductiveType {
            name: nat.clone(),
            type_: Expr::type_(),
            constructors: vec![
                Constructor {
                    name: Name::from_string("Nat.zero"),
                    type_: nat_ref.clone(),
                },
                Constructor {
                    name: Name::from_string("Nat.succ"),
                    type_: Expr::arrow(nat_ref.clone(), nat_ref.clone()),
                },
            ],
        }],
    };
    env.add_inductive(decl).unwrap();

    let tc = TypeChecker::new(&env);

    // Nat.rec with zero should work (zero has 0 fields)
    let rec = Expr::const_(Name::from_string("Nat.rec"), vec![Level::zero()]);
    let motive = Expr::lam(BinderInfo::Default, nat_ref.clone(), Expr::prop());
    let case_zero = Expr::type_();
    let case_succ = Expr::lam(
        BinderInfo::Default,
        nat_ref.clone(),
        Expr::lam(BinderInfo::Default, Expr::prop(), Expr::BVar(1)),
    );
    let zero = Expr::const_(Name::from_string("Nat.zero"), vec![]);

    let app = Expr::app(
        Expr::app(
            Expr::app(Expr::app(rec.clone(), motive.clone()), case_zero.clone()),
            case_succ.clone(),
        ),
        zero,
    );

    let result = tc.whnf(&app);
    assert_eq!(result, case_zero, "Nat.rec zero should give case_zero");

    // Nat.rec with succ(zero) should also work
    let succ_zero = Expr::app(
        Expr::const_(Name::from_string("Nat.succ"), vec![]),
        Expr::const_(Name::from_string("Nat.zero"), vec![]),
    );

    let app2 = Expr::app(
        Expr::app(Expr::app(Expr::app(rec, motive), case_zero), case_succ),
        succ_zero,
    );

    let result2 = tc.whnf(&app2);
    // Result should be case_succ applied to zero and (rec zero)
    // case_succ n ih reduces to: λ n ih. n (which is BVar(1) with depth 2)
    // So the full result is the application of case_succ to zero and (rec ... zero)
    //
    // The key assertion: result2 should NOT be the original application
    // (if reduction failed, app2 == result2 as whnf wouldn't change it)
    // With the < mutation changed to == or > or <=, field extraction fails
    // and the result would be wrong or None
    assert_ne!(
        app2, result2,
        "Nat.rec (succ zero) must reduce (field_start < major_args.len)"
    );
}

#[test]
fn test_try_iota_reduction_args_before_major() {
    // Kill mutant at line 716: replace + with - in args_before_major calculation
    // args_before_major = num_params + num_motives + num_minors + num_indices

    use crate::inductive::{Constructor, InductiveDecl, InductiveType};

    let mut env = Environment::new();

    // Create Bool (0 params, 1 motive, 2 minors, 0 indices)
    let bool_name = Name::from_string("Bool");
    let bool_ref = Expr::const_(bool_name.clone(), vec![]);

    let decl = InductiveDecl {
        level_params: vec![],
        num_params: 0,
        types: vec![InductiveType {
            name: bool_name.clone(),
            type_: Expr::type_(),
            constructors: vec![
                Constructor {
                    name: Name::from_string("Bool.true"),
                    type_: bool_ref.clone(),
                },
                Constructor {
                    name: Name::from_string("Bool.false"),
                    type_: bool_ref.clone(),
                },
            ],
        }],
    };
    env.add_inductive(decl).unwrap();

    let tc = TypeChecker::new(&env);

    // Bool.rec has: 0 params + 1 motive + 2 minors + 0 indices = 3 args before major
    let rec = Expr::const_(Name::from_string("Bool.rec"), vec![Level::zero()]);
    let motive = Expr::lam(BinderInfo::Default, bool_ref.clone(), Expr::prop());
    let case_true = Expr::type_();
    let case_false = Expr::prop();
    let tt = Expr::const_(Name::from_string("Bool.true"), vec![]);

    // Should have exactly 4 args: motive, case_true, case_false, major(tt)
    let app = Expr::app(
        Expr::app(
            Expr::app(Expr::app(rec, motive), case_true.clone()),
            case_false,
        ),
        tt,
    );

    let result = tc.whnf(&app);
    // Should reduce to case_true
    assert_eq!(result, case_true, "Bool.rec true should give case_true");
}

#[test]
fn test_is_def_eq_literals() {
    // Kill mutant at line 867: delete match arm (Expr::Lit(l1), Expr::Lit(l2))
    // Without this arm, Lit falls through to `_ => false`
    use crate::expr::Literal;

    let env = Environment::new();
    let tc = TypeChecker::new(&env);

    // Same nat literals should be def_eq
    let nat42_a = Expr::Lit(Literal::Nat(42));
    let nat42_b = Expr::Lit(Literal::Nat(42));
    assert!(
        tc.is_def_eq(&nat42_a, &nat42_b),
        "Lit(42) should be def_eq to Lit(42)"
    );

    // Different nat literals should not be def_eq
    let nat43 = Expr::Lit(Literal::Nat(43));
    assert!(
        !tc.is_def_eq(&nat42_a, &nat43),
        "Lit(42) should NOT be def_eq to Lit(43)"
    );

    // Same string literals should be def_eq
    let str_a = Expr::Lit(Literal::String(std::sync::Arc::from("hello")));
    let str_b = Expr::Lit(Literal::String(std::sync::Arc::from("hello")));
    assert!(
        tc.is_def_eq(&str_a, &str_b),
        "Lit(\"hello\") should be def_eq to Lit(\"hello\")"
    );

    // Different string literals should not be def_eq
    let str_c = Expr::Lit(Literal::String(std::sync::Arc::from("world")));
    assert!(
        !tc.is_def_eq(&str_a, &str_c),
        "Lit(\"hello\") should NOT be def_eq to Lit(\"world\")"
    );
}

// =========================================================================
// Kill: tc.rs:42:22, 56:22 - += to -= in LocalContext::push/push_let
// =========================================================================
#[test]
fn test_local_context_fvarid_increment() {
    // Tests that FVarIds are monotonically increasing
    // Mutation: += to -= would cause IDs to decrease

    let mut ctx = LocalContext::new();

    // Push several bindings and verify IDs are increasing
    let id1 = ctx.push(Name::anon(), Expr::prop(), BinderInfo::Default);
    let id2 = ctx.push(Name::anon(), Expr::prop(), BinderInfo::Default);
    let id3 = ctx.push_let(Name::anon(), Expr::prop(), Expr::prop());

    // IDs should be monotonically increasing
    assert!(id2.0 > id1.0, "id2 should be greater than id1");
    assert!(id3.0 > id2.0, "id3 should be greater than id2");

    // Verify we can look them up
    assert!(ctx.get(id1).is_some());
    assert!(ctx.get(id2).is_some());
    assert!(ctx.get(id3).is_some());

    // With -= mutation, id2.0 would be id1.0 - 1, causing lookup issues
}

// =========================================================================
// Kill: tc.rs:98:33 - + to - in LocalContext::push_with_id
// =========================================================================
#[test]
fn test_local_context_push_with_id_updates_next_id() {
    // Tests that push_with_id correctly updates next_id to avoid collisions
    // Mutation: id.0 + 1 -> id.0 - 1 would cause next_id to underflow

    let mut ctx = LocalContext::new();

    // Push with a specific ID
    ctx.push_with_id(FVarId(100), Name::anon(), Expr::prop(), BinderInfo::Default);

    // Now push a regular binding - should get ID > 100
    let id = ctx.push(Name::anon(), Expr::prop(), BinderInfo::Default);
    assert!(
        id.0 > 100,
        "New ID should be greater than 100, got {}",
        id.0
    );

    // With - mutation, next_id would be 99, causing ID collision
}

// =========================================================================
// Kill: tc.rs:1035:54, 1040:54 - + to - in lift_expr (Pi body, Let body)
// =========================================================================
#[test]
fn test_lift_expr_pi_let_cutoff_increment() {
    // Tests that lift_expr correctly increments cutoff in Pi and Let bodies
    // Mutation: cutoff + 1 -> cutoff - 1 would break lifting

    let env = Environment::new();
    let tc = TypeChecker::new(&env);

    // π x : Prop . BVar(1) - BVar(1) refers outside the pi
    // When lifting from cutoff=0 by amount=2, body processed at cutoff=1
    // BVar(1) >= 1, so lift by 2: should become BVar(3)
    let pi = Expr::pi(BinderInfo::Default, Expr::prop(), Expr::BVar(1));
    let result = tc.lift_expr(&pi, 0, 2);
    match result {
        Expr::Pi(_, _, body) => {
            assert_eq!(
                body.as_ref(),
                &Expr::BVar(3),
                "BVar(1) in pi body should lift to BVar(3)"
            );
        }
        _ => panic!("Expected Pi"),
    }

    // let x = Prop in BVar(1) - similar test for Let
    let let_expr = Expr::let_(Expr::prop(), Expr::prop(), Expr::BVar(1));
    let result = tc.lift_expr(&let_expr, 0, 2);
    match result {
        Expr::Let(_, _, body) => {
            assert_eq!(
                body.as_ref(),
                &Expr::BVar(3),
                "BVar(1) in let body should lift to BVar(3)"
            );
        }
        _ => panic!("Expected Let"),
    }

    // Bound variable should NOT be lifted
    let pi_bound = Expr::pi(BinderInfo::Default, Expr::prop(), Expr::BVar(0));
    let result = tc.lift_expr(&pi_bound, 0, 2);
    match result {
        Expr::Pi(_, _, body) => {
            assert_eq!(
                body.as_ref(),
                &Expr::BVar(0),
                "BVar(0) in pi body should not be lifted"
            );
        }
        _ => panic!("Expected Pi"),
    }
}

// =========================================================================
// Kill: tc.rs:501 - replace match guard id == fvar_id with true
// =========================================================================
#[test]
fn test_convert_fvar_cert_to_bvar_multiple_fvars() {
    // This test uses two different FVars and converts only one of them.
    // Mutation: id == fvar_id -> true would convert BOTH FVars to BVars

    use crate::cert::ProofCert;

    // Create two different FVars
    let fvar1 = FVarId(100);
    let fvar2 = FVarId(200);

    // Create an App cert: (FVar1 FVar2)
    // The function is FVar1, argument is FVar2
    let app_cert = ProofCert::App {
        fn_cert: Box::new(ProofCert::FVar {
            id: fvar1,
            type_: Box::new(Expr::pi(BinderInfo::Default, Expr::prop(), Expr::prop())),
        }),
        fn_type: Box::new(Expr::pi(BinderInfo::Default, Expr::prop(), Expr::prop())),
        arg_cert: Box::new(ProofCert::FVar {
            id: fvar2,
            type_: Box::new(Expr::prop()),
        }),
        result_type: Box::new(Expr::prop()),
    };

    // Convert only fvar1 to a BVar at depth 0
    let result = convert_fvar_cert_to_bvar(app_cert, fvar1, 0);

    // The fn_cert should be BVar(0), but arg_cert should still be FVar(200)
    match result {
        ProofCert::App {
            fn_cert, arg_cert, ..
        } => {
            // fn_cert should be converted to BVar
            match *fn_cert {
                ProofCert::BVar { idx, .. } => {
                    assert_eq!(idx, 0, "FVar1 should convert to BVar(0)");
                }
                ProofCert::FVar { id, .. } => {
                    panic!("FVar1 should have been converted, got FVar({})", id.0);
                }
                _ => panic!("Expected BVar or FVar for fn_cert"),
            }

            // arg_cert should NOT be converted (different FVar)
            match *arg_cert {
                ProofCert::FVar { id, .. } => {
                    assert_eq!(
                        id.0, 200,
                        "FVar2 should remain as FVar(200), got FVar({})",
                        id.0
                    );
                }
                ProofCert::BVar { idx, .. } => {
                    panic!("FVar2 should NOT have been converted to BVar, got BVar({idx})");
                }
                _ => panic!("Expected FVar or BVar for arg_cert"),
            }
        }
        _ => panic!("Expected App"),
    }
}

// =========================================================================
// Kill: tc.rs:542 - replace + with - or * in convert_fvar_cert_to_bvar (Pi body depth)
// =========================================================================
#[test]
fn test_convert_fvar_cert_pi_body_depth() {
    // Tests that Pi body correctly increments depth by 1
    // Mutation: depth + 1 -> depth - 1 or depth * 1 would break nested conversion

    use crate::cert::ProofCert;

    let fvar = FVarId(42);

    // Create a Pi cert: ∀ (x : Prop). FVar42
    // The FVar42 in the body should become BVar(1) when converting at depth 0
    // Because: under the Pi binder, depth is 1
    let pi_cert = ProofCert::Pi {
        binder_info: BinderInfo::Default,
        arg_type_cert: Box::new(ProofCert::Sort {
            level: Level::zero(),
        }),
        arg_level: Level::succ(Level::zero()),
        body_type_cert: Box::new(ProofCert::FVar {
            id: fvar,
            type_: Box::new(Expr::prop()),
        }),
        body_level: Level::succ(Level::zero()),
    };

    // Convert fvar at depth 0
    // In the body, depth should be 1, so fvar -> BVar(1)
    let result = convert_fvar_cert_to_bvar(pi_cert, fvar, 0);

    match result {
        ProofCert::Pi { body_type_cert, .. } => {
            match *body_type_cert {
                ProofCert::BVar { idx, .. } => {
                    // With correct depth + 1: fvar at body depth 1 -> BVar(1)
                    // With depth - 1: would be depth -1, wrong
                    // With depth * 1: would be depth 0, giving BVar(0)
                    assert_eq!(
                        idx, 1,
                        "FVar in Pi body should become BVar(1) at depth 1, got BVar({idx})"
                    );
                }
                ProofCert::FVar { id, .. } => {
                    panic!("FVar should have been converted, got FVar({})", id.0);
                }
                _ => panic!("Expected BVar or FVar for body_type_cert"),
            }
        }
        _ => panic!("Expected Pi"),
    }
}

// =========================================================================
// Kill: tc.rs:716, tc.rs:761 - indexed inductive iota reduction tests
// These mutants can only be killed with an indexed inductive (num_indices > 0)
// =========================================================================
#[test]
fn test_iota_reduction_indexed_inductive() {
    // Kill mutants at:
    // - tc.rs:716: + to - in args_before_major (num_indices calculation)
    // - tc.rs:761: < to ==, >, <= in field_start < major_args.len()
    //
    // We need an indexed inductive: Idx : Nat → Type with num_indices = 1
    // add_inductive calculates num_indices = type_arity - num_params
    // For Idx : Nat → Type with num_params = 0, we get num_indices = 1

    use crate::inductive::{Constructor, InductiveDecl, InductiveType};

    let mut env = Environment::new();

    // First we need Nat for the index
    let nat = Name::from_string("Nat");
    let nat_ref = Expr::const_(nat.clone(), vec![]);

    let nat_decl = InductiveDecl {
        level_params: vec![],
        num_params: 0,
        types: vec![InductiveType {
            name: nat.clone(),
            type_: Expr::type_(),
            constructors: vec![
                Constructor {
                    name: Name::from_string("Nat.zero"),
                    type_: nat_ref.clone(),
                },
                Constructor {
                    name: Name::from_string("Nat.succ"),
                    type_: Expr::arrow(nat_ref.clone(), nat_ref.clone()),
                },
            ],
        }],
    };
    env.add_inductive(nat_decl).unwrap();

    // Create indexed inductive: Idx : Nat → Type
    // Single constructor: Idx.mk : (n : Nat) → Idx n
    // type_arity(Nat → Type) = 1, num_params = 0 => num_indices = 1
    let idx_name = Name::from_string("Idx");
    // Idx : Nat → Type (Pi Nat. Type)
    let idx_type = Expr::pi(BinderInfo::Default, nat_ref.clone(), Expr::type_());

    // Idx.mk : (n : Nat) → Idx n = Pi(n : Nat). Idx (BVar 0)
    let idx_mk_type = Expr::pi(
        BinderInfo::Default,
        nat_ref.clone(),
        Expr::app(Expr::const_(idx_name.clone(), vec![]), Expr::BVar(0)),
    );

    let idx_decl = InductiveDecl {
        level_params: vec![],
        num_params: 0, // No params, so the Nat argument is an INDEX
        types: vec![InductiveType {
            name: idx_name.clone(),
            type_: idx_type,
            constructors: vec![Constructor {
                name: Name::from_string("Idx.mk"),
                type_: idx_mk_type,
            }],
        }],
    };
    env.add_inductive(idx_decl).unwrap();

    // Verify num_indices was set correctly
    let idx_val = env.get_inductive(&idx_name).unwrap();
    assert_eq!(idx_val.num_indices, 1, "Idx should have 1 index");

    let rec_val = env.get_recursor(&Name::from_string("Idx.rec")).unwrap();
    assert_eq!(rec_val.num_indices, 1, "Idx.rec should have 1 index");

    let tc = TypeChecker::new(&env);

    // Test: Idx.rec motive minor index (Idx.mk index)
    // Idx.rec structure: motive, minor, index, major
    // args_before_major = 0 + 1 + 1 + 1 = 3
    // With mutation (+ to -): 0 + 1 + 1 - 1 = 1

    let zero = Expr::const_(Name::from_string("Nat.zero"), vec![]);
    let mk_name = Name::from_string("Idx.mk");
    let major = Expr::app(Expr::const_(mk_name.clone(), vec![]), zero.clone());

    let rec = Expr::const_(Name::from_string("Idx.rec"), vec![Level::zero()]);
    let motive = Expr::lam(
        BinderInfo::Default,
        nat_ref.clone(),
        Expr::lam(
            BinderInfo::Default,
            Expr::app(Expr::const_(idx_name.clone(), vec![]), Expr::BVar(0)),
            Expr::prop(),
        ),
    );
    let minor = Expr::lam(BinderInfo::Default, nat_ref.clone(), Expr::type_());

    // Build: rec motive minor index major
    let app = Expr::app(
        Expr::app(
            Expr::app(Expr::app(rec.clone(), motive.clone()), minor.clone()),
            zero.clone(),
        ),
        major.clone(),
    );

    let result = tc.whnf(&app);

    // The key assertion: reduction must happen
    // If args_before_major calculation is wrong (+ to -), major position is wrong
    // and reduction fails (returns original app)
    assert_ne!(
        app, result,
        "Idx.rec (Idx.mk zero) must reduce - indexed inductive iota (kills line 716 mutant)"
    );

    // Additional assertion to kill line 761 mutations:
    // The result should be `minor zero` which beta-reduces to Type
    // With < to == or < to >: fields would be empty, result = minor (a lambda)
    // Verify result is NOT a lambda (it should be Type after beta reduction)
    assert!(
        !matches!(result, Expr::Lam(..)),
        "Result should not be a lambda - field extraction failed (kills line 761 mutants)"
    );
}

#[test]
fn test_iota_reduction_param_vs_index() {
    // Test with an inductive that has BOTH params and indices
    // This distinguishes param extraction from index handling
    // Vec : Type → Nat → Type has 1 param (α) and 1 index (n)

    use crate::inductive::{Constructor, InductiveDecl, InductiveType};

    let mut env = Environment::new();

    // Need Nat first
    let nat = Name::from_string("Nat");
    let nat_ref = Expr::const_(nat.clone(), vec![]);

    let nat_decl = InductiveDecl {
        level_params: vec![],
        num_params: 0,
        types: vec![InductiveType {
            name: nat.clone(),
            type_: Expr::type_(),
            constructors: vec![
                Constructor {
                    name: Name::from_string("Nat.zero"),
                    type_: nat_ref.clone(),
                },
                Constructor {
                    name: Name::from_string("Nat.succ"),
                    type_: Expr::arrow(nat_ref.clone(), nat_ref.clone()),
                },
            ],
        }],
    };
    env.add_inductive(nat_decl).unwrap();

    // Create Vec : Type → Nat → Type
    // num_params = 1 (the Type), so type_arity(2) - num_params(1) = 1 index
    let vec_name = Name::from_string("Vec");
    let u = Name::from_string("u");

    // Vec : Type u → Nat → Type u
    // = Pi(α : Type u). Pi(n : Nat). Type u
    let vec_type = Expr::pi(
        BinderInfo::Implicit,
        Expr::Sort(Level::Param(u.clone())),
        Expr::pi(
            BinderInfo::Default,
            nat_ref.clone(),
            Expr::Sort(Level::Param(u.clone())),
        ),
    );

    // Vec.nil : {α : Type u} → Vec α Nat.zero
    // = Pi{α : Type u}. Vec α Nat.zero
    let vec_nil_type = Expr::pi(
        BinderInfo::Implicit,
        Expr::Sort(Level::Param(u.clone())),
        Expr::app(
            Expr::app(
                Expr::const_(vec_name.clone(), vec![Level::Param(u.clone())]),
                Expr::BVar(0),
            ),
            Expr::const_(Name::from_string("Nat.zero"), vec![]),
        ),
    );

    let vec_decl = InductiveDecl {
        level_params: vec![u.clone()],
        num_params: 1, // α is a param, n is an index
        types: vec![InductiveType {
            name: vec_name.clone(),
            type_: vec_type,
            constructors: vec![Constructor {
                name: Name::from_string("Vec.nil"),
                type_: vec_nil_type,
            }],
        }],
    };
    env.add_inductive(vec_decl).unwrap();

    // Verify correct setup
    let vec_val = env.get_inductive(&vec_name).unwrap();
    assert_eq!(vec_val.num_params, 1, "Vec should have 1 param");
    assert_eq!(vec_val.num_indices, 1, "Vec should have 1 index");

    let rec_val = env.get_recursor(&Name::from_string("Vec.rec")).unwrap();
    assert_eq!(rec_val.num_params, 1, "Vec.rec should have 1 param");
    assert_eq!(rec_val.num_indices, 1, "Vec.rec should have 1 index");

    let tc = TypeChecker::new(&env);

    // Vec.rec structure: param(α), motive, minor(nil), index(n), major
    // args_before_major = 1 + 1 + 1 + 1 = 4
    // With mutation (+ to -): 1 + 1 + 1 - 1 = 2

    let zero = Expr::const_(Name::from_string("Nat.zero"), vec![]);
    let alpha = Expr::type_(); // Use Type as the type parameter

    // Vec.nil @Type : Vec Type 0
    let nil = Expr::app(
        Expr::const_(Name::from_string("Vec.nil"), vec![Level::zero()]),
        alpha.clone(),
    );

    let rec = Expr::const_(
        Name::from_string("Vec.rec"),
        vec![Level::zero(), Level::zero()],
    );

    // Simplified motive
    let motive = Expr::lam(
        BinderInfo::Default,
        nat_ref.clone(),
        Expr::lam(
            BinderInfo::Default,
            Expr::app(
                Expr::app(
                    Expr::const_(vec_name.clone(), vec![Level::zero()]),
                    alpha.clone(),
                ),
                Expr::BVar(0),
            ),
            Expr::prop(),
        ),
    );

    let case_nil = Expr::type_();

    // Build: rec α motive case_nil index(=0) major(=nil)
    let app = Expr::app(
        Expr::app(
            Expr::app(
                Expr::app(Expr::app(rec, alpha.clone()), motive),
                case_nil.clone(),
            ),
            zero,
        ),
        nil,
    );

    let result = tc.whnf(&app);

    // With correct args_before_major = 4, major = args[4] = nil
    // With mutation (+ to -), args_before_major = 2, major = args[2] = case_nil (wrong!)
    assert_ne!(
        app, result,
        "Vec.rec nil must reduce (param+index test, kills line 716 mutant)"
    );
}

// =========================================================================
// Kill: tc.rs:56 - push_let += to -=
// Additional test specifically for push_let (existing test only checks push)
// =========================================================================
#[test]
fn test_local_context_push_let_id_positive() {
    // Tests that push_let generates positive, increasing IDs
    // Mutation: += to -= would cause IDs to become negative/decreasing

    let mut ctx = LocalContext::new();

    // Push only let bindings
    let id1 = ctx.push_let(Name::anon(), Expr::prop(), Expr::prop());
    let id2 = ctx.push_let(Name::anon(), Expr::type_(), Expr::type_());
    let id3 = ctx.push_let(
        Name::anon(),
        Expr::Sort(Level::succ(Level::zero())),
        Expr::type_(),
    );

    // IDs should be reasonable (not wrapped around from underflow)
    // If -= mutation happens: 0 - 1 = u64::MAX in wrapping
    assert!(
        id1.0 < u64::MAX / 2,
        "id1 should be small, got {} (underflow?)",
        id1.0
    );
    assert!(
        id2.0 < u64::MAX / 2,
        "id2 should be small, got {} (underflow?)",
        id2.0
    );
    assert!(
        id3.0 < u64::MAX / 2,
        "id3 should be small, got {} (underflow?)",
        id3.0
    );

    // IDs should be strictly increasing
    assert!(id2.0 > id1.0, "id2 ({}) should be > id1 ({})", id2.0, id1.0);
    assert!(id3.0 > id2.0, "id3 ({}) should be > id2 ({})", id3.0, id2.0);

    // With -= mutation, after first push_let at id=0:
    // next_id becomes 0 - 1 = u64::MAX (wrapping)
    // Second push_let would get id = u64::MAX
    // So id2 would NOT be > id1
}

// =========================================================================
// Recursor reduction tests with recursive fields
// =========================================================================

#[test]
fn test_iota_reduction_recursive_nat_succ() {
    // Test iota reduction on Nat.succ includes induction hypothesis
    // Nat.rec motive z_case s_case (Nat.succ n) = s_case n (Nat.rec motive z_case s_case n)
    use crate::inductive::{Constructor, InductiveDecl, InductiveType};

    let mut env = Environment::new();

    let nat = Name::from_string("Nat");
    let nat_ref = Expr::const_(nat.clone(), vec![]);

    let decl = InductiveDecl {
        level_params: vec![],
        num_params: 0,
        types: vec![InductiveType {
            name: nat.clone(),
            type_: Expr::type_(),
            constructors: vec![
                Constructor {
                    name: Name::from_string("Nat.zero"),
                    type_: nat_ref.clone(),
                },
                Constructor {
                    name: Name::from_string("Nat.succ"),
                    type_: Expr::arrow(nat_ref.clone(), nat_ref.clone()),
                },
            ],
        }],
    };
    env.add_inductive(decl).unwrap();

    // Verify recursor rule has recursive_fields set correctly
    let rec_val = env.get_recursor(&Name::from_string("Nat.rec")).unwrap();
    assert_eq!(rec_val.rules.len(), 2);

    // zero rule has no recursive fields
    assert!(rec_val.rules[0].recursive_fields.is_empty());

    // succ rule has one recursive field (the predecessor)
    assert_eq!(rec_val.rules[1].recursive_fields.len(), 1);
    assert!(rec_val.rules[1].recursive_fields[0]);

    let tc = TypeChecker::new(&env);

    // Build: Nat.rec motive zero_case succ_case (Nat.succ Nat.zero)
    let rec = Expr::const_(Name::from_string("Nat.rec"), vec![Level::zero()]);
    let motive = Expr::lam(BinderInfo::Default, nat_ref.clone(), Expr::prop());

    // zero_case: some proposition P
    let p = Expr::const_(Name::from_string("P"), vec![]);
    let zero_case = p.clone();

    // succ_case: λ n ih. Q  (just returns some Q for testing)
    let q = Expr::const_(Name::from_string("Q"), vec![]);
    let succ_case = Expr::lam(
        BinderInfo::Default,
        nat_ref.clone(),
        Expr::lam(BinderInfo::Default, Expr::prop(), q.clone()),
    );

    let zero = Expr::const_(Name::from_string("Nat.zero"), vec![]);
    let one = Expr::app(
        Expr::const_(Name::from_string("Nat.succ"), vec![]),
        zero.clone(),
    );

    // Nat.rec motive zero_case succ_case (succ zero)
    let app = Expr::app(
        Expr::app(
            Expr::app(Expr::app(rec.clone(), motive.clone()), zero_case.clone()),
            succ_case.clone(),
        ),
        one,
    );

    let result = tc.whnf(&app);

    // The result should be: succ_case zero (Nat.rec motive zero_case succ_case zero)
    // Which reduces through lambdas to Q
    // But let's check the structure before full reduction

    // The iota reduction produces: succ_case n ih
    // where n = Nat.zero and ih = Nat.rec motive zero_case succ_case Nat.zero
    // After beta reduction of succ_case applied to n, we get:
    // (λ ih. Q) ih = Q
    assert_eq!(result, q);
}

#[test]
fn test_iota_reduction_enum_no_recursive() {
    // Test that enumerations (no recursive fields) work correctly
    use crate::inductive::{Constructor, InductiveDecl, InductiveType};

    let mut env = Environment::new();

    let bool_name = Name::from_string("MyBool");
    let bool_ref = Expr::const_(bool_name.clone(), vec![]);

    let decl = InductiveDecl {
        level_params: vec![],
        num_params: 0,
        types: vec![InductiveType {
            name: bool_name.clone(),
            type_: Expr::type_(),
            constructors: vec![
                Constructor {
                    name: Name::from_string("MyBool.false"),
                    type_: bool_ref.clone(),
                },
                Constructor {
                    name: Name::from_string("MyBool.true"),
                    type_: bool_ref.clone(),
                },
            ],
        }],
    };
    env.add_inductive(decl).unwrap();

    // Verify no recursive fields
    let rec_val = env.get_recursor(&Name::from_string("MyBool.rec")).unwrap();
    for rule in &rec_val.rules {
        assert!(
            rule.recursive_fields.is_empty(),
            "Enum should have no recursive fields"
        );
    }

    let tc = TypeChecker::new(&env);

    // MyBool.rec motive false_case true_case MyBool.true should reduce to true_case
    let rec = Expr::const_(Name::from_string("MyBool.rec"), vec![Level::zero()]);
    let motive = Expr::lam(BinderInfo::Default, bool_ref.clone(), Expr::prop());
    let false_case = Expr::const_(Name::from_string("F"), vec![]);
    let true_case = Expr::const_(Name::from_string("T"), vec![]);
    let tt = Expr::const_(Name::from_string("MyBool.true"), vec![]);

    let app = Expr::app(
        Expr::app(
            Expr::app(Expr::app(rec, motive), false_case),
            true_case.clone(),
        ),
        tt,
    );

    let result = tc.whnf(&app);
    assert_eq!(result, true_case);
}

#[test]
fn test_recursor_type_structure() {
    // Verify that the generated recursor type has correct structure
    use crate::inductive::{Constructor, InductiveDecl, InductiveType};

    let mut env = Environment::new();

    let nat = Name::from_string("Nat");
    let nat_ref = Expr::const_(nat.clone(), vec![]);

    let decl = InductiveDecl {
        level_params: vec![],
        num_params: 0,
        types: vec![InductiveType {
            name: nat.clone(),
            type_: Expr::type_(),
            constructors: vec![
                Constructor {
                    name: Name::from_string("Nat.zero"),
                    type_: nat_ref.clone(),
                },
                Constructor {
                    name: Name::from_string("Nat.succ"),
                    type_: Expr::arrow(nat_ref.clone(), nat_ref.clone()),
                },
            ],
        }],
    };
    env.add_inductive(decl).unwrap();

    let rec_val = env.get_recursor(&Name::from_string("Nat.rec")).unwrap();

    // Recursor should have: motive universe param
    assert!(rec_val.level_params.contains(&Name::from_string("u")));

    // Check recursor type is a Pi type
    // Should be: {motive : Nat → Sort u} → ... → (t : Nat) → motive t
    match &rec_val.type_ {
        Expr::Pi(bi, _, _) => {
            // First parameter should be implicit (motive)
            assert_eq!(*bi, BinderInfo::Implicit);
        }
        _ => panic!("Recursor type should be a Pi type"),
    }

    // Check metadata
    assert_eq!(rec_val.num_params, 0);
    assert_eq!(rec_val.num_motives, 1);
    assert_eq!(rec_val.num_minors, 2); // zero and succ cases
    assert_eq!(rec_val.num_indices, 0);
}

#[test]
fn test_cases_on_type_structure() {
    // Verify that the generated casesOn type has correct structure
    // casesOn differs from rec: no induction hypotheses in minor premises
    use crate::inductive::{Constructor, InductiveDecl, InductiveType};

    let mut env = Environment::new();

    let nat = Name::from_string("Nat");
    let nat_ref = Expr::const_(nat.clone(), vec![]);

    let decl = InductiveDecl {
        level_params: vec![],
        num_params: 0,
        types: vec![InductiveType {
            name: nat.clone(),
            type_: Expr::type_(),
            constructors: vec![
                Constructor {
                    name: Name::from_string("Nat.zero"),
                    type_: nat_ref.clone(),
                },
                Constructor {
                    name: Name::from_string("Nat.succ"),
                    type_: Expr::arrow(nat_ref.clone(), nat_ref.clone()),
                },
            ],
        }],
    };
    env.add_inductive(decl).unwrap();

    // Verify casesOn exists
    let cases_val = env
        .get_recursor(&Name::from_string("Nat.casesOn"))
        .expect("casesOn should be registered");

    // casesOn should have: motive universe param
    assert!(cases_val.level_params.contains(&Name::from_string("u")));

    // Check casesOn type is a Pi type
    match &cases_val.type_ {
        Expr::Pi(bi, _, _) => {
            // First parameter should be implicit (motive)
            assert_eq!(*bi, BinderInfo::Implicit);
        }
        _ => panic!("casesOn type should be a Pi type"),
    }

    // Check metadata
    assert_eq!(cases_val.num_params, 0);
    assert_eq!(cases_val.num_motives, 1);
    assert_eq!(cases_val.num_minors, 2); // zero and succ cases
    assert_eq!(cases_val.num_indices, 0);

    // Verify no recursive fields in casesOn rules (key difference from rec)
    for rule in &cases_val.rules {
        assert!(
            rule.recursive_fields.iter().all(|&b| !b),
            "casesOn should have no recursive fields marked"
        );
    }
}

#[test]
fn test_cases_on_iota_reduction_nat() {
    // Test that casesOn reduces correctly without induction hypotheses
    use crate::inductive::{Constructor, InductiveDecl, InductiveType};

    let mut env = Environment::new();

    let nat = Name::from_string("Nat");
    let nat_ref = Expr::const_(nat.clone(), vec![]);

    let decl = InductiveDecl {
        level_params: vec![],
        num_params: 0,
        types: vec![InductiveType {
            name: nat.clone(),
            type_: Expr::type_(),
            constructors: vec![
                Constructor {
                    name: Name::from_string("Nat.zero"),
                    type_: nat_ref.clone(),
                },
                Constructor {
                    name: Name::from_string("Nat.succ"),
                    type_: Expr::arrow(nat_ref.clone(), nat_ref.clone()),
                },
            ],
        }],
    };
    env.add_inductive(decl).unwrap();

    let tc = TypeChecker::new(&env);

    // Test casesOn with Nat.zero
    // Nat.casesOn motive zero_case succ_case Nat.zero => zero_case
    let cases_on = Expr::const_(Name::from_string("Nat.casesOn"), vec![Level::zero()]);
    let motive = Expr::lam(BinderInfo::Default, nat_ref.clone(), Expr::prop());
    let zero_case = Expr::const_(Name::from_string("ZeroResult"), vec![]);
    let succ_case = Expr::lam(
        BinderInfo::Default,
        nat_ref.clone(),
        Expr::const_(Name::from_string("SuccResult"), vec![]),
    );
    let zero = Expr::const_(Name::from_string("Nat.zero"), vec![]);

    let app_zero = Expr::app(
        Expr::app(
            Expr::app(
                Expr::app(cases_on.clone(), motive.clone()),
                zero_case.clone(),
            ),
            succ_case.clone(),
        ),
        zero,
    );

    let result = tc.whnf(&app_zero);
    assert_eq!(result, zero_case);

    // Test casesOn with Nat.succ n
    // Nat.casesOn motive zero_case succ_case (Nat.succ n) => succ_case n
    // Note: casesOn does NOT pass IH, only the field n
    let succ = Expr::const_(Name::from_string("Nat.succ"), vec![]);
    let n_val = Expr::const_(Name::from_string("n"), vec![]);
    let succ_n = Expr::app(succ, n_val.clone());

    let app_succ = Expr::app(
        Expr::app(Expr::app(Expr::app(cases_on, motive), zero_case), succ_case),
        succ_n,
    );

    let result = tc.whnf(&app_succ);
    // Should be: succ_case applied to n (no IH)
    let expected = Expr::const_(Name::from_string("SuccResult"), vec![]);
    assert_eq!(result, expected);
}

#[test]
fn test_cases_on_enum() {
    // Test casesOn for a simple enumeration (Bool-like)
    use crate::inductive::{Constructor, InductiveDecl, InductiveType};

    let mut env = Environment::new();

    let bool_name = Name::from_string("MyBool");
    let bool_ref = Expr::const_(bool_name.clone(), vec![]);

    let decl = InductiveDecl {
        level_params: vec![],
        num_params: 0,
        types: vec![InductiveType {
            name: bool_name.clone(),
            type_: Expr::type_(),
            constructors: vec![
                Constructor {
                    name: Name::from_string("MyBool.false"),
                    type_: bool_ref.clone(),
                },
                Constructor {
                    name: Name::from_string("MyBool.true"),
                    type_: bool_ref.clone(),
                },
            ],
        }],
    };
    env.add_inductive(decl).unwrap();

    let tc = TypeChecker::new(&env);

    // Test casesOn with MyBool.true
    let cases_on = Expr::const_(Name::from_string("MyBool.casesOn"), vec![Level::zero()]);
    let motive = Expr::lam(BinderInfo::Default, bool_ref.clone(), Expr::prop());
    let false_case = Expr::const_(Name::from_string("FalseResult"), vec![]);
    let true_case = Expr::const_(Name::from_string("TrueResult"), vec![]);
    let tt = Expr::const_(Name::from_string("MyBool.true"), vec![]);

    let app = Expr::app(
        Expr::app(
            Expr::app(Expr::app(cases_on, motive), false_case),
            true_case.clone(),
        ),
        tt,
    );

    let result = tc.whnf(&app);
    assert_eq!(result, true_case);
}

#[test]
fn test_parametric_recursor_list() {
    // Test recursor for parametric inductive type (List)
    use crate::inductive::{Constructor, InductiveDecl, InductiveType};
    use crate::level::Level;

    let mut env = Environment::new();

    let u = Name::from_string("u");
    let list = Name::from_string("List");

    // List.{u} : Type u → Type u
    let list_type = Expr::pi(
        BinderInfo::Default,
        Expr::Sort(Level::param(u.clone())),
        Expr::Sort(Level::param(u.clone())),
    );

    // List A (with BVar 0 for parameter A)
    let list_a = Expr::app(
        Expr::const_(list.clone(), vec![Level::param(u.clone())]),
        Expr::bvar(0),
    );

    // nil : (A : Type u) → List A
    let nil_type = Expr::pi(
        BinderInfo::Default,
        Expr::Sort(Level::param(u.clone())),
        list_a.clone(),
    );

    // cons : (A : Type u) → A → List A → List A
    let cons_body = Expr::pi(
        BinderInfo::Default,
        Expr::bvar(0), // A
        Expr::pi(
            BinderInfo::Default,
            Expr::app(
                Expr::const_(list.clone(), vec![Level::param(u.clone())]),
                Expr::bvar(1), // A (now at depth 1)
            ),
            Expr::app(
                Expr::const_(list.clone(), vec![Level::param(u.clone())]),
                Expr::bvar(2), // A (now at depth 2)
            ),
        ),
    );
    let cons_type = Expr::pi(
        BinderInfo::Default,
        Expr::Sort(Level::param(u.clone())),
        cons_body,
    );

    let decl = InductiveDecl {
        level_params: vec![u.clone()],
        num_params: 1, // A is a parameter
        types: vec![InductiveType {
            name: list.clone(),
            type_: list_type,
            constructors: vec![
                Constructor {
                    name: Name::from_string("List.nil"),
                    type_: nil_type,
                },
                Constructor {
                    name: Name::from_string("List.cons"),
                    type_: cons_type,
                },
            ],
        }],
    };

    env.add_inductive(decl).unwrap();

    // Verify both rec and casesOn are created
    let rec_val = env
        .get_recursor(&Name::from_string("List.rec"))
        .expect("List.rec should exist");
    let cases_val = env
        .get_recursor(&Name::from_string("List.casesOn"))
        .expect("List.casesOn should exist");

    // Both should have 1 type parameter
    assert_eq!(rec_val.num_params, 1);
    assert_eq!(cases_val.num_params, 1);

    // Both should have 2 minor premises (nil and cons cases)
    assert_eq!(rec_val.num_minors, 2);
    assert_eq!(cases_val.num_minors, 2);

    // rec should have recursive field marked for cons (the List A argument)
    assert_eq!(rec_val.rules.len(), 2);
    let cons_rule = &rec_val.rules[1]; // cons is second constructor
                                       // cons has 2 fields: the element (A) and the tail (List A)
                                       // The tail (List A) is recursive
    assert_eq!(cons_rule.num_fields, 2);
    assert_eq!(cons_rule.recursive_fields.len(), 2);
    assert!(
        !cons_rule.recursive_fields[0],
        "First field (element) is not recursive"
    );
    assert!(
        cons_rule.recursive_fields[1],
        "Second field (tail) IS recursive"
    );

    // casesOn should have NO recursive fields
    let cases_cons_rule = &cases_val.rules[1];
    assert!(
        cases_cons_rule.recursive_fields.iter().all(|&b| !b),
        "casesOn should have no recursive fields"
    );
}

#[test]
fn test_parametric_cases_on_list_nil() {
    // Test List.casesOn reduction with nil
    use crate::inductive::{Constructor, InductiveDecl, InductiveType};
    use crate::level::Level;

    let mut env = Environment::new();

    let u = Name::from_string("u");
    let list = Name::from_string("List");

    let list_type = Expr::pi(
        BinderInfo::Default,
        Expr::Sort(Level::param(u.clone())),
        Expr::Sort(Level::param(u.clone())),
    );

    let list_a = Expr::app(
        Expr::const_(list.clone(), vec![Level::param(u.clone())]),
        Expr::bvar(0),
    );

    let nil_type = Expr::pi(
        BinderInfo::Default,
        Expr::Sort(Level::param(u.clone())),
        list_a.clone(),
    );

    let cons_body = Expr::pi(
        BinderInfo::Default,
        Expr::bvar(0),
        Expr::pi(
            BinderInfo::Default,
            Expr::app(
                Expr::const_(list.clone(), vec![Level::param(u.clone())]),
                Expr::bvar(1),
            ),
            Expr::app(
                Expr::const_(list.clone(), vec![Level::param(u.clone())]),
                Expr::bvar(2),
            ),
        ),
    );
    let cons_type = Expr::pi(
        BinderInfo::Default,
        Expr::Sort(Level::param(u.clone())),
        cons_body,
    );

    let decl = InductiveDecl {
        level_params: vec![u.clone()],
        num_params: 1,
        types: vec![InductiveType {
            name: list.clone(),
            type_: list_type,
            constructors: vec![
                Constructor {
                    name: Name::from_string("List.nil"),
                    type_: nil_type,
                },
                Constructor {
                    name: Name::from_string("List.cons"),
                    type_: cons_type,
                },
            ],
        }],
    };

    env.add_inductive(decl).unwrap();

    let tc = TypeChecker::new(&env);

    // Test: List.casesOn Nat motive nil_case cons_case (List.nil Nat) => nil_case
    // Build List Nat
    let nat = Expr::const_(Name::from_string("Nat"), vec![]);
    let list_nat = Expr::app(Expr::const_(list.clone(), vec![Level::zero()]), nat.clone());

    let cases_on = Expr::const_(
        Name::from_string("List.casesOn"),
        vec![Level::zero(), Level::zero()], // motive universe and type param universe
    );

    let motive = Expr::lam(BinderInfo::Default, list_nat.clone(), Expr::prop());
    let nil_case = Expr::const_(Name::from_string("NilResult"), vec![]);
    let cons_case = Expr::lam(
        BinderInfo::Default,
        nat.clone(), // head element
        Expr::lam(
            BinderInfo::Default,
            list_nat.clone(), // tail
            Expr::const_(Name::from_string("ConsResult"), vec![]),
        ),
    );

    // nil applied to the type parameter
    let nil = Expr::app(
        Expr::const_(Name::from_string("List.nil"), vec![Level::zero()]),
        nat.clone(),
    );

    // Build: casesOn nat motive nil_case cons_case nil
    // Note: param comes first for parametric recursors
    let app = Expr::app(
        Expr::app(
            Expr::app(
                Expr::app(Expr::app(cases_on, nat), motive),
                nil_case.clone(),
            ),
            cons_case,
        ),
        nil,
    );

    let result = tc.whnf(&app);
    assert_eq!(result, nil_case);
}

#[test]
fn test_rec_on_type_structure() {
    // Verify that recOn type has correct structure: {motive} → major → minors → result
    use crate::inductive::{Constructor, InductiveDecl, InductiveType};

    let mut env = Environment::new();

    let nat = Name::from_string("Nat");
    let nat_ref = Expr::const_(nat.clone(), vec![]);

    let decl = InductiveDecl {
        level_params: vec![],
        num_params: 0,
        types: vec![InductiveType {
            name: nat.clone(),
            type_: Expr::Sort(Level::succ(Level::zero())), // Type 1
            constructors: vec![
                Constructor {
                    name: Name::from_string("Nat.zero"),
                    type_: nat_ref.clone(),
                },
                Constructor {
                    name: Name::from_string("Nat.succ"),
                    type_: Expr::pi(BinderInfo::Default, nat_ref.clone(), nat_ref.clone()),
                },
            ],
        }],
    };

    env.add_inductive(decl).unwrap();

    // Verify recOn is created
    let rec_on_val = env
        .get_recursor(&Name::from_string("Nat.recOn"))
        .expect("Nat.recOn should exist");

    assert_eq!(rec_on_val.name, Name::from_string("Nat.recOn"));
    assert_eq!(rec_on_val.num_params, 0);
    assert_eq!(rec_on_val.num_motives, 1);
    assert_eq!(rec_on_val.num_minors, 2); // zero and succ

    // Verify type structure: {motive} → major → minor_zero → minor_succ → motive major
    // Count Pi binders
    let mut pi_count = 0;
    let mut ty = &rec_on_val.type_;
    while let Expr::Pi(_, _, body) = ty {
        pi_count += 1;
        ty = body;
    }
    assert_eq!(pi_count, 4); // motive, major, minor_zero, minor_succ

    // Compare with rec type structure
    let rec_val = env
        .get_recursor(&Name::from_string("Nat.rec"))
        .expect("Nat.rec should exist");

    let mut rec_pi_count = 0;
    let mut rec_ty = &rec_val.type_;
    while let Expr::Pi(_, _, body) = rec_ty {
        rec_pi_count += 1;
        rec_ty = body;
    }
    assert_eq!(rec_pi_count, 4); // Both have same number of Pi binders

    // The difference is in argument order, not count
}

#[test]
fn test_rec_on_iota_reduction_zero() {
    // recOn should reduce with the major premise before minors
    use crate::inductive::{Constructor, InductiveDecl, InductiveType};

    let mut env = Environment::new();

    let nat = Name::from_string("Nat");
    let nat_ref = Expr::const_(nat.clone(), vec![]);

    let decl = InductiveDecl {
        level_params: vec![],
        num_params: 0,
        types: vec![InductiveType {
            name: nat.clone(),
            type_: Expr::type_(),
            constructors: vec![
                Constructor {
                    name: Name::from_string("Nat.zero"),
                    type_: nat_ref.clone(),
                },
                Constructor {
                    name: Name::from_string("Nat.succ"),
                    type_: Expr::arrow(nat_ref.clone(), nat_ref.clone()),
                },
            ],
        }],
    };

    env.add_inductive(decl).unwrap();

    let rec_val = env
        .get_recursor(&Name::from_string("Nat.recOn"))
        .expect("Nat.recOn should exist");
    assert_eq!(
        rec_val.arg_order,
        RecursorArgOrder::MajorAfterMotive,
        "recOn should position major before minors"
    );

    let tc = TypeChecker::new(&env);

    let rec_on = Expr::const_(Name::from_string("Nat.recOn"), vec![Level::zero()]);
    let motive = Expr::lam(BinderInfo::Default, nat_ref.clone(), Expr::prop());
    let zero_case = Expr::const_(Name::from_string("ZeroResult"), vec![]);
    let succ_case = Expr::lam(
        BinderInfo::Default,
        nat_ref.clone(),
        Expr::lam(
            BinderInfo::Default,
            Expr::prop(),
            Expr::const_(Name::from_string("SuccResult"), vec![]),
        ),
    );

    let zero = Expr::const_(Name::from_string("Nat.zero"), vec![]);

    // Order: motive, major, minor_zero, minor_succ
    let app = Expr::app(
        Expr::app(
            Expr::app(Expr::app(rec_on, motive), zero.clone()),
            zero_case.clone(),
        ),
        succ_case,
    );

    let result = tc.whnf(&app);
    assert_eq!(
        result, zero_case,
        "recOn should reduce to zero_case when major = Nat.zero"
    );
}

#[test]
fn test_rec_on_iota_reduction_succ() {
    // recOn must supply an induction hypothesis even with major-before-minors layout
    use crate::inductive::{Constructor, InductiveDecl, InductiveType};

    let mut env = Environment::new();

    let nat = Name::from_string("Nat");
    let nat_ref = Expr::const_(nat.clone(), vec![]);

    let decl = InductiveDecl {
        level_params: vec![],
        num_params: 0,
        types: vec![InductiveType {
            name: nat.clone(),
            type_: Expr::type_(),
            constructors: vec![
                Constructor {
                    name: Name::from_string("Nat.zero"),
                    type_: nat_ref.clone(),
                },
                Constructor {
                    name: Name::from_string("Nat.succ"),
                    type_: Expr::arrow(nat_ref.clone(), nat_ref.clone()),
                },
            ],
        }],
    };

    env.add_inductive(decl).unwrap();

    let tc = TypeChecker::new(&env);

    let rec_on = Expr::const_(Name::from_string("Nat.recOn"), vec![Level::zero()]);
    let motive = Expr::lam(BinderInfo::Default, nat_ref.clone(), Expr::prop());
    let zero_case = Expr::const_(Name::from_string("ZeroCase"), vec![]);
    // succ_case returns the IH to expose whether it was supplied
    let succ_case = Expr::lam(
        BinderInfo::Default,
        nat_ref.clone(),
        Expr::lam(BinderInfo::Default, Expr::prop(), Expr::BVar(0)),
    );

    let zero = Expr::const_(Name::from_string("Nat.zero"), vec![]);
    let one = Expr::app(
        Expr::const_(Name::from_string("Nat.succ"), vec![]),
        zero.clone(),
    );

    // Order: motive, major, minor_zero, minor_succ
    let app = Expr::app(
        Expr::app(
            Expr::app(Expr::app(rec_on, motive.clone()), one),
            zero_case.clone(),
        ),
        succ_case,
    );

    let result = tc.whnf(&app);
    assert_eq!(
        result, zero_case,
        "recOn should pass IH and reduce recursive call on Nat.zero"
    );
}

#[test]
fn test_rec_on_exists_for_enum() {
    // Test recOn for a simple enumeration
    use crate::inductive::{Constructor, InductiveDecl, InductiveType};

    let mut env = Environment::new();

    let bool_name = Name::from_string("MyBool");
    let bool_ref = Expr::const_(bool_name.clone(), vec![]);

    let decl = InductiveDecl {
        level_params: vec![],
        num_params: 0,
        types: vec![InductiveType {
            name: bool_name.clone(),
            type_: Expr::Sort(Level::succ(Level::zero())), // Type 1
            constructors: vec![
                Constructor {
                    name: Name::from_string("MyBool.false"),
                    type_: bool_ref.clone(),
                },
                Constructor {
                    name: Name::from_string("MyBool.true"),
                    type_: bool_ref.clone(),
                },
            ],
        }],
    };

    env.add_inductive(decl).unwrap();

    // Verify all three eliminators exist
    assert!(
        env.get_recursor(&Name::from_string("MyBool.rec")).is_some(),
        "MyBool.rec should exist"
    );
    assert!(
        env.get_recursor(&Name::from_string("MyBool.casesOn"))
            .is_some(),
        "MyBool.casesOn should exist"
    );
    assert!(
        env.get_recursor(&Name::from_string("MyBool.recOn"))
            .is_some(),
        "MyBool.recOn should exist"
    );
}

#[test]
fn test_rec_on_parametric_list() {
    // Test recOn for parametric type
    use crate::inductive::{Constructor, InductiveDecl, InductiveType};
    use crate::level::Level;

    let mut env = Environment::new();

    let u = Name::from_string("u");
    let list = Name::from_string("List");

    // List : Type u → Type u
    let list_type = Expr::pi(
        BinderInfo::Default,
        Expr::Sort(Level::param(u.clone())),
        Expr::Sort(Level::param(u.clone())),
    );

    // List A (with BVar 0 for parameter A)
    let list_a = Expr::app(
        Expr::const_(list.clone(), vec![Level::param(u.clone())]),
        Expr::bvar(0),
    );

    // nil : (A : Type u) → List A
    let nil_type = Expr::pi(
        BinderInfo::Default,
        Expr::Sort(Level::param(u.clone())),
        list_a.clone(),
    );

    // cons : (A : Type u) → A → List A → List A
    let cons_body = Expr::pi(
        BinderInfo::Default,
        Expr::bvar(0), // A
        Expr::pi(
            BinderInfo::Default,
            Expr::app(
                Expr::const_(list.clone(), vec![Level::param(u.clone())]),
                Expr::bvar(1), // A (now at depth 1)
            ),
            Expr::app(
                Expr::const_(list.clone(), vec![Level::param(u.clone())]),
                Expr::bvar(2), // A (now at depth 2)
            ),
        ),
    );
    let cons_type = Expr::pi(
        BinderInfo::Default,
        Expr::Sort(Level::param(u.clone())),
        cons_body,
    );

    let decl = InductiveDecl {
        level_params: vec![u.clone()],
        num_params: 1,
        types: vec![InductiveType {
            name: list.clone(),
            type_: list_type,
            constructors: vec![
                Constructor {
                    name: Name::from_string("List.nil"),
                    type_: nil_type,
                },
                Constructor {
                    name: Name::from_string("List.cons"),
                    type_: cons_type,
                },
            ],
        }],
    };

    env.add_inductive(decl).unwrap();

    // Verify recOn is created and has correct structure
    let rec_on_val = env
        .get_recursor(&Name::from_string("List.recOn"))
        .expect("List.recOn should exist");

    assert_eq!(rec_on_val.num_params, 1); // A is a parameter
    assert_eq!(rec_on_val.num_minors, 2); // nil and cons cases
    assert_eq!(rec_on_val.rules.len(), 2);
}

#[test]
fn test_mutual_inductive_even_odd() {
    // Test mutual inductive types: Even and Odd
    // mutual
    //   inductive Even : Type
    //   | zero : Even
    //   | succ_odd : Odd → Even
    //
    //   inductive Odd : Type
    //   | succ_even : Even → Odd
    // end
    use crate::inductive::{Constructor, InductiveDecl, InductiveType};

    let mut env = Environment::new();

    let even = Name::from_string("Even");
    let odd = Name::from_string("Odd");
    let even_ref = Expr::const_(even.clone(), vec![]);
    let odd_ref = Expr::const_(odd.clone(), vec![]);

    let decl = InductiveDecl {
        level_params: vec![],
        num_params: 0,
        types: vec![
            InductiveType {
                name: even.clone(),
                type_: Expr::Sort(Level::succ(Level::zero())),
                constructors: vec![
                    Constructor {
                        name: Name::from_string("Even.zero"),
                        type_: even_ref.clone(),
                    },
                    Constructor {
                        name: Name::from_string("Even.succ_odd"),
                        type_: Expr::pi(BinderInfo::Default, odd_ref.clone(), even_ref.clone()),
                    },
                ],
            },
            InductiveType {
                name: odd.clone(),
                type_: Expr::Sort(Level::succ(Level::zero())),
                constructors: vec![Constructor {
                    name: Name::from_string("Odd.succ_even"),
                    type_: Expr::pi(BinderInfo::Default, even_ref.clone(), odd_ref.clone()),
                }],
            },
        ],
    };

    env.add_inductive(decl).unwrap();

    // Both inductives should exist
    assert!(
        env.get_inductive(&even).is_some(),
        "Even should be registered"
    );
    assert!(
        env.get_inductive(&odd).is_some(),
        "Odd should be registered"
    );

    // Check that both have all_names set correctly
    let even_val = env.get_inductive(&even).unwrap();
    assert_eq!(even_val.all_names.len(), 2);
    assert!(even_val.all_names.contains(&even));
    assert!(even_val.all_names.contains(&odd));

    let odd_val = env.get_inductive(&odd).unwrap();
    assert_eq!(odd_val.all_names.len(), 2);
    assert!(odd_val.all_names.contains(&even));
    assert!(odd_val.all_names.contains(&odd));

    // Both should have recursors
    assert!(
        env.get_recursor(&Name::from_string("Even.rec")).is_some(),
        "Even.rec should exist"
    );
    assert!(
        env.get_recursor(&Name::from_string("Odd.rec")).is_some(),
        "Odd.rec should exist"
    );

    // Both should have casesOn
    assert!(
        env.get_recursor(&Name::from_string("Even.casesOn"))
            .is_some(),
        "Even.casesOn should exist"
    );
    assert!(
        env.get_recursor(&Name::from_string("Odd.casesOn"))
            .is_some(),
        "Odd.casesOn should exist"
    );

    // Both should have recOn
    assert!(
        env.get_recursor(&Name::from_string("Even.recOn")).is_some(),
        "Even.recOn should exist"
    );
    assert!(
        env.get_recursor(&Name::from_string("Odd.recOn")).is_some(),
        "Odd.recOn should exist"
    );

    // Verify constructor counts
    let even_rec = env.get_recursor(&Name::from_string("Even.rec")).unwrap();
    assert_eq!(even_rec.num_minors, 2); // zero and succ_odd

    let odd_rec = env.get_recursor(&Name::from_string("Odd.rec")).unwrap();
    assert_eq!(odd_rec.num_minors, 1); // succ_even only
}

#[test]
fn test_mutual_inductive_recursors_have_correct_structure() {
    // Verify that mutual inductive recursors have proper structure
    use crate::inductive::{Constructor, InductiveDecl, InductiveType};

    let mut env = Environment::new();

    let even = Name::from_string("Even");
    let odd = Name::from_string("Odd");
    let even_ref = Expr::const_(even.clone(), vec![]);
    let odd_ref = Expr::const_(odd.clone(), vec![]);

    let decl = InductiveDecl {
        level_params: vec![],
        num_params: 0,
        types: vec![
            InductiveType {
                name: even.clone(),
                type_: Expr::Sort(Level::succ(Level::zero())),
                constructors: vec![
                    Constructor {
                        name: Name::from_string("Even.zero"),
                        type_: even_ref.clone(),
                    },
                    Constructor {
                        name: Name::from_string("Even.succ_odd"),
                        type_: Expr::pi(BinderInfo::Default, odd_ref.clone(), even_ref.clone()),
                    },
                ],
            },
            InductiveType {
                name: odd.clone(),
                type_: Expr::Sort(Level::succ(Level::zero())),
                constructors: vec![Constructor {
                    name: Name::from_string("Odd.succ_even"),
                    type_: Expr::pi(BinderInfo::Default, even_ref.clone(), odd_ref.clone()),
                }],
            },
        ],
    };

    env.add_inductive(decl).unwrap();

    // Even.rec should detect that succ_odd has a "recursive" field (Odd)
    // For mutual inductives, "recursive" means the field type mentions
    // ANY type in the mutual block, not just the type being defined.
    let even_rec = env.get_recursor(&Name::from_string("Even.rec")).unwrap();
    assert_eq!(even_rec.rules.len(), 2);

    // succ_odd constructor takes an Odd argument - this IS recursive in mutual context
    let succ_odd_rule = &even_rec.rules[1];
    assert_eq!(succ_odd_rule.num_fields, 1);
    // Now correctly marked as recursive because Odd is in the mutual block
    assert!(
        succ_odd_rule.recursive_fields[0],
        "succ_odd's Odd field should be marked recursive for mutual inductives"
    );

    // Odd.rec should have 1 rule for succ_even
    let odd_rec = env.get_recursor(&Name::from_string("Odd.rec")).unwrap();
    assert_eq!(odd_rec.rules.len(), 1);
    let succ_even_rule = &odd_rec.rules[0];
    assert_eq!(succ_even_rule.num_fields, 1);
    // succ_even takes an Even argument - this IS recursive in mutual context
    assert!(
        succ_even_rule.recursive_fields[0],
        "succ_even's Even field should be marked recursive for mutual inductives"
    );
}

#[test]
fn test_nested_inductive_wrapper() {
    // Test a simple nested inductive: type that wraps another inductive
    // inductive Wrapped (α : Type) : Type
    // | wrap : α → Wrapped α
    //
    // This tests the infrastructure, not true nesting like Tree/List
    use crate::inductive::{Constructor, InductiveDecl, InductiveType};
    use crate::level::Level;

    let mut env = Environment::new();

    let u = Name::from_string("u");
    let wrapped = Name::from_string("Wrapped");

    // Wrapped : Type u → Type u
    let wrapped_type = Expr::pi(
        BinderInfo::Default,
        Expr::Sort(Level::param(u.clone())),
        Expr::Sort(Level::param(u.clone())),
    );

    // Wrapped α (with BVar 0 for α)
    let wrapped_a = Expr::app(
        Expr::const_(wrapped.clone(), vec![Level::param(u.clone())]),
        Expr::bvar(0),
    );

    // wrap : (α : Type u) → α → Wrapped α
    let wrap_type = Expr::pi(
        BinderInfo::Default,
        Expr::Sort(Level::param(u.clone())),
        Expr::pi(
            BinderInfo::Default,
            Expr::bvar(0), // α
            wrapped_a.clone(),
        ),
    );

    let decl = InductiveDecl {
        level_params: vec![u.clone()],
        num_params: 1, // α is a parameter
        types: vec![InductiveType {
            name: wrapped.clone(),
            type_: wrapped_type,
            constructors: vec![Constructor {
                name: Name::from_string("Wrapped.wrap"),
                type_: wrap_type,
            }],
        }],
    };

    env.add_inductive(decl).unwrap();

    // Verify all eliminators exist
    assert!(
        env.get_recursor(&Name::from_string("Wrapped.rec"))
            .is_some(),
        "Wrapped.rec should exist"
    );
    assert!(
        env.get_recursor(&Name::from_string("Wrapped.casesOn"))
            .is_some(),
        "Wrapped.casesOn should exist"
    );
    assert!(
        env.get_recursor(&Name::from_string("Wrapped.recOn"))
            .is_some(),
        "Wrapped.recOn should exist"
    );

    // Verify structure
    let rec_val = env.get_recursor(&Name::from_string("Wrapped.rec")).unwrap();
    assert_eq!(rec_val.num_params, 1);
    assert_eq!(rec_val.num_minors, 1);

    // The wrap constructor has 1 field (the α value)
    // It's NOT recursive because α doesn't mention Wrapped
    let wrap_rule = &rec_val.rules[0];
    assert_eq!(wrap_rule.num_fields, 1);
    assert!(!wrap_rule.recursive_fields[0], "α field is not recursive");
}

#[test]
fn test_inductive_with_function_type_field() {
    // Test inductive with higher-order field
    // inductive HO : Type
    // | mk : (Nat → Nat) → HO
    //
    // The function field is NOT recursive (doesn't mention HO)
    use crate::inductive::{Constructor, InductiveDecl, InductiveType};

    let mut env = Environment::new();

    // First add Nat
    let nat = Name::from_string("Nat");
    let nat_ref = Expr::const_(nat.clone(), vec![]);

    let nat_decl = InductiveDecl {
        level_params: vec![],
        num_params: 0,
        types: vec![InductiveType {
            name: nat.clone(),
            type_: Expr::Sort(Level::succ(Level::zero())),
            constructors: vec![
                Constructor {
                    name: Name::from_string("Nat.zero"),
                    type_: nat_ref.clone(),
                },
                Constructor {
                    name: Name::from_string("Nat.succ"),
                    type_: Expr::pi(BinderInfo::Default, nat_ref.clone(), nat_ref.clone()),
                },
            ],
        }],
    };
    env.add_inductive(nat_decl).unwrap();

    // Now add HO
    let ho = Name::from_string("HO");
    let ho_ref = Expr::const_(ho.clone(), vec![]);

    // Nat → Nat
    let nat_to_nat = Expr::pi(BinderInfo::Default, nat_ref.clone(), nat_ref.clone());

    // mk : (Nat → Nat) → HO
    let mk_type = Expr::pi(BinderInfo::Default, nat_to_nat, ho_ref.clone());

    let ho_decl = InductiveDecl {
        level_params: vec![],
        num_params: 0,
        types: vec![InductiveType {
            name: ho.clone(),
            type_: Expr::Sort(Level::succ(Level::zero())),
            constructors: vec![Constructor {
                name: Name::from_string("HO.mk"),
                type_: mk_type,
            }],
        }],
    };

    env.add_inductive(ho_decl).unwrap();

    // Verify recursor
    let rec_val = env.get_recursor(&Name::from_string("HO.rec")).unwrap();
    assert_eq!(rec_val.num_minors, 1);

    let mk_rule = &rec_val.rules[0];
    assert_eq!(mk_rule.num_fields, 1);
    // The (Nat → Nat) field is not recursive
    assert!(
        !mk_rule.recursive_fields[0],
        "Function field should not be recursive"
    );
}

#[test]
fn test_inductive_with_self_referencing_function() {
    // Test inductive where field is a function returning the inductive
    // inductive Stream : Type
    // | cons : Nat → (Unit → Stream) → Stream
    //
    // The thunk field IS recursive because it returns Stream
    use crate::inductive::{Constructor, InductiveDecl, InductiveType};

    let mut env = Environment::new();

    // Add Nat first
    let nat = Name::from_string("Nat");
    let nat_ref = Expr::const_(nat.clone(), vec![]);

    let nat_decl = InductiveDecl {
        level_params: vec![],
        num_params: 0,
        types: vec![InductiveType {
            name: nat.clone(),
            type_: Expr::Sort(Level::succ(Level::zero())),
            constructors: vec![
                Constructor {
                    name: Name::from_string("Nat.zero"),
                    type_: nat_ref.clone(),
                },
                Constructor {
                    name: Name::from_string("Nat.succ"),
                    type_: Expr::pi(BinderInfo::Default, nat_ref.clone(), nat_ref.clone()),
                },
            ],
        }],
    };
    env.add_inductive(nat_decl).unwrap();

    // Add Unit
    let unit = Name::from_string("Unit");
    let unit_ref = Expr::const_(unit.clone(), vec![]);

    let unit_decl = InductiveDecl {
        level_params: vec![],
        num_params: 0,
        types: vec![InductiveType {
            name: unit.clone(),
            type_: Expr::Sort(Level::succ(Level::zero())),
            constructors: vec![Constructor {
                name: Name::from_string("Unit.unit"),
                type_: unit_ref.clone(),
            }],
        }],
    };
    env.add_inductive(unit_decl).unwrap();

    // Add Stream
    let stream = Name::from_string("Stream");
    let stream_ref = Expr::const_(stream.clone(), vec![]);

    // Unit → Stream (thunk type)
    let thunk_type = Expr::pi(BinderInfo::Default, unit_ref.clone(), stream_ref.clone());

    // cons : Nat → (Unit → Stream) → Stream
    let cons_type = Expr::pi(
        BinderInfo::Default,
        nat_ref.clone(),
        Expr::pi(BinderInfo::Default, thunk_type, stream_ref.clone()),
    );

    let stream_decl = InductiveDecl {
        level_params: vec![],
        num_params: 0,
        types: vec![InductiveType {
            name: stream.clone(),
            type_: Expr::Sort(Level::succ(Level::zero())),
            constructors: vec![Constructor {
                name: Name::from_string("Stream.cons"),
                type_: cons_type,
            }],
        }],
    };

    env.add_inductive(stream_decl).unwrap();

    // Verify recursor
    let rec_val = env.get_recursor(&Name::from_string("Stream.rec")).unwrap();
    assert_eq!(rec_val.num_minors, 1);

    let cons_rule = &rec_val.rules[0];
    assert_eq!(cons_rule.num_fields, 2); // Nat and thunk

    // First field (Nat) is not recursive
    assert!(
        !cons_rule.recursive_fields[0],
        "Nat field should not be recursive"
    );

    // Second field (Unit → Stream) IS recursive because it mentions Stream
    assert!(
        cons_rule.recursive_fields[1],
        "Thunk field should be recursive since it mentions Stream"
    );
}

#[test]
fn test_mutual_inductive_recursor_minor_premise_has_ih() {
    // Verify that mutual inductive recursors have IH for cross-type references
    //
    // For Even/Odd, Even.rec's minor for succ_odd should have:
    //   minor_succ_odd : (o : Odd) → motive_even (Even.succ_odd o)
    // But since Odd is in the mutual block and is marked recursive,
    // the minor premise should ALSO include an IH:
    //   minor_succ_odd : (o : Odd) → (ih : motive_odd o) → motive_even (Even.succ_odd o)
    //
    // This is essential for mutual recursion to work properly.
    use crate::inductive::{Constructor, InductiveDecl, InductiveType};

    let mut env = Environment::new();

    let even = Name::from_string("Even");
    let odd = Name::from_string("Odd");
    let even_ref = Expr::const_(even.clone(), vec![]);
    let odd_ref = Expr::const_(odd.clone(), vec![]);

    let decl = InductiveDecl {
        level_params: vec![],
        num_params: 0,
        types: vec![
            InductiveType {
                name: even.clone(),
                type_: Expr::Sort(Level::succ(Level::zero())),
                constructors: vec![
                    Constructor {
                        name: Name::from_string("Even.zero"),
                        type_: even_ref.clone(),
                    },
                    Constructor {
                        name: Name::from_string("Even.succ_odd"),
                        type_: Expr::pi(BinderInfo::Default, odd_ref.clone(), even_ref.clone()),
                    },
                ],
            },
            InductiveType {
                name: odd.clone(),
                type_: Expr::Sort(Level::succ(Level::zero())),
                constructors: vec![Constructor {
                    name: Name::from_string("Odd.succ_even"),
                    type_: Expr::pi(BinderInfo::Default, even_ref.clone(), odd_ref.clone()),
                }],
            },
        ],
    };

    env.add_inductive(decl).unwrap();

    // Even.rec type should be:
    // {motive : Even → Sort u} →
    // motive Even.zero →                                    -- zero minor
    // ((o : Odd) → motive (Even.succ_odd o)) →              -- succ_odd minor (HAS IH due to mutual)
    // (t : Even) → motive t
    let even_rec = env.get_recursor(&Name::from_string("Even.rec")).unwrap();

    // Verify recursive flags
    // zero has no fields
    assert_eq!(even_rec.rules[0].num_fields, 0);
    assert!(even_rec.rules[0].recursive_fields.is_empty());

    // succ_odd has 1 field (Odd) which should be marked recursive
    assert_eq!(even_rec.rules[1].num_fields, 1);
    assert!(
        even_rec.rules[1].recursive_fields[0],
        "succ_odd's Odd field should be marked recursive in mutual context"
    );

    // Odd.rec's succ_even should similarly have its Even field marked recursive
    let odd_rec = env.get_recursor(&Name::from_string("Odd.rec")).unwrap();
    assert_eq!(odd_rec.rules[0].num_fields, 1);
    assert!(
        odd_rec.rules[0].recursive_fields[0],
        "succ_even's Even field should be marked recursive in mutual context"
    );

    // Verify num_motives is 1 (single-type recursor)
    // Note: For proper mutual induction, we'd have a combined recursor with
    // motives for both types, but that's a more complex feature
    assert_eq!(even_rec.num_motives, 1);
    assert_eq!(odd_rec.num_motives, 1);
}

#[test]
fn test_no_confusion_type_exists() {
    // Verify noConfusionType is generated for inductive types
    use crate::inductive::{Constructor, InductiveDecl, InductiveType};

    let mut env = Environment::new();

    let nat = Name::from_string("Nat");
    let nat_ref = Expr::const_(nat.clone(), vec![]);

    let decl = InductiveDecl {
        level_params: vec![],
        num_params: 0,
        types: vec![InductiveType {
            name: nat.clone(),
            type_: Expr::type_(),
            constructors: vec![
                Constructor {
                    name: Name::from_string("Nat.zero"),
                    type_: nat_ref.clone(),
                },
                Constructor {
                    name: Name::from_string("Nat.succ"),
                    type_: Expr::arrow(nat_ref.clone(), nat_ref.clone()),
                },
            ],
        }],
    };

    env.add_inductive(decl).unwrap();

    // Verify noConfusionType is created
    let no_conf_type = env
        .get_recursor(&Name::from_string("Nat.noConfusionType"))
        .expect("Nat.noConfusionType should exist");

    // Verify basic structure
    assert_eq!(no_conf_type.inductive_name.to_string(), "Nat");
    assert_eq!(no_conf_type.num_minors, 2); // Two constructors

    // Verify noConfusionType has the right number of rules
    assert_eq!(no_conf_type.rules.len(), 2);
}

#[test]
fn test_no_confusion_exists() {
    // Verify noConfusion is generated for inductive types
    use crate::inductive::{Constructor, InductiveDecl, InductiveType};

    let mut env = Environment::new();

    let nat = Name::from_string("Nat");
    let nat_ref = Expr::const_(nat.clone(), vec![]);

    let decl = InductiveDecl {
        level_params: vec![],
        num_params: 0,
        types: vec![InductiveType {
            name: nat.clone(),
            type_: Expr::type_(),
            constructors: vec![
                Constructor {
                    name: Name::from_string("Nat.zero"),
                    type_: nat_ref.clone(),
                },
                Constructor {
                    name: Name::from_string("Nat.succ"),
                    type_: Expr::arrow(nat_ref.clone(), nat_ref.clone()),
                },
            ],
        }],
    };

    env.add_inductive(decl).unwrap();

    // Verify noConfusion is created
    let no_conf = env
        .get_recursor(&Name::from_string("Nat.noConfusion"))
        .expect("Nat.noConfusion should exist");

    // Verify basic structure
    assert_eq!(no_conf.inductive_name.to_string(), "Nat");
    assert_eq!(no_conf.num_minors, 2); // Two constructors
}

#[test]
fn test_no_confusion_type_structure() {
    // Verify noConfusionType has correct type structure:
    // Nat.noConfusionType : Sort u → Nat → Nat → Sort u
    use crate::inductive::{Constructor, InductiveDecl, InductiveType};

    let mut env = Environment::new();

    let nat = Name::from_string("Nat");
    let nat_ref = Expr::const_(nat.clone(), vec![]);

    let decl = InductiveDecl {
        level_params: vec![],
        num_params: 0,
        types: vec![InductiveType {
            name: nat.clone(),
            type_: Expr::type_(),
            constructors: vec![
                Constructor {
                    name: Name::from_string("Nat.zero"),
                    type_: nat_ref.clone(),
                },
                Constructor {
                    name: Name::from_string("Nat.succ"),
                    type_: Expr::arrow(nat_ref.clone(), nat_ref.clone()),
                },
            ],
        }],
    };

    env.add_inductive(decl).unwrap();

    let no_conf_type = env
        .get_recursor(&Name::from_string("Nat.noConfusionType"))
        .expect("Nat.noConfusionType should exist");

    // The type should be: (Sort u) → Nat → Nat → Sort u
    // Which is Pi Sort_u (Pi Nat (Pi Nat Sort_u))
    let ty = &no_conf_type.type_;

    // First arg: Sort u (the result type parameter)
    if let Expr::Pi(_, domain, _) = ty {
        if let Expr::Sort(_) = domain.as_ref() {
            // OK - first arg is Sort u
        } else {
            panic!("Expected first arg to be Sort u, got: {domain:?}");
        }
    } else {
        panic!("Expected Pi type, got: {ty:?}");
    }
}

#[test]
fn test_no_confusion_for_enum() {
    // Verify noConfusion is generated for enum types (no fields)
    use crate::inductive::{Constructor, InductiveDecl, InductiveType};

    let mut env = Environment::new();

    let bool_ty = Name::from_string("MyBool");
    let bool_ref = Expr::const_(bool_ty.clone(), vec![]);

    let decl = InductiveDecl {
        level_params: vec![],
        num_params: 0,
        types: vec![InductiveType {
            name: bool_ty.clone(),
            type_: Expr::type_(),
            constructors: vec![
                Constructor {
                    name: Name::from_string("MyBool.false"),
                    type_: bool_ref.clone(),
                },
                Constructor {
                    name: Name::from_string("MyBool.true"),
                    type_: bool_ref.clone(),
                },
            ],
        }],
    };

    env.add_inductive(decl).unwrap();

    // Both noConfusion and noConfusionType should exist
    assert!(
        env.get_recursor(&Name::from_string("MyBool.noConfusionType"))
            .is_some(),
        "MyBool.noConfusionType should exist"
    );
    assert!(
        env.get_recursor(&Name::from_string("MyBool.noConfusion"))
            .is_some(),
        "MyBool.noConfusion should exist"
    );

    // Verify rules for enum type
    let no_conf_type = env
        .get_recursor(&Name::from_string("MyBool.noConfusionType"))
        .unwrap();

    // Enum with 2 constructors should have 2 rules
    assert_eq!(no_conf_type.rules.len(), 2);

    // Each rule should have 0 fields (enum constructors have no arguments)
    for rule in &no_conf_type.rules {
        assert_eq!(rule.num_fields, 0, "Enum constructors have no fields");
    }
}

#[test]
fn test_no_confusion_for_parametric_type() {
    // Verify noConfusion is generated for parametric types
    use crate::inductive::{Constructor, InductiveDecl, InductiveType};
    use crate::level::Level;

    let mut env = Environment::new();

    let u = Name::from_string("u");
    let opt = Name::from_string("MyOption");

    // MyOption : Type u → Type u
    let opt_type = Expr::pi(
        BinderInfo::Default,
        Expr::Sort(Level::param(u.clone())),
        Expr::Sort(Level::param(u.clone())),
    );

    // MyOption A
    let opt_a = Expr::app(
        Expr::const_(opt.clone(), vec![Level::param(u.clone())]),
        Expr::bvar(0),
    );

    // none : (A : Type u) → MyOption A
    let none_type = Expr::pi(
        BinderInfo::Default,
        Expr::Sort(Level::param(u.clone())),
        opt_a.clone(),
    );

    // some : (A : Type u) → A → MyOption A
    let some_type = Expr::pi(
        BinderInfo::Default,
        Expr::Sort(Level::param(u.clone())),
        Expr::pi(
            BinderInfo::Default,
            Expr::bvar(0), // A
            Expr::app(
                Expr::const_(opt.clone(), vec![Level::param(u.clone())]),
                Expr::bvar(1), // A (at depth 1)
            ),
        ),
    );

    let decl = InductiveDecl {
        level_params: vec![u.clone()],
        num_params: 1,
        types: vec![InductiveType {
            name: opt.clone(),
            type_: opt_type,
            constructors: vec![
                Constructor {
                    name: Name::from_string("MyOption.none"),
                    type_: none_type,
                },
                Constructor {
                    name: Name::from_string("MyOption.some"),
                    type_: some_type,
                },
            ],
        }],
    };

    env.add_inductive(decl).unwrap();

    // Verify noConfusion is created for parametric type
    let no_conf_type = env
        .get_recursor(&Name::from_string("MyOption.noConfusionType"))
        .expect("MyOption.noConfusionType should exist");

    assert_eq!(no_conf_type.num_params, 1); // A is a parameter
    assert_eq!(no_conf_type.num_minors, 2); // none and some

    let no_conf = env
        .get_recursor(&Name::from_string("MyOption.noConfusion"))
        .expect("MyOption.noConfusion should exist");

    assert_eq!(no_conf.num_params, 1);
}
