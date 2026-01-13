//! Quotient Types
//!
//! Lean's quotient types allow construction of types modulo an equivalence relation.
//! This module implements the four quotient primitives:
//!
//! - `Quot.{u} {α : Sort u} (r : α → α → Prop) : Sort u` - The quotient type
//! - `Quot.mk.{u} {α : Sort u} (r : α → α → Prop) (a : α) : @Quot.{u} α r` - Constructor
//! - `Quot.lift.{u v} {α : Sort u} {r : α → α → Prop} {β : Sort v} (f : α → β) :
//!     (∀ a b : α, r a b → f a = f b) → @Quot.{u} α r → β` - Eliminator
//! - `Quot.ind.{u} {α : Sort u} {r : α → α → Prop} {β : @Quot.{u} α r → Prop} :
//!     (∀ a : α, β (@Quot.mk.{u} α r a)) → ∀ q : @Quot.{u} α r, β q` - Induction
//!
//! The key computation rule (iota/quot reduction):
//! `Quot.lift f h (Quot.mk r a) ≡ f a`
//!
//! This means when `lift` is applied to a `mk`, we can reduce directly to the function
//! application, discarding the proof obligation.
//!
//! References:
//! - Lean 4 kernel: src/kernel/quot.cpp
//! - lean4lean: Lean4Lean/Quot.lean

use crate::expr::{BinderInfo, Expr};
use crate::level::Level;
use crate::name::Name;
use serde::{Deserialize, Serialize};

/// Names of the quotient constants
pub mod names {
    use crate::name::Name;
    use std::sync::LazyLock;

    pub static QUOT: LazyLock<Name> = LazyLock::new(|| Name::from_string("Quot"));
    pub static QUOT_MK: LazyLock<Name> = LazyLock::new(|| Name::from_string("Quot.mk"));
    pub static QUOT_LIFT: LazyLock<Name> = LazyLock::new(|| Name::from_string("Quot.lift"));
    pub static QUOT_IND: LazyLock<Name> = LazyLock::new(|| Name::from_string("Quot.ind"));
}

/// Information about a quotient primitive
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuotVal {
    /// Name of the quotient primitive
    pub name: Name,
    /// Universe parameters
    pub level_params: Vec<Name>,
    /// Type of the primitive
    pub type_: Expr,
    /// Which quotient primitive this is
    pub kind: QuotKind,
}

/// The kind of quotient primitive
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuotKind {
    /// Quot - the quotient type former
    Type,
    /// Quot.mk - the constructor
    Mk,
    /// Quot.lift - the eliminator/recursor
    Lift,
    /// Quot.ind - induction principle
    Ind,
}

impl QuotKind {
    /// Get the name for this quotient kind
    pub fn name(&self) -> Name {
        match self {
            QuotKind::Type => names::QUOT.clone(),
            QuotKind::Mk => names::QUOT_MK.clone(),
            QuotKind::Lift => names::QUOT_LIFT.clone(),
            QuotKind::Ind => names::QUOT_IND.clone(),
        }
    }
}

/// Build the type of `Quot`:
/// `Quot.{u} : {α : Sort u} → (r : α → α → Prop) → Sort u`
pub fn quot_type(u: &Name) -> Expr {
    let sort_u = Expr::Sort(Level::param(u.clone()));
    // {α : Sort u}
    let alpha = Expr::bvar(0);
    // r : α → α → Prop
    let r_type = Expr::pi(
        BinderInfo::Default,
        alpha.clone(),
        Expr::pi(BinderInfo::Default, Expr::bvar(1), Expr::prop()),
    );
    // The result type: Sort u
    let result = Expr::Sort(Level::param(u.clone()));

    // Build: {α : Sort u} → (r : α → α → Prop) → Sort u
    Expr::pi(
        BinderInfo::Implicit,
        sort_u,
        Expr::pi(BinderInfo::Default, r_type, result),
    )
}

/// Build the type of `Quot.mk`:
/// `Quot.mk.{u} : {α : Sort u} → (r : α → α → Prop) → (a : α) → @Quot.{u} α r`
pub fn quot_mk_type(u: &Name) -> Expr {
    let sort_u = Expr::Sort(Level::param(u.clone()));

    // α is BVar 2 after binding α, r, a
    // r is BVar 1 after binding α, r, a
    // a is BVar 0 after binding α, r, a

    // r : α → α → Prop (α is BVar 0 at this point)
    let r_type = Expr::pi(
        BinderInfo::Default,
        Expr::bvar(0), // α
        Expr::pi(BinderInfo::Default, Expr::bvar(1), Expr::prop()),
    );

    // Build @Quot.{u} α r where α is BVar 2, r is BVar 1
    let quot_app = Expr::app(
        Expr::app(
            Expr::const_(names::QUOT.clone(), vec![Level::param(u.clone())]),
            Expr::bvar(2), // α
        ),
        Expr::bvar(1), // r
    );

    // Build: {α : Sort u} → (r : α → α → Prop) → (a : α) → @Quot.{u} α r
    Expr::pi(
        BinderInfo::Implicit,
        sort_u,
        Expr::pi(
            BinderInfo::Default,
            r_type,
            Expr::pi(
                BinderInfo::Default,
                Expr::bvar(1), // a : α (α is now BVar 1)
                quot_app,
            ),
        ),
    )
}

/// Build the type of `Quot.lift`:
/// `Quot.lift.{u v} : {α : Sort u} → {r : α → α → Prop} → {β : Sort v} →
///   (f : α → β) → (∀ a b : α, r a b → f a = f b) → @Quot.{u} α r → β`
pub fn quot_lift_type(u: &Name, v: &Name) -> Expr {
    let sort_u = Expr::Sort(Level::param(u.clone()));
    let sort_v = Expr::Sort(Level::param(v.clone()));

    // r : α → α → Prop (α is BVar 0 at this point)
    let r_type = Expr::pi(
        BinderInfo::Default,
        Expr::bvar(0), // α
        Expr::pi(BinderInfo::Default, Expr::bvar(1), Expr::prop()),
    );

    // After binding α, r, β: α is BVar 2, r is BVar 1, β is BVar 0
    // f : α → β
    let f_type = Expr::pi(BinderInfo::Default, Expr::bvar(2), Expr::bvar(1));

    // Build the proof obligation type:
    // ∀ a b : α, r a b → f a = f b
    // After binding α, r, β, f: α is BVar 3, r is BVar 2, β is BVar 1, f is BVar 0
    // In the proof type, we bind a, b internally
    let proof_type = build_lift_proof_type();

    // @Quot.{u} α r
    // After all bindings: α is BVar 4, r is BVar 3
    let quot_type_app = Expr::app(
        Expr::app(
            Expr::const_(names::QUOT.clone(), vec![Level::param(u.clone())]),
            Expr::bvar(4), // α
        ),
        Expr::bvar(3), // r
    );

    // Result type: β (which is BVar 2 after binding α, r, β, f, proof)
    let result = Expr::bvar(2);

    // Build the full type with all binders
    Expr::pi(
        BinderInfo::Implicit,
        sort_u, // α : Sort u
        Expr::pi(
            BinderInfo::Implicit,
            r_type, // r : α → α → Prop
            Expr::pi(
                BinderInfo::Implicit,
                sort_v, // β : Sort v
                Expr::pi(
                    BinderInfo::Default,
                    f_type, // f : α → β
                    Expr::pi(
                        BinderInfo::Default,
                        proof_type, // proof : ∀ a b, r a b → f a = f b
                        Expr::pi(
                            BinderInfo::Default,
                            quot_type_app, // q : @Quot α r
                            result,        // β
                        ),
                    ),
                ),
            ),
        ),
    )
}

/// Build the proof obligation type for Quot.lift:
/// `∀ a b : α, r a b → f a = f b`
/// At the point this is used: α is BVar 3, r is BVar 2, β is BVar 1, f is BVar 0
fn build_lift_proof_type() -> Expr {
    // α at BVar 3
    let alpha = Expr::bvar(3);
    // r at BVar 2 (used in the body after binding)
    let _r = Expr::bvar(2);
    // f at BVar 0 (used in the body after binding)
    let _f = Expr::bvar(0);

    // After binding a, b: a is BVar 1, b is BVar 0
    // α becomes BVar 5, r becomes BVar 4, f becomes BVar 2
    // r a b (application)
    let r_a_b = Expr::app(Expr::app(Expr::bvar(4), Expr::bvar(1)), Expr::bvar(0));

    // f a, f b (applications)
    // f becomes BVar 3 after binding a, b, h
    let f_a = Expr::app(Expr::bvar(3), Expr::bvar(2)); // in the body after binding h
    let f_b = Expr::app(Expr::bvar(3), Expr::bvar(1)); // after binding h

    // f a = f b (using Eq)
    // This is a bit complex - we need the Eq type
    // In Lean: @Eq β (f a) (f b)
    // β is BVar 3 at top level, becomes BVar 5 after binding a, b, then BVar 6 after h
    let eq_type = make_eq_type(Expr::bvar(5), f_a, f_b);

    // Build: ∀ a b : α, r a b → f a = f b
    // Note: α is BVar 3 -> becomes BVar 4 after binding a -> becomes BVar 5 after binding b
    Expr::pi(
        BinderInfo::Default,
        alpha.clone(), // a : α
        Expr::pi(
            BinderInfo::Default,
            Expr::bvar(4), // b : α (α shifted by 1)
            Expr::pi(
                BinderInfo::Default,
                r_a_b,   // h : r a b
                eq_type, // f a = f b
            ),
        ),
    )
}

/// Build @Eq.{v} β a b
/// This creates the type for equality of two terms in type β
fn make_eq_type(beta: Expr, a: Expr, b: Expr) -> Expr {
    // Construct @Eq β a b
    // Eq is universe-polymorphic but lives in Prop, so no universe arguments needed here
    Expr::app(
        Expr::app(
            Expr::app(
                // Eq requires the type as first argument
                Expr::const_(Name::from_string("Eq"), vec![]), // Eq is in Prop
                beta,
            ),
            a,
        ),
        b,
    )
}

/// Build the type of `Quot.ind`:
/// `Quot.ind.{u} : {α : Sort u} → {r : α → α → Prop} →
///   {β : @Quot.{u} α r → Prop} →
///   (∀ a : α, β (@Quot.mk.{u} α r a)) → ∀ q : @Quot.{u} α r, β q`
pub fn quot_ind_type(u: &Name) -> Expr {
    let sort_u = Expr::Sort(Level::param(u.clone()));

    // r : α → α → Prop
    let r_type = Expr::pi(
        BinderInfo::Default,
        Expr::bvar(0),
        Expr::pi(BinderInfo::Default, Expr::bvar(1), Expr::prop()),
    );

    // @Quot.{u} α r (after binding α, r)
    // α is BVar 1, r is BVar 0
    let quot_alpha_r = Expr::app(
        Expr::app(
            Expr::const_(names::QUOT.clone(), vec![Level::param(u.clone())]),
            Expr::bvar(1),
        ),
        Expr::bvar(0),
    );

    // β : @Quot.{u} α r → Prop (after binding α, r)
    // When used later, shifts accordingly
    let beta_type = Expr::pi(BinderInfo::Default, quot_alpha_r.clone(), Expr::prop());

    // Build the induction hypothesis type:
    // ∀ a : α, β (@Quot.mk.{u} α r a)
    // After binding α, r, β: α is BVar 2, r is BVar 1, β is BVar 0
    let ih_type = build_ind_hyp_type(u);

    // @Quot.{u} α r for the final argument
    // After binding α, r, β, h: α is BVar 3, r is BVar 2
    let quot_final = Expr::app(
        Expr::app(
            Expr::const_(names::QUOT.clone(), vec![Level::param(u.clone())]),
            Expr::bvar(3),
        ),
        Expr::bvar(2),
    );

    // β q (β is BVar 1 after binding α, r, β, h, q)
    let beta_q = Expr::app(Expr::bvar(1), Expr::bvar(0));

    // Build the full type
    Expr::pi(
        BinderInfo::Implicit,
        sort_u,
        Expr::pi(
            BinderInfo::Implicit,
            r_type,
            Expr::pi(
                BinderInfo::Implicit,
                beta_type,
                Expr::pi(
                    BinderInfo::Default,
                    ih_type,
                    Expr::pi(BinderInfo::Default, quot_final, beta_q),
                ),
            ),
        ),
    )
}

/// Build the induction hypothesis type:
/// `∀ a : α, β (@Quot.mk.{u} α r a)`
/// At the point this is used: α is BVar 2, r is BVar 1, β is BVar 0
fn build_ind_hyp_type(u: &Name) -> Expr {
    // α is BVar 2
    let alpha = Expr::bvar(2);

    // After binding a: α is BVar 3, r is BVar 2, β is BVar 1, a is BVar 0
    // @Quot.mk.{u} α r a
    let mk_a = Expr::app(
        Expr::app(
            Expr::app(
                Expr::const_(names::QUOT_MK.clone(), vec![Level::param(u.clone())]),
                Expr::bvar(3), // α
            ),
            Expr::bvar(2), // r
        ),
        Expr::bvar(0), // a
    );

    // β (@Quot.mk α r a) where β is BVar 1 after binding a
    let beta_mk_a = Expr::app(Expr::bvar(1), mk_a);

    // ∀ a : α, β (@Quot.mk α r a)
    Expr::pi(BinderInfo::Default, alpha, beta_mk_a)
}

/// Initialize the quotient types in an environment
/// Returns the four QuotVal primitives to be added
pub fn init_quot_vals() -> Vec<QuotVal> {
    let u = Name::from_string("u");
    let v = Name::from_string("v");

    vec![
        QuotVal {
            name: names::QUOT.clone(),
            level_params: vec![u.clone()],
            type_: quot_type(&u),
            kind: QuotKind::Type,
        },
        QuotVal {
            name: names::QUOT_MK.clone(),
            level_params: vec![u.clone()],
            type_: quot_mk_type(&u),
            kind: QuotKind::Mk,
        },
        QuotVal {
            name: names::QUOT_LIFT.clone(),
            level_params: vec![u.clone(), v.clone()],
            type_: quot_lift_type(&u, &v),
            kind: QuotKind::Lift,
        },
        QuotVal {
            name: names::QUOT_IND.clone(),
            level_params: vec![u.clone()],
            type_: quot_ind_type(&u),
            kind: QuotKind::Ind,
        },
    ]
}

/// Check if a name is a quotient primitive
pub fn is_quot_name(name: &Name) -> bool {
    *name == *names::QUOT
        || *name == *names::QUOT_MK
        || *name == *names::QUOT_LIFT
        || *name == *names::QUOT_IND
}

/// Get the QuotKind for a name, if it's a quotient primitive
pub fn get_quot_kind(name: &Name) -> Option<QuotKind> {
    if *name == *names::QUOT {
        Some(QuotKind::Type)
    } else if *name == *names::QUOT_MK {
        Some(QuotKind::Mk)
    } else if *name == *names::QUOT_LIFT {
        Some(QuotKind::Lift)
    } else if *name == *names::QUOT_IND {
        Some(QuotKind::Ind)
    } else {
        None
    }
}

/// Try to reduce a Quot.lift application
///
/// The reduction rule is:
/// `Quot.lift.{u v} α r β f h (Quot.mk.{u} α r a) ≡ f a`
///
/// Returns Some(reduced) if the expression can be reduced, None otherwise.
pub fn try_quot_lift_reduction(
    fn_head: &Expr,
    args: &[&Expr],
    whnf: impl Fn(&Expr) -> Expr,
) -> Option<Expr> {
    // Check if the head is Quot.lift
    if let Expr::Const(name, _levels) = fn_head {
        if *name != *names::QUOT_LIFT {
            return None;
        }
    } else {
        return None;
    }

    // Quot.lift has 6 arguments: α, r, β, f, h, q
    // The 6th argument (q) should be Quot.mk applied to something
    if args.len() < 6 {
        return None;
    }

    // Get the major premise (the quotient value)
    let major = args[5];
    let major_whnf = whnf(major);

    // Check if major is Quot.mk applied to arguments
    let major_head = major_whnf.get_app_fn();
    let major_args = major_whnf.get_app_args();

    if let Expr::Const(name, _) = major_head {
        if *name != *names::QUOT_MK {
            return None;
        }
    } else {
        return None;
    }

    // Quot.mk has 3 arguments: α, r, a
    if major_args.len() < 3 {
        return None;
    }

    // The value being quoted
    let a = major_args[2];

    // f is args[3]
    let f = args[3];

    // Result: f a
    Some(Expr::app(f.clone(), a.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quot_names() {
        assert_eq!(names::QUOT.to_string(), "Quot");
        assert_eq!(names::QUOT_MK.to_string(), "Quot.mk");
        assert_eq!(names::QUOT_LIFT.to_string(), "Quot.lift");
        assert_eq!(names::QUOT_IND.to_string(), "Quot.ind");
    }

    #[test]
    fn test_is_quot_name() {
        assert!(is_quot_name(&Name::from_string("Quot")));
        assert!(is_quot_name(&Name::from_string("Quot.mk")));
        assert!(is_quot_name(&Name::from_string("Quot.lift")));
        assert!(is_quot_name(&Name::from_string("Quot.ind")));
        assert!(!is_quot_name(&Name::from_string("Nat")));
        assert!(!is_quot_name(&Name::from_string("List")));
    }

    #[test]
    fn test_get_quot_kind() {
        assert_eq!(
            get_quot_kind(&Name::from_string("Quot")),
            Some(QuotKind::Type)
        );
        assert_eq!(
            get_quot_kind(&Name::from_string("Quot.mk")),
            Some(QuotKind::Mk)
        );
        assert_eq!(
            get_quot_kind(&Name::from_string("Quot.lift")),
            Some(QuotKind::Lift)
        );
        assert_eq!(
            get_quot_kind(&Name::from_string("Quot.ind")),
            Some(QuotKind::Ind)
        );
        assert_eq!(get_quot_kind(&Name::from_string("Nat")), None);
    }

    #[test]
    fn test_quot_type_structure() {
        let u = Name::from_string("u");
        let typ = quot_type(&u);

        // Should be a Pi type
        match typ {
            Expr::Pi(bi, _, _) => {
                // First binder should be implicit (α)
                assert_eq!(bi, BinderInfo::Implicit);
            }
            _ => panic!("Expected Pi type"),
        }
    }

    #[test]
    fn test_quot_mk_type_structure() {
        let u = Name::from_string("u");
        let typ = quot_mk_type(&u);

        // Should be a Pi type
        match typ {
            Expr::Pi(bi, _, _) => {
                // First binder should be implicit (α)
                assert_eq!(bi, BinderInfo::Implicit);
            }
            _ => panic!("Expected Pi type"),
        }
    }

    #[test]
    fn test_init_quot_vals() {
        let vals = init_quot_vals();
        assert_eq!(vals.len(), 4);

        // Check all four primitives are present
        let names: Vec<&Name> = vals.iter().map(|v| &v.name).collect();
        assert!(names.iter().any(|n| n.to_string() == "Quot"));
        assert!(names.iter().any(|n| n.to_string() == "Quot.mk"));
        assert!(names.iter().any(|n| n.to_string() == "Quot.lift"));
        assert!(names.iter().any(|n| n.to_string() == "Quot.ind"));

        // Check kinds
        for val in &vals {
            match val.kind {
                QuotKind::Type => assert_eq!(val.name.to_string(), "Quot"),
                QuotKind::Mk => assert_eq!(val.name.to_string(), "Quot.mk"),
                QuotKind::Lift => assert_eq!(val.name.to_string(), "Quot.lift"),
                QuotKind::Ind => assert_eq!(val.name.to_string(), "Quot.ind"),
            }
        }
    }

    #[test]
    fn test_quot_lift_reduction_not_lift() {
        // Test that non-Quot.lift heads return None
        let head = Expr::const_(Name::from_string("Nat"), vec![]);
        let args: Vec<&Expr> = vec![];
        let result = try_quot_lift_reduction(&head, &args, Expr::clone);
        assert!(result.is_none());
    }

    #[test]
    fn test_quot_lift_reduction_insufficient_args() {
        // Test that Quot.lift with insufficient args returns None
        let head = Expr::const_(names::QUOT_LIFT.clone(), vec![Level::zero()]);
        let arg1 = Expr::prop();
        let args: Vec<&Expr> = vec![&arg1]; // Only 1 arg, need 6
        let result = try_quot_lift_reduction(&head, &args, Expr::clone);
        assert!(result.is_none());
    }

    #[test]
    fn test_quot_lift_reduction_success() {
        // Build a reducible expression: Quot.lift α r β f h (Quot.mk α r a)
        // This should reduce to: f a

        let u = Level::param(Name::from_string("u"));
        let v = Level::param(Name::from_string("v"));

        // Dummy values for α, r, β
        let alpha = Expr::type_();
        let r = Expr::lam(
            BinderInfo::Default,
            Expr::type_(),
            Expr::lam(BinderInfo::Default, Expr::bvar(0), Expr::prop()),
        );
        let beta = Expr::type_();

        // f : α → β (identity function for simplicity)
        let f = Expr::lam(BinderInfo::Default, Expr::type_(), Expr::bvar(0));

        // h : proof (dummy)
        let h = Expr::prop();

        // a : α (some value)
        let a = Expr::const_(Name::from_string("x"), vec![]);

        // Build Quot.mk α r a
        let mk_app = Expr::app(
            Expr::app(
                Expr::app(
                    Expr::const_(names::QUOT_MK.clone(), vec![u.clone()]),
                    alpha.clone(),
                ),
                r.clone(),
            ),
            a.clone(),
        );

        // Build the head and args for Quot.lift
        let head = Expr::const_(names::QUOT_LIFT.clone(), vec![u, v]);
        let args: Vec<&Expr> = vec![&alpha, &r, &beta, &f, &h, &mk_app];

        // The reduction should give us f a
        let result = try_quot_lift_reduction(&head, &args, Expr::clone);
        assert!(result.is_some());

        let reduced = result.unwrap();
        // Check it's f applied to a
        match reduced {
            Expr::App(func, arg) => {
                // func should be f (the lambda)
                assert!(matches!(func.as_ref(), Expr::Lam(..)));
                // arg should be a
                assert_eq!(*arg, a);
            }
            _ => panic!("Expected App, got {reduced:?}"),
        }
    }

    // =========================================================================
    // Mutation Testing Kill Tests - quot.rs survivors
    // =========================================================================

    #[test]
    fn test_quot_lift_reduction_args_boundary() {
        // Kill mutant: replace < with > in try_quot_lift_reduction (line 462)
        // The check is: if args.len() < 6 { return None; }
        // With exactly 5 args, should return None. With 6+, should potentially work.

        let u = Level::param(Name::from_string("u"));
        let v = Level::param(Name::from_string("v"));
        let head = Expr::const_(names::QUOT_LIFT.clone(), vec![u, v]);

        let alpha = Expr::type_();
        let r = Expr::prop();
        let beta = Expr::type_();
        let f = Expr::prop();
        let h = Expr::prop();

        // Exactly 5 args - should NOT reduce (less than 6)
        let args_5: Vec<&Expr> = vec![&alpha, &r, &beta, &f, &h];
        let result_5 = try_quot_lift_reduction(&head, &args_5, Expr::clone);
        assert!(result_5.is_none(), "5 args < 6, should return None");

        // Exactly 6 args but last is NOT Quot.mk - should not reduce
        let not_mk = Expr::type_();
        let args_6: Vec<&Expr> = vec![&alpha, &r, &beta, &f, &h, &not_mk];
        let result_6 = try_quot_lift_reduction(&head, &args_6, Expr::clone);
        assert!(
            result_6.is_none(),
            "6 args but no Quot.mk, should return None"
        );

        // For completeness, 4 args should also return None
        let args_4: Vec<&Expr> = vec![&alpha, &r, &beta, &f];
        let result_4 = try_quot_lift_reduction(&head, &args_4, Expr::clone);
        assert!(result_4.is_none(), "4 args < 6, should return None");
    }

    #[test]
    fn test_quot_lift_reduction_major_args_boundary() {
        // Kill mutant at line 462: replace < with > in `major_args.len() < 3`
        // This checks if Quot.mk has enough arguments (needs 3: α, r, a)

        let u = Level::param(Name::from_string("u"));
        let v = Level::param(Name::from_string("v"));
        let head = Expr::const_(names::QUOT_LIFT.clone(), vec![u.clone(), v]);

        let alpha = Expr::type_();
        let r = Expr::prop();
        let beta = Expr::type_();
        let f_func = Expr::lam(BinderInfo::Default, Expr::type_(), Expr::bvar(0));
        let h = Expr::prop();

        // Build Quot.mk with ONLY 2 arguments (not enough - needs 3)
        // Quot.mk α r (missing the value 'a')
        let mk_partial = Expr::app(
            Expr::app(
                Expr::const_(names::QUOT_MK.clone(), vec![u.clone()]),
                alpha.clone(),
            ),
            r.clone(),
        );

        // Call with 6 outer args where major (6th) is Quot.mk with only 2 inner args
        let args_with_partial_mk: Vec<&Expr> = vec![&alpha, &r, &beta, &f_func, &h, &mk_partial];
        let result = try_quot_lift_reduction(&head, &args_with_partial_mk, Expr::clone);
        assert!(
            result.is_none(),
            "Quot.mk with only 2 args (< 3), should return None"
        );

        // Now with exactly 3 args for Quot.mk (should work)
        let a_value = Expr::const_(Name::from_string("x"), vec![]);
        let mk_complete = Expr::app(mk_partial.clone(), a_value.clone());

        let args_with_complete_mk: Vec<&Expr> = vec![&alpha, &r, &beta, &f_func, &h, &mk_complete];
        let result = try_quot_lift_reduction(&head, &args_with_complete_mk, Expr::clone);
        assert!(
            result.is_some(),
            "Quot.mk with exactly 3 args (>= 3), should reduce"
        );

        // Verify the result is f a
        let reduced = result.unwrap();
        match &reduced {
            Expr::App(func, arg) => {
                assert!(
                    matches!(func.as_ref(), Expr::Lam(..)),
                    "Result function should be the lambda f"
                );
                assert_eq!(arg.as_ref(), &a_value, "Result arg should be a");
            }
            _ => panic!("Expected App"),
        }

        // Edge case: Quot.mk with exactly 1 arg
        let mk_1arg = Expr::app(
            Expr::const_(names::QUOT_MK.clone(), vec![u.clone()]),
            alpha.clone(),
        );
        let args_1arg: Vec<&Expr> = vec![&alpha, &r, &beta, &f_func, &h, &mk_1arg];
        let result = try_quot_lift_reduction(&head, &args_1arg, Expr::clone);
        assert!(
            result.is_none(),
            "Quot.mk with 1 arg (< 3), should return None"
        );

        // Edge case: Quot.mk with 0 args (just the constant)
        let mk_0arg = Expr::const_(names::QUOT_MK.clone(), vec![u]);
        let args_0arg: Vec<&Expr> = vec![&alpha, &r, &beta, &f_func, &h, &mk_0arg];
        let result = try_quot_lift_reduction(&head, &args_0arg, Expr::clone);
        assert!(
            result.is_none(),
            "Quot.mk with 0 args (< 3), should return None"
        );
    }

    // =========================================================================
    // Tests for quot_lift_type and quot_ind_type type structures
    // =========================================================================

    #[test]
    fn test_quot_lift_type_structure() {
        let u = Name::from_string("u");
        let v = Name::from_string("v");
        let typ = quot_lift_type(&u, &v);

        // quot_lift_type has 6 binders:
        // {α : Sort u} → {r : α → α → Prop} → {β : Sort v} →
        //   (f : α → β) → (h : ∀ a b, r a b → f a = f b) → (@Quot α r → β)

        // Count the nested Pi types (should be 6 total binders)
        fn count_pis(e: &Expr) -> usize {
            match e {
                Expr::Pi(_, _, body) => 1 + count_pis(body),
                _ => 0,
            }
        }

        assert_eq!(count_pis(&typ), 6, "quot_lift_type should have 6 Pi binders");

        // Verify first 3 binders are implicit
        match &typ {
            Expr::Pi(bi1, _, body1) => {
                assert_eq!(*bi1, BinderInfo::Implicit, "First binder (α) should be implicit");
                match body1.as_ref() {
                    Expr::Pi(bi2, _, body2) => {
                        assert_eq!(*bi2, BinderInfo::Implicit, "Second binder (r) should be implicit");
                        match body2.as_ref() {
                            Expr::Pi(bi3, _, body3) => {
                                assert_eq!(*bi3, BinderInfo::Implicit, "Third binder (β) should be implicit");
                                // Next 3 should be explicit
                                match body3.as_ref() {
                                    Expr::Pi(bi4, _, body4) => {
                                        assert_eq!(*bi4, BinderInfo::Default, "Fourth binder (f) should be explicit");
                                        match body4.as_ref() {
                                            Expr::Pi(bi5, _, body5) => {
                                                assert_eq!(*bi5, BinderInfo::Default, "Fifth binder (h) should be explicit");
                                                match body5.as_ref() {
                                                    Expr::Pi(bi6, _, _) => {
                                                        assert_eq!(*bi6, BinderInfo::Default, "Sixth binder (q) should be explicit");
                                                    }
                                                    _ => panic!("Expected 6th Pi"),
                                                }
                                            }
                                            _ => panic!("Expected 5th Pi"),
                                        }
                                    }
                                    _ => panic!("Expected 4th Pi"),
                                }
                            }
                            _ => panic!("Expected 3rd Pi"),
                        }
                    }
                    _ => panic!("Expected 2nd Pi"),
                }
            }
            _ => panic!("Expected Pi type"),
        }
    }

    #[test]
    fn test_quot_ind_type_structure() {
        let u = Name::from_string("u");
        let typ = quot_ind_type(&u);

        // quot_ind_type has 4 binders:
        // {α : Sort u} → {r : α → α → Prop} →
        //   {β : @Quot α r → Prop} →
        //   (ih : ∀ a : α, β (Quot.mk α r a)) → ∀ q : @Quot α r, β q

        fn count_pis(e: &Expr) -> usize {
            match e {
                Expr::Pi(_, _, body) => 1 + count_pis(body),
                _ => 0,
            }
        }

        assert_eq!(count_pis(&typ), 5, "quot_ind_type should have 5 Pi binders");

        // Verify first 3 binders are implicit
        match &typ {
            Expr::Pi(bi1, _, body1) => {
                assert_eq!(*bi1, BinderInfo::Implicit, "First binder (α) should be implicit");
                match body1.as_ref() {
                    Expr::Pi(bi2, _, body2) => {
                        assert_eq!(*bi2, BinderInfo::Implicit, "Second binder (r) should be implicit");
                        match body2.as_ref() {
                            Expr::Pi(bi3, _, body3) => {
                                assert_eq!(*bi3, BinderInfo::Implicit, "Third binder (β) should be implicit");
                                // Next 2 should be explicit
                                match body3.as_ref() {
                                    Expr::Pi(bi4, _, body4) => {
                                        assert_eq!(*bi4, BinderInfo::Default, "Fourth binder (ih) should be explicit");
                                        match body4.as_ref() {
                                            Expr::Pi(bi5, _, _) => {
                                                assert_eq!(*bi5, BinderInfo::Default, "Fifth binder (q) should be explicit");
                                            }
                                            _ => panic!("Expected 5th Pi"),
                                        }
                                    }
                                    _ => panic!("Expected 4th Pi"),
                                }
                            }
                            _ => panic!("Expected 3rd Pi"),
                        }
                    }
                    _ => panic!("Expected 2nd Pi"),
                }
            }
            _ => panic!("Expected Pi type"),
        }
    }

    #[test]
    fn test_quot_type_has_correct_universe_params() {
        let u = Name::from_string("u");
        let typ = quot_type(&u);

        // The first binder's type should be Sort(u)
        match &typ {
            Expr::Pi(_, domain, _) => {
                assert!(
                    matches!(domain.as_ref(), Expr::Sort(l) if matches!(l, Level::Param(_))),
                    "First binder domain should be Sort(u)"
                );
            }
            _ => panic!("Expected Pi type"),
        }
    }

    #[test]
    fn test_quot_mk_type_returns_quot_type() {
        let u = Name::from_string("u");
        let typ = quot_mk_type(&u);

        // Navigate to the innermost return type
        fn get_return_type(e: &Expr) -> &Expr {
            match e {
                Expr::Pi(_, _, body) => get_return_type(body),
                _ => e,
            }
        }

        let ret = get_return_type(&typ);

        // Return type should be @Quot α r which is App(App(Const(...), ...), ...)
        assert!(
            matches!(ret, Expr::App(_, _)),
            "Return type of Quot.mk should be an application (Quot α r)"
        );
    }

    #[test]
    fn test_quot_lift_returns_correct_type() {
        let u = Name::from_string("u");
        let v = Name::from_string("v");
        let typ = quot_lift_type(&u, &v);

        fn get_return_type(e: &Expr) -> &Expr {
            match e {
                Expr::Pi(_, _, body) => get_return_type(body),
                _ => e,
            }
        }

        let ret = get_return_type(&typ);

        // Return type should be β (a BVar referring to the third binder)
        assert!(
            matches!(ret, Expr::BVar(_)),
            "Return type of Quot.lift should be a bound variable (β)"
        );
    }

    #[test]
    fn test_quot_ind_returns_correct_type() {
        let u = Name::from_string("u");
        let typ = quot_ind_type(&u);

        fn get_return_type(e: &Expr) -> &Expr {
            match e {
                Expr::Pi(_, _, body) => get_return_type(body),
                _ => e,
            }
        }

        let ret = get_return_type(&typ);

        // Return type should be (β q) which is App(β, q)
        assert!(
            matches!(ret, Expr::App(_, _)),
            "Return type of Quot.ind should be an application (β q)"
        );
    }

    #[test]
    fn test_quot_kind_name_mapping() {
        // Verify that QuotKind::name() returns the correct name for each variant
        assert_eq!(QuotKind::Type.name().to_string(), "Quot");
        assert_eq!(QuotKind::Mk.name().to_string(), "Quot.mk");
        assert_eq!(QuotKind::Lift.name().to_string(), "Quot.lift");
        assert_eq!(QuotKind::Ind.name().to_string(), "Quot.ind");
    }

    #[test]
    fn test_init_quot_vals_level_params() {
        let vals = init_quot_vals();

        for val in &vals {
            match val.kind {
                QuotKind::Type | QuotKind::Mk | QuotKind::Ind => {
                    // These have 1 universe parameter (u)
                    assert_eq!(
                        val.level_params.len(),
                        1,
                        "{:?} should have 1 level param",
                        val.kind
                    );
                }
                QuotKind::Lift => {
                    // Lift has 2 universe parameters (u, v)
                    assert_eq!(val.level_params.len(), 2, "Lift should have 2 level params");
                }
            }
        }
    }
}
