//! Inductive Types
//!
//! Validation and compilation of inductive type definitions.
//!
//! In Lean/CIC, inductive types are introduced with:
//! - A type former (the inductive type itself)
//! - Constructors that build inhabitants
//! - A recursor (eliminator) for case analysis and recursion
//!
//! # Example: Natural Numbers
//! ```text
//! inductive Nat : Type
//! | zero : Nat
//! | succ : Nat → Nat
//! ```
//!
//! Generates:
//! - `Nat : Type`
//! - `Nat.zero : Nat`
//! - `Nat.succ : Nat → Nat`
//! - `Nat.rec : {C : Nat → Sort u} → C Nat.zero → ((n : Nat) → C n → C (Nat.succ n)) → (n : Nat) → C n`

use crate::expr::Expr;
use crate::name::Name;
use serde::{Deserialize, Serialize};

/// Minimum stack space to reserve before recursive calls (32 KB).
const MIN_STACK_RED_ZONE: usize = 32 * 1024;

/// Stack size to grow to when running low (1 MB).
const STACK_GROWTH_SIZE: usize = 1024 * 1024;

/// A constructor declaration for an inductive type
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Constructor {
    /// Name of the constructor (e.g., "Nat.zero")
    pub name: Name,
    /// Type of the constructor (must return the inductive type)
    pub type_: Expr,
}

/// A single inductive type declaration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InductiveType {
    /// Name of the inductive type
    pub name: Name,
    /// Type of the inductive (e.g., Type, Type → Type, etc.)
    pub type_: Expr,
    /// Constructors
    pub constructors: Vec<Constructor>,
}

/// Declaration of one or more mutually inductive types
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InductiveDecl {
    /// Universe parameters
    pub level_params: Vec<Name>,
    /// Number of parameters (shared prefix in all types and constructors)
    pub num_params: u32,
    /// The inductive types (length > 1 for mutual inductives)
    pub types: Vec<InductiveType>,
}

/// Information stored in the environment about an inductive type
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InductiveVal {
    /// Name of the inductive type
    pub name: Name,
    /// Universe parameters
    pub level_params: Vec<Name>,
    /// Type of the inductive
    pub type_: Expr,
    /// Number of parameters
    pub num_params: u32,
    /// Number of indices (arguments after parameters)
    pub num_indices: u32,
    /// Names of all inductive types in mutual block
    pub all_names: Vec<Name>,
    /// Names of constructors
    pub constructor_names: Vec<Name>,
    /// Whether the type is recursive
    pub is_recursive: bool,
    /// Whether this is a reflexive inductive (contains inductive → inductive)
    pub is_reflexive: bool,
    /// Whether large elimination is allowed (eliminating into Type u for u > 0)
    pub is_large_elim: bool,
}

/// Information stored in the environment about a constructor
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConstructorVal {
    /// Name of the constructor
    pub name: Name,
    /// Name of the inductive type this constructs
    pub inductive_name: Name,
    /// Universe parameters (same as inductive)
    pub level_params: Vec<Name>,
    /// Type of the constructor
    pub type_: Expr,
    /// Number of parameters
    pub num_params: u32,
    /// Number of fields (arguments after parameters)
    pub num_fields: u32,
    /// Index of this constructor in the inductive's constructor list
    pub constructor_idx: u32,
}

/// Information stored in the environment about a recursor
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecursorVal {
    /// Name of the recursor (e.g., "Nat.rec")
    pub name: Name,
    /// Ordering of arguments relative to the major premise
    pub arg_order: RecursorArgOrder,
    /// Universe parameters (includes motive universe)
    pub level_params: Vec<Name>,
    /// Type of the recursor
    pub type_: Expr,
    /// Name of the inductive type
    pub inductive_name: Name,
    /// Number of parameters
    pub num_params: u32,
    /// Number of indices
    pub num_indices: u32,
    /// Number of motives (1 for simple inductives, n for mutual)
    pub num_motives: u32,
    /// Number of minor premises (one per constructor)
    pub num_minors: u32,
    /// Recursor rules (one per constructor)
    pub rules: Vec<RecursorRule>,
    /// Whether K-like reduction is used
    pub is_k: bool,
}

/// A recursor rule: how the recursor computes on a constructor
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecursorRule {
    /// Constructor this rule applies to
    pub constructor_name: Name,
    /// Number of fields in the constructor
    pub num_fields: u32,
    /// Which fields are recursive (require induction hypothesis)
    /// Length equals num_fields; true means the field has type involving the inductive
    pub recursive_fields: Vec<bool>,
    /// The right-hand side of the rule
    /// `rec ... (ctor fields) = rhs[fields, recursive_results]`
    pub rhs: Expr,
}

/// Where the major premise appears in the recursor argument list.
///
/// Lean-style recursors (rec/casesOn) put the major after minors and indices.
/// recOn variants move the major immediately after motives/indices so users can
/// supply the major premise earlier.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum RecursorArgOrder {
    /// Standard layout: params → motives → minors → indices → major
    MajorAfterMinors,
    /// recOn layout: params → motives → indices → major → minors
    MajorAfterMotive,
}

/// Errors during inductive type checking
#[derive(Debug, thiserror::Error)]
pub enum InductiveError {
    #[error("Empty inductive declaration")]
    EmptyDecl,
    #[error("No constructors for inductive type {0}")]
    NoConstructors(Name),
    #[error("Constructor {0} has invalid type")]
    InvalidConstructorType(Name),
    #[error("Non-positive occurrence of {0} in {1}")]
    NonPositive(Name, Name),
    #[error("Invalid inductive type: {0}")]
    InvalidType(String),
    #[error("Universe level mismatch in constructor {0}")]
    UniverseMismatch(Name),
    #[error("Constructor {0} does not return the inductive type {1}")]
    ConstructorReturnType(Name, Name),
    #[error("Duplicate constructor name: {0}")]
    DuplicateConstructor(Name),
    #[error("Invalid number of parameters")]
    InvalidParams,
}

/// Check if an inductive type occurs strictly positively in an expression.
///
/// Positivity is required for logical consistency. The rule is:
/// - An occurrence in a constructor's return type is positive
/// - An occurrence in the domain of a Pi is positive IFF the inductive doesn't
///   appear in a nested negative position within that domain
/// - Specifically: `I → R` is fine (I just appears directly)
/// - But `(I → X) → R` is NOT fine (I appears left of an arrow within the domain)
///
/// In practice, for a constructor like `succ : Nat → Nat`:
/// - The first Nat (domain) is checked with "strictly positive" rules
/// - The second Nat (codomain) is fine
///
/// The strictly positive check means: the inductive can appear, but not to the
/// left of any arrows within that subexpression.
pub fn check_positivity(
    inductive_name: &Name,
    expr: &Expr,
    param_count: u32,
) -> Result<(), InductiveError> {
    // For constructor types, we check the whole type with standard rules
    check_positivity_in_ctor_type(inductive_name, expr, param_count)
}

/// Check positivity in a constructor type: (args) → I params indices
fn check_positivity_in_ctor_type(
    inductive_name: &Name,
    expr: &Expr,
    param_count: u32,
) -> Result<(), InductiveError> {
    stacker::maybe_grow(MIN_STACK_RED_ZONE, STACK_GROWTH_SIZE, || {
        check_positivity_in_ctor_type_impl(inductive_name, expr, param_count)
    })
}

/// Implementation (called via stacker::maybe_grow)
fn check_positivity_in_ctor_type_impl(
    inductive_name: &Name,
    expr: &Expr,
    param_count: u32,
) -> Result<(), InductiveError> {
    match expr {
        Expr::Pi(_, domain, codomain) => {
            // In a constructor argument type, the inductive can appear
            // but must be "strictly positive" (not left of any arrow)
            check_strictly_positive(inductive_name, domain, param_count)?;
            check_positivity_in_ctor_type(inductive_name, codomain, param_count)?;
            Ok(())
        }
        _ => {
            // Return type - any occurrence is fine (it should be the inductive itself)
            Ok(())
        }
    }
}

/// Check strict positivity: the inductive may appear, but not to the left of any arrow
fn check_strictly_positive(
    inductive_name: &Name,
    expr: &Expr,
    param_count: u32,
) -> Result<(), InductiveError> {
    stacker::maybe_grow(MIN_STACK_RED_ZONE, STACK_GROWTH_SIZE, || {
        check_strictly_positive_impl(inductive_name, expr, param_count)
    })
}

/// Implementation (called via stacker::maybe_grow)
fn check_strictly_positive_impl(
    inductive_name: &Name,
    expr: &Expr,
    _param_count: u32,
) -> Result<(), InductiveError> {
    match expr {
        Expr::BVar(_) | Expr::FVar(_) | Expr::Sort(_) | Expr::Lit(_) => Ok(()),

        Expr::Const(_name, _) => {
            // Direct occurrence of the inductive is fine
            Ok(())
        }

        Expr::App(f, a) => {
            // Check if head is the inductive type
            let head = expr.get_app_fn();
            if let Expr::Const(name, _) = head {
                if name == inductive_name {
                    // I applied to args - args must not mention I negatively
                    let args = expr.get_app_args();
                    for arg in args {
                        check_no_negative_occurrence(inductive_name, arg)?;
                    }
                    return Ok(());
                }
            }
            // General application: check both parts for strict positivity
            check_strictly_positive(inductive_name, f, _param_count)?;
            check_strictly_positive(inductive_name, a, _param_count)?;
            Ok(())
        }

        Expr::Pi(_, domain, codomain) => {
            // This is the critical case: (A → B) appears in a constructor argument
            // The inductive CANNOT appear in A (that would be negative)
            check_no_negative_occurrence(inductive_name, domain)?;
            // But it CAN appear in B (still positive, just nested)
            check_strictly_positive(inductive_name, codomain, _param_count)?;
            Ok(())
        }

        Expr::Lam(_, ty, body) => {
            check_strictly_positive(inductive_name, ty, _param_count)?;
            check_strictly_positive(inductive_name, body, _param_count)?;
            Ok(())
        }

        Expr::Let(ty, val, body) => {
            check_strictly_positive(inductive_name, ty, _param_count)?;
            check_strictly_positive(inductive_name, val, _param_count)?;
            check_strictly_positive(inductive_name, body, _param_count)?;
            Ok(())
        }

        Expr::Proj(_, _, e) => {
            check_strictly_positive(inductive_name, e, _param_count)?;
            Ok(())
        }

        // MData is transparent - check the inner expression
        Expr::MData(_, inner) => check_strictly_positive(inductive_name, inner, _param_count),

        // Mode-specific extensions - conservative: check all subexpressions
        Expr::CubicalInterval | Expr::CubicalI0 | Expr::CubicalI1 => Ok(()),
        Expr::CubicalPath { ty, left, right } => {
            check_strictly_positive(inductive_name, ty, _param_count)?;
            check_strictly_positive(inductive_name, left, _param_count)?;
            check_strictly_positive(inductive_name, right, _param_count)
        }
        Expr::CubicalPathLam { body } => check_strictly_positive(inductive_name, body, _param_count),
        Expr::CubicalPathApp { path, arg } => {
            check_strictly_positive(inductive_name, path, _param_count)?;
            check_strictly_positive(inductive_name, arg, _param_count)
        }
        Expr::CubicalHComp { ty, phi, u, base } => {
            check_strictly_positive(inductive_name, ty, _param_count)?;
            check_strictly_positive(inductive_name, phi, _param_count)?;
            check_strictly_positive(inductive_name, u, _param_count)?;
            check_strictly_positive(inductive_name, base, _param_count)
        }
        Expr::CubicalTransp { ty, phi, base } => {
            check_strictly_positive(inductive_name, ty, _param_count)?;
            check_strictly_positive(inductive_name, phi, _param_count)?;
            check_strictly_positive(inductive_name, base, _param_count)
        }
        Expr::ClassicalChoice {
            ty,
            pred,
            exists_proof,
        } => {
            check_strictly_positive(inductive_name, ty, _param_count)?;
            check_strictly_positive(inductive_name, pred, _param_count)?;
            check_strictly_positive(inductive_name, exists_proof, _param_count)
        }
        Expr::ClassicalEpsilon { ty, pred } => {
            check_strictly_positive(inductive_name, ty, _param_count)?;
            check_strictly_positive(inductive_name, pred, _param_count)
        }
        Expr::ZFCSet(_) | Expr::ZFCMem { .. } | Expr::ZFCComprehension { .. } => {
            // ZFC set expressions don't typically interact with inductive types
            Ok(())
        }
        // Impredicative mode extensions
        Expr::SProp => Ok(()),
        Expr::Squash(inner) => check_strictly_positive(inductive_name, inner, _param_count),
    }
}

/// Check that the inductive does not appear in a negative position
/// (i.e., the inductive should not appear in this expression at all in the domain of an arrow)
fn check_no_negative_occurrence(inductive_name: &Name, expr: &Expr) -> Result<(), InductiveError> {
    if mentions_name(expr, inductive_name) {
        Err(InductiveError::NonPositive(
            inductive_name.clone(),
            inductive_name.clone(),
        ))
    } else {
        Ok(())
    }
}

/// Check if an expression mentions a name
pub fn mentions_name(expr: &Expr, name: &Name) -> bool {
    stacker::maybe_grow(MIN_STACK_RED_ZONE, STACK_GROWTH_SIZE, || {
        mentions_name_impl(expr, name)
    })
}

/// Implementation (called via stacker::maybe_grow)
fn mentions_name_impl(expr: &Expr, name: &Name) -> bool {
    match expr {
        Expr::BVar(_) | Expr::FVar(_) | Expr::Sort(_) | Expr::Lit(_) => false,
        Expr::Const(n, _) => n == name,
        Expr::App(f, a) => mentions_name(f, name) || mentions_name(a, name),
        Expr::Lam(_, ty, body) | Expr::Pi(_, ty, body) => {
            mentions_name(ty, name) || mentions_name(body, name)
        }
        Expr::Let(ty, val, body) => {
            mentions_name(ty, name) || mentions_name(val, name) || mentions_name(body, name)
        }
        Expr::Proj(_, _, e) => mentions_name(e, name),
        // MData is transparent - check the inner expression
        Expr::MData(_, inner) => mentions_name(inner, name),

        // Mode-specific extensions
        Expr::CubicalInterval | Expr::CubicalI0 | Expr::CubicalI1 => false,
        Expr::CubicalPath { ty, left, right } => {
            mentions_name(ty, name) || mentions_name(left, name) || mentions_name(right, name)
        }
        Expr::CubicalPathLam { body } => mentions_name(body, name),
        Expr::CubicalPathApp { path, arg } => {
            mentions_name(path, name) || mentions_name(arg, name)
        }
        Expr::CubicalHComp { ty, phi, u, base } => {
            mentions_name(ty, name)
                || mentions_name(phi, name)
                || mentions_name(u, name)
                || mentions_name(base, name)
        }
        Expr::CubicalTransp { ty, phi, base } => {
            mentions_name(ty, name) || mentions_name(phi, name) || mentions_name(base, name)
        }
        Expr::ClassicalChoice {
            ty,
            pred,
            exists_proof,
        } => {
            mentions_name(ty, name)
                || mentions_name(pred, name)
                || mentions_name(exists_proof, name)
        }
        Expr::ClassicalEpsilon { ty, pred } => {
            mentions_name(ty, name) || mentions_name(pred, name)
        }
        // ZFC set expressions - conservative check
        Expr::ZFCSet(_) | Expr::ZFCMem { .. } | Expr::ZFCComprehension { .. } => {
            // These are ZFC primitives, typically don't contain user-defined names
            false
        }
        // Impredicative mode extensions
        Expr::SProp => false,
        Expr::Squash(inner) => mentions_name(inner, name),
    }
}

/// Count the number of Pi types at the head of an expression
pub fn count_pi_args(expr: &Expr) -> u32 {
    match expr {
        Expr::Pi(_, _, body) => 1 + count_pi_args(body),
        _ => 0,
    }
}

/// Strip `n` Pi types from the front of an expression, returning the body
pub fn strip_pi(expr: &Expr, n: u32) -> &Expr {
    if n == 0 {
        return expr;
    }
    match expr {
        Expr::Pi(_, _, body) => strip_pi(body, n - 1),
        _ => expr,
    }
}

/// Get the return type of a Pi-telescope (strip all Pi's)
pub fn get_return_type(expr: &Expr) -> &Expr {
    match expr {
        Expr::Pi(_, _, body) => get_return_type(body),
        _ => expr,
    }
}

/// Validate an inductive declaration
pub fn validate_inductive(decl: &InductiveDecl) -> Result<(), InductiveError> {
    if decl.types.is_empty() {
        return Err(InductiveError::EmptyDecl);
    }

    // Check each inductive type
    for ind_type in &decl.types {
        // Validate constructors
        for ctor in &ind_type.constructors {
            // Check positivity
            check_positivity(&ind_type.name, &ctor.type_, decl.num_params)?;

            // Check that constructor returns the inductive type
            let return_type = get_return_type(&ctor.type_);
            let head = return_type.get_app_fn();
            match head {
                Expr::Const(name, _) if name == &ind_type.name => {}
                _ => {
                    return Err(InductiveError::ConstructorReturnType(
                        ctor.name.clone(),
                        ind_type.name.clone(),
                    ));
                }
            }
        }
    }

    Ok(())
}

/// Check if an inductive type is recursive
///
/// An inductive is recursive if any constructor has an argument of the inductive type.
/// (The return type doesn't count - all constructors return the inductive type.)
pub fn is_recursive(inductive_name: &Name, constructors: &[Constructor]) -> bool {
    constructors
        .iter()
        .any(|ctor| mentions_name_in_args(&ctor.type_, inductive_name))
}

/// Check if an inductive type is reflexive
///
/// A reflexive inductive is one where the inductive type appears in the domain
/// of a function type in a constructor argument. This means the inductive type
/// is used both to construct elements AND to index into data.
///
/// Example: W-types (well-founded trees)
/// ```text
/// inductive W (A : Type) (B : A → Type) : Type
/// | sup : (a : A) → (B a → W A B) → W A B
/// ```
/// Here `W A B` appears in the domain of `B a → W A B`, making it reflexive.
///
/// Contrast with Nat which is recursive but NOT reflexive:
/// ```text
/// inductive Nat : Type
/// | succ : Nat → Nat
/// ```
/// Nat appears directly as an argument, not in a function domain.
pub fn is_reflexive(inductive_name: &Name, constructors: &[Constructor]) -> bool {
    constructors
        .iter()
        .any(|ctor| has_reflexive_occurrence(&ctor.type_, inductive_name))
}

/// Check if an expression has a reflexive occurrence of the inductive
///
/// A reflexive occurrence is when the inductive appears in the domain of a
/// function type that is itself an argument to a constructor.
fn has_reflexive_occurrence(expr: &Expr, name: &Name) -> bool {
    match expr {
        Expr::Pi(_, domain, codomain) => {
            // Check if this domain is a function type where the inductive appears
            // somewhere in that function type (making this a reflexive argument)
            if is_function_mentioning_name(domain, name) {
                return true;
            }
            // Continue checking in the codomain
            has_reflexive_occurrence(codomain, name)
        }
        _ => false, // Return type doesn't matter
    }
}

/// Check if expr is a function type (Pi) that mentions `name` anywhere
/// This detects reflexive arguments like `(B a → W A B)` where W appears in the codomain
fn is_function_mentioning_name(expr: &Expr, name: &Name) -> bool {
    match expr {
        Expr::Pi(_, domain, codomain) => {
            // This is a function type - check if name appears anywhere in it
            mentions_name(domain, name) || mentions_name(codomain, name)
        }
        _ => false, // Not a function type
    }
}

/// Check if an expression mentions a name in its argument types (Pi domains), not the return type
fn mentions_name_in_args(expr: &Expr, name: &Name) -> bool {
    match expr {
        Expr::Pi(_, domain, codomain) => {
            mentions_name(domain, name) || mentions_name_in_args(codomain, name)
        }
        _ => false, // Return type doesn't count
    }
}

/// Check if large elimination is allowed
///
/// Large elimination (eliminating into Type u for u > 0) is allowed when:
/// - The inductive is in Prop and has at most one constructor with no arguments
///   (singleton elimination for proof-irrelevant types)
/// - The inductive is not in Prop
pub fn allows_large_elim(inductive_type: &Expr, constructors: &[Constructor]) -> bool {
    // Check if inductive is in Prop
    let result_sort = get_return_type(inductive_type);
    if let Expr::Sort(level) = result_sort {
        if level.is_zero() {
            // In Prop - check singleton condition
            if constructors.len() <= 1 {
                // Check if constructor has no non-param arguments
                // For now, allow large elim conservatively
                return true;
            }
            return false;
        }
    }
    // Not in Prop, large elimination is allowed
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::BinderInfo;

    #[test]
    fn test_positivity_simple() {
        // Nat : Type
        // zero : Nat (positive - Nat only in return type)
        let nat = Name::from_string("Nat");
        let zero_type = Expr::const_(nat.clone(), vec![]);
        assert!(check_positivity(&nat, &zero_type, 0).is_ok());
    }

    #[test]
    fn test_positivity_arrow() {
        // succ : Nat → Nat (positive - Nat in domain is OK for non-dependent arrow)
        let nat = Name::from_string("Nat");
        let nat_ref = Expr::const_(nat.clone(), vec![]);
        let succ_type = Expr::arrow(nat_ref.clone(), nat_ref.clone());
        assert!(check_positivity(&nat, &succ_type, 0).is_ok());
    }

    #[test]
    fn test_positivity_negative() {
        // Bad : (Bad → Nat) → Bad (negative - Bad appears left of arrow)
        let bad = Name::from_string("Bad");
        let nat = Name::from_string("Nat");
        let bad_ref = Expr::const_(bad.clone(), vec![]);
        let nat_ref = Expr::const_(nat.clone(), vec![]);

        // (Bad → Nat) → Bad
        let inner_arrow = Expr::arrow(bad_ref.clone(), nat_ref);
        let bad_type = Expr::arrow(inner_arrow, bad_ref);

        assert!(check_positivity(&bad, &bad_type, 0).is_err());
    }

    #[test]
    fn test_positivity_nested_positive() {
        // Tree : Type
        // node : List Tree → Tree
        // This is positive because Tree appears as argument to List, not directly in arrow domain

        let tree = Name::from_string("Tree");
        let list = Name::from_string("List");
        let tree_ref = Expr::const_(tree.clone(), vec![]);
        let list_tree = Expr::app(Expr::const_(list, vec![]), tree_ref.clone());
        let node_type = Expr::arrow(list_tree, tree_ref);

        // This should be positive (Tree is applied to List, then List Tree → Tree)
        assert!(check_positivity(&tree, &node_type, 0).is_ok());
    }

    #[test]
    fn test_mentions_name() {
        let nat = Name::from_string("Nat");
        let nat_ref = Expr::const_(nat.clone(), vec![]);

        assert!(mentions_name(&nat_ref, &nat));
        assert!(!mentions_name(&Expr::prop(), &nat));

        let arrow = Expr::arrow(nat_ref.clone(), Expr::prop());
        assert!(mentions_name(&arrow, &nat));
    }

    #[test]
    fn test_count_pi_args() {
        // Nat → Nat → Nat has 2 Pi's
        let nat_ref = Expr::const_(Name::from_string("Nat"), vec![]);
        let ty = Expr::arrow(nat_ref.clone(), Expr::arrow(nat_ref.clone(), nat_ref));
        assert_eq!(count_pi_args(&ty), 2);

        // Nat has 0 Pi's
        assert_eq!(
            count_pi_args(&Expr::const_(Name::from_string("Nat"), vec![])),
            0
        );
    }

    #[test]
    fn test_get_return_type() {
        let nat_ref = Expr::const_(Name::from_string("Nat"), vec![]);
        let ty = Expr::arrow(Expr::prop(), Expr::arrow(Expr::prop(), nat_ref.clone()));

        let ret = get_return_type(&ty);
        assert!(matches!(ret, Expr::Const(n, _) if n == &Name::from_string("Nat")));
    }

    #[test]
    fn test_is_recursive() {
        let nat = Name::from_string("Nat");
        let nat_ref = Expr::const_(nat.clone(), vec![]);

        let zero = Constructor {
            name: Name::from_string("Nat.zero"),
            type_: nat_ref.clone(),
        };
        let succ = Constructor {
            name: Name::from_string("Nat.succ"),
            type_: Expr::arrow(nat_ref.clone(), nat_ref),
        };

        // Just zero is not recursive
        assert!(!is_recursive(&nat, std::slice::from_ref(&zero)));

        // With succ it is recursive
        assert!(is_recursive(&nat, &[zero, succ]));
    }

    #[test]
    fn test_is_reflexive_nat() {
        // Nat is recursive but NOT reflexive
        // succ : Nat → Nat - Nat appears directly as an argument, not in a function domain
        let nat = Name::from_string("Nat");
        let nat_ref = Expr::const_(nat.clone(), vec![]);

        let zero = Constructor {
            name: Name::from_string("Nat.zero"),
            type_: nat_ref.clone(),
        };
        let succ = Constructor {
            name: Name::from_string("Nat.succ"),
            type_: Expr::arrow(nat_ref.clone(), nat_ref),
        };

        assert!(!is_reflexive(&nat, std::slice::from_ref(&zero)));
        assert!(!is_reflexive(&nat, &[zero, succ]));
    }

    #[test]
    fn test_is_reflexive_w_type() {
        // W-type (well-founded trees) IS reflexive
        // sup : (a : A) → (B a → W A B) → W A B
        // W appears in the domain of (B a → W A B)
        let w = Name::from_string("W");
        let a = Name::from_string("A");
        let b = Name::from_string("B");

        let w_ref = Expr::const_(w.clone(), vec![]);
        let a_ref = Expr::const_(a.clone(), vec![]);
        let b_ref = Expr::const_(b.clone(), vec![]);

        // B a → W A B (the function type with W in domain position is the argument)
        let b_a = Expr::app(b_ref.clone(), a_ref.clone());
        let inner_arrow = Expr::arrow(b_a, w_ref.clone());

        // (a : A) → (B a → W A B) → W A B
        let sup_type = Expr::pi(
            BinderInfo::Default,
            a_ref.clone(),
            Expr::arrow(inner_arrow, w_ref),
        );

        let sup = Constructor {
            name: Name::from_string("W.sup"),
            type_: sup_type,
        };

        assert!(is_reflexive(&w, &[sup]));
    }

    #[test]
    fn test_is_reflexive_list() {
        // List is recursive but NOT reflexive
        // cons : A → List A → List A
        // List appears directly as an argument, not in a function domain
        let list = Name::from_string("List");
        let a = Name::from_string("A");

        let list_a = Expr::app(
            Expr::const_(list.clone(), vec![]),
            Expr::const_(a.clone(), vec![]),
        );

        let nil = Constructor {
            name: Name::from_string("List.nil"),
            type_: list_a.clone(),
        };
        let cons = Constructor {
            name: Name::from_string("List.cons"),
            // A → List A → List A
            type_: Expr::arrow(Expr::const_(a, vec![]), Expr::arrow(list_a.clone(), list_a)),
        };

        assert!(!is_reflexive(&list, &[nil, cons]));
    }

    #[test]
    fn test_is_reflexive_nested_function() {
        // Test with nested function types
        // T : Type
        // mk : ((T → T) → T) → T
        // T appears in domain of (T → T), which is in domain of outer arrow
        // This IS reflexive
        let t = Name::from_string("T");
        let t_ref = Expr::const_(t.clone(), vec![]);

        // T → T
        let t_to_t = Expr::arrow(t_ref.clone(), t_ref.clone());
        // (T → T) → T
        let inner = Expr::arrow(t_to_t, t_ref.clone());
        // ((T → T) → T) → T
        let mk_type = Expr::arrow(inner, t_ref);

        let mk = Constructor {
            name: Name::from_string("T.mk"),
            type_: mk_type,
        };

        assert!(is_reflexive(&t, &[mk]));
    }

    #[test]
    fn test_validate_inductive_nat() {
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
                        type_: Expr::arrow(nat_ref.clone(), nat_ref),
                    },
                ],
            }],
        };

        assert!(validate_inductive(&decl).is_ok());
    }

    #[test]
    fn test_validate_inductive_negative() {
        // Try to define a type that violates positivity
        let bad = Name::from_string("Bad");
        let bad_ref = Expr::const_(bad.clone(), vec![]);

        let decl = InductiveDecl {
            level_params: vec![],
            num_params: 0,
            types: vec![InductiveType {
                name: bad.clone(),
                type_: Expr::type_(),
                constructors: vec![Constructor {
                    name: Name::from_string("Bad.mk"),
                    // (Bad → Bad) → Bad is negative
                    type_: Expr::arrow(Expr::arrow(bad_ref.clone(), bad_ref.clone()), bad_ref),
                }],
            }],
        };

        assert!(validate_inductive(&decl).is_err());
    }

    #[test]
    fn test_allows_large_elim() {
        // Type in Type allows large elim
        let nat_type = Expr::type_();
        assert!(allows_large_elim(&nat_type, &[]));

        // Type in Prop with no constructors allows large elim (Empty/False)
        let empty_type = Expr::prop();
        assert!(allows_large_elim(&empty_type, &[]));

        // Type in Prop with one constructor allows large elim
        let unit_ctor = Constructor {
            name: Name::from_string("Unit.unit"),
            type_: Expr::const_(Name::from_string("Unit"), vec![]),
        };
        assert!(allows_large_elim(&empty_type, &[unit_ctor]));
    }

    // =========================================================================
    // Mutation Testing Kill Tests - inductive.rs survivors
    // =========================================================================

    #[test]
    fn test_mentions_name_logic_operators() {
        // Kill mutants: replace || with && in mentions_name (lines 293)
        // mentions_name returns true if ANY subexpression contains the name

        let nat = Name::from_string("Nat");
        let nat_ref = Expr::const_(nat.clone(), vec![]);
        let other = Expr::const_(Name::from_string("Other"), vec![]);

        // App: should find name in either f OR a
        let app_in_f = Expr::app(nat_ref.clone(), other.clone());
        let app_in_a = Expr::app(other.clone(), nat_ref.clone());
        let app_neither = Expr::app(other.clone(), other.clone());

        assert!(
            mentions_name(&app_in_f, &nat),
            "Should find Nat in function position"
        );
        assert!(
            mentions_name(&app_in_a, &nat),
            "Should find Nat in argument position"
        );
        assert!(
            !mentions_name(&app_neither, &nat),
            "Should not find Nat when absent"
        );

        // Let: should find name in ANY of ty, val, or body
        let let_in_ty = Expr::let_(nat_ref.clone(), other.clone(), other.clone());
        let let_in_val = Expr::let_(other.clone(), nat_ref.clone(), other.clone());
        let let_in_body = Expr::let_(other.clone(), other.clone(), nat_ref.clone());
        let let_none = Expr::let_(other.clone(), other.clone(), other.clone());

        assert!(
            mentions_name(&let_in_ty, &nat),
            "Should find Nat in let type"
        );
        assert!(
            mentions_name(&let_in_val, &nat),
            "Should find Nat in let value"
        );
        assert!(
            mentions_name(&let_in_body, &nat),
            "Should find Nat in let body"
        );
        assert!(
            !mentions_name(&let_none, &nat),
            "Should not find Nat when absent from let"
        );
    }

    #[test]
    fn test_strip_pi_comparison_and_arithmetic() {
        // Kill mutants:
        // - replace == with != in strip_pi (line 309)
        // - delete match arm Expr::Pi in strip_pi (line 313)
        // - replace - with / or + in strip_pi (line 313)

        let nat_ref = Expr::const_(Name::from_string("Nat"), vec![]);

        // n=0 should return the expression unchanged
        let simple = nat_ref.clone();
        assert!(
            std::ptr::eq(strip_pi(&simple, 0), &simple),
            "strip_pi(e, 0) should return e"
        );

        // Single Pi - strip 1 should return body
        let single_pi = Expr::pi(BinderInfo::Default, Expr::prop(), nat_ref.clone());
        let stripped1 = strip_pi(&single_pi, 1);
        assert!(matches!(stripped1, Expr::Const(n, _) if n == &Name::from_string("Nat")));

        // Strip 0 from Pi should return the Pi itself
        let stripped0 = strip_pi(&single_pi, 0);
        assert!(matches!(stripped0, Expr::Pi(_, _, _)));

        // Two Pis - strip 1 should return inner Pi, strip 2 should return body
        let inner_pi = Expr::pi(BinderInfo::Default, Expr::type_(), nat_ref.clone());
        let double_pi = Expr::pi(BinderInfo::Default, Expr::prop(), inner_pi);

        let stripped_1of2 = strip_pi(&double_pi, 1);
        assert!(
            matches!(stripped_1of2, Expr::Pi(_, _, _)),
            "Stripping 1 from 2 Pis should leave 1 Pi"
        );

        let stripped_2of2 = strip_pi(&double_pi, 2);
        assert!(matches!(stripped_2of2, Expr::Const(n, _) if n == &Name::from_string("Nat")));

        // Try to strip more than available - should return the final body
        let stripped_3of2 = strip_pi(&double_pi, 3);
        assert!(matches!(stripped_3of2, Expr::Const(n, _) if n == &Name::from_string("Nat")));
    }

    #[test]
    fn test_strip_pi_arithmetic_precise() {
        // Kill mutants: n - 1 replaced with n / 1, n + 1, etc.
        // We need tests that distinguish between n-1 and n/1, n+1

        let nat_ref = Expr::const_(Name::from_string("Nat"), vec![]);

        // Build a chain of 5 Pi types
        let mut expr = nat_ref.clone();
        for i in 0..5 {
            expr = Expr::pi(
                BinderInfo::Default,
                Expr::const_(Name::from_string(&format!("Arg{i}")), vec![]),
                expr,
            );
        }

        // strip_pi(e, 1) should strip exactly 1 (body is Pi with 4 remaining)
        // n-1: after stripping 1, count_pi_args should be 4
        // n/1: would strip 1, still 4
        // n+1: would try to strip 2, leaving 3
        let after_1 = strip_pi(&expr, 1);
        assert_eq!(count_pi_args(after_1), 4, "Stripping 1 should leave 4 Pis");

        // strip_pi(e, 2) should strip exactly 2
        let after_2 = strip_pi(&expr, 2);
        assert_eq!(count_pi_args(after_2), 3, "Stripping 2 should leave 3 Pis");

        // strip_pi(e, 3) should strip exactly 3
        let after_3 = strip_pi(&expr, 3);
        assert_eq!(count_pi_args(after_3), 2, "Stripping 3 should leave 2 Pis");

        // This test specifically catches n-1 vs n+1: with 5 Pis, stripping 2:
        // Correct (n-1): strips 2, leaves 3
        // Wrong (n+1): would try to strip 3, leaves 2
        assert_ne!(
            count_pi_args(after_2),
            2,
            "n-1 vs n+1 distinction: should not be 2"
        );
    }

    #[test]
    fn test_validate_inductive_match_guard() {
        // Kill mutant: replace match guard name == &ind_type.name with true (line 428)
        // This should verify constructor returns the correct inductive type

        let nat = Name::from_string("Nat");
        let nat_ref = Expr::const_(nat.clone(), vec![]);
        let other = Name::from_string("Other");
        let other_ref = Expr::const_(other.clone(), vec![]);

        // Valid: constructor returns the inductive type
        let valid_decl = InductiveDecl {
            level_params: vec![],
            num_params: 0,
            types: vec![InductiveType {
                name: nat.clone(),
                type_: Expr::type_(),
                constructors: vec![Constructor {
                    name: Name::from_string("Nat.zero"),
                    type_: nat_ref.clone(), // Returns Nat
                }],
            }],
        };
        assert!(validate_inductive(&valid_decl).is_ok());

        // Invalid: constructor returns a DIFFERENT type
        let invalid_decl = InductiveDecl {
            level_params: vec![],
            num_params: 0,
            types: vec![InductiveType {
                name: nat.clone(),
                type_: Expr::type_(),
                constructors: vec![Constructor {
                    name: Name::from_string("Nat.bad"),
                    type_: other_ref.clone(), // Returns Other, not Nat!
                }],
            }],
        };
        assert!(
            validate_inductive(&invalid_decl).is_err(),
            "Constructor returning wrong type should fail validation"
        );

        // Invalid: constructor returns wrong type after arrow
        let invalid_arrow = InductiveDecl {
            level_params: vec![],
            num_params: 0,
            types: vec![InductiveType {
                name: nat.clone(),
                type_: Expr::type_(),
                constructors: vec![Constructor {
                    name: Name::from_string("Nat.bad2"),
                    type_: Expr::arrow(nat_ref.clone(), other_ref.clone()), // Nat → Other
                }],
            }],
        };
        assert!(
            validate_inductive(&invalid_arrow).is_err(),
            "Constructor with wrong return type should fail"
        );
    }

    #[test]
    fn test_allows_large_elim_prop_multiple_constructors() {
        // Kill mutant: replace allows_large_elim with true (line 470)
        // Prop types with multiple constructors should NOT allow large elimination

        let prop_type = Expr::prop();

        // In Prop with 2 constructors - should NOT allow large elim
        let ctor1 = Constructor {
            name: Name::from_string("Or.inl"),
            type_: Expr::const_(Name::from_string("Or"), vec![]),
        };
        let ctor2 = Constructor {
            name: Name::from_string("Or.inr"),
            type_: Expr::const_(Name::from_string("Or"), vec![]),
        };

        assert!(
            !allows_large_elim(&prop_type, &[ctor1.clone(), ctor2.clone()]),
            "Prop type with 2 constructors should NOT allow large elimination"
        );

        // In Prop with 3 constructors
        let ctor3 = Constructor {
            name: Name::from_string("Or.third"),
            type_: Expr::const_(Name::from_string("Or"), vec![]),
        };
        let ctors = vec![
            Constructor {
                name: Name::from_string("C1"),
                type_: Expr::const_(Name::from_string("T"), vec![]),
            },
            Constructor {
                name: Name::from_string("C2"),
                type_: Expr::const_(Name::from_string("T"), vec![]),
            },
            ctor3,
        ];

        assert!(
            !allows_large_elim(&prop_type, &ctors),
            "Prop type with 3 constructors should NOT allow large elimination"
        );

        // But Type (not Prop) with multiple constructors DOES allow large elim
        let type_type = Expr::type_();
        assert!(
            allows_large_elim(&type_type, &[ctor1, ctor2]),
            "Type (not Prop) should allow large elimination"
        );
    }
}
