//! Lean5 expression representation
//!
//! This module defines Lean5 expressions suitable for proof generation.
//! Unlike the full Lean5 kernel expressions, these are simplified for
//! the subset needed for invariant proofs.

use std::fmt;

/// A Lean5 name (hierarchical identifier)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Lean5Name(pub Vec<String>);

impl Lean5Name {
    /// Create a simple name
    pub fn simple(s: impl Into<String>) -> Self {
        Lean5Name(vec![s.into()])
    }

    /// Create a hierarchical name from dot-separated string
    pub fn from_string(s: &str) -> Self {
        Lean5Name(s.split('.').map(|p| p.to_string()).collect())
    }

    /// Create a name in a namespace
    pub fn in_namespace(&self, namespace: &str) -> Self {
        let mut parts = vec![namespace.to_string()];
        parts.extend(self.0.clone());
        Lean5Name(parts)
    }
}

impl fmt::Display for Lean5Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = self.0.join(".");
        write!(f, "{name}")
    }
}

/// A Lean5 type
#[derive(Debug, Clone, PartialEq)]
pub enum Lean5Type {
    /// Natural numbers
    Nat,
    /// Integers
    Int,
    /// Booleans (Prop for propositions)
    Bool,
    /// Prop (the type of propositions)
    Prop,
    /// Type (the type of types)
    Type,
    /// Function type: A → B
    Arrow(Box<Lean5Type>, Box<Lean5Type>),
    /// Application: T x
    App(Box<Lean5Type>, Box<Lean5Expr>),
    /// Named type constant
    Const(Lean5Name),
    /// Dependent product: (x : A) → B
    Pi(String, Box<Lean5Type>, Box<Lean5Type>),
}

impl Lean5Type {
    /// Create an arrow type
    pub fn arrow(from: Lean5Type, to: Lean5Type) -> Self {
        Lean5Type::Arrow(Box::new(from), Box::new(to))
    }

    /// Create a pi type
    pub fn pi(var: impl Into<String>, ty: Lean5Type, body: Lean5Type) -> Self {
        Lean5Type::Pi(var.into(), Box::new(ty), Box::new(body))
    }

    /// Create a constant type
    pub fn const_(name: Lean5Name) -> Self {
        Lean5Type::Const(name)
    }
}

impl fmt::Display for Lean5Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Lean5Type::Nat => write!(f, "Nat"),
            Lean5Type::Int => write!(f, "Int"),
            Lean5Type::Bool => write!(f, "Bool"),
            Lean5Type::Prop => write!(f, "Prop"),
            Lean5Type::Type => write!(f, "Type"),
            Lean5Type::Arrow(a, b) => write!(f, "{a} → {b}"),
            Lean5Type::App(t, arg) => write!(f, "({t} {arg})"),
            Lean5Type::Const(n) => write!(f, "{n}"),
            Lean5Type::Pi(var, ty, body) => write!(f, "({var} : {ty}) → {body}"),
        }
    }
}

/// A Lean5 expression
#[derive(Debug, Clone, PartialEq)]
pub enum Lean5Expr {
    /// Variable reference
    Var(String),
    /// Natural number literal
    NatLit(u64),
    /// Integer literal (can be negative)
    IntLit(i64),
    /// Boolean literal
    BoolLit(bool),
    /// Constant reference
    Const(Lean5Name),
    /// Function application
    App(Box<Lean5Expr>, Box<Lean5Expr>),
    /// Lambda abstraction: fun (x : T) => body
    Lam(String, Lean5Type, Box<Lean5Expr>),
    /// Let binding: let x : T := val in body
    Let(String, Lean5Type, Box<Lean5Expr>, Box<Lean5Expr>),
    /// Forall/Pi: ∀ (x : T), body
    Forall(String, Lean5Type, Box<Lean5Expr>),
    /// Exists: ∃ (x : T), body
    Exists(String, Lean5Type, Box<Lean5Expr>),
    /// Conjunction: a ∧ b
    And(Box<Lean5Expr>, Box<Lean5Expr>),
    /// Disjunction: a ∨ b
    Or(Box<Lean5Expr>, Box<Lean5Expr>),
    /// Negation: ¬ a
    Not(Box<Lean5Expr>),
    /// Implication: a → b
    Implies(Box<Lean5Expr>, Box<Lean5Expr>),
    /// Equality: a = b
    Eq(Box<Lean5Expr>, Box<Lean5Expr>),
    /// Less than: a < b
    Lt(Box<Lean5Expr>, Box<Lean5Expr>),
    /// Less than or equal: a ≤ b
    Le(Box<Lean5Expr>, Box<Lean5Expr>),
    /// Greater than: a > b
    Gt(Box<Lean5Expr>, Box<Lean5Expr>),
    /// Greater than or equal: a ≥ b
    Ge(Box<Lean5Expr>, Box<Lean5Expr>),
    /// Addition: a + b
    Add(Box<Lean5Expr>, Box<Lean5Expr>),
    /// Subtraction: a - b
    Sub(Box<Lean5Expr>, Box<Lean5Expr>),
    /// Multiplication: a * b
    Mul(Box<Lean5Expr>, Box<Lean5Expr>),
    /// Division: a / b
    Div(Box<Lean5Expr>, Box<Lean5Expr>),
    /// Modulo: a % b
    Mod(Box<Lean5Expr>, Box<Lean5Expr>),
    /// Negation (arithmetic): -a
    Neg(Box<Lean5Expr>),
    /// If-then-else: if c then t else e
    Ite(Box<Lean5Expr>, Box<Lean5Expr>, Box<Lean5Expr>),
    /// Type ascription: (e : T)
    Ascribe(Box<Lean5Expr>, Lean5Type),
    /// Placeholder (hole to be filled)
    Hole,
    /// Sorry (admitted proof)
    Sorry,
}

impl Lean5Expr {
    /// Create a variable
    pub fn var(name: impl Into<String>) -> Self {
        Lean5Expr::Var(name.into())
    }

    /// Create a constant
    pub fn const_(name: Lean5Name) -> Self {
        Lean5Expr::Const(name)
    }

    /// Create an application
    pub fn app(func: Lean5Expr, arg: Lean5Expr) -> Self {
        Lean5Expr::App(Box::new(func), Box::new(arg))
    }

    /// Create multiple applications
    pub fn apps(func: Lean5Expr, args: impl IntoIterator<Item = Lean5Expr>) -> Self {
        args.into_iter().fold(func, Lean5Expr::app)
    }

    /// Create a lambda
    pub fn lam(var: impl Into<String>, ty: Lean5Type, body: Lean5Expr) -> Self {
        Lean5Expr::Lam(var.into(), ty, Box::new(body))
    }

    /// Create a forall
    pub fn forall_(var: impl Into<String>, ty: Lean5Type, body: Lean5Expr) -> Self {
        Lean5Expr::Forall(var.into(), ty, Box::new(body))
    }

    /// Create an exists
    pub fn exists_(var: impl Into<String>, ty: Lean5Type, body: Lean5Expr) -> Self {
        Lean5Expr::Exists(var.into(), ty, Box::new(body))
    }

    /// Create conjunction
    pub fn and(a: Lean5Expr, b: Lean5Expr) -> Self {
        Lean5Expr::And(Box::new(a), Box::new(b))
    }

    /// Create disjunction
    pub fn or(a: Lean5Expr, b: Lean5Expr) -> Self {
        Lean5Expr::Or(Box::new(a), Box::new(b))
    }

    /// Create negation (logical NOT)
    #[allow(clippy::should_implement_trait)]
    pub fn not(a: Lean5Expr) -> Self {
        Lean5Expr::Not(Box::new(a))
    }

    /// Create implication
    pub fn implies(a: Lean5Expr, b: Lean5Expr) -> Self {
        Lean5Expr::Implies(Box::new(a), Box::new(b))
    }

    /// Create equality
    pub fn eq(a: Lean5Expr, b: Lean5Expr) -> Self {
        Lean5Expr::Eq(Box::new(a), Box::new(b))
    }

    /// Create less than
    pub fn lt(a: Lean5Expr, b: Lean5Expr) -> Self {
        Lean5Expr::Lt(Box::new(a), Box::new(b))
    }

    /// Create less than or equal
    pub fn le(a: Lean5Expr, b: Lean5Expr) -> Self {
        Lean5Expr::Le(Box::new(a), Box::new(b))
    }

    /// Create greater than
    pub fn gt(a: Lean5Expr, b: Lean5Expr) -> Self {
        Lean5Expr::Gt(Box::new(a), Box::new(b))
    }

    /// Create greater than or equal
    pub fn ge(a: Lean5Expr, b: Lean5Expr) -> Self {
        Lean5Expr::Ge(Box::new(a), Box::new(b))
    }

    /// Create addition
    #[allow(clippy::should_implement_trait)]
    pub fn add(a: Lean5Expr, b: Lean5Expr) -> Self {
        Lean5Expr::Add(Box::new(a), Box::new(b))
    }

    /// Create subtraction
    #[allow(clippy::should_implement_trait)]
    pub fn sub(a: Lean5Expr, b: Lean5Expr) -> Self {
        Lean5Expr::Sub(Box::new(a), Box::new(b))
    }

    /// Create multiplication
    #[allow(clippy::should_implement_trait)]
    pub fn mul(a: Lean5Expr, b: Lean5Expr) -> Self {
        Lean5Expr::Mul(Box::new(a), Box::new(b))
    }

    /// Create if-then-else
    pub fn ite(cond: Lean5Expr, then_: Lean5Expr, else_: Lean5Expr) -> Self {
        Lean5Expr::Ite(Box::new(cond), Box::new(then_), Box::new(else_))
    }
}

impl fmt::Display for Lean5Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Lean5Expr::Var(name) => write!(f, "{name}"),
            Lean5Expr::NatLit(n) => write!(f, "{n}"),
            Lean5Expr::IntLit(n) => {
                if *n < 0 {
                    write!(f, "({n})")
                } else {
                    write!(f, "{n}")
                }
            }
            Lean5Expr::BoolLit(b) => write!(f, "{b}"),
            Lean5Expr::Const(name) => write!(f, "{name}"),
            Lean5Expr::App(func, arg) => write!(f, "({func} {arg})"),
            Lean5Expr::Lam(var, ty, body) => write!(f, "fun ({var} : {ty}) => {body}"),
            Lean5Expr::Let(var, ty, val, body) => {
                write!(f, "let {var} : {ty} := {val} in {body}")
            }
            Lean5Expr::Forall(var, ty, body) => write!(f, "∀ ({var} : {ty}), {body}"),
            Lean5Expr::Exists(var, ty, body) => write!(f, "∃ ({var} : {ty}), {body}"),
            Lean5Expr::And(a, b) => write!(f, "({a} ∧ {b})"),
            Lean5Expr::Or(a, b) => write!(f, "({a} ∨ {b})"),
            Lean5Expr::Not(a) => write!(f, "¬{a}"),
            Lean5Expr::Implies(a, b) => write!(f, "({a} → {b})"),
            Lean5Expr::Eq(a, b) => write!(f, "({a} = {b})"),
            Lean5Expr::Lt(a, b) => write!(f, "({a} < {b})"),
            Lean5Expr::Le(a, b) => write!(f, "({a} ≤ {b})"),
            Lean5Expr::Gt(a, b) => write!(f, "({a} > {b})"),
            Lean5Expr::Ge(a, b) => write!(f, "({a} ≥ {b})"),
            Lean5Expr::Add(a, b) => write!(f, "({a} + {b})"),
            Lean5Expr::Sub(a, b) => write!(f, "({a} - {b})"),
            Lean5Expr::Mul(a, b) => write!(f, "({a} * {b})"),
            Lean5Expr::Div(a, b) => write!(f, "({a} / {b})"),
            Lean5Expr::Mod(a, b) => write!(f, "({a} % {b})"),
            Lean5Expr::Neg(a) => write!(f, "(-{a})"),
            Lean5Expr::Ite(c, t, e) => write!(f, "if {c} then {t} else {e}"),
            Lean5Expr::Ascribe(e, ty) => write!(f, "({e} : {ty})"),
            Lean5Expr::Hole => write!(f, "_"),
            Lean5Expr::Sorry => write!(f, "sorry"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===========================================
    // Lean5Name tests
    // ===========================================

    #[test]
    fn test_lean5_name_simple() {
        let name = Lean5Name::simple("x");
        assert_eq!(format!("{}", name), "x");
        assert_eq!(name.0, vec!["x".to_string()]);
    }

    #[test]
    fn test_lean5_name_simple_with_string_type() {
        let name = Lean5Name::simple(String::from("myVar"));
        assert_eq!(format!("{}", name), "myVar");
    }

    #[test]
    fn test_lean5_name_from_string_single() {
        let name = Lean5Name::from_string("Nat");
        assert_eq!(format!("{}", name), "Nat");
        assert_eq!(name.0, vec!["Nat".to_string()]);
    }

    #[test]
    fn test_lean5_name_from_string_multi() {
        let name = Lean5Name::from_string("Nat.add");
        assert_eq!(format!("{}", name), "Nat.add");
        assert_eq!(name.0, vec!["Nat".to_string(), "add".to_string()]);
    }

    #[test]
    fn test_lean5_name_from_string_deep_hierarchy() {
        let name = Lean5Name::from_string("Mathlib.Data.Nat.Basic.succ");
        assert_eq!(format!("{}", name), "Mathlib.Data.Nat.Basic.succ");
        assert_eq!(name.0.len(), 5);
    }

    #[test]
    fn test_lean5_name_in_namespace() {
        let name = Lean5Name::simple("x");
        let namespaced = name.in_namespace("KaniFast");
        assert_eq!(format!("{}", namespaced), "KaniFast.x");
        assert_eq!(namespaced, Lean5Name(vec!["KaniFast".into(), "x".into()]));
    }

    #[test]
    fn test_lean5_name_in_namespace_hierarchical() {
        let name = Lean5Name::from_string("Nat.add");
        let namespaced = name.in_namespace("KaniFast");
        assert_eq!(format!("{}", namespaced), "KaniFast.Nat.add");
        assert_eq!(
            namespaced,
            Lean5Name(vec!["KaniFast".into(), "Nat".into(), "add".into()])
        );
    }

    #[test]
    fn test_lean5_name_clone() {
        let name = Lean5Name::from_string("Test.Name");
        let cloned = name.clone();
        assert_eq!(name, cloned);
    }

    #[test]
    fn test_lean5_name_debug() {
        let name = Lean5Name::simple("debug_test");
        let debug_str = format!("{:?}", name);
        assert!(debug_str.contains("Lean5Name"));
        assert!(debug_str.contains("debug_test"));
    }

    #[test]
    fn test_lean5_name_eq() {
        let n1 = Lean5Name::from_string("Nat.add");
        let n2 = Lean5Name::from_string("Nat.add");
        let n3 = Lean5Name::from_string("Nat.sub");
        assert_eq!(n1, n2);
        assert_ne!(n1, n3);
    }

    #[test]
    fn test_lean5_name_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Lean5Name::from_string("Nat.add"));
        set.insert(Lean5Name::from_string("Nat.sub"));
        assert_eq!(set.len(), 2);
        assert!(set.contains(&Lean5Name::from_string("Nat.add")));
    }

    // ===========================================
    // Lean5Type tests
    // ===========================================

    #[test]
    fn test_lean5_type_nat_display() {
        assert_eq!(format!("{}", Lean5Type::Nat), "Nat");
    }

    #[test]
    fn test_lean5_type_int_display() {
        assert_eq!(format!("{}", Lean5Type::Int), "Int");
    }

    #[test]
    fn test_lean5_type_bool_display() {
        assert_eq!(format!("{}", Lean5Type::Bool), "Bool");
    }

    #[test]
    fn test_lean5_type_prop_display() {
        assert_eq!(format!("{}", Lean5Type::Prop), "Prop");
    }

    #[test]
    fn test_lean5_type_type_display() {
        assert_eq!(format!("{}", Lean5Type::Type), "Type");
    }

    #[test]
    fn test_lean5_type_arrow() {
        let ty = Lean5Type::arrow(Lean5Type::Int, Lean5Type::Bool);
        assert_eq!(format!("{}", ty), "Int → Bool");
    }

    #[test]
    fn test_lean5_type_arrow_nested() {
        let ty = Lean5Type::arrow(
            Lean5Type::Nat,
            Lean5Type::arrow(Lean5Type::Nat, Lean5Type::Bool),
        );
        assert_eq!(format!("{}", ty), "Nat → Nat → Bool");
    }

    #[test]
    fn test_lean5_type_pi() {
        let ty = Lean5Type::pi("x", Lean5Type::Int, Lean5Type::Prop);
        assert_eq!(format!("{}", ty), "(x : Int) → Prop");
    }

    #[test]
    fn test_lean5_type_pi_with_string() {
        let ty = Lean5Type::pi(String::from("var"), Lean5Type::Nat, Lean5Type::Bool);
        assert_eq!(format!("{}", ty), "(var : Nat) → Bool");
    }

    #[test]
    fn test_lean5_type_app() {
        let app_ty = Lean5Type::App(Box::new(Lean5Type::Int), Box::new(Lean5Expr::var("x")));
        assert_eq!(format!("{}", app_ty), "(Int x)");
    }

    #[test]
    fn test_lean5_type_const() {
        let const_ty = Lean5Type::const_(Lean5Name::from_string("Mathlib.Real"));
        assert_eq!(format!("{}", const_ty), "Mathlib.Real");
    }

    #[test]
    fn test_lean5_type_clone() {
        let ty = Lean5Type::arrow(Lean5Type::Int, Lean5Type::Bool);
        let cloned = ty.clone();
        assert_eq!(ty, cloned);
    }

    #[test]
    fn test_lean5_type_debug() {
        let ty = Lean5Type::Int;
        let debug_str = format!("{:?}", ty);
        assert!(debug_str.contains("Int"));
    }

    #[test]
    fn test_lean5_type_eq() {
        let t1 = Lean5Type::Int;
        let t2 = Lean5Type::Int;
        let t3 = Lean5Type::Nat;
        assert_eq!(t1, t2);
        assert_ne!(t1, t3);
    }

    // ===========================================
    // Lean5Expr literal tests
    // ===========================================

    #[test]
    fn test_lean5_expr_var() {
        let expr = Lean5Expr::var("x");
        assert_eq!(format!("{}", expr), "x");
    }

    #[test]
    fn test_lean5_expr_var_with_string() {
        let expr = Lean5Expr::var(String::from("myVariable"));
        assert_eq!(format!("{}", expr), "myVariable");
    }

    #[test]
    fn test_lean5_expr_nat_lit() {
        assert_eq!(format!("{}", Lean5Expr::NatLit(0)), "0");
        assert_eq!(format!("{}", Lean5Expr::NatLit(42)), "42");
        assert_eq!(
            format!("{}", Lean5Expr::NatLit(u64::MAX)),
            format!("{}", u64::MAX)
        );
    }

    #[test]
    fn test_lean5_expr_int_lit_positive() {
        assert_eq!(format!("{}", Lean5Expr::IntLit(0)), "0");
        assert_eq!(format!("{}", Lean5Expr::IntLit(42)), "42");
    }

    #[test]
    fn test_lean5_expr_int_lit_negative() {
        assert_eq!(format!("{}", Lean5Expr::IntLit(-1)), "(-1)");
        assert_eq!(format!("{}", Lean5Expr::IntLit(-42)), "(-42)");
    }

    #[test]
    fn test_lean5_expr_bool_lit() {
        assert_eq!(format!("{}", Lean5Expr::BoolLit(true)), "true");
        assert_eq!(format!("{}", Lean5Expr::BoolLit(false)), "false");
    }

    #[test]
    fn test_lean5_expr_const() {
        let expr = Lean5Expr::const_(Lean5Name::from_string("Nat.zero"));
        assert_eq!(format!("{}", expr), "Nat.zero");
    }

    #[test]
    fn test_lean5_expr_hole() {
        assert_eq!(format!("{}", Lean5Expr::Hole), "_");
    }

    #[test]
    fn test_lean5_expr_sorry() {
        assert_eq!(format!("{}", Lean5Expr::Sorry), "sorry");
    }

    // ===========================================
    // Lean5Expr application tests
    // ===========================================

    #[test]
    fn test_lean5_expr_app_simple() {
        let expr = Lean5Expr::app(Lean5Expr::var("f"), Lean5Expr::var("x"));
        assert_eq!(format!("{}", expr), "(f x)");
    }

    #[test]
    fn test_lean5_expr_apps_multiple() {
        let expr = Lean5Expr::apps(
            Lean5Expr::const_(Lean5Name::simple("f")),
            vec![Lean5Expr::var("x"), Lean5Expr::var("y")],
        );
        assert_eq!(format!("{}", expr), "((f x) y)");
    }

    #[test]
    fn test_lean5_expr_apps_empty() {
        let no_args = Lean5Expr::apps(Lean5Expr::var("id"), Vec::new());
        assert_eq!(no_args, Lean5Expr::var("id"));
    }

    #[test]
    fn test_lean5_expr_apps_single() {
        let expr = Lean5Expr::apps(Lean5Expr::var("f"), vec![Lean5Expr::NatLit(5)]);
        assert_eq!(format!("{}", expr), "(f 5)");
    }

    // ===========================================
    // Lean5Expr lambda and let tests
    // ===========================================

    #[test]
    fn test_lean5_expr_lam() {
        let lam = Lean5Expr::lam("x", Lean5Type::Int, Lean5Expr::var("x"));
        assert_eq!(format!("{}", lam), "fun (x : Int) => x");
    }

    #[test]
    fn test_lean5_expr_lam_with_string() {
        let lam = Lean5Expr::lam(String::from("n"), Lean5Type::Nat, Lean5Expr::NatLit(0));
        assert_eq!(format!("{}", lam), "fun (n : Nat) => 0");
    }

    #[test]
    fn test_lean5_expr_lam_complex() {
        let lam = Lean5Expr::lam(
            "n",
            Lean5Type::Nat,
            Lean5Expr::implies(
                Lean5Expr::ge(Lean5Expr::var("n"), Lean5Expr::NatLit(0)),
                Lean5Expr::BoolLit(true),
            ),
        );
        assert!(format!("{}", lam).contains("fun (n : Nat)"));
    }

    #[test]
    fn test_lean5_expr_let() {
        let let_expr = Lean5Expr::Let(
            "a".into(),
            Lean5Type::Int,
            Box::new(Lean5Expr::IntLit(5)),
            Box::new(Lean5Expr::add(Lean5Expr::var("a"), Lean5Expr::IntLit(1))),
        );
        assert_eq!(format!("{}", let_expr), "let a : Int := 5 in (a + 1)");
    }

    // ===========================================
    // Lean5Expr quantifier tests
    // ===========================================

    #[test]
    fn test_lean5_expr_forall() {
        let expr = Lean5Expr::forall_(
            "x",
            Lean5Type::Int,
            Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
        );
        assert_eq!(format!("{}", expr), "∀ (x : Int), (x ≥ 0)");
    }

    #[test]
    fn test_lean5_expr_forall_with_string() {
        let expr = Lean5Expr::forall_(String::from("n"), Lean5Type::Nat, Lean5Expr::BoolLit(true));
        assert_eq!(format!("{}", expr), "∀ (n : Nat), true");
    }

    #[test]
    fn test_lean5_expr_exists() {
        let expr = Lean5Expr::exists_(
            "y",
            Lean5Type::Int,
            Lean5Expr::eq(Lean5Expr::var("y"), Lean5Expr::IntLit(0)),
        );
        assert_eq!(format!("{}", expr), "∃ (y : Int), (y = 0)");
    }

    #[test]
    fn test_lean5_expr_exists_with_string() {
        let expr = Lean5Expr::exists_(
            String::from("n"),
            Lean5Type::Nat,
            Lean5Expr::gt(Lean5Expr::var("n"), Lean5Expr::NatLit(0)),
        );
        assert_eq!(format!("{}", expr), "∃ (n : Nat), (n > 0)");
    }

    // ===========================================
    // Lean5Expr logical operator tests
    // ===========================================

    #[test]
    fn test_lean5_expr_and() {
        let expr = Lean5Expr::and(Lean5Expr::BoolLit(true), Lean5Expr::BoolLit(false));
        assert_eq!(format!("{}", expr), "(true ∧ false)");
    }

    #[test]
    fn test_lean5_expr_or() {
        let expr = Lean5Expr::or(Lean5Expr::BoolLit(true), Lean5Expr::BoolLit(false));
        assert_eq!(format!("{}", expr), "(true ∨ false)");
    }

    #[test]
    fn test_lean5_expr_not() {
        let expr = Lean5Expr::not(Lean5Expr::BoolLit(true));
        assert_eq!(format!("{}", expr), "¬true");
    }

    #[test]
    fn test_lean5_expr_implies() {
        let expr = Lean5Expr::implies(Lean5Expr::var("p"), Lean5Expr::var("q"));
        assert_eq!(format!("{}", expr), "(p → q)");
    }

    #[test]
    fn test_lean5_expr_logical_nested() {
        let expr = Lean5Expr::implies(
            Lean5Expr::and(Lean5Expr::var("p"), Lean5Expr::var("q")),
            Lean5Expr::or(Lean5Expr::var("r"), Lean5Expr::not(Lean5Expr::var("s"))),
        );
        assert_eq!(format!("{}", expr), "((p ∧ q) → (r ∨ ¬s))");
    }

    // ===========================================
    // Lean5Expr comparison tests
    // ===========================================

    #[test]
    fn test_lean5_expr_eq() {
        let expr = Lean5Expr::eq(Lean5Expr::var("x"), Lean5Expr::IntLit(5));
        assert_eq!(format!("{}", expr), "(x = 5)");
    }

    #[test]
    fn test_lean5_expr_lt() {
        let expr = Lean5Expr::lt(Lean5Expr::var("x"), Lean5Expr::IntLit(10));
        assert_eq!(format!("{}", expr), "(x < 10)");
    }

    #[test]
    fn test_lean5_expr_le() {
        let expr = Lean5Expr::le(Lean5Expr::var("x"), Lean5Expr::IntLit(10));
        assert_eq!(format!("{}", expr), "(x ≤ 10)");
    }

    #[test]
    fn test_lean5_expr_gt() {
        let expr = Lean5Expr::gt(Lean5Expr::var("x"), Lean5Expr::IntLit(0));
        assert_eq!(format!("{}", expr), "(x > 0)");
    }

    #[test]
    fn test_lean5_expr_ge() {
        let expr = Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0));
        assert_eq!(format!("{}", expr), "(x ≥ 0)");
    }

    // ===========================================
    // Lean5Expr arithmetic tests
    // ===========================================

    #[test]
    fn test_lean5_expr_add() {
        let expr = Lean5Expr::add(Lean5Expr::var("x"), Lean5Expr::IntLit(1));
        assert_eq!(format!("{}", expr), "(x + 1)");
    }

    #[test]
    fn test_lean5_expr_sub() {
        let expr = Lean5Expr::sub(Lean5Expr::var("x"), Lean5Expr::IntLit(1));
        assert_eq!(format!("{}", expr), "(x - 1)");
    }

    #[test]
    fn test_lean5_expr_mul() {
        let expr = Lean5Expr::mul(Lean5Expr::var("x"), Lean5Expr::IntLit(2));
        assert_eq!(format!("{}", expr), "(x * 2)");
    }

    #[test]
    fn test_lean5_expr_div() {
        let expr = Lean5Expr::Div(
            Box::new(Lean5Expr::var("x")),
            Box::new(Lean5Expr::IntLit(2)),
        );
        assert_eq!(format!("{}", expr), "(x / 2)");
    }

    #[test]
    fn test_lean5_expr_mod() {
        let expr = Lean5Expr::Mod(
            Box::new(Lean5Expr::var("x")),
            Box::new(Lean5Expr::IntLit(3)),
        );
        assert_eq!(format!("{}", expr), "(x % 3)");
    }

    #[test]
    fn test_lean5_expr_neg() {
        let expr = Lean5Expr::Neg(Box::new(Lean5Expr::var("x")));
        assert_eq!(format!("{}", expr), "(-x)");
    }

    #[test]
    fn test_lean5_expr_arithmetic_nested() {
        let expr = Lean5Expr::add(
            Lean5Expr::mul(Lean5Expr::var("a"), Lean5Expr::var("b")),
            Lean5Expr::IntLit(1),
        );
        assert_eq!(format!("{}", expr), "((a * b) + 1)");
    }

    // ===========================================
    // Lean5Expr ite (if-then-else) tests
    // ===========================================

    #[test]
    fn test_lean5_expr_ite() {
        let ite = Lean5Expr::ite(
            Lean5Expr::gt(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
            Lean5Expr::NatLit(1),
            Lean5Expr::NatLit(0),
        );
        assert_eq!(format!("{}", ite), "if (x > 0) then 1 else 0");
    }

    #[test]
    fn test_lean5_expr_ite_with_bool() {
        let ite = Lean5Expr::ite(
            Lean5Expr::BoolLit(true),
            Lean5Expr::var("a"),
            Lean5Expr::var("b"),
        );
        assert_eq!(format!("{}", ite), "if true then a else b");
    }

    #[test]
    fn test_lean5_expr_ite_nested() {
        let ite = Lean5Expr::ite(
            Lean5Expr::var("cond1"),
            Lean5Expr::ite(
                Lean5Expr::var("cond2"),
                Lean5Expr::NatLit(1),
                Lean5Expr::NatLit(2),
            ),
            Lean5Expr::NatLit(3),
        );
        assert_eq!(
            format!("{}", ite),
            "if cond1 then if cond2 then 1 else 2 else 3"
        );
    }

    // ===========================================
    // Lean5Expr ascribe tests
    // ===========================================

    #[test]
    fn test_lean5_expr_ascribe() {
        let expr = Lean5Expr::Ascribe(Box::new(Lean5Expr::NatLit(42)), Lean5Type::Nat);
        assert_eq!(format!("{}", expr), "(42 : Nat)");
    }

    #[test]
    fn test_lean5_expr_ascribe_complex() {
        let expr = Lean5Expr::Ascribe(
            Box::new(Lean5Expr::add(Lean5Expr::var("x"), Lean5Expr::IntLit(1))),
            Lean5Type::Int,
        );
        assert_eq!(format!("{}", expr), "((x + 1) : Int)");
    }

    // ===========================================
    // Lean5Expr clone and eq tests
    // ===========================================

    #[test]
    fn test_lean5_expr_clone() {
        let expr = Lean5Expr::add(Lean5Expr::var("x"), Lean5Expr::IntLit(1));
        let cloned = expr.clone();
        assert_eq!(expr, cloned);
    }

    #[test]
    fn test_lean5_expr_eq_same() {
        let e1 = Lean5Expr::var("x");
        let e2 = Lean5Expr::var("x");
        assert_eq!(e1, e2);
    }

    #[test]
    fn test_lean5_expr_eq_different() {
        let e1 = Lean5Expr::var("x");
        let e2 = Lean5Expr::var("y");
        assert_ne!(e1, e2);
    }

    #[test]
    fn test_lean5_expr_debug() {
        let expr = Lean5Expr::var("test");
        let debug_str = format!("{:?}", expr);
        assert!(debug_str.contains("Var"));
        assert!(debug_str.contains("test"));
    }

    // ===========================================
    // Complex expression tests
    // ===========================================

    #[test]
    fn test_lean5_expr_complex_invariant() {
        // Build: ∀ (n : Nat), n ≥ 0 ∧ n < 100 → n + 1 ≤ 100
        let invariant = Lean5Expr::forall_(
            "n",
            Lean5Type::Nat,
            Lean5Expr::implies(
                Lean5Expr::and(
                    Lean5Expr::ge(Lean5Expr::var("n"), Lean5Expr::NatLit(0)),
                    Lean5Expr::lt(Lean5Expr::var("n"), Lean5Expr::NatLit(100)),
                ),
                Lean5Expr::le(
                    Lean5Expr::add(Lean5Expr::var("n"), Lean5Expr::NatLit(1)),
                    Lean5Expr::NatLit(100),
                ),
            ),
        );
        let s = format!("{}", invariant);
        assert!(s.contains("∀ (n : Nat)"));
        assert!(s.contains("≥"));
        assert!(s.contains("∧"));
        assert!(s.contains("→"));
    }

    #[test]
    fn test_lean5_expr_complex_exists() {
        // ∃ (x : Int), x > 0 ∧ x < 10
        let expr = Lean5Expr::exists_(
            "x",
            Lean5Type::Int,
            Lean5Expr::and(
                Lean5Expr::gt(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
                Lean5Expr::lt(Lean5Expr::var("x"), Lean5Expr::IntLit(10)),
            ),
        );
        let s = format!("{}", expr);
        assert!(s.contains("∃ (x : Int)"));
        assert!(s.contains("x > 0"));
        assert!(s.contains("x < 10"));
    }

    #[test]
    fn test_lean5_expr_deeply_nested() {
        // Build a deeply nested expression
        let base = Lean5Expr::IntLit(0);
        let mut expr = base;
        for i in 0..5 {
            expr = Lean5Expr::add(Lean5Expr::var(format!("x{}", i)), expr);
        }
        let s = format!("{}", expr);
        assert!(s.contains("x0"));
        assert!(s.contains("x4"));
    }

    #[test]
    fn test_lean5_type_deeply_nested_arrow() {
        // Build Int → Int → Int → Bool
        let ty = Lean5Type::arrow(
            Lean5Type::Int,
            Lean5Type::arrow(
                Lean5Type::Int,
                Lean5Type::arrow(Lean5Type::Int, Lean5Type::Bool),
            ),
        );
        assert_eq!(format!("{}", ty), "Int → Int → Int → Bool");
    }

    #[test]
    fn test_lean5_name_empty_parts() {
        // Edge case: empty string creates single empty part
        let name = Lean5Name::from_string("");
        assert_eq!(name.0.len(), 1);
        assert_eq!(name.0[0], "");
    }

    #[test]
    fn test_lean5_expr_all_comparison_ops() {
        let x = Lean5Expr::var("x");
        let y = Lean5Expr::var("y");

        assert!(format!("{}", Lean5Expr::lt(x.clone(), y.clone())).contains("<"));
        assert!(format!("{}", Lean5Expr::le(x.clone(), y.clone())).contains("≤"));
        assert!(format!("{}", Lean5Expr::gt(x.clone(), y.clone())).contains(">"));
        assert!(format!("{}", Lean5Expr::ge(x.clone(), y.clone())).contains("≥"));
        assert!(format!("{}", Lean5Expr::eq(x, y)).contains("="));
    }
}
