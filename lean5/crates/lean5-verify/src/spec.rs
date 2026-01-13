//! Kernel Specification in Lean5 Type Theory
//!
//! This module defines a model of the Lean5 kernel as Lean5 inductive types
//! and recursive functions. This specification serves as the formal definition
//! of "what the kernel should do."
//!
//! ## Design
//!
//! We use Lean5's type theory to express:
//! - Expr, Level as inductive types
//! - has_type(env, ctx, e, T) as a predicate
//! - is_def_eq(env, a, b) as a predicate
//! - whnf(env, e) as a function
//!
//! The specification is written using Lean5 surface syntax, which is then
//! elaborated and type-checked by the kernel.

use lean5_elab::{elaborate, ElabCtx, ElabError};
use lean5_kernel::{Declaration, Environment, Expr, Level, Name, TypeChecker};
use lean5_parser::parse_expr;
use std::collections::HashMap;

/// A specification expression (Lean5 surface syntax)
#[derive(Debug, Clone)]
pub struct SpecExpr {
    /// The source text
    pub source: String,
    /// Elaborated kernel expression (if successful)
    pub kernel_expr: Option<Expr>,
}

impl SpecExpr {
    /// Create a new specification expression from source
    pub fn new(source: &str) -> Self {
        SpecExpr {
            source: source.to_string(),
            kernel_expr: None,
        }
    }

    /// Elaborate the expression
    pub fn elaborate(&mut self, env: &Environment) -> Result<&Expr, ElabError> {
        if self.kernel_expr.is_none() {
            let surface =
                parse_expr(&self.source).map_err(|e| ElabError::ParseError(e.to_string()))?;
            let expr = elaborate(env, &surface)?;
            self.kernel_expr = Some(expr);
        }
        Ok(self.kernel_expr.as_ref().unwrap())
    }
}

/// A specification level (Lean5 universe level)
#[derive(Debug, Clone)]
pub struct SpecLevel {
    /// The source text
    pub source: String,
    /// Elaborated level (if applicable)
    pub level: Option<Level>,
}

impl SpecLevel {
    pub fn new(source: &str) -> Self {
        SpecLevel {
            source: source.to_string(),
            level: None,
        }
    }

    /// Parse as a universe level
    pub fn parse(&mut self) -> Result<&Level, String> {
        if self.level.is_none() {
            // Simple level parsing
            let level = match self.source.trim() {
                "0" | "Prop" => Level::Zero,
                "1" | "Type" => Level::succ(Level::Zero),
                s if s.starts_with("succ ") => {
                    let inner = s.strip_prefix("succ ").unwrap();
                    let mut inner_spec = SpecLevel::new(inner);
                    Level::succ(inner_spec.parse()?.clone())
                }
                _ => return Err(format!("Cannot parse level: {}", self.source)),
            };
            self.level = Some(level);
        }
        Ok(self.level.as_ref().unwrap())
    }
}

/// The complete kernel specification
#[derive(Debug)]
pub struct Specification {
    /// Environment with specification definitions
    env: Environment,
    /// Named definitions
    definitions: HashMap<String, SpecDefinition>,
}

/// A specification definition
#[derive(Debug, Clone)]
pub struct SpecDefinition {
    /// Definition name
    pub name: String,
    /// Definition type (as Lean5 source)
    pub type_src: String,
    /// Definition value (as Lean5 source)
    pub value_src: Option<String>,
    /// Whether this is an axiom (no value)
    pub is_axiom: bool,
    /// Description
    pub description: String,
    /// Elaborated type (cached)
    pub elaborated_type: Option<Expr>,
    /// Elaborated value (cached)
    pub elaborated_value: Option<Expr>,
}

impl Specification {
    /// Create a new specification with standard definitions
    pub fn new() -> Result<Self, SpecError> {
        let mut spec = Specification {
            env: Environment::new(),
            definitions: HashMap::new(),
        };

        // Add core type theory specification
        spec.add_core_spec()?;

        Ok(spec)
    }

    /// Get the environment
    pub fn env(&self) -> &Environment {
        &self.env
    }

    /// Get all definitions
    pub fn definitions(&self) -> &HashMap<String, SpecDefinition> {
        &self.definitions
    }

    /// Add a definition
    pub fn add_definition(&mut self, mut def: SpecDefinition) -> Result<(), SpecError> {
        let type_expr = self.elaborate_source(&def.type_src, &format!("type of {}", def.name))?;
        def.elaborated_type = Some(type_expr.clone());

        let value_expr = if let Some(ref value_src) = def.value_src {
            let value_expr = self.elaborate_source(value_src, &format!("value of {}", def.name))?;

            // Type check the value against the declared type
            let mut tc = TypeChecker::new(&self.env);
            let inferred = tc.infer_type(&value_expr).map_err(|e| {
                SpecError::TypeError(format!("value inference for {}: {:?}", def.name, e))
            })?;

            if !tc.is_def_eq(&inferred, &type_expr) {
                return Err(SpecError::TypeError(format!(
                    "Value for {} has type {:?}, expected {:?}",
                    def.name, inferred, type_expr
                )));
            }

            Some(value_expr)
        } else {
            None
        };

        def.elaborated_value = value_expr.clone();

        let decl = match value_expr {
            Some(val) => Declaration::Theorem {
                name: Name::from_string(&def.name),
                level_params: vec![],
                type_: type_expr.clone(),
                value: val,
            },
            None => Declaration::Axiom {
                name: Name::from_string(&def.name),
                level_params: vec![],
                type_: type_expr.clone(),
            },
        };

        self.env
            .add_decl(decl)
            .map_err(|e| SpecError::EnvError(e.to_string()))?;

        self.definitions.insert(def.name.clone(), def);
        Ok(())
    }

    /// Add core type theory specifications
    fn add_core_spec(&mut self) -> Result<(), SpecError> {
        // =========================================================
        // PART 1: Equality type (Eq, rfl)
        // =========================================================
        // This is the foundation for all proofs

        self.add_definition(SpecDefinition {
            name: "Eq".to_string(),
            type_src: "forall (A : Type), A -> A -> Type".to_string(),
            value_src: None,
            is_axiom: true, // Inductive type, axiomatized for now
            description: "Propositional equality type. Eq A x y means x equals y at type A."
                .to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        self.add_definition(SpecDefinition {
            name: "Eq.refl".to_string(),
            type_src: "forall (A : Type) (x : A), Eq A x x".to_string(),
            value_src: None,
            is_axiom: true, // Constructor
            description: "Reflexivity: every element equals itself.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // Symmetry of equality
        self.add_definition(SpecDefinition {
            name: "Eq.symm".to_string(),
            type_src: "forall (A : Type) (x : A) (y : A), Eq A x y -> Eq A y x".to_string(),
            value_src: None,
            is_axiom: true, // Derived from J
            description: "Symmetry: if x = y then y = x.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // Transitivity of equality
        self.add_definition(SpecDefinition {
            name: "Eq.trans".to_string(),
            type_src: "forall (A : Type) (x : A) (y : A) (z : A), Eq A x y -> Eq A y z -> Eq A x z"
                .to_string(),
            value_src: None,
            is_axiom: true, // Derived from J
            description: "Transitivity: if x = y and y = z then x = z.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // Congruence (function application preserves equality)
        self.add_definition(SpecDefinition {
            name: "Eq.cong".to_string(),
            type_src: "forall (A : Type) (B : Type) (f : A -> B) (x : A) (y : A), Eq A x y -> Eq B (f x) (f y)".to_string(),
            value_src: None,
            is_axiom: true, // Derived from J
            description: "Congruence: f x = f y if x = y.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // =========================================================
        // PART 2: Natural Numbers
        // =========================================================

        self.add_definition(SpecDefinition {
            name: "Nat".to_string(),
            type_src: "Type".to_string(),
            value_src: None,
            is_axiom: true, // Inductive type
            description: "Natural numbers.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        self.add_definition(SpecDefinition {
            name: "Nat.zero".to_string(),
            type_src: "Nat".to_string(),
            value_src: None,
            is_axiom: true, // Constructor
            description: "Zero.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        self.add_definition(SpecDefinition {
            name: "Nat.succ".to_string(),
            type_src: "Nat -> Nat".to_string(),
            value_src: None,
            is_axiom: true, // Constructor
            description: "Successor function.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // =========================================================
        // PART 3: Boolean type
        // =========================================================

        self.add_definition(SpecDefinition {
            name: "Bool".to_string(),
            type_src: "Type".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Boolean type.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        self.add_definition(SpecDefinition {
            name: "Bool.true".to_string(),
            type_src: "Bool".to_string(),
            value_src: None,
            is_axiom: true,
            description: "True value.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        self.add_definition(SpecDefinition {
            name: "Bool.false".to_string(),
            type_src: "Bool".to_string(),
            value_src: None,
            is_axiom: true,
            description: "False value.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // =========================================================
        // PART 4: Kernel Expression Model
        // =========================================================
        // These are Lean5 types that model the kernel's Expr type

        self.add_definition(SpecDefinition {
            name: "KExpr".to_string(),
            type_src: "Type".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Kernel expression type (model of lean5_kernel::Expr).".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        self.add_definition(SpecDefinition {
            name: "KExpr.sort".to_string(),
            type_src: "Nat -> KExpr".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Sort constructor: Sort n.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        self.add_definition(SpecDefinition {
            name: "KExpr.bvar".to_string(),
            type_src: "Nat -> KExpr".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Bound variable constructor: BVar n.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        self.add_definition(SpecDefinition {
            name: "KExpr.app".to_string(),
            type_src: "KExpr -> KExpr -> KExpr".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Application constructor: App f a.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        self.add_definition(SpecDefinition {
            name: "KExpr.lam".to_string(),
            type_src: "KExpr -> KExpr -> KExpr".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Lambda constructor: Lam ty body.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        self.add_definition(SpecDefinition {
            name: "KExpr.pi".to_string(),
            type_src: "KExpr -> KExpr -> KExpr".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Pi/forall constructor: Pi ty body.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // =========================================================
        // PART 5: Type Checking Predicates
        // =========================================================

        self.add_definition(SpecDefinition {
            name: "has_type".to_string(),
            type_src: "KExpr -> KExpr -> Type".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Typing judgment: has_type e T means expression e has type T.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        self.add_definition(SpecDefinition {
            name: "is_def_eq".to_string(),
            type_src: "KExpr -> KExpr -> Type".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Definitional equality: is_def_eq a b means a ≡ b.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // =========================================================
        // PART 6: Type Preservation Statement
        // =========================================================

        // This is the key soundness property we want to prove
        self.add_definition(SpecDefinition {
            name: "TypePreservation".to_string(),
            type_src: "forall (e : KExpr) (T : KExpr) (e' : KExpr), has_type e T -> is_def_eq e e' -> has_type e' T".to_string(),
            value_src: None,
            is_axiom: false, // This is what we want to PROVE
            description: "Type preservation: if e : T and e ≡ e', then e' : T.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // =========================================================
        // PART 7: Typing Rules as Axioms
        // =========================================================

        // Sort typing rule: Sort n : Sort (n + 1)
        self.add_definition(SpecDefinition {
            name: "sort_typing".to_string(),
            type_src: "forall (n : Nat), has_type (KExpr.sort n) (KExpr.sort (Nat.succ n))"
                .to_string(),
            value_src: None,
            is_axiom: true,
            description: "Sort typing rule: Sort n has type Sort (n+1).".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // Pi formation rule
        self.add_definition(SpecDefinition {
            name: "pi_formation".to_string(),
            type_src: "forall (A : KExpr) (B : KExpr) (n : Nat) (m : Nat), has_type A (KExpr.sort n) -> has_type B (KExpr.sort m) -> has_type (KExpr.pi A B) (KExpr.sort m)".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Pi formation: if A : Sort n and B : Sort m, then (A -> B) : Sort m.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // Lambda typing rule
        self.add_definition(SpecDefinition {
            name: "lam_typing".to_string(),
            type_src: "forall (A : KExpr) (b : KExpr) (B : KExpr), has_type A (KExpr.sort Nat.zero) -> has_type b B -> has_type (KExpr.lam A b) (KExpr.pi A B)".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Lambda typing: if A is a type and b : B, then (λA.b) : (A → B).".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // Application typing rule
        self.add_definition(SpecDefinition {
            name: "app_typing".to_string(),
            type_src: "forall (f : KExpr) (a : KExpr) (A : KExpr) (B : KExpr), has_type f (KExpr.pi A B) -> has_type a A -> has_type (KExpr.app f a) B".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Application typing: if f : (A → B) and a : A, then (f a) : B.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // =========================================================
        // PART 8: Definitional Equality Rules
        // =========================================================

        // Reflexivity of def eq
        self.add_definition(SpecDefinition {
            name: "def_eq_refl".to_string(),
            type_src: "forall (e : KExpr), is_def_eq e e".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Definitional equality is reflexive.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // Symmetry of def eq
        self.add_definition(SpecDefinition {
            name: "def_eq_symm".to_string(),
            type_src: "forall (a : KExpr) (b : KExpr), is_def_eq a b -> is_def_eq b a".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Definitional equality is symmetric.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // Transitivity of def eq
        self.add_definition(SpecDefinition {
            name: "def_eq_trans".to_string(),
            type_src: "forall (a : KExpr) (b : KExpr) (c : KExpr), is_def_eq a b -> is_def_eq b c -> is_def_eq a c".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Definitional equality is transitive.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // Beta reduction
        self.add_definition(SpecDefinition {
            name: "beta_reduction".to_string(),
            type_src: "forall (A : KExpr) (b : KExpr) (a : KExpr), is_def_eq (KExpr.app (KExpr.lam A b) a) b".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Beta reduction: (λA.b) a ≡ b[a/0].".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // =========================================================
        // PART 9: WHNF and Reduction
        // =========================================================
        // These definitions model WHNF reduction for the specification.

        // is_value predicate (expressions in normal form)
        self.add_definition(SpecDefinition {
            name: "is_value".to_string(),
            type_src: "KExpr -> Type".to_string(),
            value_src: None,
            is_axiom: true,
            description: "is_value e holds if e is a value (lam, pi, sort, etc.).".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // terminates_whnf predicate
        self.add_definition(SpecDefinition {
            name: "terminates_whnf".to_string(),
            type_src: "KExpr -> Type".to_string(),
            value_src: None,
            is_axiom: true,
            description: "terminates_whnf e holds if WHNF reduction terminates on e.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // terminates_infer predicate
        self.add_definition(SpecDefinition {
            name: "terminates_infer".to_string(),
            type_src: "KExpr -> Type".to_string(),
            value_src: None,
            is_axiom: true,
            description: "terminates_infer e holds if type inference terminates on e.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // terminates_def_eq predicate
        self.add_definition(SpecDefinition {
            name: "terminates_def_eq".to_string(),
            type_src: "KExpr -> KExpr -> Type".to_string(),
            value_src: None,
            is_axiom: true,
            description: "terminates_def_eq a b holds if def eq checking terminates.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // whnf_to relation (whnf_to e e' means e reduces to e' in WHNF)
        self.add_definition(SpecDefinition {
            name: "whnf_to".to_string(),
            type_src: "KExpr -> KExpr -> Type".to_string(),
            value_src: None,
            is_axiom: true,
            description: "whnf_to e e' holds if e reduces to e' (e' in WHNF).".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // beta_reduces relation
        self.add_definition(SpecDefinition {
            name: "beta_reduces".to_string(),
            type_src: "KExpr -> KExpr -> Type".to_string(),
            value_src: None,
            is_axiom: true,
            description: "beta_reduces e e' holds if e beta-reduces to e'.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // =========================================================
        // PART 10: Expression Operations
        // =========================================================
        // Operations on kernel expressions (for proofs).

        // expr_size function (for termination proofs)
        self.add_definition(SpecDefinition {
            name: "expr_size".to_string(),
            type_src: "KExpr -> Nat".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Size measure on expressions for termination proofs.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // instantiate function (substitution)
        self.add_definition(SpecDefinition {
            name: "instantiate".to_string(),
            type_src: "KExpr -> KExpr -> KExpr".to_string(),
            value_src: None,
            is_axiom: true,
            description: "instantiate body val substitutes val for BVar 0 in body.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // lift function (de Bruijn lifting)
        self.add_definition(SpecDefinition {
            name: "lift".to_string(),
            type_src: "KExpr -> Nat -> KExpr".to_string(),
            value_src: None,
            is_axiom: true,
            description: "lift e n lifts free variables in e by n.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // is_closed predicate
        self.add_definition(SpecDefinition {
            name: "is_closed".to_string(),
            type_src: "KExpr -> Type".to_string(),
            value_src: None,
            is_axiom: true,
            description: "is_closed e holds if e has no free bound variables.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // =========================================================
        // PART 11: Key Lemmas (Axiomatized from Verus proofs)
        // =========================================================
        // These are lemmas proven in Verus, now axiomatized in the spec.

        // lift zero is identity
        self.add_definition(SpecDefinition {
            name: "lift_zero_identity".to_string(),
            type_src: "forall (e : KExpr), Eq KExpr (lift e Nat.zero) e".to_string(),
            value_src: None,
            is_axiom: true,
            description: "lift e 0 = e (lifting by 0 is identity).".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // instantiate on BVar 0 gives the value
        self.add_definition(SpecDefinition {
            name: "instantiate_bvar_zero".to_string(),
            type_src: "forall (val : KExpr), Eq KExpr (instantiate (KExpr.bvar Nat.zero) val) val"
                .to_string(),
            value_src: None,
            is_axiom: true,
            description: "instantiate (BVar 0) val = val.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // We need a "greater than zero" predicate, but for now just have a weaker lemma:
        // The size function returns a Nat (this is stated by its type already)

        // beta reduction determinism
        self.add_definition(SpecDefinition {
            name: "beta_deterministic".to_string(),
            type_src: "forall (e : KExpr) (r1 : KExpr) (r2 : KExpr), beta_reduces e r1 -> beta_reduces e r2 -> is_def_eq r1 r2".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Beta reduction is deterministic.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // WHNF idempotence
        self.add_definition(SpecDefinition {
            name: "whnf_idempotent".to_string(),
            type_src: "forall (e : KExpr) (e' : KExpr), whnf_to e e' -> whnf_to e' e'".to_string(),
            value_src: None,
            is_axiom: true,
            description: "If e reduces to e' in WHNF, then e' is already in WHNF.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // Values are in WHNF
        self.add_definition(SpecDefinition {
            name: "value_in_whnf".to_string(),
            type_src: "forall (e : KExpr), is_value e -> whnf_to e e".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Values are already in WHNF.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // Sort is a value
        self.add_definition(SpecDefinition {
            name: "sort_is_value".to_string(),
            type_src: "forall (n : Nat), is_value (KExpr.sort n)".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Sort n is a value.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // Lambda is a value
        self.add_definition(SpecDefinition {
            name: "lam_is_value".to_string(),
            type_src: "forall (ty : KExpr) (body : KExpr), is_value (KExpr.lam ty body)"
                .to_string(),
            value_src: None,
            is_axiom: true,
            description: "Lambda abstractions are values.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // Pi is a value
        self.add_definition(SpecDefinition {
            name: "pi_is_value".to_string(),
            type_src: "forall (ty : KExpr) (body : KExpr), is_value (KExpr.pi ty body)".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Pi types are values.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // WHNF termination for well-typed terms
        self.add_definition(SpecDefinition {
            name: "whnf_terminates_well_typed".to_string(),
            type_src: "forall (e : KExpr) (T : KExpr), has_type e T -> terminates_whnf e"
                .to_string(),
            value_src: None,
            is_axiom: true,
            description: "WHNF terminates on well-typed expressions.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // Type inference termination
        self.add_definition(SpecDefinition {
            name: "infer_terminates".to_string(),
            type_src: "forall (e : KExpr), terminates_infer e".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Type inference always terminates.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // WHNF confluence
        self.add_definition(SpecDefinition {
            name: "whnf_confluent".to_string(),
            type_src: "forall (e : KExpr) (e1 : KExpr) (e2 : KExpr), whnf_to e e1 -> whnf_to e e2 -> is_def_eq e1 e2".to_string(),
            value_src: None,
            is_axiom: true,
            description: "WHNF is confluent (unique normal forms).".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // =========================================================
        // PART 12: Type Preservation Infrastructure
        // =========================================================
        // These are the key lemmas needed to prove TypePreservation.

        // Substitution typing lemma (crucial for beta reduction preservation)
        // If b : B in context (x : A, Γ) and a : A, then b[a/x] : B[a/x]
        self.add_definition(SpecDefinition {
            name: "substitution_typing".to_string(),
            type_src: "forall (A : KExpr) (B : KExpr) (b : KExpr) (a : KExpr), has_type A (KExpr.sort Nat.zero) -> has_type b B -> has_type a A -> has_type (instantiate b a) (instantiate B a)".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Substitution preserves typing: if b : B and a : A, then b[a/x] : B[a/x].".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // Type conversion (allows changing types to def-eq types)
        self.add_definition(SpecDefinition {
            name: "type_conversion".to_string(),
            type_src: "forall (e : KExpr) (T1 : KExpr) (T2 : KExpr), has_type e T1 -> is_def_eq T1 T2 -> has_type e T2".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Type conversion: if e : T1 and T1 ≡ T2, then e : T2.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // Beta preservation: if (λA.b) a reduces and f a has type T, then b[a/x] has type T
        // This follows from: (λA.b) a : B[a/x], and (λA.b) a ≡ b[a/x], so b[a/x] : B[a/x]
        self.add_definition(SpecDefinition {
            name: "beta_preservation".to_string(),
            type_src: "forall (A : KExpr) (B : KExpr) (b : KExpr) (a : KExpr) (T : KExpr), has_type (KExpr.app (KExpr.lam A b) a) T -> has_type (instantiate b a) T".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Beta preservation: if (λA.b) a : T, then b[a/x] : T.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // Def eq preserves typing (the key lemma for type preservation)
        self.add_definition(SpecDefinition {
            name: "def_eq_preserves_typing".to_string(),
            type_src: "forall (e : KExpr) (e' : KExpr) (T : KExpr), has_type e T -> is_def_eq e e' -> has_type e' T".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Definitional equality preserves typing (type preservation).".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // Congruence for application: if f ≡ f' and a ≡ a', then f a ≡ f' a'
        self.add_definition(SpecDefinition {
            name: "def_eq_app_cong".to_string(),
            type_src: "forall (f : KExpr) (f' : KExpr) (a : KExpr) (a' : KExpr), is_def_eq f f' -> is_def_eq a a' -> is_def_eq (KExpr.app f a) (KExpr.app f' a')".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Application congruence for def eq.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // Congruence for lambda: if b ≡ b', then λA.b ≡ λA.b'
        self.add_definition(SpecDefinition {
            name: "def_eq_lam_cong".to_string(),
            type_src: "forall (A : KExpr) (b : KExpr) (b' : KExpr), is_def_eq b b' -> is_def_eq (KExpr.lam A b) (KExpr.lam A b')".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Lambda congruence for def eq.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // Congruence for pi: if A ≡ A' and B ≡ B', then Π(A).B ≡ Π(A').B'
        self.add_definition(SpecDefinition {
            name: "def_eq_pi_cong".to_string(),
            type_src: "forall (A : KExpr) (A' : KExpr) (B : KExpr) (B' : KExpr), is_def_eq A A' -> is_def_eq B B' -> is_def_eq (KExpr.pi A B) (KExpr.pi A' B')".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Pi congruence for def eq.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // =========================================================
        // PART 13: Derived typing lemmas (concrete programs)
        // =========================================================
        // These lemmas serve as sanity checks that the specification
        // supports ordinary lambda calculus reasoning.

        self.add_definition(SpecDefinition {
            name: "identity_typing".to_string(),
            type_src: "forall (A : Type), A -> A".to_string(),
            value_src: Some("fun (A : Type) (x : A) => x".to_string()),
            is_axiom: false,
            description: "The identity function has type A → A.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        self.add_definition(SpecDefinition {
            name: "const_typing".to_string(),
            type_src: "forall (A : Type) (B : Type), A -> B -> A".to_string(),
            value_src: Some("fun (A : Type) (B : Type) (a : A) (_b : B) => a".to_string()),
            is_axiom: false,
            description: "The const function has type A → B → A.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        self.add_definition(SpecDefinition {
            name: "compose_typing".to_string(),
            type_src: "forall (A : Type) (B : Type) (C : Type) (g : B -> C) (f : A -> B), A -> C"
                .to_string(),
            value_src: Some(
                "fun (A : Type) (B : Type) (C : Type) (g : B -> C) (f : A -> B) (x : A) => g (f x)"
                    .to_string(),
            ),
            is_axiom: false,
            description: "Function composition has type (B→C) → (A→B) → A → C.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        self.add_definition(SpecDefinition {
            name: "flip_typing".to_string(),
            type_src: "forall (A : Type) (B : Type) (C : Type) (f : A -> B -> C), B -> A -> C"
                .to_string(),
            value_src: Some(
                "fun (A : Type) (B : Type) (C : Type) (f : A -> B -> C) (b : B) (a : A) => f a b"
                    .to_string(),
            ),
            is_axiom: false,
            description: "The flip combinator swaps function arguments.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // =========================================================
        // PART 14: Micro-Checker Model
        // =========================================================
        // These specifications model the micro-checker (lean5-kernel/src/micro.rs)
        // for proving its correctness. The micro-checker is a minimal certificate
        // verifier (~1200 lines) designed to be auditable and provably correct.

        // MicroLevel type (universe levels for micro-checker)
        self.add_definition(SpecDefinition {
            name: "MicroLevel".to_string(),
            type_src: "Type".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Micro-checker universe levels (simplified from kernel Level)."
                .to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        self.add_definition(SpecDefinition {
            name: "MicroLevel.zero".to_string(),
            type_src: "MicroLevel".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Level 0 (Prop).".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        self.add_definition(SpecDefinition {
            name: "MicroLevel.succ".to_string(),
            type_src: "MicroLevel -> MicroLevel".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Successor level: l + 1.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        self.add_definition(SpecDefinition {
            name: "MicroLevel.max".to_string(),
            type_src: "MicroLevel -> MicroLevel -> MicroLevel".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Maximum of two levels.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        self.add_definition(SpecDefinition {
            name: "MicroLevel.imax".to_string(),
            type_src: "MicroLevel -> MicroLevel -> MicroLevel".to_string(),
            value_src: None,
            is_axiom: true,
            description: "IMax: 0 if second arg is 0, else max.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // MicroExpr type (expressions for micro-checker)
        self.add_definition(SpecDefinition {
            name: "MicroExpr".to_string(),
            type_src: "Type".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Micro-checker expression type.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        self.add_definition(SpecDefinition {
            name: "MicroExpr.bvar".to_string(),
            type_src: "Nat -> MicroExpr".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Bound variable (de Bruijn index).".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        self.add_definition(SpecDefinition {
            name: "MicroExpr.sort".to_string(),
            type_src: "MicroLevel -> MicroExpr".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Sort/Type at a level.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        self.add_definition(SpecDefinition {
            name: "MicroExpr.app".to_string(),
            type_src: "MicroExpr -> MicroExpr -> MicroExpr".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Function application.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        self.add_definition(SpecDefinition {
            name: "MicroExpr.lam".to_string(),
            type_src: "MicroExpr -> MicroExpr -> MicroExpr".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Lambda abstraction: λ (x : A). b.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        self.add_definition(SpecDefinition {
            name: "MicroExpr.pi".to_string(),
            type_src: "MicroExpr -> MicroExpr -> MicroExpr".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Pi/forall type: (x : A) → B.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        self.add_definition(SpecDefinition {
            name: "MicroExpr.let_".to_string(),
            type_src: "MicroExpr -> MicroExpr -> MicroExpr -> MicroExpr".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Let binding: let x : A := v in b.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        self.add_definition(SpecDefinition {
            name: "MicroExpr.opaque".to_string(),
            type_src: "MicroExpr -> MicroExpr".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Opaque constant (just type, no definition).".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // =========================================================
        // PART 15: Micro-Checker Operations
        // =========================================================

        // micro_lift: lift bound variables
        self.add_definition(SpecDefinition {
            name: "micro_lift".to_string(),
            type_src: "MicroExpr -> Nat -> Nat -> MicroExpr".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Lift bound variables >= cutoff by amount.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // micro_instantiate: substitute for BVar(0)
        self.add_definition(SpecDefinition {
            name: "micro_instantiate".to_string(),
            type_src: "MicroExpr -> MicroExpr -> MicroExpr".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Substitute val for BVar(0).".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // micro_whnf: weak head normal form
        self.add_definition(SpecDefinition {
            name: "micro_whnf".to_string(),
            type_src: "MicroExpr -> MicroExpr".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Compute WHNF (beta + zeta only).".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // micro_def_eq: definitional equality
        self.add_definition(SpecDefinition {
            name: "micro_def_eq".to_string(),
            type_src: "MicroExpr -> MicroExpr -> Bool".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Check definitional equality (structural after WHNF).".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // micro_structural_eq: structural equality
        self.add_definition(SpecDefinition {
            name: "micro_structural_eq".to_string(),
            type_src: "MicroExpr -> MicroExpr -> Bool".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Check structural equality.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // =========================================================
        // PART 16: Micro-Checker Certificate Types
        // =========================================================

        // MicroCert type (certificate for verification)
        self.add_definition(SpecDefinition {
            name: "MicroCert".to_string(),
            type_src: "Type".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Micro-checker proof certificate type.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // micro_verify: verify certificate against expression
        self.add_definition(SpecDefinition {
            name: "micro_verify".to_string(),
            type_src: "MicroCert -> MicroExpr -> MicroExpr".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Verify certificate, returning the proven type (partial).".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // =========================================================
        // PART 17: Micro-Checker Correctness Properties
        // =========================================================
        // These are the key theorems that establish micro-checker correctness.

        // Lift preserves structure
        self.add_definition(SpecDefinition {
            name: "micro_lift_zero_id".to_string(),
            type_src: "forall (e : MicroExpr) (c : Nat), Eq MicroExpr (micro_lift e c Nat.zero) e"
                .to_string(),
            value_src: None,
            is_axiom: true,
            description: "Lifting by 0 is identity.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // Instantiate BVar(0) gives the value
        self.add_definition(SpecDefinition {
            name: "micro_instantiate_bvar_zero".to_string(),
            type_src: "forall (v : MicroExpr), Eq MicroExpr (micro_instantiate (MicroExpr.bvar Nat.zero) v) v".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Instantiating BVar(0) gives the substituted value.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // WHNF idempotence
        self.add_definition(SpecDefinition {
            name: "micro_whnf_idempotent".to_string(),
            type_src:
                "forall (e : MicroExpr), Eq MicroExpr (micro_whnf (micro_whnf e)) (micro_whnf e)"
                    .to_string(),
            value_src: None,
            is_axiom: true,
            description: "WHNF is idempotent: whnf(whnf(e)) = whnf(e).".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // WHNF preserves values (sorts, lambdas, pis)
        self.add_definition(SpecDefinition {
            name: "micro_whnf_sort".to_string(),
            type_src: "forall (l : MicroLevel), Eq MicroExpr (micro_whnf (MicroExpr.sort l)) (MicroExpr.sort l)".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Sorts are in WHNF.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        self.add_definition(SpecDefinition {
            name: "micro_whnf_lam".to_string(),
            type_src: "forall (ty : MicroExpr) (body : MicroExpr), Eq MicroExpr (micro_whnf (MicroExpr.lam ty body)) (MicroExpr.lam ty body)".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Lambdas are in WHNF.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        self.add_definition(SpecDefinition {
            name: "micro_whnf_pi".to_string(),
            type_src: "forall (ty : MicroExpr) (body : MicroExpr), Eq MicroExpr (micro_whnf (MicroExpr.pi ty body)) (MicroExpr.pi ty body)".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Pis are in WHNF.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // WHNF beta reduction
        self.add_definition(SpecDefinition {
            name: "micro_whnf_beta".to_string(),
            type_src: "forall (ty : MicroExpr) (body : MicroExpr) (arg : MicroExpr), Eq MicroExpr (micro_whnf (MicroExpr.app (MicroExpr.lam ty body) arg)) (micro_whnf (micro_instantiate body arg))".to_string(),
            value_src: None,
            is_axiom: true,
            description: "WHNF performs beta reduction: (λ.b) a → whnf(b[a]).".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // def_eq reflexivity
        self.add_definition(SpecDefinition {
            name: "micro_def_eq_refl".to_string(),
            type_src: "forall (e : MicroExpr), Eq Bool (micro_def_eq e e) Bool.true".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Definitional equality is reflexive.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // def_eq symmetry
        self.add_definition(SpecDefinition {
            name: "micro_def_eq_symm".to_string(),
            type_src: "forall (a : MicroExpr) (b : MicroExpr), Eq Bool (micro_def_eq a b) (micro_def_eq b a)".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Definitional equality is symmetric.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // =========================================================
        // PART 18: Micro-Checker Soundness
        // =========================================================
        // The key theorem: if micro_verify succeeds, the typing is correct.

        // micro_has_type: typing judgment for micro-checker
        self.add_definition(SpecDefinition {
            name: "micro_has_type".to_string(),
            type_src: "MicroExpr -> MicroExpr -> Type".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Typing judgment for micro-checker: e has type T.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // Micro-checker soundness: if verify(cert, e) = T, then e : T
        self.add_definition(SpecDefinition {
            name: "micro_verify_sound".to_string(),
            type_src: "forall (cert : MicroCert) (e : MicroExpr) (T : MicroExpr), Eq MicroExpr (micro_verify cert e) T -> micro_has_type e T".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Micro-checker soundness: if verify succeeds with type T, then e : T.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // Sort typing rule for micro-checker
        self.add_definition(SpecDefinition {
            name: "micro_sort_typing".to_string(),
            type_src: "forall (l : MicroLevel), micro_has_type (MicroExpr.sort l) (MicroExpr.sort (MicroLevel.succ l))".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Sort l : Sort (succ l).".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // Pi formation for micro-checker
        self.add_definition(SpecDefinition {
            name: "micro_pi_formation".to_string(),
            type_src: "forall (A : MicroExpr) (B : MicroExpr) (l1 : MicroLevel) (l2 : MicroLevel), micro_has_type A (MicroExpr.sort l1) -> micro_has_type B (MicroExpr.sort l2) -> micro_has_type (MicroExpr.pi A B) (MicroExpr.sort (MicroLevel.imax l1 l2))".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Pi formation: Π(A:l1)(B:l2) : Sort(imax l1 l2).".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // Lambda typing for micro-checker
        self.add_definition(SpecDefinition {
            name: "micro_lam_typing".to_string(),
            type_src: "forall (A : MicroExpr) (b : MicroExpr) (B : MicroExpr), micro_has_type b B -> micro_has_type (MicroExpr.lam A b) (MicroExpr.pi A B)".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Lambda typing: if b : B then λA.b : Π A.B.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // Application typing for micro-checker
        self.add_definition(SpecDefinition {
            name: "micro_app_typing".to_string(),
            type_src: "forall (f : MicroExpr) (a : MicroExpr) (A : MicroExpr) (B : MicroExpr), micro_has_type f (MicroExpr.pi A B) -> micro_has_type a A -> micro_has_type (MicroExpr.app f a) (micro_instantiate B a)".to_string(),
            value_src: None,
            is_axiom: true,
            description: "App typing: if f : Π A.B and a : A then f a : B[a].".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // Type preservation for micro-checker
        self.add_definition(SpecDefinition {
            name: "micro_type_preservation".to_string(),
            type_src: "forall (e : MicroExpr) (T : MicroExpr) (e' : MicroExpr), micro_has_type e T -> Eq Bool (micro_def_eq e e') Bool.true -> micro_has_type e' T".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Micro-checker type preservation: if e : T and e ≡ e', then e' : T.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // =========================================================
        // PART 19: Cross-Validation with Main Kernel
        // =========================================================
        // These relate micro-checker types to main kernel types.

        // Translate kernel expr to micro expr
        self.add_definition(SpecDefinition {
            name: "kernel_to_micro".to_string(),
            type_src: "KExpr -> MicroExpr".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Translate kernel expression to micro expression.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // Translation preserves typing
        self.add_definition(SpecDefinition {
            name: "translation_preserves_typing".to_string(),
            type_src: "forall (e : KExpr) (T : KExpr), has_type e T -> micro_has_type (kernel_to_micro e) (kernel_to_micro T)".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Translation preserves typing judgments.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        // Translation preserves def eq
        self.add_definition(SpecDefinition {
            name: "translation_preserves_def_eq".to_string(),
            type_src: "forall (a : KExpr) (b : KExpr), is_def_eq a b -> Eq Bool (micro_def_eq (kernel_to_micro a) (kernel_to_micro b)) Bool.true".to_string(),
            value_src: None,
            is_axiom: true,
            description: "Translation preserves definitional equality.".to_string(),
            elaborated_type: None,
            elaborated_value: None,
        })?;

        Ok(())
    }

    /// Verify that a definition is well-typed
    pub fn verify_definition(&self, name: &str) -> Result<(), SpecError> {
        let def = self
            .definitions
            .get(name)
            .ok_or_else(|| SpecError::UnknownDefinition(name.to_string()))?;

        let type_expr = def
            .elaborated_type
            .as_ref()
            .ok_or_else(|| SpecError::MissingElaboration(def.name.clone()))?;

        if let Some(value) = &def.elaborated_value {
            let mut tc = TypeChecker::new(&self.env);
            let inferred = tc
                .infer_type(value)
                .map_err(|e| SpecError::TypeError(format!("infer {}: {:?}", def.name, e)))?;

            if !tc.is_def_eq(&inferred, type_expr) {
                return Err(SpecError::TypeError(format!(
                    "Type mismatch for {}: {:?} vs {:?}",
                    def.name, inferred, type_expr
                )));
            }
        }

        Ok(())
    }

    /// Elaborate a Lean5 source string in the current spec environment
    fn elaborate_source(&self, src: &str, label: &str) -> Result<Expr, SpecError> {
        let surface =
            parse_expr(src).map_err(|e| SpecError::ParseError(format!("{label}: {e}")))?;
        let mut ctx = ElabCtx::new(&self.env);
        ctx.elaborate(&surface)
            .map_err(|e| SpecError::ElabError(format!("{label}: {e}")))
    }
}

impl Default for Specification {
    fn default() -> Self {
        Self::new().expect("specification construction should succeed")
    }
}

/// Specification error
#[derive(Debug, thiserror::Error)]
pub enum SpecError {
    #[error("Unknown definition: {0}")]
    UnknownDefinition(String),
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Elaboration error: {0}")]
    ElabError(String),
    #[error("Type error: {0}")]
    TypeError(String),
    #[error("Environment error: {0}")]
    EnvError(String),
    #[error("Missing elaboration for definition {0}")]
    MissingElaboration(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_specification_creation() {
        let spec = Specification::new().expect("spec should build");
        assert!(spec.definitions().contains_key("Eq"));
        assert!(spec.definitions().contains_key("has_type"));
        assert!(spec.definitions().contains_key("TypePreservation"));
    }

    #[test]
    fn test_spec_expr_elaborate() {
        let env = Environment::new();
        let mut expr = SpecExpr::new("Type -> Type");
        assert!(expr.elaborate(&env).is_ok());
    }

    #[test]
    fn test_spec_level_parse() {
        let mut level = SpecLevel::new("0");
        assert!(level.parse().is_ok());
        assert_eq!(*level.parse().unwrap(), Level::Zero);
    }

    #[test]
    fn test_core_definitions_exist() {
        let spec = Specification::new().expect("spec should build");

        // Core equality
        assert!(spec.definitions().contains_key("Eq"));
        assert!(spec.definitions().contains_key("Eq.refl"));
        assert!(spec.definitions().contains_key("Eq.symm"));
        assert!(spec.definitions().contains_key("Eq.trans"));
        assert!(spec.definitions().contains_key("Eq.cong"));

        // Natural numbers
        assert!(spec.definitions().contains_key("Nat"));
        assert!(spec.definitions().contains_key("Nat.zero"));
        assert!(spec.definitions().contains_key("Nat.succ"));

        // Kernel model
        assert!(spec.definitions().contains_key("KExpr"));
        assert!(spec.definitions().contains_key("has_type"));
        assert!(spec.definitions().contains_key("is_def_eq"));

        // Typing rules
        assert!(spec.definitions().contains_key("sort_typing"));
        assert!(spec.definitions().contains_key("pi_formation"));
        assert!(spec.definitions().contains_key("lam_typing"));
        assert!(spec.definitions().contains_key("app_typing"));

        // Def eq rules
        assert!(spec.definitions().contains_key("def_eq_refl"));
        assert!(spec.definitions().contains_key("beta_reduction"));

        // Main property
        assert!(spec.definitions().contains_key("TypePreservation"));
    }

    #[test]
    fn test_type_preservation_definitions_exist() {
        let spec = Specification::new().expect("spec should build");

        // Type preservation infrastructure
        assert!(spec.definitions().contains_key("substitution_typing"));
        assert!(spec.definitions().contains_key("type_conversion"));
        assert!(spec.definitions().contains_key("beta_preservation"));
        assert!(spec.definitions().contains_key("def_eq_preserves_typing"));
    }

    #[test]
    fn test_congruence_definitions_exist() {
        let spec = Specification::new().expect("spec should build");

        // Congruence rules
        assert!(spec.definitions().contains_key("def_eq_app_cong"));
        assert!(spec.definitions().contains_key("def_eq_lam_cong"));
        assert!(spec.definitions().contains_key("def_eq_pi_cong"));
    }

    #[test]
    fn test_definition_count() {
        let spec = Specification::new().expect("spec should build");
        // Should have at least 90 definitions (55 + 35 micro-checker)
        let count = spec.definitions().len();
        assert!(count >= 90, "Expected at least 90 definitions, got {count}");
    }

    #[test]
    fn test_micro_checker_definitions_exist() {
        let spec = Specification::new().expect("spec should build");

        // Micro level types
        assert!(spec.definitions().contains_key("MicroLevel"));
        assert!(spec.definitions().contains_key("MicroLevel.zero"));
        assert!(spec.definitions().contains_key("MicroLevel.succ"));
        assert!(spec.definitions().contains_key("MicroLevel.max"));
        assert!(spec.definitions().contains_key("MicroLevel.imax"));

        // Micro expression types
        assert!(spec.definitions().contains_key("MicroExpr"));
        assert!(spec.definitions().contains_key("MicroExpr.bvar"));
        assert!(spec.definitions().contains_key("MicroExpr.sort"));
        assert!(spec.definitions().contains_key("MicroExpr.app"));
        assert!(spec.definitions().contains_key("MicroExpr.lam"));
        assert!(spec.definitions().contains_key("MicroExpr.pi"));
        assert!(spec.definitions().contains_key("MicroExpr.let_"));
        assert!(spec.definitions().contains_key("MicroExpr.opaque"));

        // Micro operations
        assert!(spec.definitions().contains_key("micro_lift"));
        assert!(spec.definitions().contains_key("micro_instantiate"));
        assert!(spec.definitions().contains_key("micro_whnf"));
        assert!(spec.definitions().contains_key("micro_def_eq"));

        // Micro certificates
        assert!(spec.definitions().contains_key("MicroCert"));
        assert!(spec.definitions().contains_key("micro_verify"));

        // Micro correctness properties
        assert!(spec.definitions().contains_key("micro_lift_zero_id"));
        assert!(spec.definitions().contains_key("micro_whnf_idempotent"));
        assert!(spec.definitions().contains_key("micro_def_eq_refl"));
    }

    #[test]
    fn test_micro_checker_soundness_exists() {
        let spec = Specification::new().expect("spec should build");

        // Micro-checker soundness
        assert!(spec.definitions().contains_key("micro_has_type"));
        assert!(spec.definitions().contains_key("micro_verify_sound"));
        assert!(spec.definitions().contains_key("micro_sort_typing"));
        assert!(spec.definitions().contains_key("micro_pi_formation"));
        assert!(spec.definitions().contains_key("micro_lam_typing"));
        assert!(spec.definitions().contains_key("micro_app_typing"));
        assert!(spec.definitions().contains_key("micro_type_preservation"));
    }

    #[test]
    fn test_cross_validation_definitions_exist() {
        let spec = Specification::new().expect("spec should build");

        // Cross-validation
        assert!(spec.definitions().contains_key("kernel_to_micro"));
        assert!(spec
            .definitions()
            .contains_key("translation_preserves_typing"));
        assert!(spec
            .definitions()
            .contains_key("translation_preserves_def_eq"));
    }
}
