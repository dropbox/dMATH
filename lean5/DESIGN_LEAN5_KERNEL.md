# Lean5 Universal Kernel Specification

**Status:** Design
**Author:** Andrew Yates
**Date:** 2026-01-07
**Purpose:** Formal specification for Lean5's type-theoretic kernel supporting all mathematical proof traditions

---

## 1. Design Principles

### 1.1 Core Philosophy

**Math is math.** The Lean5 kernel supports ALL formal mathematics from ALL traditions:
- Constructive (Lean 4, Agda, Coq)
- Classical (HOL Light, HOL4, Isabelle/HOL)
- Impredicative (Coq's Prop)
- Cubical (Cubical Agda, cubicaltt)
- Set-theoretic (Mizar, Metamath/ZFC)

Lean5 is NOT bound by Lean 4 compatibility. It is a clean-slate implementation that extends the type theory to accommodate all systems.

### 1.2 Mode System

Different mathematical traditions have different logical foundations. Rather than pick one, Lean5 supports multiple **modes** with proven-safe combinations:

```rust
/// Logical mode controlling which axioms and features are available
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Lean5Mode {
    /// Pure Martin-Lof Type Theory - no axioms, decidable type checking
    /// Compatible with: All modes (most restrictive)
    Constructive,

    /// Calculus of Inductive Constructions - impredicative Prop, no large elimination
    /// Compatible with: Constructive, Classical
    Impredicative,

    /// Cubical Type Theory - Path types, hcomp, transp, Glue, univalence provable
    /// Compatible with: Constructive (NOT with Classical or Impredicative)
    Cubical,

    /// Classical logic - LEM, Choice as axioms
    /// Compatible with: Constructive, Impredicative
    Classical,

    /// ZFC set theory - sets as first-class, no dependent types required
    /// Compatible with: Classical
    SetTheoretic,
}

impl Lean5Mode {
    /// Check if proofs from `source` mode can be used in `target` mode
    pub fn can_import(source: Lean5Mode, target: Lean5Mode) -> bool {
        use Lean5Mode::*;
        match (source, target) {
            // Constructive proofs work everywhere
            (Constructive, _) => true,

            // Same mode always works
            (m1, m2) if m1 == m2 => true,

            // Impredicative works in Classical (both accept proof irrelevance)
            (Impredicative, Classical) => true,

            // Classical works in SetTheoretic (SetTheoretic extends Classical with ZFC axioms)
            (Classical, SetTheoretic) => true,
            (Impredicative, SetTheoretic) => true,

            // Cubical is isolated - different equality/computation rules (Path/Glue/hcomp); no cross-mode translation (yet)
            (Cubical, _) => false,
            (_, Cubical) => false,

            // SetTheoretic only imports from Classical hierarchy
            (SetTheoretic, _) => false,

            // Default: not compatible
            _ => false,
        }
    }

    /// Get the default mode for a source system
    pub fn from_source_system(system: SourceSystem) -> Self {
        use SourceSystem::*;
        match system {
            Lean4 => Lean5Mode::Constructive,
            Coq => Lean5Mode::Impredicative,
            Agda => Lean5Mode::Constructive,
            CubicalAgda => Lean5Mode::Cubical,
            IsabelleHOL | HOLLight | HOL4 => Lean5Mode::Classical,
            Mizar | MetamathZFC => Lean5Mode::SetTheoretic,
            MetamathSet | ACL2 => Lean5Mode::Classical,
            PVS => Lean5Mode::Classical,
        }
    }
}

/// Source proof system
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SourceSystem {
    Lean4,
    Coq,
    Agda,
    CubicalAgda,
    IsabelleHOL,
    HOLLight,
    HOL4,
    Mizar,
    MetamathZFC,
    MetamathSet,
    PVS,
    ACL2,
}
```

---

## 2. Universal Expression Type

### 2.1 Core Expressions

The Lean5 expression type extends Lean 4's calculus with additional constructors for other type theories:

```rust
/// Lean5 universal expression type
/// Supports all mathematical proof traditions through mode-gated extensions
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Expr {
    // ══════════════════════════════════════════════════════════════════
    // CORE (all modes)
    // ══════════════════════════════════════════════════════════════════

    /// Bound variable (de Bruijn index)
    BVar(u32),

    /// Free variable with unique ID
    FVar(FVarId),

    /// Sort (Type u, Prop, SProp)
    Sort(Level),

    /// Constant reference with universe level instantiation
    Const {
        name: Name,
        levels: Vec<Level>,
    },

    /// Function application
    App {
        func: Box<Expr>,
        arg: Box<Expr>,
    },

    /// Lambda abstraction
    Lam {
        binder_name: Name,
        binder_info: BinderInfo,
        domain: Box<Expr>,
        body: Box<Expr>,
    },

    /// Dependent function type (Pi/forall)
    Pi {
        binder_name: Name,
        binder_info: BinderInfo,
        domain: Box<Expr>,
        codomain: Box<Expr>,
    },

    /// Let binding
    Let {
        name: Name,
        ty: Box<Expr>,
        value: Box<Expr>,
        body: Box<Expr>,
    },

    /// Literal values
    Lit(Literal),

    /// Metavariable (for elaboration)
    MVar(MVarId),

    /// Projection from structure
    Proj {
        struct_name: Name,
        field_idx: u32,
        struct_expr: Box<Expr>,
    },

    // ══════════════════════════════════════════════════════════════════
    // IMPREDICATIVE MODE EXTENSIONS
    // ══════════════════════════════════════════════════════════════════

    /// Strict proposition (proof-irrelevant, no large elimination)
    /// Mode: Impredicative
    SProp,

    /// Squash type (truncation to SProp)
    /// Mode: Impredicative
    Squash(Box<Expr>),

    // ══════════════════════════════════════════════════════════════════
    // CUBICAL MODE EXTENSIONS
    // ══════════════════════════════════════════════════════════════════

    /// Interval type I with endpoints 0 and 1
    /// Mode: Cubical
    Interval,

    /// Interval endpoints
    /// Mode: Cubical
    I0,
    I1,

    /// Path type: Path A a b (heterogeneous equality)
    /// Mode: Cubical
    PathType {
        ty: Box<Expr>,      // A : I -> Type
        left: Box<Expr>,    // a : A 0
        right: Box<Expr>,   // b : A 1
    },

    /// Path lambda: <i> e
    /// Mode: Cubical
    PathLam {
        var: Name,
        body: Box<Expr>,
    },

    /// Path application: p @ i
    /// Mode: Cubical
    PathApp {
        path: Box<Expr>,
        arg: Box<Expr>,
    },

    /// Homogeneous composition
    /// hcomp {A} {φ} (u : I -> Partial φ A) (a : A) : A
    /// Mode: Cubical
    HComp {
        ty: Box<Expr>,
        phi: Box<Expr>,
        u: Box<Expr>,
        base: Box<Expr>,
    },

    /// Transport along a path
    /// transp (A : I -> Type) (φ : I) (a : A 0) : A 1
    /// Mode: Cubical
    Transp {
        ty: Box<Expr>,
        phi: Box<Expr>,
        base: Box<Expr>,
    },

    /// Glue type for univalence
    /// Mode: Cubical
    Glue {
        base: Box<Expr>,
        phi: Box<Expr>,
        types: Box<Expr>,
        equivs: Box<Expr>,
    },

    /// Glue term constructor
    /// Mode: Cubical
    GlueTerm {
        base: Box<Expr>,
        phi: Box<Expr>,
        term: Box<Expr>,
    },

    /// Unglue projection
    /// Mode: Cubical
    Unglue {
        ty: Box<Expr>,
        phi: Box<Expr>,
        term: Box<Expr>,
    },

    // ══════════════════════════════════════════════════════════════════
    // CLASSICAL MODE EXTENSIONS
    // ══════════════════════════════════════════════════════════════════

    /// Classical choice operator
    /// choice : (∃ x, P x) -> {x // P x}
    /// Mode: Classical
    Choice {
        ty: Box<Expr>,
        pred: Box<Expr>,
        exists_proof: Box<Expr>,
    },

    /// Hilbert epsilon (indefinite description)
    /// epsilon : (α -> Prop) -> α
    /// Mode: Classical
    Epsilon {
        ty: Box<Expr>,
        pred: Box<Expr>,
    },

    // ══════════════════════════════════════════════════════════════════
    // SET-THEORETIC MODE EXTENSIONS
    // ══════════════════════════════════════════════════════════════════

    /// ZFC set literal
    /// Mode: SetTheoretic
    ZFCSet(ZFCSetExpr),

    /// Set membership: a ∈ b
    /// Mode: SetTheoretic
    SetMem {
        element: Box<Expr>,
        set: Box<Expr>,
    },

    /// Set comprehension: {x ∈ A | P(x)}
    /// Mode: SetTheoretic
    SetComprehension {
        var: Name,
        domain: Box<Expr>,
        pred: Box<Expr>,
    },

    // ══════════════════════════════════════════════════════════════════
    // COINDUCTIVE EXTENSIONS (all modes)
    // ══════════════════════════════════════════════════════════════════

    /// Coinductive type (greatest fixpoint)
    CoInductive {
        name: Name,
        params: Vec<Expr>,
    },

    /// Corecursive definition with productivity guard
    CoRec {
        ty: Box<Expr>,
        body: Box<Expr>,
        guard: ProductivityGuard,
    },

    /// Bisimulation equality for coinductives
    Bisim {
        ty: Box<Expr>,
        left: Box<Expr>,
        right: Box<Expr>,
        relation: Box<Expr>,
    },
}

/// ZFC set expressions
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ZFCSetExpr {
    Empty,                           // ∅
    Singleton(Box<Expr>),            // {a}
    Pair(Box<Expr>, Box<Expr>),      // {a, b}
    Union(Box<Expr>),                // ⋃A
    PowerSet(Box<Expr>),             // P(A)
    Separation {                     // {x ∈ A | φ(x)}
        set: Box<Expr>,
        pred: Box<Expr>,
    },
    Replacement {                    // {F(x) | x ∈ A}
        set: Box<Expr>,
        func: Box<Expr>,
    },
    Infinity,                        // ω
    Choice(Box<Expr>),               // AC choice function
}

/// Productivity guard for corecursion
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ProductivityGuard {
    /// Guarded by constructor (standard guardedness)
    Guarded,
    /// Sized types with size annotation
    Sized(Box<Expr>),
    /// Clock-based (for Clocked Type Theory)
    Clocked(Name),
}
```

### 2.2 Universe Levels

```rust
/// Universe levels with extensions for all modes
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Level {
    /// Universe zero
    Zero,

    /// Successor universe
    Succ(Box<Level>),

    /// Universe maximum
    Max(Box<Level>, Box<Level>),

    /// Impredicative maximum (for Prop)
    IMax(Box<Level>, Box<Level>),

    /// Universe parameter (polymorphism)
    Param(Name),

    /// Universe metavariable
    MVar(LMVarId),

    /// Prop (impredicative, proof-irrelevant in some modes)
    Prop,

    /// SProp (strict propositions, always proof-irrelevant)
    /// Mode: Impredicative
    SProp,

    /// Set (predicative base universe in Coq-style)
    /// Mode: Impredicative
    Set,
}
```

---

## 3. Type Checking Rules

### 3.1 Mode-Aware Type Checker

```rust
/// Type checking context
pub struct TypeChecker {
    /// Current logical mode
    mode: Lean5Mode,

    /// Environment with definitions
    env: Environment,

    /// Local context
    lctx: LocalContext,

    /// Metavariable context
    mctx: MetavarContext,

    /// Universe constraint solver
    level_solver: LevelSolver,
}

impl TypeChecker {
    /// Check that an expression is well-formed in the current mode
    pub fn check(&mut self, expr: &Expr) -> Result<Expr, TypeError> {
        match expr {
            // Core expressions - all modes
            Expr::BVar(_) | Expr::FVar(_) | Expr::Sort(_) |
            Expr::Const { .. } | Expr::App { .. } | Expr::Lam { .. } |
            Expr::Pi { .. } | Expr::Let { .. } | Expr::Lit(_) |
            Expr::MVar(_) | Expr::Proj { .. } => self.check_core(expr),

            // Impredicative extensions
            Expr::SProp | Expr::Squash(_) => {
                self.require_mode(Lean5Mode::Impredicative)?;
                self.check_impredicative(expr)
            }

            // Cubical extensions
            Expr::Interval | Expr::I0 | Expr::I1 |
            Expr::PathType { .. } | Expr::PathLam { .. } | Expr::PathApp { .. } |
            Expr::HComp { .. } | Expr::Transp { .. } |
            Expr::Glue { .. } | Expr::GlueTerm { .. } | Expr::Unglue { .. } => {
                self.require_mode(Lean5Mode::Cubical)?;
                self.check_cubical(expr)
            }

            // Classical extensions
            Expr::Choice { .. } | Expr::Epsilon { .. } => {
                self.require_mode(Lean5Mode::Classical)?;
                self.check_classical(expr)
            }

            // Set-theoretic extensions
            Expr::ZFCSet(_) | Expr::SetMem { .. } | Expr::SetComprehension { .. } => {
                self.require_mode(Lean5Mode::SetTheoretic)?;
                self.check_set_theoretic(expr)
            }

            // Coinductives - all modes but with different guards
            Expr::CoInductive { .. } | Expr::CoRec { .. } | Expr::Bisim { .. } => {
                self.check_coinductive(expr)
            }
        }
    }

    /// Verify mode is active or compatible
    fn require_mode(&self, required: Lean5Mode) -> Result<(), TypeError> {
        if self.mode == required || Lean5Mode::can_import(required, self.mode) {
            Ok(())
        } else {
            Err(TypeError::ModeMismatch {
                required,
                current: self.mode,
            })
        }
    }
}
```

### 3.2 Cubical Type Checking

```rust
impl TypeChecker {
    /// Check cubical-specific expressions
    fn check_cubical(&mut self, expr: &Expr) -> Result<Expr, TypeError> {
        match expr {
            Expr::Interval => Ok(Expr::Sort(Level::Zero)),

            Expr::I0 | Expr::I1 => Ok(Expr::Interval),

            Expr::PathType { ty, left, right } => {
                // A : I -> Type u
                let ty_ty = self.infer(ty)?;
                self.check_is_type_family(&ty_ty, &Expr::Interval)?;

                // left : A 0
                let left_expected = self.apply(ty, &Expr::I0)?;
                self.check_type(left, &left_expected)?;

                // right : A 1
                let right_expected = self.apply(ty, &Expr::I1)?;
                self.check_type(right, &right_expected)?;

                // Path A left right : Type u
                let u = self.infer_type_level(&ty_ty)?;
                Ok(Expr::Sort(u))
            }

            Expr::PathLam { var, body } => {
                // Introduce interval variable
                self.lctx.push_local(var.clone(), Expr::Interval);
                let body_ty = self.infer(body)?;
                self.lctx.pop();

                // Infer path type
                let left = self.subst(body, var, &Expr::I0)?;
                let right = self.subst(body, var, &Expr::I1)?;
                let ty = Expr::Lam {
                    binder_name: var.clone(),
                    binder_info: BinderInfo::Default,
                    domain: Box::new(Expr::Interval),
                    body: Box::new(body_ty),
                };

                Ok(Expr::PathType {
                    ty: Box::new(ty),
                    left: Box::new(left),
                    right: Box::new(right),
                })
            }

            Expr::HComp { ty, phi, u, base } => {
                // ty : Type
                self.check_is_type(ty)?;

                // phi : I (cofibration)
                self.check_type(phi, &Expr::Interval)?;

                // u : I -> Partial phi ty
                // (partial element varying over interval)
                self.check_partial_path(u, phi, ty)?;

                // base : ty [phi -> u 0]
                self.check_type(base, ty)?;
                self.check_boundary(base, phi, u, &Expr::I0)?;

                Ok((**ty).clone())
            }

            Expr::Transp { ty, phi, base } => {
                // ty : I -> Type (line of types)
                self.check_type_path(ty)?;

                // phi : I (extent of constancy)
                self.check_type(phi, &Expr::Interval)?;

                // base : ty 0
                let ty_at_0 = self.apply(ty, &Expr::I0)?;
                self.check_type(base, &ty_at_0)?;

                // Result : ty 1
                let ty_at_1 = self.apply(ty, &Expr::I1)?;
                Ok(ty_at_1)
            }

            Expr::Glue { base, phi, types, equivs } => {
                // Glue type for univalence
                self.check_is_type(base)?;
                self.check_type(phi, &Expr::Interval)?;

                // types : Partial phi Type
                // equivs : (i : I) -> (h : IsOne phi) -> Equiv (types i h) base
                self.check_glue_data(base, phi, types, equivs)?;

                Ok(Expr::Sort(self.infer_type_level(base)?))
            }

            _ => Err(TypeError::InvalidCubical(expr.clone())),
        }
    }
}
```

### 3.3 Classical Type Checking

```rust
impl TypeChecker {
    fn check_classical(&mut self, expr: &Expr) -> Result<Expr, TypeError> {
        match expr {
            Expr::Choice { ty, pred, exists_proof } => {
                // ty : Type
                self.check_is_type(ty)?;

                // pred : ty -> Prop
                let pred_ty = Expr::Pi {
                    binder_name: Name::anonymous(),
                    binder_info: BinderInfo::Default,
                    domain: ty.clone(),
                    codomain: Box::new(Expr::Sort(Level::Prop)),
                };
                self.check_type(pred, &pred_ty)?;

                // exists_proof : ∃ x : ty, pred x
                let exists_ty = self.make_exists(ty, pred)?;
                self.check_type(exists_proof, &exists_ty)?;

                // Result : {x : ty // pred x} (subtype)
                Ok(self.make_subtype(ty, pred)?)
            }

            Expr::Epsilon { ty, pred } => {
                // Hilbert's epsilon - indefinite description
                // If ∃x, P(x) is true, returns some x satisfying P
                // If not, returns arbitrary element

                self.check_is_type(ty)?;

                let pred_ty = Expr::Pi {
                    binder_name: Name::anonymous(),
                    binder_info: BinderInfo::Default,
                    domain: ty.clone(),
                    codomain: Box::new(Expr::Sort(Level::Prop)),
                };
                self.check_type(pred, &pred_ty)?;

                // Result : ty
                Ok((**ty).clone())
            }

            _ => Err(TypeError::InvalidClassical(expr.clone())),
        }
    }
}
```

---

## 4. Kernel Architecture

### 4.1 Layered Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Lean5 Kernel                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 4: Import Translators                                            │
│  ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐         │
│  │  Lean4  │   Coq   │  Agda   │  HOL    │  Mizar  │Metamath │         │
│  └────┬────┴────┬────┴────┬────┴────┬────┴────┬────┴────┬────┘         │
│       │         │         │         │         │         │               │
├───────▼─────────▼─────────▼─────────▼─────────▼─────────▼───────────────┤
│  Layer 3: Mode System                                                   │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │  Constructive ◄──► Impredicative ◄──► Classical ──► SetTheory│      │
│  │        ▲                                                      │      │
│  │        │ (isolated)                                           │      │
│  │     Cubical                                                   │      │
│  └──────────────────────────────────────────────────────────────┘      │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 2: Universal Type Checker                                        │
│  ┌────────────┬────────────┬────────────┬────────────┐                 │
│  │   Typing   │ Reduction  │ Conversion │  Universes │                 │
│  └────────────┴────────────┴────────────┴────────────┘                 │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 1: Core Calculus                                                 │
│  ┌────────────┬────────────┬────────────┬────────────┐                 │
│  │    Expr    │   Level    │    Name    │   Decl     │                 │
│  └────────────┴────────────┴────────────┴────────────┘                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Environment

```rust
/// Global environment containing all definitions
pub struct Environment {
    /// All declarations indexed by name
    declarations: HashMap<Name, Declaration>,

    /// Mode annotations for each declaration
    modes: HashMap<Name, Lean5Mode>,

    /// Source system for imported declarations
    sources: HashMap<Name, SourceInfo>,

    /// Quotient type flag
    quot_enabled: bool,

    /// Proof certificates (for verified imports)
    certificates: HashMap<Name, ProofCertificate>,
}

/// Declaration in the environment
#[derive(Clone, Debug)]
pub enum Declaration {
    /// Axiom (no definition, trusted)
    Axiom {
        name: Name,
        levels: Vec<Name>,
        ty: Expr,
    },

    /// Definition (has value)
    Definition {
        name: Name,
        levels: Vec<Name>,
        ty: Expr,
        value: Expr,
        hints: ReducibilityHints,
    },

    /// Theorem (value is proof, can be erased)
    Theorem {
        name: Name,
        levels: Vec<Name>,
        ty: Expr,
        value: Expr,
    },

    /// Opaque (value hidden from reduction)
    Opaque {
        name: Name,
        levels: Vec<Name>,
        ty: Expr,
        value: Expr,
    },

    /// Inductive type
    Inductive {
        name: Name,
        levels: Vec<Name>,
        num_params: u32,
        ty: Expr,
        constructors: Vec<Constructor>,
        is_recursive: bool,
        is_nested: bool,
    },

    /// Coinductive type
    CoInductive {
        name: Name,
        levels: Vec<Name>,
        num_params: u32,
        ty: Expr,
        destructors: Vec<Destructor>,
        guard: ProductivityGuard,
    },

    /// Quotient type
    Quotient {
        name: Name,
        levels: Vec<Name>,
        ty: Expr,
        relation: Expr,
    },
}

/// Source information for imported declarations
#[derive(Clone, Debug)]
pub struct SourceInfo {
    /// Original system
    pub system: SourceSystem,

    /// Original name in source system
    pub original_name: String,

    /// Import timestamp
    pub imported_at: u64,

    /// Hash of original proof
    pub source_hash: [u8; 32],

    /// Translation method used
    pub translation: TranslationMethod,
}

/// How the declaration was translated
#[derive(Clone, Debug)]
pub enum TranslationMethod {
    /// Direct port (Lean4 -> Lean5)
    Direct,

    /// Algorithmic translation (Coq -> Lean5)
    Translated { translator_version: String },

    /// Deep embedding (HOL -> Lean5)
    Embedded { embedding_theory: Name },

    /// Proof reconstruction (Metamath -> Lean5)
    Reconstructed { checker_version: String },
}
```

---

## 5. Axiom Management

### 5.1 Built-in Axioms by Mode

```rust
/// Axioms available in each mode
impl Lean5Mode {
    pub fn available_axioms(&self) -> Vec<AxiomId> {
        use Lean5Mode::*;
        match self {
            Constructive => vec![
                // No logical axioms - pure MLTT
            ],

            Impredicative => vec![
                AxiomId::PropExt,        // Propositional extensionality
                AxiomId::ProofIrrel,     // Proof irrelevance for Prop
            ],

            Cubical => vec![
                // Univalence is PROVABLE, not an axiom
                // But we expose it as a theorem
            ],

            Classical => vec![
                AxiomId::PropExt,
                AxiomId::ProofIrrel,
                AxiomId::LEM,            // Law of excluded middle
                AxiomId::Choice,         // Axiom of choice
                AxiomId::FunExt,         // Function extensionality
            ],

            SetTheoretic => vec![
                // All classical axioms plus ZFC
                AxiomId::PropExt,
                AxiomId::ProofIrrel,
                AxiomId::LEM,
                AxiomId::Choice,
                AxiomId::FunExt,
                AxiomId::ZFCExtensionality,
                AxiomId::ZFCPairing,
                AxiomId::ZFCUnion,
                AxiomId::ZFCPowerSet,
                AxiomId::ZFCInfinity,
                AxiomId::ZFCSeparation,
                AxiomId::ZFCReplacement,
                AxiomId::ZFCFoundation,
            ],
        }
    }
}

/// Axiom identifiers
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AxiomId {
    // Logical axioms
    PropExt,
    ProofIrrel,
    LEM,
    Choice,
    FunExt,

    // ZFC axioms
    ZFCExtensionality,
    ZFCPairing,
    ZFCUnion,
    ZFCPowerSet,
    ZFCInfinity,
    ZFCSeparation,
    ZFCReplacement,
    ZFCFoundation,
    ZFCChoice,  // AC as set-theoretic axiom
}
```

### 5.2 Consistency Guards

```rust
/// Verify that axiom combinations are consistent
pub fn check_consistency(axioms: &[AxiomId]) -> Result<(), ConsistencyError> {
    // Known inconsistent combinations

    // Univalence + LEM is inconsistent
    // (But we handle this through mode separation - Cubical mode doesn't have LEM)

    // Large elimination from impredicative Prop + certain inductives is inconsistent
    // (We restrict large elimination in Impredicative mode)

    // Type-in-Type is inconsistent
    // (We use predicative universes, except for Prop which is impredicative in some modes)

    Ok(())
}

/// Restrictions on operations in each mode
impl Lean5Mode {
    pub fn allows_large_elimination(&self, from_sort: &Level) -> bool {
        match self {
            Lean5Mode::Constructive => true,  // Always allowed
            Lean5Mode::Impredicative => {
                // Only small elimination from Prop
                // Large elim only for singletons (Empty, Unit, Eq)
                !matches!(from_sort, Level::Prop)
            }
            Lean5Mode::Cubical => true,
            Lean5Mode::Classical => {
                // Same as impredicative
                !matches!(from_sort, Level::Prop)
            }
            Lean5Mode::SetTheoretic => true,  // Sets can eliminate
        }
    }
}
```

---

## 6. Import Pipelines

### 6.1 Lean 4 Import (Direct)

```rust
/// Direct import from Lean 4 .olean files
pub struct Lean4Importer {
    mode: Lean5Mode,
}

impl Lean4Importer {
    pub fn import(&self, olean: &[u8]) -> Result<Vec<Declaration>, ImportError> {
        // Parse .olean format
        let module = parse_olean(olean)?;

        // Translate expressions (1:1 mapping for core)
        let decls = module.declarations
            .into_iter()
            .map(|d| self.translate_decl(d))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(decls)
    }

    fn translate_decl(&self, decl: Lean4Decl) -> Result<Declaration, ImportError> {
        // Lean 4 maps directly to Lean5 Constructive mode
        // Quot becomes our native quotient
        // Everything else is 1:1
        todo!()
    }
}
```

### 6.2 Coq Import (Translation)

```rust
/// Translate Coq terms to Lean5
pub struct CoqImporter {
    mode: Lean5Mode,  // Usually Impredicative
}

impl CoqImporter {
    pub fn import(&self, vo: &[u8]) -> Result<Vec<Declaration>, ImportError> {
        // Parse .vo format
        let module = parse_coq_vo(vo)?;

        // Key translations:
        // - Coq's Prop -> Lean5 Prop (impredicative)
        // - Coq's Set -> Lean5 Type 0
        // - Coq's Type -> Lean5 Type (with universe polymorphism)
        // - Coq's SProp -> Lean5 SProp
        // - Coq's match -> Lean5 match (may need eta expansion)
        // - Coq's fix -> Lean5 rec (termination proof translation)
        // - Coq's cofix -> Lean5 corec

        todo!()
    }
}
```

### 6.3 HOL Import (Embedding)

```rust
/// Embed HOL proofs via shallow embedding
pub struct HOLImporter {
    mode: Lean5Mode,  // Classical
}

impl HOLImporter {
    pub fn import(&self, article: &[u8]) -> Result<Vec<Declaration>, ImportError> {
        // HOL has simple types, we embed into dependent types
        // HOL bool -> Lean5 Prop
        // HOL fun -> Lean5 Pi (non-dependent)
        // HOL = -> Lean5 Eq

        // Key insight: HOL proofs use LEM and Choice
        // These are axioms in Classical mode, so proofs transfer

        todo!()
    }
}
```

### 6.4 Metamath Import (Reconstruction)

```rust
/// Reconstruct proofs from Metamath proof steps
pub struct MetamathImporter {
    mode: Lean5Mode,  // SetTheoretic for ZFC, Classical for set.mm
}

impl MetamathImporter {
    pub fn import(&self, mm: &str) -> Result<Vec<Declaration>, ImportError> {
        // Metamath has explicit proof steps
        // We reconstruct the proof in Lean5:
        // 1. Parse the .mm file
        // 2. For each theorem, replay the proof steps
        // 3. Each step becomes a Lean5 tactic application
        // 4. Verify the resulting proof term

        todo!()
    }
}
```

---

## 7. Proof Certificates

### 7.1 Certificate Structure

```rust
/// Proof certificate for imported proofs
#[derive(Clone, Debug)]
pub struct ProofCertificate {
    /// Original source
    pub source: SourceInfo,

    /// Type of verification performed
    pub verification: VerificationType,

    /// Hash of the verified proof term
    pub proof_hash: [u8; 32],

    /// Timestamp of verification
    pub verified_at: u64,

    /// Verifier version
    pub verifier_version: String,
}

/// Types of verification
#[derive(Clone, Debug)]
pub enum VerificationType {
    /// Proof was type-checked in Lean5 kernel
    TypeChecked,

    /// Proof was verified in original system and translated
    TranslatedVerified {
        original_system: SourceSystem,
        translator_version: String,
    },

    /// Proof was cross-validated against another system
    CrossValidated {
        systems: Vec<SourceSystem>,
    },

    /// Proof was formally verified correct (bootstrap)
    FormallyVerified {
        verification_proof: Name,  // Points to the verification proof
    },
}
```

---

## 8. Performance Considerations

### 8.1 Fast Path

```rust
impl TypeChecker {
    /// Optimized checking for common cases
    pub fn check_fast(&mut self, expr: &Expr) -> Result<Expr, TypeError> {
        // Fast path for Constructive mode (most common)
        if self.mode == Lean5Mode::Constructive {
            return self.check_constructive_fast(expr);
        }

        // Fast path for non-extended expressions
        if !expr.uses_extensions() {
            return self.check_core_fast(expr);
        }

        // Full checking for extended expressions
        self.check(expr)
    }
}

impl Expr {
    /// Check if expression uses any mode-specific extensions
    pub fn uses_extensions(&self) -> bool {
        match self {
            // Core - no extensions
            Expr::BVar(_) | Expr::FVar(_) | Expr::Sort(_) |
            Expr::Const { .. } | Expr::Lit(_) | Expr::MVar(_) => false,

            // Recursive check for compound expressions
            Expr::App { func, arg } => func.uses_extensions() || arg.uses_extensions(),
            Expr::Lam { domain, body, .. } => domain.uses_extensions() || body.uses_extensions(),
            Expr::Pi { domain, codomain, .. } => domain.uses_extensions() || codomain.uses_extensions(),
            Expr::Let { ty, value, body, .. } => {
                ty.uses_extensions() || value.uses_extensions() || body.uses_extensions()
            }
            Expr::Proj { struct_expr, .. } => struct_expr.uses_extensions(),

            // All extensions
            _ => true,
        }
    }
}
```

### 8.2 Caching

```rust
/// Type checking cache
pub struct TypeCheckCache {
    /// Cached types for expressions (by hash)
    types: HashMap<u64, Expr>,

    /// Cached reduction results
    reductions: HashMap<u64, Expr>,

    /// Cached conversion checks
    conversions: HashMap<(u64, u64), bool>,
}
```

---

## 9. Future Extensions

### 9.1 Planned Features

1. **Higher Inductive Types (HITs)** - Define types with path constructors
2. **Observational Type Theory** - Definitional proof irrelevance
3. **Modal Type Theory** - For staged computation and metaprogramming
4. **Linear Types** - Resource tracking
5. **Effect Systems** - Algebraic effects with handlers

### 9.2 Extension Points

```rust
/// Extension hook for future type theory features
pub trait KernelExtension {
    /// New expression variants
    fn expr_variants(&self) -> Vec<ExprVariant>;

    /// Type checking rules
    fn check(&self, ctx: &mut TypeChecker, expr: &Expr) -> Option<Result<Expr, TypeError>>;

    /// Reduction rules
    fn reduce(&self, ctx: &mut TypeChecker, expr: &Expr) -> Option<Expr>;

    /// Compatibility with modes
    fn compatible_modes(&self) -> Vec<Lean5Mode>;
}
```

---

## 10. Verification Strategy

### 10.1 Self-Verification

The Lean5 kernel will be verified within Lean5 itself:

1. **Bootstrap Phase**: Minimal trusted kernel (< 5000 LOC Rust)
2. **Specification**: Type theory rules specified in Lean5
3. **Implementation**: Full kernel implemented in Lean5
4. **Proof**: Prove implementation matches specification
5. **Extraction**: Extract verified kernel back to Rust

### 10.2 Cross-Validation

- Run same proofs through multiple systems
- Compare proof terms structurally
- Hash-based integrity checking
- Differential testing against Lean 4

---

## Appendix A: Mode Compatibility Matrix

| Source Mode    | → Constructive | → Impredicative | → Cubical | → Classical | → SetTheoretic |
|----------------|----------------|-----------------|-----------|-------------|----------------|
| Constructive   | ✓              | ✓               | ✓         | ✓           | ✓              |
| Impredicative  | ✗              | ✓               | ✗         | ✓           | ✓              |
| Cubical        | ✗              | ✗               | ✓         | ✗           | ✗              |
| Classical      | ✗              | ✗               | ✗         | ✓           | ✓              |
| SetTheoretic   | ✗              | ✗               | ✗         | ✗           | ✓              |

---

## Appendix B: Source System Mapping

| System        | Default Mode   | Key Features to Handle                    |
|---------------|----------------|-------------------------------------------|
| Lean 4        | Constructive   | Quot, structure eta, auto-bound implicits |
| Coq           | Impredicative  | SProp, template polymorphism, cofix       |
| Agda          | Constructive   | Sized types, instance args, copatterns    |
| Cubical Agda  | Cubical        | Interval, Path, hcomp, Glue, univalence   |
| Isabelle/HOL  | Classical      | Typedef, type classes, locales            |
| HOL Light     | Classical      | Small kernel, proof recording             |
| HOL4          | Classical      | Term quotations, proof tactics            |
| Mizar         | SetTheoretic   | Soft types, registrations, clusters       |
| Metamath/ZFC  | SetTheoretic   | Pure set theory, explicit proof steps     |
| Metamath/set  | Classical      | set.mm logic, explicit steps              |
| PVS           | Classical      | Subtyping, dependent types, TCCs          |
| ACL2          | Classical      | First-order, total functions, induction   |

---

*Document version: 1.0*
*Last updated: 2026-01-07*
