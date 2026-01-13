//! Expression representation
//!
//! The core expression type used throughout Lean5.
//! Uses de Bruijn indices for bound variables.

use crate::level::Level;
use crate::name::Name;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::sync::Arc;

/// Minimum stack space to reserve before recursive calls (32 KB).
const MIN_STACK_RED_ZONE: usize = 32 * 1024;

/// Stack size to grow to when running low (1 MB).
const STACK_GROWTH_SIZE: usize = 1024 * 1024;

/// Type alias for universe level lists in Expr::Const.
///
/// Most constants have 0-2 universe levels (97.1% in Init.Prelude),
/// so we use SmallVec to avoid heap allocation for the common case.
/// This reduces allocation overhead during .olean loading.
pub type LevelVec = SmallVec<[Level; 2]>;

/// Binder information (how a variable is bound)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BinderInfo {
    /// Regular explicit binding
    Default,
    /// Implicit binding (inferred by unification) `{x : T}`
    Implicit,
    /// Strict implicit (must be inferrable) `{{x : T}}`
    StrictImplicit,
    /// Instance implicit (resolved by type class) `[x : T]`
    InstImplicit,
}

/// Literal values
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Literal {
    /// Natural number literal
    Nat(u64),
    /// String literal
    String(Arc<str>),
}

/// Metadata value for MData expressions
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MDataValue {
    /// Boolean metadata
    Bool(bool),
    /// Natural number metadata
    Nat(u64),
    /// String metadata
    String(Arc<str>),
    /// Name metadata
    Name(Name),
}

/// Key-value metadata map for MData expressions
pub type MDataMap = Vec<(Name, MDataValue)>;

/// Unique identifier for free variables
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FVarId(pub u64);

/// Core expression type
///
/// The Lean5 expression type supports multiple mathematical traditions through
/// mode-gated extensions. The core variants (BVar through MData) work in all modes.
/// Extended variants require specific modes (Cubical, Classical, SetTheoretic).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Expr {
    // ════════════════════════════════════════════════════════════════════════
    // CORE (all modes)
    // ════════════════════════════════════════════════════════════════════════
    /// Bound variable (de Bruijn index, 0 = innermost)
    BVar(u32),
    /// Free variable
    FVar(FVarId),
    /// Sort (Type u or Prop)
    Sort(Level),
    /// Constant with universe level instantiation
    Const(Name, LevelVec),
    /// Function application
    App(Arc<Expr>, Arc<Expr>),
    /// Lambda abstraction: λ (x : A), body
    Lam(BinderInfo, Arc<Expr>, Arc<Expr>),
    /// Pi/forall type: (x : A) → B
    Pi(BinderInfo, Arc<Expr>, Arc<Expr>),
    /// Let binding: let x : A := val in body
    Let(Arc<Expr>, Arc<Expr>, Arc<Expr>),
    /// Literal value
    Lit(Literal),
    /// Structure projection
    Proj(Name, u32, Arc<Expr>),
    /// Metadata wrapper (transparent to type checking)
    /// MData(metadata, inner_expr) - the metadata is carried but type is of inner_expr
    MData(MDataMap, Arc<Expr>),

    // ════════════════════════════════════════════════════════════════════════
    // IMPREDICATIVE MODE EXTENSIONS
    // These expressions are only valid in Impredicative mode (or Classical/SetTheoretic).
    // ════════════════════════════════════════════════════════════════════════
    /// Strict proposition sort (proof-irrelevant, no large elimination).
    /// SProp is always proof-irrelevant (unlike Prop which is only proof-irrelevant
    /// when proof irrelevance axiom is enabled).
    /// Mode: Impredicative
    SProp,

    /// Squash type (truncation to SProp).
    /// `Squash A` is a strict proposition that is inhabited iff A is inhabited.
    /// All proofs of `Squash A` are definitionally equal.
    /// Mode: Impredicative
    Squash(Arc<Expr>),

    // ════════════════════════════════════════════════════════════════════════
    // CUBICAL MODE EXTENSIONS
    // These expressions are only valid in Cubical mode.
    // ════════════════════════════════════════════════════════════════════════
    /// Interval type I with endpoints 0 and 1.
    /// Mode: Cubical
    CubicalInterval,

    /// Interval endpoint 0.
    /// Mode: Cubical
    CubicalI0,

    /// Interval endpoint 1.
    /// Mode: Cubical
    CubicalI1,

    /// Path type: Path A a b (heterogeneous equality).
    /// `ty` is `A : I -> Type`, `left` is `a : A 0`, `right` is `b : A 1`.
    /// Mode: Cubical
    CubicalPath {
        ty: Arc<Expr>,
        left: Arc<Expr>,
        right: Arc<Expr>,
    },

    /// Path lambda: `<i> e` (introduce a path by abstracting over interval).
    /// Mode: Cubical
    CubicalPathLam { body: Arc<Expr> },

    /// Path application: p @ i (apply a path to an interval point).
    /// Mode: Cubical
    CubicalPathApp { path: Arc<Expr>, arg: Arc<Expr> },

    /// Homogeneous composition: hcomp {A} {φ} u base.
    /// Computes a filler for a partial element along a cofibration.
    /// Mode: Cubical
    CubicalHComp {
        ty: Arc<Expr>,
        phi: Arc<Expr>,
        u: Arc<Expr>,
        base: Arc<Expr>,
    },

    /// Transport along a path: transp A φ base.
    /// Transports `base : A 0` to `A 1` along a line of types.
    /// Mode: Cubical
    CubicalTransp {
        ty: Arc<Expr>,
        phi: Arc<Expr>,
        base: Arc<Expr>,
    },

    // ════════════════════════════════════════════════════════════════════════
    // CLASSICAL MODE EXTENSIONS
    // These expressions are only valid in Classical mode.
    // ════════════════════════════════════════════════════════════════════════
    /// Classical choice operator: choice ty pred proof.
    /// Given `proof : ∃ x : ty, pred x`, returns an element `x : ty` with `pred x`.
    /// Mode: Classical
    ClassicalChoice {
        ty: Arc<Expr>,
        pred: Arc<Expr>,
        exists_proof: Arc<Expr>,
    },

    /// Hilbert epsilon (indefinite description): epsilon ty pred.
    /// Returns some `x : ty` satisfying `pred x` if one exists, arbitrary otherwise.
    /// Mode: Classical
    ClassicalEpsilon { ty: Arc<Expr>, pred: Arc<Expr> },

    // ════════════════════════════════════════════════════════════════════════
    // SET-THEORETIC MODE EXTENSIONS
    // These expressions are only valid in SetTheoretic mode.
    // ════════════════════════════════════════════════════════════════════════
    /// ZFC set expression (various set constructions).
    /// Mode: SetTheoretic
    ZFCSet(ZFCSetExpr),

    /// Set membership: element ∈ set.
    /// Mode: SetTheoretic
    ZFCMem {
        element: Arc<Expr>,
        set: Arc<Expr>,
    },

    /// Set comprehension: {x ∈ domain | pred x}.
    /// Mode: SetTheoretic
    ZFCComprehension {
        domain: Arc<Expr>,
        pred: Arc<Expr>,
    },
}

/// ZFC set expressions for SetTheoretic mode.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ZFCSetExpr {
    /// Empty set: ∅
    Empty,
    /// Singleton set: {a}
    Singleton(Arc<Expr>),
    /// Unordered pair: {a, b}
    Pair(Arc<Expr>, Arc<Expr>),
    /// Union: ⋃A (union of all sets in A)
    Union(Arc<Expr>),
    /// Power set: P(A) (set of all subsets of A)
    PowerSet(Arc<Expr>),
    /// Separation: {x ∈ A | φ(x)}
    Separation { set: Arc<Expr>, pred: Arc<Expr> },
    /// Replacement: {F(x) | x ∈ A}
    Replacement { set: Arc<Expr>, func: Arc<Expr> },
    /// Infinity: ω (the first infinite ordinal)
    Infinity,
    /// Choice function application
    Choice(Arc<Expr>),
}

impl ZFCSetExpr {
    /// Instantiate bound variable 0 with the given expression at depth
    fn instantiate_at(&self, val: &Expr, depth: u32) -> Self {
        match self {
            ZFCSetExpr::Empty | ZFCSetExpr::Infinity => self.clone(),
            ZFCSetExpr::Singleton(e) => {
                ZFCSetExpr::Singleton(Arc::new(e.instantiate_at(val, depth)))
            }
            ZFCSetExpr::Pair(a, b) => ZFCSetExpr::Pair(
                Arc::new(a.instantiate_at(val, depth)),
                Arc::new(b.instantiate_at(val, depth)),
            ),
            ZFCSetExpr::Union(e) => ZFCSetExpr::Union(Arc::new(e.instantiate_at(val, depth))),
            ZFCSetExpr::PowerSet(e) => ZFCSetExpr::PowerSet(Arc::new(e.instantiate_at(val, depth))),
            ZFCSetExpr::Separation { set, pred } => ZFCSetExpr::Separation {
                set: Arc::new(set.instantiate_at(val, depth)),
                pred: Arc::new(pred.instantiate_at(val, depth + 1)),
            },
            ZFCSetExpr::Replacement { set, func } => ZFCSetExpr::Replacement {
                set: Arc::new(set.instantiate_at(val, depth)),
                func: Arc::new(func.instantiate_at(val, depth + 1)),
            },
            ZFCSetExpr::Choice(e) => ZFCSetExpr::Choice(Arc::new(e.instantiate_at(val, depth))),
        }
    }

    /// Lift loose bound variables >= start by amount
    fn lift_at(&self, start: u32, amount: u32) -> Self {
        match self {
            ZFCSetExpr::Empty | ZFCSetExpr::Infinity => self.clone(),
            ZFCSetExpr::Singleton(e) => ZFCSetExpr::Singleton(Arc::new(e.lift_at(start, amount))),
            ZFCSetExpr::Pair(a, b) => ZFCSetExpr::Pair(
                Arc::new(a.lift_at(start, amount)),
                Arc::new(b.lift_at(start, amount)),
            ),
            ZFCSetExpr::Union(e) => ZFCSetExpr::Union(Arc::new(e.lift_at(start, amount))),
            ZFCSetExpr::PowerSet(e) => ZFCSetExpr::PowerSet(Arc::new(e.lift_at(start, amount))),
            ZFCSetExpr::Separation { set, pred } => ZFCSetExpr::Separation {
                set: Arc::new(set.lift_at(start, amount)),
                pred: Arc::new(pred.lift_at(start + 1, amount)),
            },
            ZFCSetExpr::Replacement { set, func } => ZFCSetExpr::Replacement {
                set: Arc::new(set.lift_at(start, amount)),
                func: Arc::new(func.lift_at(start + 1, amount)),
            },
            ZFCSetExpr::Choice(e) => ZFCSetExpr::Choice(Arc::new(e.lift_at(start, amount))),
        }
    }

    /// Check if expression has loose bound variables in range [start, end)
    fn has_loose_bvar_in_range(&self, start: u32, end: u32) -> bool {
        match self {
            ZFCSetExpr::Empty | ZFCSetExpr::Infinity => false,
            ZFCSetExpr::Singleton(e) => e.has_loose_bvar_in_range(start, end),
            ZFCSetExpr::Pair(a, b) => {
                a.has_loose_bvar_in_range(start, end) || b.has_loose_bvar_in_range(start, end)
            }
            ZFCSetExpr::Union(e) | ZFCSetExpr::PowerSet(e) | ZFCSetExpr::Choice(e) => {
                e.has_loose_bvar_in_range(start, end)
            }
            ZFCSetExpr::Separation { set, pred } | ZFCSetExpr::Replacement { set, func: pred } => {
                set.has_loose_bvar_in_range(start, end)
                    || pred.has_loose_bvar_in_range(start + 1, end.saturating_add(1))
            }
        }
    }

    /// Abstract: replace FVar(id) with BVar(depth), shifting other bound variables up
    fn abstract_fvar_at(&self, id: FVarId, depth: u32) -> Self {
        match self {
            ZFCSetExpr::Empty | ZFCSetExpr::Infinity => self.clone(),
            ZFCSetExpr::Singleton(e) => {
                ZFCSetExpr::Singleton(Arc::new(e.abstract_fvar_at(id, depth)))
            }
            ZFCSetExpr::Pair(a, b) => ZFCSetExpr::Pair(
                Arc::new(a.abstract_fvar_at(id, depth)),
                Arc::new(b.abstract_fvar_at(id, depth)),
            ),
            ZFCSetExpr::Union(e) => ZFCSetExpr::Union(Arc::new(e.abstract_fvar_at(id, depth))),
            ZFCSetExpr::PowerSet(e) => {
                ZFCSetExpr::PowerSet(Arc::new(e.abstract_fvar_at(id, depth)))
            }
            ZFCSetExpr::Separation { set, pred } => ZFCSetExpr::Separation {
                set: Arc::new(set.abstract_fvar_at(id, depth)),
                pred: Arc::new(pred.abstract_fvar_at(id, depth + 1)),
            },
            ZFCSetExpr::Replacement { set, func } => ZFCSetExpr::Replacement {
                set: Arc::new(set.abstract_fvar_at(id, depth)),
                func: Arc::new(func.abstract_fvar_at(id, depth + 1)),
            },
            ZFCSetExpr::Choice(e) => ZFCSetExpr::Choice(Arc::new(e.abstract_fvar_at(id, depth))),
        }
    }

    /// Substitute a free variable with an expression
    fn subst_fvar(&self, id: FVarId, replacement: &Expr) -> Self {
        match self {
            ZFCSetExpr::Empty | ZFCSetExpr::Infinity => self.clone(),
            ZFCSetExpr::Singleton(e) => {
                ZFCSetExpr::Singleton(Arc::new(e.subst_fvar(id, replacement)))
            }
            ZFCSetExpr::Pair(a, b) => ZFCSetExpr::Pair(
                Arc::new(a.subst_fvar(id, replacement)),
                Arc::new(b.subst_fvar(id, replacement)),
            ),
            ZFCSetExpr::Union(e) => ZFCSetExpr::Union(Arc::new(e.subst_fvar(id, replacement))),
            ZFCSetExpr::PowerSet(e) => {
                ZFCSetExpr::PowerSet(Arc::new(e.subst_fvar(id, replacement)))
            }
            ZFCSetExpr::Separation { set, pred } => ZFCSetExpr::Separation {
                set: Arc::new(set.subst_fvar(id, replacement)),
                pred: Arc::new(pred.subst_fvar(id, replacement)),
            },
            ZFCSetExpr::Replacement { set, func } => ZFCSetExpr::Replacement {
                set: Arc::new(set.subst_fvar(id, replacement)),
                func: Arc::new(func.subst_fvar(id, replacement)),
            },
            ZFCSetExpr::Choice(e) => ZFCSetExpr::Choice(Arc::new(e.subst_fvar(id, replacement))),
        }
    }

    /// Substitute universe parameters
    fn instantiate_level_params(&self, subst: &[(Name, Level)]) -> Self {
        match self {
            ZFCSetExpr::Empty | ZFCSetExpr::Infinity => self.clone(),
            ZFCSetExpr::Singleton(e) => {
                ZFCSetExpr::Singleton(Arc::new(e.instantiate_level_params(subst)))
            }
            ZFCSetExpr::Pair(a, b) => ZFCSetExpr::Pair(
                Arc::new(a.instantiate_level_params(subst)),
                Arc::new(b.instantiate_level_params(subst)),
            ),
            ZFCSetExpr::Union(e) => ZFCSetExpr::Union(Arc::new(e.instantiate_level_params(subst))),
            ZFCSetExpr::PowerSet(e) => {
                ZFCSetExpr::PowerSet(Arc::new(e.instantiate_level_params(subst)))
            }
            ZFCSetExpr::Separation { set, pred } => ZFCSetExpr::Separation {
                set: Arc::new(set.instantiate_level_params(subst)),
                pred: Arc::new(pred.instantiate_level_params(subst)),
            },
            ZFCSetExpr::Replacement { set, func } => ZFCSetExpr::Replacement {
                set: Arc::new(set.instantiate_level_params(subst)),
                func: Arc::new(func.instantiate_level_params(subst)),
            },
            ZFCSetExpr::Choice(e) => {
                ZFCSetExpr::Choice(Arc::new(e.instantiate_level_params(subst)))
            }
        }
    }
}

impl Expr {
    /// Create a bound variable
    pub fn bvar(idx: u32) -> Self {
        Expr::BVar(idx)
    }

    /// Create a free variable
    pub fn fvar(id: FVarId) -> Self {
        Expr::FVar(id)
    }

    /// Create a sort (Type u)
    pub fn sort(level: Level) -> Self {
        Expr::Sort(level)
    }

    /// Create Prop (Sort 0)
    pub fn prop() -> Self {
        Expr::Sort(Level::zero())
    }

    /// Create Type (Sort 1)
    pub fn type_() -> Self {
        Expr::Sort(Level::succ(Level::zero()))
    }

    /// Create a constant reference
    pub fn const_(name: Name, levels: impl Into<LevelVec>) -> Self {
        Expr::Const(name, levels.into())
    }

    /// Create an application
    pub fn app(func: Expr, arg: Expr) -> Self {
        Expr::App(Arc::new(func), Arc::new(arg))
    }

    /// Create a lambda
    pub fn lam(bi: BinderInfo, ty: Expr, body: Expr) -> Self {
        Expr::Lam(bi, Arc::new(ty), Arc::new(body))
    }

    /// Create a pi type
    pub fn pi(bi: BinderInfo, ty: Expr, body: Expr) -> Self {
        Expr::Pi(bi, Arc::new(ty), Arc::new(body))
    }

    /// Create an arrow type (non-dependent pi)
    pub fn arrow(from: Expr, to: Expr) -> Self {
        Expr::Pi(BinderInfo::Default, Arc::new(from), Arc::new(to))
    }

    /// Create a let binding
    pub fn let_(ty: Expr, val: Expr, body: Expr) -> Self {
        Expr::Let(Arc::new(ty), Arc::new(val), Arc::new(body))
    }

    /// Create a natural number literal
    pub fn nat_lit(n: u64) -> Self {
        Expr::Lit(Literal::Nat(n))
    }

    /// Create a string literal
    pub fn str_lit(s: impl AsRef<str>) -> Self {
        Expr::Lit(Literal::String(Arc::from(s.as_ref())))
    }

    /// Create a projection
    pub fn proj(struct_name: Name, idx: u32, expr: Expr) -> Self {
        Expr::Proj(struct_name, idx, Arc::new(expr))
    }

    /// Create a metadata wrapper
    pub fn mdata(metadata: MDataMap, expr: Expr) -> Self {
        Expr::MData(metadata, Arc::new(expr))
    }

    /// Get the inner expression if this is an MData, otherwise self
    pub fn strip_mdata(&self) -> &Expr {
        match self {
            Expr::MData(_, inner) => inner.strip_mdata(),
            _ => self,
        }
    }

    /// Check if this expression is a sort
    pub fn is_sort(&self) -> bool {
        matches!(self, Expr::Sort(_))
    }

    /// Check if this is Prop
    pub fn is_prop(&self) -> bool {
        matches!(self, Expr::Sort(l) if l.is_zero())
    }

    /// Get the head of an application spine
    pub fn get_app_fn(&self) -> &Expr {
        match self {
            Expr::App(f, _) => f.get_app_fn(),
            _ => self,
        }
    }

    /// Get all arguments of an application spine
    pub fn get_app_args(&self) -> Vec<&Expr> {
        let mut args = Vec::new();
        let mut curr = self;
        while let Expr::App(f, a) = curr {
            args.push(a.as_ref());
            curr = f.as_ref();
        }
        args.reverse();
        args
    }

    /// Substitute bound variable 0 with the given expression
    #[must_use]
    pub fn instantiate(&self, val: &Expr) -> Expr {
        self.instantiate_at(val, 0)
    }

    fn instantiate_at(&self, val: &Expr, depth: u32) -> Expr {
        stacker::maybe_grow(MIN_STACK_RED_ZONE, STACK_GROWTH_SIZE, || {
            self.instantiate_at_impl(val, depth)
        })
    }

    /// Implementation (called via stacker::maybe_grow)
    fn instantiate_at_impl(&self, val: &Expr, depth: u32) -> Expr {
        match self {
            Expr::BVar(idx) => {
                use std::cmp::Ordering;
                match idx.cmp(&depth) {
                    Ordering::Equal => val.lift(depth),
                    Ordering::Greater => Expr::BVar(idx - 1),
                    Ordering::Less => Expr::BVar(*idx),
                }
            }
            Expr::FVar(_) | Expr::Sort(_) | Expr::Const(_, _) | Expr::Lit(_) => self.clone(),
            Expr::App(f, a) => Expr::App(
                Arc::new(f.instantiate_at(val, depth)),
                Arc::new(a.instantiate_at(val, depth)),
            ),
            Expr::Lam(bi, ty, body) => Expr::Lam(
                *bi,
                Arc::new(ty.instantiate_at(val, depth)),
                Arc::new(body.instantiate_at(val, depth + 1)),
            ),
            Expr::Pi(bi, ty, body) => Expr::Pi(
                *bi,
                Arc::new(ty.instantiate_at(val, depth)),
                Arc::new(body.instantiate_at(val, depth + 1)),
            ),
            Expr::Let(ty, v, body) => Expr::Let(
                Arc::new(ty.instantiate_at(val, depth)),
                Arc::new(v.instantiate_at(val, depth)),
                Arc::new(body.instantiate_at(val, depth + 1)),
            ),
            Expr::Proj(name, idx, e) => {
                Expr::Proj(name.clone(), *idx, Arc::new(e.instantiate_at(val, depth)))
            }
            Expr::MData(meta, inner) => {
                Expr::MData(meta.clone(), Arc::new(inner.instantiate_at(val, depth)))
            }

            // Impredicative mode extensions
            Expr::SProp => Expr::SProp,
            Expr::Squash(inner) => Expr::Squash(Arc::new(inner.instantiate_at(val, depth))),

            // Cubical mode extensions
            Expr::CubicalInterval | Expr::CubicalI0 | Expr::CubicalI1 => self.clone(),
            Expr::CubicalPath { ty, left, right } => Expr::CubicalPath {
                ty: Arc::new(ty.instantiate_at(val, depth)),
                left: Arc::new(left.instantiate_at(val, depth)),
                right: Arc::new(right.instantiate_at(val, depth)),
            },
            Expr::CubicalPathLam { body } => Expr::CubicalPathLam {
                body: Arc::new(body.instantiate_at(val, depth + 1)),
            },
            Expr::CubicalPathApp { path, arg } => Expr::CubicalPathApp {
                path: Arc::new(path.instantiate_at(val, depth)),
                arg: Arc::new(arg.instantiate_at(val, depth)),
            },
            Expr::CubicalHComp { ty, phi, u, base } => Expr::CubicalHComp {
                ty: Arc::new(ty.instantiate_at(val, depth)),
                phi: Arc::new(phi.instantiate_at(val, depth)),
                u: Arc::new(u.instantiate_at(val, depth)),
                base: Arc::new(base.instantiate_at(val, depth)),
            },
            Expr::CubicalTransp { ty, phi, base } => Expr::CubicalTransp {
                ty: Arc::new(ty.instantiate_at(val, depth)),
                phi: Arc::new(phi.instantiate_at(val, depth)),
                base: Arc::new(base.instantiate_at(val, depth)),
            },

            // Classical mode extensions
            Expr::ClassicalChoice {
                ty,
                pred,
                exists_proof,
            } => Expr::ClassicalChoice {
                ty: Arc::new(ty.instantiate_at(val, depth)),
                pred: Arc::new(pred.instantiate_at(val, depth)),
                exists_proof: Arc::new(exists_proof.instantiate_at(val, depth)),
            },
            Expr::ClassicalEpsilon { ty, pred } => Expr::ClassicalEpsilon {
                ty: Arc::new(ty.instantiate_at(val, depth)),
                pred: Arc::new(pred.instantiate_at(val, depth)),
            },

            // SetTheoretic mode extensions
            Expr::ZFCSet(set_expr) => Expr::ZFCSet(set_expr.instantiate_at(val, depth)),
            Expr::ZFCMem { element, set } => Expr::ZFCMem {
                element: Arc::new(element.instantiate_at(val, depth)),
                set: Arc::new(set.instantiate_at(val, depth)),
            },
            Expr::ZFCComprehension { domain, pred } => Expr::ZFCComprehension {
                domain: Arc::new(domain.instantiate_at(val, depth)),
                pred: Arc::new(pred.instantiate_at(val, depth + 1)),
            },
        }
    }

    /// Lift loose bound variables >= `start` by `amount`
    ///
    /// This is used when substituting into a binder. For example, when we substitute
    /// `val` into `body` where body is inside a lambda, we need to lift the free
    /// variables in `val` by 1 so they refer to the right things.
    #[must_use]
    pub fn lift(&self, amount: u32) -> Expr {
        self.lift_at(0, amount)
    }

    /// Lift loose bound variables >= `start` by `amount`
    fn lift_at(&self, start: u32, amount: u32) -> Expr {
        stacker::maybe_grow(MIN_STACK_RED_ZONE, STACK_GROWTH_SIZE, || {
            self.lift_at_impl(start, amount)
        })
    }

    /// Implementation (called via stacker::maybe_grow)
    fn lift_at_impl(&self, start: u32, amount: u32) -> Expr {
        if amount == 0 {
            return self.clone();
        }
        match self {
            Expr::BVar(idx) => {
                if *idx >= start {
                    Expr::BVar(idx + amount)
                } else {
                    Expr::BVar(*idx)
                }
            }
            Expr::FVar(_) | Expr::Sort(_) | Expr::Const(_, _) | Expr::Lit(_) => self.clone(),
            Expr::App(f, a) => Expr::App(
                Arc::new(f.lift_at(start, amount)),
                Arc::new(a.lift_at(start, amount)),
            ),
            Expr::Lam(bi, ty, body) => Expr::Lam(
                *bi,
                Arc::new(ty.lift_at(start, amount)),
                Arc::new(body.lift_at(start + 1, amount)),
            ),
            Expr::Pi(bi, ty, body) => Expr::Pi(
                *bi,
                Arc::new(ty.lift_at(start, amount)),
                Arc::new(body.lift_at(start + 1, amount)),
            ),
            Expr::Let(ty, val, body) => Expr::Let(
                Arc::new(ty.lift_at(start, amount)),
                Arc::new(val.lift_at(start, amount)),
                Arc::new(body.lift_at(start + 1, amount)),
            ),
            Expr::Proj(name, idx, e) => {
                Expr::Proj(name.clone(), *idx, Arc::new(e.lift_at(start, amount)))
            }
            Expr::MData(meta, inner) => {
                Expr::MData(meta.clone(), Arc::new(inner.lift_at(start, amount)))
            }

            // Impredicative mode extensions
            Expr::SProp => Expr::SProp,
            Expr::Squash(inner) => Expr::Squash(Arc::new(inner.lift_at(start, amount))),

            // Cubical mode extensions
            Expr::CubicalInterval | Expr::CubicalI0 | Expr::CubicalI1 => self.clone(),
            Expr::CubicalPath { ty, left, right } => Expr::CubicalPath {
                ty: Arc::new(ty.lift_at(start, amount)),
                left: Arc::new(left.lift_at(start, amount)),
                right: Arc::new(right.lift_at(start, amount)),
            },
            Expr::CubicalPathLam { body } => Expr::CubicalPathLam {
                body: Arc::new(body.lift_at(start + 1, amount)),
            },
            Expr::CubicalPathApp { path, arg } => Expr::CubicalPathApp {
                path: Arc::new(path.lift_at(start, amount)),
                arg: Arc::new(arg.lift_at(start, amount)),
            },
            Expr::CubicalHComp { ty, phi, u, base } => Expr::CubicalHComp {
                ty: Arc::new(ty.lift_at(start, amount)),
                phi: Arc::new(phi.lift_at(start, amount)),
                u: Arc::new(u.lift_at(start, amount)),
                base: Arc::new(base.lift_at(start, amount)),
            },
            Expr::CubicalTransp { ty, phi, base } => Expr::CubicalTransp {
                ty: Arc::new(ty.lift_at(start, amount)),
                phi: Arc::new(phi.lift_at(start, amount)),
                base: Arc::new(base.lift_at(start, amount)),
            },

            // Classical mode extensions
            Expr::ClassicalChoice {
                ty,
                pred,
                exists_proof,
            } => Expr::ClassicalChoice {
                ty: Arc::new(ty.lift_at(start, amount)),
                pred: Arc::new(pred.lift_at(start, amount)),
                exists_proof: Arc::new(exists_proof.lift_at(start, amount)),
            },
            Expr::ClassicalEpsilon { ty, pred } => Expr::ClassicalEpsilon {
                ty: Arc::new(ty.lift_at(start, amount)),
                pred: Arc::new(pred.lift_at(start, amount)),
            },

            // SetTheoretic mode extensions
            Expr::ZFCSet(set_expr) => Expr::ZFCSet(set_expr.lift_at(start, amount)),
            Expr::ZFCMem { element, set } => Expr::ZFCMem {
                element: Arc::new(element.lift_at(start, amount)),
                set: Arc::new(set.lift_at(start, amount)),
            },
            Expr::ZFCComprehension { domain, pred } => Expr::ZFCComprehension {
                domain: Arc::new(domain.lift_at(start, amount)),
                pred: Arc::new(pred.lift_at(start + 1, amount)),
            },
        }
    }

    /// Check if expression has any loose bound variables
    pub fn has_loose_bvars(&self) -> bool {
        self.has_loose_bvar_in_range(0, u32::MAX)
    }

    /// Check if expression has loose bound variables in range [start, end)
    fn has_loose_bvar_in_range(&self, start: u32, end: u32) -> bool {
        match self {
            Expr::BVar(idx) => *idx >= start && *idx < end,
            Expr::FVar(_) | Expr::Sort(_) | Expr::Const(_, _) | Expr::Lit(_) => false,
            Expr::App(f, a) => {
                f.has_loose_bvar_in_range(start, end) || a.has_loose_bvar_in_range(start, end)
            }
            Expr::Lam(_, ty, body) | Expr::Pi(_, ty, body) => {
                ty.has_loose_bvar_in_range(start, end)
                    || body.has_loose_bvar_in_range(start + 1, end.saturating_add(1))
            }
            Expr::Let(ty, val, body) => {
                ty.has_loose_bvar_in_range(start, end)
                    || val.has_loose_bvar_in_range(start, end)
                    || body.has_loose_bvar_in_range(start + 1, end.saturating_add(1))
            }
            Expr::Proj(_, _, e) => e.has_loose_bvar_in_range(start, end),
            Expr::MData(_, inner) => inner.has_loose_bvar_in_range(start, end),

            // Impredicative mode extensions
            Expr::SProp => false,
            Expr::Squash(inner) => inner.has_loose_bvar_in_range(start, end),

            // Cubical mode extensions
            Expr::CubicalInterval | Expr::CubicalI0 | Expr::CubicalI1 => false,
            Expr::CubicalPath { ty, left, right } => {
                ty.has_loose_bvar_in_range(start, end)
                    || left.has_loose_bvar_in_range(start, end)
                    || right.has_loose_bvar_in_range(start, end)
            }
            Expr::CubicalPathLam { body } => {
                body.has_loose_bvar_in_range(start + 1, end.saturating_add(1))
            }
            Expr::CubicalPathApp { path, arg } => {
                path.has_loose_bvar_in_range(start, end) || arg.has_loose_bvar_in_range(start, end)
            }
            Expr::CubicalHComp { ty, phi, u, base } => {
                ty.has_loose_bvar_in_range(start, end)
                    || phi.has_loose_bvar_in_range(start, end)
                    || u.has_loose_bvar_in_range(start, end)
                    || base.has_loose_bvar_in_range(start, end)
            }
            Expr::CubicalTransp { ty, phi, base } => {
                ty.has_loose_bvar_in_range(start, end)
                    || phi.has_loose_bvar_in_range(start, end)
                    || base.has_loose_bvar_in_range(start, end)
            }

            // Classical mode extensions
            Expr::ClassicalChoice {
                ty,
                pred,
                exists_proof,
            } => {
                ty.has_loose_bvar_in_range(start, end)
                    || pred.has_loose_bvar_in_range(start, end)
                    || exists_proof.has_loose_bvar_in_range(start, end)
            }
            Expr::ClassicalEpsilon { ty, pred } => {
                ty.has_loose_bvar_in_range(start, end) || pred.has_loose_bvar_in_range(start, end)
            }

            // SetTheoretic mode extensions
            Expr::ZFCSet(set_expr) => set_expr.has_loose_bvar_in_range(start, end),
            Expr::ZFCMem { element, set } => {
                element.has_loose_bvar_in_range(start, end)
                    || set.has_loose_bvar_in_range(start, end)
            }
            Expr::ZFCComprehension { domain, pred } => {
                domain.has_loose_bvar_in_range(start, end)
                    || pred.has_loose_bvar_in_range(start + 1, end.saturating_add(1))
            }
        }
    }

    /// Abstract: replace FVar(id) with BVar(0), shifting other bound variables up
    #[must_use]
    pub fn abstract_fvar(&self, id: FVarId) -> Expr {
        self.abstract_fvar_at(id, 0)
    }

    fn abstract_fvar_at(&self, id: FVarId, depth: u32) -> Expr {
        stacker::maybe_grow(MIN_STACK_RED_ZONE, STACK_GROWTH_SIZE, || {
            self.abstract_fvar_at_impl(id, depth)
        })
    }

    /// Implementation (called via stacker::maybe_grow)
    fn abstract_fvar_at_impl(&self, id: FVarId, depth: u32) -> Expr {
        match self {
            Expr::FVar(fid) if *fid == id => Expr::BVar(depth),
            Expr::FVar(_) | Expr::Sort(_) | Expr::Const(_, _) | Expr::Lit(_) => self.clone(),
            Expr::BVar(idx) => {
                // Bound variables >= depth need to be shifted up
                if *idx >= depth {
                    Expr::BVar(idx + 1)
                } else {
                    Expr::BVar(*idx)
                }
            }
            Expr::App(f, a) => Expr::App(
                Arc::new(f.abstract_fvar_at(id, depth)),
                Arc::new(a.abstract_fvar_at(id, depth)),
            ),
            Expr::Lam(bi, ty, body) => Expr::Lam(
                *bi,
                Arc::new(ty.abstract_fvar_at(id, depth)),
                Arc::new(body.abstract_fvar_at(id, depth + 1)),
            ),
            Expr::Pi(bi, ty, body) => Expr::Pi(
                *bi,
                Arc::new(ty.abstract_fvar_at(id, depth)),
                Arc::new(body.abstract_fvar_at(id, depth + 1)),
            ),
            Expr::Let(ty, val, body) => Expr::Let(
                Arc::new(ty.abstract_fvar_at(id, depth)),
                Arc::new(val.abstract_fvar_at(id, depth)),
                Arc::new(body.abstract_fvar_at(id, depth + 1)),
            ),
            Expr::Proj(name, idx, e) => {
                Expr::Proj(name.clone(), *idx, Arc::new(e.abstract_fvar_at(id, depth)))
            }
            Expr::MData(meta, inner) => {
                Expr::MData(meta.clone(), Arc::new(inner.abstract_fvar_at(id, depth)))
            }

            // Impredicative mode extensions
            Expr::SProp => Expr::SProp,
            Expr::Squash(inner) => Expr::Squash(Arc::new(inner.abstract_fvar_at(id, depth))),

            // Cubical mode extensions
            Expr::CubicalInterval | Expr::CubicalI0 | Expr::CubicalI1 => self.clone(),
            Expr::CubicalPath { ty, left, right } => Expr::CubicalPath {
                ty: Arc::new(ty.abstract_fvar_at(id, depth)),
                left: Arc::new(left.abstract_fvar_at(id, depth)),
                right: Arc::new(right.abstract_fvar_at(id, depth)),
            },
            Expr::CubicalPathLam { body } => Expr::CubicalPathLam {
                body: Arc::new(body.abstract_fvar_at(id, depth + 1)),
            },
            Expr::CubicalPathApp { path, arg } => Expr::CubicalPathApp {
                path: Arc::new(path.abstract_fvar_at(id, depth)),
                arg: Arc::new(arg.abstract_fvar_at(id, depth)),
            },
            Expr::CubicalHComp { ty, phi, u, base } => Expr::CubicalHComp {
                ty: Arc::new(ty.abstract_fvar_at(id, depth)),
                phi: Arc::new(phi.abstract_fvar_at(id, depth)),
                u: Arc::new(u.abstract_fvar_at(id, depth)),
                base: Arc::new(base.abstract_fvar_at(id, depth)),
            },
            Expr::CubicalTransp { ty, phi, base } => Expr::CubicalTransp {
                ty: Arc::new(ty.abstract_fvar_at(id, depth)),
                phi: Arc::new(phi.abstract_fvar_at(id, depth)),
                base: Arc::new(base.abstract_fvar_at(id, depth)),
            },

            // Classical mode extensions
            Expr::ClassicalChoice {
                ty,
                pred,
                exists_proof,
            } => Expr::ClassicalChoice {
                ty: Arc::new(ty.abstract_fvar_at(id, depth)),
                pred: Arc::new(pred.abstract_fvar_at(id, depth)),
                exists_proof: Arc::new(exists_proof.abstract_fvar_at(id, depth)),
            },
            Expr::ClassicalEpsilon { ty, pred } => Expr::ClassicalEpsilon {
                ty: Arc::new(ty.abstract_fvar_at(id, depth)),
                pred: Arc::new(pred.abstract_fvar_at(id, depth)),
            },

            // SetTheoretic mode extensions
            Expr::ZFCSet(set_expr) => Expr::ZFCSet(set_expr.abstract_fvar_at(id, depth)),
            Expr::ZFCMem { element, set } => Expr::ZFCMem {
                element: Arc::new(element.abstract_fvar_at(id, depth)),
                set: Arc::new(set.abstract_fvar_at(id, depth)),
            },
            Expr::ZFCComprehension { domain, pred } => Expr::ZFCComprehension {
                domain: Arc::new(domain.abstract_fvar_at(id, depth)),
                pred: Arc::new(pred.abstract_fvar_at(id, depth + 1)),
            },
        }
    }

    /// Substitute a free variable with an expression
    /// This is similar to instantiate but for free variables instead of bound variables
    #[must_use]
    pub fn subst_fvar(&self, id: FVarId, replacement: &Expr) -> Expr {
        stacker::maybe_grow(MIN_STACK_RED_ZONE, STACK_GROWTH_SIZE, || {
            self.subst_fvar_impl(id, replacement)
        })
    }

    /// Implementation (called via stacker::maybe_grow)
    fn subst_fvar_impl(&self, id: FVarId, replacement: &Expr) -> Expr {
        match self {
            Expr::FVar(fid) if *fid == id => replacement.clone(),
            Expr::FVar(_) | Expr::BVar(_) | Expr::Sort(_) | Expr::Const(_, _) | Expr::Lit(_) => {
                self.clone()
            }
            Expr::App(f, a) => Expr::App(
                Arc::new(f.subst_fvar(id, replacement)),
                Arc::new(a.subst_fvar(id, replacement)),
            ),
            Expr::Lam(bi, ty, body) => Expr::Lam(
                *bi,
                Arc::new(ty.subst_fvar(id, replacement)),
                Arc::new(body.subst_fvar(id, replacement)),
            ),
            Expr::Pi(bi, ty, body) => Expr::Pi(
                *bi,
                Arc::new(ty.subst_fvar(id, replacement)),
                Arc::new(body.subst_fvar(id, replacement)),
            ),
            Expr::Let(ty, val, body) => Expr::Let(
                Arc::new(ty.subst_fvar(id, replacement)),
                Arc::new(val.subst_fvar(id, replacement)),
                Arc::new(body.subst_fvar(id, replacement)),
            ),
            Expr::Proj(name, idx, e) => {
                Expr::Proj(name.clone(), *idx, Arc::new(e.subst_fvar(id, replacement)))
            }
            Expr::MData(meta, inner) => {
                Expr::MData(meta.clone(), Arc::new(inner.subst_fvar(id, replacement)))
            }

            // Impredicative mode extensions
            Expr::SProp => Expr::SProp,
            Expr::Squash(inner) => Expr::Squash(Arc::new(inner.subst_fvar(id, replacement))),

            // Cubical mode extensions
            Expr::CubicalInterval | Expr::CubicalI0 | Expr::CubicalI1 => self.clone(),
            Expr::CubicalPath { ty, left, right } => Expr::CubicalPath {
                ty: Arc::new(ty.subst_fvar(id, replacement)),
                left: Arc::new(left.subst_fvar(id, replacement)),
                right: Arc::new(right.subst_fvar(id, replacement)),
            },
            Expr::CubicalPathLam { body } => Expr::CubicalPathLam {
                body: Arc::new(body.subst_fvar(id, replacement)),
            },
            Expr::CubicalPathApp { path, arg } => Expr::CubicalPathApp {
                path: Arc::new(path.subst_fvar(id, replacement)),
                arg: Arc::new(arg.subst_fvar(id, replacement)),
            },
            Expr::CubicalHComp { ty, phi, u, base } => Expr::CubicalHComp {
                ty: Arc::new(ty.subst_fvar(id, replacement)),
                phi: Arc::new(phi.subst_fvar(id, replacement)),
                u: Arc::new(u.subst_fvar(id, replacement)),
                base: Arc::new(base.subst_fvar(id, replacement)),
            },
            Expr::CubicalTransp { ty, phi, base } => Expr::CubicalTransp {
                ty: Arc::new(ty.subst_fvar(id, replacement)),
                phi: Arc::new(phi.subst_fvar(id, replacement)),
                base: Arc::new(base.subst_fvar(id, replacement)),
            },

            // Classical mode extensions
            Expr::ClassicalChoice {
                ty,
                pred,
                exists_proof,
            } => Expr::ClassicalChoice {
                ty: Arc::new(ty.subst_fvar(id, replacement)),
                pred: Arc::new(pred.subst_fvar(id, replacement)),
                exists_proof: Arc::new(exists_proof.subst_fvar(id, replacement)),
            },
            Expr::ClassicalEpsilon { ty, pred } => Expr::ClassicalEpsilon {
                ty: Arc::new(ty.subst_fvar(id, replacement)),
                pred: Arc::new(pred.subst_fvar(id, replacement)),
            },

            // SetTheoretic mode extensions
            Expr::ZFCSet(set_expr) => Expr::ZFCSet(set_expr.subst_fvar(id, replacement)),
            Expr::ZFCMem { element, set } => Expr::ZFCMem {
                element: Arc::new(element.subst_fvar(id, replacement)),
                set: Arc::new(set.subst_fvar(id, replacement)),
            },
            Expr::ZFCComprehension { domain, pred } => Expr::ZFCComprehension {
                domain: Arc::new(domain.subst_fvar(id, replacement)),
                pred: Arc::new(pred.subst_fvar(id, replacement)),
            },
        }
    }

    /// Substitute universe parameters in this expression
    #[must_use]
    pub fn instantiate_level_params(&self, subst: &[(Name, Level)]) -> Expr {
        stacker::maybe_grow(MIN_STACK_RED_ZONE, STACK_GROWTH_SIZE, || {
            self.instantiate_level_params_impl(subst)
        })
    }

    /// Implementation (called via stacker::maybe_grow)
    fn instantiate_level_params_impl(&self, subst: &[(Name, Level)]) -> Expr {
        if subst.is_empty() {
            return self.clone();
        }
        match self {
            Expr::BVar(_) | Expr::FVar(_) | Expr::Lit(_) => self.clone(),
            Expr::Sort(l) => Expr::Sort(l.substitute(subst)),
            Expr::Const(name, levels) => {
                let new_levels: LevelVec = levels.iter().map(|l| l.substitute(subst)).collect();
                Expr::Const(name.clone(), new_levels)
            }
            Expr::App(f, a) => Expr::App(
                Arc::new(f.instantiate_level_params(subst)),
                Arc::new(a.instantiate_level_params(subst)),
            ),
            Expr::Lam(bi, ty, body) => Expr::Lam(
                *bi,
                Arc::new(ty.instantiate_level_params(subst)),
                Arc::new(body.instantiate_level_params(subst)),
            ),
            Expr::Pi(bi, ty, body) => Expr::Pi(
                *bi,
                Arc::new(ty.instantiate_level_params(subst)),
                Arc::new(body.instantiate_level_params(subst)),
            ),
            Expr::Let(ty, val, body) => Expr::Let(
                Arc::new(ty.instantiate_level_params(subst)),
                Arc::new(val.instantiate_level_params(subst)),
                Arc::new(body.instantiate_level_params(subst)),
            ),
            Expr::Proj(name, idx, e) => Expr::Proj(
                name.clone(),
                *idx,
                Arc::new(e.instantiate_level_params(subst)),
            ),
            Expr::MData(meta, inner) => Expr::MData(
                meta.clone(),
                Arc::new(inner.instantiate_level_params(subst)),
            ),

            // Impredicative mode extensions
            Expr::SProp => Expr::SProp,
            Expr::Squash(inner) => {
                Expr::Squash(Arc::new(inner.instantiate_level_params(subst)))
            }

            // Cubical mode extensions
            Expr::CubicalInterval | Expr::CubicalI0 | Expr::CubicalI1 => self.clone(),
            Expr::CubicalPath { ty, left, right } => Expr::CubicalPath {
                ty: Arc::new(ty.instantiate_level_params(subst)),
                left: Arc::new(left.instantiate_level_params(subst)),
                right: Arc::new(right.instantiate_level_params(subst)),
            },
            Expr::CubicalPathLam { body } => Expr::CubicalPathLam {
                body: Arc::new(body.instantiate_level_params(subst)),
            },
            Expr::CubicalPathApp { path, arg } => Expr::CubicalPathApp {
                path: Arc::new(path.instantiate_level_params(subst)),
                arg: Arc::new(arg.instantiate_level_params(subst)),
            },
            Expr::CubicalHComp { ty, phi, u, base } => Expr::CubicalHComp {
                ty: Arc::new(ty.instantiate_level_params(subst)),
                phi: Arc::new(phi.instantiate_level_params(subst)),
                u: Arc::new(u.instantiate_level_params(subst)),
                base: Arc::new(base.instantiate_level_params(subst)),
            },
            Expr::CubicalTransp { ty, phi, base } => Expr::CubicalTransp {
                ty: Arc::new(ty.instantiate_level_params(subst)),
                phi: Arc::new(phi.instantiate_level_params(subst)),
                base: Arc::new(base.instantiate_level_params(subst)),
            },

            // Classical mode extensions
            Expr::ClassicalChoice {
                ty,
                pred,
                exists_proof,
            } => Expr::ClassicalChoice {
                ty: Arc::new(ty.instantiate_level_params(subst)),
                pred: Arc::new(pred.instantiate_level_params(subst)),
                exists_proof: Arc::new(exists_proof.instantiate_level_params(subst)),
            },
            Expr::ClassicalEpsilon { ty, pred } => Expr::ClassicalEpsilon {
                ty: Arc::new(ty.instantiate_level_params(subst)),
                pred: Arc::new(pred.instantiate_level_params(subst)),
            },

            // SetTheoretic mode extensions
            Expr::ZFCSet(set_expr) => Expr::ZFCSet(set_expr.instantiate_level_params(subst)),
            Expr::ZFCMem { element, set } => Expr::ZFCMem {
                element: Arc::new(element.instantiate_level_params(subst)),
                set: Arc::new(set.instantiate_level_params(subst)),
            },
            Expr::ZFCComprehension { domain, pred } => Expr::ZFCComprehension {
                domain: Arc::new(domain.instantiate_level_params(subst)),
                pred: Arc::new(pred.instantiate_level_params(subst)),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lift_bvar() {
        // BVar(0) lifted by 1 at depth 0 should become BVar(1)
        let e = Expr::BVar(0);
        assert_eq!(e.lift(1), Expr::BVar(1));

        // BVar(0) inside a lambda should NOT be lifted (it's bound)
        // We test lift_at directly for this
        let e = Expr::BVar(0);
        assert_eq!(e.lift_at(1, 1), Expr::BVar(0)); // BVar(0) < start=1, no change
        assert_eq!(e.lift_at(0, 1), Expr::BVar(1)); // BVar(0) >= start=0, lifted

        let e = Expr::BVar(2);
        assert_eq!(e.lift_at(1, 3), Expr::BVar(5)); // BVar(2) >= start=1, lifted by 3
    }

    #[test]
    fn test_instantiate() {
        // (λ x. x) instantiated with y should give y
        let body = Expr::BVar(0);
        let val = Expr::fvar(FVarId(42));
        let result = body.instantiate(&val);
        assert_eq!(result, Expr::fvar(FVarId(42)));

        // (λ x. λ y. x) - inner body should have BVar(1) for x
        // instantiate outer: BVar(1) -> should become BVar(0) after shift
        let inner_body = Expr::BVar(1); // refers to outer x
        let val = Expr::fvar(FVarId(99));
        let result = inner_body.instantiate(&val);
        // BVar(1) at depth 0: 1 > 0, so becomes BVar(0) (shifted down)
        assert_eq!(result, Expr::BVar(0));
    }

    #[test]
    fn test_abstract_fvar() {
        let fvar = Expr::fvar(FVarId(42));
        let result = fvar.abstract_fvar(FVarId(42));
        assert_eq!(result, Expr::BVar(0));

        // Different fvar should not be abstracted
        let result = fvar.abstract_fvar(FVarId(99));
        assert_eq!(result, Expr::fvar(FVarId(42)));
    }

    #[test]
    fn test_has_loose_bvars() {
        assert!(Expr::BVar(0).has_loose_bvars());
        assert!(!Expr::fvar(FVarId(0)).has_loose_bvars());
        assert!(!Expr::prop().has_loose_bvars());

        // Lambda binds the BVar(0), so no loose bvars
        let lam = Expr::lam(BinderInfo::Default, Expr::prop(), Expr::BVar(0));
        assert!(!lam.has_loose_bvars());

        // BVar(1) inside lambda is loose (refers outside)
        let lam = Expr::lam(BinderInfo::Default, Expr::prop(), Expr::BVar(1));
        assert!(lam.has_loose_bvars());
    }

    // =========================================================================
    // Mutation Testing Kill Tests
    // =========================================================================

    #[test]
    fn test_is_sort_predicates() {
        // Kill mutants: is_sort can return true always
        assert!(Expr::prop().is_sort());
        assert!(Expr::type_().is_sort());
        assert!(!Expr::BVar(0).is_sort());
        assert!(!Expr::fvar(FVarId(0)).is_sort());
        assert!(!Expr::lam(BinderInfo::Default, Expr::prop(), Expr::BVar(0)).is_sort());
        assert!(!Expr::nat_lit(42).is_sort());
    }

    #[test]
    fn test_is_prop_predicate() {
        // Kill mutant: is_prop can return true always
        assert!(Expr::prop().is_prop());
        assert!(!Expr::type_().is_prop());
        assert!(!Expr::Sort(Level::succ(Level::succ(Level::zero()))).is_prop());
        assert!(!Expr::BVar(0).is_prop());
    }

    #[test]
    fn test_instantiate_at_boundary_conditions() {
        // Kill mutants: instantiate_at > vs >= comparison

        // BVar(0) at depth 0 should be replaced
        let body = Expr::BVar(0);
        let val = Expr::prop();
        assert_eq!(body.instantiate(&val), Expr::prop());

        // BVar(1) at depth 0 should be decremented to BVar(0)
        let body = Expr::BVar(1);
        assert_eq!(body.instantiate(&val), Expr::BVar(0));

        // BVar(2) at depth 0 should become BVar(1)
        let body = Expr::BVar(2);
        assert_eq!(body.instantiate(&val), Expr::BVar(1));

        // Inside a binder, BVar(0) refers to the binder, not substituted
        // λ (x : Prop). x -> x is BVar(0) at depth 1, so no substitution
        let inner = Expr::lam(BinderInfo::Default, Expr::prop(), Expr::BVar(0));
        // Instantiating the outer level shouldn't change inner BVar(0)
        let result = inner.instantiate(&val);
        assert_eq!(
            result,
            Expr::lam(BinderInfo::Default, Expr::prop(), Expr::BVar(0))
        );
    }

    #[test]
    fn test_instantiate_arithmetic() {
        // Kill mutant: instantiate_at + with * in body.instantiate_at(val, depth + 1)
        // This tests that depth is incremented correctly under binders

        // Simple case: λ x. BVar(1) -- BVar(1) refers to the substitution target
        // When instantiated at depth 0, the body is processed at depth 1
        // BVar(1) at depth 1: 1 == 1, so gets replaced with val.lift(1)
        let lam = Expr::lam(BinderInfo::Default, Expr::prop(), Expr::BVar(1));
        let val = Expr::type_();
        let result = lam.instantiate(&val);

        // Body should have BVar(1) replaced with Type.lift(1) = Type
        // (Type has no loose bvars so lifting doesn't change it)
        match result {
            Expr::Lam(_, _, body) => {
                assert_eq!(body.as_ref(), &Expr::type_());
            }
            _ => panic!("Expected lambda"),
        }

        // More complex: test that BVar references above the binder are decremented
        // λ x. BVar(2) -- BVar(2) is above the substitution depth
        // At depth 1: BVar(2) > 1, so becomes BVar(1)
        let lam = Expr::lam(BinderInfo::Default, Expr::prop(), Expr::BVar(2));
        let result = lam.instantiate(&val);
        match result {
            Expr::Lam(_, _, body) => {
                assert_eq!(body.as_ref(), &Expr::BVar(1));
            }
            _ => panic!("Expected lambda"),
        }
    }

    #[test]
    fn test_lift_at_arithmetic() {
        // Kill mutants: lift_at + with * or -

        // BVar(2) at start=1 lifted by 3 should be BVar(5), not BVar(6) or BVar(-1)
        let e = Expr::BVar(2);
        assert_eq!(e.lift_at(1, 3), Expr::BVar(5)); // 2 >= 1, so 2+3=5

        // BVar(0) at start=1 should NOT be lifted
        let e = Expr::BVar(0);
        assert_eq!(e.lift_at(1, 3), Expr::BVar(0)); // 0 < 1, no change

        // Test inside nested binders - start should increment
        // λ x. λ y. BVar(0)  -- BVar(0) refers to y, shouldn't be lifted
        let inner = Expr::lam(BinderInfo::Default, Expr::prop(), Expr::BVar(0));
        let outer = Expr::lam(BinderInfo::Default, Expr::prop(), inner);
        let result = outer.lift(1);

        // Inner BVar(0) is under 2 binders, so start becomes 0+1+1=2
        // BVar(0) < 2, no change
        match result {
            Expr::Lam(_, _, body) => match body.as_ref() {
                Expr::Lam(_, _, inner_body) => {
                    assert_eq!(inner_body.as_ref(), &Expr::BVar(0));
                }
                _ => panic!("Expected nested lambda"),
            },
            _ => panic!("Expected lambda"),
        }

        // λ x. BVar(1) -- BVar(1) refers outside lambda, should be lifted
        let lam = Expr::lam(BinderInfo::Default, Expr::prop(), Expr::BVar(1));
        let result = lam.lift(1);
        match result {
            Expr::Lam(_, _, body) => {
                // Under one binder, start=1. BVar(1) >= 1, so lifted to BVar(2)
                assert_eq!(body.as_ref(), &Expr::BVar(2));
            }
            _ => panic!("Expected lambda"),
        }
    }

    #[test]
    fn test_has_loose_bvar_in_range_logic() {
        // Kill mutants: has_loose_bvar_in_range && to ||, < to <=, etc.

        // BVar(5) in range [3, 7) should be loose
        assert!(Expr::BVar(5).has_loose_bvar_in_range(3, 7));

        // BVar(3) in range [3, 7) should be loose (inclusive start)
        assert!(Expr::BVar(3).has_loose_bvar_in_range(3, 7));

        // BVar(7) in range [3, 7) should NOT be loose (exclusive end)
        assert!(!Expr::BVar(7).has_loose_bvar_in_range(3, 7));

        // BVar(2) in range [3, 7) should NOT be loose (below start)
        assert!(!Expr::BVar(2).has_loose_bvar_in_range(3, 7));

        // App: requires EITHER f OR a to have loose bvar (||, not &&)
        let app_with_loose = Expr::app(Expr::BVar(5), Expr::prop());
        assert!(app_with_loose.has_loose_bvar_in_range(3, 7));

        let app_without_loose = Expr::app(Expr::prop(), Expr::type_());
        assert!(!app_without_loose.has_loose_bvar_in_range(3, 7));
    }

    #[test]
    fn test_has_loose_bvar_nested_binders_arithmetic() {
        // Kill mutant: has_loose_bvar_in_range start + 1 with * or -

        // λ x. BVar(0) -- BVar(0) at depth 1 is NOT loose (bound by lambda)
        let lam = Expr::lam(BinderInfo::Default, Expr::prop(), Expr::BVar(0));
        assert!(!lam.has_loose_bvars());

        // λ x. BVar(1) -- BVar(1) at depth 1 IS loose (refers outside)
        let lam = Expr::lam(BinderInfo::Default, Expr::prop(), Expr::BVar(1));
        assert!(lam.has_loose_bvars());

        // λ x. λ y. BVar(2) -- BVar(2) at depth 2 IS loose
        let inner = Expr::lam(BinderInfo::Default, Expr::prop(), Expr::BVar(2));
        let outer = Expr::lam(BinderInfo::Default, Expr::prop(), inner);
        assert!(outer.has_loose_bvars());

        // λ x. λ y. BVar(1) -- BVar(1) at depth 2 is NOT loose (refers to x)
        let inner = Expr::lam(BinderInfo::Default, Expr::prop(), Expr::BVar(1));
        let outer = Expr::lam(BinderInfo::Default, Expr::prop(), inner);
        assert!(!outer.has_loose_bvars());
    }

    #[test]
    fn test_abstract_fvar_at_arithmetic() {
        // Kill mutants: abstract_fvar_at + with * or -

        // FVar(42) at depth 0 becomes BVar(0)
        let fvar = Expr::fvar(FVarId(42));
        assert_eq!(fvar.abstract_fvar(FVarId(42)), Expr::BVar(0));

        // Inside a lambda, FVar(42) becomes BVar(1) (depth increases)
        let lam = Expr::lam(BinderInfo::Default, Expr::prop(), Expr::fvar(FVarId(42)));
        let result = lam.abstract_fvar(FVarId(42));
        match result {
            Expr::Lam(_, _, body) => {
                assert_eq!(body.as_ref(), &Expr::BVar(1)); // depth was 0+1=1
            }
            _ => panic!("Expected lambda"),
        }

        // Doubly nested: FVar becomes BVar(2)
        let inner = Expr::lam(BinderInfo::Default, Expr::prop(), Expr::fvar(FVarId(42)));
        let outer = Expr::lam(BinderInfo::Default, Expr::prop(), inner);
        let result = outer.abstract_fvar(FVarId(42));
        match result {
            Expr::Lam(_, _, body) => match body.as_ref() {
                Expr::Lam(_, _, inner_body) => {
                    assert_eq!(inner_body.as_ref(), &Expr::BVar(2)); // depth was 0+1+1=2
                }
                _ => panic!("Expected nested lambda"),
            },
            _ => panic!("Expected lambda"),
        }
    }

    #[test]
    fn test_abstract_fvar_bvar_shift() {
        // Kill mutant: BVar(idx) >= depth should increment, testing idx + 1

        // BVar(0) at depth 0: 0 >= 0, should shift to BVar(1)
        let bvar = Expr::BVar(0);
        let result = bvar.abstract_fvar(FVarId(42));
        assert_eq!(result, Expr::BVar(1)); // shifted up by 1

        // BVar(5) at depth 3: 5 >= 3, should shift to BVar(6)
        let bvar = Expr::BVar(5);
        let result = bvar.abstract_fvar_at(FVarId(42), 3);
        assert_eq!(result, Expr::BVar(6));

        // BVar(2) at depth 5: 2 < 5, should NOT shift
        let bvar = Expr::BVar(2);
        let result = bvar.abstract_fvar_at(FVarId(42), 5);
        assert_eq!(result, Expr::BVar(2));
    }

    // =========================================================================
    // Additional Mutation Testing Kill Tests - expr.rs survivors
    // =========================================================================

    #[test]
    fn test_instantiate_at_greater_than() {
        // Kill mutant: replace > with >= in Expr::instantiate_at (line 173)
        // BVar(idx) > depth, not >=, because if idx == depth we substitute, not decrement

        // BVar(0) at depth 0: 0 == 0, gets substituted (not 0 > 0)
        let body = Expr::BVar(0);
        let val = Expr::type_();
        let result = body.instantiate_at(&val, 0);
        assert_eq!(
            result,
            Expr::type_(),
            "BVar(0) at depth 0 should be substituted"
        );

        // BVar(1) at depth 0: 1 > 0, gets decremented to BVar(0)
        let body = Expr::BVar(1);
        let result = body.instantiate_at(&val, 0);
        assert_eq!(
            result,
            Expr::BVar(0),
            "BVar(1) at depth 0 should become BVar(0)"
        );

        // BVar(1) at depth 1: 1 == 1, gets substituted
        let body = Expr::BVar(1);
        let result = body.instantiate_at(&val, 1);
        assert_eq!(
            result,
            val.lift(1),
            "BVar(1) at depth 1 should be substituted with lifted val"
        );
    }

    #[test]
    fn test_lift_at_plus_vs_times() {
        // Kill mutants: replace + with * in Expr::lift_at (lines 240, 245)
        // Tests that idx + amount is used, not idx * amount

        // BVar(2) with amount=3: should be 2+3=5, not 2*3=6
        let e = Expr::BVar(2);
        assert_eq!(e.lift_at(0, 3), Expr::BVar(5), "2 + 3 = 5, not 2 * 3 = 6");

        // BVar(3) with amount=2: should be 3+2=5, not 3*2=6
        let e = Expr::BVar(3);
        assert_eq!(e.lift_at(0, 2), Expr::BVar(5), "3 + 2 = 5, not 3 * 2 = 6");

        // BVar(1) with amount=1: + and * give same result (2), so test with larger values
        let e = Expr::BVar(4);
        assert_eq!(e.lift_at(0, 3), Expr::BVar(7), "4 + 3 = 7, not 4 * 3 = 12");
    }

    #[test]
    fn test_has_loose_bvar_or_vs_and() {
        // Kill mutants: replace || with && in Expr::has_loose_bvar_in_range (lines 272, 276, 277)
        // The function should return true if ANY part has a loose bvar (||), not if ALL do (&&)

        // App: only function has loose bvar
        let app_f_loose = Expr::app(Expr::BVar(5), Expr::prop());
        assert!(
            app_f_loose.has_loose_bvar_in_range(0, 10),
            "f has loose bvar"
        );

        // App: only argument has loose bvar
        let app_a_loose = Expr::app(Expr::prop(), Expr::BVar(5));
        assert!(
            app_a_loose.has_loose_bvar_in_range(0, 10),
            "a has loose bvar"
        );

        // Pi: only domain has loose bvar
        let pi_dom_loose = Expr::pi(BinderInfo::Default, Expr::BVar(5), Expr::prop());
        assert!(
            pi_dom_loose.has_loose_bvar_in_range(0, 10),
            "domain has loose bvar"
        );

        // Let: only type has loose bvar
        let let_ty_loose = Expr::let_(Expr::BVar(5), Expr::prop(), Expr::prop());
        assert!(
            let_ty_loose.has_loose_bvar_in_range(0, 10),
            "type has loose bvar"
        );

        // Let: only value has loose bvar
        let let_val_loose = Expr::let_(Expr::prop(), Expr::BVar(5), Expr::prop());
        assert!(
            let_val_loose.has_loose_bvar_in_range(0, 10),
            "value has loose bvar"
        );
    }

    #[test]
    fn test_has_loose_bvar_range_plus_arithmetic() {
        // Kill mutants: replace + with * or - in Expr::has_loose_bvar_in_range (lines 272, 277)
        // Tests that end.saturating_add(1) and start + 1 work correctly

        // Under 1 binder, BVar(0) is bound, BVar(1) is loose
        // With start=0, end=MAX, under binder start becomes 1, end becomes MAX+1 (saturated)
        // BVar(0) < 1, so not in range (bound by lambda)
        let lam_bound = Expr::lam(BinderInfo::Default, Expr::prop(), Expr::BVar(0));
        assert!(
            !lam_bound.has_loose_bvars(),
            "BVar(0) under lambda is bound"
        );

        // BVar(1) under lambda: with start=1, 1 >= 1 and 1 < MAX, so it IS loose
        let lam_loose = Expr::lam(BinderInfo::Default, Expr::prop(), Expr::BVar(1));
        assert!(lam_loose.has_loose_bvars(), "BVar(1) under lambda is loose");

        // Double nested: BVar(1) at depth 2 is bound (bound by inner), BVar(2) is loose
        let inner = Expr::lam(
            BinderInfo::Default,
            Expr::prop(),
            Expr::lam(BinderInfo::Default, Expr::prop(), Expr::BVar(1)),
        );
        assert!(!inner.has_loose_bvars(), "BVar(1) under 2 lambdas is bound");

        // BVar(2) under 2 lambdas IS loose (indices 0 and 1 are bound)
        let inner_loose = Expr::lam(
            BinderInfo::Default,
            Expr::prop(),
            Expr::lam(BinderInfo::Default, Expr::prop(), Expr::BVar(2)),
        );
        assert!(
            inner_loose.has_loose_bvars(),
            "BVar(2) under 2 lambdas is loose"
        );
    }

    #[test]
    fn test_abstract_fvar_at_plus_vs_other() {
        // Kill mutants: replace + with - or * in Expr::abstract_fvar_at (line 317)
        // Tests idx + 1 for shifting BVars

        // BVar(0) at depth 0: 0 >= 0, should become BVar(1) (0 + 1)
        let bvar = Expr::BVar(0);
        let result = bvar.abstract_fvar_at(FVarId(99), 0);
        assert_eq!(result, Expr::BVar(1), "BVar(0) + 1 = BVar(1)");

        // BVar(3) at depth 2: 3 >= 2, should become BVar(4) (3 + 1)
        let bvar = Expr::BVar(3);
        let result = bvar.abstract_fvar_at(FVarId(99), 2);
        assert_eq!(
            result,
            Expr::BVar(4),
            "BVar(3) + 1 = BVar(4), not BVar(2) or BVar(3)"
        );

        // BVar(5) at depth 3: 5 >= 3, should become BVar(6) (5 + 1, not 5 - 1 = 4 or 5 * 1 = 5)
        let bvar = Expr::BVar(5);
        let result = bvar.abstract_fvar_at(FVarId(99), 3);
        assert_eq!(result, Expr::BVar(6), "BVar(5) + 1 = BVar(6)");
    }

    // =========================================================================
    // Targeted Mutation Kill Tests - depth+1 vs depth*1 vs depth-1
    // =========================================================================

    #[test]
    fn test_lift_at_binder_depth_increment() {
        // Kill mutants at lines 240, 245: replace + with * in body.lift_at(start + 1, amount)
        // When start=0, start+1=1 vs start*1=0 behaves differently

        // Test: lift λ x. BVar(0) by 5
        // The inner BVar(0) is at depth start=0 in the outer expression
        // Under the lambda, start becomes 0+1=1
        // BVar(0) < 1, so NO lift (it's bound by the lambda)
        // If start*1=0 instead, BVar(0) >= 0, it WOULD be lifted (wrong!)
        let lam = Expr::lam(BinderInfo::Default, Expr::prop(), Expr::BVar(0));
        let result = lam.lift(5);
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

        // Test: lift λ x. BVar(1) by 5
        // Under lambda, start becomes 1. BVar(1) >= 1, so lift to BVar(6)
        let lam = Expr::lam(BinderInfo::Default, Expr::prop(), Expr::BVar(1));
        let result = lam.lift(5);
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

        // Test: lift (Π x: Prop. BVar(0)) by 3
        // Pi also increments depth. BVar(0) should NOT be lifted.
        let pi = Expr::pi(BinderInfo::Default, Expr::prop(), Expr::BVar(0));
        let result = pi.lift(3);
        match &result {
            Expr::Pi(_, _, body) => {
                assert_eq!(
                    body.as_ref(),
                    &Expr::BVar(0),
                    "BVar(0) under Pi should NOT be lifted (bound)"
                );
            }
            _ => panic!("Expected Pi"),
        }

        // Test: let x = Prop in BVar(0) lifted by 3
        // Let binds the body at +1 depth
        let let_expr = Expr::let_(Expr::prop(), Expr::prop(), Expr::BVar(0));
        let result = let_expr.lift(3);
        match &result {
            Expr::Let(_, _, body) => {
                assert_eq!(
                    body.as_ref(),
                    &Expr::BVar(0),
                    "BVar(0) in let body should NOT be lifted (bound)"
                );
            }
            _ => panic!("Expected Let"),
        }

        // Test Pi body lift with start > 0
        // Π x. BVar(2) lifted starting at cutoff 1
        // Under Pi, cutoff becomes 1+1=2. BVar(2) >= 2, so lifted.
        let pi = Expr::pi(BinderInfo::Default, Expr::prop(), Expr::BVar(2));
        let result = pi.lift_at(1, 3);
        match &result {
            Expr::Pi(_, _, body) => {
                // Under binder, cutoff is 1+1=2. BVar(2) >= 2, so add 3 = BVar(5)
                assert_eq!(
                    body.as_ref(),
                    &Expr::BVar(5),
                    "BVar(2) should be lifted to BVar(5)"
                );
            }
            _ => panic!("Expected Pi"),
        }
    }

    #[test]
    fn test_has_loose_bvar_nested_depth_arithmetic() {
        // Kill mutants at lines 272, 277: replace + with * or - in nested binder checks

        // λ x. Π y. BVar(1)
        // BVar(1) at depth 2 refers to x (index 1), so is BOUND, NOT loose
        // Range check: start=0, end=MAX
        // Under first λ: start=1
        // Under Π: start=2
        // BVar(1) >= 2? NO, so NOT in loose range
        let inner_pi = Expr::pi(BinderInfo::Default, Expr::prop(), Expr::BVar(1));
        let outer_lam = Expr::lam(BinderInfo::Default, Expr::prop(), inner_pi);
        assert!(
            !outer_lam.has_loose_bvars(),
            "BVar(1) under 2 binders refers to outer binder, NOT loose"
        );

        // λ x. Π y. BVar(2)
        // BVar(2) at depth 2 is exactly at the boundary
        // Under 2 binders, start=2. BVar(2) >= 2? YES, so IS loose
        let inner_pi = Expr::pi(BinderInfo::Default, Expr::prop(), Expr::BVar(2));
        let outer_lam = Expr::lam(BinderInfo::Default, Expr::prop(), inner_pi);
        assert!(
            outer_lam.has_loose_bvars(),
            "BVar(2) under 2 binders IS loose (refers outside)"
        );

        // Let with nested structures
        // let x = BVar(5) in λ y. BVar(1)
        // The value BVar(5) is loose (not under any extra binder)
        // The body λ y. BVar(1) has BVar(1) under 2 total binders (let + lambda)
        let inner_lam = Expr::lam(BinderInfo::Default, Expr::prop(), Expr::BVar(1));
        let let_expr = Expr::let_(Expr::prop(), Expr::BVar(5), inner_lam);
        assert!(
            let_expr.has_loose_bvars(),
            "let with loose BVar(5) in value is loose"
        );

        // let x = Prop in λ y. BVar(2)
        // Under let (start=1), then under λ (start=2)
        // BVar(2) >= 2, so IS loose
        let inner_lam = Expr::lam(BinderInfo::Default, Expr::prop(), Expr::BVar(2));
        let let_expr = Expr::let_(Expr::prop(), Expr::prop(), inner_lam);
        assert!(
            let_expr.has_loose_bvars(),
            "BVar(2) under let+lambda IS loose"
        );

        // let x = Prop in λ y. BVar(1)
        // BVar(1) at depth 2 (under let + lambda) refers to let-bound x
        // 1 < 2, so NOT loose
        let inner_lam = Expr::lam(BinderInfo::Default, Expr::prop(), Expr::BVar(1));
        let let_expr = Expr::let_(Expr::prop(), Expr::prop(), inner_lam);
        assert!(
            !let_expr.has_loose_bvars(),
            "BVar(1) under let+lambda refers to let binding, NOT loose"
        );
    }

    #[test]
    fn test_abstract_fvar_nested_depth_plus_one() {
        // Kill mutants at line 317: replace depth + 1 with depth - 1 or depth * 1

        // λ x. (FVar(42)) should become λ x. BVar(1)
        // Under λ, depth becomes 0+1=1, so FVar -> BVar(1)
        let lam = Expr::lam(BinderInfo::Default, Expr::prop(), Expr::fvar(FVarId(42)));
        let result = lam.abstract_fvar(FVarId(42));
        match &result {
            Expr::Lam(_, _, body) => {
                assert_eq!(
                    body.as_ref(),
                    &Expr::BVar(1),
                    "FVar under lambda becomes BVar(1), not BVar(0) or BVar(-1)"
                );
            }
            _ => panic!("Expected Lam"),
        }

        // Π x. (FVar(42)) should become Π x. BVar(1)
        let pi = Expr::pi(BinderInfo::Default, Expr::prop(), Expr::fvar(FVarId(42)));
        let result = pi.abstract_fvar(FVarId(42));
        match &result {
            Expr::Pi(_, _, body) => {
                assert_eq!(
                    body.as_ref(),
                    &Expr::BVar(1),
                    "FVar under Pi becomes BVar(1)"
                );
            }
            _ => panic!("Expected Pi"),
        }

        // let x = FVar(42) in FVar(42)
        // Value: at depth 0, FVar -> BVar(0)
        // Body: at depth 0+1=1, FVar -> BVar(1)
        let let_expr = Expr::let_(Expr::prop(), Expr::fvar(FVarId(42)), Expr::fvar(FVarId(42)));
        let result = let_expr.abstract_fvar(FVarId(42));
        match &result {
            Expr::Let(_, val, body) => {
                assert_eq!(
                    val.as_ref(),
                    &Expr::BVar(0),
                    "FVar in let value becomes BVar(0)"
                );
                assert_eq!(
                    body.as_ref(),
                    &Expr::BVar(1),
                    "FVar in let body becomes BVar(1)"
                );
            }
            _ => panic!("Expected Let"),
        }

        // Triple nested: λ x. λ y. λ z. FVar(42)
        // At innermost level, depth = 3, so FVar -> BVar(3)
        let inner = Expr::fvar(FVarId(42));
        let l1 = Expr::lam(BinderInfo::Default, Expr::prop(), inner);
        let l2 = Expr::lam(BinderInfo::Default, Expr::prop(), l1);
        let l3 = Expr::lam(BinderInfo::Default, Expr::prop(), l2);
        let result = l3.abstract_fvar(FVarId(42));

        // Navigate to innermost body
        let body1 = match &result {
            Expr::Lam(_, _, b) => b.as_ref(),
            _ => panic!("Expected Lam"),
        };
        let body2 = match body1 {
            Expr::Lam(_, _, b) => b.as_ref(),
            _ => panic!("Expected Lam"),
        };
        let body3 = match body2 {
            Expr::Lam(_, _, b) => b.as_ref(),
            _ => panic!("Expected Lam"),
        };
        assert_eq!(
            body3,
            &Expr::BVar(3),
            "FVar under 3 lambdas becomes BVar(3)"
        );
    }

    #[test]
    fn test_instantiate_at_gt_vs_gte() {
        // Kill mutant at line 173: replace > with >= in instantiate_at
        // When idx == depth, we substitute. When idx > depth, we decrement.
        // With >=, idx == depth would ALSO decrement (wrong!)

        // BVar(0) at depth 0: idx == depth, should SUBSTITUTE
        let body = Expr::BVar(0);
        let val = Expr::type_();
        let result = body.instantiate_at(&val, 0);
        assert_eq!(
            result,
            Expr::type_(),
            "BVar(0) at depth 0: == case, should substitute"
        );

        // BVar(1) at depth 0: idx > depth, should DECREMENT to BVar(0)
        let body = Expr::BVar(1);
        let result = body.instantiate_at(&val, 0);
        assert_eq!(
            result,
            Expr::BVar(0),
            "BVar(1) at depth 0: > case, should decrement"
        );

        // BVar(1) at depth 1: idx == depth, should SUBSTITUTE with val.lift(1)
        let body = Expr::BVar(1);
        let result = body.instantiate_at(&val, 1);
        // val is Type, which has no loose bvars, so lift(1) = Type
        assert_eq!(
            result,
            Expr::type_(),
            "BVar(1) at depth 1: == case, should substitute"
        );

        // BVar(2) at depth 1: idx > depth, should decrement to BVar(1)
        let body = Expr::BVar(2);
        let result = body.instantiate_at(&val, 1);
        assert_eq!(
            result,
            Expr::BVar(1),
            "BVar(2) at depth 1: > case, should decrement"
        );

        // Nested: λ x. BVar(1) instantiated with Type
        // Body BVar(1) is at depth 1, idx 1 == depth 1, so substitute
        let lam = Expr::lam(BinderInfo::Default, Expr::prop(), Expr::BVar(1));
        let result = lam.instantiate(&val);
        match &result {
            Expr::Lam(_, _, body) => {
                // BVar(1) at depth 1 gets substituted with Type.lift(1) = Type
                assert_eq!(
                    body.as_ref(),
                    &Expr::type_(),
                    "BVar(1) under lambda at depth 0 refers to substitution target"
                );
            }
            _ => panic!("Expected Lam"),
        }
    }

    // =========================================================================
    // Kill: expr.rs:196:57 - Let body in instantiate_at (depth + 1)
    // =========================================================================
    #[test]
    fn test_instantiate_at_let_body_depth() {
        // This tests that the Let body correctly increments depth by 1
        // Mutation: depth + 1 -> depth - 1 or depth * 1 should fail
        let val = Expr::type_();

        // let x = Prop in BVar(1) - BVar(1) at depth 1 should be substituted
        // If depth is wrong (e.g., depth - 1 = -1 = u32::MAX), this breaks
        // If depth is wrong (e.g., depth * 1 = 0), BVar(1) > 0 so gets decremented
        let let_expr = Expr::let_(Expr::prop(), Expr::prop(), Expr::BVar(1));
        let result = let_expr.instantiate(&val);
        match result {
            Expr::Let(_, _, body) => {
                // BVar(1) at depth 1: 1 == 1, so substitute with val.lift(1) = Type
                assert_eq!(
                    body.as_ref(),
                    &Expr::type_(),
                    "BVar(1) in let body should be substituted at depth 1"
                );
            }
            _ => panic!("Expected Let"),
        }

        // let x = Prop in BVar(0) - BVar(0) at depth 1 is the let-bound variable
        // Should NOT be substituted (0 < 1)
        let let_expr = Expr::let_(Expr::prop(), Expr::prop(), Expr::BVar(0));
        let result = let_expr.instantiate(&val);
        match result {
            Expr::Let(_, _, body) => {
                assert_eq!(
                    body.as_ref(),
                    &Expr::BVar(0),
                    "BVar(0) in let body refers to let binding, not substituted"
                );
            }
            _ => panic!("Expected Let"),
        }

        // let x = Prop in BVar(2) - BVar(2) at depth 1 should decrement to BVar(1)
        let let_expr = Expr::let_(Expr::prop(), Expr::prop(), Expr::BVar(2));
        let result = let_expr.instantiate(&val);
        match result {
            Expr::Let(_, _, body) => {
                assert_eq!(
                    body.as_ref(),
                    &Expr::BVar(1),
                    "BVar(2) in let body should decrement to BVar(1)"
                );
            }
            _ => panic!("Expected Let"),
        }
    }

    // =========================================================================
    // Kill: expr.rs:239:45 - Pi body in lift_at (start + 1)
    // =========================================================================
    #[test]
    fn test_lift_at_pi_body_start() {
        // This tests that the Pi body correctly increments start by 1
        // Mutation: start + 1 -> start - 1 should fail

        // π x : Prop . BVar(1) - BVar(1) is outside the pi (refers to external)
        // When lifting from start=0 by amount=2, the body is processed at start=1
        // BVar(1) >= 1, so it should become BVar(3)
        let pi = Expr::pi(BinderInfo::Default, Expr::prop(), Expr::BVar(1));
        let result = pi.lift(2);
        match result {
            Expr::Pi(_, _, body) => {
                // At start=1, BVar(1) >= 1, so lift by 2: BVar(3)
                assert_eq!(
                    body.as_ref(),
                    &Expr::BVar(3),
                    "BVar(1) in pi body should lift to BVar(3)"
                );
            }
            _ => panic!("Expected Pi"),
        }

        // π x : Prop . BVar(0) - BVar(0) is the pi-bound variable
        // At start=1: BVar(0) < 1, should NOT be lifted
        let pi = Expr::pi(BinderInfo::Default, Expr::prop(), Expr::BVar(0));
        let result = pi.lift(2);
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
    // Kill: expr.rs:244:45 - Let body in lift_at (start + 1)
    // =========================================================================
    #[test]
    fn test_lift_at_let_body_start() {
        // This tests that the Let body correctly increments start by 1
        // Mutation: start + 1 -> start - 1 should fail

        // let x = Prop in BVar(1) - BVar(1) refers outside the let
        // When lifting from start=0 by amount=3, the body is processed at start=1
        // BVar(1) >= 1, so it should become BVar(4)
        let let_expr = Expr::let_(Expr::prop(), Expr::prop(), Expr::BVar(1));
        let result = let_expr.lift(3);
        match result {
            Expr::Let(_, _, body) => {
                assert_eq!(
                    body.as_ref(),
                    &Expr::BVar(4),
                    "BVar(1) in let body should lift to BVar(4)"
                );
            }
            _ => panic!("Expected Let"),
        }

        // let x = Prop in BVar(0) - BVar(0) is the let-bound variable
        // At start=1: BVar(0) < 1, should NOT be lifted
        let let_expr = Expr::let_(Expr::prop(), Expr::prop(), Expr::BVar(0));
        let result = let_expr.lift(3);
        match result {
            Expr::Let(_, _, body) => {
                assert_eq!(
                    body.as_ref(),
                    &Expr::BVar(0),
                    "BVar(0) in let body should not be lifted"
                );
            }
            _ => panic!("Expected Let"),
        }

        // Critical case: let x = Prop in BVar(2)
        // If mutation is start - 1 = 0 - 1 = u32::MAX, this would break
        // At start=1: BVar(2) >= 1, lift by 3: BVar(5)
        let let_expr = Expr::let_(Expr::prop(), Expr::prop(), Expr::BVar(2));
        let result = let_expr.lift(3);
        match result {
            Expr::Let(_, _, body) => {
                assert_eq!(
                    body.as_ref(),
                    &Expr::BVar(5),
                    "BVar(2) in let body should lift to BVar(5)"
                );
            }
            _ => panic!("Expected Let"),
        }
    }

    // =========================================================================
    // lean4lean Theorem Coverage: Lift-Instantiate Commutation
    // Reference: Theory/VExpr.lean - lift_instN_lo, lift_inst_hi, inst_liftN
    // =========================================================================

    #[test]
    fn test_lift_inst_commutation_lo() {
        // lean4lean theorem: lift_instN_lo
        // For e with no loose bvars < lo, and lo <= k:
        //   lift(inst(e, v, lo), k, n) = inst(lift(e, k+1, n), lift(v, k, n), lo)
        //
        // Simplified case where lo = 0 (most common):
        //   lift(inst(e, v), k, n) = inst(lift(e, k+1, n), lift(v, k, n))

        // Case 1: Simple substitution
        // e = BVar(0), v = Prop
        // inst(BVar(0), Prop) = Prop
        // lift(Prop, 0, 5) = Prop (Prop has no BVars)
        // LHS = lift(inst(BVar(0), Prop), 0, 5) = lift(Prop, 0, 5) = Prop
        //
        // lift(BVar(0), 1, 5) = BVar(0) (0 < 1, not lifted)
        // lift(Prop, 0, 5) = Prop
        // inst(BVar(0), Prop) = Prop
        // RHS = inst(lift(BVar(0), 1, 5), lift(Prop, 0, 5)) = inst(BVar(0), Prop) = Prop
        let e = Expr::BVar(0);
        let v = Expr::prop();
        let lhs = e.clone().instantiate(&v).lift_at(0, 5);
        let rhs = e.lift_at(1, 5).instantiate(&v.lift_at(0, 5));
        assert_eq!(lhs, rhs, "lift_inst_commutation_lo case 1");

        // Case 2: BVar survives substitution
        // e = BVar(2), v = Prop at depth 0
        // inst(BVar(2), Prop, 0) = BVar(1) (2 > 0, decrement)
        // lift(BVar(1), 0, 5) = BVar(6)
        //
        // lift(BVar(2), 1, 5) = BVar(7) (2 >= 1, add 5)
        // inst(BVar(7), Prop) = BVar(6) (7 > 0, decrement)
        let e = Expr::BVar(2);
        let v = Expr::prop();
        let lhs = e.clone().instantiate(&v).lift_at(0, 5);
        let rhs = e.lift_at(1, 5).instantiate(&v.lift_at(0, 5));
        assert_eq!(lhs, rhs, "lift_inst_commutation_lo case 2");

        // Case 3: Nested expression
        // e = App(BVar(0), BVar(1))
        // v = Prop
        // inst: App(Prop, BVar(0))
        // lift by 3: App(Prop, BVar(3))
        let e = Expr::app(Expr::BVar(0), Expr::BVar(1));
        let v = Expr::prop();
        let lhs = e.clone().instantiate(&v).lift_at(0, 3);
        let rhs = e.lift_at(1, 3).instantiate(&v.lift_at(0, 3));
        assert_eq!(lhs, rhs, "lift_inst_commutation_lo case 3");

        // Case 4: Lambda expression
        // e = λ x: Prop. BVar(1) (BVar(1) refers to outer var)
        // v = Type
        // inst(λ x. BVar(1), Type) at depth 0:
        //   body: inst(BVar(1), Type, 1) = Type (1 == 1, substitute)
        // Result: λ x: Prop. Type
        let e = Expr::lam(BinderInfo::Default, Expr::prop(), Expr::BVar(1));
        let v = Expr::type_();
        let lhs = e.clone().instantiate(&v).lift_at(0, 2);
        let rhs = e.lift_at(1, 2).instantiate(&v.lift_at(0, 2));
        assert_eq!(lhs, rhs, "lift_inst_commutation_lo case 4 - lambda");
    }

    #[test]
    fn test_lift_inst_commutation_hi() {
        // lean4lean theorem: lift_inst_hi (when k < lo)
        // For k < lo:
        //   lift(inst(e, v, lo), k, n) = inst(lift(e, k, n), v, lo + n)
        //
        // This handles the case where we lift below the substitution point.

        // Case: e = BVar(2), substitute at lo=1, lift at k=0
        // k=0 < lo=1, so use lift_inst_hi formula
        // inst(BVar(2), v, 1) = BVar(1) (2 > 1, decrement)
        // lift(BVar(1), 0, 3) = BVar(4)
        //
        // lift(BVar(2), 0, 3) = BVar(5)
        // inst(BVar(5), v, 4) = BVar(4) (5 > 4, decrement)
        let e = Expr::BVar(2);
        let v = Expr::prop();
        let lhs = e.clone().instantiate_at(&v, 1).lift_at(0, 3);
        let rhs = e.lift_at(0, 3).instantiate_at(&v, 4); // lo + n = 1 + 3 = 4
        assert_eq!(lhs, rhs, "lift_inst_commutation_hi");
    }

    #[test]
    fn test_inst_lift_identity() {
        // lean4lean theorem: inst_liftN / inst_lift
        // For closed v (no loose bvars):
        //   inst(lift(e, 0, 1), v, 0) = e
        //
        // Lifting by 1 then instantiating at 0 is identity for closed v.

        // Case 1: Simple BVar
        // lift(BVar(0), 0, 1) = BVar(1)
        // inst(BVar(1), Prop, 0) = BVar(0) (1 > 0, decrement)
        let e = Expr::BVar(0);
        let v = Expr::prop(); // closed
        let result = e.clone().lift(1).instantiate(&v);
        assert_eq!(result, e, "inst_lift_identity for BVar(0)");

        // Case 2: Higher BVar
        // lift(BVar(5), 0, 1) = BVar(6)
        // inst(BVar(6), Prop, 0) = BVar(5)
        let e = Expr::BVar(5);
        let result = e.clone().lift(1).instantiate(&v);
        assert_eq!(result, e, "inst_lift_identity for BVar(5)");

        // Case 3: Nested expression
        let e = Expr::app(Expr::BVar(0), Expr::app(Expr::BVar(1), Expr::prop()));
        let result = e.clone().lift(1).instantiate(&v);
        assert_eq!(result, e, "inst_lift_identity for nested App");

        // Case 4: Lambda
        // lift(λ x. BVar(1), 0, 1) = λ x. BVar(2)
        // inst at depth 0, body at depth 1: inst(BVar(2), Prop, 1) = BVar(1)
        // Result: λ x. BVar(1)
        let e = Expr::lam(BinderInfo::Default, Expr::prop(), Expr::BVar(1));
        let result = e.clone().lift(1).instantiate(&v);
        assert_eq!(result, e, "inst_lift_identity for lambda");
    }

    // =========================================================================
    // MData Tests
    // =========================================================================

    #[test]
    fn test_mdata_basic() {
        // Create metadata with a simple key-value pair
        let metadata: MDataMap = vec![(Name::from_string("key"), MDataValue::Bool(true))];
        let inner = Expr::prop();
        let mdata = Expr::mdata(metadata.clone(), inner.clone());

        // Check construction
        match &mdata {
            Expr::MData(m, i) => {
                assert_eq!(m, &metadata);
                assert_eq!(i.as_ref(), &inner);
            }
            _ => panic!("Expected MData"),
        }
    }

    #[test]
    fn test_mdata_strip() {
        // strip_mdata should recursively remove MData wrappers
        let inner = Expr::type_();
        let mdata1 = Expr::mdata(vec![], inner.clone());
        let mdata2 = Expr::mdata(vec![], mdata1);

        // strip_mdata should return the innermost non-MData expression
        assert_eq!(mdata2.strip_mdata(), &inner);
    }

    #[test]
    fn test_mdata_instantiate() {
        // MData should pass through instantiate
        let metadata: MDataMap = vec![];
        let inner = Expr::BVar(0);
        let mdata = Expr::mdata(metadata.clone(), inner);

        let val = Expr::prop();
        let result = mdata.instantiate(&val);

        // Should be MData wrapping the instantiated result
        match result {
            Expr::MData(_, inner) => {
                assert_eq!(inner.as_ref(), &Expr::prop());
            }
            _ => panic!("Expected MData after instantiate"),
        }
    }

    #[test]
    fn test_mdata_lift() {
        // MData should pass through lift
        let inner = Expr::BVar(0);
        let mdata = Expr::mdata(vec![], inner);

        let result = mdata.lift(1);

        match result {
            Expr::MData(_, inner) => {
                assert_eq!(inner.as_ref(), &Expr::BVar(1));
            }
            _ => panic!("Expected MData after lift"),
        }
    }

    #[test]
    fn test_mdata_has_loose_bvars() {
        // MData should check inner for loose bvars
        let inner_with_bvar = Expr::BVar(0);
        let mdata_with_bvar = Expr::mdata(vec![], inner_with_bvar);
        assert!(mdata_with_bvar.has_loose_bvars());

        let inner_without_bvar = Expr::prop();
        let mdata_without_bvar = Expr::mdata(vec![], inner_without_bvar);
        assert!(!mdata_without_bvar.has_loose_bvars());
    }

    #[test]
    fn test_mdata_level_params() {
        // MData should propagate level parameter instantiation
        let u = Name::from_string("u");
        let inner = Expr::Sort(Level::param(u.clone()));
        let mdata = Expr::mdata(vec![], inner);

        let subst = vec![(u, Level::succ(Level::zero()))];
        let result = mdata.instantiate_level_params(&subst);

        match result {
            Expr::MData(_, inner) => {
                assert_eq!(inner.as_ref(), &Expr::Sort(Level::succ(Level::zero())));
            }
            _ => panic!("Expected MData after level param instantiation"),
        }
    }

    // =========================================================================
    // Tests for subst_fvar
    // =========================================================================

    #[test]
    fn test_subst_fvar_basic() {
        // Substituting FVar(42) with Prop should replace it
        let fvar = Expr::fvar(FVarId(42));
        let result = fvar.subst_fvar(FVarId(42), &Expr::prop());
        assert_eq!(result, Expr::prop());

        // Different FVar should not be replaced
        let fvar = Expr::fvar(FVarId(42));
        let result = fvar.subst_fvar(FVarId(99), &Expr::prop());
        assert_eq!(result, Expr::fvar(FVarId(42)));
    }

    #[test]
    fn test_subst_fvar_unchanged() {
        // BVar should not be affected
        let bvar = Expr::BVar(5);
        let result = bvar.subst_fvar(FVarId(42), &Expr::prop());
        assert_eq!(result, Expr::BVar(5));

        // Sort should not be affected
        let sort = Expr::type_();
        let result = sort.subst_fvar(FVarId(42), &Expr::prop());
        assert_eq!(result, Expr::type_());

        // Const should not be affected
        let c = Expr::Const(Name::from_string("Nat"), LevelVec::new());
        let result = c.subst_fvar(FVarId(42), &Expr::prop());
        assert_eq!(result, Expr::Const(Name::from_string("Nat"), LevelVec::new()));

        // Literal should not be affected
        let lit = Expr::nat_lit(123);
        let result = lit.subst_fvar(FVarId(42), &Expr::prop());
        assert_eq!(result, Expr::nat_lit(123));
    }

    #[test]
    fn test_subst_fvar_in_app() {
        // Substitute FVar in function position
        let app = Expr::app(Expr::fvar(FVarId(42)), Expr::type_());
        let result = app.subst_fvar(FVarId(42), &Expr::prop());
        assert_eq!(result, Expr::app(Expr::prop(), Expr::type_()));

        // Substitute FVar in argument position
        let app = Expr::app(Expr::type_(), Expr::fvar(FVarId(42)));
        let result = app.subst_fvar(FVarId(42), &Expr::prop());
        assert_eq!(result, Expr::app(Expr::type_(), Expr::prop()));

        // Substitute FVar in both positions
        let app = Expr::app(Expr::fvar(FVarId(42)), Expr::fvar(FVarId(42)));
        let result = app.subst_fvar(FVarId(42), &Expr::prop());
        assert_eq!(result, Expr::app(Expr::prop(), Expr::prop()));
    }

    #[test]
    fn test_subst_fvar_in_binders() {
        // Lambda: substitute in type and body
        let lam = Expr::lam(BinderInfo::Default, Expr::fvar(FVarId(42)), Expr::fvar(FVarId(42)));
        let result = lam.subst_fvar(FVarId(42), &Expr::type_());
        match result {
            Expr::Lam(_, ty, body) => {
                assert_eq!(ty.as_ref(), &Expr::type_());
                assert_eq!(body.as_ref(), &Expr::type_());
            }
            _ => panic!("Expected Lam"),
        }

        // Pi: substitute in type and body
        let pi = Expr::pi(BinderInfo::Default, Expr::fvar(FVarId(42)), Expr::fvar(FVarId(42)));
        let result = pi.subst_fvar(FVarId(42), &Expr::type_());
        match result {
            Expr::Pi(_, ty, body) => {
                assert_eq!(ty.as_ref(), &Expr::type_());
                assert_eq!(body.as_ref(), &Expr::type_());
            }
            _ => panic!("Expected Pi"),
        }

        // Let: substitute in type, value, and body
        let let_expr = Expr::Let(
            Arc::new(Expr::fvar(FVarId(42))),
            Arc::new(Expr::fvar(FVarId(42))),
            Arc::new(Expr::fvar(FVarId(42))),
        );
        let result = let_expr.subst_fvar(FVarId(42), &Expr::prop());
        match result {
            Expr::Let(ty, val, body) => {
                assert_eq!(ty.as_ref(), &Expr::prop());
                assert_eq!(val.as_ref(), &Expr::prop());
                assert_eq!(body.as_ref(), &Expr::prop());
            }
            _ => panic!("Expected Let"),
        }
    }

    #[test]
    fn test_subst_fvar_in_proj() {
        // Projection with FVar
        let proj = Expr::Proj(Name::from_string("fst"), 0, Arc::new(Expr::fvar(FVarId(42))));
        let result = proj.subst_fvar(FVarId(42), &Expr::type_());
        match result {
            Expr::Proj(name, idx, e) => {
                assert_eq!(name, Name::from_string("fst"));
                assert_eq!(idx, 0);
                assert_eq!(e.as_ref(), &Expr::type_());
            }
            _ => panic!("Expected Proj"),
        }
    }

    #[test]
    fn test_subst_fvar_in_mdata() {
        // MData should propagate substitution
        let mdata = Expr::mdata(vec![], Expr::fvar(FVarId(42)));
        let result = mdata.subst_fvar(FVarId(42), &Expr::prop());
        match result {
            Expr::MData(_, inner) => {
                assert_eq!(inner.as_ref(), &Expr::prop());
            }
            _ => panic!("Expected MData"),
        }
    }

    #[test]
    fn test_subst_fvar_nested() {
        // Deep nesting: λx. λy. FVar(42)
        let inner = Expr::lam(BinderInfo::Default, Expr::prop(), Expr::fvar(FVarId(42)));
        let outer = Expr::lam(BinderInfo::Default, Expr::prop(), inner);
        let result = outer.subst_fvar(FVarId(42), &Expr::type_());

        match result {
            Expr::Lam(_, _, body) => match body.as_ref() {
                Expr::Lam(_, _, inner_body) => {
                    assert_eq!(inner_body.as_ref(), &Expr::type_());
                }
                _ => panic!("Expected nested Lam"),
            },
            _ => panic!("Expected Lam"),
        }
    }

    // =========================================================================
    // Tests for instantiate_level_params
    // =========================================================================

    #[test]
    fn test_instantiate_level_params_sort() {
        // Sort(u) with u -> Type1 should become Sort(Type1)
        let u = Name::from_string("u");
        let sort = Expr::Sort(Level::param(u.clone()));
        let subst = vec![(u, Level::succ(Level::zero()))];
        let result = sort.instantiate_level_params(&subst);
        assert_eq!(result, Expr::Sort(Level::succ(Level::zero())));
    }

    #[test]
    fn test_instantiate_level_params_const() {
        // Const with level params
        let u = Name::from_string("u");
        let v = Name::from_string("v");
        let levels: LevelVec = smallvec::smallvec![Level::param(u.clone()), Level::param(v.clone())];
        let c = Expr::Const(Name::from_string("List"), levels);

        let subst = vec![
            (u, Level::zero()),
            (v, Level::succ(Level::succ(Level::zero()))),
        ];
        let result = c.instantiate_level_params(&subst);

        match result {
            Expr::Const(name, lvls) => {
                assert_eq!(name, Name::from_string("List"));
                assert_eq!(lvls.len(), 2);
                assert_eq!(lvls[0], Level::zero());
                assert_eq!(lvls[1], Level::succ(Level::succ(Level::zero())));
            }
            _ => panic!("Expected Const"),
        }
    }

    #[test]
    fn test_instantiate_level_params_empty_subst() {
        // Empty substitution should return same expression
        let u = Name::from_string("u");
        let sort = Expr::Sort(Level::param(u));
        let subst: Vec<(Name, Level)> = vec![];
        let result = sort.instantiate_level_params(&subst);
        assert_eq!(result, sort);
    }

    #[test]
    fn test_instantiate_level_params_unchanged() {
        // BVar, FVar, Lit should pass through unchanged
        let u = Name::from_string("u");
        let subst = vec![(u, Level::zero())];

        assert_eq!(Expr::BVar(0).instantiate_level_params(&subst), Expr::BVar(0));
        assert_eq!(
            Expr::fvar(FVarId(42)).instantiate_level_params(&subst),
            Expr::fvar(FVarId(42))
        );
        assert_eq!(
            Expr::nat_lit(123).instantiate_level_params(&subst),
            Expr::nat_lit(123)
        );
    }

    #[test]
    fn test_instantiate_level_params_in_binders() {
        // Lambda with Sort(u) in type and body
        let u = Name::from_string("u");
        let lam = Expr::lam(
            BinderInfo::Default,
            Expr::Sort(Level::param(u.clone())),
            Expr::Sort(Level::param(u.clone())),
        );
        let subst = vec![(u, Level::succ(Level::zero()))];
        let result = lam.instantiate_level_params(&subst);

        match result {
            Expr::Lam(_, ty, body) => {
                assert_eq!(ty.as_ref(), &Expr::Sort(Level::succ(Level::zero())));
                assert_eq!(body.as_ref(), &Expr::Sort(Level::succ(Level::zero())));
            }
            _ => panic!("Expected Lam"),
        }

        // Pi with levels
        let v = Name::from_string("v");
        let pi = Expr::pi(
            BinderInfo::Default,
            Expr::Sort(Level::param(v.clone())),
            Expr::Sort(Level::param(v.clone())),
        );
        let subst = vec![(v, Level::zero())];
        let result = pi.instantiate_level_params(&subst);

        match result {
            Expr::Pi(_, ty, body) => {
                assert_eq!(ty.as_ref(), &Expr::prop());
                assert_eq!(body.as_ref(), &Expr::prop());
            }
            _ => panic!("Expected Pi"),
        }
    }

    #[test]
    fn test_instantiate_level_params_in_app() {
        // App with levels in subexpressions
        let u = Name::from_string("u");
        let app = Expr::app(
            Expr::Sort(Level::param(u.clone())),
            Expr::Sort(Level::param(u.clone())),
        );
        let subst = vec![(u, Level::succ(Level::zero()))];
        let result = app.instantiate_level_params(&subst);

        match result {
            Expr::App(f, a) => {
                assert_eq!(f.as_ref(), &Expr::Sort(Level::succ(Level::zero())));
                assert_eq!(a.as_ref(), &Expr::Sort(Level::succ(Level::zero())));
            }
            _ => panic!("Expected App"),
        }
    }

    #[test]
    fn test_instantiate_level_params_in_let() {
        // Let with levels in all parts
        let u = Name::from_string("u");
        let let_expr = Expr::Let(
            Arc::new(Expr::Sort(Level::param(u.clone()))),
            Arc::new(Expr::Sort(Level::param(u.clone()))),
            Arc::new(Expr::Sort(Level::param(u.clone()))),
        );
        let subst = vec![(u, Level::zero())];
        let result = let_expr.instantiate_level_params(&subst);

        match result {
            Expr::Let(ty, val, body) => {
                assert_eq!(ty.as_ref(), &Expr::prop());
                assert_eq!(val.as_ref(), &Expr::prop());
                assert_eq!(body.as_ref(), &Expr::prop());
            }
            _ => panic!("Expected Let"),
        }
    }

    #[test]
    fn test_instantiate_level_params_in_proj() {
        // Proj with level in inner expression
        let u = Name::from_string("u");
        let proj = Expr::Proj(
            Name::from_string("fst"),
            0,
            Arc::new(Expr::Sort(Level::param(u.clone()))),
        );
        let subst = vec![(u, Level::succ(Level::zero()))];
        let result = proj.instantiate_level_params(&subst);

        match result {
            Expr::Proj(name, idx, e) => {
                assert_eq!(name, Name::from_string("fst"));
                assert_eq!(idx, 0);
                assert_eq!(e.as_ref(), &Expr::Sort(Level::succ(Level::zero())));
            }
            _ => panic!("Expected Proj"),
        }
    }

    #[test]
    fn test_instantiate_level_params_multiple() {
        // Multiple different params
        let u = Name::from_string("u");
        let v = Name::from_string("v");
        let w = Name::from_string("w");

        let app = Expr::app(
            Expr::app(
                Expr::Sort(Level::param(u.clone())),
                Expr::Sort(Level::param(v.clone())),
            ),
            Expr::Sort(Level::param(w.clone())),
        );

        let subst = vec![
            (u, Level::zero()),
            (v, Level::succ(Level::zero())),
            (w, Level::succ(Level::succ(Level::zero()))),
        ];
        let result = app.instantiate_level_params(&subst);

        // Check the result has all three params substituted
        match result {
            Expr::App(f, a) => {
                assert_eq!(a.as_ref(), &Expr::Sort(Level::succ(Level::succ(Level::zero()))));
                match f.as_ref() {
                    Expr::App(f2, a2) => {
                        assert_eq!(f2.as_ref(), &Expr::prop());
                        assert_eq!(a2.as_ref(), &Expr::Sort(Level::succ(Level::zero())));
                    }
                    _ => panic!("Expected nested App"),
                }
            }
            _ => panic!("Expected App"),
        }
    }

    // =========================================================================
    // Tests for strip_mdata
    // =========================================================================

    #[test]
    fn test_strip_mdata_basic() {
        // Non-MData expression returns self
        let prop = Expr::prop();
        assert_eq!(prop.strip_mdata(), &prop);

        let bvar = Expr::BVar(5);
        assert_eq!(bvar.strip_mdata(), &bvar);
    }

    #[test]
    fn test_strip_mdata_single() {
        // Single layer of MData
        let inner = Expr::type_();
        let mdata = Expr::mdata(vec![], inner.clone());
        assert_eq!(mdata.strip_mdata(), &inner);
    }

    #[test]
    fn test_strip_mdata_nested() {
        // Nested MData layers should all be stripped
        let inner = Expr::prop();
        let mdata1 = Expr::mdata(vec![], inner.clone());
        let mdata2 = Expr::mdata(vec![], mdata1);
        let mdata3 = Expr::mdata(vec![], mdata2);
        assert_eq!(mdata3.strip_mdata(), &inner);
    }

    #[test]
    fn test_strip_mdata_with_metadata() {
        // MData with actual metadata
        let inner = Expr::fvar(FVarId(42));
        let metadata = vec![
            (Name::from_string("key1"), MDataValue::Bool(true)),
            (Name::from_string("key2"), MDataValue::Nat(100)),
        ];
        let mdata = Expr::mdata(metadata, inner.clone());
        assert_eq!(mdata.strip_mdata(), &inner);
    }

    #[test]
    fn test_strip_mdata_various_inner() {
        // Test with various inner expression types
        let exprs = vec![
            Expr::BVar(0),
            Expr::fvar(FVarId(1)),
            Expr::type_(),
            Expr::prop(),
            Expr::nat_lit(42),
            Expr::lam(BinderInfo::Default, Expr::prop(), Expr::BVar(0)),
            Expr::pi(BinderInfo::Default, Expr::type_(), Expr::BVar(0)),
            Expr::app(Expr::fvar(FVarId(1)), Expr::fvar(FVarId(2))),
        ];

        for inner in exprs {
            let mdata = Expr::mdata(vec![], inner.clone());
            assert_eq!(
                mdata.strip_mdata(),
                &inner,
                "strip_mdata failed for {:?}",
                inner
            );
        }
    }

    // =========================================================================
    // Tests for get_app_fn and get_app_args
    // =========================================================================

    #[test]
    fn test_get_app_fn_basic() {
        // Non-App returns self
        let prop = Expr::prop();
        assert_eq!(prop.get_app_fn(), &prop);

        let fvar = Expr::fvar(FVarId(42));
        assert_eq!(fvar.get_app_fn(), &fvar);
    }

    #[test]
    fn test_get_app_fn_single() {
        // Single application: f(x) -> f
        let f = Expr::fvar(FVarId(1));
        let x = Expr::fvar(FVarId(2));
        let app = Expr::app(f.clone(), x);
        assert_eq!(app.get_app_fn(), &f);
    }

    #[test]
    fn test_get_app_fn_nested() {
        // Nested applications: f(x)(y)(z) -> f
        let f = Expr::fvar(FVarId(1));
        let x = Expr::fvar(FVarId(2));
        let y = Expr::fvar(FVarId(3));
        let z = Expr::fvar(FVarId(4));
        let app = Expr::app(Expr::app(Expr::app(f.clone(), x), y), z);
        assert_eq!(app.get_app_fn(), &f);
    }

    #[test]
    fn test_get_app_args_empty() {
        // Non-App returns empty vec
        let prop = Expr::prop();
        assert_eq!(prop.get_app_args(), Vec::<&Expr>::new());
    }

    #[test]
    fn test_get_app_args_single() {
        // Single application: f(x) -> [x]
        let f = Expr::fvar(FVarId(1));
        let x = Expr::fvar(FVarId(2));
        let app = Expr::app(f, x.clone());

        let args = app.get_app_args();
        assert_eq!(args.len(), 1);
        assert_eq!(args[0], &x);
    }

    #[test]
    fn test_get_app_args_multiple() {
        // Multiple applications: f(x)(y)(z) -> [x, y, z]
        let f = Expr::fvar(FVarId(1));
        let x = Expr::fvar(FVarId(2));
        let y = Expr::fvar(FVarId(3));
        let z = Expr::fvar(FVarId(4));
        let app = Expr::app(Expr::app(Expr::app(f, x.clone()), y.clone()), z.clone());

        let args = app.get_app_args();
        assert_eq!(args.len(), 3);
        assert_eq!(args[0], &x);
        assert_eq!(args[1], &y);
        assert_eq!(args[2], &z);
    }
}
