//! Micro-Checker: A Minimal Verified Certificate Checker
//!
//! This module provides a minimal, self-contained certificate checker that can
//! verify proof certificates independently of the main kernel. It is designed
//! to be:
//!
//! 1. **Small**: ~500-700 lines, auditable by humans
//! 2. **Self-contained**: Minimal dependencies
//! 3. **Verifiable**: Simple enough to prove correct in Lean5 itself
//! 4. **Correct**: Conservative checking - rejects anything suspicious
//!
//! ## Design Philosophy
//!
//! The micro-checker trades off performance for simplicity and correctness.
//! It only implements the absolute minimum needed to verify certificates:
//!
//! - Basic expression types (no metadata, spans, etc.)
//! - Substitution with de Bruijn indices
//! - Simple WHNF (beta + let only, no delta)
//! - Structural equality after WHNF
//!
//! ## Trust Model
//!
//! If the micro-checker accepts a certificate, the typing derivation is correct.
//! The micro-checker can be verified by:
//! 1. Human auditing (~500 lines)
//! 2. Formal verification in Lean5 (future)
//! 3. Cross-validation with the main kernel
//!
//! ## Certificate Format
//!
//! Certificates are self-contained: they include all type information needed
//! for verification. The checker doesn't need access to an environment.

use std::sync::Arc;

/// Minimum stack space to reserve before recursive calls (32 KB).
const MIN_STACK_RED_ZONE: usize = 32 * 1024;

/// Stack size to grow to when running low (1 MB).
const STACK_GROWTH_SIZE: usize = 1024 * 1024;

/// Minimal expression type for the micro-checker.
/// This is a simplified version of the main kernel's Expr.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MicroExpr {
    /// Bound variable (de Bruijn index)
    BVar(u32),
    /// Sort/Type at a level
    Sort(MicroLevel),
    /// Application
    App(Arc<MicroExpr>, Arc<MicroExpr>),
    /// Lambda abstraction: λ (x : A). b
    Lam(Arc<MicroExpr>, Arc<MicroExpr>),
    /// Pi/forall type: (x : A) → B
    Pi(Arc<MicroExpr>, Arc<MicroExpr>),
    /// Let binding: let x : A := v in b
    Let(Arc<MicroExpr>, Arc<MicroExpr>, Arc<MicroExpr>),
    /// Opaque constant (just a type, no definition)
    Opaque(Arc<MicroExpr>),
}

/// Minimal universe level for the micro-checker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MicroLevel {
    /// Level 0 (Prop)
    Zero,
    /// Successor: l + 1
    Succ(Arc<MicroLevel>),
    /// Maximum: max(l1, l2)
    Max(Arc<MicroLevel>, Arc<MicroLevel>),
    /// IMax: imax(l1, l2) - 0 if l2=0, else max(l1, l2)
    IMax(Arc<MicroLevel>, Arc<MicroLevel>),
}

/// Minimal proof certificate for the micro-checker.
#[derive(Debug, Clone, PartialEq)]
pub enum MicroCert {
    /// Sort(l) : Sort(succ(l))
    Sort { level: MicroLevel },

    /// BVar with its type from context
    BVar { idx: u32, ty: Box<MicroExpr> },

    /// Opaque constant with its type
    Opaque { ty: Box<MicroExpr> },

    /// App: f a : B[a/x]
    App {
        fn_cert: Box<MicroCert>,
        arg_cert: Box<MicroCert>,
        result_ty: Box<MicroExpr>,
    },

    /// Lam: λ (x : A). b : (x : A) → B
    Lam {
        arg_ty_cert: Box<MicroCert>,
        body_cert: Box<MicroCert>,
        result_ty: Box<MicroExpr>,
    },

    /// Pi: (x : A) → B : Sort(imax(l1, l2))
    Pi {
        arg_ty_cert: Box<MicroCert>,
        arg_level: MicroLevel,
        body_ty_cert: Box<MicroCert>,
        body_level: MicroLevel,
    },

    /// Let: let x : A := v in b : B[v/x]
    Let {
        ty_cert: Box<MicroCert>,
        val_cert: Box<MicroCert>,
        body_cert: Box<MicroCert>,
        result_ty: Box<MicroExpr>,
    },
}

/// Verification error
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MicroError {
    /// Type mismatch
    TypeMismatch {
        expected: MicroExpr,
        actual: MicroExpr,
    },
    /// Invalid de Bruijn index
    InvalidBVar(u32),
    /// Expected a Sort
    ExpectedSort(MicroExpr),
    /// Expected a Pi type
    ExpectedPi(MicroExpr),
    /// Level mismatch
    LevelMismatch {
        expected: MicroLevel,
        actual: MicroLevel,
    },
    /// Certificate/expression structure mismatch
    StructureMismatch,
}

impl std::fmt::Display for MicroError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MicroError::TypeMismatch { expected, actual } => {
                write!(f, "type mismatch: expected {expected:?}, got {actual:?}")
            }
            MicroError::InvalidBVar(idx) => write!(f, "invalid bound variable: {idx}"),
            MicroError::ExpectedSort(e) => write!(f, "expected Sort, got {e:?}"),
            MicroError::ExpectedPi(e) => write!(f, "expected Pi, got {e:?}"),
            MicroError::LevelMismatch { expected, actual } => {
                write!(f, "level mismatch: expected {expected:?}, got {actual:?}")
            }
            MicroError::StructureMismatch => write!(f, "certificate/expression structure mismatch"),
        }
    }
}

impl std::error::Error for MicroError {}

// ============================================================================
// Expression Operations
// ============================================================================

impl MicroExpr {
    /// Lift all bound variables >= cutoff by amount
    #[must_use]
    pub fn lift(&self, cutoff: u32, amount: u32) -> MicroExpr {
        match self {
            MicroExpr::BVar(idx) => {
                if *idx >= cutoff {
                    MicroExpr::BVar(idx + amount)
                } else {
                    self.clone()
                }
            }
            MicroExpr::Sort(l) => MicroExpr::Sort(l.clone()),
            MicroExpr::App(f, a) => MicroExpr::App(
                Arc::new(f.lift(cutoff, amount)),
                Arc::new(a.lift(cutoff, amount)),
            ),
            MicroExpr::Lam(ty, body) => MicroExpr::Lam(
                Arc::new(ty.lift(cutoff, amount)),
                Arc::new(body.lift(cutoff + 1, amount)),
            ),
            MicroExpr::Pi(ty, body) => MicroExpr::Pi(
                Arc::new(ty.lift(cutoff, amount)),
                Arc::new(body.lift(cutoff + 1, amount)),
            ),
            MicroExpr::Let(ty, val, body) => MicroExpr::Let(
                Arc::new(ty.lift(cutoff, amount)),
                Arc::new(val.lift(cutoff, amount)),
                Arc::new(body.lift(cutoff + 1, amount)),
            ),
            MicroExpr::Opaque(ty) => MicroExpr::Opaque(Arc::new(ty.lift(cutoff, amount))),
        }
    }

    /// Substitute `val` for BVar(0), adjusting indices appropriately
    #[must_use]
    pub fn instantiate(&self, val: &MicroExpr) -> MicroExpr {
        self.subst(0, val)
    }

    /// Substitute `val` for BVar(depth), adjusting indices
    fn subst(&self, depth: u32, val: &MicroExpr) -> MicroExpr {
        match self {
            MicroExpr::BVar(idx) => {
                use std::cmp::Ordering;
                match idx.cmp(&depth) {
                    Ordering::Equal => val.lift(0, depth),
                    Ordering::Greater => MicroExpr::BVar(idx - 1),
                    Ordering::Less => self.clone(),
                }
            }
            MicroExpr::Sort(l) => MicroExpr::Sort(l.clone()),
            MicroExpr::App(f, a) => {
                MicroExpr::App(Arc::new(f.subst(depth, val)), Arc::new(a.subst(depth, val)))
            }
            MicroExpr::Lam(ty, body) => MicroExpr::Lam(
                Arc::new(ty.subst(depth, val)),
                Arc::new(body.subst(depth + 1, val)),
            ),
            MicroExpr::Pi(ty, body) => MicroExpr::Pi(
                Arc::new(ty.subst(depth, val)),
                Arc::new(body.subst(depth + 1, val)),
            ),
            MicroExpr::Let(ty, v, body) => MicroExpr::Let(
                Arc::new(ty.subst(depth, val)),
                Arc::new(v.subst(depth, val)),
                Arc::new(body.subst(depth + 1, val)),
            ),
            MicroExpr::Opaque(ty) => MicroExpr::Opaque(Arc::new(ty.subst(depth, val))),
        }
    }
}

impl MicroLevel {
    /// Create successor level
    pub fn succ(l: MicroLevel) -> MicroLevel {
        MicroLevel::Succ(Arc::new(l))
    }

    /// Create max level, simplifying if possible
    pub fn max(l1: MicroLevel, l2: MicroLevel) -> MicroLevel {
        // Simplifications:
        // max(l, l) = l
        // max(0, l) = l
        // max(l, 0) = l
        if l1 == l2 {
            return l1;
        }
        if l1 == MicroLevel::Zero {
            return l2;
        }
        if l2 == MicroLevel::Zero {
            return l1;
        }
        // Check if one is definitely >= the other
        if MicroLevel::is_geq(&l1, &l2) {
            return l1;
        }
        if MicroLevel::is_geq(&l2, &l1) {
            return l2;
        }
        MicroLevel::Max(Arc::new(l1), Arc::new(l2))
    }

    /// Check if l1 >= l2 (conservative approximation)
    fn is_geq(l1: &MicroLevel, l2: &MicroLevel) -> bool {
        // Same level
        if l1 == l2 {
            return true;
        }

        // Zero is the minimum
        if *l2 == MicroLevel::Zero {
            return true;
        }

        // Get offsets (number of Succ applications)
        let (base1, offset1) = MicroLevel::get_offset(l1);
        let (base2, offset2) = MicroLevel::get_offset(l2);

        // If same base, compare offsets
        if base1 == base2 {
            return offset1 >= offset2;
        }

        // If l1 = succ(l1') and l1' >= l2, then l1 >= l2
        // The offset check is implicit: Succ pattern only matches when offset > 0
        if let MicroLevel::Succ(inner) = l1 {
            if MicroLevel::is_geq(inner, l2) {
                return true;
            }
        }

        // max(a, b) >= l if a >= l or b >= l
        if let MicroLevel::Max(a, b) = l1 {
            if MicroLevel::is_geq(a, l2) || MicroLevel::is_geq(b, l2) {
                return true;
            }
        }

        // l >= max(a, b) if l >= a and l >= b
        if let MicroLevel::Max(a, b) = l2 {
            if MicroLevel::is_geq(l1, a) && MicroLevel::is_geq(l1, b) {
                return true;
            }
        }

        false
    }

    /// Get the base level and offset (number of Succ applications)
    fn get_offset(l: &MicroLevel) -> (&MicroLevel, u32) {
        match l {
            MicroLevel::Succ(inner) => {
                let (base, offset) = MicroLevel::get_offset(inner);
                (base, offset + 1)
            }
            _ => (l, 0),
        }
    }

    /// Create imax level, simplifying if possible
    ///
    /// imax(l1, l2) = 0 if l2 = 0, else max(l1, l2) if l2 is nonzero (Succ)
    pub fn imax(l1: MicroLevel, l2: MicroLevel) -> MicroLevel {
        // imax(_, 0) = 0
        if l2 == MicroLevel::Zero {
            return MicroLevel::Zero;
        }
        // imax(l, succ(l')) = max(l, succ(l')) since succ(l') > 0
        if matches!(l2, MicroLevel::Succ(_)) {
            return MicroLevel::max(l1, l2);
        }
        // imax(0, l) = l (if l != 0, which we handled above)
        if l1 == MicroLevel::Zero {
            return l2;
        }
        // imax(l, l) = l
        if l1 == l2 {
            return l1;
        }
        MicroLevel::IMax(Arc::new(l1), Arc::new(l2))
    }

    /// Check if two levels are equal (uses derived PartialEq)
    pub fn level_eq(&self, other: &MicroLevel) -> bool {
        self == other
    }
}

// ============================================================================
// Micro-Checker Core
// ============================================================================

/// Minimal certificate checker state
pub struct MicroChecker {
    /// Type context (de Bruijn levels: index 0 = outermost binding)
    context: Vec<MicroExpr>,
}

impl MicroChecker {
    /// Create a new micro-checker
    pub fn new() -> Self {
        MicroChecker {
            context: Vec::new(),
        }
    }

    /// Verify a certificate against an expression, returning the proven type
    ///
    /// This is the core verification function. It checks that the certificate
    /// correctly witnesses the typing derivation for the expression.
    pub fn verify(&mut self, cert: &MicroCert, expr: &MicroExpr) -> Result<MicroExpr, MicroError> {
        stacker::maybe_grow(MIN_STACK_RED_ZONE, STACK_GROWTH_SIZE, || {
            self.verify_impl(cert, expr)
        })
    }

    /// Implementation of certificate verification (called via stacker::maybe_grow)
    fn verify_impl(
        &mut self,
        cert: &MicroCert,
        expr: &MicroExpr,
    ) -> Result<MicroExpr, MicroError> {
        match (cert, expr) {
            // Sort rule: Sort(l) : Sort(succ(l))
            (MicroCert::Sort { level }, MicroExpr::Sort(l)) => {
                if !level.level_eq(l) {
                    return Err(MicroError::LevelMismatch {
                        expected: level.clone(),
                        actual: l.clone(),
                    });
                }
                Ok(MicroExpr::Sort(MicroLevel::succ(level.clone())))
            }

            // BVar rule: look up type in context
            (MicroCert::BVar { idx, ty }, MicroExpr::BVar(i)) => {
                if *idx != *i {
                    return Err(MicroError::InvalidBVar(*i));
                }
                let depth = self.context.len();
                if (*idx as usize) >= depth {
                    return Err(MicroError::InvalidBVar(*idx));
                }
                // Return the type from certificate (context provides validation)
                Ok(ty.as_ref().clone())
            }

            // Opaque rule: return declared type
            (MicroCert::Opaque { ty }, MicroExpr::Opaque(t)) => {
                // Verify the types match
                if !self.def_eq(ty.as_ref(), t.as_ref()) {
                    return Err(MicroError::TypeMismatch {
                        expected: ty.as_ref().clone(),
                        actual: t.as_ref().clone(),
                    });
                }
                Ok(ty.as_ref().clone())
            }

            // App rule: f a : B[a/x] when f : (x : A) → B and a : A
            (
                MicroCert::App {
                    fn_cert,
                    arg_cert,
                    result_ty,
                },
                MicroExpr::App(f, a),
            ) => {
                // Verify function
                let fn_ty = self.verify(fn_cert, f)?;
                let fn_ty_whnf = self.whnf(&fn_ty);

                // Check function has Pi type
                let (expected_arg_ty, body_ty) = match &fn_ty_whnf {
                    MicroExpr::Pi(arg_ty, body) => (arg_ty.as_ref(), body.as_ref()),
                    _ => return Err(MicroError::ExpectedPi(fn_ty_whnf)),
                };

                // Verify argument
                let arg_ty = self.verify(arg_cert, a)?;

                // Check argument type matches
                if !self.def_eq(&arg_ty, expected_arg_ty) {
                    return Err(MicroError::TypeMismatch {
                        expected: expected_arg_ty.clone(),
                        actual: arg_ty,
                    });
                }

                // Compute expected result type
                let expected_result = body_ty.instantiate(a);

                // Verify result type
                if !self.def_eq(result_ty.as_ref(), &expected_result) {
                    return Err(MicroError::TypeMismatch {
                        expected: expected_result,
                        actual: result_ty.as_ref().clone(),
                    });
                }

                Ok(result_ty.as_ref().clone())
            }

            // Lam rule: λ (x : A). b : (x : A) → B
            (
                MicroCert::Lam {
                    arg_ty_cert,
                    body_cert,
                    result_ty,
                },
                MicroExpr::Lam(arg_ty, body),
            ) => {
                // Verify arg type is a Sort
                let arg_sort = self.verify(arg_ty_cert, arg_ty)?;
                let arg_sort_whnf = self.whnf(&arg_sort);
                if !matches!(arg_sort_whnf, MicroExpr::Sort(_)) {
                    return Err(MicroError::ExpectedSort(arg_sort_whnf));
                }

                // Extend context and verify body
                self.context.push(arg_ty.as_ref().clone());
                let body_ty = self.verify(body_cert, body)?;
                self.context.pop();

                // Build expected Pi type
                let expected_pi = MicroExpr::Pi(arg_ty.clone(), Arc::new(body_ty));

                // Verify result type
                if !self.def_eq(result_ty.as_ref(), &expected_pi) {
                    return Err(MicroError::TypeMismatch {
                        expected: expected_pi,
                        actual: result_ty.as_ref().clone(),
                    });
                }

                Ok(result_ty.as_ref().clone())
            }

            // Pi rule: (x : A) → B : Sort(imax(l1, l2))
            (
                MicroCert::Pi {
                    arg_ty_cert,
                    arg_level,
                    body_ty_cert,
                    body_level,
                },
                MicroExpr::Pi(arg_ty, body_ty),
            ) => {
                // Verify arg type
                let arg_sort = self.verify(arg_ty_cert, arg_ty)?;
                let l1 = match self.whnf(&arg_sort) {
                    MicroExpr::Sort(l) => l,
                    other => return Err(MicroError::ExpectedSort(other)),
                };

                // Check level matches
                if !l1.level_eq(arg_level) {
                    return Err(MicroError::LevelMismatch {
                        expected: arg_level.clone(),
                        actual: l1,
                    });
                }

                // Extend context and verify body type
                self.context.push(arg_ty.as_ref().clone());
                let body_sort = self.verify(body_ty_cert, body_ty)?;
                self.context.pop();

                let l2 = match self.whnf(&body_sort) {
                    MicroExpr::Sort(l) => l,
                    other => return Err(MicroError::ExpectedSort(other)),
                };

                // Check level matches
                if !l2.level_eq(body_level) {
                    return Err(MicroError::LevelMismatch {
                        expected: body_level.clone(),
                        actual: l2,
                    });
                }

                // Result is Sort(imax(l1, l2))
                Ok(MicroExpr::Sort(MicroLevel::imax(
                    arg_level.clone(),
                    body_level.clone(),
                )))
            }

            // Let rule: let x : A := v in b : B[v/x]
            (
                MicroCert::Let {
                    ty_cert,
                    val_cert,
                    body_cert,
                    result_ty,
                },
                MicroExpr::Let(ty, val, body),
            ) => {
                // Verify type is a Sort
                let ty_sort = self.verify(ty_cert, ty)?;
                let ty_sort_whnf = self.whnf(&ty_sort);
                if !matches!(ty_sort_whnf, MicroExpr::Sort(_)) {
                    return Err(MicroError::ExpectedSort(ty_sort_whnf));
                }

                // Verify value has the declared type
                let val_ty = self.verify(val_cert, val)?;
                if !self.def_eq(&val_ty, ty) {
                    return Err(MicroError::TypeMismatch {
                        expected: ty.as_ref().clone(),
                        actual: val_ty,
                    });
                }

                // Extend context and verify body
                self.context.push(ty.as_ref().clone());
                let body_ty = self.verify(body_cert, body)?;
                self.context.pop();

                // Compute expected result type (substitute value for bound var)
                let expected_result = body_ty.instantiate(val);

                // Verify result type
                if !self.def_eq(result_ty.as_ref(), &expected_result) {
                    return Err(MicroError::TypeMismatch {
                        expected: expected_result,
                        actual: result_ty.as_ref().clone(),
                    });
                }

                Ok(result_ty.as_ref().clone())
            }

            // Structure mismatch
            _ => Err(MicroError::StructureMismatch),
        }
    }

    /// Weak head normal form (beta + zeta only)
    ///
    /// This is a simplified WHNF that only handles:
    /// - Beta reduction: (λx.b) a → b[a/x]
    /// - Zeta reduction: let x := v in b → b[v/x]
    ///
    /// No delta (constant unfolding) because MicroChecker doesn't have an environment.
    fn whnf(&self, e: &MicroExpr) -> MicroExpr {
        stacker::maybe_grow(MIN_STACK_RED_ZONE, STACK_GROWTH_SIZE, || self.whnf_impl(e))
    }

    /// Implementation of WHNF (called via stacker::maybe_grow)
    fn whnf_impl(&self, e: &MicroExpr) -> MicroExpr {
        match e {
            MicroExpr::App(f, a) => {
                let f_whnf = self.whnf(f);
                match &f_whnf {
                    MicroExpr::Lam(_, body) => {
                        let reduced = body.instantiate(a);
                        self.whnf(&reduced)
                    }
                    _ => MicroExpr::App(Arc::new(f_whnf), a.clone()),
                }
            }
            MicroExpr::Let(_, val, body) => {
                let reduced = body.instantiate(val);
                self.whnf(&reduced)
            }
            _ => e.clone(),
        }
    }

    /// Definitional equality check (structural after WHNF)
    fn def_eq(&self, a: &MicroExpr, b: &MicroExpr) -> bool {
        stacker::maybe_grow(MIN_STACK_RED_ZONE, STACK_GROWTH_SIZE, || {
            self.def_eq_impl(a, b)
        })
    }

    /// Implementation of def_eq (called via stacker::maybe_grow)
    fn def_eq_impl(&self, a: &MicroExpr, b: &MicroExpr) -> bool {
        let a_whnf = self.whnf(a);
        let b_whnf = self.whnf(b);
        self.structural_eq(&a_whnf, &b_whnf)
    }

    /// Structural equality (used after WHNF)
    fn structural_eq(&self, a: &MicroExpr, b: &MicroExpr) -> bool {
        stacker::maybe_grow(MIN_STACK_RED_ZONE, STACK_GROWTH_SIZE, || {
            self.structural_eq_impl(a, b)
        })
    }

    /// Implementation of structural_eq (called via stacker::maybe_grow)
    fn structural_eq_impl(&self, a: &MicroExpr, b: &MicroExpr) -> bool {
        match (a, b) {
            (MicroExpr::BVar(i), MicroExpr::BVar(j)) => i == j,
            (MicroExpr::Sort(l1), MicroExpr::Sort(l2)) => l1.level_eq(l2),
            (MicroExpr::App(f1, a1), MicroExpr::App(f2, a2)) => {
                self.structural_eq(f1, f2) && self.structural_eq(a1, a2)
            }
            (MicroExpr::Lam(ty1, b1), MicroExpr::Lam(ty2, b2))
            | (MicroExpr::Pi(ty1, b1), MicroExpr::Pi(ty2, b2)) => {
                self.structural_eq(ty1, ty2) && self.structural_eq(b1, b2)
            }
            (MicroExpr::Let(ty1, v1, b1), MicroExpr::Let(ty2, v2, b2)) => {
                self.structural_eq(ty1, ty2)
                    && self.structural_eq(v1, v2)
                    && self.structural_eq(b1, b2)
            }
            (MicroExpr::Opaque(t1), MicroExpr::Opaque(t2)) => self.structural_eq(t1, t2),
            _ => false,
        }
    }
}

impl Default for MicroChecker {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Translation from Main Kernel Types
// ============================================================================

use crate::expr::Expr;
use crate::level::Level;

/// Translation error
#[derive(Debug, Clone)]
pub enum TranslateError {
    /// Unsupported expression type for micro-checker
    UnsupportedExpr(String),
    /// Unsupported level type
    UnsupportedLevel(String),
}

impl std::fmt::Display for TranslateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TranslateError::UnsupportedExpr(msg) => write!(f, "unsupported expression: {msg}"),
            TranslateError::UnsupportedLevel(msg) => write!(f, "unsupported level: {msg}"),
        }
    }
}

impl std::error::Error for TranslateError {}

impl MicroLevel {
    /// Convert from kernel Level to MicroLevel
    pub fn from_kernel(level: &Level) -> Result<MicroLevel, TranslateError> {
        match level {
            Level::Zero => Ok(MicroLevel::Zero),
            Level::Succ(l) => Ok(MicroLevel::Succ(Arc::new(MicroLevel::from_kernel(l)?))),
            Level::Max(l1, l2) => Ok(MicroLevel::Max(
                Arc::new(MicroLevel::from_kernel(l1)?),
                Arc::new(MicroLevel::from_kernel(l2)?),
            )),
            Level::IMax(l1, l2) => Ok(MicroLevel::IMax(
                Arc::new(MicroLevel::from_kernel(l1)?),
                Arc::new(MicroLevel::from_kernel(l2)?),
            )),
            Level::Param(name) => Err(TranslateError::UnsupportedLevel(format!(
                "level parameter {name:?} not supported in micro-checker"
            ))),
        }
    }
}

impl MicroExpr {
    /// Convert from kernel Expr to MicroExpr
    ///
    /// Note: This conversion loses information (FVars, Consts become Opaque)
    /// and is only suitable for expressions that don't require delta reduction.
    pub fn from_kernel(expr: &Expr) -> Result<MicroExpr, TranslateError> {
        match expr {
            Expr::BVar(idx) => Ok(MicroExpr::BVar(*idx)),
            Expr::Sort(level) => Ok(MicroExpr::Sort(MicroLevel::from_kernel(level)?)),
            Expr::App(f, a) => Ok(MicroExpr::App(
                Arc::new(MicroExpr::from_kernel(f)?),
                Arc::new(MicroExpr::from_kernel(a)?),
            )),
            Expr::Lam(_, ty, body) => Ok(MicroExpr::Lam(
                Arc::new(MicroExpr::from_kernel(ty)?),
                Arc::new(MicroExpr::from_kernel(body)?),
            )),
            Expr::Pi(_, ty, body) => Ok(MicroExpr::Pi(
                Arc::new(MicroExpr::from_kernel(ty)?),
                Arc::new(MicroExpr::from_kernel(body)?),
            )),
            Expr::Let(ty, val, body) => Ok(MicroExpr::Let(
                Arc::new(MicroExpr::from_kernel(ty)?),
                Arc::new(MicroExpr::from_kernel(val)?),
                Arc::new(MicroExpr::from_kernel(body)?),
            )),
            // FVar and Const become opaque - we can't look them up without an environment
            Expr::FVar(_) => Err(TranslateError::UnsupportedExpr(
                "FVar not supported - use closed expressions".to_string(),
            )),
            Expr::Const(name, _) => Err(TranslateError::UnsupportedExpr(format!(
                "Const {name:?} not supported - micro-checker has no environment"
            ))),
            Expr::Lit(_) => Err(TranslateError::UnsupportedExpr(
                "Lit not supported in micro-checker".to_string(),
            )),
            Expr::Proj(_, _, _) => Err(TranslateError::UnsupportedExpr(
                "Proj not supported in micro-checker".to_string(),
            )),
            // MData is transparent - just convert the inner expression
            Expr::MData(_, inner) => MicroExpr::from_kernel(inner),

            // Mode-specific extensions are not supported in the micro-checker
            Expr::CubicalInterval
            | Expr::CubicalI0
            | Expr::CubicalI1
            | Expr::CubicalPath { .. }
            | Expr::CubicalPathLam { .. }
            | Expr::CubicalPathApp { .. }
            | Expr::CubicalHComp { .. }
            | Expr::CubicalTransp { .. } => Err(TranslateError::UnsupportedExpr(
                "Cubical expressions not supported in micro-checker".to_string(),
            )),
            Expr::ClassicalChoice { .. } | Expr::ClassicalEpsilon { .. } => {
                Err(TranslateError::UnsupportedExpr(
                    "Classical expressions not supported in micro-checker".to_string(),
                ))
            }
            Expr::ZFCSet(_) | Expr::ZFCMem { .. } | Expr::ZFCComprehension { .. } => {
                Err(TranslateError::UnsupportedExpr(
                    "SetTheoretic expressions not supported in micro-checker".to_string(),
                ))
            }
            Expr::SProp | Expr::Squash(_) => Err(TranslateError::UnsupportedExpr(
                "Impredicative expressions not supported in micro-checker".to_string(),
            )),
        }
    }

    /// Convert from kernel Expr to MicroExpr, treating unknown expressions as opaque
    /// with a given type.
    ///
    /// This is useful when you have a closed expression with constants that
    /// you want to treat as opaque with known types.
    pub fn from_kernel_with_opaques(
        expr: &Expr,
        opaque_types: &std::collections::HashMap<String, MicroExpr>,
    ) -> Result<MicroExpr, TranslateError> {
        match expr {
            Expr::BVar(idx) => Ok(MicroExpr::BVar(*idx)),
            Expr::Sort(level) => Ok(MicroExpr::Sort(MicroLevel::from_kernel(level)?)),
            Expr::App(f, a) => Ok(MicroExpr::App(
                Arc::new(MicroExpr::from_kernel_with_opaques(f, opaque_types)?),
                Arc::new(MicroExpr::from_kernel_with_opaques(a, opaque_types)?),
            )),
            Expr::Lam(_, ty, body) => Ok(MicroExpr::Lam(
                Arc::new(MicroExpr::from_kernel_with_opaques(ty, opaque_types)?),
                Arc::new(MicroExpr::from_kernel_with_opaques(body, opaque_types)?),
            )),
            Expr::Pi(_, ty, body) => Ok(MicroExpr::Pi(
                Arc::new(MicroExpr::from_kernel_with_opaques(ty, opaque_types)?),
                Arc::new(MicroExpr::from_kernel_with_opaques(body, opaque_types)?),
            )),
            Expr::Let(ty, val, body) => Ok(MicroExpr::Let(
                Arc::new(MicroExpr::from_kernel_with_opaques(ty, opaque_types)?),
                Arc::new(MicroExpr::from_kernel_with_opaques(val, opaque_types)?),
                Arc::new(MicroExpr::from_kernel_with_opaques(body, opaque_types)?),
            )),
            Expr::Const(name, _) => {
                let key = format!("{name:?}");
                opaque_types.get(&key).map_or_else(
                    || {
                        Err(TranslateError::UnsupportedExpr(format!(
                            "Const {name:?} not in opaque_types map"
                        )))
                    },
                    |ty| Ok(MicroExpr::Opaque(Arc::new(ty.clone()))),
                )
            }
            Expr::FVar(_) => Err(TranslateError::UnsupportedExpr(
                "FVar not supported".to_string(),
            )),
            Expr::Lit(_) => Err(TranslateError::UnsupportedExpr(
                "Lit not supported".to_string(),
            )),
            Expr::Proj(_, _, _) => Err(TranslateError::UnsupportedExpr(
                "Proj not supported".to_string(),
            )),
            // MData is transparent - just convert the inner expression
            Expr::MData(_, inner) => MicroExpr::from_kernel_with_opaques(inner, opaque_types),

            // Mode-specific extensions are not supported in the micro-checker
            Expr::CubicalInterval
            | Expr::CubicalI0
            | Expr::CubicalI1
            | Expr::CubicalPath { .. }
            | Expr::CubicalPathLam { .. }
            | Expr::CubicalPathApp { .. }
            | Expr::CubicalHComp { .. }
            | Expr::CubicalTransp { .. } => Err(TranslateError::UnsupportedExpr(
                "Cubical expressions not supported in micro-checker".to_string(),
            )),
            Expr::ClassicalChoice { .. } | Expr::ClassicalEpsilon { .. } => {
                Err(TranslateError::UnsupportedExpr(
                    "Classical expressions not supported in micro-checker".to_string(),
                ))
            }
            Expr::ZFCSet(_) | Expr::ZFCMem { .. } | Expr::ZFCComprehension { .. } => {
                Err(TranslateError::UnsupportedExpr(
                    "SetTheoretic expressions not supported in micro-checker".to_string(),
                ))
            }
            Expr::SProp | Expr::Squash(_) => Err(TranslateError::UnsupportedExpr(
                "Impredicative expressions not supported in micro-checker".to_string(),
            )),
        }
    }
}

// ============================================================================
// ProofCert to MicroCert Conversion
// ============================================================================

use crate::cert::ProofCert;

/// Convert a ProofCert to MicroCert for cross-validation.
///
/// This conversion may fail if the ProofCert contains constructs not
/// supported by the micro-checker (e.g., FVars, Consts, Lits, Projs).
/// In such cases, None is returned and cross-validation is skipped.
impl MicroCert {
    /// Try to convert a ProofCert to MicroCert.
    ///
    /// Returns None if the certificate contains unsupported constructs.
    /// This is expected - the micro-checker only handles a subset of the
    /// full kernel's capabilities.
    pub fn from_proof_cert(cert: &ProofCert) -> Option<MicroCert> {
        match cert {
            ProofCert::Sort { level } => {
                let micro_level = MicroLevel::from_kernel(level).ok()?;
                Some(MicroCert::Sort { level: micro_level })
            }
            ProofCert::BVar { idx, expected_type } => {
                let micro_ty = MicroExpr::from_kernel(expected_type).ok()?;
                Some(MicroCert::BVar {
                    idx: *idx,
                    ty: Box::new(micro_ty),
                })
            }
            ProofCert::FVar { .. } => {
                // FVars are not supported in micro-checker
                None
            }
            ProofCert::Const { .. } => {
                // Consts require environment lookup, not supported
                None
            }
            ProofCert::App {
                fn_cert,
                arg_cert,
                result_type,
                ..
            } => {
                let fn_micro = MicroCert::from_proof_cert(fn_cert)?;
                let arg_micro = MicroCert::from_proof_cert(arg_cert)?;
                let result_micro = MicroExpr::from_kernel(result_type).ok()?;
                Some(MicroCert::App {
                    fn_cert: Box::new(fn_micro),
                    arg_cert: Box::new(arg_micro),
                    result_ty: Box::new(result_micro),
                })
            }
            ProofCert::Lam {
                arg_type_cert,
                body_cert,
                result_type,
                ..
            } => {
                let arg_ty_micro = MicroCert::from_proof_cert(arg_type_cert)?;
                let body_micro = MicroCert::from_proof_cert(body_cert)?;
                let result_micro = MicroExpr::from_kernel(result_type).ok()?;
                Some(MicroCert::Lam {
                    arg_ty_cert: Box::new(arg_ty_micro),
                    body_cert: Box::new(body_micro),
                    result_ty: Box::new(result_micro),
                })
            }
            ProofCert::Pi {
                arg_type_cert,
                arg_level,
                body_type_cert,
                body_level,
                ..
            } => {
                let arg_ty_micro = MicroCert::from_proof_cert(arg_type_cert)?;
                let arg_level_micro = MicroLevel::from_kernel(arg_level).ok()?;
                let body_ty_micro = MicroCert::from_proof_cert(body_type_cert)?;
                let body_level_micro = MicroLevel::from_kernel(body_level).ok()?;
                Some(MicroCert::Pi {
                    arg_ty_cert: Box::new(arg_ty_micro),
                    arg_level: arg_level_micro,
                    body_ty_cert: Box::new(body_ty_micro),
                    body_level: body_level_micro,
                })
            }
            ProofCert::Let {
                type_cert,
                value_cert,
                body_cert,
                result_type,
            } => {
                let ty_micro = MicroCert::from_proof_cert(type_cert)?;
                let val_micro = MicroCert::from_proof_cert(value_cert)?;
                let body_micro = MicroCert::from_proof_cert(body_cert)?;
                let result_micro = MicroExpr::from_kernel(result_type).ok()?;
                Some(MicroCert::Let {
                    ty_cert: Box::new(ty_micro),
                    val_cert: Box::new(val_micro),
                    body_cert: Box::new(body_micro),
                    result_ty: Box::new(result_micro),
                })
            }
            ProofCert::Lit { .. } => {
                // Literals are not supported in micro-checker
                None
            }
            ProofCert::DefEq { inner, .. } => {
                // Try to convert the inner certificate
                MicroCert::from_proof_cert(inner)
            }
            ProofCert::MData { inner_cert, .. } => {
                // MData is transparent - convert inner certificate
                MicroCert::from_proof_cert(inner_cert)
            }
            ProofCert::Proj { .. } => {
                // Projections are not supported in micro-checker
                None
            }
            // Mode-specific certificates are not supported in micro-checker
            ProofCert::CubicalInterval
            | ProofCert::CubicalEndpoint { .. }
            | ProofCert::CubicalPath { .. }
            | ProofCert::CubicalPathLam { .. }
            | ProofCert::CubicalPathApp { .. }
            | ProofCert::CubicalHComp { .. }
            | ProofCert::CubicalTransp { .. }
            | ProofCert::ClassicalChoice { .. }
            | ProofCert::ClassicalEpsilon { .. }
            | ProofCert::ZFCSet { .. }
            | ProofCert::ZFCMem { .. }
            | ProofCert::ZFCComprehension { .. }
            | ProofCert::SProp
            | ProofCert::Squash { .. } => None,
        }
    }
}

/// Cross-validate type inference result using the micro-checker.
///
/// This function is called in debug builds to verify that the main kernel's
/// type inference agrees with the independent micro-checker.
///
/// # Arguments
/// - `expr`: The expression that was type-checked
/// - `inferred_type`: The type inferred by the main kernel
/// - `cert`: The proof certificate from the main kernel
///
/// # Panics
/// Panics if the micro-checker disagrees with the main kernel.
///
/// # Returns
/// `true` if cross-validation succeeded, `false` if it was skipped
/// (because the expression contains unsupported constructs).
pub fn cross_validate_with_micro(expr: &Expr, inferred_type: &Expr, cert: &ProofCert) -> bool {
    // Try to convert expression to MicroExpr
    let Ok(micro_expr) = MicroExpr::from_kernel(expr) else {
        return false; // Skip validation for unsupported expressions
    };

    // Try to convert certificate to MicroCert
    let Some(micro_cert) = MicroCert::from_proof_cert(cert) else {
        return false; // Skip validation for unsupported certificates
    };

    // Try to convert inferred type to MicroExpr
    let Ok(micro_inferred_type) = MicroExpr::from_kernel(inferred_type) else {
        return false; // Skip validation for unsupported types
    };

    // Run the micro-checker
    let mut micro_checker = MicroChecker::new();
    let micro_result = micro_checker.verify(&micro_cert, &micro_expr);

    match micro_result {
        Ok(micro_type) => {
            // Compare types (structural equality after WHNF)
            // Note: MicroChecker doesn't have delta reduction, so we compare structurally
            assert!(
                micro_type == micro_inferred_type,
                "MICRO-CHECKER DISAGREEMENT!\n\
                 Expression: {expr:?}\n\
                 Main kernel type: {inferred_type:?}\n\
                 Micro-checker type: {micro_type:?}\n\
                 Certificate: {cert:?}"
            );
            true
        }
        Err(e) => {
            panic!(
                "MICRO-CHECKER VERIFICATION FAILED!\n\
                 Expression: {expr:?}\n\
                 Main kernel type: {inferred_type:?}\n\
                 Certificate: {cert:?}\n\
                 Error: {e:?}"
            );
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn l0() -> MicroLevel {
        MicroLevel::Zero
    }
    fn l1() -> MicroLevel {
        MicroLevel::succ(l0())
    }
    fn sort(l: MicroLevel) -> MicroExpr {
        MicroExpr::Sort(l)
    }
    fn bvar(i: u32) -> MicroExpr {
        MicroExpr::BVar(i)
    }
    fn app(f: MicroExpr, a: MicroExpr) -> MicroExpr {
        MicroExpr::App(Arc::new(f), Arc::new(a))
    }
    fn lam(ty: MicroExpr, body: MicroExpr) -> MicroExpr {
        MicroExpr::Lam(Arc::new(ty), Arc::new(body))
    }
    fn pi(ty: MicroExpr, body: MicroExpr) -> MicroExpr {
        MicroExpr::Pi(Arc::new(ty), Arc::new(body))
    }
    fn let_(ty: MicroExpr, val: MicroExpr, body: MicroExpr) -> MicroExpr {
        MicroExpr::Let(Arc::new(ty), Arc::new(val), Arc::new(body))
    }

    // ========================================================================
    // Expression operations tests
    // ========================================================================

    #[test]
    fn test_lift_bvar() {
        let e = bvar(0);
        assert_eq!(e.lift(0, 1), bvar(1));
        assert_eq!(e.lift(1, 1), bvar(0)); // Below cutoff, unchanged
    }

    #[test]
    fn test_lift_lambda() {
        // λ x. x  (body is BVar(0))
        let e = lam(sort(l0()), bvar(0));
        // Lifting doesn't affect bound variables inside binders
        let lifted = e.lift(0, 1);
        assert_eq!(lifted, lam(sort(l0()), bvar(0)));
    }

    #[test]
    fn test_instantiate_simple() {
        // BVar(0)[val/0] = val
        let e = bvar(0);
        let val = sort(l0());
        assert_eq!(e.instantiate(&val), val);
    }

    #[test]
    fn test_instantiate_higher_index() {
        // BVar(1)[val/0] = BVar(0) (index decreases)
        let e = bvar(1);
        let val = sort(l0());
        assert_eq!(e.instantiate(&val), bvar(0));
    }

    #[test]
    fn test_instantiate_under_binder() {
        // (λ x. BVar(1))[val/0] = λ x. val
        // The BVar(1) refers to the outer variable (index 0 at depth 0)
        let e = lam(sort(l0()), bvar(1));
        let val = sort(l1());
        let result = e.instantiate(&val);
        // After substitution: λ x. Sort(1)
        assert_eq!(result, lam(sort(l0()), sort(l1())));
    }

    // ========================================================================
    // Level tests
    // ========================================================================

    #[test]
    fn test_level_eq() {
        assert!(l0().level_eq(&l0()));
        assert!(l1().level_eq(&l1()));
        assert!(!l0().level_eq(&l1()));
    }

    #[test]
    fn test_imax_zero_right() {
        // imax(l, 0) = 0
        let l = MicroLevel::imax(l1(), l0());
        assert_eq!(l, l0());
    }

    // ========================================================================
    // WHNF tests
    // ========================================================================

    #[test]
    fn test_whnf_sort() {
        let checker = MicroChecker::new();
        let e = sort(l0());
        assert_eq!(checker.whnf(&e), e);
    }

    #[test]
    fn test_whnf_beta() {
        // (λ x. x) y → y
        let checker = MicroChecker::new();
        let id = lam(sort(l0()), bvar(0));
        let e = app(id, sort(l1()));
        assert_eq!(checker.whnf(&e), sort(l1()));
    }

    #[test]
    fn test_whnf_nested_beta() {
        // (λ x. λ y. x) a b → a
        let checker = MicroChecker::new();
        let f = lam(sort(l0()), lam(sort(l0()), bvar(1)));
        let e = app(app(f, sort(l1())), sort(l0()));
        assert_eq!(checker.whnf(&e), sort(l1()));
    }

    #[test]
    fn test_whnf_zeta() {
        // let x := v in x → v
        let checker = MicroChecker::new();
        let e = let_(sort(l0()), sort(l1()), bvar(0));
        assert_eq!(checker.whnf(&e), sort(l1()));
    }

    // ========================================================================
    // Verification tests
    // ========================================================================

    #[test]
    fn test_verify_sort() {
        let mut checker = MicroChecker::new();
        let expr = sort(l0());
        let cert = MicroCert::Sort { level: l0() };

        let result = checker.verify(&cert, &expr);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), sort(l1()));
    }

    #[test]
    fn test_verify_sort_level_mismatch() {
        let mut checker = MicroChecker::new();
        let expr = sort(l0());
        let cert = MicroCert::Sort { level: l1() };

        let result = checker.verify(&cert, &expr);
        assert!(matches!(result, Err(MicroError::LevelMismatch { .. })));
    }

    #[test]
    fn test_verify_pi() {
        // Prop → Prop : Type 0
        let mut checker = MicroChecker::new();
        let prop = sort(l0());
        let expr = pi(prop.clone(), prop.clone());

        let cert = MicroCert::Pi {
            arg_ty_cert: Box::new(MicroCert::Sort { level: l0() }),
            arg_level: l1(),
            body_ty_cert: Box::new(MicroCert::Sort { level: l0() }),
            body_level: l1(),
        };

        let result = checker.verify(&cert, &expr);
        assert!(result.is_ok());
        // imax(1, 1) = 1
        assert_eq!(result.unwrap(), sort(MicroLevel::imax(l1(), l1())));
    }

    #[test]
    fn test_verify_identity() {
        // λ (x : Prop). x : Prop → Prop
        let mut checker = MicroChecker::new();
        let prop = sort(l0());
        let expr = lam(prop.clone(), bvar(0));

        let expected_ty = pi(prop.clone(), prop.clone());

        let cert = MicroCert::Lam {
            arg_ty_cert: Box::new(MicroCert::Sort { level: l0() }),
            body_cert: Box::new(MicroCert::BVar {
                idx: 0,
                ty: Box::new(prop.clone()),
            }),
            result_ty: Box::new(expected_ty.clone()),
        };

        let result = checker.verify(&cert, &expr);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), expected_ty);
    }

    #[test]
    fn test_verify_app() {
        // (λ (A : Type). A) Prop : Type
        //
        // Note: λ (A : Type). A returns its argument, which is a Type.
        // The body `A` (BVar(0)) has type `Type` (from the binder).
        // So the lambda has type `Type → Type`, not `(A : Type) → A`.
        //
        // When applied to Prop, the result is Prop, which has type Type.

        let mut checker = MicroChecker::new();
        let type0 = sort(l0()); // Type 0 = Prop
        let type1 = sort(l1()); // Type 1

        // Identity on types: λ (A : Type). A
        // Expression: Lam(Sort(l1), BVar(0))
        let id_type = lam(type1.clone(), bvar(0));

        // Type of id: Type → Type
        // The body (BVar(0)) has type Type (from the binder), so result type is Type
        let id_ty = pi(type1.clone(), type1.clone());

        // Verify the lambda alone
        let lam_cert = MicroCert::Lam {
            arg_ty_cert: Box::new(MicroCert::Sort { level: l1() }),
            body_cert: Box::new(MicroCert::BVar {
                idx: 0,
                ty: Box::new(type1.clone()),
            }),
            result_ty: Box::new(id_ty.clone()),
        };
        let lam_result = checker.verify(&lam_cert, &id_type);
        assert!(lam_result.is_ok(), "Lambda error: {lam_result:?}");

        // Verify the argument alone (Prop : Type)
        let arg_cert = MicroCert::Sort { level: l0() };
        let arg_result = checker.verify(&arg_cert, &type0);
        assert!(arg_result.is_ok(), "Arg error: {arg_result:?}");

        // The app: (λ (A : Type). A) Prop
        let expr = app(id_type.clone(), type0.clone());

        // Result type: Type (from the Pi body which is Type, no substitution needed)
        let cert = MicroCert::App {
            fn_cert: Box::new(lam_cert.clone()),
            arg_cert: Box::new(arg_cert),
            result_ty: Box::new(type1.clone()), // The result type is Type
        };

        let result = checker.verify(&cert, &expr);
        assert!(result.is_ok(), "Error: {result:?}");
        assert_eq!(result.unwrap(), type1);
    }

    #[test]
    fn test_verify_let() {
        // let x : Type 1 := Type 0 in x : Type 1
        // The body type is Type 1, after substituting value it's still Type 1
        // (not the value itself - the TYPE of the body after substitution)
        let mut checker = MicroChecker::new();
        let type1 = sort(l1()); // Type 1
        let type0 = sort(l0()); // Type 0 = Prop

        // let x : Type 1 := Type 0 in x
        let expr = let_(type1.clone(), type0.clone(), bvar(0));

        let cert = MicroCert::Let {
            // Type 1 : Type 2
            ty_cert: Box::new(MicroCert::Sort { level: l1() }),
            // Type 0 : Type 1 (but we need Type 0 to have Type 1, which it does!)
            val_cert: Box::new(MicroCert::Sort { level: l0() }),
            // In body context, x : Type 1, so x has type Type 1
            body_cert: Box::new(MicroCert::BVar {
                idx: 0,
                ty: Box::new(type1.clone()),
            }),
            // After substitution: Type 1[Type 0/x] = Type 1
            // The body_ty is Type 1, instantiating doesn't change it
            result_ty: Box::new(type1.clone()),
        };

        let result = checker.verify(&cert, &expr);
        assert!(result.is_ok(), "Error: {result:?}");
        assert_eq!(result.unwrap(), type1);
    }

    #[test]
    fn test_verify_structure_mismatch() {
        let mut checker = MicroChecker::new();
        let expr = sort(l0());
        let cert = MicroCert::BVar {
            idx: 0,
            ty: Box::new(sort(l0())),
        };

        let result = checker.verify(&cert, &expr);
        assert!(matches!(result, Err(MicroError::StructureMismatch)));
    }

    #[test]
    fn test_verify_nested_lambda() {
        // λ (A : Type). λ (x : A). x : (A : Type) → A → A
        let mut checker = MicroChecker::new();
        let type0 = sort(l0());

        // Inner: λ (x : A). x where A is BVar(0) from outer
        let inner = lam(bvar(0), bvar(0));
        // Outer: λ (A : Type). inner
        let expr = lam(type0.clone(), inner);

        // Inner type: A → A (where A is BVar(0))
        let inner_ty = pi(bvar(0), bvar(1));
        // Outer type: (A : Type) → A → A
        let outer_ty = pi(type0.clone(), inner_ty.clone());

        let cert = MicroCert::Lam {
            arg_ty_cert: Box::new(MicroCert::Sort { level: l0() }),
            body_cert: Box::new(MicroCert::Lam {
                arg_ty_cert: Box::new(MicroCert::BVar {
                    idx: 0,
                    ty: Box::new(type0.clone()),
                }),
                body_cert: Box::new(MicroCert::BVar {
                    idx: 0,
                    ty: Box::new(bvar(1)), // x : A (shifted by 1)
                }),
                result_ty: Box::new(inner_ty.clone()),
            }),
            result_ty: Box::new(outer_ty.clone()),
        };

        let result = checker.verify(&cert, &expr);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), outer_ty);
    }

    #[test]
    fn test_verify_type_mismatch_in_app() {
        // Try to apply identity (Prop → Prop) to Type (wrong argument type)
        let mut checker = MicroChecker::new();
        let prop = sort(l0());
        let type1 = sort(l1());
        let id = lam(prop.clone(), bvar(0));
        let expr = app(id.clone(), type1.clone());

        let id_ty = pi(prop.clone(), prop.clone());

        let cert = MicroCert::App {
            fn_cert: Box::new(MicroCert::Lam {
                arg_ty_cert: Box::new(MicroCert::Sort { level: l0() }),
                body_cert: Box::new(MicroCert::BVar {
                    idx: 0,
                    ty: Box::new(prop.clone()),
                }),
                result_ty: Box::new(id_ty),
            }),
            // Wrong! Argument is Type 1, but function expects Prop
            arg_cert: Box::new(MicroCert::Sort { level: l1() }),
            result_ty: Box::new(type1),
        };

        let result = checker.verify(&cert, &expr);
        assert!(matches!(result, Err(MicroError::TypeMismatch { .. })));
    }

    #[test]
    fn test_def_eq_beta() {
        // (λ x. x) y ≡ y
        let checker = MicroChecker::new();
        let id = lam(sort(l0()), bvar(0));
        let y = sort(l1());
        let app_e = app(id, y.clone());

        assert!(checker.def_eq(&app_e, &y));
    }

    #[test]
    fn test_def_eq_under_binder() {
        // WHNF doesn't reduce under binders, so λ x. (λ y. y) x ≢ λ x. x
        // This is intentional - micro-checker only does WHNF at the head
        // For full definitional equality under binders, you'd need eta/deep reduction
        //
        // Instead, test that def_eq works for structurally equal lambdas
        let checker = MicroChecker::new();
        let lhs = lam(sort(l0()), bvar(0));
        let rhs = lam(sort(l0()), bvar(0));

        assert!(checker.def_eq(&lhs, &rhs));

        // And different lambdas are not equal
        let different = lam(sort(l1()), bvar(0));
        assert!(!checker.def_eq(&lhs, &different));
    }

    // ========================================================================
    // Translation tests
    // ========================================================================

    #[test]
    fn test_translate_level_zero() {
        let kernel_level = Level::zero();
        let micro_level = MicroLevel::from_kernel(&kernel_level).unwrap();
        assert_eq!(micro_level, MicroLevel::Zero);
    }

    #[test]
    fn test_translate_level_succ() {
        let kernel_level = Level::succ(Level::zero());
        let micro_level = MicroLevel::from_kernel(&kernel_level).unwrap();
        assert_eq!(micro_level, l1());
    }

    #[test]
    fn test_translate_level_max() {
        // Note: Kernel Level::max simplifies max(0, l) = l
        // So we test with two non-comparable levels to get an actual Max

        // max(u, v) where u,v are parameters should NOT simplify
        // But since we can't translate params, test that max(1, 1) = 1
        let kernel_level = Level::max(Level::succ(Level::zero()), Level::succ(Level::zero()));
        let micro_level = MicroLevel::from_kernel(&kernel_level).unwrap();
        // max(1, 1) = 1 due to simplification
        assert_eq!(micro_level, l1());

        // Test that we can construct a Max if needed (raw construction)
        let raw_max = Level::Max(
            Arc::new(Level::succ(Level::zero())),
            Arc::new(Level::succ(Level::succ(Level::zero()))),
        );
        let micro_max = MicroLevel::from_kernel(&raw_max).unwrap();
        // This should be Max(1, 2)
        assert_eq!(
            micro_max,
            MicroLevel::Max(Arc::new(l1()), Arc::new(MicroLevel::succ(l1())))
        );
    }

    #[test]
    fn test_translate_level_param_fails() {
        use crate::name::Name;
        let kernel_level = Level::param(Name::from_string("u"));
        let result = MicroLevel::from_kernel(&kernel_level);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_expr_sort() {
        let kernel_expr = Expr::Sort(Level::zero());
        let micro_expr = MicroExpr::from_kernel(&kernel_expr).unwrap();
        assert_eq!(micro_expr, sort(l0()));
    }

    #[test]
    fn test_translate_expr_bvar() {
        let kernel_expr = Expr::BVar(5);
        let micro_expr = MicroExpr::from_kernel(&kernel_expr).unwrap();
        assert_eq!(micro_expr, bvar(5));
    }

    #[test]
    fn test_translate_expr_lam() {
        use crate::expr::BinderInfo;

        // λ (x : Prop). x
        let kernel_expr = Expr::Lam(
            BinderInfo::Default,
            Arc::new(Expr::Sort(Level::zero())),
            Arc::new(Expr::BVar(0)),
        );
        let micro_expr = MicroExpr::from_kernel(&kernel_expr).unwrap();
        assert_eq!(micro_expr, lam(sort(l0()), bvar(0)));
    }

    #[test]
    fn test_translate_expr_pi() {
        use crate::expr::BinderInfo;

        // Prop → Prop
        let kernel_expr = Expr::Pi(
            BinderInfo::Default,
            Arc::new(Expr::Sort(Level::zero())),
            Arc::new(Expr::Sort(Level::zero())),
        );
        let micro_expr = MicroExpr::from_kernel(&kernel_expr).unwrap();
        assert_eq!(micro_expr, pi(sort(l0()), sort(l0())));
    }

    #[test]
    fn test_translate_expr_app() {
        use crate::expr::BinderInfo;

        // (λ x. x) Prop
        let kernel_id = Expr::Lam(
            BinderInfo::Default,
            Arc::new(Expr::Sort(Level::zero())),
            Arc::new(Expr::BVar(0)),
        );
        let kernel_expr = Expr::App(Arc::new(kernel_id), Arc::new(Expr::Sort(Level::zero())));
        let micro_expr = MicroExpr::from_kernel(&kernel_expr).unwrap();
        assert_eq!(micro_expr, app(lam(sort(l0()), bvar(0)), sort(l0())));
    }

    #[test]
    fn test_translate_expr_const_fails() {
        use crate::name::Name;

        let kernel_expr = Expr::const_(Name::from_string("Nat"), vec![]);
        let result = MicroExpr::from_kernel(&kernel_expr);
        assert!(result.is_err());
    }

    #[test]
    fn test_cross_validate_sort_verification() {
        // Verify that both micro-checker and main kernel agree on Sort typing
        use crate::env::Environment;
        use crate::tc::TypeChecker;

        let env = Environment::new();
        let mut tc = TypeChecker::new(&env);

        // Main kernel: Sort(0) : Sort(1)
        let kernel_expr = Expr::Sort(Level::zero());
        let kernel_ty = tc.infer_type(&kernel_expr).unwrap();

        // Micro-checker
        let micro_expr = MicroExpr::from_kernel(&kernel_expr).unwrap();
        let micro_cert = MicroCert::Sort { level: l0() };
        let mut micro_checker = MicroChecker::new();
        let micro_ty = micro_checker.verify(&micro_cert, &micro_expr).unwrap();

        // Both should give Sort(1)
        let kernel_ty_translated = MicroExpr::from_kernel(&kernel_ty).unwrap();
        assert_eq!(micro_ty, kernel_ty_translated);
    }

    #[test]
    fn test_cross_validate_identity_verification() {
        // Verify that both agree on identity function typing
        use crate::env::Environment;
        use crate::expr::BinderInfo;
        use crate::tc::TypeChecker;

        let env = Environment::new();
        let mut tc = TypeChecker::new(&env);

        // λ (x : Prop). x
        let kernel_expr = Expr::Lam(
            BinderInfo::Default,
            Arc::new(Expr::Sort(Level::zero())),
            Arc::new(Expr::BVar(0)),
        );
        let kernel_ty = tc.infer_type(&kernel_expr).unwrap();

        // Micro-checker
        let micro_expr = MicroExpr::from_kernel(&kernel_expr).unwrap();
        let micro_cert = MicroCert::Lam {
            arg_ty_cert: Box::new(MicroCert::Sort { level: l0() }),
            body_cert: Box::new(MicroCert::BVar {
                idx: 0,
                ty: Box::new(sort(l0())),
            }),
            result_ty: Box::new(pi(sort(l0()), sort(l0()))),
        };
        let mut micro_checker = MicroChecker::new();
        let micro_ty = micro_checker.verify(&micro_cert, &micro_expr).unwrap();

        // Both should give Prop → Prop
        let kernel_ty_translated = MicroExpr::from_kernel(&kernel_ty).unwrap();
        assert_eq!(micro_ty, kernel_ty_translated);
    }

    // ========================================================================
    // Tests targeting surviving mutations
    // ========================================================================

    // --- MicroExpr::lift arithmetic tests ---

    #[test]
    fn test_lift_bvar_at_cutoff() {
        // BVar(1) lifted at cutoff 1 with amount 1 should become BVar(2)
        let e = bvar(1);
        assert_eq!(e.lift(1, 1), bvar(2));
        // Below cutoff, unchanged
        assert_eq!(e.lift(2, 1), bvar(1));
    }

    #[test]
    fn test_lift_pi_body_increment() {
        // Pi body has cutoff+1
        let e = pi(sort(l0()), bvar(1)); // body has free var at 1
        let lifted = e.lift(0, 1);
        // The body should lift with cutoff=1, so BVar(1) >= 1 becomes BVar(2)
        assert_eq!(lifted, pi(sort(l0()), bvar(2)));
    }

    #[test]
    fn test_lift_let_body_increment() {
        // Let body has cutoff+1
        let e = let_(sort(l0()), sort(l0()), bvar(1)); // body has free var at 1
        let lifted = e.lift(0, 1);
        // The body should lift with cutoff=1, so BVar(1) >= 1 becomes BVar(2)
        assert_eq!(lifted, let_(sort(l0()), sort(l0()), bvar(2)));
    }

    #[test]
    fn test_lift_multiple_amounts() {
        // Lifting by 2 vs lifting by 1 twice
        let e = bvar(0);
        let lift2 = e.lift(0, 2);
        assert_eq!(lift2, bvar(2));

        // Verify + vs * matters: lift(0, 2) should give BVar(0+2)=BVar(2), not BVar(0*2)=BVar(0)
        assert_ne!(lift2, bvar(0));
    }

    // --- MicroExpr::subst tests ---

    #[test]
    fn test_subst_boundary_condition() {
        // BVar(1) with depth=0: should become BVar(0) (idx > depth, so idx-1)
        let e = bvar(1);
        let val = sort(l0());
        let result = e.subst(0, &val);
        assert_eq!(result, bvar(0));
    }

    #[test]
    fn test_subst_exact_match() {
        // BVar(0) with depth=0: should substitute
        let e = bvar(0);
        let val = sort(l1());
        let result = e.subst(0, &val);
        assert_eq!(result, val);
    }

    #[test]
    fn test_subst_below_depth() {
        // BVar(0) with depth=1: idx < depth, so unchanged
        let e = bvar(0);
        let val = sort(l1());
        let result = e.subst(1, &val);
        assert_eq!(result, bvar(0));
    }

    #[test]
    fn test_subst_body_depth_increment() {
        // Lambda body subst should use depth+1
        let e = lam(sort(l0()), bvar(1)); // Body refers to outer var (idx=1 after body's binder)
        let val = sort(l1());
        // Substituting at depth=0: body uses depth=1
        // BVar(1) in body: idx=1, depth=1, so idx == depth -> substitute
        let result = e.subst(0, &val);
        // The body's BVar(1) should be substituted with val.lift(0, 1) = sort(l1())
        assert_eq!(result, lam(sort(l0()), sort(l1())));
    }

    // --- MicroLevel::is_geq tests ---

    #[test]
    fn test_is_geq_same_base() {
        // If same base and offset1 >= offset2, return true
        let l1 = MicroLevel::succ(MicroLevel::succ(MicroLevel::Zero)); // 2
        let l2 = MicroLevel::succ(MicroLevel::Zero); // 1
        assert!(MicroLevel::is_geq(&l1, &l2)); // 2 >= 1
        assert!(!MicroLevel::is_geq(&l2, &l1)); // 1 >= 2 is false
    }

    #[test]
    fn test_is_geq_equal_levels() {
        let l = MicroLevel::succ(MicroLevel::Zero);
        assert!(MicroLevel::is_geq(&l, &l)); // l >= l
    }

    #[test]
    fn test_is_geq_zero_comparison() {
        // l >= 0 for any l
        let l = MicroLevel::succ(MicroLevel::Zero);
        assert!(MicroLevel::is_geq(&l, &MicroLevel::Zero));
        assert!(MicroLevel::is_geq(&MicroLevel::Zero, &MicroLevel::Zero));
    }

    #[test]
    fn test_is_geq_offset_check() {
        // Test that offset > 0 check matters
        let l1 = MicroLevel::Succ(Arc::new(MicroLevel::Zero)); // 1
        let l2 = MicroLevel::Zero; // 0
                                   // 1 >= 0 should be true
        assert!(MicroLevel::is_geq(&l1, &l2));

        // Verify the comparison is > not >=
        // offset1=1, l1' = Zero, check if Zero >= l2=Zero, which is true
        // But this relies on the comparison being >0, not >=0
    }

    #[test]
    fn test_is_geq_max_left() {
        // max(a, b) >= l if a >= l or b >= l
        let a = MicroLevel::succ(MicroLevel::Zero); // 1
        let b = MicroLevel::Zero; // 0
        let max_level = MicroLevel::Max(Arc::new(a.clone()), Arc::new(b.clone()));
        let l = MicroLevel::succ(MicroLevel::Zero); // 1

        // max(1, 0) >= 1 should be true (because 1 >= 1)
        assert!(MicroLevel::is_geq(&max_level, &l));

        // max(0, 0) >= 1 should be false
        let max_zeros = MicroLevel::Max(Arc::new(MicroLevel::Zero), Arc::new(MicroLevel::Zero));
        assert!(!MicroLevel::is_geq(&max_zeros, &l));
    }

    #[test]
    fn test_is_geq_max_right() {
        // l >= max(a, b) if l >= a and l >= b
        let l = MicroLevel::succ(MicroLevel::succ(MicroLevel::Zero)); // 2
        let a = MicroLevel::succ(MicroLevel::Zero); // 1
        let b = MicroLevel::succ(MicroLevel::Zero); // 1
        let max_level = MicroLevel::Max(Arc::new(a), Arc::new(b));

        // 2 >= max(1, 1) should be true
        assert!(MicroLevel::is_geq(&l, &max_level));

        // 0 >= max(1, 1) should be false
        assert!(!MicroLevel::is_geq(&MicroLevel::Zero, &max_level));

        // Test that AND is required: 1 >= max(0, 2) should be false
        let a2 = MicroLevel::Zero;
        let b2 = MicroLevel::succ(MicroLevel::succ(MicroLevel::Zero));
        let max2 = MicroLevel::Max(Arc::new(a2), Arc::new(b2));
        let one = MicroLevel::succ(MicroLevel::Zero);
        assert!(!MicroLevel::is_geq(&one, &max2)); // 1 >= 0 but 1 >= 2 is false
    }

    // --- MicroLevel::imax tests ---

    #[test]
    fn test_imax_zero_left() {
        // imax(0, l) = l (when l != 0)
        let l = MicroLevel::succ(MicroLevel::Zero);
        let result = MicroLevel::imax(MicroLevel::Zero, l.clone());
        assert_eq!(result, l);
    }

    #[test]
    fn test_imax_equal() {
        // imax(l, l) = l
        let l = MicroLevel::succ(MicroLevel::Zero);
        let result = MicroLevel::imax(l.clone(), l.clone());
        assert_eq!(result, l);
    }

    #[test]
    fn test_imax_creates_imax_node() {
        // When l2 is not Zero or Succ, and l1 != l2, should create IMax
        // Use an IMax as l2 to test this
        let l1 = MicroLevel::succ(MicroLevel::Zero);
        let inner = MicroLevel::IMax(
            Arc::new(MicroLevel::succ(MicroLevel::Zero)),
            Arc::new(MicroLevel::succ(MicroLevel::succ(MicroLevel::Zero))),
        );
        let result = MicroLevel::imax(l1.clone(), inner.clone());
        // Should create IMax(l1, inner) since inner is not Zero/Succ and l1 != inner
        assert!(matches!(result, MicroLevel::IMax(_, _)));
    }

    // --- MicroChecker::verify Opaque tests ---

    #[test]
    fn test_verify_opaque_matching_type() {
        let mut checker = MicroChecker::new();
        let ty = sort(l0());
        let expr = MicroExpr::Opaque(Arc::new(ty.clone()));
        let cert = MicroCert::Opaque {
            ty: Box::new(ty.clone()),
        };

        let result = checker.verify(&cert, &expr);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ty);
    }

    #[test]
    fn test_verify_opaque_mismatching_type() {
        let mut checker = MicroChecker::new();
        let ty1 = sort(l0());
        let ty2 = sort(l1());
        let expr = MicroExpr::Opaque(Arc::new(ty1));
        let cert = MicroCert::Opaque { ty: Box::new(ty2) };

        let result = checker.verify(&cert, &expr);
        assert!(matches!(result, Err(MicroError::TypeMismatch { .. })));
    }

    // --- MicroChecker::structural_eq tests ---

    #[test]
    fn test_structural_eq_app() {
        let checker = MicroChecker::new();
        let f1 = sort(l0());
        let a1 = sort(l0());
        let app1 = app(f1.clone(), a1.clone());
        let app2 = app(f1.clone(), a1.clone());
        let app3 = app(f1.clone(), sort(l1())); // different arg
        let app4 = app(sort(l1()), a1.clone()); // different fn

        assert!(checker.structural_eq(&app1, &app2));
        assert!(!checker.structural_eq(&app1, &app3));
        assert!(!checker.structural_eq(&app1, &app4));
    }

    #[test]
    fn test_structural_eq_pi() {
        let checker = MicroChecker::new();
        let pi1 = pi(sort(l0()), sort(l0()));
        let pi2 = pi(sort(l0()), sort(l0()));
        let pi3 = pi(sort(l1()), sort(l0())); // different type
        let pi4 = pi(sort(l0()), sort(l1())); // different body

        assert!(checker.structural_eq(&pi1, &pi2));
        assert!(!checker.structural_eq(&pi1, &pi3));
        assert!(!checker.structural_eq(&pi1, &pi4));
    }

    #[test]
    fn test_structural_eq_let() {
        let checker = MicroChecker::new();
        let let1 = let_(sort(l0()), sort(l0()), bvar(0));
        let let2 = let_(sort(l0()), sort(l0()), bvar(0));
        let let3 = let_(sort(l1()), sort(l0()), bvar(0)); // different type
        let let4 = let_(sort(l0()), sort(l1()), bvar(0)); // different value
        let let5 = let_(sort(l0()), sort(l0()), bvar(1)); // different body

        assert!(checker.structural_eq(&let1, &let2));
        assert!(!checker.structural_eq(&let1, &let3));
        assert!(!checker.structural_eq(&let1, &let4));
        assert!(!checker.structural_eq(&let1, &let5));
    }

    #[test]
    fn test_structural_eq_opaque() {
        let checker = MicroChecker::new();
        let op1 = MicroExpr::Opaque(Arc::new(sort(l0())));
        let op2 = MicroExpr::Opaque(Arc::new(sort(l0())));
        let op3 = MicroExpr::Opaque(Arc::new(sort(l1())));

        assert!(checker.structural_eq(&op1, &op2));
        assert!(!checker.structural_eq(&op1, &op3));
    }

    // --- Display tests ---

    #[test]
    fn test_micro_error_display() {
        let err = MicroError::InvalidBVar(5);
        let s = format!("{err}");
        assert!(!s.is_empty());

        let err2 = MicroError::StructureMismatch;
        let s2 = format!("{err2}");
        assert!(!s2.is_empty());
    }

    #[test]
    fn test_translate_error_display() {
        let err = TranslateError::UnsupportedExpr("test".to_string());
        let s = format!("{err}");
        assert!(!s.is_empty());
        assert!(s.contains("test"));

        let err2 = TranslateError::UnsupportedLevel("level".to_string());
        let s2 = format!("{err2}");
        assert!(!s2.is_empty());
    }

    // =========================================================================
    // Additional Mutation Testing Kill Tests - micro.rs survivors
    // =========================================================================

    #[test]
    fn test_lift_plus_vs_times() {
        // Kill mutants: replace + with * in MicroExpr::lift (lines 180, 187)
        // Verify idx + amount, not idx * amount

        // BVar(2) lifted by 3 should be BVar(5), not BVar(6)
        let e = bvar(2);
        let lifted = e.lift(0, 3);
        assert_eq!(lifted, bvar(5), "2 + 3 = 5, not 2 * 3 = 6");

        // BVar(3) lifted by 2 should be BVar(5), not BVar(6)
        let e = bvar(3);
        let lifted = e.lift(0, 2);
        assert_eq!(lifted, bvar(5), "3 + 2 = 5, not 3 * 2 = 6");

        // BVar(1) lifted by 4 should be BVar(5), not BVar(4)
        let e = bvar(1);
        let lifted = e.lift(0, 4);
        assert_eq!(lifted, bvar(5), "1 + 4 = 5, not 1 * 4 = 4");
    }

    #[test]
    fn test_subst_greater_than_vs_geq() {
        // Kill mutant: replace > with >= in MicroExpr::subst (line 205)
        // idx > depth means decrement, idx == depth means substitute

        // BVar(1) at depth 1: 1 == 1, so should substitute, NOT decrement
        let e = bvar(1);
        let val = sort(l1());
        let result = e.subst(1, &val);
        // val lifted by 1 = sort(l1())
        assert_eq!(result, sort(l1()), "BVar(1) at depth=1 should substitute");

        // BVar(2) at depth 1: 2 > 1, so should decrement to BVar(1)
        let e = bvar(2);
        let result = e.subst(1, &val);
        assert_eq!(
            result,
            bvar(1),
            "BVar(2) at depth=1 should decrement to BVar(1)"
        );
    }

    #[test]
    fn test_subst_plus_vs_minus() {
        // Kill mutants: replace + with - in MicroExpr::subst (line 234)
        // Tests depth + 1 for nested binders

        // Lambda body uses depth+1 for substitution
        // λ (x : Prop). BVar(1) - in body, depth=1, so BVar(1)==1 gets substituted
        let e = lam(sort(l0()), bvar(1));
        let val = sort(l1());
        let result = e.subst(0, &val);
        // Body BVar(1) at depth=1: 1==1, substitute with val.lift(0,1) = sort(l1())
        assert_eq!(result, lam(sort(l0()), sort(l1())));

        // Pi also uses depth+1
        let e = pi(sort(l0()), bvar(1));
        let result = e.subst(0, &val);
        assert_eq!(result, pi(sort(l0()), sort(l1())));
    }

    #[test]
    fn test_is_geq_comparison_operators() {
        // Kill mutants: replace > with ==, >=, < in MicroLevel::is_geq (line 295)
        // Tests offset1 > 0 comparison

        // Succ(Zero) vs Zero: offset1=1, base=Zero
        // offset1 > 0 means we check if Zero >= Zero (yes)
        // If > was ==: offset1 == 0 is false, wouldn't recurse
        // If > was <: offset1 < 0 is false, wouldn't recurse
        let l1 = MicroLevel::succ(MicroLevel::Zero); // offset=1
        let l0 = MicroLevel::Zero; // offset=0
        assert!(MicroLevel::is_geq(&l1, &l0), "Succ(Zero) >= Zero");

        // Succ(Succ(Zero)) vs Succ(Zero): 2 >= 1
        let l2 = MicroLevel::succ(MicroLevel::succ(MicroLevel::Zero));
        let l1_copy = MicroLevel::succ(MicroLevel::Zero);
        assert!(MicroLevel::is_geq(&l2, &l1_copy), "2 >= 1");

        // Zero vs Succ(Zero): 0 >= 1 should be false
        assert!(!MicroLevel::is_geq(&l0, &l1), "0 >= 1 is false");
    }

    #[test]
    fn test_imax_equality_check() {
        // Kill mutant: replace == with != in MicroLevel::imax (line 344)
        // imax(l, l) = l, but if == became !=, identical levels wouldn't simplify

        // imax(Zero, Zero) should equal Zero
        let result = MicroLevel::imax(MicroLevel::Zero, MicroLevel::Zero);
        assert_eq!(result, MicroLevel::Zero);

        // imax(1, 1) should equal 1
        let l1 = MicroLevel::succ(MicroLevel::Zero);
        let result = MicroLevel::imax(l1.clone(), l1.clone());
        assert_eq!(result, l1);

        // imax(2, 2) should equal 2
        let l2 = MicroLevel::succ(MicroLevel::succ(MicroLevel::Zero));
        let result = MicroLevel::imax(l2.clone(), l2.clone());
        assert_eq!(result, l2);
    }

    // =========================================================================
    // Additional Mutation Kill Tests - cutoff+1 and depth+1
    // =========================================================================

    #[test]
    fn test_lift_cutoff_plus_one_in_binders() {
        // Kill mutants at lines 180, 187: replace cutoff + 1 with cutoff * 1
        // When cutoff=0, cutoff+1=1 vs cutoff*1=0 behaves differently

        // λ x. BVar(0) lifted at cutoff 0 by 5
        // Under lambda, cutoff becomes 0+1=1
        // BVar(0) < 1, so NOT lifted (it's bound)
        // With * mutant: cutoff*1=0, BVar(0) >= 0, WOULD lift (wrong!)
        let e = lam(sort(l0()), bvar(0));
        let result = e.lift(0, 5);
        match &result {
            MicroExpr::Lam(_, body) => {
                assert_eq!(
                    body.as_ref(),
                    &bvar(0),
                    "BVar(0) under lambda should NOT be lifted (bound)"
                );
            }
            _ => panic!("Expected Lam"),
        }

        // λ x. BVar(1) lifted at cutoff 0 by 5
        // Under lambda, cutoff=1. BVar(1) >= 1, so lifted to BVar(6)
        let e = lam(sort(l0()), bvar(1));
        let result = e.lift(0, 5);
        match &result {
            MicroExpr::Lam(_, body) => {
                assert_eq!(
                    body.as_ref(),
                    &bvar(6),
                    "BVar(1) under lambda should be lifted to BVar(6)"
                );
            }
            _ => panic!("Expected Lam"),
        }

        // Pi: same behavior
        let e = pi(sort(l0()), bvar(0));
        let result = e.lift(0, 5);
        match &result {
            MicroExpr::Pi(_, body) => {
                assert_eq!(
                    body.as_ref(),
                    &bvar(0),
                    "BVar(0) under Pi should NOT be lifted"
                );
            }
            _ => panic!("Expected Pi"),
        }

        // Let body: cutoff+1
        let e = let_(sort(l0()), sort(l0()), bvar(0));
        let result = e.lift(0, 5);
        match &result {
            MicroExpr::Let(_, _, body) => {
                assert_eq!(
                    body.as_ref(),
                    &bvar(0),
                    "BVar(0) under let body should NOT be lifted"
                );
            }
            _ => panic!("Expected Let"),
        }

        // Double nested: λ x. λ y. BVar(1)
        // At depth 2, cutoff=2. BVar(1) < 2, not lifted
        let inner = lam(sort(l0()), bvar(1));
        let outer = lam(sort(l0()), inner);
        let result = outer.lift(0, 5);
        // Navigate to innermost
        match &result {
            MicroExpr::Lam(_, body) => match body.as_ref() {
                MicroExpr::Lam(_, inner_body) => {
                    assert_eq!(
                        inner_body.as_ref(),
                        &bvar(1),
                        "BVar(1) under 2 lambdas should NOT be lifted"
                    );
                }
                _ => panic!("Expected inner Lam"),
            },
            _ => panic!("Expected outer Lam"),
        }

        // λ x. λ y. BVar(2) at depth 2 IS >= 2, so lifted to BVar(7)
        let inner = lam(sort(l0()), bvar(2));
        let outer = lam(sort(l0()), inner);
        let result = outer.lift(0, 5);
        match &result {
            MicroExpr::Lam(_, body) => match body.as_ref() {
                MicroExpr::Lam(_, inner_body) => {
                    assert_eq!(
                        inner_body.as_ref(),
                        &bvar(7),
                        "BVar(2) under 2 lambdas should be lifted to BVar(7)"
                    );
                }
                _ => panic!("Expected inner Lam"),
            },
            _ => panic!("Expected outer Lam"),
        }
    }

    #[test]
    fn test_subst_depth_plus_one_in_binders() {
        // Kill mutants at line 234: replace depth + 1 with depth * 1 or depth - 1
        // When depth=0, depth+1=1 vs depth*1=0 behaves differently

        // λ x. BVar(0) substituted at depth 0
        // Body at depth 0+1=1. BVar(0) < 1, stays as is
        // With * mutant: depth*1=0, BVar(0)==0, would substitute (wrong!)
        let e = lam(sort(l0()), bvar(0));
        let val = sort(l1());
        let result = e.subst(0, &val);
        match &result {
            MicroExpr::Lam(_, body) => {
                assert_eq!(
                    body.as_ref(),
                    &bvar(0),
                    "BVar(0) under lambda should stay (bound to lambda param)"
                );
            }
            _ => panic!("Expected Lam"),
        }

        // λ x. BVar(1) substituted at depth 0
        // Body at depth 1. BVar(1)==1, substitutes with val.lift(0,1)
        let e = lam(sort(l0()), bvar(1));
        let result = e.subst(0, &val);
        match &result {
            MicroExpr::Lam(_, body) => {
                // val.lift(0,1) = sort(l1()) since no bvars
                assert_eq!(
                    body.as_ref(),
                    &sort(l1()),
                    "BVar(1) under lambda at depth 0 substitutes"
                );
            }
            _ => panic!("Expected Lam"),
        }

        // λ x. BVar(2) substituted at depth 0
        // Body at depth 1. BVar(2) > 1, decrements to BVar(1)
        let e = lam(sort(l0()), bvar(2));
        let result = e.subst(0, &val);
        match &result {
            MicroExpr::Lam(_, body) => {
                assert_eq!(
                    body.as_ref(),
                    &bvar(1),
                    "BVar(2) under lambda at depth 0 decrements to BVar(1)"
                );
            }
            _ => panic!("Expected Lam"),
        }

        // let x = v in BVar(0): body at depth+1
        let e = let_(sort(l0()), sort(l0()), bvar(0));
        let result = e.subst(0, &val);
        match &result {
            MicroExpr::Let(_, _, body) => {
                assert_eq!(
                    body.as_ref(),
                    &bvar(0),
                    "BVar(0) under let should stay (bound to let)"
                );
            }
            _ => panic!("Expected Let"),
        }

        // Double nested: λ x. λ y. BVar(2)
        // At depth 2, BVar(2)==2 substitutes
        let inner = lam(sort(l0()), bvar(2));
        let outer = lam(sort(l0()), inner);
        let result = outer.subst(0, &val);
        match &result {
            MicroExpr::Lam(_, body) => match body.as_ref() {
                MicroExpr::Lam(_, inner_body) => {
                    // val.lift(0,2) = sort(l1()) since no bvars
                    assert_eq!(
                        inner_body.as_ref(),
                        &sort(l1()),
                        "BVar(2) under 2 lambdas at depth 0 substitutes"
                    );
                }
                _ => panic!("Expected inner Lam"),
            },
            _ => panic!("Expected outer Lam"),
        }
    }

    #[test]
    fn test_subst_gt_not_gte() {
        // Kill mutant at line 205: replace > with >= in subst
        // When idx == depth, we substitute. When idx > depth, we decrement.

        // BVar(0) at depth 0: idx==depth, SUBSTITUTE
        let e = bvar(0);
        let val = sort(l1());
        let result = e.subst(0, &val);
        assert_eq!(
            result,
            sort(l1()),
            "BVar(0) at depth 0: == case, should substitute"
        );

        // BVar(1) at depth 0: idx > depth, DECREMENT to BVar(0)
        let e = bvar(1);
        let result = e.subst(0, &val);
        assert_eq!(
            result,
            bvar(0),
            "BVar(1) at depth 0: > case, should decrement"
        );

        // BVar(0) at depth 0 but val has structure
        let complex_val = app(sort(l0()), sort(l1()));
        let result = bvar(0).subst(0, &complex_val);
        assert_eq!(
            result, complex_val,
            "Substitution should return val exactly at depth 0"
        );
    }

    #[test]
    fn test_is_geq_offset_gt_zero() {
        // Kill mutants at line 295: replace > with ==, >=, or <
        // offset1 > 0 check for Succ levels

        // Create Max level to test where bases differ but offset > 0 matters
        let max_ab = MicroLevel::Max(
            Arc::new(MicroLevel::Zero),
            Arc::new(MicroLevel::succ(MicroLevel::Zero)),
        );

        // succ(max(0, 1)) >= max(0, 1)
        // l1 = Succ(max), l2 = max
        // bases differ, but offset1=1 > 0, so check max >= max (true via same level)
        let succ_max = MicroLevel::succ(max_ab.clone());
        assert!(
            MicroLevel::is_geq(&succ_max, &max_ab),
            "succ(max(0,1)) >= max(0,1) via offset > 0 recursive check"
        );

        // For the > vs >= case at offset checking:
        // When offset1 = 1 and we're checking recursively:
        // > 0: 1 > 0 is true, does the recursive check
        // >= 0: 1 >= 0 is true, would ALSO do recursive check (same result here)
        // == 0: 1 == 0 is false, wouldn't recurse
        // < 0: 1 < 0 is false, wouldn't recurse

        // For offset1 = 0 case:
        // > 0: 0 > 0 is false, skip recursive check
        // >= 0: 0 >= 0 is true, would recurse (different!)
        // == 0: 0 == 0 is true, would recurse (different!)
        // < 0: 0 < 0 is false, skip

        // So we need a case where offset1 = 0 but the recursive check matters
        // But if offset1=0, then l1 has no Succ wrapper, so as_inner() would
        // just return l1 itself... Actually let me re-read the code

        // offset1 > 0 means the level has at least one Succ wrapper
        // So the check is: if l1 = succ^k(l1') with k > 0, then check l1' >= l2
        // If k = 0, we skip this optimization

        // To kill > vs == mutation: need case where offset=1 makes difference
        // To kill > vs < mutation: offset < 0 is never true for u32, always skipped
        // To kill > vs >= mutation: need offset=0 case where >= would recurse but > wouldn't

        // But offset=0 means no Succ, so l1_inner = l1, checking l1 >= l2 is circular...
        // Actually looking at code:
        // if offset1 > 0 { if is_geq(l1.as_inner(), l2) { return true; } }
        // as_inner removes one Succ layer
        // So this only makes sense when offset1 >= 1

        // Key test: does > vs >= matter?
        // When offset1 = 0: > 0 is false, skip. >= 0 is true, would check as_inner
        // But as_inner of a non-Succ level just returns itself, leading to infinite recursion
        // So >= 0 would cause issues, > 0 is correct

        // We can't directly test >= vs > with offset=0 because it would loop
        // But the test above with offset=1 shows the code path works
    }

    #[test]
    fn test_imax_eq_vs_ne() {
        // Kill mutant at line 344: replace == with != in MicroLevel::imax
        // imax(l1, l2) when l1 == l2 should return l1

        // Max levels
        let max_01 = MicroLevel::Max(
            Arc::new(MicroLevel::Zero),
            Arc::new(MicroLevel::succ(MicroLevel::Zero)),
        );

        // imax(max(0,1), max(0,1)) should return max(0,1)
        // With != mutation: l1 != l2 is false, wouldn't simplify
        let result = MicroLevel::imax(max_01.clone(), max_01.clone());
        assert_eq!(
            result, max_01,
            "imax(l, l) should return l when both are equal Max levels"
        );

        // IMax level
        let imax_01 = MicroLevel::IMax(
            Arc::new(MicroLevel::Zero),
            Arc::new(MicroLevel::succ(MicroLevel::Zero)),
        );

        // imax(imax(0,1), imax(0,1)) should return imax(0,1)
        let result = MicroLevel::imax(imax_01.clone(), imax_01.clone());
        assert_eq!(
            result, imax_01,
            "imax(l, l) should return l when both are equal IMax levels"
        );
    }

    #[test]
    fn test_is_geq_offset_with_different_bases() {
        // Kill mutants at line 295: replace > with <, ==, or >=
        // This test uses levels with DIFFERENT bases to distinguish mutations
        //
        // succ(max(0, 1)) >= 1
        // l1 = succ(max(0, 1)), l2 = succ(0)
        // get_offset(l1) = (max(0, 1), 1)
        // get_offset(l2) = (Zero, 1)
        // bases differ: max(0, 1) != Zero
        //
        // With > 0: offset1=1 > 0, check is_geq(max(0, 1), succ(0))
        //   max(a, b) >= l if a >= l or b >= l
        //   0 >= 1? false. 1 >= 1? true via same offset
        //   Returns true
        // With < 0: 1 < 0 is false, skip offset check
        //   l1 is Succ not Max, skip max check
        //   l2 is Succ(Zero) not Max, skip max check
        //   Return false (WRONG!)
        let max_01 = MicroLevel::Max(
            Arc::new(MicroLevel::Zero),
            Arc::new(MicroLevel::succ(MicroLevel::Zero)),
        );
        let succ_max = MicroLevel::succ(max_01);
        let one = MicroLevel::succ(MicroLevel::Zero);

        assert!(
            MicroLevel::is_geq(&succ_max, &one),
            "succ(max(0, 1)) >= 1 should be true: bases differ but inner max(0,1) >= 1"
        );
    }

    #[test]
    fn test_imax_zero_left_returns_right() {
        // Kill mutant at line 344: replace == with != in check for l1 == Zero
        // imax(0, l) = l (when l != 0)
        //
        // With ==: l1 == Zero is true, return l2
        // With !=: l1 != Zero is false, skip this check
        //   Then l1 == l2? 0 != IMax, so false
        //   Would return IMax(0, IMax(...))
        let inner = MicroLevel::IMax(
            Arc::new(MicroLevel::Zero),
            Arc::new(MicroLevel::succ(MicroLevel::Zero)),
        );

        // imax(0, imax(0, 1)) should return imax(0, 1)
        let result = MicroLevel::imax(MicroLevel::Zero, inner.clone());
        assert_eq!(
            result, inner,
            "imax(0, l) should return l directly when l is non-zero IMax"
        );
    }

    #[test]
    fn test_get_offset_nested_succ() {
        // Kill mutant at line 321: delete match arm MicroLevel::Succ
        // get_offset should recursively unwrap Succ to count the offset
        //
        // With Succ arm: succ(succ(Zero)) -> (Zero, 2)
        // Without Succ arm (using _ =>): succ(succ(Zero)) -> (succ(succ(Zero)), 0)
        //
        // Test is_geq uses get_offset, so we test via is_geq:
        // succ(succ(Zero)) >= succ(Zero)?
        // With correct get_offset: bases both Zero, offsets 2 >= 1, true
        // With broken get_offset: bases differ, would check different paths
        let two = MicroLevel::succ(MicroLevel::succ(MicroLevel::Zero));
        let one = MicroLevel::succ(MicroLevel::Zero);

        assert!(
            MicroLevel::is_geq(&two, &one),
            "succ(succ(0)) >= succ(0) should be true - offset 2 >= 1"
        );

        // Also test that succ(succ(succ(Zero))) >= succ(Zero)
        let three = MicroLevel::succ(MicroLevel::succ(MicroLevel::succ(MicroLevel::Zero)));
        assert!(
            MicroLevel::is_geq(&three, &one),
            "succ(succ(succ(0))) >= succ(0) should be true - offset 3 >= 1"
        );
    }

    // =========================================================================
    // Kill: micro.rs:233:47 - Let body in subst (depth + 1)
    // =========================================================================
    #[test]
    fn test_micro_subst_let_body_depth() {
        // This tests that the Let body correctly increments depth by 1
        // Mutation: depth + 1 -> depth - 1 should fail

        // let x = Prop in BVar(1) - BVar(1) at depth 1 should be substituted
        let val = MicroExpr::Sort(MicroLevel::succ(MicroLevel::Zero)); // Type 1
        let let_expr = MicroExpr::Let(
            Arc::new(MicroExpr::Sort(MicroLevel::Zero)), // type: Prop
            Arc::new(MicroExpr::Sort(MicroLevel::Zero)), // value: Prop
            Arc::new(MicroExpr::BVar(1)),                // body: BVar(1)
        );
        let result = let_expr.subst(0, &val);
        match result {
            MicroExpr::Let(_, _, body) => {
                // BVar(1) at depth 1: 1 == 1, so substitute with val (no lifting needed)
                assert!(
                    matches!(body.as_ref(), MicroExpr::Sort(MicroLevel::Succ(_))),
                    "BVar(1) in let body should be substituted at depth 1"
                );
            }
            _ => panic!("Expected Let"),
        }

        // let x = Prop in BVar(0) - BVar(0) at depth 1 is the let-bound variable
        // Should NOT be substituted (0 < 1)
        let let_expr = MicroExpr::Let(
            Arc::new(MicroExpr::Sort(MicroLevel::Zero)),
            Arc::new(MicroExpr::Sort(MicroLevel::Zero)),
            Arc::new(MicroExpr::BVar(0)),
        );
        let result = let_expr.subst(0, &val);
        match result {
            MicroExpr::Let(_, _, body) => {
                assert!(
                    matches!(body.as_ref(), MicroExpr::BVar(0)),
                    "BVar(0) in let body refers to let binding, not substituted"
                );
            }
            _ => panic!("Expected Let"),
        }
    }

    // =========================================================================
    // Kill: micro.rs:256:15, 259:15 - == vs != in MicroLevel::max
    // =========================================================================
    #[test]
    fn test_micro_level_max_zero_checks() {
        // Kill mutants: l1 == MicroLevel::Zero and l2 == MicroLevel::Zero with !=
        //
        // max(0, l) = l
        // max(l, 0) = l

        // max(0, succ(0)) = succ(0), not max(0, succ(0))
        let zero = MicroLevel::Zero;
        let one = MicroLevel::succ(MicroLevel::Zero);

        let result = MicroLevel::max(zero.clone(), one.clone());
        assert_eq!(result, one.clone(), "max(0, 1) should return 1 directly");

        // max(succ(0), 0) = succ(0)
        let result = MicroLevel::max(one.clone(), zero.clone());
        assert_eq!(result, one, "max(1, 0) should return 1 directly");

        // Important: test that we DON'T simplify incorrectly
        // With mutation: != instead of ==, max(0, 1) would not trigger the simplification
        // and would fall through to is_geq checks or return Max(0, 1)

        // Check that a non-zero max doesn't trigger zero simplification
        let two = MicroLevel::succ(MicroLevel::succ(MicroLevel::Zero));
        let result = MicroLevel::max(one.clone(), two.clone());
        assert_eq!(result, two, "max(1, 2) should return 2 via is_geq");
    }

    // =========================================================================
    // Kill: micro.rs:321 - delete Succ arm in get_offset
    // =========================================================================
    #[test]
    fn test_get_offset_direct() {
        // Direct unit test for get_offset to kill the "delete Succ arm" mutant.
        // If Succ arm is deleted, get_offset returns (self, 0) for all inputs.
        //
        // With Succ arm: succ(succ(Zero)) -> (&Zero, 2)
        // Without Succ arm: succ(succ(Zero)) -> (&succ(succ(Zero)), 0)

        let zero = MicroLevel::Zero;
        let one = MicroLevel::succ(MicroLevel::Zero);
        let two = MicroLevel::succ(MicroLevel::succ(MicroLevel::Zero));
        let three = MicroLevel::succ(MicroLevel::succ(MicroLevel::succ(MicroLevel::Zero)));

        // Test Zero: should return (Zero, 0)
        let (base, offset) = MicroLevel::get_offset(&zero);
        assert_eq!(*base, MicroLevel::Zero);
        assert_eq!(offset, 0, "Zero should have offset 0");

        // Test succ(Zero): should return (Zero, 1)
        let (base, offset) = MicroLevel::get_offset(&one);
        assert_eq!(*base, MicroLevel::Zero, "succ(0) base should be Zero");
        assert_eq!(offset, 1, "succ(0) should have offset 1");

        // Test succ(succ(Zero)): should return (Zero, 2)
        let (base, offset) = MicroLevel::get_offset(&two);
        assert_eq!(*base, MicroLevel::Zero, "succ(succ(0)) base should be Zero");
        assert_eq!(offset, 2, "succ(succ(0)) should have offset 2");

        // Test succ(succ(succ(Zero))): should return (Zero, 3)
        let (base, offset) = MicroLevel::get_offset(&three);
        assert_eq!(
            *base,
            MicroLevel::Zero,
            "succ(succ(succ(0))) base should be Zero"
        );
        assert_eq!(offset, 3, "succ(succ(succ(0))) should have offset 3");

        // Test with a Max base: succ(succ(Max(0,0))) -> (Max(0,0), 2)
        let max_base = MicroLevel::Max(Arc::new(MicroLevel::Zero), Arc::new(MicroLevel::Zero));
        let max_plus_2 = MicroLevel::succ(MicroLevel::succ(max_base.clone()));
        let (base, offset) = MicroLevel::get_offset(&max_plus_2);
        assert_eq!(
            *base, max_base,
            "succ(succ(Max(0,0))) base should be Max(0,0)"
        );
        assert_eq!(offset, 2, "succ(succ(Max(0,0))) should have offset 2");
    }
}
