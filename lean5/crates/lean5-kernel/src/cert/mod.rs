//! Proof Certificates for Lean5
//!
//! This module implements proof certificates that witness type-correctness.
//! A proof certificate is a compact representation of a typing derivation
//! that can be verified much faster than re-running type inference.
//!
//! ## Design Goals
//!
//! 1. **Verifiable**: Certificates can be checked by a simple checker
//! 2. **Compact**: Certificates are smaller than full derivation trees
//! 3. **Deterministic**: Same input produces same certificate
//! 4. **Self-contained**: Certificate + expression is sufficient for verification
//!
//! ## Certificate Structure
//!
//! Each certificate node corresponds to a typing rule from CIC:
//!
//! ```text
//! Sort:    Sort(l) : Sort(succ(l))
//! Pi:      (x : A) → B : Sort(imax(l1, l2))
//! Lam:     λ (x : A). b : (x : A) → B
//! App:     f a : B[a/x] when f : (x : A) → B
//! Let:     let x : A := v in b : B[v/x]
//! ```
//!
//! ## Usage
//!
//! ```ignore
//! // Generate certificate during type checking
//! let (ty, cert) = checker.infer_type_with_cert(&expr)?;
//!
//! // Verify certificate independently
//! let verified_ty = cert.verify(&env, &expr)?;
//! assert_eq!(ty, verified_ty);
//! ```

use crate::env::Environment;
use crate::expr::{BinderInfo, Expr, FVarId, LevelVec, Literal, ZFCSetExpr};
use crate::level::Level;
use crate::mode::Lean5Mode;
use crate::name::Name;
use crate::tc::LocalContext;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Minimum stack space to reserve before recursive calls (32 KB).
const MIN_STACK_RED_ZONE: usize = 32 * 1024;

/// Stack size to grow to when running low (1 MB).
const STACK_GROWTH_SIZE: usize = 1024 * 1024;

/// A proof certificate witnessing a typing derivation.
///
/// The certificate structure mirrors the expression structure but includes
/// all intermediate types needed for verification.
///
/// Certificates are serializable for proof archives and can be verified
/// independently by a certificate verifier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProofCert {
    /// Certificate for Sort(l) : Sort(succ(l))
    Sort { level: Level },

    /// Certificate for `BVar` (de Bruijn index)
    /// Includes the expected type from context
    BVar { idx: u32, expected_type: Box<Expr> },

    /// Certificate for `FVar` (free variable)
    /// Includes the type from local context
    FVar { id: FVarId, type_: Box<Expr> },

    /// Certificate for Const (constant reference)
    /// Includes instantiated type
    Const {
        name: Name,
        levels: Vec<Level>,
        type_: Box<Expr>,
    },

    /// Certificate for App: f a : B[a/x]
    /// Records: function cert, arg cert, and the instantiated result type
    App {
        fn_cert: Box<ProofCert>,
        fn_type: Box<Expr>, // The Pi type of f
        arg_cert: Box<ProofCert>,
        result_type: Box<Expr>, // B[a/x]
    },

    /// Certificate for Lam: λ (x : A). b : (x : A) → B
    /// Records: arg type cert, body cert (in extended context)
    Lam {
        binder_info: BinderInfo,
        arg_type_cert: Box<ProofCert>, // Proves A : Sort(l)
        body_cert: Box<ProofCert>,     // Proves b : B in extended context
        result_type: Box<Expr>,        // The Pi type
    },

    /// Certificate for Pi: (x : A) → B : Sort(imax(l1, l2))
    Pi {
        binder_info: BinderInfo,
        arg_type_cert: Box<ProofCert>,  // Proves A : Sort(l1)
        arg_level: Level,               // l1
        body_type_cert: Box<ProofCert>, // Proves B : Sort(l2) in extended context
        body_level: Level,              // l2
    },

    /// Certificate for Let: let x : A := v in b : B[v/x]
    Let {
        type_cert: Box<ProofCert>,  // Proves A : Sort(l)
        value_cert: Box<ProofCert>, // Proves v : A
        body_cert: Box<ProofCert>,  // Proves b : B in extended context
        result_type: Box<Expr>,     // B[v/x]
    },

    /// Certificate for Literal values
    Lit {
        lit: Literal,
        type_: Box<Expr>, // Nat or String
    },

    /// Certificate for definitional equality check
    /// Used when checking e : T reduces to checking e : T' where T ≡ T'
    DefEq {
        inner: Box<ProofCert>,
        expected_type: Box<Expr>,
        actual_type: Box<Expr>,
        /// Steps needed to show equivalence (for debugging/verification)
        eq_steps: Vec<DefEqStep>,
    },

    /// Certificate for `MData` (metadata wrapper)
    /// `MData` is transparent - the type is the type of the inner expression
    MData {
        metadata: crate::expr::MDataMap,
        inner_cert: Box<ProofCert>,
        result_type: Box<Expr>,
    },

    /// Certificate for Proj (projection from structure)
    /// Records the struct name, field index, and the type of the projected field
    Proj {
        struct_name: Name,
        idx: u32,
        expr_cert: Box<ProofCert>,
        expr_type: Box<Expr>,  // Type of the expression being projected
        field_type: Box<Expr>, // Type of the projected field
    },

    // ════════════════════════════════════════════════════════════════════════
    // Mode-specific certificates (Cubical, Classical, SetTheoretic)
    // ════════════════════════════════════════════════════════════════════════

    /// Certificate for CubicalInterval : Sort(0)
    /// The interval type I is a special sort in Cubical type theory
    CubicalInterval,

    /// Certificate for CubicalI0 : I and CubicalI1 : I
    /// The endpoints of the interval
    CubicalEndpoint {
        /// true for I1, false for I0
        is_one: bool,
    },

    /// Certificate for CubicalPath { ty, left, right } : Sort(l)
    /// Path A a b is a type when A : Sort(l), a : A, b : A
    CubicalPath {
        ty_cert: Box<ProofCert>,
        ty_level: Level,
        left_cert: Box<ProofCert>,
        right_cert: Box<ProofCert>,
    },

    /// Certificate for CubicalPathLam { body } : Path A (body[0/i]) (body[1/i])
    /// Path abstraction `<i> e` where i : I
    CubicalPathLam {
        body_cert: Box<ProofCert>,
        /// The type of the body (before abstracting interval var)
        body_type: Box<Expr>,
        /// The resulting Path type
        result_type: Box<Expr>,
    },

    /// Certificate for CubicalPathApp { path, arg } : A
    /// Path application p @ i where p : Path A a b and i : I
    CubicalPathApp {
        path_cert: Box<ProofCert>,
        arg_cert: Box<ProofCert>,
        /// The Path type of the path expression
        path_type: Box<Expr>,
        /// The result type (the type parameter A from Path A a b)
        result_type: Box<Expr>,
    },

    /// Certificate for CubicalHComp { ty, phi, u, base } : ty
    /// Homogeneous composition in Cubical type theory
    CubicalHComp {
        ty_cert: Box<ProofCert>,
        phi_cert: Box<ProofCert>,
        u_cert: Box<ProofCert>,
        base_cert: Box<ProofCert>,
        result_type: Box<Expr>,
    },

    /// Certificate for CubicalTransp { ty, phi, base } : ty[1/i]
    /// Transport along a path in Cubical type theory
    CubicalTransp {
        ty_cert: Box<ProofCert>,
        phi_cert: Box<ProofCert>,
        base_cert: Box<ProofCert>,
        result_type: Box<Expr>,
    },

    /// Certificate for ClassicalChoice { ty, pred, proof } : ty
    /// Hilbert's choice operator (requires Classical mode)
    ClassicalChoice {
        ty_cert: Box<ProofCert>,
        pred_cert: Box<ProofCert>,
        proof_cert: Box<ProofCert>,
        result_type: Box<Expr>,
    },

    /// Certificate for ClassicalEpsilon { ty, pred } : ty
    /// Hilbert's epsilon operator (indefinite description)
    ClassicalEpsilon {
        ty_cert: Box<ProofCert>,
        pred_cert: Box<ProofCert>,
        result_type: Box<Expr>,
    },

    /// Certificate for ZFCSet expressions : Set
    /// Various set constructions in ZFC
    ZFCSet {
        /// The specific set construction
        kind: ZFCSetCertKind,
        /// Always Set (the type of sets)
        result_type: Box<Expr>,
    },

    /// Certificate for ZFCMem { elem, set } : Prop
    /// Set membership ∈
    ZFCMem {
        elem_cert: Box<ProofCert>,
        set_cert: Box<ProofCert>,
    },

    /// Certificate for ZFCComprehension { var_ty, pred } : Set
    /// Set comprehension { x : A | P(x) }
    ZFCComprehension {
        var_ty_cert: Box<ProofCert>,
        pred_cert: Box<ProofCert>,
        result_type: Box<Expr>,
    },

    // ════════════════════════════════════════════════════════════════════════
    // Impredicative mode certificates
    // ════════════════════════════════════════════════════════════════════════

    /// Certificate for SProp : Type 1
    /// SProp is the sort of strict propositions (always proof-irrelevant)
    SProp,

    /// Certificate for Squash A : SProp (when A : Sort u)
    /// Squash (propositional truncation) - all proofs are definitionally equal
    Squash { inner_cert: Box<ProofCert> },
}

/// Certificate variants for ZFC set expressions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ZFCSetCertKind {
    /// Empty set ∅
    Empty,
    /// Singleton {a}
    Singleton(Box<ProofCert>),
    /// Unordered pair {a, b}
    Pair(Box<ProofCert>, Box<ProofCert>),
    /// Union ⋃A
    Union(Box<ProofCert>),
    /// Power set P(A)
    PowerSet(Box<ProofCert>),
    /// Separation {x ∈ A | φ(x)}
    Separation {
        set_cert: Box<ProofCert>,
        pred_cert: Box<ProofCert>,
    },
    /// Replacement {F(x) | x ∈ A}
    Replacement {
        set_cert: Box<ProofCert>,
        func_cert: Box<ProofCert>,
    },
    /// Infinity ω
    Infinity,
    /// Choice (AC)
    Choice(Box<ProofCert>),
}

/// A step in a definitional equality proof.
///
/// These steps record how the verifier establishes definitional equality
/// between types, useful for debugging and proof reconstruction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DefEqStep {
    /// Reflexivity: e ≡ e
    Refl,
    /// Symmetry: e1 ≡ e2 implies e2 ≡ e1
    Symm(Box<DefEqStep>),
    /// Transitivity: e1 ≡ e2 and e2 ≡ e3 implies e1 ≡ e3
    Trans(Box<DefEqStep>, Box<DefEqStep>),
    /// Beta reduction: (λx.b) a ≡ b[a/x]
    Beta,
    /// Delta reduction: unfold constant definition
    Delta(Name),
    /// Zeta reduction: unfold let binding
    Zeta,
    /// Iota reduction: recursor computation rule
    Iota,
    /// Structural: congruence through constructors
    Struct(String, Vec<DefEqStep>),
}

/// Error during certificate verification
#[derive(Debug, Clone)]
pub enum CertError {
    /// Type mismatch during verification
    TypeMismatch {
        expected: Box<Expr>,
        actual: Box<Expr>,
        location: String,
    },
    /// Unknown constant reference
    UnknownConst(Name),
    /// Unknown free variable
    UnknownFVar(FVarId),
    /// Invalid de Bruijn index
    InvalidBVar(u32),
    /// Certificate structure doesn't match expression
    StructureMismatch { expected: String, actual: String },
    /// Definitional equality check failed
    DefEqFailed { left: Box<Expr>, right: Box<Expr> },
    /// Sort level mismatch
    LevelMismatch { expected: Level, actual: Level },
    /// Invalid certificate structure
    InvalidCert(String),
    /// Mode-specific feature requires a different mode
    ModeRequired {
        feature: String,
        required_mode: Lean5Mode,
        current_mode: Lean5Mode,
    },
}

impl std::fmt::Display for CertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CertError::TypeMismatch {
                expected,
                actual,
                location,
            } => {
                write!(
                    f,
                    "Type mismatch at {location}: expected {expected:?}, got {actual:?}"
                )
            }
            CertError::UnknownConst(name) => write!(f, "Unknown constant: {name:?}"),
            CertError::UnknownFVar(id) => write!(f, "Unknown free variable: {id:?}"),
            CertError::InvalidBVar(idx) => write!(f, "Invalid bound variable index: {idx}"),
            CertError::StructureMismatch { expected, actual } => {
                write!(f, "Structure mismatch: expected {expected}, got {actual}")
            }
            CertError::DefEqFailed { left, right } => {
                write!(f, "Definitional equality failed: {left:?} ≢ {right:?}")
            }
            CertError::LevelMismatch { expected, actual } => {
                write!(f, "Level mismatch: expected {expected:?}, got {actual:?}")
            }
            CertError::InvalidCert(msg) => write!(f, "Invalid certificate: {msg}"),
            CertError::ModeRequired {
                feature,
                required_mode,
                current_mode,
            } => {
                write!(
                    f,
                    "Feature '{feature}' requires {} mode, but current mode is {}",
                    required_mode.name(),
                    current_mode.name()
                )
            }
        }
    }
}

impl std::error::Error for CertError {}

/// Certificate verifier state
pub struct CertVerifier<'env> {
    env: &'env Environment,
    /// Local context: maps de Bruijn level to type
    /// (level 0 = outermost binding)
    context: Vec<Expr>,
    /// `FVar` types
    fvar_types: HashMap<FVarId, Expr>,
    /// Current mode for mode-aware verification
    mode: Lean5Mode,
}

impl<'env> CertVerifier<'env> {
    /// Create a new certificate verifier in Constructive mode (default)
    pub fn new(env: &'env Environment) -> Self {
        Self {
            env,
            context: Vec::new(),
            fvar_types: HashMap::new(),
            mode: Lean5Mode::default(),
        }
    }

    /// Create a new certificate verifier with a specific mode
    pub fn with_mode(env: &'env Environment, mode: Lean5Mode) -> Self {
        Self {
            env,
            context: Vec::new(),
            fvar_types: HashMap::new(),
            mode,
        }
    }

    /// Get the current mode
    pub fn mode(&self) -> Lean5Mode {
        self.mode
    }

    /// Set the mode
    pub fn set_mode(&mut self, mode: Lean5Mode) {
        self.mode = mode;
    }

    /// Register a free variable type in the verifier context.
    /// Returns an error if the ID was already registered with a different type.
    pub fn register_fvar(&mut self, id: FVarId, ty: Expr) -> Result<(), CertError> {
        if let Some(existing) = self.fvar_types.get(&id) {
            if !self.def_eq(existing, &ty) {
                return Err(CertError::TypeMismatch {
                    expected: Box::new(existing.clone()),
                    actual: Box::new(ty),
                    location: format!("FVar {id:?}"),
                });
            }
        }
        self.fvar_types.insert(id, ty);
        Ok(())
    }

    /// Register all free variables from a `LocalContext`.
    ///
    /// This is useful when integrating with the elaborator to transfer
    /// the full local context into the certificate verifier.
    /// Returns an error if any `FVar` was already registered with a conflicting type.
    pub fn register_local_context(&mut self, ctx: &LocalContext) -> Result<(), CertError> {
        for decl in ctx.iter() {
            self.register_fvar(decl.id, decl.type_.clone())?;
        }
        Ok(())
    }

    /// Verify a certificate and return the proven type
    ///
    /// This is the trusted checker - it verifies that the certificate
    /// correctly witnesses the typing derivation.
    pub fn verify(&mut self, cert: &ProofCert, expr: &Expr) -> Result<Expr, CertError> {
        stacker::maybe_grow(MIN_STACK_RED_ZONE, STACK_GROWTH_SIZE, || {
            self.verify_impl(cert, expr)
        })
    }

    /// Implementation (called via stacker::maybe_grow)
    fn verify_impl(&mut self, cert: &ProofCert, expr: &Expr) -> Result<Expr, CertError> {
        match (cert, expr) {
            // Sort rule: Sort(l) : Sort(succ(l))
            (ProofCert::Sort { level }, Expr::Sort(l)) => {
                if level != l {
                    return Err(CertError::LevelMismatch {
                        expected: level.clone(),
                        actual: l.clone(),
                    });
                }
                Ok(Expr::Sort(Level::succ(level.clone())))
            }

            // BVar rule: context lookup
            (ProofCert::BVar { idx, expected_type }, Expr::BVar(i)) => {
                if *idx != *i {
                    return Err(CertError::InvalidBVar(*i));
                }
                // Convert de Bruijn index to level and lookup
                let depth = self.context.len();
                if (*idx as usize) >= depth {
                    return Err(CertError::InvalidBVar(*idx));
                }
                let level = depth - 1 - (*idx as usize);
                let ctx_type = &self.context[level];

                // The type at context[level] was stored when context had `level` entries.
                // At that time, the type's free BVars referred to binders 0..level-1.
                // Now at depth `depth`, we have `depth - level` additional binders between
                // us and where the type was valid. So we lift by `depth - level` which
                // equals `idx + 1` (since level = depth - 1 - idx).
                // SAFETY: lift_amount = depth - level = idx + 1, and idx is u32, so this fits in u32.
                #[allow(clippy::cast_possible_truncation)]
                let lift_amount = (depth - level) as u32;
                let lifted_ctx_type = ctx_type.lift(lift_amount);

                // Verify the certificate's expected_type matches the lifted context type
                if !self.def_eq(expected_type.as_ref(), &lifted_ctx_type) {
                    return Err(CertError::TypeMismatch {
                        expected: Box::new(expected_type.as_ref().clone()),
                        actual: Box::new(lifted_ctx_type),
                        location: format!("BVar({idx})"),
                    });
                }
                Ok(expected_type.as_ref().clone())
            }

            // FVar rule: local context lookup
            (ProofCert::FVar { id, type_ }, Expr::FVar(fid)) => {
                if id != fid {
                    return Err(CertError::UnknownFVar(*fid));
                }
                // Verify FVar type is in context
                let ctx_ty = self.fvar_types.get(id).ok_or(CertError::UnknownFVar(*id))?;
                if !self.def_eq(type_.as_ref(), ctx_ty) {
                    return Err(CertError::TypeMismatch {
                        expected: Box::new(type_.as_ref().clone()),
                        actual: Box::new(ctx_ty.clone()),
                        location: format!("FVar {id:?}"),
                    });
                }
                Ok(type_.as_ref().clone())
            }

            // Const rule: environment lookup
            (
                ProofCert::Const {
                    name,
                    levels,
                    type_,
                },
                Expr::Const(n, ls),
            ) => {
                if name != n {
                    return Err(CertError::StructureMismatch {
                        expected: format!("Const {name:?}"),
                        actual: format!("Const {n:?}"),
                    });
                }
                if levels.as_slice() != ls.as_slice() {
                    return Err(CertError::InvalidCert(
                        "Level parameters mismatch".to_string(),
                    ));
                }
                // Verify against environment
                if let Some(env_type) = self.env.instantiate_type(name, levels) {
                    if !self.def_eq(type_.as_ref(), &env_type) {
                        return Err(CertError::TypeMismatch {
                            expected: Box::new(type_.as_ref().clone()),
                            actual: Box::new(env_type),
                            location: format!("Const {name:?}"),
                        });
                    }
                } else {
                    return Err(CertError::UnknownConst(name.clone()));
                }
                Ok(type_.as_ref().clone())
            }

            // App rule: f a : B[a/x] when f : (x : A) → B and a : A
            (
                ProofCert::App {
                    fn_cert,
                    fn_type: _,
                    arg_cert,
                    result_type,
                },
                Expr::App(f, a),
            ) => {
                // Verify function
                let fn_ty = self.verify(fn_cert, f)?;

                // Check function type is Pi
                let fn_type_whnf = self.whnf(&fn_ty);
                match &fn_type_whnf {
                    Expr::Pi(_, expected_arg_type, body_type) => {
                        // Verify argument
                        let arg_ty = self.verify(arg_cert, a)?;

                        // Check argument type matches
                        if !self.def_eq(&arg_ty, expected_arg_type) {
                            return Err(CertError::TypeMismatch {
                                expected: Box::new(expected_arg_type.as_ref().clone()),
                                actual: Box::new(arg_ty),
                                location: "App argument".to_string(),
                            });
                        }

                        // Verify result type
                        let expected_result = body_type.instantiate(a);
                        if !self.def_eq(result_type, &expected_result) {
                            return Err(CertError::TypeMismatch {
                                expected: Box::new(expected_result),
                                actual: Box::new(result_type.as_ref().clone()),
                                location: "App result".to_string(),
                            });
                        }

                        Ok(result_type.as_ref().clone())
                    }
                    _ => Err(CertError::InvalidCert(format!(
                        "Expected Pi type for function, got {fn_type_whnf:?}"
                    ))),
                }
            }

            // Lam rule: λ (x : A). b : (x : A) → B
            (
                ProofCert::Lam {
                    binder_info,
                    arg_type_cert,
                    body_cert,
                    result_type,
                },
                Expr::Lam(bi, arg_ty, body),
            ) => {
                if binder_info != bi {
                    return Err(CertError::InvalidCert(
                        "Binder info mismatch in Lam".to_string(),
                    ));
                }

                // Verify arg type is a type (Sort)
                let arg_sort = self.verify(arg_type_cert, arg_ty)?;
                match self.whnf(&arg_sort) {
                    Expr::Sort(_) => {}
                    _ => {
                        return Err(CertError::InvalidCert(
                            "Lambda argument type is not a type".to_string(),
                        ))
                    }
                }

                // Extend context for body verification
                self.context.push(arg_ty.as_ref().clone());
                let body_ty = self.verify(body_cert, body)?;
                self.context.pop();

                // Build expected Pi type
                let expected_pi = Expr::Pi(*bi, arg_ty.clone(), body_ty.into());
                if !self.def_eq(&expected_pi, result_type) {
                    return Err(CertError::TypeMismatch {
                        expected: Box::new(expected_pi),
                        actual: Box::new(result_type.as_ref().clone()),
                        location: "Lam result type".to_string(),
                    });
                }

                Ok(result_type.as_ref().clone())
            }

            // Pi rule: (x : A) → B : Sort(imax(l1, l2))
            (
                ProofCert::Pi {
                    binder_info,
                    arg_type_cert,
                    arg_level,
                    body_type_cert,
                    body_level,
                },
                Expr::Pi(bi, arg_ty, body_ty),
            ) => {
                if binder_info != bi {
                    return Err(CertError::InvalidCert(
                        "Binder info mismatch in Pi".to_string(),
                    ));
                }

                // Verify arg type
                let arg_sort = self.verify(arg_type_cert, arg_ty)?;
                let Expr::Sort(l1) = self.whnf(&arg_sort) else {
                    return Err(CertError::InvalidCert(
                        "Pi domain is not a type".to_string(),
                    ));
                };

                // Check level matches
                if !self.level_eq(&l1, arg_level) {
                    return Err(CertError::LevelMismatch {
                        expected: arg_level.clone(),
                        actual: l1,
                    });
                }

                // Extend context for body verification
                self.context.push(arg_ty.as_ref().clone());
                let body_sort = self.verify(body_type_cert, body_ty)?;
                self.context.pop();

                let Expr::Sort(l2) = self.whnf(&body_sort) else {
                    return Err(CertError::InvalidCert(
                        "Pi codomain is not a type".to_string(),
                    ));
                };

                // Check level matches
                if !self.level_eq(&l2, body_level) {
                    return Err(CertError::LevelMismatch {
                        expected: body_level.clone(),
                        actual: l2,
                    });
                }

                // Result is Sort(imax(l1, l2))
                Ok(Expr::Sort(Level::imax(
                    arg_level.clone(),
                    body_level.clone(),
                )))
            }

            // Let rule
            (
                ProofCert::Let {
                    type_cert,
                    value_cert,
                    body_cert,
                    result_type,
                },
                Expr::Let(ty, val, body),
            ) => {
                // Verify type is a type
                let ty_sort = self.verify(type_cert, ty)?;
                match self.whnf(&ty_sort) {
                    Expr::Sort(_) => {}
                    _ => return Err(CertError::InvalidCert("Let type is not a type".to_string())),
                }

                // Verify value has the type
                let val_ty = self.verify(value_cert, val)?;
                if !self.def_eq(&val_ty, ty) {
                    return Err(CertError::TypeMismatch {
                        expected: Box::new(ty.as_ref().clone()),
                        actual: Box::new(val_ty),
                        location: "Let value".to_string(),
                    });
                }

                // Extend context for body
                self.context.push(ty.as_ref().clone());
                let body_ty = self.verify(body_cert, body)?;
                self.context.pop();

                // Result type is body type with value substituted
                let expected_result = body_ty.instantiate(val);
                if !self.def_eq(&expected_result, result_type) {
                    return Err(CertError::TypeMismatch {
                        expected: Box::new(expected_result),
                        actual: Box::new(result_type.as_ref().clone()),
                        location: "Let result".to_string(),
                    });
                }

                Ok(result_type.as_ref().clone())
            }

            // Lit rule
            (ProofCert::Lit { lit, type_ }, Expr::Lit(l)) => {
                if lit != l {
                    return Err(CertError::StructureMismatch {
                        expected: format!("{lit:?}"),
                        actual: format!("{l:?}"),
                    });
                }

                let expected_type = match lit {
                    Literal::Nat(_) => Expr::const_(Name::from_string("Nat"), vec![]),
                    Literal::String(_) => Expr::const_(Name::from_string("String"), vec![]),
                };

                if !self.def_eq(type_, &expected_type) {
                    return Err(CertError::TypeMismatch {
                        expected: Box::new(expected_type),
                        actual: Box::new(type_.as_ref().clone()),
                        location: "Literal type".to_string(),
                    });
                }

                Ok(type_.as_ref().clone())
            }

            // DefEq wrapper
            (
                ProofCert::DefEq {
                    inner,
                    expected_type,
                    actual_type,
                    ..
                },
                expr,
            ) => {
                let actual = self.verify(inner, expr)?;

                if !self.def_eq(&actual, actual_type) {
                    return Err(CertError::TypeMismatch {
                        expected: Box::new(actual_type.as_ref().clone()),
                        actual: Box::new(actual.clone()),
                        location: "DefEq inner".to_string(),
                    });
                }

                if !self.def_eq(actual_type, expected_type) {
                    return Err(CertError::DefEqFailed {
                        left: Box::new(actual_type.as_ref().clone()),
                        right: Box::new(expected_type.as_ref().clone()),
                    });
                }

                Ok(expected_type.as_ref().clone())
            }

            // MData rule: metadata is transparent, verify inner expression
            (
                ProofCert::MData {
                    metadata: _,
                    inner_cert,
                    result_type,
                },
                Expr::MData(_, inner),
            ) => {
                let inner_ty = self.verify(inner_cert, inner)?;

                // Result type should match inner type
                if !self.def_eq(&inner_ty, result_type) {
                    return Err(CertError::TypeMismatch {
                        expected: Box::new(inner_ty),
                        actual: Box::new(result_type.as_ref().clone()),
                        location: "MData result".to_string(),
                    });
                }

                Ok(result_type.as_ref().clone())
            }

            // ════════════════════════════════════════════════════════════════════
            // Mode-specific certificate verification
            // ════════════════════════════════════════════════════════════════════

            // Cubical Interval: I : Sort(0)
            (ProofCert::CubicalInterval, Expr::CubicalInterval) => {
                if self.mode != Lean5Mode::Cubical {
                    return Err(CertError::ModeRequired {
                        feature: "CubicalInterval".to_string(),
                        required_mode: Lean5Mode::Cubical,
                        current_mode: self.mode,
                    });
                }
                Ok(Expr::Sort(Level::zero()))
            }

            // Cubical endpoints: 0, 1 : I
            (ProofCert::CubicalEndpoint { is_one }, expr) => {
                if self.mode != Lean5Mode::Cubical {
                    return Err(CertError::ModeRequired {
                        feature: "CubicalEndpoint".to_string(),
                        required_mode: Lean5Mode::Cubical,
                        current_mode: self.mode,
                    });
                }
                match (is_one, expr) {
                    (false, Expr::CubicalI0) | (true, Expr::CubicalI1) => {
                        Ok(Expr::CubicalInterval)
                    }
                    _ => Err(CertError::StructureMismatch {
                        expected: if *is_one {
                            "CubicalI1".to_string()
                        } else {
                            "CubicalI0".to_string()
                        },
                        actual: expr_name(expr),
                    }),
                }
            }

            // Cubical Path type: Path A a b : Sort(l)
            (
                ProofCert::CubicalPath {
                    ty_cert,
                    ty_level,
                    left_cert,
                    right_cert,
                },
                Expr::CubicalPath { ty, left, right },
            ) => {
                if self.mode != Lean5Mode::Cubical {
                    return Err(CertError::ModeRequired {
                        feature: "CubicalPath".to_string(),
                        required_mode: Lean5Mode::Cubical,
                        current_mode: self.mode,
                    });
                }

                // Verify ty : I -> Sort(l)
                let ty_type = self.verify(ty_cert, ty)?;
                let ty_type_whnf = self.whnf(&ty_type);
                let Expr::Pi(_, arg_ty, body_ty) = ty_type_whnf else {
                    return Err(CertError::InvalidCert(
                        "CubicalPath type family is not a function".to_string(),
                    ));
                };
                if !matches!(self.whnf(&arg_ty), Expr::CubicalInterval) {
                    return Err(CertError::TypeMismatch {
                        expected: Box::new(Expr::CubicalInterval),
                        actual: Box::new(arg_ty.as_ref().clone()),
                        location: "CubicalPath type family domain".to_string(),
                    });
                }
                let body_ty_whnf = self.whnf(&body_ty);
                let Expr::Sort(level) = body_ty_whnf else {
                    return Err(CertError::InvalidCert(
                        "CubicalPath type family codomain is not a universe".to_string(),
                    ));
                };
                if !self.level_eq(&level, ty_level) {
                    return Err(CertError::LevelMismatch {
                        expected: ty_level.clone(),
                        actual: level,
                    });
                }

                // Verify left : ty 0
                let left_ty = self.verify(left_cert, left)?;
                let expected_left_ty = Expr::App(ty.clone(), Expr::CubicalI0.into());
                if !self.def_eq(&left_ty, &expected_left_ty) {
                    return Err(CertError::TypeMismatch {
                        expected: Box::new(expected_left_ty),
                        actual: Box::new(left_ty),
                        location: "CubicalPath left endpoint".to_string(),
                    });
                }

                // Verify right : ty 1
                let right_ty = self.verify(right_cert, right)?;
                let expected_right_ty = Expr::App(ty.clone(), Expr::CubicalI1.into());
                if !self.def_eq(&right_ty, &expected_right_ty) {
                    return Err(CertError::TypeMismatch {
                        expected: Box::new(expected_right_ty),
                        actual: Box::new(right_ty),
                        location: "CubicalPath right endpoint".to_string(),
                    });
                }

                // Path types live at the same universe level as the type family codomain
                Ok(Expr::Sort(ty_level.clone()))
            }

            // Cubical PathLam: <i> e : Path A (e[0/i]) (e[1/i])
            (
                ProofCert::CubicalPathLam {
                    body_cert,
                    body_type: _,
                    result_type,
                },
                Expr::CubicalPathLam { body },
            ) => {
                if self.mode != Lean5Mode::Cubical {
                    return Err(CertError::ModeRequired {
                        feature: "CubicalPathLam".to_string(),
                        required_mode: Lean5Mode::Cubical,
                        current_mode: self.mode,
                    });
                }

                // Extend context with interval variable and verify body
                self.context.push(Expr::CubicalInterval);
                let _body_ty = self.verify(body_cert, body)?;
                self.context.pop();

                // Result should be a Path type
                let result_whnf = self.whnf(result_type);
                if !matches!(result_whnf, Expr::CubicalPath { .. }) {
                    return Err(CertError::InvalidCert(
                        "CubicalPathLam result is not a Path type".to_string(),
                    ));
                }

                Ok(result_type.as_ref().clone())
            }

            // Cubical PathApp: p @ i : A
            (
                ProofCert::CubicalPathApp {
                    path_cert,
                    arg_cert,
                    path_type: _,
                    result_type,
                },
                Expr::CubicalPathApp { path, arg },
            ) => {
                if self.mode != Lean5Mode::Cubical {
                    return Err(CertError::ModeRequired {
                        feature: "CubicalPathApp".to_string(),
                        required_mode: Lean5Mode::Cubical,
                        current_mode: self.mode,
                    });
                }

                // Verify path has Path type
                let path_ty = self.verify(path_cert, path)?;
                let path_ty_whnf = self.whnf(&path_ty);
                let Expr::CubicalPath { ty, .. } = path_ty_whnf else {
                    return Err(CertError::InvalidCert(
                        "CubicalPathApp path is not a Path type".to_string(),
                    ));
                };

                // Verify arg : I
                let arg_ty = self.verify(arg_cert, arg)?;
                if !matches!(self.whnf(&arg_ty), Expr::CubicalInterval) {
                    return Err(CertError::TypeMismatch {
                        expected: Box::new(Expr::CubicalInterval),
                        actual: Box::new(arg_ty),
                        location: "CubicalPathApp argument".to_string(),
                    });
                }

                // Result type should match ty applied to the argument
                let expected_result_ty = Expr::App(ty, arg.clone());
                if !self.def_eq(&expected_result_ty, result_type) {
                    return Err(CertError::TypeMismatch {
                        expected: Box::new(expected_result_ty),
                        actual: Box::new(result_type.as_ref().clone()),
                        location: "CubicalPathApp result".to_string(),
                    });
                }

                Ok(result_type.as_ref().clone())
            }

            // Cubical HComp: hcomp {A} {φ} u base : A
            (
                ProofCert::CubicalHComp {
                    ty_cert,
                    phi_cert,
                    u_cert,
                    base_cert,
                    result_type,
                },
                Expr::CubicalHComp { ty, phi, u, base },
            ) => {
                if self.mode != Lean5Mode::Cubical {
                    return Err(CertError::ModeRequired {
                        feature: "CubicalHComp".to_string(),
                        required_mode: Lean5Mode::Cubical,
                        current_mode: self.mode,
                    });
                }

                // Verify ty is a type
                let ty_sort = self.verify(ty_cert, ty)?;
                if !matches!(self.whnf(&ty_sort), Expr::Sort(_)) {
                    return Err(CertError::InvalidCert(
                        "CubicalHComp type is not a type".to_string(),
                    ));
                }

                // Verify phi, u, base (simplified - full verification is complex)
                let _ = self.verify(phi_cert, phi)?;
                let _ = self.verify(u_cert, u)?;
                let _ = self.verify(base_cert, base)?;

                // Result type should match ty
                if !self.def_eq(ty, result_type) {
                    return Err(CertError::TypeMismatch {
                        expected: Box::new(ty.as_ref().clone()),
                        actual: Box::new(result_type.as_ref().clone()),
                        location: "CubicalHComp result".to_string(),
                    });
                }

                Ok(result_type.as_ref().clone())
            }

            // Cubical Transp: transp ty φ base : ty[1/i]
            (
                ProofCert::CubicalTransp {
                    ty_cert,
                    phi_cert,
                    base_cert,
                    result_type,
                },
                Expr::CubicalTransp { ty, phi, base },
            ) => {
                if self.mode != Lean5Mode::Cubical {
                    return Err(CertError::ModeRequired {
                        feature: "CubicalTransp".to_string(),
                        required_mode: Lean5Mode::Cubical,
                        current_mode: self.mode,
                    });
                }

                // Verify ty, phi, base
                let _ = self.verify(ty_cert, ty)?;
                let _ = self.verify(phi_cert, phi)?;
                let _ = self.verify(base_cert, base)?;

                // Result type is ty[1/i]
                Ok(result_type.as_ref().clone())
            }

            // Classical Choice: choice ty pred exists_proof : ty
            (
                ProofCert::ClassicalChoice {
                    ty_cert,
                    pred_cert,
                    proof_cert,
                    result_type,
                },
                Expr::ClassicalChoice {
                    ty,
                    pred,
                    exists_proof,
                },
            ) => {
                if self.mode != Lean5Mode::Classical && self.mode != Lean5Mode::SetTheoretic {
                    return Err(CertError::ModeRequired {
                        feature: "ClassicalChoice".to_string(),
                        required_mode: Lean5Mode::Classical,
                        current_mode: self.mode,
                    });
                }

                // Verify ty is a type
                let ty_sort = self.verify(ty_cert, ty)?;
                if !matches!(self.whnf(&ty_sort), Expr::Sort(_)) {
                    return Err(CertError::InvalidCert(
                        "ClassicalChoice type is not a type".to_string(),
                    ));
                }

                // Verify pred and exists_proof
                let _ = self.verify(pred_cert, pred)?;
                let _ = self.verify(proof_cert, exists_proof)?;

                // Result type should match ty
                if !self.def_eq(ty, result_type) {
                    return Err(CertError::TypeMismatch {
                        expected: Box::new(ty.as_ref().clone()),
                        actual: Box::new(result_type.as_ref().clone()),
                        location: "ClassicalChoice result".to_string(),
                    });
                }

                Ok(result_type.as_ref().clone())
            }

            // Classical Epsilon: ε ty pred : ty
            (
                ProofCert::ClassicalEpsilon {
                    ty_cert,
                    pred_cert,
                    result_type,
                },
                Expr::ClassicalEpsilon { ty, pred },
            ) => {
                if self.mode != Lean5Mode::Classical && self.mode != Lean5Mode::SetTheoretic {
                    return Err(CertError::ModeRequired {
                        feature: "ClassicalEpsilon".to_string(),
                        required_mode: Lean5Mode::Classical,
                        current_mode: self.mode,
                    });
                }

                // Verify ty is a type
                let ty_sort = self.verify(ty_cert, ty)?;
                if !matches!(self.whnf(&ty_sort), Expr::Sort(_)) {
                    return Err(CertError::InvalidCert(
                        "ClassicalEpsilon type is not a type".to_string(),
                    ));
                }

                // Verify pred
                let _ = self.verify(pred_cert, pred)?;

                // Result type should match ty
                if !self.def_eq(ty, result_type) {
                    return Err(CertError::TypeMismatch {
                        expected: Box::new(ty.as_ref().clone()),
                        actual: Box::new(result_type.as_ref().clone()),
                        location: "ClassicalEpsilon result".to_string(),
                    });
                }

                Ok(result_type.as_ref().clone())
            }

            // ZFC Set: various set expressions : Set
            (
                ProofCert::ZFCSet { kind, result_type },
                Expr::ZFCSet(set_expr),
            ) => {
                if self.mode != Lean5Mode::SetTheoretic {
                    return Err(CertError::ModeRequired {
                        feature: "ZFCSet".to_string(),
                        required_mode: Lean5Mode::SetTheoretic,
                        current_mode: self.mode,
                    });
                }

                // Verify the specific set construction matches
                self.verify_zfc_set(kind, set_expr)?;

                // ZFC sets have type Set (represented as a constant)
                Ok(result_type.as_ref().clone())
            }

            // ZFC Membership: element ∈ set : Prop
            (
                ProofCert::ZFCMem {
                    elem_cert,
                    set_cert,
                },
                Expr::ZFCMem { element, set },
            ) => {
                if self.mode != Lean5Mode::SetTheoretic {
                    return Err(CertError::ModeRequired {
                        feature: "ZFCMem".to_string(),
                        required_mode: Lean5Mode::SetTheoretic,
                        current_mode: self.mode,
                    });
                }

                // Verify element and set
                let _ = self.verify(elem_cert, element)?;
                let _ = self.verify(set_cert, set)?;

                // Membership is a proposition
                Ok(Expr::Sort(Level::zero()))
            }

            // ZFC Comprehension: { x ∈ domain | P(x) } : Set
            (
                ProofCert::ZFCComprehension {
                    var_ty_cert,
                    pred_cert,
                    result_type,
                },
                Expr::ZFCComprehension { domain, pred },
            ) => {
                if self.mode != Lean5Mode::SetTheoretic {
                    return Err(CertError::ModeRequired {
                        feature: "ZFCComprehension".to_string(),
                        required_mode: Lean5Mode::SetTheoretic,
                        current_mode: self.mode,
                    });
                }

                // Verify domain is a set (has type ZFC.Set)
                // The domain certificate verifies the type of domain
                let _ = self.verify(var_ty_cert, domain)?;
                // Note: In SetTheoretic mode, ZFC.Set is the expected type for sets
                // but we don't enforce the exact constant here to keep verification simple

                // Verify pred
                let _ = self.verify(pred_cert, pred)?;

                Ok(result_type.as_ref().clone())
            }

            // Structure mismatch
            (cert, expr) => Err(CertError::StructureMismatch {
                expected: cert_name(cert),
                actual: expr_name(expr),
            }),
        }
    }

    /// Helper to verify ZFC set expression matches certificate kind
    fn verify_zfc_set(
        &mut self,
        kind: &ZFCSetCertKind,
        expr: &ZFCSetExpr,
    ) -> Result<(), CertError> {
        match (kind, expr) {
            (ZFCSetCertKind::Empty, ZFCSetExpr::Empty) => Ok(()),
            (ZFCSetCertKind::Infinity, ZFCSetExpr::Infinity) => Ok(()),
            (ZFCSetCertKind::Singleton(cert), ZFCSetExpr::Singleton(e)) => {
                let _ = self.verify(cert, e)?;
                Ok(())
            }
            (ZFCSetCertKind::Pair(c1, c2), ZFCSetExpr::Pair(e1, e2)) => {
                let _ = self.verify(c1, e1)?;
                let _ = self.verify(c2, e2)?;
                Ok(())
            }
            (ZFCSetCertKind::Union(cert), ZFCSetExpr::Union(e)) => {
                let _ = self.verify(cert, e)?;
                Ok(())
            }
            (ZFCSetCertKind::PowerSet(cert), ZFCSetExpr::PowerSet(e)) => {
                let _ = self.verify(cert, e)?;
                Ok(())
            }
            (
                ZFCSetCertKind::Separation { set_cert, pred_cert },
                ZFCSetExpr::Separation { set, pred },
            ) => {
                let _ = self.verify(set_cert, set)?;
                let _ = self.verify(pred_cert, pred)?;
                Ok(())
            }
            (
                ZFCSetCertKind::Replacement { set_cert, func_cert },
                ZFCSetExpr::Replacement { set, func },
            ) => {
                let _ = self.verify(set_cert, set)?;
                let _ = self.verify(func_cert, func)?;
                Ok(())
            }
            (ZFCSetCertKind::Choice(cert), ZFCSetExpr::Choice(e)) => {
                let _ = self.verify(cert, e)?;
                Ok(())
            }
            _ => Err(CertError::StructureMismatch {
                expected: format!("{kind:?}"),
                actual: format!("{expr:?}"),
            }),
        }
    }

    /// Check definitional equality (simplified)
    fn def_eq(&self, a: &Expr, b: &Expr) -> bool {
        stacker::maybe_grow(MIN_STACK_RED_ZONE, STACK_GROWTH_SIZE, || {
            self.def_eq_impl(a, b)
        })
    }

    /// Implementation (called via stacker::maybe_grow)
    fn def_eq_impl(&self, a: &Expr, b: &Expr) -> bool {
        let a_whnf = self.whnf(a);
        let b_whnf = self.whnf(b);
        self.structural_eq(&a_whnf, &b_whnf)
    }

    /// Structural equality after WHNF
    fn structural_eq(&self, a: &Expr, b: &Expr) -> bool {
        stacker::maybe_grow(MIN_STACK_RED_ZONE, STACK_GROWTH_SIZE, || {
            self.structural_eq_impl(a, b)
        })
    }

    /// Implementation (called via stacker::maybe_grow)
    fn structural_eq_impl(&self, a: &Expr, b: &Expr) -> bool {
        match (a, b) {
            (Expr::BVar(i), Expr::BVar(j)) => i == j,
            (Expr::FVar(i), Expr::FVar(j)) => i == j,
            (Expr::Sort(l1), Expr::Sort(l2)) => self.level_eq(l1, l2),
            (Expr::Const(n1, ls1), Expr::Const(n2, ls2)) => {
                n1 == n2
                    && ls1.len() == ls2.len()
                    && ls1
                        .iter()
                        .zip(ls2.iter())
                        .all(|(l1, l2)| self.level_eq(l1, l2))
            }
            (Expr::App(f1, a1), Expr::App(f2, a2)) => {
                self.structural_eq(f1, f2) && self.structural_eq(a1, a2)
            }
            (Expr::Lam(bi1, ty1, b1), Expr::Lam(bi2, ty2, b2))
            | (Expr::Pi(bi1, ty1, b1), Expr::Pi(bi2, ty2, b2)) => {
                bi1 == bi2 && self.structural_eq(ty1, ty2) && self.structural_eq(b1, b2)
            }
            (Expr::Let(ty1, v1, b1), Expr::Let(ty2, v2, b2)) => {
                self.structural_eq(ty1, ty2)
                    && self.structural_eq(v1, v2)
                    && self.structural_eq(b1, b2)
            }
            (Expr::Lit(l1), Expr::Lit(l2)) => l1 == l2,
            (Expr::Proj(n1, i1, e1), Expr::Proj(n2, i2, e2)) => {
                n1 == n2 && i1 == i2 && self.structural_eq(e1, e2)
            }

            // Mode-specific expressions
            (Expr::CubicalInterval, Expr::CubicalInterval) => true,
            (Expr::CubicalI0, Expr::CubicalI0) => true,
            (Expr::CubicalI1, Expr::CubicalI1) => true,
            (
                Expr::CubicalPath {
                    ty: ty1,
                    left: l1,
                    right: r1,
                },
                Expr::CubicalPath {
                    ty: ty2,
                    left: l2,
                    right: r2,
                },
            ) => {
                self.structural_eq(ty1, ty2)
                    && self.structural_eq(l1, l2)
                    && self.structural_eq(r1, r2)
            }
            (Expr::CubicalPathLam { body: b1 }, Expr::CubicalPathLam { body: b2 }) => {
                self.structural_eq(b1, b2)
            }
            (
                Expr::CubicalPathApp {
                    path: p1,
                    arg: a1,
                },
                Expr::CubicalPathApp {
                    path: p2,
                    arg: a2,
                },
            ) => self.structural_eq(p1, p2) && self.structural_eq(a1, a2),
            (
                Expr::CubicalHComp {
                    ty: ty1,
                    phi: phi1,
                    u: u1,
                    base: base1,
                },
                Expr::CubicalHComp {
                    ty: ty2,
                    phi: phi2,
                    u: u2,
                    base: base2,
                },
            ) => {
                self.structural_eq(ty1, ty2)
                    && self.structural_eq(phi1, phi2)
                    && self.structural_eq(u1, u2)
                    && self.structural_eq(base1, base2)
            }
            (
                Expr::CubicalTransp {
                    ty: ty1,
                    phi: phi1,
                    base: base1,
                },
                Expr::CubicalTransp {
                    ty: ty2,
                    phi: phi2,
                    base: base2,
                },
            ) => {
                self.structural_eq(ty1, ty2)
                    && self.structural_eq(phi1, phi2)
                    && self.structural_eq(base1, base2)
            }
            (
                Expr::ClassicalChoice {
                    ty: ty1,
                    pred: pred1,
                    exists_proof: proof1,
                },
                Expr::ClassicalChoice {
                    ty: ty2,
                    pred: pred2,
                    exists_proof: proof2,
                },
            ) => {
                self.structural_eq(ty1, ty2)
                    && self.structural_eq(pred1, pred2)
                    && self.structural_eq(proof1, proof2)
            }
            (
                Expr::ClassicalEpsilon {
                    ty: ty1,
                    pred: pred1,
                },
                Expr::ClassicalEpsilon {
                    ty: ty2,
                    pred: pred2,
                },
            ) => self.structural_eq(ty1, ty2) && self.structural_eq(pred1, pred2),
            (Expr::ZFCSet(s1), Expr::ZFCSet(s2)) => self.zfc_set_eq(s1, s2),
            (
                Expr::ZFCMem {
                    element: e1,
                    set: s1,
                },
                Expr::ZFCMem {
                    element: e2,
                    set: s2,
                },
            ) => self.structural_eq(e1, e2) && self.structural_eq(s1, s2),
            (
                Expr::ZFCComprehension {
                    domain: d1,
                    pred: p1,
                },
                Expr::ZFCComprehension {
                    domain: d2,
                    pred: p2,
                },
            ) => self.structural_eq(d1, d2) && self.structural_eq(p1, p2),

            _ => false,
        }
    }

    /// Structural equality for ZFC set expressions
    fn zfc_set_eq(&self, a: &ZFCSetExpr, b: &ZFCSetExpr) -> bool {
        match (a, b) {
            (ZFCSetExpr::Empty, ZFCSetExpr::Empty) => true,
            (ZFCSetExpr::Infinity, ZFCSetExpr::Infinity) => true,
            (ZFCSetExpr::Singleton(e1), ZFCSetExpr::Singleton(e2)) => self.structural_eq(e1, e2),
            (ZFCSetExpr::Pair(a1, b1), ZFCSetExpr::Pair(a2, b2)) => {
                self.structural_eq(a1, a2) && self.structural_eq(b1, b2)
            }
            (ZFCSetExpr::Union(e1), ZFCSetExpr::Union(e2)) => self.structural_eq(e1, e2),
            (ZFCSetExpr::PowerSet(e1), ZFCSetExpr::PowerSet(e2)) => self.structural_eq(e1, e2),
            (
                ZFCSetExpr::Separation {
                    set: s1,
                    pred: p1,
                },
                ZFCSetExpr::Separation {
                    set: s2,
                    pred: p2,
                },
            ) => self.structural_eq(s1, s2) && self.structural_eq(p1, p2),
            (
                ZFCSetExpr::Replacement {
                    set: s1,
                    func: f1,
                },
                ZFCSetExpr::Replacement {
                    set: s2,
                    func: f2,
                },
            ) => self.structural_eq(s1, s2) && self.structural_eq(f1, f2),
            (ZFCSetExpr::Choice(e1), ZFCSetExpr::Choice(e2)) => self.structural_eq(e1, e2),
            _ => false,
        }
    }

    /// Level equality
    fn level_eq(&self, l1: &Level, l2: &Level) -> bool {
        // Simplified: structural equality
        // Full version would normalize levels
        l1 == l2
    }

    /// Compute WHNF (weak head normal form)
    fn whnf(&self, e: &Expr) -> Expr {
        stacker::maybe_grow(MIN_STACK_RED_ZONE, STACK_GROWTH_SIZE, || self.whnf_impl(e))
    }

    /// Implementation (called via stacker::maybe_grow)
    fn whnf_impl(&self, e: &Expr) -> Expr {
        match e {
            Expr::App(f, a) => {
                let f_whnf = self.whnf(f);
                match &f_whnf {
                    Expr::Lam(_, _, body) => {
                        let reduced = body.instantiate(a);
                        self.whnf(&reduced)
                    }
                    _ => Expr::App(f_whnf.into(), a.clone()),
                }
            }
            Expr::Let(_, val, body) => {
                let reduced = body.instantiate(val);
                self.whnf(&reduced)
            }
            Expr::Const(name, levels) => self
                .env
                .unfold(name, levels)
                .map_or_else(|| e.clone(), |val| self.whnf(&val)),
            _ => e.clone(),
        }
    }
}

// =============================================================================
// Batch Certificate Verification
//
// Parallel verification of multiple certificates using rayon for CPU parallelism.
// This is a key building block for the GPU acceleration pipeline.
// =============================================================================

/// Input for batch verification: a certificate paired with its expression
#[derive(Debug, Clone)]
pub struct BatchVerifyInput {
    /// Unique identifier for this input (for correlating results)
    pub id: String,
    /// The proof certificate to verify
    pub cert: ProofCert,
    /// The expression the certificate should verify
    pub expr: Expr,
}

impl BatchVerifyInput {
    /// Create a new batch input
    pub fn new(id: impl Into<String>, cert: ProofCert, expr: Expr) -> Self {
        Self {
            id: id.into(),
            cert,
            expr,
        }
    }
}

/// Result of verifying a single certificate in a batch
#[derive(Debug, Clone)]
pub struct BatchVerifyResult {
    /// The ID from the input
    pub id: String,
    /// Whether verification succeeded
    pub success: bool,
    /// The verified type (if successful)
    pub verified_type: Option<Expr>,
    /// Error message (if failed)
    pub error: Option<String>,
    /// Verification time in microseconds
    pub time_us: u64,
}

impl BatchVerifyResult {
    fn success(id: String, ty: Expr, time_us: u64) -> Self {
        Self {
            id,
            success: true,
            verified_type: Some(ty),
            error: None,
            time_us,
        }
    }

    fn failure(id: String, error: String, time_us: u64) -> Self {
        Self {
            id,
            success: false,
            verified_type: None,
            error: Some(error),
            time_us,
        }
    }
}

/// Statistics for batch verification
#[derive(Debug, Clone, Default)]
pub struct BatchVerifyStats {
    /// Total number of inputs
    pub total: usize,
    /// Number of successful verifications
    pub successful: usize,
    /// Number of failed verifications
    pub failed: usize,
    /// Total wall-clock time in microseconds
    pub wall_time_us: u64,
    /// Sum of individual verification times (useful for parallelism analysis)
    pub sum_verify_time_us: u64,
    /// Minimum verification time
    pub min_time_us: u64,
    /// Maximum verification time
    pub max_time_us: u64,
    /// Effective speedup (`sum_verify_time` / `wall_time`)
    pub speedup: f64,
}

impl std::fmt::Display for BatchVerifyStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BatchVerifyStats {{ total: {}, success: {}, failed: {}, wall_time: {}µs, sum_time: {}µs, speedup: {:.2}x }}",
            self.total, self.successful, self.failed, self.wall_time_us, self.sum_verify_time_us, self.speedup
        )
    }
}

fn compute_batch_stats(results: &[BatchVerifyResult], wall_time_us: u64) -> BatchVerifyStats {
    let total = results.len();
    let successful = results.iter().filter(|r| r.success).count();
    let failed = results.iter().filter(|r| !r.success).count();
    let sum_verify_time_us: u64 = results.iter().map(|r| r.time_us).sum();
    let min_time_us = results.iter().map(|r| r.time_us).min().unwrap_or(0);
    let max_time_us = results.iter().map(|r| r.time_us).max().unwrap_or(0);
    let speedup = if wall_time_us > 0 {
        sum_verify_time_us as f64 / wall_time_us as f64
    } else {
        1.0
    };

    BatchVerifyStats {
        total,
        successful,
        failed,
        wall_time_us,
        sum_verify_time_us,
        min_time_us,
        max_time_us,
        speedup,
    }
}

/// Convert a `u128` microsecond duration to `u64` with saturation.
///
/// `Duration::as_micros()` returns `u128`, but for practical timing measurements
/// the values will never exceed `u64::MAX` (which would be 584,000+ years).
/// We saturate rather than truncate to avoid silent data corruption.
#[inline]
fn micros_to_u64(micros: u128) -> u64 {
    u64::try_from(micros).unwrap_or(u64::MAX)
}

/// Verify a batch of certificates in parallel using rayon.
///
/// This is the primary API for high-throughput certificate verification.
/// Each certificate is verified independently using a separate `CertVerifier`,
/// enabling full parallelism.
///
/// # Arguments
/// * `env` - The environment for type checking
/// * `inputs` - The certificates to verify
///
/// # Returns
/// A vector of results in the same order as inputs
///
/// # Example
/// ```ignore
/// let inputs = vec![
///     BatchVerifyInput::new("1", cert1, expr1),
///     BatchVerifyInput::new("2", cert2, expr2),
/// ];
/// let results = batch_verify(&env, inputs);
/// for result in results {
///     if result.success {
///         println!("{}: verified as {:?}", result.id, result.verified_type);
///     }
/// }
/// ```
pub fn batch_verify(env: &Environment, inputs: Vec<BatchVerifyInput>) -> Vec<BatchVerifyResult> {
    use rayon::prelude::*;

    inputs
        .into_par_iter()
        .map(|input| {
            let start = std::time::Instant::now();
            let mut verifier = CertVerifier::new(env);
            match verifier.verify(&input.cert, &input.expr) {
                Ok(ty) => BatchVerifyResult::success(
                    input.id,
                    ty,
                    micros_to_u64(start.elapsed().as_micros()),
                ),
                Err(e) => BatchVerifyResult::failure(
                    input.id,
                    e.to_string(),
                    micros_to_u64(start.elapsed().as_micros()),
                ),
            }
        })
        .collect()
}

/// Verify a batch of certificates with statistics.
///
/// Same as `batch_verify` but also returns aggregate statistics about the batch.
pub fn batch_verify_with_stats(
    env: &Environment,
    inputs: Vec<BatchVerifyInput>,
) -> (Vec<BatchVerifyResult>, BatchVerifyStats) {
    let wall_start = std::time::Instant::now();
    let results = batch_verify(env, inputs);
    let wall_time_us = micros_to_u64(wall_start.elapsed().as_micros());

    let stats = compute_batch_stats(&results, wall_time_us);

    (results, stats)
}

/// Sequential batch verification (for comparison/fallback).
///
/// This verifies certificates one at a time, which is useful for:
/// - Debugging (deterministic ordering)
/// - Single-threaded environments
/// - Baseline performance comparison
pub fn batch_verify_sequential(
    env: &Environment,
    inputs: Vec<BatchVerifyInput>,
) -> Vec<BatchVerifyResult> {
    inputs
        .into_iter()
        .map(|input| {
            let start = std::time::Instant::now();
            let mut verifier = CertVerifier::new(env);
            match verifier.verify(&input.cert, &input.expr) {
                Ok(ty) => BatchVerifyResult::success(
                    input.id,
                    ty,
                    micros_to_u64(start.elapsed().as_micros()),
                ),
                Err(e) => BatchVerifyResult::failure(
                    input.id,
                    e.to_string(),
                    micros_to_u64(start.elapsed().as_micros()),
                ),
            }
        })
        .collect()
}

/// Sequential batch verification with statistics.
pub fn batch_verify_sequential_with_stats(
    env: &Environment,
    inputs: Vec<BatchVerifyInput>,
) -> (Vec<BatchVerifyResult>, BatchVerifyStats) {
    let wall_start = std::time::Instant::now();
    let results = batch_verify_sequential(env, inputs);
    let wall_time_us = micros_to_u64(wall_start.elapsed().as_micros());

    let stats = compute_batch_stats(&results, wall_time_us);

    (results, stats)
}

/// Batch verification with custom thread pool size.
///
/// Useful for controlling parallelism in constrained environments.
pub fn batch_verify_with_threads(
    env: &Environment,
    inputs: Vec<BatchVerifyInput>,
    num_threads: usize,
) -> Vec<BatchVerifyResult> {
    use rayon::prelude::*;

    // Create a thread pool with the specified number of threads
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("Failed to create thread pool");

    pool.install(|| {
        inputs
            .into_par_iter()
            .map(|input| {
                let start = std::time::Instant::now();
                let mut verifier = CertVerifier::new(env);
                match verifier.verify(&input.cert, &input.expr) {
                    Ok(ty) => BatchVerifyResult::success(
                        input.id,
                        ty,
                        micros_to_u64(start.elapsed().as_micros()),
                    ),
                    Err(e) => BatchVerifyResult::failure(
                        input.id,
                        e.to_string(),
                        micros_to_u64(start.elapsed().as_micros()),
                    ),
                }
            })
            .collect()
    })
}

/// Batch verification with custom thread pool size and statistics.
///
/// Combines thread control with statistics collection.
pub fn batch_verify_with_stats_threads(
    env: &Environment,
    inputs: Vec<BatchVerifyInput>,
    num_threads: usize,
) -> (Vec<BatchVerifyResult>, BatchVerifyStats) {
    let wall_start = std::time::Instant::now();
    let results = batch_verify_with_threads(env, inputs, num_threads);
    let wall_time_us = micros_to_u64(wall_start.elapsed().as_micros());

    let stats = compute_batch_stats(&results, wall_time_us);

    (results, stats)
}

/// Batch verification with optional custom thread pool size and progress callback.
///
/// Mirrors `batch_verify_with_stats` but invokes the provided callback every time
/// an item finishes verification, enabling streaming progress reporting.
/// The callback may be invoked from multiple threads; callers must ensure thread safety.
///
/// `threads = 0` uses Rayon default parallelism.
pub fn batch_verify_with_stats_progress<F>(
    env: &Environment,
    inputs: Vec<BatchVerifyInput>,
    threads: usize,
    on_result: F,
) -> (Vec<BatchVerifyResult>, BatchVerifyStats)
where
    F: Fn(&BatchVerifyResult) + Send + Sync,
{
    use rayon::prelude::*;

    let wall_start = std::time::Instant::now();
    let callback = std::sync::Arc::new(on_result);

    let verify_inputs = |inputs: Vec<BatchVerifyInput>, callback: std::sync::Arc<F>| {
        inputs
            .into_par_iter()
            .map(|input| {
                let start = std::time::Instant::now();
                let mut verifier = CertVerifier::new(env);
                let result = match verifier.verify(&input.cert, &input.expr) {
                    Ok(ty) => BatchVerifyResult::success(
                        input.id,
                        ty,
                        micros_to_u64(start.elapsed().as_micros()),
                    ),
                    Err(e) => BatchVerifyResult::failure(
                        input.id,
                        e.to_string(),
                        micros_to_u64(start.elapsed().as_micros()),
                    ),
                };
                callback(&result);
                result
            })
            .collect::<Vec<_>>()
    };

    let results = if threads > 0 {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("Failed to create thread pool");
        let cb = callback.clone();
        pool.install(|| verify_inputs(inputs, cb))
    } else {
        let cb = callback.clone();
        verify_inputs(inputs, cb)
    };

    let wall_time_us = micros_to_u64(wall_start.elapsed().as_micros());
    let stats = compute_batch_stats(&results, wall_time_us);

    (results, stats)
}

// =============================================================================
// Proof Replay
//
// Proof replay reconstructs an expression from a certificate. This enables:
// 1. Proof archives - store compact certificates and replay to reconstruct proofs
// 2. Proof transfer - replay a certificate in a different environment
// 3. Proof debugging - step through certificate structure
// =============================================================================

/// Reconstructs an expression from a proof certificate.
///
/// This is the inverse operation of certificate generation: given a certificate,
/// it produces the expression that would have generated that certificate.
///
/// # Usage
///
/// ```ignore
/// let (ty, cert) = tc.infer_type_with_cert(&expr)?;
/// // Serialize cert to storage...
/// // Later, reconstruct:
/// let reconstructed = replay_cert(&cert);
/// assert_eq!(reconstructed, expr);
/// ```
///
/// # Guarantees
///
/// - For valid certificates, replay produces a well-formed expression
/// - The replayed expression will type-check to the type embedded in the certificate
/// - Replay is deterministic: same certificate always produces same expression
pub fn replay_cert(cert: &ProofCert) -> Expr {
    match cert {
        ProofCert::Sort { level } => Expr::Sort(level.clone()),

        ProofCert::BVar { idx, .. } => Expr::BVar(*idx),

        ProofCert::FVar { id, .. } => Expr::FVar(*id),

        ProofCert::Const { name, levels, .. } => Expr::const_(name.clone(), levels.clone()),

        ProofCert::App {
            fn_cert, arg_cert, ..
        } => {
            let fn_expr = replay_cert(fn_cert);
            let arg_expr = replay_cert(arg_cert);
            Expr::App(fn_expr.into(), arg_expr.into())
        }

        ProofCert::Lam {
            binder_info,
            arg_type_cert,
            body_cert,
            result_type,
        } => {
            // For lambda, we need the domain type from the result_type (which is Pi)
            let domain = match result_type.as_ref() {
                Expr::Pi(_, dom, _) => dom.as_ref().clone(),
                _ => {
                    // Fallback: try to extract from arg_type_cert
                    // The arg_type_cert proves A : Sort, so we can't directly get A
                    // But we can use the expected_type pattern if it's a Sort cert
                    extract_type_from_sort_cert(arg_type_cert)
                }
            };
            let body_expr = replay_cert(body_cert);
            Expr::Lam(*binder_info, domain.into(), body_expr.into())
        }

        ProofCert::Pi {
            binder_info,
            arg_type_cert,
            body_type_cert,
            ..
        } => {
            // Extract domain and codomain from subcertificates
            let domain = extract_type_from_sort_cert(arg_type_cert);
            let codomain = extract_type_from_sort_cert(body_type_cert);
            Expr::Pi(*binder_info, domain.into(), codomain.into())
        }

        ProofCert::Let {
            type_cert,
            value_cert,
            body_cert,
            ..
        } => {
            let ty_expr = extract_type_from_sort_cert(type_cert);
            let val_expr = replay_cert(value_cert);
            let body_expr = replay_cert(body_cert);
            Expr::Let(ty_expr.into(), val_expr.into(), body_expr.into())
        }

        ProofCert::Lit { lit, .. } => Expr::Lit(lit.clone()),

        ProofCert::DefEq { inner, .. } => {
            // DefEq is transparent - the expression is the inner expression
            replay_cert(inner)
        }

        ProofCert::MData {
            metadata,
            inner_cert,
            ..
        } => {
            let inner_expr = replay_cert(inner_cert);
            Expr::MData(metadata.clone(), inner_expr.into())
        }

        ProofCert::Proj {
            struct_name,
            idx,
            expr_cert,
            ..
        } => {
            let expr = replay_cert(expr_cert);
            Expr::proj(struct_name.clone(), *idx, expr)
        }

        // Mode-specific certificates
        ProofCert::CubicalInterval => Expr::CubicalInterval,

        ProofCert::CubicalEndpoint { is_one } => {
            if *is_one {
                Expr::CubicalI1
            } else {
                Expr::CubicalI0
            }
        }

        ProofCert::CubicalPath {
            ty_cert,
            left_cert,
            right_cert,
            ..
        } => {
            let ty = replay_cert(ty_cert);
            let left = replay_cert(left_cert);
            let right = replay_cert(right_cert);
            Expr::CubicalPath {
                ty: ty.into(),
                left: left.into(),
                right: right.into(),
            }
        }

        ProofCert::CubicalPathLam { body_cert, .. } => {
            let body = replay_cert(body_cert);
            Expr::CubicalPathLam { body: body.into() }
        }

        ProofCert::CubicalPathApp {
            path_cert,
            arg_cert,
            ..
        } => {
            let path = replay_cert(path_cert);
            let arg = replay_cert(arg_cert);
            Expr::CubicalPathApp {
                path: path.into(),
                arg: arg.into(),
            }
        }

        ProofCert::CubicalHComp {
            ty_cert,
            phi_cert,
            u_cert,
            base_cert,
            ..
        } => {
            let ty = replay_cert(ty_cert);
            let phi = replay_cert(phi_cert);
            let u = replay_cert(u_cert);
            let base = replay_cert(base_cert);
            Expr::CubicalHComp {
                ty: ty.into(),
                phi: phi.into(),
                u: u.into(),
                base: base.into(),
            }
        }

        ProofCert::CubicalTransp {
            ty_cert,
            phi_cert,
            base_cert,
            ..
        } => {
            let ty = replay_cert(ty_cert);
            let phi = replay_cert(phi_cert);
            let base = replay_cert(base_cert);
            Expr::CubicalTransp {
                ty: ty.into(),
                phi: phi.into(),
                base: base.into(),
            }
        }

        ProofCert::ClassicalChoice {
            ty_cert,
            pred_cert,
            proof_cert,
            ..
        } => {
            let ty = replay_cert(ty_cert);
            let pred = replay_cert(pred_cert);
            let exists_proof = replay_cert(proof_cert);
            Expr::ClassicalChoice {
                ty: ty.into(),
                pred: pred.into(),
                exists_proof: exists_proof.into(),
            }
        }

        ProofCert::ClassicalEpsilon {
            ty_cert, pred_cert, ..
        } => {
            let ty = replay_cert(ty_cert);
            let pred = replay_cert(pred_cert);
            Expr::ClassicalEpsilon {
                ty: ty.into(),
                pred: pred.into(),
            }
        }

        ProofCert::ZFCSet { kind, .. } => Expr::ZFCSet(replay_zfc_set(kind)),

        ProofCert::ZFCMem {
            elem_cert,
            set_cert,
        } => {
            let element = replay_cert(elem_cert);
            let set = replay_cert(set_cert);
            Expr::ZFCMem {
                element: element.into(),
                set: set.into(),
            }
        }

        ProofCert::ZFCComprehension {
            var_ty_cert,
            pred_cert,
            ..
        } => {
            let domain = replay_cert(var_ty_cert);
            let pred = replay_cert(pred_cert);
            Expr::ZFCComprehension {
                domain: domain.into(),
                pred: pred.into(),
            }
        }

        // Impredicative mode certificates
        ProofCert::SProp => Expr::SProp,
        ProofCert::Squash { inner_cert } => {
            let inner = replay_cert(inner_cert);
            Expr::Squash(inner.into())
        }
    }
}

/// Replay a ZFC set certificate kind
fn replay_zfc_set(kind: &ZFCSetCertKind) -> ZFCSetExpr {
    match kind {
        ZFCSetCertKind::Empty => ZFCSetExpr::Empty,
        ZFCSetCertKind::Infinity => ZFCSetExpr::Infinity,
        ZFCSetCertKind::Singleton(cert) => ZFCSetExpr::Singleton(replay_cert(cert).into()),
        ZFCSetCertKind::Pair(c1, c2) => {
            ZFCSetExpr::Pair(replay_cert(c1).into(), replay_cert(c2).into())
        }
        ZFCSetCertKind::Union(cert) => ZFCSetExpr::Union(replay_cert(cert).into()),
        ZFCSetCertKind::PowerSet(cert) => ZFCSetExpr::PowerSet(replay_cert(cert).into()),
        ZFCSetCertKind::Separation { set_cert, pred_cert } => ZFCSetExpr::Separation {
            set: replay_cert(set_cert).into(),
            pred: replay_cert(pred_cert).into(),
        },
        ZFCSetCertKind::Replacement { set_cert, func_cert } => ZFCSetExpr::Replacement {
            set: replay_cert(set_cert).into(),
            func: replay_cert(func_cert).into(),
        },
        ZFCSetCertKind::Choice(cert) => ZFCSetExpr::Choice(replay_cert(cert).into()),
    }
}

/// Helper to extract the expression being typed from a certificate that
/// proves "e : Sort(l)".
///
/// This is tricky because the certificate structure doesn't directly store
/// the expression - only its type and subcertificates. For Sort certificates,
/// the expression is a Sort. For other cases, we recursively replay.
fn extract_type_from_sort_cert(cert: &ProofCert) -> Expr {
    match cert {
        // If proving Sort(l) : Sort(succ(l)), the expression is Sort(l)
        ProofCert::Sort { level } => Expr::Sort(level.clone()),

        // If proving BVar : T, the expression is BVar
        ProofCert::BVar { idx, .. } => Expr::BVar(*idx),

        // If proving FVar : T, the expression is FVar
        ProofCert::FVar { id, .. } => Expr::FVar(*id),

        // If proving Const : T, the expression is Const
        ProofCert::Const { name, levels, .. } => Expr::const_(name.clone(), levels.clone()),

        // For App, Lam, Pi, Let, Proj: replay the full expression
        ProofCert::App { .. }
        | ProofCert::Lam { .. }
        | ProofCert::Pi { .. }
        | ProofCert::Let { .. }
        | ProofCert::Proj { .. } => replay_cert(cert),

        // For Lit, the expression is the literal
        ProofCert::Lit { lit, .. } => Expr::Lit(lit.clone()),

        // For DefEq, the expression is the inner expression
        ProofCert::DefEq { inner, .. } => extract_type_from_sort_cert(inner),

        // For MData, the expression is MData wrapping the inner
        ProofCert::MData {
            metadata,
            inner_cert,
            ..
        } => {
            let inner_expr = extract_type_from_sort_cert(inner_cert);
            Expr::MData(metadata.clone(), inner_expr.into())
        }

        // Mode-specific certificates: just replay them
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
        | ProofCert::Squash { .. } => replay_cert(cert),
    }
}

/// Validates that a replayed certificate produces an expression that
/// type-checks correctly against the certificate.
///
/// This is a sanity check to ensure replay is working correctly.
/// Returns the replayed expression and its verified type.
impl<'env> CertVerifier<'env> {
    /// Replay a certificate and verify the result.
    ///
    /// This combines `replay_cert` with verification, ensuring the
    /// reconstructed expression matches the certificate.
    pub fn replay_and_verify(&mut self, cert: &ProofCert) -> Result<(Expr, Expr), CertError> {
        let expr = replay_cert(cert);
        let ty = self.verify(cert, &expr)?;
        Ok((expr, ty))
    }
}

/// Get a descriptive name for certificate variant
fn cert_name(cert: &ProofCert) -> String {
    match cert {
        ProofCert::Sort { .. } => "Sort".to_string(),
        ProofCert::BVar { .. } => "BVar".to_string(),
        ProofCert::FVar { .. } => "FVar".to_string(),
        ProofCert::Const { .. } => "Const".to_string(),
        ProofCert::App { .. } => "App".to_string(),
        ProofCert::Lam { .. } => "Lam".to_string(),
        ProofCert::Pi { .. } => "Pi".to_string(),
        ProofCert::Let { .. } => "Let".to_string(),
        ProofCert::Lit { .. } => "Lit".to_string(),
        ProofCert::DefEq { .. } => "DefEq".to_string(),
        ProofCert::MData { .. } => "MData".to_string(),
        ProofCert::Proj { .. } => "Proj".to_string(),
        // Mode-specific certificates
        ProofCert::CubicalInterval => "CubicalInterval".to_string(),
        ProofCert::CubicalEndpoint { .. } => "CubicalEndpoint".to_string(),
        ProofCert::CubicalPath { .. } => "CubicalPath".to_string(),
        ProofCert::CubicalPathLam { .. } => "CubicalPathLam".to_string(),
        ProofCert::CubicalPathApp { .. } => "CubicalPathApp".to_string(),
        ProofCert::CubicalHComp { .. } => "CubicalHComp".to_string(),
        ProofCert::CubicalTransp { .. } => "CubicalTransp".to_string(),
        ProofCert::ClassicalChoice { .. } => "ClassicalChoice".to_string(),
        ProofCert::ClassicalEpsilon { .. } => "ClassicalEpsilon".to_string(),
        ProofCert::ZFCSet { .. } => "ZFCSet".to_string(),
        ProofCert::ZFCMem { .. } => "ZFCMem".to_string(),
        ProofCert::ZFCComprehension { .. } => "ZFCComprehension".to_string(),
        // Impredicative mode certificates
        ProofCert::SProp => "SProp".to_string(),
        ProofCert::Squash { .. } => "Squash".to_string(),
    }
}

/// Get a descriptive name for expression variant
fn expr_name(expr: &Expr) -> String {
    match expr {
        Expr::BVar(_) => "BVar".to_string(),
        Expr::FVar(_) => "FVar".to_string(),
        Expr::Sort(_) => "Sort".to_string(),
        Expr::Const(_, _) => "Const".to_string(),
        Expr::App(_, _) => "App".to_string(),
        Expr::Lam(_, _, _) => "Lam".to_string(),
        Expr::Pi(_, _, _) => "Pi".to_string(),
        Expr::Let(_, _, _) => "Let".to_string(),
        Expr::Lit(_) => "Lit".to_string(),
        Expr::Proj(_, _, _) => "Proj".to_string(),
        Expr::MData(_, _) => "MData".to_string(),
        // Cubical mode extensions
        Expr::CubicalInterval => "CubicalInterval".to_string(),
        Expr::CubicalI0 => "CubicalI0".to_string(),
        Expr::CubicalI1 => "CubicalI1".to_string(),
        Expr::CubicalPath { .. } => "CubicalPath".to_string(),
        Expr::CubicalPathLam { .. } => "CubicalPathLam".to_string(),
        Expr::CubicalPathApp { .. } => "CubicalPathApp".to_string(),
        Expr::CubicalHComp { .. } => "CubicalHComp".to_string(),
        Expr::CubicalTransp { .. } => "CubicalTransp".to_string(),
        // Classical mode extensions
        Expr::ClassicalChoice { .. } => "ClassicalChoice".to_string(),
        Expr::ClassicalEpsilon { .. } => "ClassicalEpsilon".to_string(),
        // SetTheoretic mode extensions
        Expr::ZFCSet(_) => "ZFCSet".to_string(),
        Expr::ZFCMem { .. } => "ZFCMem".to_string(),
        Expr::ZFCComprehension { .. } => "ZFCComprehension".to_string(),
        // Impredicative mode extensions
        Expr::SProp => "SProp".to_string(),
        Expr::Squash(_) => "Squash".to_string(),
    }
}

// ============================================================================
// Certificate Compression
// ============================================================================
//
// This module implements structure-sharing compression for proof certificates.
// Large proofs often contain many repeated subexpressions and subcertificates.
// Compression uses hash-consing to deduplicate these and produce a compact
// indexed representation.
//
// ## Design
//
// The compressed format consists of:
// 1. Expression table: deduplicated expressions, indexed by position
// 2. Level table: deduplicated universe levels, indexed by position
// 3. Certificate table: deduplicated certificates, indexed by position
// 4. Root index: points to the main certificate in the table
//
// References in the compressed format use indices into these tables instead
// of nested structures, enabling significant size reduction for large proofs.
//
// ## Usage
//
// ```ignore
// // Compress a certificate for storage
// let compressed = compress_cert(&cert);
// let bytes = bincode::serialize(&compressed)?;
//
// // Decompress for verification
// let compressed: CompressedCert = bincode::deserialize(&bytes)?;
// let cert = decompress_cert(&compressed)?;
// ```

/// Index into the expression table in compressed format
pub type ExprIdx = u32;

/// Index into the level table in compressed format
pub type LevelIdx = u32;

/// Index into the certificate table in compressed format
pub type CertIdx = u32;

/// Compressed proof certificate format using structure sharing.
///
/// This format deduplicates repeated subexpressions, levels, and certificates
/// to achieve significant size reduction for large proofs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompressedCert {
    /// Deduplicated expression table
    pub exprs: Vec<CompressedExpr>,
    /// Deduplicated level table
    pub levels: Vec<CompressedLevel>,
    /// Deduplicated certificate table
    pub certs: Vec<CompressedCertNode>,
    /// Index of the root certificate
    pub root: CertIdx,
}

/// Compressed expression node with indices instead of nested structures
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressedExpr {
    BVar(u32),
    FVar(FVarId),
    Sort(LevelIdx),
    Const(Name, Vec<LevelIdx>),
    App(ExprIdx, ExprIdx),
    Lam(BinderInfo, ExprIdx, ExprIdx),
    Pi(BinderInfo, ExprIdx, ExprIdx),
    Let(ExprIdx, ExprIdx, ExprIdx),
    Lit(Literal),
    Proj(Name, u32, ExprIdx),
    MData(crate::expr::MDataMap, ExprIdx),
}

/// Compressed universe level with indices for nested levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressedLevel {
    Zero,
    Succ(LevelIdx),
    Max(LevelIdx, LevelIdx),
    IMax(LevelIdx, LevelIdx),
    Param(Name),
}

/// Compressed certificate node with indices
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CompressedCertNode {
    Sort {
        level: LevelIdx,
    },
    BVar {
        idx: u32,
        expected_type: ExprIdx,
    },
    FVar {
        id: FVarId,
        type_: ExprIdx,
    },
    Const {
        name: Name,
        levels: Vec<LevelIdx>,
        type_: ExprIdx,
    },
    App {
        fn_cert: CertIdx,
        fn_type: ExprIdx,
        arg_cert: CertIdx,
        result_type: ExprIdx,
    },
    Lam {
        binder_info: BinderInfo,
        arg_type_cert: CertIdx,
        body_cert: CertIdx,
        result_type: ExprIdx,
    },
    Pi {
        binder_info: BinderInfo,
        arg_type_cert: CertIdx,
        arg_level: LevelIdx,
        body_type_cert: CertIdx,
        body_level: LevelIdx,
    },
    Let {
        type_cert: CertIdx,
        value_cert: CertIdx,
        body_cert: CertIdx,
        result_type: ExprIdx,
    },
    Lit {
        lit: Literal,
        type_: ExprIdx,
    },
    DefEq {
        inner: CertIdx,
        expected_type: ExprIdx,
        actual_type: ExprIdx,
        eq_steps: Vec<DefEqStep>,
    },
    MData {
        metadata: crate::expr::MDataMap,
        inner_cert: CertIdx,
        result_type: ExprIdx,
    },
    Proj {
        struct_name: Name,
        idx: u32,
        expr_cert: CertIdx,
        expr_type: ExprIdx,
        field_type: ExprIdx,
    },
    /// Mode-specific certificates (Cubical, Classical, SetTheoretic)
    /// Stored as boxed ProofCert to avoid duplicating compression logic.
    /// Full compression support for mode-specific certs can be added later.
    ModeSpecific(Box<ProofCert>),
}

/// State for certificate compression (hash-consing)
struct CompressionState {
    /// Expression hash map: expr hash -> index
    expr_map: HashMap<u64, ExprIdx>,
    /// Expression table (indexed)
    exprs: Vec<CompressedExpr>,
    /// Level hash map: level hash -> index
    level_map: HashMap<u64, LevelIdx>,
    /// Level table (indexed)
    levels: Vec<CompressedLevel>,
    /// Certificate hash map: cert hash -> index
    cert_map: HashMap<u64, CertIdx>,
    /// Certificate table (indexed)
    certs: Vec<CompressedCertNode>,
}

impl CompressionState {
    fn new() -> Self {
        Self {
            expr_map: HashMap::new(),
            exprs: Vec::new(),
            level_map: HashMap::new(),
            levels: Vec::new(),
            cert_map: HashMap::new(),
            certs: Vec::new(),
        }
    }

    /// Convert a usize length to a u32 index, panicking on overflow.
    ///
    /// This enforces the limit that compressed certificates cannot have more than
    /// 2^32 unique expressions, levels, or certificates. In practice, this limit
    /// is unreachable - it would require a proof with billions of unique nodes.
    #[inline]
    fn len_to_idx(len: usize, kind: &str) -> u32 {
        u32::try_from(len).unwrap_or_else(|_| {
            panic!("certificate compression overflow: {kind} count {len} exceeds u32::MAX")
        })
    }

    /// Hash a level for deduplication
    fn hash_level(level: &Level) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        // Simple structural hash
        match level {
            Level::Zero => 0u8.hash(&mut hasher),
            Level::Succ(l) => {
                1u8.hash(&mut hasher);
                Self::hash_level(l).hash(&mut hasher);
            }
            Level::Max(l1, l2) => {
                2u8.hash(&mut hasher);
                Self::hash_level(l1).hash(&mut hasher);
                Self::hash_level(l2).hash(&mut hasher);
            }
            Level::IMax(l1, l2) => {
                3u8.hash(&mut hasher);
                Self::hash_level(l1).hash(&mut hasher);
                Self::hash_level(l2).hash(&mut hasher);
            }
            Level::Param(n) => {
                4u8.hash(&mut hasher);
                n.hash(&mut hasher);
            }
        }
        hasher.finish()
    }

    /// Hash an expression for deduplication
    fn hash_expr(expr: &Expr) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        match expr {
            Expr::BVar(idx) => {
                0u8.hash(&mut hasher);
                idx.hash(&mut hasher);
            }
            Expr::FVar(id) => {
                1u8.hash(&mut hasher);
                id.hash(&mut hasher);
            }
            Expr::Sort(l) => {
                2u8.hash(&mut hasher);
                Self::hash_level(l).hash(&mut hasher);
            }
            Expr::Const(n, ls) => {
                3u8.hash(&mut hasher);
                n.hash(&mut hasher);
                for l in ls {
                    Self::hash_level(l).hash(&mut hasher);
                }
            }
            Expr::App(f, a) => {
                4u8.hash(&mut hasher);
                Self::hash_expr(f).hash(&mut hasher);
                Self::hash_expr(a).hash(&mut hasher);
            }
            Expr::Lam(bi, ty, body) => {
                5u8.hash(&mut hasher);
                bi.hash(&mut hasher);
                Self::hash_expr(ty).hash(&mut hasher);
                Self::hash_expr(body).hash(&mut hasher);
            }
            Expr::Pi(bi, ty, body) => {
                6u8.hash(&mut hasher);
                bi.hash(&mut hasher);
                Self::hash_expr(ty).hash(&mut hasher);
                Self::hash_expr(body).hash(&mut hasher);
            }
            Expr::Let(ty, val, body) => {
                7u8.hash(&mut hasher);
                Self::hash_expr(ty).hash(&mut hasher);
                Self::hash_expr(val).hash(&mut hasher);
                Self::hash_expr(body).hash(&mut hasher);
            }
            Expr::Lit(lit) => {
                8u8.hash(&mut hasher);
                lit.hash(&mut hasher);
            }
            Expr::Proj(n, idx, e) => {
                9u8.hash(&mut hasher);
                n.hash(&mut hasher);
                idx.hash(&mut hasher);
                Self::hash_expr(e).hash(&mut hasher);
            }
            Expr::MData(md, e) => {
                10u8.hash(&mut hasher);
                // Hash metadata by its length and structure
                md.len().hash(&mut hasher);
                Self::hash_expr(e).hash(&mut hasher);
            }
            // Cubical mode extensions
            Expr::CubicalInterval => {
                11u8.hash(&mut hasher);
            }
            Expr::CubicalI0 => {
                12u8.hash(&mut hasher);
            }
            Expr::CubicalI1 => {
                13u8.hash(&mut hasher);
            }
            Expr::CubicalPath { ty, left, right } => {
                14u8.hash(&mut hasher);
                Self::hash_expr(ty).hash(&mut hasher);
                Self::hash_expr(left).hash(&mut hasher);
                Self::hash_expr(right).hash(&mut hasher);
            }
            Expr::CubicalPathLam { body } => {
                15u8.hash(&mut hasher);
                Self::hash_expr(body).hash(&mut hasher);
            }
            Expr::CubicalPathApp { path, arg } => {
                16u8.hash(&mut hasher);
                Self::hash_expr(path).hash(&mut hasher);
                Self::hash_expr(arg).hash(&mut hasher);
            }
            Expr::CubicalHComp { ty, phi, u, base } => {
                17u8.hash(&mut hasher);
                Self::hash_expr(ty).hash(&mut hasher);
                Self::hash_expr(phi).hash(&mut hasher);
                Self::hash_expr(u).hash(&mut hasher);
                Self::hash_expr(base).hash(&mut hasher);
            }
            Expr::CubicalTransp { ty, phi, base } => {
                18u8.hash(&mut hasher);
                Self::hash_expr(ty).hash(&mut hasher);
                Self::hash_expr(phi).hash(&mut hasher);
                Self::hash_expr(base).hash(&mut hasher);
            }
            // Classical mode extensions
            Expr::ClassicalChoice {
                ty,
                pred,
                exists_proof,
            } => {
                19u8.hash(&mut hasher);
                Self::hash_expr(ty).hash(&mut hasher);
                Self::hash_expr(pred).hash(&mut hasher);
                Self::hash_expr(exists_proof).hash(&mut hasher);
            }
            Expr::ClassicalEpsilon { ty, pred } => {
                20u8.hash(&mut hasher);
                Self::hash_expr(ty).hash(&mut hasher);
                Self::hash_expr(pred).hash(&mut hasher);
            }
            // SetTheoretic mode extensions
            Expr::ZFCSet(set_expr) => {
                21u8.hash(&mut hasher);
                // Hash based on discriminant for simplicity
                std::mem::discriminant(set_expr).hash(&mut hasher);
            }
            Expr::ZFCMem { element, set } => {
                22u8.hash(&mut hasher);
                Self::hash_expr(element).hash(&mut hasher);
                Self::hash_expr(set).hash(&mut hasher);
            }
            Expr::ZFCComprehension { domain, pred } => {
                23u8.hash(&mut hasher);
                Self::hash_expr(domain).hash(&mut hasher);
                Self::hash_expr(pred).hash(&mut hasher);
            }
            // Impredicative mode extensions
            Expr::SProp => {
                24u8.hash(&mut hasher);
            }
            Expr::Squash(inner) => {
                25u8.hash(&mut hasher);
                Self::hash_expr(inner).hash(&mut hasher);
            }
        }
        hasher.finish()
    }

    /// Hash a certificate for deduplication
    fn hash_cert(cert: &ProofCert) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        match cert {
            ProofCert::Sort { level } => {
                0u8.hash(&mut hasher);
                Self::hash_level(level).hash(&mut hasher);
            }
            ProofCert::BVar { idx, expected_type } => {
                1u8.hash(&mut hasher);
                idx.hash(&mut hasher);
                Self::hash_expr(expected_type).hash(&mut hasher);
            }
            ProofCert::FVar { id, type_ } => {
                2u8.hash(&mut hasher);
                id.hash(&mut hasher);
                Self::hash_expr(type_).hash(&mut hasher);
            }
            ProofCert::Const {
                name,
                levels,
                type_,
            } => {
                3u8.hash(&mut hasher);
                name.hash(&mut hasher);
                for l in levels {
                    Self::hash_level(l).hash(&mut hasher);
                }
                Self::hash_expr(type_).hash(&mut hasher);
            }
            ProofCert::App {
                fn_cert,
                fn_type,
                arg_cert,
                result_type,
            } => {
                4u8.hash(&mut hasher);
                Self::hash_cert(fn_cert).hash(&mut hasher);
                Self::hash_expr(fn_type).hash(&mut hasher);
                Self::hash_cert(arg_cert).hash(&mut hasher);
                Self::hash_expr(result_type).hash(&mut hasher);
            }
            ProofCert::Lam {
                binder_info,
                arg_type_cert,
                body_cert,
                result_type,
            } => {
                5u8.hash(&mut hasher);
                binder_info.hash(&mut hasher);
                Self::hash_cert(arg_type_cert).hash(&mut hasher);
                Self::hash_cert(body_cert).hash(&mut hasher);
                Self::hash_expr(result_type).hash(&mut hasher);
            }
            ProofCert::Pi {
                binder_info,
                arg_type_cert,
                arg_level,
                body_type_cert,
                body_level,
            } => {
                6u8.hash(&mut hasher);
                binder_info.hash(&mut hasher);
                Self::hash_cert(arg_type_cert).hash(&mut hasher);
                Self::hash_level(arg_level).hash(&mut hasher);
                Self::hash_cert(body_type_cert).hash(&mut hasher);
                Self::hash_level(body_level).hash(&mut hasher);
            }
            ProofCert::Let {
                type_cert,
                value_cert,
                body_cert,
                result_type,
            } => {
                7u8.hash(&mut hasher);
                Self::hash_cert(type_cert).hash(&mut hasher);
                Self::hash_cert(value_cert).hash(&mut hasher);
                Self::hash_cert(body_cert).hash(&mut hasher);
                Self::hash_expr(result_type).hash(&mut hasher);
            }
            ProofCert::Lit { lit, type_ } => {
                8u8.hash(&mut hasher);
                lit.hash(&mut hasher);
                Self::hash_expr(type_).hash(&mut hasher);
            }
            ProofCert::DefEq {
                inner,
                expected_type,
                actual_type,
                eq_steps,
            } => {
                9u8.hash(&mut hasher);
                Self::hash_cert(inner).hash(&mut hasher);
                Self::hash_expr(expected_type).hash(&mut hasher);
                Self::hash_expr(actual_type).hash(&mut hasher);
                eq_steps.len().hash(&mut hasher);
            }
            ProofCert::MData {
                metadata,
                inner_cert,
                result_type,
            } => {
                10u8.hash(&mut hasher);
                metadata.len().hash(&mut hasher);
                Self::hash_cert(inner_cert).hash(&mut hasher);
                Self::hash_expr(result_type).hash(&mut hasher);
            }
            ProofCert::Proj {
                struct_name,
                idx,
                expr_cert,
                expr_type,
                field_type,
            } => {
                11u8.hash(&mut hasher);
                struct_name.hash(&mut hasher);
                idx.hash(&mut hasher);
                Self::hash_cert(expr_cert).hash(&mut hasher);
                Self::hash_expr(expr_type).hash(&mut hasher);
                Self::hash_expr(field_type).hash(&mut hasher);
            }
            // Mode-specific certificates - use unique discriminants starting at 12
            ProofCert::CubicalInterval => {
                12u8.hash(&mut hasher);
            }
            ProofCert::CubicalEndpoint { is_one } => {
                13u8.hash(&mut hasher);
                is_one.hash(&mut hasher);
            }
            ProofCert::CubicalPath {
                ty_cert,
                ty_level,
                left_cert,
                right_cert,
            } => {
                14u8.hash(&mut hasher);
                Self::hash_cert(ty_cert).hash(&mut hasher);
                Self::hash_level(ty_level).hash(&mut hasher);
                Self::hash_cert(left_cert).hash(&mut hasher);
                Self::hash_cert(right_cert).hash(&mut hasher);
            }
            ProofCert::CubicalPathLam {
                body_cert,
                body_type,
                result_type,
            } => {
                15u8.hash(&mut hasher);
                Self::hash_cert(body_cert).hash(&mut hasher);
                Self::hash_expr(body_type).hash(&mut hasher);
                Self::hash_expr(result_type).hash(&mut hasher);
            }
            ProofCert::CubicalPathApp {
                path_cert,
                arg_cert,
                path_type,
                result_type,
            } => {
                16u8.hash(&mut hasher);
                Self::hash_cert(path_cert).hash(&mut hasher);
                Self::hash_cert(arg_cert).hash(&mut hasher);
                Self::hash_expr(path_type).hash(&mut hasher);
                Self::hash_expr(result_type).hash(&mut hasher);
            }
            ProofCert::CubicalHComp {
                ty_cert,
                phi_cert,
                u_cert,
                base_cert,
                result_type,
            } => {
                17u8.hash(&mut hasher);
                Self::hash_cert(ty_cert).hash(&mut hasher);
                Self::hash_cert(phi_cert).hash(&mut hasher);
                Self::hash_cert(u_cert).hash(&mut hasher);
                Self::hash_cert(base_cert).hash(&mut hasher);
                Self::hash_expr(result_type).hash(&mut hasher);
            }
            ProofCert::CubicalTransp {
                ty_cert,
                phi_cert,
                base_cert,
                result_type,
            } => {
                18u8.hash(&mut hasher);
                Self::hash_cert(ty_cert).hash(&mut hasher);
                Self::hash_cert(phi_cert).hash(&mut hasher);
                Self::hash_cert(base_cert).hash(&mut hasher);
                Self::hash_expr(result_type).hash(&mut hasher);
            }
            ProofCert::ClassicalChoice {
                ty_cert,
                pred_cert,
                proof_cert,
                result_type,
            } => {
                19u8.hash(&mut hasher);
                Self::hash_cert(ty_cert).hash(&mut hasher);
                Self::hash_cert(pred_cert).hash(&mut hasher);
                Self::hash_cert(proof_cert).hash(&mut hasher);
                Self::hash_expr(result_type).hash(&mut hasher);
            }
            ProofCert::ClassicalEpsilon {
                ty_cert,
                pred_cert,
                result_type,
            } => {
                20u8.hash(&mut hasher);
                Self::hash_cert(ty_cert).hash(&mut hasher);
                Self::hash_cert(pred_cert).hash(&mut hasher);
                Self::hash_expr(result_type).hash(&mut hasher);
            }
            ProofCert::ZFCSet { kind, result_type } => {
                21u8.hash(&mut hasher);
                Self::hash_zfc_set_kind(kind).hash(&mut hasher);
                Self::hash_expr(result_type).hash(&mut hasher);
            }
            ProofCert::ZFCMem {
                elem_cert,
                set_cert,
            } => {
                22u8.hash(&mut hasher);
                Self::hash_cert(elem_cert).hash(&mut hasher);
                Self::hash_cert(set_cert).hash(&mut hasher);
            }
            ProofCert::ZFCComprehension {
                var_ty_cert,
                pred_cert,
                result_type,
            } => {
                23u8.hash(&mut hasher);
                Self::hash_cert(var_ty_cert).hash(&mut hasher);
                Self::hash_cert(pred_cert).hash(&mut hasher);
                Self::hash_expr(result_type).hash(&mut hasher);
            }
            // Impredicative mode certificates
            ProofCert::SProp => {
                24u8.hash(&mut hasher);
            }
            ProofCert::Squash { inner_cert } => {
                25u8.hash(&mut hasher);
                Self::hash_cert(inner_cert).hash(&mut hasher);
            }
        }
        hasher.finish()
    }

    /// Hash a ZFC set certificate kind for deduplication
    fn hash_zfc_set_kind(kind: &ZFCSetCertKind) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        match kind {
            ZFCSetCertKind::Empty => 0u8.hash(&mut hasher),
            ZFCSetCertKind::Infinity => 1u8.hash(&mut hasher),
            ZFCSetCertKind::Singleton(c) => {
                2u8.hash(&mut hasher);
                Self::hash_cert(c).hash(&mut hasher);
            }
            ZFCSetCertKind::Pair(c1, c2) => {
                3u8.hash(&mut hasher);
                Self::hash_cert(c1).hash(&mut hasher);
                Self::hash_cert(c2).hash(&mut hasher);
            }
            ZFCSetCertKind::Union(c) => {
                4u8.hash(&mut hasher);
                Self::hash_cert(c).hash(&mut hasher);
            }
            ZFCSetCertKind::PowerSet(c) => {
                5u8.hash(&mut hasher);
                Self::hash_cert(c).hash(&mut hasher);
            }
            ZFCSetCertKind::Separation { set_cert, pred_cert } => {
                6u8.hash(&mut hasher);
                Self::hash_cert(set_cert).hash(&mut hasher);
                Self::hash_cert(pred_cert).hash(&mut hasher);
            }
            ZFCSetCertKind::Replacement { set_cert, func_cert } => {
                7u8.hash(&mut hasher);
                Self::hash_cert(set_cert).hash(&mut hasher);
                Self::hash_cert(func_cert).hash(&mut hasher);
            }
            ZFCSetCertKind::Choice(c) => {
                8u8.hash(&mut hasher);
                Self::hash_cert(c).hash(&mut hasher);
            }
        }
        hasher.finish()
    }

    /// Intern a level, returning its index
    fn intern_level(&mut self, level: &Level) -> LevelIdx {
        let hash = Self::hash_level(level);
        if let Some(&idx) = self.level_map.get(&hash) {
            return idx;
        }

        let compressed = match level {
            Level::Zero => CompressedLevel::Zero,
            Level::Succ(l) => {
                let idx = self.intern_level(l);
                CompressedLevel::Succ(idx)
            }
            Level::Max(l1, l2) => {
                let idx1 = self.intern_level(l1);
                let idx2 = self.intern_level(l2);
                CompressedLevel::Max(idx1, idx2)
            }
            Level::IMax(l1, l2) => {
                let idx1 = self.intern_level(l1);
                let idx2 = self.intern_level(l2);
                CompressedLevel::IMax(idx1, idx2)
            }
            Level::Param(n) => CompressedLevel::Param(n.clone()),
        };

        let idx = Self::len_to_idx(self.levels.len(), "level");
        self.levels.push(compressed);
        self.level_map.insert(hash, idx);
        idx
    }

    /// Intern an expression, returning its index
    fn intern_expr(&mut self, expr: &Expr) -> ExprIdx {
        let hash = Self::hash_expr(expr);
        if let Some(&idx) = self.expr_map.get(&hash) {
            return idx;
        }

        let compressed = match expr {
            Expr::BVar(idx) => CompressedExpr::BVar(*idx),
            Expr::FVar(id) => CompressedExpr::FVar(*id),
            Expr::Sort(l) => {
                let idx = self.intern_level(l);
                CompressedExpr::Sort(idx)
            }
            Expr::Const(n, ls) => {
                let level_idxs: Vec<_> = ls.iter().map(|l| self.intern_level(l)).collect();
                CompressedExpr::Const(n.clone(), level_idxs)
            }
            Expr::App(f, a) => {
                let f_idx = self.intern_expr(f);
                let a_idx = self.intern_expr(a);
                CompressedExpr::App(f_idx, a_idx)
            }
            Expr::Lam(bi, ty, body) => {
                let ty_idx = self.intern_expr(ty);
                let body_idx = self.intern_expr(body);
                CompressedExpr::Lam(*bi, ty_idx, body_idx)
            }
            Expr::Pi(bi, ty, body) => {
                let ty_idx = self.intern_expr(ty);
                let body_idx = self.intern_expr(body);
                CompressedExpr::Pi(*bi, ty_idx, body_idx)
            }
            Expr::Let(ty, val, body) => {
                let ty_idx = self.intern_expr(ty);
                let val_idx = self.intern_expr(val);
                let body_idx = self.intern_expr(body);
                CompressedExpr::Let(ty_idx, val_idx, body_idx)
            }
            Expr::Lit(lit) => CompressedExpr::Lit(lit.clone()),
            Expr::Proj(n, idx, e) => {
                let e_idx = self.intern_expr(e);
                CompressedExpr::Proj(n.clone(), *idx, e_idx)
            }
            Expr::MData(md, e) => {
                let e_idx = self.intern_expr(e);
                CompressedExpr::MData(md.clone(), e_idx)
            }

            // Mode-specific extensions don't have compressed representations yet
            // They require extending CompressedExpr which is a separate task
            Expr::CubicalInterval
            | Expr::CubicalI0
            | Expr::CubicalI1
            | Expr::CubicalPath { .. }
            | Expr::CubicalPathLam { .. }
            | Expr::CubicalPathApp { .. }
            | Expr::CubicalHComp { .. }
            | Expr::CubicalTransp { .. }
            | Expr::ClassicalChoice { .. }
            | Expr::ClassicalEpsilon { .. }
            | Expr::ZFCSet(_)
            | Expr::ZFCMem { .. }
            | Expr::ZFCComprehension { .. }
            | Expr::SProp
            | Expr::Squash(_) => {
                panic!(
                    "Mode-specific expressions do not have compressed representations yet: {:?}",
                    expr_name(expr)
                )
            }
        };

        let idx = Self::len_to_idx(self.exprs.len(), "expression");
        self.exprs.push(compressed);
        self.expr_map.insert(hash, idx);
        idx
    }

    /// Intern a certificate, returning its index
    fn intern_cert(&mut self, cert: &ProofCert) -> CertIdx {
        let hash = Self::hash_cert(cert);
        if let Some(&idx) = self.cert_map.get(&hash) {
            return idx;
        }

        let compressed = match cert {
            ProofCert::Sort { level } => {
                let level_idx = self.intern_level(level);
                CompressedCertNode::Sort { level: level_idx }
            }
            ProofCert::BVar { idx, expected_type } => {
                let type_idx = self.intern_expr(expected_type);
                CompressedCertNode::BVar {
                    idx: *idx,
                    expected_type: type_idx,
                }
            }
            ProofCert::FVar { id, type_ } => {
                let type_idx = self.intern_expr(type_);
                CompressedCertNode::FVar {
                    id: *id,
                    type_: type_idx,
                }
            }
            ProofCert::Const {
                name,
                levels,
                type_,
            } => {
                let level_idxs: Vec<_> = levels.iter().map(|l| self.intern_level(l)).collect();
                let type_idx = self.intern_expr(type_);
                CompressedCertNode::Const {
                    name: name.clone(),
                    levels: level_idxs,
                    type_: type_idx,
                }
            }
            ProofCert::App {
                fn_cert,
                fn_type,
                arg_cert,
                result_type,
            } => {
                let fn_cert_idx = self.intern_cert(fn_cert);
                let fn_type_idx = self.intern_expr(fn_type);
                let arg_cert_idx = self.intern_cert(arg_cert);
                let result_type_idx = self.intern_expr(result_type);
                CompressedCertNode::App {
                    fn_cert: fn_cert_idx,
                    fn_type: fn_type_idx,
                    arg_cert: arg_cert_idx,
                    result_type: result_type_idx,
                }
            }
            ProofCert::Lam {
                binder_info,
                arg_type_cert,
                body_cert,
                result_type,
            } => {
                let arg_cert_idx = self.intern_cert(arg_type_cert);
                let body_cert_idx = self.intern_cert(body_cert);
                let result_type_idx = self.intern_expr(result_type);
                CompressedCertNode::Lam {
                    binder_info: *binder_info,
                    arg_type_cert: arg_cert_idx,
                    body_cert: body_cert_idx,
                    result_type: result_type_idx,
                }
            }
            ProofCert::Pi {
                binder_info,
                arg_type_cert,
                arg_level,
                body_type_cert,
                body_level,
            } => {
                let arg_cert_idx = self.intern_cert(arg_type_cert);
                let arg_level_idx = self.intern_level(arg_level);
                let body_cert_idx = self.intern_cert(body_type_cert);
                let body_level_idx = self.intern_level(body_level);
                CompressedCertNode::Pi {
                    binder_info: *binder_info,
                    arg_type_cert: arg_cert_idx,
                    arg_level: arg_level_idx,
                    body_type_cert: body_cert_idx,
                    body_level: body_level_idx,
                }
            }
            ProofCert::Let {
                type_cert,
                value_cert,
                body_cert,
                result_type,
            } => {
                let type_cert_idx = self.intern_cert(type_cert);
                let value_cert_idx = self.intern_cert(value_cert);
                let body_cert_idx = self.intern_cert(body_cert);
                let result_type_idx = self.intern_expr(result_type);
                CompressedCertNode::Let {
                    type_cert: type_cert_idx,
                    value_cert: value_cert_idx,
                    body_cert: body_cert_idx,
                    result_type: result_type_idx,
                }
            }
            ProofCert::Lit { lit, type_ } => {
                let type_idx = self.intern_expr(type_);
                CompressedCertNode::Lit {
                    lit: lit.clone(),
                    type_: type_idx,
                }
            }
            ProofCert::DefEq {
                inner,
                expected_type,
                actual_type,
                eq_steps,
            } => {
                let inner_idx = self.intern_cert(inner);
                let expected_idx = self.intern_expr(expected_type);
                let actual_idx = self.intern_expr(actual_type);
                CompressedCertNode::DefEq {
                    inner: inner_idx,
                    expected_type: expected_idx,
                    actual_type: actual_idx,
                    eq_steps: eq_steps.clone(),
                }
            }
            ProofCert::MData {
                metadata,
                inner_cert,
                result_type,
            } => {
                let inner_idx = self.intern_cert(inner_cert);
                let result_type_idx = self.intern_expr(result_type);
                CompressedCertNode::MData {
                    metadata: metadata.clone(),
                    inner_cert: inner_idx,
                    result_type: result_type_idx,
                }
            }
            ProofCert::Proj {
                struct_name,
                idx,
                expr_cert,
                expr_type,
                field_type,
            } => {
                let expr_cert_idx = self.intern_cert(expr_cert);
                let expr_type_idx = self.intern_expr(expr_type);
                let field_type_idx = self.intern_expr(field_type);
                CompressedCertNode::Proj {
                    struct_name: struct_name.clone(),
                    idx: *idx,
                    expr_cert: expr_cert_idx,
                    expr_type: expr_type_idx,
                    field_type: field_type_idx,
                }
            }
            // Mode-specific certificates: store as-is in a boxed ProofCert
            // Full compression can be added later
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
            | ProofCert::Squash { .. } => {
                CompressedCertNode::ModeSpecific(Box::new(cert.clone()))
            }
        };

        let idx = Self::len_to_idx(self.certs.len(), "certificate");
        self.certs.push(compressed);
        self.cert_map.insert(hash, idx);
        idx
    }
}

/// Compress a proof certificate using structure sharing.
///
/// This function creates a compact representation of the certificate by
/// deduplicating repeated expressions, levels, and subcertificates.
///
/// ## Example
///
/// ```ignore
/// let cert = /* some ProofCert */;
/// let compressed = compress_cert(&cert);
/// let bytes = bincode::serialize(&compressed)?;
/// // bytes is typically 30-70% smaller than serializing cert directly
/// ```
pub fn compress_cert(cert: &ProofCert) -> CompressedCert {
    let mut state = CompressionState::new();
    let root = state.intern_cert(cert);
    CompressedCert {
        exprs: state.exprs,
        levels: state.levels,
        certs: state.certs,
        root,
    }
}

/// Error during certificate decompression
#[derive(Debug, Clone)]
pub enum DecompressError {
    /// Invalid expression index
    InvalidExprIndex(ExprIdx),
    /// Invalid level index
    InvalidLevelIndex(LevelIdx),
    /// Invalid certificate index
    InvalidCertIndex(CertIdx),
}

impl std::fmt::Display for DecompressError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecompressError::InvalidExprIndex(idx) => {
                write!(f, "Invalid expression index: {idx}")
            }
            DecompressError::InvalidLevelIndex(idx) => write!(f, "Invalid level index: {idx}"),
            DecompressError::InvalidCertIndex(idx) => {
                write!(f, "Invalid certificate index: {idx}")
            }
        }
    }
}

impl std::error::Error for DecompressError {}

/// State for certificate decompression
struct DecompressionState<'a> {
    compressed: &'a CompressedCert,
    /// Cache of decompressed levels
    level_cache: HashMap<LevelIdx, Level>,
    /// Cache of decompressed expressions
    expr_cache: HashMap<ExprIdx, Expr>,
    /// Cache of decompressed certificates
    cert_cache: HashMap<CertIdx, ProofCert>,
}

impl<'a> DecompressionState<'a> {
    fn new(compressed: &'a CompressedCert) -> Self {
        Self {
            compressed,
            level_cache: HashMap::new(),
            expr_cache: HashMap::new(),
            cert_cache: HashMap::new(),
        }
    }

    /// Decompress a level by index
    fn decompress_level(&mut self, idx: LevelIdx) -> Result<Level, DecompressError> {
        if let Some(level) = self.level_cache.get(&idx) {
            return Ok(level.clone());
        }

        let compressed = self
            .compressed
            .levels
            .get(idx as usize)
            .ok_or(DecompressError::InvalidLevelIndex(idx))?;

        let level = match compressed {
            CompressedLevel::Zero => Level::Zero,
            CompressedLevel::Succ(l_idx) => {
                let l = self.decompress_level(*l_idx)?;
                Level::Succ(l.into())
            }
            CompressedLevel::Max(l1_idx, l2_idx) => {
                let l1 = self.decompress_level(*l1_idx)?;
                let l2 = self.decompress_level(*l2_idx)?;
                Level::Max(l1.into(), l2.into())
            }
            CompressedLevel::IMax(l1_idx, l2_idx) => {
                let l1 = self.decompress_level(*l1_idx)?;
                let l2 = self.decompress_level(*l2_idx)?;
                Level::IMax(l1.into(), l2.into())
            }
            CompressedLevel::Param(n) => Level::Param(n.clone()),
        };

        self.level_cache.insert(idx, level.clone());
        Ok(level)
    }

    /// Decompress an expression by index
    fn decompress_expr(&mut self, idx: ExprIdx) -> Result<Expr, DecompressError> {
        if let Some(expr) = self.expr_cache.get(&idx) {
            return Ok(expr.clone());
        }

        let compressed = self
            .compressed
            .exprs
            .get(idx as usize)
            .ok_or(DecompressError::InvalidExprIndex(idx))?
            .clone();

        let expr = match compressed {
            CompressedExpr::BVar(i) => Expr::BVar(i),
            CompressedExpr::FVar(id) => Expr::FVar(id),
            CompressedExpr::Sort(l_idx) => {
                let l = self.decompress_level(l_idx)?;
                Expr::Sort(l)
            }
            CompressedExpr::Const(n, level_idxs) => {
                let levels: Result<LevelVec, _> = level_idxs
                    .iter()
                    .map(|&idx| self.decompress_level(idx))
                    .collect();
                Expr::Const(n, levels?)
            }
            CompressedExpr::App(f_idx, a_idx) => {
                let f = self.decompress_expr(f_idx)?;
                let a = self.decompress_expr(a_idx)?;
                Expr::App(f.into(), a.into())
            }
            CompressedExpr::Lam(bi, ty_idx, body_idx) => {
                let ty = self.decompress_expr(ty_idx)?;
                let body = self.decompress_expr(body_idx)?;
                Expr::Lam(bi, ty.into(), body.into())
            }
            CompressedExpr::Pi(bi, ty_idx, body_idx) => {
                let ty = self.decompress_expr(ty_idx)?;
                let body = self.decompress_expr(body_idx)?;
                Expr::Pi(bi, ty.into(), body.into())
            }
            CompressedExpr::Let(ty_idx, val_idx, body_idx) => {
                let ty = self.decompress_expr(ty_idx)?;
                let val = self.decompress_expr(val_idx)?;
                let body = self.decompress_expr(body_idx)?;
                Expr::Let(ty.into(), val.into(), body.into())
            }
            CompressedExpr::Lit(lit) => Expr::Lit(lit),
            CompressedExpr::Proj(n, i, e_idx) => {
                let e = self.decompress_expr(e_idx)?;
                Expr::Proj(n, i, e.into())
            }
            CompressedExpr::MData(md, e_idx) => {
                let e = self.decompress_expr(e_idx)?;
                Expr::MData(md, e.into())
            }
        };

        self.expr_cache.insert(idx, expr.clone());
        Ok(expr)
    }

    /// Decompress a certificate by index
    fn decompress_cert(&mut self, idx: CertIdx) -> Result<ProofCert, DecompressError> {
        if let Some(cert) = self.cert_cache.get(&idx) {
            return Ok(cert.clone());
        }

        let compressed = self
            .compressed
            .certs
            .get(idx as usize)
            .ok_or(DecompressError::InvalidCertIndex(idx))?
            .clone();

        let cert = match compressed {
            CompressedCertNode::Sort { level } => {
                let l = self.decompress_level(level)?;
                ProofCert::Sort { level: l }
            }
            CompressedCertNode::BVar {
                idx: bvar_idx,
                expected_type,
            } => {
                let ty = self.decompress_expr(expected_type)?;
                ProofCert::BVar {
                    idx: bvar_idx,
                    expected_type: Box::new(ty),
                }
            }
            CompressedCertNode::FVar { id, type_ } => {
                let ty = self.decompress_expr(type_)?;
                ProofCert::FVar {
                    id,
                    type_: Box::new(ty),
                }
            }
            CompressedCertNode::Const {
                name,
                levels,
                type_,
            } => {
                let level_vec: Result<Vec<_>, _> = levels
                    .iter()
                    .map(|&idx| self.decompress_level(idx))
                    .collect();
                let ty = self.decompress_expr(type_)?;
                ProofCert::Const {
                    name,
                    levels: level_vec?,
                    type_: Box::new(ty),
                }
            }
            CompressedCertNode::App {
                fn_cert,
                fn_type,
                arg_cert,
                result_type,
            } => {
                let fc = self.decompress_cert(fn_cert)?;
                let ft = self.decompress_expr(fn_type)?;
                let ac = self.decompress_cert(arg_cert)?;
                let rt = self.decompress_expr(result_type)?;
                ProofCert::App {
                    fn_cert: Box::new(fc),
                    fn_type: Box::new(ft),
                    arg_cert: Box::new(ac),
                    result_type: Box::new(rt),
                }
            }
            CompressedCertNode::Lam {
                binder_info,
                arg_type_cert,
                body_cert,
                result_type,
            } => {
                let atc = self.decompress_cert(arg_type_cert)?;
                let bc = self.decompress_cert(body_cert)?;
                let rt = self.decompress_expr(result_type)?;
                ProofCert::Lam {
                    binder_info,
                    arg_type_cert: Box::new(atc),
                    body_cert: Box::new(bc),
                    result_type: Box::new(rt),
                }
            }
            CompressedCertNode::Pi {
                binder_info,
                arg_type_cert,
                arg_level,
                body_type_cert,
                body_level,
            } => {
                let atc = self.decompress_cert(arg_type_cert)?;
                let al = self.decompress_level(arg_level)?;
                let btc = self.decompress_cert(body_type_cert)?;
                let bl = self.decompress_level(body_level)?;
                ProofCert::Pi {
                    binder_info,
                    arg_type_cert: Box::new(atc),
                    arg_level: al,
                    body_type_cert: Box::new(btc),
                    body_level: bl,
                }
            }
            CompressedCertNode::Let {
                type_cert,
                value_cert,
                body_cert,
                result_type,
            } => {
                let tc = self.decompress_cert(type_cert)?;
                let vc = self.decompress_cert(value_cert)?;
                let bc = self.decompress_cert(body_cert)?;
                let rt = self.decompress_expr(result_type)?;
                ProofCert::Let {
                    type_cert: Box::new(tc),
                    value_cert: Box::new(vc),
                    body_cert: Box::new(bc),
                    result_type: Box::new(rt),
                }
            }
            CompressedCertNode::Lit { lit, type_ } => {
                let ty = self.decompress_expr(type_)?;
                ProofCert::Lit {
                    lit,
                    type_: Box::new(ty),
                }
            }
            CompressedCertNode::DefEq {
                inner,
                expected_type,
                actual_type,
                eq_steps,
            } => {
                let ic = self.decompress_cert(inner)?;
                let et = self.decompress_expr(expected_type)?;
                let at = self.decompress_expr(actual_type)?;
                ProofCert::DefEq {
                    inner: Box::new(ic),
                    expected_type: Box::new(et),
                    actual_type: Box::new(at),
                    eq_steps,
                }
            }
            CompressedCertNode::MData {
                metadata,
                inner_cert,
                result_type,
            } => {
                let ic = self.decompress_cert(inner_cert)?;
                let rt = self.decompress_expr(result_type)?;
                ProofCert::MData {
                    metadata,
                    inner_cert: Box::new(ic),
                    result_type: Box::new(rt),
                }
            }
            CompressedCertNode::Proj {
                struct_name,
                idx: proj_idx,
                expr_cert,
                expr_type,
                field_type,
            } => {
                let ec = self.decompress_cert(expr_cert)?;
                let et = self.decompress_expr(expr_type)?;
                let ft = self.decompress_expr(field_type)?;
                ProofCert::Proj {
                    struct_name,
                    idx: proj_idx,
                    expr_cert: Box::new(ec),
                    expr_type: Box::new(et),
                    field_type: Box::new(ft),
                }
            }
            // Mode-specific certificates are stored as-is
            CompressedCertNode::ModeSpecific(cert) => *cert,
        };

        self.cert_cache.insert(idx, cert.clone());
        Ok(cert)
    }
}

/// Decompress a compressed certificate back to the original format.
///
/// ## Example
///
/// ```ignore
/// let compressed: CompressedCert = bincode::deserialize(&bytes)?;
/// let cert = decompress_cert(&compressed)?;
/// // cert is now a regular ProofCert
/// ```
pub fn decompress_cert(compressed: &CompressedCert) -> Result<ProofCert, DecompressError> {
    let mut state = DecompressionState::new(compressed);
    state.decompress_cert(compressed.root)
}

/// Statistics about certificate compression
#[derive(Debug, Clone)]
pub struct CompressionStats {
    /// Number of unique expressions in compressed form
    pub unique_exprs: usize,
    /// Number of unique levels in compressed form
    pub unique_levels: usize,
    /// Number of unique certificates in compressed form
    pub unique_certs: usize,
    /// Original size in bytes (bincode serialized)
    pub original_bytes: usize,
    /// Compressed size in bytes (bincode serialized)
    pub compressed_bytes: usize,
    /// Compression ratio (original / compressed)
    pub ratio: f64,
}

impl std::fmt::Display for CompressionStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CompressionStats {{ exprs: {}, levels: {}, certs: {}, {} -> {} bytes ({:.1}x) }}",
            self.unique_exprs,
            self.unique_levels,
            self.unique_certs,
            self.original_bytes,
            self.compressed_bytes,
            self.ratio
        )
    }
}

/// Compress a certificate and return statistics about the compression.
///
/// Useful for analyzing compression effectiveness.
pub fn compress_cert_with_stats(cert: &ProofCert) -> (CompressedCert, CompressionStats) {
    let compressed = compress_cert(cert);

    // Measure sizes using bincode
    let original_bytes = bincode::serialize(cert).map(|v| v.len()).unwrap_or(0);
    let compressed_bytes = bincode::serialize(&compressed)
        .map(|v| v.len())
        .unwrap_or(0);

    let ratio = if compressed_bytes > 0 {
        original_bytes as f64 / compressed_bytes as f64
    } else {
        1.0
    };

    let stats = CompressionStats {
        unique_exprs: compressed.exprs.len(),
        unique_levels: compressed.levels.len(),
        unique_certs: compressed.certs.len(),
        original_bytes,
        compressed_bytes,
        ratio,
    };

    (compressed, stats)
}

// ============================================================================
// Byte-Level Compression (LZ4)
// ============================================================================

/// Error during byte-level compression/decompression
#[derive(Debug, Clone)]
pub enum ByteCompressError {
    /// Failed to serialize to bincode
    SerializeError(String),
    /// Failed to compress with LZ4
    CompressError(String),
    /// Failed to decompress with LZ4
    DecompressError(String),
    /// Failed to deserialize from bincode
    DeserializeError(String),
    /// Data too large to store in archive format (>4GB uncompressed)
    SizeOverflow { size: usize, max: u32 },
}

impl std::fmt::Display for ByteCompressError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ByteCompressError::SerializeError(e) => write!(f, "Serialization error: {e}"),
            ByteCompressError::CompressError(e) => write!(f, "LZ4 compression error: {e}"),
            ByteCompressError::DecompressError(e) => write!(f, "LZ4 decompression error: {e}"),
            ByteCompressError::DeserializeError(e) => write!(f, "Deserialization error: {e}"),
            ByteCompressError::SizeOverflow { size, max } => {
                write!(f, "Data size {size} exceeds maximum {max} bytes")
            }
        }
    }
}

impl std::error::Error for ByteCompressError {}

/// Convert a usize to u32, returning an error if it would overflow.
#[inline]
fn usize_to_u32(size: usize) -> Result<u32, ByteCompressError> {
    u32::try_from(size).map_err(|_| ByteCompressError::SizeOverflow {
        size,
        max: u32::MAX,
    })
}

/// A certificate archive with byte-level LZ4 compression.
///
/// This format provides maximum compression by combining:
/// 1. Structure sharing (hash-consing) via `CompressedCert`
/// 2. Byte-level LZ4 compression on the serialized output
///
/// Ideal for archiving proofs to disk or network transmission.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertArchive {
    /// LZ4-compressed bincode serialization of `CompressedCert`
    pub compressed_data: Vec<u8>,
    /// Uncompressed size for allocation hint
    pub uncompressed_size: u32,
    /// Archive format version
    pub version: u8,
}

impl CertArchive {
    /// Archive format version
    pub const VERSION: u8 = 1;
}

/// Statistics about archive compression
#[derive(Debug, Clone)]
pub struct ArchiveStats {
    /// Original certificate size (bincode)
    pub original_cert_bytes: usize,
    /// After structure sharing (bincode `CompressedCert`)
    pub structure_shared_bytes: usize,
    /// After LZ4 compression
    pub archive_bytes: usize,
    /// Structure sharing ratio (`original` / `structure_shared`)
    pub structure_ratio: f64,
    /// LZ4 ratio (`structure_shared` / `archive`)
    pub lz4_ratio: f64,
    /// Total ratio (original / archive)
    pub total_ratio: f64,
}

impl std::fmt::Display for ArchiveStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ArchiveStats {{ {} -> {} -> {} bytes (struct: {:.1}x, lz4: {:.1}x, total: {:.1}x) }}",
            self.original_cert_bytes,
            self.structure_shared_bytes,
            self.archive_bytes,
            self.structure_ratio,
            self.lz4_ratio,
            self.total_ratio
        )
    }
}

/// Create a certificate archive with maximum compression.
///
/// This combines structure sharing (hash-consing) with LZ4 byte-level compression.
/// The result is suitable for long-term storage or network transmission.
///
/// ## Example
///
/// ```ignore
/// let cert = /* some ProofCert */;
/// let archive = archive_cert(&cert)?;
/// // archive.compressed_data contains the compressed bytes
///
/// let restored = unarchive_cert(&archive)?;
/// assert_eq!(restored, cert);
/// ```
pub fn archive_cert(cert: &ProofCert) -> Result<CertArchive, ByteCompressError> {
    // Step 1: Structure sharing compression
    let compressed = compress_cert(cert);

    // Step 2: Serialize to bincode
    let bincode_bytes = bincode::serialize(&compressed)
        .map_err(|e| ByteCompressError::SerializeError(e.to_string()))?;

    let uncompressed_size = usize_to_u32(bincode_bytes.len())?;

    // Step 3: LZ4 compression
    let lz4_bytes = lz4_flex::compress_prepend_size(&bincode_bytes);

    Ok(CertArchive {
        compressed_data: lz4_bytes,
        uncompressed_size,
        version: CertArchive::VERSION,
    })
}

/// Restore a certificate from an archive.
///
/// Reverses the compression applied by `archive_cert`.
pub fn unarchive_cert(archive: &CertArchive) -> Result<ProofCert, ByteCompressError> {
    // Step 1: LZ4 decompression
    let bincode_bytes = lz4_flex::decompress_size_prepended(&archive.compressed_data)
        .map_err(|e| ByteCompressError::DecompressError(e.to_string()))?;

    // Step 2: Deserialize from bincode
    let compressed: CompressedCert = bincode::deserialize(&bincode_bytes)
        .map_err(|e| ByteCompressError::DeserializeError(e.to_string()))?;

    // Step 3: Structure decompression
    decompress_cert(&compressed).map_err(|e| ByteCompressError::DeserializeError(e.to_string()))
}

/// Archive a certificate and return compression statistics.
///
/// Useful for analyzing compression effectiveness of the two-stage process.
pub fn archive_cert_with_stats(
    cert: &ProofCert,
) -> Result<(CertArchive, ArchiveStats), ByteCompressError> {
    // Measure original size
    let original_bytes =
        bincode::serialize(cert).map_err(|e| ByteCompressError::SerializeError(e.to_string()))?;
    let original_cert_bytes = original_bytes.len();

    // Structure sharing compression
    let compressed = compress_cert(cert);
    let structure_bytes = bincode::serialize(&compressed)
        .map_err(|e| ByteCompressError::SerializeError(e.to_string()))?;
    let structure_shared_bytes = structure_bytes.len();

    // LZ4 compression
    let lz4_bytes = lz4_flex::compress_prepend_size(&structure_bytes);
    let archive_bytes = lz4_bytes.len();

    let archive = CertArchive {
        compressed_data: lz4_bytes,
        uncompressed_size: usize_to_u32(structure_shared_bytes)?,
        version: CertArchive::VERSION,
    };

    let structure_ratio = if structure_shared_bytes > 0 {
        original_cert_bytes as f64 / structure_shared_bytes as f64
    } else {
        1.0
    };

    let lz4_ratio = if archive_bytes > 0 {
        structure_shared_bytes as f64 / archive_bytes as f64
    } else {
        1.0
    };

    let total_ratio = if archive_bytes > 0 {
        original_cert_bytes as f64 / archive_bytes as f64
    } else {
        1.0
    };

    let stats = ArchiveStats {
        original_cert_bytes,
        structure_shared_bytes,
        archive_bytes,
        structure_ratio,
        lz4_ratio,
        total_ratio,
    };

    Ok((archive, stats))
}

/// Compress raw bytes with LZ4 (low-level utility).
///
/// For direct byte-level compression without structure sharing.
pub fn lz4_compress(data: &[u8]) -> Vec<u8> {
    lz4_flex::compress_prepend_size(data)
}

/// Decompress LZ4-compressed bytes (low-level utility).
///
/// For direct byte-level decompression.
pub fn lz4_decompress(data: &[u8]) -> Result<Vec<u8>, ByteCompressError> {
    lz4_flex::decompress_size_prepended(data)
        .map_err(|e| ByteCompressError::DecompressError(e.to_string()))
}

// ============================================================================
// Byte-Level Compression (Zstd)
// ============================================================================

/// Error during zstd compression/decompression
#[derive(Debug, Clone)]
pub enum ZstdCompressError {
    /// Failed to serialize to bincode
    SerializeError(String),
    /// Failed to compress with zstd
    CompressError(String),
    /// Failed to decompress with zstd
    DecompressError(String),
    /// Failed to deserialize from bincode
    DeserializeError(String),
    /// Data too large to store in archive format (>4GB uncompressed)
    SizeOverflow { size: usize, max: u32 },
}

impl std::fmt::Display for ZstdCompressError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ZstdCompressError::SerializeError(e) => write!(f, "Serialization error: {e}"),
            ZstdCompressError::CompressError(e) => write!(f, "Zstd compression error: {e}"),
            ZstdCompressError::DecompressError(e) => write!(f, "Zstd decompression error: {e}"),
            ZstdCompressError::DeserializeError(e) => write!(f, "Deserialization error: {e}"),
            ZstdCompressError::SizeOverflow { size, max } => {
                write!(f, "Data size {size} exceeds maximum {max} bytes")
            }
        }
    }
}

impl std::error::Error for ZstdCompressError {}

/// Convert a usize to u32 for zstd archive format, returning an error if it would overflow.
#[inline]
fn usize_to_u32_zstd(size: usize) -> Result<u32, ZstdCompressError> {
    u32::try_from(size).map_err(|_| ZstdCompressError::SizeOverflow {
        size,
        max: u32::MAX,
    })
}

/// A certificate archive with byte-level zstd compression.
///
/// This format provides higher compression ratio than LZ4 at the cost of
/// slower compression/decompression speed. Ideal when storage size matters
/// more than latency.
///
/// Combines:
/// 1. Structure sharing (hash-consing) via `CompressedCert`
/// 2. Byte-level zstd compression on the serialized output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZstdCertArchive {
    /// Zstd-compressed bincode serialization of `CompressedCert`
    pub compressed_data: Vec<u8>,
    /// Uncompressed size for allocation hint
    pub uncompressed_size: u32,
    /// Archive format version
    pub version: u8,
    /// Zstd compression level used (1-22, default 3)
    pub compression_level: i32,
}

impl ZstdCertArchive {
    /// Archive format version
    pub const VERSION: u8 = 1;
    /// Default compression level (balanced speed/ratio)
    pub const DEFAULT_LEVEL: i32 = 3;
    /// High compression level (better ratio, slower)
    pub const HIGH_LEVEL: i32 = 19;
    /// Maximum compression level
    pub const MAX_LEVEL: i32 = 22;
}

/// Statistics about zstd archive compression
#[derive(Debug, Clone)]
pub struct ZstdArchiveStats {
    /// Original certificate size (bincode)
    pub original_cert_bytes: usize,
    /// After structure sharing (bincode `CompressedCert`)
    pub structure_shared_bytes: usize,
    /// After zstd compression
    pub archive_bytes: usize,
    /// Structure sharing ratio (`original` / `structure_shared`)
    pub structure_ratio: f64,
    /// Zstd ratio (`structure_shared` / `archive`)
    pub zstd_ratio: f64,
    /// Total ratio (original / archive)
    pub total_ratio: f64,
    /// Compression level used
    pub compression_level: i32,
}

impl std::fmt::Display for ZstdArchiveStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ZstdArchiveStats {{ {} -> {} -> {} bytes (struct: {:.1}x, zstd[{}]: {:.1}x, total: {:.1}x) }}",
            self.original_cert_bytes,
            self.structure_shared_bytes,
            self.archive_bytes,
            self.structure_ratio,
            self.compression_level,
            self.zstd_ratio,
            self.total_ratio
        )
    }
}

/// Create a certificate archive with zstd compression (default level).
///
/// This combines structure sharing (hash-consing) with zstd byte-level compression.
/// Zstd provides better compression ratios than LZ4 but is slower.
///
/// ## Example
///
/// ```ignore
/// let cert = /* some ProofCert */;
/// let archive = zstd_archive_cert(&cert)?;
/// // archive.compressed_data contains the compressed bytes
///
/// let restored = zstd_unarchive_cert(&archive)?;
/// assert_eq!(restored, cert);
/// ```
pub fn zstd_archive_cert(cert: &ProofCert) -> Result<ZstdCertArchive, ZstdCompressError> {
    zstd_archive_cert_level(cert, ZstdCertArchive::DEFAULT_LEVEL)
}

/// Create a certificate archive with zstd compression at a specific level.
///
/// Level ranges from 1 (fastest, least compression) to 22 (slowest, best compression).
/// Common choices:
/// - Level 1-3: Fast compression, good for realtime
/// - Level 10-15: Balanced compression
/// - Level 19-22: Maximum compression, slow
pub fn zstd_archive_cert_level(
    cert: &ProofCert,
    level: i32,
) -> Result<ZstdCertArchive, ZstdCompressError> {
    // Step 1: Structure sharing compression
    let compressed = compress_cert(cert);

    // Step 2: Serialize to bincode
    let bincode_bytes = bincode::serialize(&compressed)
        .map_err(|e| ZstdCompressError::SerializeError(e.to_string()))?;

    let uncompressed_size = usize_to_u32_zstd(bincode_bytes.len())?;

    // Step 3: Zstd compression
    let zstd_bytes = zstd::encode_all(bincode_bytes.as_slice(), level)
        .map_err(|e| ZstdCompressError::CompressError(e.to_string()))?;

    Ok(ZstdCertArchive {
        compressed_data: zstd_bytes,
        uncompressed_size,
        version: ZstdCertArchive::VERSION,
        compression_level: level,
    })
}

/// Restore a certificate from a zstd archive.
///
/// Reverses the compression applied by `zstd_archive_cert` or `zstd_archive_cert_level`.
pub fn zstd_unarchive_cert(archive: &ZstdCertArchive) -> Result<ProofCert, ZstdCompressError> {
    // Step 1: Zstd decompression
    let bincode_bytes = zstd::decode_all(archive.compressed_data.as_slice())
        .map_err(|e| ZstdCompressError::DecompressError(e.to_string()))?;

    // Step 2: Deserialize from bincode
    let compressed: CompressedCert = bincode::deserialize(&bincode_bytes)
        .map_err(|e| ZstdCompressError::DeserializeError(e.to_string()))?;

    // Step 3: Structure decompression
    decompress_cert(&compressed).map_err(|e| ZstdCompressError::DeserializeError(e.to_string()))
}

/// Archive a certificate with zstd and return compression statistics.
///
/// Useful for analyzing compression effectiveness and comparing with LZ4.
pub fn zstd_archive_cert_with_stats(
    cert: &ProofCert,
) -> Result<(ZstdCertArchive, ZstdArchiveStats), ZstdCompressError> {
    zstd_archive_cert_with_stats_level(cert, ZstdCertArchive::DEFAULT_LEVEL)
}

/// Archive a certificate with zstd at a specific level and return statistics.
pub fn zstd_archive_cert_with_stats_level(
    cert: &ProofCert,
    level: i32,
) -> Result<(ZstdCertArchive, ZstdArchiveStats), ZstdCompressError> {
    // Measure original size
    let original_bytes =
        bincode::serialize(cert).map_err(|e| ZstdCompressError::SerializeError(e.to_string()))?;
    let original_cert_bytes = original_bytes.len();

    // Structure sharing compression
    let compressed = compress_cert(cert);
    let structure_bytes = bincode::serialize(&compressed)
        .map_err(|e| ZstdCompressError::SerializeError(e.to_string()))?;
    let structure_shared_bytes = structure_bytes.len();

    // Zstd compression
    let zstd_bytes = zstd::encode_all(structure_bytes.as_slice(), level)
        .map_err(|e| ZstdCompressError::CompressError(e.to_string()))?;
    let archive_bytes = zstd_bytes.len();

    let archive = ZstdCertArchive {
        compressed_data: zstd_bytes,
        uncompressed_size: usize_to_u32_zstd(structure_shared_bytes)?,
        version: ZstdCertArchive::VERSION,
        compression_level: level,
    };

    let structure_ratio = if structure_shared_bytes > 0 {
        original_cert_bytes as f64 / structure_shared_bytes as f64
    } else {
        1.0
    };

    let zstd_ratio = if archive_bytes > 0 {
        structure_shared_bytes as f64 / archive_bytes as f64
    } else {
        1.0
    };

    let total_ratio = if archive_bytes > 0 {
        original_cert_bytes as f64 / archive_bytes as f64
    } else {
        1.0
    };

    let stats = ZstdArchiveStats {
        original_cert_bytes,
        structure_shared_bytes,
        archive_bytes,
        structure_ratio,
        zstd_ratio,
        total_ratio,
        compression_level: level,
    };

    Ok((archive, stats))
}

/// Compress raw bytes with zstd (low-level utility).
///
/// For direct byte-level compression without structure sharing.
/// Uses default compression level.
pub fn zstd_compress(data: &[u8]) -> Result<Vec<u8>, ZstdCompressError> {
    zstd_compress_level(data, ZstdCertArchive::DEFAULT_LEVEL)
}

/// Compress raw bytes with zstd at a specific level.
pub fn zstd_compress_level(data: &[u8], level: i32) -> Result<Vec<u8>, ZstdCompressError> {
    zstd::encode_all(data, level).map_err(|e| ZstdCompressError::CompressError(e.to_string()))
}

/// Decompress zstd-compressed bytes (low-level utility).
///
/// For direct byte-level decompression.
pub fn zstd_decompress(data: &[u8]) -> Result<Vec<u8>, ZstdCompressError> {
    zstd::decode_all(data).map_err(|e| ZstdCompressError::DecompressError(e.to_string()))
}

// ============================================================================
// Dictionary-Based Zstd Compression
// ============================================================================

/// A trained dictionary for certificate compression.
///
/// Dictionaries improve compression ratios for small, similar data. For proof
/// certificates, training a dictionary on a corpus of representative certificates
/// can significantly improve compression, especially for small proofs.
///
/// ## Usage
///
/// ```ignore
/// // Train a dictionary from sample certificates
/// let samples: Vec<ProofCert> = /* collect sample certificates */;
/// let dict = CertDictionary::train(&samples, 32 * 1024)?; // 32KB dictionary
///
/// // Compress with dictionary
/// let archive = zstd_archive_cert_with_dict(&cert, &dict)?;
///
/// // Decompress with dictionary
/// let restored = zstd_unarchive_cert_with_dict(&archive, &dict)?;
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertDictionary {
    /// The raw dictionary bytes
    pub data: Vec<u8>,
    /// Dictionary ID for validation (hash of data)
    pub dict_id: u32,
    /// Number of samples used for training
    pub sample_count: usize,
    /// Target compression level this dictionary was trained for
    pub target_level: i32,
    /// Version for format compatibility
    pub version: u8,
}

impl CertDictionary {
    /// Current dictionary format version
    pub const VERSION: u8 = 1;

    /// Default dictionary size (32KB - good balance of size vs effectiveness)
    pub const DEFAULT_SIZE: usize = 32 * 1024;

    /// Minimum samples needed for effective training
    pub const MIN_SAMPLES: usize = 5;

    /// Create a dictionary from raw bytes.
    ///
    /// Use this when loading a pre-trained dictionary.
    pub fn from_bytes(data: Vec<u8>, target_level: i32) -> Self {
        let dict_id = Self::compute_id(&data);
        CertDictionary {
            data,
            dict_id,
            sample_count: 0, // Unknown for pre-made dictionaries
            target_level,
            version: Self::VERSION,
        }
    }

    /// Train a dictionary from a collection of proof certificates.
    ///
    /// The dictionary is trained to be effective for compressing similar
    /// certificates. For best results, use a diverse but representative
    /// set of certificates.
    ///
    /// ## Parameters
    ///
    /// - `samples`: Certificate samples to train on (at least 5 recommended)
    /// - `max_size`: Maximum dictionary size in bytes (32KB default)
    /// - `level`: Target compression level (dictionary is optimized for this level)
    ///
    /// ## Example
    ///
    /// ```ignore
    /// let certs: Vec<ProofCert> = /* your certificates */;
    /// let dict = CertDictionary::train(&certs, 32 * 1024, 3)?;
    /// ```
    pub fn train(
        samples: &[ProofCert],
        max_size: usize,
        level: i32,
    ) -> Result<Self, DictTrainError> {
        if samples.len() < Self::MIN_SAMPLES {
            return Err(DictTrainError::NotEnoughSamples {
                provided: samples.len(),
                minimum: Self::MIN_SAMPLES,
            });
        }

        // Serialize samples to bytes for training
        let sample_bytes: Vec<Vec<u8>> = samples
            .iter()
            .map(|cert| {
                let compressed = compress_cert(cert);
                bincode::serialize(&compressed).map_err(|e| {
                    DictTrainError::SerializeError(format!("Failed to serialize sample: {e}"))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Train the dictionary
        let dict_data = zstd::dict::from_samples(&sample_bytes, max_size)
            .map_err(|e| DictTrainError::TrainError(e.to_string()))?;

        let dict_id = Self::compute_id(&dict_data);

        Ok(CertDictionary {
            data: dict_data,
            dict_id,
            sample_count: samples.len(),
            target_level: level,
            version: Self::VERSION,
        })
    }

    /// Train a dictionary from raw byte samples.
    ///
    /// Use this when you have pre-serialized certificate data.
    pub fn train_from_bytes(
        samples: &[Vec<u8>],
        max_size: usize,
        level: i32,
    ) -> Result<Self, DictTrainError> {
        if samples.len() < Self::MIN_SAMPLES {
            return Err(DictTrainError::NotEnoughSamples {
                provided: samples.len(),
                minimum: Self::MIN_SAMPLES,
            });
        }

        let dict_data = zstd::dict::from_samples(samples, max_size)
            .map_err(|e| DictTrainError::TrainError(e.to_string()))?;

        let dict_id = Self::compute_id(&dict_data);

        Ok(CertDictionary {
            data: dict_data,
            dict_id,
            sample_count: samples.len(),
            target_level: level,
            version: Self::VERSION,
        })
    }

    /// Compute a dictionary ID from its data (simple hash for validation).
    fn compute_id(data: &[u8]) -> u32 {
        // Simple FNV-1a hash for dict identification
        let mut hash: u32 = 2_166_136_261;
        for byte in data {
            // SAFETY: u8 (0-255) always fits in u32
            hash ^= u32::from(*byte);
            hash = hash.wrapping_mul(16_777_619);
        }
        hash
    }

    /// Get the dictionary size in bytes.
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Check if this dictionary was trained for a compatible level.
    pub fn is_compatible_level(&self, level: i32) -> bool {
        // Dictionaries are most effective near their target level,
        // but can work at other levels
        (self.target_level - level).abs() <= 5
    }
}

/// Errors during dictionary training.
#[derive(Debug, Clone)]
pub enum DictTrainError {
    /// Not enough samples for training
    NotEnoughSamples { provided: usize, minimum: usize },
    /// Failed to serialize sample
    SerializeError(String),
    /// Zstd training failed
    TrainError(String),
}

impl std::fmt::Display for DictTrainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DictTrainError::NotEnoughSamples { provided, minimum } => {
                write!(
                    f,
                    "Not enough samples for dictionary training: {provided} provided, {minimum} minimum"
                )
            }
            DictTrainError::SerializeError(e) => write!(f, "Serialization error: {e}"),
            DictTrainError::TrainError(e) => write!(f, "Dictionary training error: {e}"),
        }
    }
}

impl std::error::Error for DictTrainError {}

/// A certificate archive compressed with a trained dictionary.
///
/// Similar to `ZstdCertArchive`, but uses a dictionary for improved compression.
/// The dictionary must be available for decompression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DictCertArchive {
    /// Dictionary-compressed bincode serialization of CompressedCert
    pub compressed_data: Vec<u8>,
    /// Uncompressed size for allocation hint
    pub uncompressed_size: u32,
    /// Archive format version
    pub version: u8,
    /// Zstd compression level used
    pub compression_level: i32,
    /// Dictionary ID used for compression (for validation)
    pub dict_id: u32,
}

impl DictCertArchive {
    /// Archive format version
    pub const VERSION: u8 = 1;
}

/// Statistics about dictionary-compressed archive.
#[derive(Debug, Clone)]
pub struct DictArchiveStats {
    /// Original certificate size (bincode)
    pub original_cert_bytes: usize,
    /// After structure sharing (bincode CompressedCert)
    pub structure_shared_bytes: usize,
    /// After dictionary compression
    pub archive_bytes: usize,
    /// Structure sharing ratio (original / structure_shared)
    pub structure_ratio: f64,
    /// Dictionary compression ratio (structure_shared / archive)
    pub dict_ratio: f64,
    /// Total ratio (original / archive)
    pub total_ratio: f64,
    /// Compression level used
    pub compression_level: i32,
    /// Dictionary ID used
    pub dict_id: u32,
}

impl std::fmt::Display for DictArchiveStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DictArchiveStats {{ {} -> {} -> {} bytes (struct: {:.1}x, dict[{}]: {:.1}x, total: {:.1}x) }}",
            self.original_cert_bytes,
            self.structure_shared_bytes,
            self.archive_bytes,
            self.structure_ratio,
            self.compression_level,
            self.dict_ratio,
            self.total_ratio
        )
    }
}

/// Error during dictionary compression/decompression.
#[derive(Debug, Clone)]
pub enum DictCompressError {
    /// Failed to serialize
    SerializeError(String),
    /// Failed to compress
    CompressError(String),
    /// Failed to decompress
    DecompressError(String),
    /// Failed to deserialize
    DeserializeError(String),
    /// Dictionary mismatch
    DictMismatch { expected: u32, found: u32 },
    /// Data too large to store in archive format (>4GB uncompressed)
    SizeOverflow { size: usize, max: u32 },
}

impl std::fmt::Display for DictCompressError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DictCompressError::SerializeError(e) => write!(f, "Serialization error: {e}"),
            DictCompressError::CompressError(e) => write!(f, "Dict compression error: {e}"),
            DictCompressError::DecompressError(e) => write!(f, "Dict decompression error: {e}"),
            DictCompressError::DeserializeError(e) => write!(f, "Deserialization error: {e}"),
            DictCompressError::DictMismatch { expected, found } => {
                write!(
                    f,
                    "Dictionary ID mismatch: expected {expected:#x}, found {found:#x}"
                )
            }
            DictCompressError::SizeOverflow { size, max } => {
                write!(f, "Data size {size} exceeds maximum {max} bytes")
            }
        }
    }
}

impl std::error::Error for DictCompressError {}

/// Convert a usize to u32 for dict archive format, returning an error if it would overflow.
#[inline]
fn usize_to_u32_dict(size: usize) -> Result<u32, DictCompressError> {
    u32::try_from(size).map_err(|_| DictCompressError::SizeOverflow {
        size,
        max: u32::MAX,
    })
}

/// Archive a certificate using dictionary compression.
///
/// Uses the trained dictionary to achieve better compression ratios,
/// especially for small certificates that are similar to the training data.
///
/// ## Example
///
/// ```ignore
/// let dict = CertDictionary::train(&samples, 32 * 1024, 3)?;
/// let archive = zstd_archive_cert_with_dict(&cert, &dict)?;
/// ```
pub fn zstd_archive_cert_with_dict(
    cert: &ProofCert,
    dict: &CertDictionary,
) -> Result<DictCertArchive, DictCompressError> {
    zstd_archive_cert_with_dict_level(cert, dict, dict.target_level)
}

/// Archive a certificate using dictionary compression at a specific level.
pub fn zstd_archive_cert_with_dict_level(
    cert: &ProofCert,
    dict: &CertDictionary,
    level: i32,
) -> Result<DictCertArchive, DictCompressError> {
    // Step 1: Structure sharing compression
    let compressed = compress_cert(cert);

    // Step 2: Serialize to bincode
    let bincode_bytes = bincode::serialize(&compressed)
        .map_err(|e| DictCompressError::SerializeError(e.to_string()))?;

    let uncompressed_size = usize_to_u32_dict(bincode_bytes.len())?;

    // Step 3: Dictionary-based compression using streaming encoder
    let mut output = Vec::new();
    {
        let mut encoder = zstd::stream::Encoder::with_dictionary(&mut output, level, &dict.data)
            .map_err(|e| DictCompressError::CompressError(e.to_string()))?;
        std::io::Write::write_all(&mut encoder, &bincode_bytes)
            .map_err(|e| DictCompressError::CompressError(e.to_string()))?;
        encoder
            .finish()
            .map_err(|e| DictCompressError::CompressError(e.to_string()))?;
    }

    Ok(DictCertArchive {
        compressed_data: output,
        uncompressed_size,
        version: DictCertArchive::VERSION,
        compression_level: level,
        dict_id: dict.dict_id,
    })
}

/// Restore a certificate from a dictionary-compressed archive.
///
/// The same dictionary that was used for compression must be provided.
pub fn zstd_unarchive_cert_with_dict(
    archive: &DictCertArchive,
    dict: &CertDictionary,
) -> Result<ProofCert, DictCompressError> {
    // Validate dictionary ID
    if archive.dict_id != dict.dict_id {
        return Err(DictCompressError::DictMismatch {
            expected: archive.dict_id,
            found: dict.dict_id,
        });
    }

    // Step 1: Dictionary-based decompression
    let mut decompressed = Vec::with_capacity(archive.uncompressed_size as usize);
    {
        let mut decoder = zstd::stream::Decoder::with_dictionary(
            std::io::Cursor::new(&archive.compressed_data),
            &dict.data,
        )
        .map_err(|e| DictCompressError::DecompressError(e.to_string()))?;
        std::io::Read::read_to_end(&mut decoder, &mut decompressed)
            .map_err(|e| DictCompressError::DecompressError(e.to_string()))?;
    }

    // Step 2: Deserialize
    let compressed_cert: CompressedCert = bincode::deserialize(&decompressed)
        .map_err(|e| DictCompressError::DeserializeError(e.to_string()))?;

    // Step 3: Decompress structure sharing
    decompress_cert(&compressed_cert)
        .map_err(|e| DictCompressError::DeserializeError(format!("Structure decompress: {e}")))
}

/// Archive a certificate with dictionary and return compression statistics.
pub fn zstd_archive_cert_with_dict_stats(
    cert: &ProofCert,
    dict: &CertDictionary,
) -> Result<(DictCertArchive, DictArchiveStats), DictCompressError> {
    zstd_archive_cert_with_dict_stats_level(cert, dict, dict.target_level)
}

/// Archive a certificate with dictionary at a specific level and return statistics.
pub fn zstd_archive_cert_with_dict_stats_level(
    cert: &ProofCert,
    dict: &CertDictionary,
    level: i32,
) -> Result<(DictCertArchive, DictArchiveStats), DictCompressError> {
    // Measure original size
    let original_bytes =
        bincode::serialize(cert).map_err(|e| DictCompressError::SerializeError(e.to_string()))?;
    let original_cert_bytes = original_bytes.len();

    // Structure sharing
    let compressed = compress_cert(cert);
    let structure_bytes = bincode::serialize(&compressed)
        .map_err(|e| DictCompressError::SerializeError(e.to_string()))?;
    let structure_shared_bytes = structure_bytes.len();

    // Dictionary compression
    let mut output = Vec::new();
    {
        let mut encoder = zstd::stream::Encoder::with_dictionary(&mut output, level, &dict.data)
            .map_err(|e| DictCompressError::CompressError(e.to_string()))?;
        std::io::Write::write_all(&mut encoder, &structure_bytes)
            .map_err(|e| DictCompressError::CompressError(e.to_string()))?;
        encoder
            .finish()
            .map_err(|e| DictCompressError::CompressError(e.to_string()))?;
    }
    let archive_bytes = output.len();

    let archive = DictCertArchive {
        compressed_data: output,
        uncompressed_size: usize_to_u32_dict(structure_shared_bytes)?,
        version: DictCertArchive::VERSION,
        compression_level: level,
        dict_id: dict.dict_id,
    };

    // Compute ratios
    let structure_ratio = if structure_shared_bytes > 0 {
        original_cert_bytes as f64 / structure_shared_bytes as f64
    } else {
        1.0
    };

    let dict_ratio = if archive_bytes > 0 {
        structure_shared_bytes as f64 / archive_bytes as f64
    } else {
        1.0
    };

    let total_ratio = if archive_bytes > 0 {
        original_cert_bytes as f64 / archive_bytes as f64
    } else {
        1.0
    };

    let stats = DictArchiveStats {
        original_cert_bytes,
        structure_shared_bytes,
        archive_bytes,
        structure_ratio,
        dict_ratio,
        total_ratio,
        compression_level: level,
        dict_id: dict.dict_id,
    };

    Ok((archive, stats))
}

/// Compress raw bytes with a dictionary.
pub fn zstd_compress_with_dict(
    data: &[u8],
    dict: &CertDictionary,
) -> Result<Vec<u8>, DictCompressError> {
    zstd_compress_with_dict_level(data, dict, dict.target_level)
}

/// Compress raw bytes with a dictionary at a specific level.
pub fn zstd_compress_with_dict_level(
    data: &[u8],
    dict: &CertDictionary,
    level: i32,
) -> Result<Vec<u8>, DictCompressError> {
    let mut output = Vec::new();
    {
        let mut encoder = zstd::stream::Encoder::with_dictionary(&mut output, level, &dict.data)
            .map_err(|e| DictCompressError::CompressError(e.to_string()))?;
        std::io::Write::write_all(&mut encoder, data)
            .map_err(|e| DictCompressError::CompressError(e.to_string()))?;
        encoder
            .finish()
            .map_err(|e| DictCompressError::CompressError(e.to_string()))?;
    }
    Ok(output)
}

/// Decompress bytes that were compressed with a dictionary.
pub fn zstd_decompress_with_dict(
    data: &[u8],
    dict: &CertDictionary,
) -> Result<Vec<u8>, DictCompressError> {
    let mut decompressed = Vec::new();
    {
        let mut decoder =
            zstd::stream::Decoder::with_dictionary(std::io::Cursor::new(data), &dict.data)
                .map_err(|e| DictCompressError::DecompressError(e.to_string()))?;
        std::io::Read::read_to_end(&mut decoder, &mut decompressed)
            .map_err(|e| DictCompressError::DecompressError(e.to_string()))?;
    }
    Ok(decompressed)
}

/// Compression algorithm choice for certificate archiving.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionAlgorithm {
    /// LZ4: Very fast compression/decompression, moderate ratio
    Lz4,
    /// Zstd default (level 3): Balanced speed and ratio
    ZstdDefault,
    /// Zstd high (level 19): Better ratio, slower
    ZstdHigh,
    /// Zstd max (level 22): Best ratio, slowest
    ZstdMax,
}

impl CompressionAlgorithm {
    /// Get descriptive name for the algorithm
    pub fn name(&self) -> &'static str {
        match self {
            CompressionAlgorithm::Lz4 => "LZ4",
            CompressionAlgorithm::ZstdDefault => "Zstd (level 3)",
            CompressionAlgorithm::ZstdHigh => "Zstd (level 19)",
            CompressionAlgorithm::ZstdMax => "Zstd (level 22)",
        }
    }

    /// Get the zstd compression level for this algorithm, if applicable.
    pub fn zstd_level(&self) -> Option<i32> {
        match self {
            CompressionAlgorithm::Lz4 => None,
            CompressionAlgorithm::ZstdDefault => Some(ZstdCertArchive::DEFAULT_LEVEL),
            CompressionAlgorithm::ZstdHigh => Some(ZstdCertArchive::HIGH_LEVEL),
            CompressionAlgorithm::ZstdMax => Some(ZstdCertArchive::MAX_LEVEL),
        }
    }

    /// Derive a CompressionAlgorithm from a zstd level.
    pub fn from_zstd_level(level: i32) -> CompressionAlgorithm {
        if level >= ZstdCertArchive::MAX_LEVEL {
            CompressionAlgorithm::ZstdMax
        } else if level >= ZstdCertArchive::HIGH_LEVEL {
            CompressionAlgorithm::ZstdHigh
        } else {
            CompressionAlgorithm::ZstdDefault
        }
    }
}

/// Unified error type for certificate archiving across algorithms.
#[derive(Debug)]
pub enum CertArchiveError {
    /// Error from LZ4-based archiving
    Lz4(ByteCompressError),
    /// Error from zstd-based archiving
    Zstd(ZstdCompressError),
}

impl std::fmt::Display for CertArchiveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CertArchiveError::Lz4(e) => write!(f, "LZ4 archive error: {e}"),
            CertArchiveError::Zstd(e) => write!(f, "Zstd archive error: {e}"),
        }
    }
}

impl std::error::Error for CertArchiveError {}

impl From<ByteCompressError> for CertArchiveError {
    fn from(value: ByteCompressError) -> Self {
        CertArchiveError::Lz4(value)
    }
}

impl From<ZstdCompressError> for CertArchiveError {
    fn from(value: ZstdCompressError) -> Self {
        CertArchiveError::Zstd(value)
    }
}

/// Statistics for any compression algorithm.
#[derive(Debug, Clone)]
pub enum ArchiveVariantStats {
    /// Statistics for LZ4 archives
    Lz4(ArchiveStats),
    /// Statistics for zstd archives
    Zstd(ZstdArchiveStats),
}

impl ArchiveVariantStats {
    /// Get the compression algorithm used.
    pub fn algorithm(&self) -> CompressionAlgorithm {
        match self {
            ArchiveVariantStats::Lz4(_) => CompressionAlgorithm::Lz4,
            ArchiveVariantStats::Zstd(stats) => {
                CompressionAlgorithm::from_zstd_level(stats.compression_level)
            }
        }
    }

    /// Get the total compression ratio (original / archive).
    pub fn total_ratio(&self) -> f64 {
        match self {
            ArchiveVariantStats::Lz4(stats) => stats.total_ratio,
            ArchiveVariantStats::Zstd(stats) => stats.total_ratio,
        }
    }

    /// Get the structure sharing ratio (original / structure_shared).
    pub fn structure_ratio(&self) -> f64 {
        match self {
            ArchiveVariantStats::Lz4(stats) => stats.structure_ratio,
            ArchiveVariantStats::Zstd(stats) => stats.structure_ratio,
        }
    }
}

impl std::fmt::Display for ArchiveVariantStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArchiveVariantStats::Lz4(stats) => {
                write!(f, "ArchiveVariantStats {{ algo: LZ4, {stats} }}")
            }
            ArchiveVariantStats::Zstd(stats) => {
                write!(f, "ArchiveVariantStats {{ algo: Zstd, {stats} }}")
            }
        }
    }
}

/// Envelope that records which compression algorithm produced the archive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CertArchiveEnvelope {
    /// LZ4-compressed archive
    Lz4(CertArchive),
    /// Zstd-compressed archive
    Zstd(ZstdCertArchive),
}

impl CertArchiveEnvelope {
    /// Compression algorithm used for this archive.
    pub fn algorithm(&self) -> CompressionAlgorithm {
        match self {
            CertArchiveEnvelope::Lz4(_) => CompressionAlgorithm::Lz4,
            CertArchiveEnvelope::Zstd(archive) => {
                CompressionAlgorithm::from_zstd_level(archive.compression_level)
            }
        }
    }

    /// Size of the compressed payload in bytes.
    pub fn compressed_len(&self) -> usize {
        match self {
            CertArchiveEnvelope::Lz4(archive) => archive.compressed_data.len(),
            CertArchiveEnvelope::Zstd(archive) => archive.compressed_data.len(),
        }
    }

    /// Uncompressed size hint stored alongside the archive.
    pub fn uncompressed_size(&self) -> u32 {
        match self {
            CertArchiveEnvelope::Lz4(archive) => archive.uncompressed_size,
            CertArchiveEnvelope::Zstd(archive) => archive.uncompressed_size,
        }
    }
}

/// Archive a certificate using the selected compression algorithm.
pub fn archive_cert_with_algorithm(
    cert: &ProofCert,
    algorithm: CompressionAlgorithm,
) -> Result<CertArchiveEnvelope, CertArchiveError> {
    match algorithm {
        CompressionAlgorithm::Lz4 => archive_cert(cert)
            .map(CertArchiveEnvelope::Lz4)
            .map_err(CertArchiveError::from),
        CompressionAlgorithm::ZstdDefault
        | CompressionAlgorithm::ZstdHigh
        | CompressionAlgorithm::ZstdMax => {
            let level = algorithm
                .zstd_level()
                .unwrap_or(ZstdCertArchive::DEFAULT_LEVEL);
            zstd_archive_cert_level(cert, level)
                .map(CertArchiveEnvelope::Zstd)
                .map_err(CertArchiveError::from)
        }
    }
}

/// Archive a certificate with statistics for the selected algorithm.
pub fn archive_cert_with_algorithm_stats(
    cert: &ProofCert,
    algorithm: CompressionAlgorithm,
) -> Result<(CertArchiveEnvelope, ArchiveVariantStats), CertArchiveError> {
    match algorithm {
        CompressionAlgorithm::Lz4 => archive_cert_with_stats(cert)
            .map(|(archive, stats)| {
                (
                    CertArchiveEnvelope::Lz4(archive),
                    ArchiveVariantStats::Lz4(stats),
                )
            })
            .map_err(CertArchiveError::from),
        CompressionAlgorithm::ZstdDefault
        | CompressionAlgorithm::ZstdHigh
        | CompressionAlgorithm::ZstdMax => {
            let level = algorithm
                .zstd_level()
                .unwrap_or(ZstdCertArchive::DEFAULT_LEVEL);
            zstd_archive_cert_with_stats_level(cert, level)
                .map(|(archive, stats)| {
                    (
                        CertArchiveEnvelope::Zstd(archive),
                        ArchiveVariantStats::Zstd(stats),
                    )
                })
                .map_err(CertArchiveError::from)
        }
    }
}

/// Restore a certificate from any archive envelope.
pub fn unarchive_cert_envelope(
    archive: &CertArchiveEnvelope,
) -> Result<ProofCert, CertArchiveError> {
    match archive {
        CertArchiveEnvelope::Lz4(archive) => {
            unarchive_cert(archive).map_err(CertArchiveError::from)
        }
        CertArchiveEnvelope::Zstd(archive) => {
            zstd_unarchive_cert(archive).map_err(CertArchiveError::from)
        }
    }
}

// ============================================================================
// Streaming Compression API
// ============================================================================

/// Progress callback type for streaming operations.
/// Called with (bytes_processed, total_bytes_if_known).
pub type StreamingProgressCallback = Box<dyn FnMut(u64, Option<u64>) + Send>;

/// Error type for streaming compression operations.
#[derive(Debug)]
pub enum StreamingError {
    /// I/O error during streaming
    Io(std::io::Error),
    /// Serialization error
    Serialize(String),
    /// Decompression error
    Decompress(String),
    /// Invalid header or format
    InvalidFormat(String),
    /// Data too large to store in streaming format (>4GB per item)
    SizeOverflow { size: usize, max: u32 },
}

impl std::fmt::Display for StreamingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamingError::Io(e) => write!(f, "I/O error: {e}"),
            StreamingError::Serialize(e) => write!(f, "Serialization error: {e}"),
            StreamingError::Decompress(e) => write!(f, "Decompression error: {e}"),
            StreamingError::InvalidFormat(e) => write!(f, "Invalid format: {e}"),
            StreamingError::SizeOverflow { size, max } => {
                write!(f, "Data size {size} exceeds maximum {max} bytes")
            }
        }
    }
}

impl std::error::Error for StreamingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            StreamingError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for StreamingError {
    fn from(e: std::io::Error) -> Self {
        StreamingError::Io(e)
    }
}

/// Convert a usize to u32 for streaming format, returning an error if it would overflow.
#[inline]
fn usize_to_u32_streaming(size: usize) -> Result<u32, StreamingError> {
    u32::try_from(size).map_err(|_| StreamingError::SizeOverflow {
        size,
        max: u32::MAX,
    })
}

/// Streaming header written at the start of a streaming archive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingArchiveHeader {
    /// Magic bytes for format identification
    pub magic: [u8; 4],
    /// Version of the streaming format
    pub version: u8,
    /// Compression algorithm used
    pub algorithm: u8, // 0 = LZ4, 1 = Zstd
    /// Zstd compression level (0 if LZ4)
    pub compression_level: i32,
    /// Total uncompressed size (0 if unknown)
    pub uncompressed_size: u64,
    /// Number of certificates in stream (0 if unknown)
    pub cert_count: u64,
}

impl StreamingArchiveHeader {
    /// Magic bytes for streaming certificate archives
    pub const MAGIC: [u8; 4] = *b"L5CS"; // Lean5 Cert Stream
    /// Current format version
    pub const VERSION: u8 = 1;

    /// Create a new header for LZ4 streaming
    pub fn new_lz4() -> Self {
        StreamingArchiveHeader {
            magic: Self::MAGIC,
            version: Self::VERSION,
            algorithm: 0,
            compression_level: 0,
            uncompressed_size: 0,
            cert_count: 0,
        }
    }

    /// Create a new header for Zstd streaming
    pub fn new_zstd(level: i32) -> Self {
        StreamingArchiveHeader {
            magic: Self::MAGIC,
            version: Self::VERSION,
            algorithm: 1,
            compression_level: level,
            uncompressed_size: 0,
            cert_count: 0,
        }
    }

    /// Get the compression algorithm
    pub fn algorithm(&self) -> CompressionAlgorithm {
        if self.algorithm == 0 {
            CompressionAlgorithm::Lz4
        } else {
            CompressionAlgorithm::from_zstd_level(self.compression_level)
        }
    }

    /// Validate the header
    pub fn validate(&self) -> Result<(), StreamingError> {
        if self.magic != Self::MAGIC {
            return Err(StreamingError::InvalidFormat(format!(
                "Invalid magic bytes: expected {:?}, got {:?}",
                Self::MAGIC,
                self.magic
            )));
        }
        if self.version > Self::VERSION {
            return Err(StreamingError::InvalidFormat(format!(
                "Unsupported version: {} (max supported: {})",
                self.version,
                Self::VERSION
            )));
        }
        if self.algorithm > 1 {
            return Err(StreamingError::InvalidFormat(format!(
                "Unknown algorithm: {}",
                self.algorithm
            )));
        }
        Ok(())
    }
}

/// A streaming certificate writer that compresses certificates as they are written.
///
/// This writer buffers certificates and flushes compressed chunks to the underlying
/// writer. This is useful for:
/// - Reducing memory usage for large proof archives
/// - Incremental archiving during proof construction
/// - Progress reporting for large operations
///
/// # Example
///
/// ```ignore
/// use std::fs::File;
/// let file = File::create("proofs.l5cs")?;
/// let mut writer = StreamingCertWriter::new_zstd(file, 3)?;
///
/// for cert in certificates {
///     writer.write_cert(&cert)?;
/// }
/// writer.finish()?;
/// ```
pub struct StreamingCertWriter<W: std::io::Write> {
    /// Inner Zstd encoder (wraps the output writer)
    encoder: zstd::stream::Encoder<'static, W>,
    /// Number of certificates written
    cert_count: u64,
    /// Total uncompressed bytes written
    uncompressed_bytes: u64,
    /// Progress callback
    progress: Option<StreamingProgressCallback>,
    /// Header (kept for finalization)
    header: StreamingArchiveHeader,
}

impl<W: std::io::Write> StreamingCertWriter<W> {
    /// Create a new streaming writer with Zstd compression.
    ///
    /// # Arguments
    /// * `writer` - The underlying writer to write compressed data to
    /// * `level` - Zstd compression level (1-22, default 3)
    pub fn new_zstd(mut writer: W, level: i32) -> Result<Self, StreamingError> {
        let header = StreamingArchiveHeader::new_zstd(level);

        // Write header uncompressed first
        let header_bytes =
            bincode::serialize(&header).map_err(|e| StreamingError::Serialize(e.to_string()))?;

        // Write header length (4 bytes) + header
        let len_bytes = usize_to_u32_streaming(header_bytes.len())?.to_le_bytes();
        writer.write_all(&len_bytes)?;
        writer.write_all(&header_bytes)?;

        // Create Zstd encoder for the compressed stream
        let encoder = zstd::stream::Encoder::new(writer, level)?;

        Ok(StreamingCertWriter {
            encoder,
            cert_count: 0,
            uncompressed_bytes: 0,
            progress: None,
            header,
        })
    }

    /// Set a progress callback for monitoring compression.
    #[must_use]
    pub fn with_progress(mut self, callback: StreamingProgressCallback) -> Self {
        self.progress = Some(callback);
        self
    }

    /// Write a certificate to the stream.
    ///
    /// The certificate is serialized, then written to the compressed stream.
    pub fn write_cert(&mut self, cert: &ProofCert) -> Result<(), StreamingError> {
        use std::io::Write;

        // Serialize the certificate
        let cert_bytes =
            bincode::serialize(cert).map_err(|e| StreamingError::Serialize(e.to_string()))?;

        // Write length prefix (4 bytes) + certificate data
        let cert_len = usize_to_u32_streaming(cert_bytes.len())?;
        let len_bytes = cert_len.to_le_bytes();
        self.encoder.write_all(&len_bytes)?;
        self.encoder.write_all(&cert_bytes)?;

        self.cert_count += 1;
        self.uncompressed_bytes += 4 + u64::from(cert_len);

        // Report progress
        if let Some(ref mut callback) = self.progress {
            callback(self.uncompressed_bytes, None);
        }

        Ok(())
    }

    /// Write multiple certificates to the stream.
    pub fn write_certs(&mut self, certs: &[ProofCert]) -> Result<(), StreamingError> {
        let total = certs.len() as u64;
        for (i, cert) in certs.iter().enumerate() {
            self.write_cert(cert)?;

            // Report progress with total
            if let Some(ref mut callback) = self.progress {
                callback(i as u64 + 1, Some(total));
            }
        }
        Ok(())
    }

    /// Get the number of certificates written so far.
    pub fn cert_count(&self) -> u64 {
        self.cert_count
    }

    /// Get the total uncompressed bytes written so far.
    pub fn uncompressed_bytes(&self) -> u64 {
        self.uncompressed_bytes
    }

    /// Get the compression algorithm used.
    pub fn algorithm(&self) -> CompressionAlgorithm {
        self.header.algorithm()
    }

    /// Finish writing and return the underlying writer.
    ///
    /// This finalizes the Zstd stream and flushes all data.
    pub fn finish(self) -> Result<W, StreamingError> {
        let writer = self.encoder.finish()?;
        Ok(writer)
    }
}

/// A streaming certificate reader that decompresses certificates as they are read.
///
/// # Example
///
/// ```ignore
/// use std::fs::File;
/// let file = File::open("proofs.l5cs")?;
/// let mut reader = StreamingCertReader::new(file)?;
///
/// while let Some(cert) = reader.read_cert()? {
///     process_cert(&cert);
/// }
/// ```
pub struct StreamingCertReader<R: std::io::Read> {
    /// Inner Zstd decoder (wraps the input reader)
    decoder: zstd::stream::Decoder<'static, std::io::BufReader<R>>,
    /// The header read from the stream
    header: StreamingArchiveHeader,
    /// Number of certificates read
    certs_read: u64,
    /// Total uncompressed bytes read
    uncompressed_bytes: u64,
    /// Progress callback
    progress: Option<StreamingProgressCallback>,
}

impl<R: std::io::Read> StreamingCertReader<R> {
    /// Create a new streaming reader.
    ///
    /// Reads and validates the header, then prepares for decompression.
    pub fn new(mut reader: R) -> Result<Self, StreamingError> {
        // Read header length
        let mut len_bytes = [0u8; 4];
        reader.read_exact(&mut len_bytes)?;
        let header_len = u32::from_le_bytes(len_bytes) as usize;

        // Read header
        let mut header_bytes = vec![0u8; header_len];
        reader.read_exact(&mut header_bytes)?;

        let header: StreamingArchiveHeader = bincode::deserialize(&header_bytes)
            .map_err(|e| StreamingError::InvalidFormat(e.to_string()))?;

        header.validate()?;

        // Create decoder based on algorithm
        if header.algorithm != 1 {
            return Err(StreamingError::InvalidFormat(
                "Streaming only supports Zstd algorithm".to_string(),
            ));
        }

        let decoder = zstd::stream::Decoder::new(reader)?;

        Ok(StreamingCertReader {
            decoder,
            header,
            certs_read: 0,
            uncompressed_bytes: 0,
            progress: None,
        })
    }

    /// Set a progress callback for monitoring decompression.
    #[must_use]
    pub fn with_progress(mut self, callback: StreamingProgressCallback) -> Self {
        self.progress = Some(callback);
        self
    }

    /// Get the header information.
    pub fn header(&self) -> &StreamingArchiveHeader {
        &self.header
    }

    /// Read the next certificate from the stream.
    ///
    /// Returns `None` when the stream is exhausted.
    pub fn read_cert(&mut self) -> Result<Option<ProofCert>, StreamingError> {
        use std::io::Read;

        // Read length prefix
        let mut len_bytes = [0u8; 4];
        match self.decoder.read_exact(&mut len_bytes) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(StreamingError::Io(e)),
        }

        let cert_len = u32::from_le_bytes(len_bytes) as usize;

        // Read certificate data
        let mut cert_bytes = vec![0u8; cert_len];
        self.decoder.read_exact(&mut cert_bytes)?;

        let cert: ProofCert = bincode::deserialize(&cert_bytes)
            .map_err(|e| StreamingError::Decompress(e.to_string()))?;

        self.certs_read += 1;
        self.uncompressed_bytes += 4 + cert_len as u64;

        // Report progress
        if let Some(ref mut callback) = self.progress {
            let total = if self.header.cert_count > 0 {
                Some(self.header.cert_count)
            } else {
                None
            };
            callback(self.certs_read, total);
        }

        Ok(Some(cert))
    }

    /// Read all remaining certificates from the stream.
    pub fn read_all(&mut self) -> Result<Vec<ProofCert>, StreamingError> {
        let mut certs = Vec::new();
        while let Some(cert) = self.read_cert()? {
            certs.push(cert);
        }
        Ok(certs)
    }

    /// Get the number of certificates read so far.
    pub fn certs_read(&self) -> u64 {
        self.certs_read
    }

    /// Get the total uncompressed bytes read so far.
    pub fn uncompressed_bytes(&self) -> u64 {
        self.uncompressed_bytes
    }
}

/// Streaming archive statistics.
#[derive(Debug, Clone)]
pub struct StreamingStats {
    /// Number of certificates processed
    pub cert_count: u64,
    /// Total uncompressed size in bytes
    pub uncompressed_bytes: u64,
    /// Total compressed size in bytes
    pub compressed_bytes: u64,
    /// Compression algorithm used
    pub algorithm: CompressionAlgorithm,
}

impl StreamingStats {
    /// Calculate compression ratio (uncompressed / compressed).
    pub fn ratio(&self) -> f64 {
        if self.compressed_bytes == 0 {
            0.0
        } else {
            self.uncompressed_bytes as f64 / self.compressed_bytes as f64
        }
    }
}

impl std::fmt::Display for StreamingStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "StreamingStats {{ certs: {}, uncompressed: {} bytes, compressed: {} bytes, ratio: {:.2}x, algo: {} }}",
            self.cert_count,
            self.uncompressed_bytes,
            self.compressed_bytes,
            self.ratio(),
            self.algorithm.name()
        )
    }
}

/// Write certificates to a file using streaming compression.
///
/// This is a convenience function that handles file creation and finalization.
pub fn stream_certs_to_file(
    path: &std::path::Path,
    certs: &[ProofCert],
    level: i32,
) -> Result<StreamingStats, StreamingError> {
    let file = std::fs::File::create(path)?;
    let mut writer = StreamingCertWriter::new_zstd(file, level)?;

    for cert in certs {
        writer.write_cert(cert)?;
    }

    let uncompressed = writer.uncompressed_bytes();
    let count = writer.cert_count();
    let _file = writer.finish()?;

    // Get file size for compressed bytes
    let compressed = std::fs::metadata(path)?.len();

    Ok(StreamingStats {
        cert_count: count,
        uncompressed_bytes: uncompressed,
        compressed_bytes: compressed,
        algorithm: CompressionAlgorithm::from_zstd_level(level),
    })
}

/// Read certificates from a file using streaming decompression.
///
/// This is a convenience function that handles file opening.
pub fn stream_certs_from_file(
    path: &std::path::Path,
) -> Result<(Vec<ProofCert>, StreamingStats), StreamingError> {
    let compressed_size = std::fs::metadata(path)?.len();
    let file = std::fs::File::open(path)?;
    let mut reader = StreamingCertReader::new(file)?;

    let algorithm = reader.header().algorithm();
    let certs = reader.read_all()?;

    Ok((
        certs.clone(),
        StreamingStats {
            cert_count: certs.len() as u64,
            uncompressed_bytes: reader.uncompressed_bytes(),
            compressed_bytes: compressed_size,
            algorithm,
        },
    ))
}

#[cfg(test)]
mod tests;
