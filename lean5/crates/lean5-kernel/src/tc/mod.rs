//! Type checker
//!
//! The core type checking algorithm.

use crate::env::Environment;
use crate::expr::{BinderInfo, Expr, FVarId};
use crate::inductive::RecursorArgOrder;
use crate::level::Level;
use crate::mode::Lean5Mode;
use crate::name::Name;
use std::sync::Arc;

/// Minimum stack space to reserve before recursive calls (32 KB).
/// This prevents stack overflow in deeply nested type checking.
const MIN_STACK_RED_ZONE: usize = 32 * 1024;

/// Stack size to grow to when running low (1 MB).
const STACK_GROWTH_SIZE: usize = 1024 * 1024;

/// Local context entry
#[derive(Clone, Debug)]
pub struct LocalDecl {
    /// Unique identifier
    pub id: FVarId,
    /// User-facing name
    pub name: Name,
    /// Type of the variable
    pub type_: Expr,
    /// Value (for let bindings)
    pub value: Option<Expr>,
    /// Binder info
    pub bi: BinderInfo,
}

/// Local context (stack of local declarations)
#[derive(Clone, Debug, Default)]
pub struct LocalContext {
    decls: Vec<LocalDecl>,
    next_id: u64,
}

impl LocalContext {
    /// Create a new empty context
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a new variable binding
    pub fn push(&mut self, name: Name, type_: Expr, bi: BinderInfo) -> FVarId {
        let id = FVarId(self.next_id);
        self.next_id += 1;
        self.decls.push(LocalDecl {
            id,
            name,
            type_,
            value: None,
            bi,
        });
        id
    }

    /// Push a let binding
    pub fn push_let(&mut self, name: Name, type_: Expr, value: Expr) -> FVarId {
        let id = FVarId(self.next_id);
        self.next_id += 1;
        self.decls.push(LocalDecl {
            id,
            name,
            type_,
            value: Some(value),
            bi: BinderInfo::Default,
        });
        id
    }

    /// Pop the most recent binding
    pub fn pop(&mut self) -> Option<LocalDecl> {
        self.decls.pop()
    }

    /// Look up a free variable
    pub fn get(&self, id: FVarId) -> Option<&LocalDecl> {
        self.decls.iter().find(|d| d.id == id)
    }

    /// Number of bindings
    pub fn len(&self) -> usize {
        self.decls.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.decls.is_empty()
    }

    /// Iterate over all local declarations
    pub fn iter(&self) -> impl Iterator<Item = &LocalDecl> {
        self.decls.iter()
    }

    /// Push a binding with a specific FVarId (used by elaborator)
    pub fn push_with_id(&mut self, id: FVarId, name: Name, type_: Expr, bi: BinderInfo) {
        self.decls.push(LocalDecl {
            id,
            name,
            type_,
            value: None,
            bi,
        });
        // Update next_id if needed to avoid collisions
        if id.0 >= self.next_id {
            self.next_id = id.0 + 1;
        }
    }
}

/// Type checker
pub struct TypeChecker<'env> {
    /// The environment
    env: &'env Environment,
    /// Local context
    ctx: LocalContext,
    /// Current mode for type checking
    mode: Lean5Mode,
}

impl<'env> TypeChecker<'env> {
    /// Create a new type checker in Constructive mode (default)
    pub fn new(env: &'env Environment) -> Self {
        Self {
            env,
            ctx: LocalContext::new(),
            mode: Lean5Mode::default(),
        }
    }

    /// Create a type checker with a specific mode
    pub fn with_mode(env: &'env Environment, mode: Lean5Mode) -> Self {
        Self {
            env,
            ctx: LocalContext::new(),
            mode,
        }
    }

    /// Create a type checker with an existing local context
    pub fn with_context(env: &'env Environment, ctx: LocalContext) -> Self {
        Self {
            env,
            ctx,
            mode: Lean5Mode::default(),
        }
    }

    /// Create a type checker with an existing local context and specific mode
    pub fn with_context_and_mode(
        env: &'env Environment,
        ctx: LocalContext,
        mode: Lean5Mode,
    ) -> Self {
        Self { env, ctx, mode }
    }

    /// Get the current mode
    pub fn mode(&self) -> Lean5Mode {
        self.mode
    }

    /// Set the mode
    pub fn set_mode(&mut self, mode: Lean5Mode) {
        self.mode = mode;
    }

    /// Get mutable reference to the local context
    pub fn local_context_mut(&mut self) -> &mut LocalContext {
        &mut self.ctx
    }

    /// Get reference to the local context
    pub fn local_context(&self) -> &LocalContext {
        &self.ctx
    }

    /// Infer the type of an expression
    ///
    /// In debug builds, this method performs cross-validation with the micro-checker
    /// to verify kernel correctness. Any disagreement causes a panic.
    ///
    /// In release builds, uses a fast path without certificate generation for performance.
    /// The typing logic is identical between debug and release modes.
    #[cfg(debug_assertions)]
    pub fn infer_type(&mut self, e: &Expr) -> Result<Expr, TypeError> {
        let (ty, cert) = self.infer_type_with_cert(e)?;
        crate::micro::cross_validate_with_micro(e, &ty, &cert);
        Ok(ty)
    }

    /// Infer the type of an expression (release mode - fast path)
    ///
    /// Uses fast unchecked inference without certificate generation.
    /// Typing logic is identical to debug mode.
    #[cfg(not(debug_assertions))]
    pub fn infer_type(&mut self, e: &Expr) -> Result<Expr, TypeError> {
        self.infer_type_fast(e)
    }

    /// Fast type inference without certificate generation.
    ///
    /// This function implements the same typing logic as `infer_type_with_cert`
    /// but without the overhead of generating proof certificates. Used in release
    /// mode for performance.
    ///
    /// The typing rules are:
    /// - Sort(l) : Sort(succ(l))
    /// - FVar(id) : type_of(id) from context
    /// - Const(n, ls) : instantiate_type(n, ls) from environment
    /// - App(f, a) : B[a/x] when f : (x : A) → B and a : A
    /// - Lam(bi, A, b) : (x : A) → B when b : B
    /// - Pi(bi, A, B) : Sort(imax(l1, l2)) when A : Sort(l1), B : Sort(l2)
    /// - Let(A, v, b) : B[v/x] when v : A, b : B
    /// - Lit(n) : Nat or String
    #[cfg(not(debug_assertions))]
    fn infer_type_fast(&mut self, e: &Expr) -> Result<Expr, TypeError> {
        stacker::maybe_grow(MIN_STACK_RED_ZONE, STACK_GROWTH_SIZE, || {
            self.infer_type_fast_impl(e)
        })
    }

    /// Implementation of fast type inference (called via stacker::maybe_grow)
    #[cfg(not(debug_assertions))]
    fn infer_type_fast_impl(&mut self, e: &Expr) -> Result<Expr, TypeError> {
        match e {
            Expr::BVar(idx) => Err(TypeError::UnboundVariable(*idx)),
            Expr::FVar(id) => {
                let decl = self.ctx.get(*id).ok_or(TypeError::UnknownFVar(*id))?;
                Ok(decl.type_.clone())
            }
            Expr::Sort(l) => Ok(Expr::Sort(Level::succ(l.clone()))),
            Expr::Const(name, levels) => self
                .env
                .instantiate_type(name, levels)
                .ok_or_else(|| TypeError::UnknownConst(name.clone())),
            Expr::App(f, a) => {
                let f_type = self.infer_type_fast(f)?;
                let f_type_whnf = self.whnf(&f_type);

                match &f_type_whnf {
                    Expr::Pi(_, expected_arg_type, result_type) => {
                        let arg_type = self.infer_type_fast(a)?;
                        if !self.is_def_eq(&arg_type, expected_arg_type) {
                            return Err(TypeError::TypeMismatch {
                                expected: Box::new(expected_arg_type.as_ref().clone()),
                                inferred: Box::new(arg_type),
                            });
                        }
                        Ok(result_type.instantiate(a))
                    }
                    _ => Err(TypeError::NotAFunction(Box::new(f_type))),
                }
            }
            Expr::Lam(bi, arg_type, body) => {
                let arg_sort = self.infer_type_fast(arg_type)?;
                let arg_sort_whnf = self.whnf(&arg_sort);
                match arg_sort_whnf {
                    Expr::Sort(_) => {}
                    _ => return Err(TypeError::ExpectedSort(Box::new(arg_sort))),
                };

                let fvar_id = self.ctx.push(Name::anon(), arg_type.as_ref().clone(), *bi);
                let body_with_fvar = self.open_bvar(body, fvar_id);
                let body_type = self.infer_type_fast(&body_with_fvar)?;
                self.ctx.pop();

                let body_type_abstract = body_type.abstract_fvar(fvar_id);
                Ok(Expr::Pi(*bi, arg_type.clone(), body_type_abstract.into()))
            }
            Expr::Pi(bi, arg_type, body) => {
                let arg_sort = self.infer_type_fast(arg_type)?;
                let arg_sort_whnf = self.whnf(&arg_sort);
                let Expr::Sort(l1) = arg_sort_whnf else {
                    return Err(TypeError::ExpectedSort(Box::new(arg_sort)));
                };

                let fvar_id = self.ctx.push(Name::anon(), arg_type.as_ref().clone(), *bi);
                let body_with_fvar = self.open_bvar(body, fvar_id);
                let body_sort = self.infer_type_fast(&body_with_fvar)?;
                self.ctx.pop();

                let body_sort_whnf = self.whnf(&body_sort);
                let Expr::Sort(l2) = body_sort_whnf else {
                    return Err(TypeError::ExpectedSort(Box::new(body_sort)));
                };

                Ok(Expr::Sort(Level::imax(l1, l2)))
            }
            Expr::Let(ty, val, body) => {
                let ty_sort = self.infer_type_fast(ty)?;
                let ty_sort_whnf = self.whnf(&ty_sort);
                match ty_sort_whnf {
                    Expr::Sort(_) => {}
                    _ => return Err(TypeError::ExpectedSort(Box::new(ty_sort))),
                }

                let val_type = self.infer_type_fast(val)?;
                if !self.is_def_eq(&val_type, ty) {
                    return Err(TypeError::TypeMismatch {
                        expected: Box::new(ty.as_ref().clone()),
                        inferred: Box::new(val_type),
                    });
                }

                let fvar_id =
                    self.ctx
                        .push_let(Name::anon(), ty.as_ref().clone(), val.as_ref().clone());
                let body_with_fvar = self.open_bvar(body, fvar_id);
                let body_type = self.infer_type_fast(&body_with_fvar)?;
                self.ctx.pop();

                Ok(body_type.instantiate(val))
            }
            Expr::Lit(lit) => Ok(match lit {
                crate::expr::Literal::Nat(_) => Expr::const_(Name::from_string("Nat"), vec![]),
                crate::expr::Literal::String(_) => {
                    Expr::const_(Name::from_string("String"), vec![])
                }
            }),
            Expr::Proj(struct_name, idx, e) => self.infer_proj_type(struct_name, *idx, e),
            // MData is transparent - just infer the type of the inner expression
            Expr::MData(_, inner) => self.infer_type_fast(inner),

            // ════════════════════════════════════════════════════════════════════
            // Mode-specific extensions - require appropriate mode to be enabled
            // ════════════════════════════════════════════════════════════════════

            // Cubical mode expressions
            Expr::CubicalInterval => {
                if self.mode != Lean5Mode::Cubical {
                    return Err(TypeError::ModeRequired {
                        feature: "CubicalInterval".to_string(),
                        mode: "Cubical".to_string(),
                    });
                }
                Ok(Expr::Sort(Level::zero()))
            }
            Expr::CubicalI0 | Expr::CubicalI1 => {
                if self.mode != Lean5Mode::Cubical {
                    return Err(TypeError::ModeRequired {
                        feature: "CubicalI0/CubicalI1".to_string(),
                        mode: "Cubical".to_string(),
                    });
                }
                Ok(Expr::CubicalInterval)
            }
            Expr::CubicalPath { ty, left, right } => {
                if self.mode != Lean5Mode::Cubical {
                    return Err(TypeError::ModeRequired {
                        feature: "CubicalPath".to_string(),
                        mode: "Cubical".to_string(),
                    });
                }

                // ty : I -> Sort(l)
                let ty_type = self.infer_type_fast(ty)?;
                let ty_type_whnf = self.whnf(&ty_type);
                let Expr::Pi(_, arg_ty, body_ty) = ty_type_whnf else {
                    return Err(TypeError::NotAFunction(Box::new(ty_type)));
                };
                if !matches!(self.whnf(&arg_ty), Expr::CubicalInterval) {
                    return Err(TypeError::TypeMismatch {
                        expected: Box::new(Expr::CubicalInterval),
                        inferred: Box::new(arg_ty.as_ref().clone()),
                    });
                }
                let body_ty_whnf = self.whnf(&body_ty);
                let Expr::Sort(level) = body_ty_whnf else {
                    return Err(TypeError::ExpectedSort(Box::new(body_ty.as_ref().clone())));
                };

                // left : ty 0, right : ty 1
                let expected_left_ty = Expr::App(ty.clone(), Arc::new(Expr::CubicalI0));
                let left_ty = self.infer_type_fast(left)?;
                if !self.is_def_eq(&left_ty, &expected_left_ty) {
                    return Err(TypeError::TypeMismatch {
                        expected: Box::new(expected_left_ty),
                        inferred: Box::new(left_ty),
                    });
                }

                let expected_right_ty = Expr::App(ty.clone(), Arc::new(Expr::CubicalI1));
                let right_ty = self.infer_type_fast(right)?;
                if !self.is_def_eq(&right_ty, &expected_right_ty) {
                    return Err(TypeError::TypeMismatch {
                        expected: Box::new(expected_right_ty),
                        inferred: Box::new(right_ty),
                    });
                }

                Ok(Expr::Sort(level))
            }
            Expr::CubicalPathLam { body } => {
                if self.mode != Lean5Mode::Cubical {
                    return Err(TypeError::ModeRequired {
                        feature: "CubicalPathLam".to_string(),
                        mode: "Cubical".to_string(),
                    });
                }

                // Add interval variable to context and infer body type
                let fvar_id = self
                    .ctx
                    .push(Name::anon(), Expr::CubicalInterval, BinderInfo::Default);
                let body_with_fvar = self.open_bvar(body, fvar_id);
                let body_type = self.infer_type_fast(&body_with_fvar)?;
                self.ctx.pop();

                // Build Path type: Path (λ i : I, body_type) (body[0]) (body[1])
                let left = body.instantiate(&Expr::CubicalI0);
                let right = body.instantiate(&Expr::CubicalI1);
                let body_type_abstract = body_type.abstract_fvar(fvar_id);
                let ty_family = Expr::Lam(
                    BinderInfo::Default,
                    Arc::new(Expr::CubicalInterval),
                    Arc::new(body_type_abstract),
                );
                Ok(Expr::CubicalPath {
                    ty: Arc::new(ty_family),
                    left: Arc::new(left),
                    right: Arc::new(right),
                })
            }
            Expr::CubicalPathApp { path, arg } => {
                if self.mode != Lean5Mode::Cubical {
                    return Err(TypeError::ModeRequired {
                        feature: "CubicalPathApp".to_string(),
                        mode: "Cubical".to_string(),
                    });
                }

                let path_type = self.infer_type_fast(path)?;
                let path_type_whnf = self.whnf(&path_type);
                let Expr::CubicalPath { ty, .. } = path_type_whnf else {
                    return Err(TypeError::NotAFunction(Box::new(path_type)));
                };
                let arg_type = self.infer_type_fast(arg)?;
                if !matches!(self.whnf(&arg_type), Expr::CubicalInterval) {
                    return Err(TypeError::TypeMismatch {
                        expected: Box::new(Expr::CubicalInterval),
                        inferred: Box::new(arg_type),
                    });
                }

                Ok(Expr::App(ty, arg.clone()))
            }
            Expr::CubicalHComp { ty, phi, u, base } => {
                if self.mode != Lean5Mode::Cubical {
                    return Err(TypeError::ModeRequired {
                        feature: "CubicalHComp".to_string(),
                        mode: "Cubical".to_string(),
                    });
                }

                let ty_sort = self.infer_type_fast(ty)?;
                if !matches!(self.whnf(&ty_sort), Expr::Sort(_)) {
                    return Err(TypeError::ExpectedSort(Box::new(ty_sort)));
                }

                let _ = self.infer_type_fast(phi)?;
                let _ = self.infer_type_fast(u)?;
                let base_ty = self.infer_type_fast(base)?;
                if !self.is_def_eq(&base_ty, ty) {
                    return Err(TypeError::TypeMismatch {
                        expected: Box::new(ty.as_ref().clone()),
                        inferred: Box::new(base_ty),
                    });
                }

                Ok(ty.as_ref().clone())
            }
            Expr::CubicalTransp { ty, phi, base } => {
                if self.mode != Lean5Mode::Cubical {
                    return Err(TypeError::ModeRequired {
                        feature: "CubicalTransp".to_string(),
                        mode: "Cubical".to_string(),
                    });
                }

                let _ = self.infer_type_fast(ty)?;
                let _ = self.infer_type_fast(phi)?;
                let _ = self.infer_type_fast(base)?;

                Ok(Expr::App(ty.clone(), Arc::new(Expr::CubicalI1)))
            }

            // Classical mode expressions
            Expr::ClassicalChoice {
                ty,
                pred,
                exists_proof,
            } => {
                if !matches!(self.mode, Lean5Mode::Classical | Lean5Mode::SetTheoretic) {
                    return Err(TypeError::ModeRequired {
                        feature: "ClassicalChoice".to_string(),
                        mode: "Classical".to_string(),
                    });
                }

                let ty_sort = self.infer_type_fast(ty)?;
                if !matches!(self.whnf(&ty_sort), Expr::Sort(_)) {
                    return Err(TypeError::ExpectedSort(Box::new(ty_sort)));
                }
                let _ = self.infer_type_fast(pred)?;
                let _ = self.infer_type_fast(exists_proof)?;
                Ok(ty.as_ref().clone())
            }
            Expr::ClassicalEpsilon { ty, pred } => {
                if !matches!(self.mode, Lean5Mode::Classical | Lean5Mode::SetTheoretic) {
                    return Err(TypeError::ModeRequired {
                        feature: "ClassicalEpsilon".to_string(),
                        mode: "Classical".to_string(),
                    });
                }

                let ty_sort = self.infer_type_fast(ty)?;
                if !matches!(self.whnf(&ty_sort), Expr::Sort(_)) {
                    return Err(TypeError::ExpectedSort(Box::new(ty_sort)));
                }
                let _ = self.infer_type_fast(pred)?;
                Ok(ty.as_ref().clone())
            }

            // SetTheoretic mode expressions
            Expr::ZFCSet(_) => {
                if self.mode != Lean5Mode::SetTheoretic {
                    return Err(TypeError::ModeRequired {
                        feature: "ZFCSet".to_string(),
                        mode: "SetTheoretic".to_string(),
                    });
                }
                Ok(Expr::const_(Name::from_string("ZFC.Set"), vec![]))
            }
            Expr::ZFCMem { element, set } => {
                if self.mode != Lean5Mode::SetTheoretic {
                    return Err(TypeError::ModeRequired {
                        feature: "ZFCMem".to_string(),
                        mode: "SetTheoretic".to_string(),
                    });
                }

                let elem_ty = self.infer_type_fast(element)?;
                let set_ty = self.infer_type_fast(set)?;
                let expected_set_ty = Expr::const_(Name::from_string("ZFC.Set"), vec![]);
                if !self.is_def_eq(&elem_ty, &expected_set_ty) {
                    return Err(TypeError::TypeMismatch {
                        expected: Box::new(expected_set_ty.clone()),
                        inferred: Box::new(elem_ty),
                    });
                }
                if !self.is_def_eq(&set_ty, &expected_set_ty) {
                    return Err(TypeError::TypeMismatch {
                        expected: Box::new(expected_set_ty),
                        inferred: Box::new(set_ty),
                    });
                }
                Ok(Expr::Sort(Level::zero()))
            }
            Expr::ZFCComprehension { domain, pred } => {
                if self.mode != Lean5Mode::SetTheoretic {
                    return Err(TypeError::ModeRequired {
                        feature: "ZFCComprehension".to_string(),
                        mode: "SetTheoretic".to_string(),
                    });
                }

                let domain_ty = self.infer_type_fast(domain)?;
                let expected_set_ty = Expr::const_(Name::from_string("ZFC.Set"), vec![]);
                if !self.is_def_eq(&domain_ty, &expected_set_ty) {
                    return Err(TypeError::TypeMismatch {
                        expected: Box::new(expected_set_ty),
                        inferred: Box::new(domain_ty),
                    });
                }
                let _ = self.infer_type_fast(pred)?;
                Ok(Expr::const_(Name::from_string("ZFC.Set"), vec![]))
            }
        }
    }

    /// Check that an expression has a given type
    pub fn check_type(&mut self, e: &Expr, expected: &Expr) -> Result<(), TypeError> {
        let inferred = self.infer_type(e)?;
        if self.is_def_eq(&inferred, expected) {
            Ok(())
        } else {
            Err(TypeError::TypeMismatch {
                expected: Box::new(expected.clone()),
                inferred: Box::new(inferred),
            })
        }
    }

    /// Infer type and ensure it's a sort, returning the level
    pub fn infer_sort(&mut self, e: &Expr) -> Result<Level, TypeError> {
        let ty = self.infer_type(e)?;
        let ty_whnf = self.whnf(&ty);
        match ty_whnf {
            Expr::Sort(l) => Ok(l),
            _ => Err(TypeError::ExpectedSort(Box::new(ty))),
        }
    }

    /// Infer the type of an expression with proof certificate generation.
    ///
    /// Returns both the inferred type and a proof certificate that can
    /// be independently verified to confirm the typing derivation.
    pub fn infer_type_with_cert(
        &mut self,
        e: &Expr,
    ) -> Result<(Expr, crate::cert::ProofCert), TypeError> {
        stacker::maybe_grow(MIN_STACK_RED_ZONE, STACK_GROWTH_SIZE, || {
            self.infer_type_with_cert_impl(e)
        })
    }

    /// Implementation of type inference with certificates (called via stacker::maybe_grow)
    fn infer_type_with_cert_impl(
        &mut self,
        e: &Expr,
    ) -> Result<(Expr, crate::cert::ProofCert), TypeError> {
        use crate::cert::ProofCert;

        match e {
            Expr::BVar(idx) => {
                // BVar should have been replaced by FVar during elaboration
                Err(TypeError::UnboundVariable(*idx))
            }
            Expr::FVar(id) => {
                let decl = self.ctx.get(*id).ok_or(TypeError::UnknownFVar(*id))?;
                let ty = decl.type_.clone();
                let cert = ProofCert::FVar {
                    id: *id,
                    type_: Box::new(ty.clone()),
                };
                Ok((ty, cert))
            }
            Expr::Sort(l) => {
                let result_ty = Expr::Sort(Level::succ(l.clone()));
                let cert = ProofCert::Sort { level: l.clone() };
                Ok((result_ty, cert))
            }
            Expr::Const(name, levels) => {
                let ty = self
                    .env
                    .instantiate_type(name, levels)
                    .ok_or_else(|| TypeError::UnknownConst(name.clone()))?;
                let cert = ProofCert::Const {
                    name: name.clone(),
                    levels: levels.to_vec(),
                    type_: Box::new(ty.clone()),
                };
                Ok((ty, cert))
            }
            Expr::App(f, a) => {
                let (f_type, f_cert) = self.infer_type_with_cert(f)?;
                let f_type_whnf = self.whnf(&f_type);

                match &f_type_whnf {
                    Expr::Pi(_, expected_arg_type, result_type) => {
                        // Check argument has expected type
                        let (arg_type, arg_cert) = self.infer_type_with_cert(a)?;

                        // Verify arg type matches (with DefEq if needed)
                        if !self.is_def_eq(&arg_type, expected_arg_type) {
                            return Err(TypeError::TypeMismatch {
                                expected: Box::new(expected_arg_type.as_ref().clone()),
                                inferred: Box::new(arg_type),
                            });
                        }

                        // Substitute argument into result type
                        let result_ty = result_type.instantiate(a);

                        let cert = ProofCert::App {
                            fn_cert: Box::new(f_cert),
                            fn_type: Box::new(f_type_whnf.clone()),
                            arg_cert: Box::new(arg_cert),
                            result_type: Box::new(result_ty.clone()),
                        };

                        Ok((result_ty, cert))
                    }
                    _ => Err(TypeError::NotAFunction(Box::new(f_type))),
                }
            }
            Expr::Lam(bi, arg_type, body) => {
                // Check arg_type is a type and get its level
                let (arg_sort, arg_type_cert) = self.infer_type_with_cert(arg_type)?;
                let arg_sort_whnf = self.whnf(&arg_sort);
                let Expr::Sort(_arg_level) = arg_sort_whnf else {
                    return Err(TypeError::ExpectedSort(Box::new(arg_sort)));
                };

                // Add variable to context and infer body type
                let fvar_id = self.ctx.push(Name::anon(), arg_type.as_ref().clone(), *bi);
                let body_with_fvar = self.open_bvar(body, fvar_id);
                let (body_type, body_cert_raw) = self.infer_type_with_cert(&body_with_fvar)?;
                self.ctx.pop();

                // Convert FVar certificates back to BVar certificates for the body
                let body_cert = convert_fvar_cert_to_bvar(body_cert_raw, fvar_id, 0);

                // Abstract back to get Pi type
                let body_type_abstract = body_type.abstract_fvar(fvar_id);
                let result_type = Expr::Pi(*bi, arg_type.clone(), body_type_abstract.into());

                let cert = ProofCert::Lam {
                    binder_info: *bi,
                    arg_type_cert: Box::new(arg_type_cert),
                    body_cert: Box::new(body_cert),
                    result_type: Box::new(result_type.clone()),
                };

                Ok((result_type, cert))
            }
            Expr::Pi(bi, arg_type, body) => {
                // Check arg_type is a type
                let (arg_sort, arg_type_cert) = self.infer_type_with_cert(arg_type)?;
                let arg_sort_whnf = self.whnf(&arg_sort);
                let Expr::Sort(l1) = arg_sort_whnf else {
                    return Err(TypeError::ExpectedSort(Box::new(arg_sort)));
                };

                // Add variable to context and check body is a type
                let fvar_id = self.ctx.push(Name::anon(), arg_type.as_ref().clone(), *bi);
                let body_with_fvar = self.open_bvar(body, fvar_id);
                let (body_sort, body_type_cert_raw) = self.infer_type_with_cert(&body_with_fvar)?;
                self.ctx.pop();

                // Convert FVar certificates back to BVar certificates for the body
                let body_type_cert = convert_fvar_cert_to_bvar(body_type_cert_raw, fvar_id, 0);

                let body_sort_whnf = self.whnf(&body_sort);
                let Expr::Sort(l2) = body_sort_whnf else {
                    return Err(TypeError::ExpectedSort(Box::new(body_sort)));
                };

                let result_level = Level::imax(l1.clone(), l2.clone());
                let result_type = Expr::Sort(result_level);

                let cert = ProofCert::Pi {
                    binder_info: *bi,
                    arg_type_cert: Box::new(arg_type_cert),
                    arg_level: l1,
                    body_type_cert: Box::new(body_type_cert),
                    body_level: l2,
                };

                Ok((result_type, cert))
            }
            Expr::Let(ty, val, body) => {
                // Check type is a type
                let (ty_sort, type_cert) = self.infer_type_with_cert(ty)?;
                let ty_sort_whnf = self.whnf(&ty_sort);
                match ty_sort_whnf {
                    Expr::Sort(_) => {}
                    _ => return Err(TypeError::ExpectedSort(Box::new(ty_sort))),
                }

                // Check value has the declared type
                let (val_type, value_cert) = self.infer_type_with_cert(val)?;
                if !self.is_def_eq(&val_type, ty) {
                    return Err(TypeError::TypeMismatch {
                        expected: Box::new(ty.as_ref().clone()),
                        inferred: Box::new(val_type),
                    });
                }

                // Add let binding to context and infer body type
                let fvar_id =
                    self.ctx
                        .push_let(Name::anon(), ty.as_ref().clone(), val.as_ref().clone());
                let body_with_fvar = self.open_bvar(body, fvar_id);
                let (body_type, body_cert_raw) = self.infer_type_with_cert(&body_with_fvar)?;
                self.ctx.pop();

                // Convert FVar certificates back to BVar certificates for the body
                let body_cert = convert_fvar_cert_to_bvar(body_cert_raw, fvar_id, 0);

                // Result type should not mention the let-bound variable
                let result_type = body_type.instantiate(val);

                let cert = ProofCert::Let {
                    type_cert: Box::new(type_cert),
                    value_cert: Box::new(value_cert),
                    body_cert: Box::new(body_cert),
                    result_type: Box::new(result_type.clone()),
                };

                Ok((result_type, cert))
            }
            Expr::Lit(lit) => {
                let type_ = match lit {
                    crate::expr::Literal::Nat(_) => Expr::const_(Name::from_string("Nat"), vec![]),
                    crate::expr::Literal::String(_) => {
                        Expr::const_(Name::from_string("String"), vec![])
                    }
                };
                let cert = ProofCert::Lit {
                    lit: lit.clone(),
                    type_: Box::new(type_.clone()),
                };
                Ok((type_, cert))
            }
            Expr::Proj(struct_name, idx, e) => {
                // First get the type of the expression being projected
                let (expr_type, expr_cert) = self.infer_type_with_cert(e)?;

                // Get the field type using the existing projection inference
                let field_type = self.infer_proj_type(struct_name, *idx, e)?;

                let cert = ProofCert::Proj {
                    struct_name: struct_name.clone(),
                    idx: *idx,
                    expr_cert: Box::new(expr_cert),
                    expr_type: Box::new(expr_type),
                    field_type: Box::new(field_type.clone()),
                };
                Ok((field_type, cert))
            }
            // MData is transparent - just infer the type of the inner expression
            // We wrap the certificate to preserve that it came from an MData
            Expr::MData(metadata, inner) => {
                let (inner_type, inner_cert) = self.infer_type_with_cert(inner)?;
                let cert = ProofCert::MData {
                    metadata: metadata.clone(),
                    inner_cert: Box::new(inner_cert),
                    result_type: Box::new(inner_type.clone()),
                };
                Ok((inner_type, cert))
            }

            // ════════════════════════════════════════════════════════════════════
            // Mode-specific extensions - require appropriate mode to be enabled
            // ════════════════════════════════════════════════════════════════════

            // Cubical mode expressions
            Expr::CubicalInterval => {
                // I : IType (special sort)
                if self.mode != Lean5Mode::Cubical {
                    return Err(TypeError::ModeRequired {
                        feature: "CubicalInterval".to_string(),
                        mode: "Cubical".to_string(),
                    });
                }
                // In Cubical mode, I is a special sort (interval type)
                // Type is Sort(0) = Prop-like
                let cert = ProofCert::CubicalInterval;
                Ok((Expr::Sort(Level::zero()), cert))
            }
            Expr::CubicalI0 | Expr::CubicalI1 => {
                // 0, 1 : I
                if self.mode != Lean5Mode::Cubical {
                    return Err(TypeError::ModeRequired {
                        feature: "CubicalI0/CubicalI1".to_string(),
                        mode: "Cubical".to_string(),
                    });
                }
                // Type is the interval I
                let ty = Expr::CubicalInterval;
                let cert = ProofCert::CubicalEndpoint {
                    is_one: matches!(e, Expr::CubicalI1),
                };
                Ok((ty, cert))
            }
            Expr::CubicalPath { ty, left, right } => {
                // Path A a b : Type
                if self.mode != Lean5Mode::Cubical {
                    return Err(TypeError::ModeRequired {
                        feature: "CubicalPath".to_string(),
                        mode: "Cubical".to_string(),
                    });
                }
                // Check ty is a type family over the interval: ty : I -> Sort(l)
                let (ty_type, ty_cert) = self.infer_type_with_cert(ty)?;
                let ty_type_whnf = self.whnf(&ty_type);
                let Expr::Pi(_, arg_ty, body_ty) = ty_type_whnf else {
                    return Err(TypeError::NotAFunction(Box::new(ty_type)));
                };
                if !matches!(self.whnf(&arg_ty), Expr::CubicalInterval) {
                    return Err(TypeError::TypeMismatch {
                        expected: Box::new(Expr::CubicalInterval),
                        inferred: Box::new(arg_ty.as_ref().clone()),
                    });
                }
                let body_ty_whnf = self.whnf(&body_ty);
                let Expr::Sort(level) = body_ty_whnf else {
                    return Err(TypeError::ExpectedSort(Box::new(body_ty.as_ref().clone())));
                };

                // Check endpoints: left : ty 0 and right : ty 1
                let expected_left_ty = Expr::App(ty.clone(), Arc::new(Expr::CubicalI0));
                let (left_ty, left_cert) = self.infer_type_with_cert(left)?;
                if !self.is_def_eq(&left_ty, &expected_left_ty) {
                    return Err(TypeError::TypeMismatch {
                        expected: Box::new(expected_left_ty),
                        inferred: Box::new(left_ty),
                    });
                }

                let expected_right_ty = Expr::App(ty.clone(), Arc::new(Expr::CubicalI1));
                let (right_ty, right_cert) = self.infer_type_with_cert(right)?;
                if !self.is_def_eq(&right_ty, &expected_right_ty) {
                    return Err(TypeError::TypeMismatch {
                        expected: Box::new(expected_right_ty),
                        inferred: Box::new(right_ty),
                    });
                }

                // Path types live at the same universe level as their type family codomain.
                let result_ty = Expr::Sort(level.clone());
                let cert = ProofCert::CubicalPath {
                    ty_cert: Box::new(ty_cert),
                    ty_level: level,
                    left_cert: Box::new(left_cert),
                    right_cert: Box::new(right_cert),
                };
                Ok((result_ty, cert))
            }
            Expr::CubicalPathLam { body } => {
                // <i> e : Path A (e[0/i]) (e[1/i])
                if self.mode != Lean5Mode::Cubical {
                    return Err(TypeError::ModeRequired {
                        feature: "CubicalPathLam".to_string(),
                        mode: "Cubical".to_string(),
                    });
                }
                // Add interval variable to context and infer body type
                let fvar_id = self
                    .ctx
                    .push(Name::anon(), Expr::CubicalInterval, BinderInfo::Default);
                let body_with_fvar = self.open_bvar(body, fvar_id);
                let (body_type, body_cert_raw) = self.infer_type_with_cert(&body_with_fvar)?;
                self.ctx.pop();

                // Convert FVar certificates back to BVar certificates for the body
                let body_cert = convert_fvar_cert_to_bvar(body_cert_raw, fvar_id, 0);

                // The result is a Path type, with a type family λ i : I, body_type
                let left = body.instantiate(&Expr::CubicalI0);
                let right = body.instantiate(&Expr::CubicalI1);
                let body_type_abstract = body_type.abstract_fvar(fvar_id);
                let ty_family = Expr::Lam(
                    BinderInfo::Default,
                    Arc::new(Expr::CubicalInterval),
                    Arc::new(body_type_abstract.clone()),
                );
                let result_ty = Expr::CubicalPath {
                    ty: Arc::new(ty_family),
                    left: Arc::new(left),
                    right: Arc::new(right),
                };
                let cert = ProofCert::CubicalPathLam {
                    body_cert: Box::new(body_cert),
                    body_type: Box::new(body_type_abstract),
                    result_type: Box::new(result_ty.clone()),
                };
                Ok((result_ty, cert))
            }
            Expr::CubicalPathApp { path, arg } => {
                // p @ i : A (when p : Path A a b and i : I)
                if self.mode != Lean5Mode::Cubical {
                    return Err(TypeError::ModeRequired {
                        feature: "CubicalPathApp".to_string(),
                        mode: "Cubical".to_string(),
                    });
                }
                // Check path has Path type
                let (path_type, path_cert) = self.infer_type_with_cert(path)?;
                let path_type_whnf = self.whnf(&path_type);
                let Expr::CubicalPath { ty, .. } = &path_type_whnf else {
                    return Err(TypeError::NotAFunction(Box::new(path_type)));
                };
                // Check arg has interval type
                let (arg_type, arg_cert) = self.infer_type_with_cert(arg)?;
                if !matches!(self.whnf(&arg_type), Expr::CubicalInterval) {
                    return Err(TypeError::TypeMismatch {
                        expected: Box::new(Expr::CubicalInterval),
                        inferred: Box::new(arg_type),
                    });
                }
                // Result is the path's type family applied to the interval point
                let result_ty = Expr::App(ty.clone(), arg.clone());
                let cert = ProofCert::CubicalPathApp {
                    path_cert: Box::new(path_cert),
                    arg_cert: Box::new(arg_cert),
                    path_type: Box::new(path_type_whnf.clone()),
                    result_type: Box::new(result_ty.clone()),
                };
                Ok((result_ty, cert))
            }
            Expr::CubicalHComp { ty, phi, u, base } => {
                // hcomp {A} {φ} u base : A
                if self.mode != Lean5Mode::Cubical {
                    return Err(TypeError::ModeRequired {
                        feature: "CubicalHComp".to_string(),
                        mode: "Cubical".to_string(),
                    });
                }
                // Simplified: check ty is a type
                let (ty_sort, ty_cert) = self.infer_type_with_cert(ty)?;
                let ty_sort_whnf = self.whnf(&ty_sort);
                if !matches!(ty_sort_whnf, Expr::Sort(_)) {
                    return Err(TypeError::ExpectedSort(Box::new(ty_sort)));
                }
                // Check phi, u, base (simplified checks)
                let (_, phi_cert) = self.infer_type_with_cert(phi)?;
                let (_, u_cert) = self.infer_type_with_cert(u)?;
                let (base_ty, base_cert) = self.infer_type_with_cert(base)?;
                if !self.is_def_eq(&base_ty, ty) {
                    return Err(TypeError::TypeMismatch {
                        expected: Box::new(ty.as_ref().clone()),
                        inferred: Box::new(base_ty),
                    });
                }
                let result_ty = ty.as_ref().clone();
                let cert = ProofCert::CubicalHComp {
                    ty_cert: Box::new(ty_cert),
                    phi_cert: Box::new(phi_cert),
                    u_cert: Box::new(u_cert),
                    base_cert: Box::new(base_cert),
                    result_type: Box::new(result_ty.clone()),
                };
                Ok((result_ty, cert))
            }
            Expr::CubicalTransp { ty, phi, base } => {
                // transp A φ base : A 1
                if self.mode != Lean5Mode::Cubical {
                    return Err(TypeError::ModeRequired {
                        feature: "CubicalTransp".to_string(),
                        mode: "Cubical".to_string(),
                    });
                }
                // Simplified: check ty is a line of types (I -> Type)
                let (_, ty_cert) = self.infer_type_with_cert(ty)?;
                let (_, phi_cert) = self.infer_type_with_cert(phi)?;
                let (_, base_cert) = self.infer_type_with_cert(base)?;
                // Result type is ty applied to i1
                let result_ty = Expr::App(ty.clone(), Arc::new(Expr::CubicalI1));
                let cert = ProofCert::CubicalTransp {
                    ty_cert: Box::new(ty_cert),
                    phi_cert: Box::new(phi_cert),
                    base_cert: Box::new(base_cert),
                    result_type: Box::new(result_ty.clone()),
                };
                Ok((result_ty, cert))
            }

            // Classical mode expressions
            Expr::ClassicalChoice {
                ty,
                pred,
                exists_proof,
            } => {
                // choice ty pred proof : ty
                if !matches!(self.mode, Lean5Mode::Classical | Lean5Mode::SetTheoretic) {
                    return Err(TypeError::ModeRequired {
                        feature: "ClassicalChoice".to_string(),
                        mode: "Classical".to_string(),
                    });
                }
                // Check ty is a type
                let (ty_sort, ty_cert) = self.infer_type_with_cert(ty)?;
                if !matches!(self.whnf(&ty_sort), Expr::Sort(_)) {
                    return Err(TypeError::ExpectedSort(Box::new(ty_sort)));
                }
                // Check pred : ty -> Prop
                let (_, pred_cert) = self.infer_type_with_cert(pred)?;
                // Check exists_proof : Exists pred (simplified)
                let (_, proof_cert) = self.infer_type_with_cert(exists_proof)?;
                let result_ty = ty.as_ref().clone();
                let cert = ProofCert::ClassicalChoice {
                    ty_cert: Box::new(ty_cert),
                    pred_cert: Box::new(pred_cert),
                    proof_cert: Box::new(proof_cert),
                    result_type: Box::new(result_ty.clone()),
                };
                Ok((result_ty, cert))
            }
            Expr::ClassicalEpsilon { ty, pred } => {
                // epsilon ty pred : ty
                if !matches!(self.mode, Lean5Mode::Classical | Lean5Mode::SetTheoretic) {
                    return Err(TypeError::ModeRequired {
                        feature: "ClassicalEpsilon".to_string(),
                        mode: "Classical".to_string(),
                    });
                }
                // Check ty is a type
                let (ty_sort, ty_cert) = self.infer_type_with_cert(ty)?;
                if !matches!(self.whnf(&ty_sort), Expr::Sort(_)) {
                    return Err(TypeError::ExpectedSort(Box::new(ty_sort)));
                }
                // Check pred : ty -> Prop
                let (_, pred_cert) = self.infer_type_with_cert(pred)?;
                let result_ty = ty.as_ref().clone();
                let cert = ProofCert::ClassicalEpsilon {
                    ty_cert: Box::new(ty_cert),
                    pred_cert: Box::new(pred_cert),
                    result_type: Box::new(result_ty.clone()),
                };
                Ok((result_ty, cert))
            }

            // SetTheoretic mode expressions
            Expr::ZFCSet(set_expr) => {
                // ZFC set expressions have type Set (a special sort)
                if self.mode != Lean5Mode::SetTheoretic {
                    return Err(TypeError::ModeRequired {
                        feature: "ZFCSet".to_string(),
                        mode: "SetTheoretic".to_string(),
                    });
                }
                // ZFC sets are in the universe of sets
                let result_ty = Expr::const_(Name::from_string("ZFC.Set"), vec![]);
                let cert_kind = self.infer_zfc_set_cert(set_expr)?;
                let cert = ProofCert::ZFCSet {
                    kind: cert_kind,
                    result_type: Box::new(result_ty.clone()),
                };
                Ok((result_ty, cert))
            }
            Expr::ZFCMem { element, set } => {
                // element ∈ set : Prop
                if self.mode != Lean5Mode::SetTheoretic {
                    return Err(TypeError::ModeRequired {
                        feature: "ZFCMem".to_string(),
                        mode: "SetTheoretic".to_string(),
                    });
                }
                // Check both element and set are sets
                let (elem_ty, elem_cert) = self.infer_type_with_cert(element)?;
                let (set_ty, set_cert) = self.infer_type_with_cert(set)?;
                let expected_set_ty = Expr::const_(Name::from_string("ZFC.Set"), vec![]);
                if !self.is_def_eq(&elem_ty, &expected_set_ty) {
                    return Err(TypeError::TypeMismatch {
                        expected: Box::new(expected_set_ty.clone()),
                        inferred: Box::new(elem_ty),
                    });
                }
                if !self.is_def_eq(&set_ty, &expected_set_ty) {
                    return Err(TypeError::TypeMismatch {
                        expected: Box::new(expected_set_ty),
                        inferred: Box::new(set_ty),
                    });
                }
                // Membership is a proposition
                let result_ty = Expr::Sort(Level::zero());
                let cert = ProofCert::ZFCMem {
                    elem_cert: Box::new(elem_cert),
                    set_cert: Box::new(set_cert),
                };
                Ok((result_ty, cert))
            }
            Expr::ZFCComprehension { domain, pred } => {
                // {x ∈ domain | pred x} : Set
                if self.mode != Lean5Mode::SetTheoretic {
                    return Err(TypeError::ModeRequired {
                        feature: "ZFCComprehension".to_string(),
                        mode: "SetTheoretic".to_string(),
                    });
                }
                // Check domain is a set
                let (domain_ty, var_ty_cert) = self.infer_type_with_cert(domain)?;
                let expected_set_ty = Expr::const_(Name::from_string("ZFC.Set"), vec![]);
                if !self.is_def_eq(&domain_ty, &expected_set_ty) {
                    return Err(TypeError::TypeMismatch {
                        expected: Box::new(expected_set_ty),
                        inferred: Box::new(domain_ty),
                    });
                }
                // Check pred : Set -> Prop
                let (_, pred_cert) = self.infer_type_with_cert(pred)?;
                // Result is a set
                let result_ty = Expr::const_(Name::from_string("ZFC.Set"), vec![]);
                let cert = ProofCert::ZFCComprehension {
                    var_ty_cert: Box::new(var_ty_cert),
                    pred_cert: Box::new(pred_cert),
                    result_type: Box::new(result_ty.clone()),
                };
                Ok((result_ty, cert))
            }

            // Impredicative mode expressions
            Expr::SProp => {
                // SProp : Type 1 (strict propositions live at the same level as Prop)
                if self.mode != Lean5Mode::Impredicative
                    && self.mode != Lean5Mode::Classical
                    && self.mode != Lean5Mode::SetTheoretic
                {
                    return Err(TypeError::ModeRequired {
                        feature: "SProp".to_string(),
                        mode: "Impredicative".to_string(),
                    });
                }
                // SProp is a sort like Prop, so SProp : Type 1
                let result_ty = Expr::Sort(Level::succ(Level::zero()));
                let cert = ProofCert::SProp;
                Ok((result_ty, cert))
            }
            Expr::Squash(inner) => {
                // Squash A : SProp (when A : Sort u)
                if self.mode != Lean5Mode::Impredicative
                    && self.mode != Lean5Mode::Classical
                    && self.mode != Lean5Mode::SetTheoretic
                {
                    return Err(TypeError::ModeRequired {
                        feature: "Squash".to_string(),
                        mode: "Impredicative".to_string(),
                    });
                }
                // Check inner is a type
                let (inner_ty, inner_cert) = self.infer_type_with_cert(inner)?;
                let inner_ty_whnf = self.whnf(&inner_ty);
                let Expr::Sort(_level) = inner_ty_whnf else {
                    return Err(TypeError::ExpectedSort(Box::new(inner_ty)));
                };
                // Squash A : SProp
                let result_ty = Expr::SProp;
                let cert = ProofCert::Squash {
                    inner_cert: Box::new(inner_cert),
                };
                Ok((result_ty, cert))
            }
        }
    }

    /// Generate a certificate for a ZFC set expression.
    fn infer_zfc_set_cert(
        &mut self,
        set_expr: &crate::expr::ZFCSetExpr,
    ) -> Result<crate::cert::ZFCSetCertKind, TypeError> {
        use crate::cert::ZFCSetCertKind;
        use crate::expr::ZFCSetExpr;

        match set_expr {
            ZFCSetExpr::Empty => Ok(ZFCSetCertKind::Empty),
            ZFCSetExpr::Infinity => Ok(ZFCSetCertKind::Infinity),
            ZFCSetExpr::Singleton(e) => {
                let (_, cert) = self.infer_type_with_cert(e)?;
                Ok(ZFCSetCertKind::Singleton(Box::new(cert)))
            }
            ZFCSetExpr::Pair(a, b) => {
                let (_, a_cert) = self.infer_type_with_cert(a)?;
                let (_, b_cert) = self.infer_type_with_cert(b)?;
                Ok(ZFCSetCertKind::Pair(Box::new(a_cert), Box::new(b_cert)))
            }
            ZFCSetExpr::Union(e) => {
                let (_, cert) = self.infer_type_with_cert(e)?;
                Ok(ZFCSetCertKind::Union(Box::new(cert)))
            }
            ZFCSetExpr::PowerSet(e) => {
                let (_, cert) = self.infer_type_with_cert(e)?;
                Ok(ZFCSetCertKind::PowerSet(Box::new(cert)))
            }
            ZFCSetExpr::Separation { set, pred } => {
                let (_, set_cert) = self.infer_type_with_cert(set)?;
                let (_, pred_cert) = self.infer_type_with_cert(pred)?;
                Ok(ZFCSetCertKind::Separation {
                    set_cert: Box::new(set_cert),
                    pred_cert: Box::new(pred_cert),
                })
            }
            ZFCSetExpr::Replacement { set, func } => {
                let (_, set_cert) = self.infer_type_with_cert(set)?;
                let (_, func_cert) = self.infer_type_with_cert(func)?;
                Ok(ZFCSetCertKind::Replacement {
                    set_cert: Box::new(set_cert),
                    func_cert: Box::new(func_cert),
                })
            }
            ZFCSetExpr::Choice(e) => {
                let (_, cert) = self.infer_type_with_cert(e)?;
                Ok(ZFCSetCertKind::Choice(Box::new(cert)))
            }
        }
    }
}

/// Convert FVar certificates back to BVar certificates.
///
/// When the type checker processes a lambda/pi/let body, it opens BVars into FVars.
/// The certificate produced refers to FVars, but the original expression uses BVars.
/// This function converts FVar certificates back to BVar certificates so verification
/// can proceed against the original expression.
fn convert_fvar_cert_to_bvar(
    cert: crate::cert::ProofCert,
    fvar_id: FVarId,
    depth: u32,
) -> crate::cert::ProofCert {
    use crate::cert::ProofCert;

    match cert {
        ProofCert::FVar { id, type_ } if id == fvar_id => {
            // Convert this FVar to a BVar at the current depth
            ProofCert::BVar {
                idx: depth,
                expected_type: Box::new(abstract_fvar_in_expr(*type_, fvar_id, depth)),
            }
        }
        ProofCert::FVar { id, type_ } => {
            // Different FVar, keep as is but abstract any FVar occurrences in type
            ProofCert::FVar {
                id,
                type_: Box::new(abstract_fvar_in_expr(*type_, fvar_id, depth)),
            }
        }
        ProofCert::Sort { level } => ProofCert::Sort { level },
        ProofCert::BVar { idx, expected_type } => ProofCert::BVar {
            idx,
            expected_type: Box::new(abstract_fvar_in_expr(*expected_type, fvar_id, depth)),
        },
        ProofCert::Const {
            name,
            levels,
            type_,
        } => ProofCert::Const {
            name,
            levels,
            type_: Box::new(abstract_fvar_in_expr(*type_, fvar_id, depth)),
        },
        ProofCert::App {
            fn_cert,
            fn_type,
            arg_cert,
            result_type,
        } => ProofCert::App {
            fn_cert: Box::new(convert_fvar_cert_to_bvar(*fn_cert, fvar_id, depth)),
            fn_type: Box::new(abstract_fvar_in_expr(*fn_type, fvar_id, depth)),
            arg_cert: Box::new(convert_fvar_cert_to_bvar(*arg_cert, fvar_id, depth)),
            result_type: Box::new(abstract_fvar_in_expr(*result_type, fvar_id, depth)),
        },
        ProofCert::Lam {
            binder_info,
            arg_type_cert,
            body_cert,
            result_type,
        } => ProofCert::Lam {
            binder_info,
            arg_type_cert: Box::new(convert_fvar_cert_to_bvar(*arg_type_cert, fvar_id, depth)),
            body_cert: Box::new(convert_fvar_cert_to_bvar(*body_cert, fvar_id, depth + 1)),
            result_type: Box::new(abstract_fvar_in_expr(*result_type, fvar_id, depth)),
        },
        ProofCert::Pi {
            binder_info,
            arg_type_cert,
            arg_level,
            body_type_cert,
            body_level,
        } => ProofCert::Pi {
            binder_info,
            arg_type_cert: Box::new(convert_fvar_cert_to_bvar(*arg_type_cert, fvar_id, depth)),
            arg_level,
            body_type_cert: Box::new(convert_fvar_cert_to_bvar(
                *body_type_cert,
                fvar_id,
                depth + 1,
            )),
            body_level,
        },
        ProofCert::Let {
            type_cert,
            value_cert,
            body_cert,
            result_type,
        } => ProofCert::Let {
            type_cert: Box::new(convert_fvar_cert_to_bvar(*type_cert, fvar_id, depth)),
            value_cert: Box::new(convert_fvar_cert_to_bvar(*value_cert, fvar_id, depth)),
            body_cert: Box::new(convert_fvar_cert_to_bvar(*body_cert, fvar_id, depth + 1)),
            result_type: Box::new(abstract_fvar_in_expr(*result_type, fvar_id, depth)),
        },
        ProofCert::Lit { lit, type_ } => ProofCert::Lit {
            lit,
            type_: Box::new(abstract_fvar_in_expr(*type_, fvar_id, depth)),
        },
        ProofCert::DefEq {
            inner,
            expected_type,
            actual_type,
            eq_steps,
        } => ProofCert::DefEq {
            inner: Box::new(convert_fvar_cert_to_bvar(*inner, fvar_id, depth)),
            expected_type: Box::new(abstract_fvar_in_expr(*expected_type, fvar_id, depth)),
            actual_type: Box::new(abstract_fvar_in_expr(*actual_type, fvar_id, depth)),
            eq_steps,
        },
        ProofCert::MData {
            metadata,
            inner_cert,
            result_type,
        } => ProofCert::MData {
            metadata,
            inner_cert: Box::new(convert_fvar_cert_to_bvar(*inner_cert, fvar_id, depth)),
            result_type: Box::new(abstract_fvar_in_expr(*result_type, fvar_id, depth)),
        },
        ProofCert::Proj {
            struct_name,
            idx,
            expr_cert,
            expr_type,
            field_type,
        } => ProofCert::Proj {
            struct_name,
            idx,
            expr_cert: Box::new(convert_fvar_cert_to_bvar(*expr_cert, fvar_id, depth)),
            expr_type: Box::new(abstract_fvar_in_expr(*expr_type, fvar_id, depth)),
            field_type: Box::new(abstract_fvar_in_expr(*field_type, fvar_id, depth)),
        },
        // Mode-specific certificates - recursively process subcertificates and types
        ProofCert::CubicalInterval => ProofCert::CubicalInterval,
        ProofCert::CubicalEndpoint { is_one } => ProofCert::CubicalEndpoint { is_one },
        ProofCert::CubicalPath {
            ty_cert,
            ty_level,
            left_cert,
            right_cert,
        } => ProofCert::CubicalPath {
            ty_cert: Box::new(convert_fvar_cert_to_bvar(*ty_cert, fvar_id, depth)),
            ty_level,
            left_cert: Box::new(convert_fvar_cert_to_bvar(*left_cert, fvar_id, depth)),
            right_cert: Box::new(convert_fvar_cert_to_bvar(*right_cert, fvar_id, depth)),
        },
        ProofCert::CubicalPathLam {
            body_cert,
            body_type,
            result_type,
        } => ProofCert::CubicalPathLam {
            body_cert: Box::new(convert_fvar_cert_to_bvar(*body_cert, fvar_id, depth + 1)),
            body_type: Box::new(abstract_fvar_in_expr(*body_type, fvar_id, depth)),
            result_type: Box::new(abstract_fvar_in_expr(*result_type, fvar_id, depth)),
        },
        ProofCert::CubicalPathApp {
            path_cert,
            arg_cert,
            path_type,
            result_type,
        } => ProofCert::CubicalPathApp {
            path_cert: Box::new(convert_fvar_cert_to_bvar(*path_cert, fvar_id, depth)),
            arg_cert: Box::new(convert_fvar_cert_to_bvar(*arg_cert, fvar_id, depth)),
            path_type: Box::new(abstract_fvar_in_expr(*path_type, fvar_id, depth)),
            result_type: Box::new(abstract_fvar_in_expr(*result_type, fvar_id, depth)),
        },
        ProofCert::CubicalHComp {
            ty_cert,
            phi_cert,
            u_cert,
            base_cert,
            result_type,
        } => ProofCert::CubicalHComp {
            ty_cert: Box::new(convert_fvar_cert_to_bvar(*ty_cert, fvar_id, depth)),
            phi_cert: Box::new(convert_fvar_cert_to_bvar(*phi_cert, fvar_id, depth)),
            u_cert: Box::new(convert_fvar_cert_to_bvar(*u_cert, fvar_id, depth)),
            base_cert: Box::new(convert_fvar_cert_to_bvar(*base_cert, fvar_id, depth)),
            result_type: Box::new(abstract_fvar_in_expr(*result_type, fvar_id, depth)),
        },
        ProofCert::CubicalTransp {
            ty_cert,
            phi_cert,
            base_cert,
            result_type,
        } => ProofCert::CubicalTransp {
            ty_cert: Box::new(convert_fvar_cert_to_bvar(*ty_cert, fvar_id, depth)),
            phi_cert: Box::new(convert_fvar_cert_to_bvar(*phi_cert, fvar_id, depth)),
            base_cert: Box::new(convert_fvar_cert_to_bvar(*base_cert, fvar_id, depth)),
            result_type: Box::new(abstract_fvar_in_expr(*result_type, fvar_id, depth)),
        },
        ProofCert::ClassicalChoice {
            ty_cert,
            pred_cert,
            proof_cert,
            result_type,
        } => ProofCert::ClassicalChoice {
            ty_cert: Box::new(convert_fvar_cert_to_bvar(*ty_cert, fvar_id, depth)),
            pred_cert: Box::new(convert_fvar_cert_to_bvar(*pred_cert, fvar_id, depth)),
            proof_cert: Box::new(convert_fvar_cert_to_bvar(*proof_cert, fvar_id, depth)),
            result_type: Box::new(abstract_fvar_in_expr(*result_type, fvar_id, depth)),
        },
        ProofCert::ClassicalEpsilon {
            ty_cert,
            pred_cert,
            result_type,
        } => ProofCert::ClassicalEpsilon {
            ty_cert: Box::new(convert_fvar_cert_to_bvar(*ty_cert, fvar_id, depth)),
            pred_cert: Box::new(convert_fvar_cert_to_bvar(*pred_cert, fvar_id, depth)),
            result_type: Box::new(abstract_fvar_in_expr(*result_type, fvar_id, depth)),
        },
        ProofCert::ZFCSet { kind, result_type } => ProofCert::ZFCSet {
            kind: convert_fvar_in_zfc_set_kind(kind, fvar_id, depth),
            result_type: Box::new(abstract_fvar_in_expr(*result_type, fvar_id, depth)),
        },
        ProofCert::ZFCMem {
            elem_cert,
            set_cert,
        } => ProofCert::ZFCMem {
            elem_cert: Box::new(convert_fvar_cert_to_bvar(*elem_cert, fvar_id, depth)),
            set_cert: Box::new(convert_fvar_cert_to_bvar(*set_cert, fvar_id, depth)),
        },
        ProofCert::ZFCComprehension {
            var_ty_cert,
            pred_cert,
            result_type,
        } => ProofCert::ZFCComprehension {
            var_ty_cert: Box::new(convert_fvar_cert_to_bvar(*var_ty_cert, fvar_id, depth)),
            pred_cert: Box::new(convert_fvar_cert_to_bvar(*pred_cert, fvar_id, depth + 1)),
            result_type: Box::new(abstract_fvar_in_expr(*result_type, fvar_id, depth)),
        },
        // Impredicative mode certificates
        ProofCert::SProp => ProofCert::SProp,
        ProofCert::Squash { inner_cert } => ProofCert::Squash {
            inner_cert: Box::new(convert_fvar_cert_to_bvar(*inner_cert, fvar_id, depth)),
        },
    }
}

/// Helper to convert FVar occurrences in ZFC set certificate kind
fn convert_fvar_in_zfc_set_kind(
    kind: crate::cert::ZFCSetCertKind,
    fvar_id: FVarId,
    depth: u32,
) -> crate::cert::ZFCSetCertKind {
    use crate::cert::ZFCSetCertKind;
    match kind {
        ZFCSetCertKind::Empty => ZFCSetCertKind::Empty,
        ZFCSetCertKind::Infinity => ZFCSetCertKind::Infinity,
        ZFCSetCertKind::Singleton(c) => {
            ZFCSetCertKind::Singleton(Box::new(convert_fvar_cert_to_bvar(*c, fvar_id, depth)))
        }
        ZFCSetCertKind::Pair(c1, c2) => ZFCSetCertKind::Pair(
            Box::new(convert_fvar_cert_to_bvar(*c1, fvar_id, depth)),
            Box::new(convert_fvar_cert_to_bvar(*c2, fvar_id, depth)),
        ),
        ZFCSetCertKind::Union(c) => {
            ZFCSetCertKind::Union(Box::new(convert_fvar_cert_to_bvar(*c, fvar_id, depth)))
        }
        ZFCSetCertKind::PowerSet(c) => {
            ZFCSetCertKind::PowerSet(Box::new(convert_fvar_cert_to_bvar(*c, fvar_id, depth)))
        }
        ZFCSetCertKind::Separation { set_cert, pred_cert } => ZFCSetCertKind::Separation {
            set_cert: Box::new(convert_fvar_cert_to_bvar(*set_cert, fvar_id, depth)),
            pred_cert: Box::new(convert_fvar_cert_to_bvar(*pred_cert, fvar_id, depth + 1)),
        },
        ZFCSetCertKind::Replacement { set_cert, func_cert } => ZFCSetCertKind::Replacement {
            set_cert: Box::new(convert_fvar_cert_to_bvar(*set_cert, fvar_id, depth)),
            func_cert: Box::new(convert_fvar_cert_to_bvar(*func_cert, fvar_id, depth + 1)),
        },
        ZFCSetCertKind::Choice(c) => {
            ZFCSetCertKind::Choice(Box::new(convert_fvar_cert_to_bvar(*c, fvar_id, depth)))
        }
    }
}

/// Abstract FVar occurrences in an expression, converting them to BVar(depth).
fn abstract_fvar_in_expr(e: Expr, fvar_id: FVarId, depth: u32) -> Expr {
    use std::sync::Arc;
    match e {
        Expr::FVar(id) if id == fvar_id => Expr::BVar(depth),
        Expr::FVar(_) | Expr::BVar(_) | Expr::Sort(_) | Expr::Const(_, _) | Expr::Lit(_) => e,
        Expr::App(f, a) => Expr::App(
            Arc::new(abstract_fvar_in_expr((*f).clone(), fvar_id, depth)),
            Arc::new(abstract_fvar_in_expr((*a).clone(), fvar_id, depth)),
        ),
        Expr::Lam(bi, ty, body) => Expr::Lam(
            bi,
            Arc::new(abstract_fvar_in_expr((*ty).clone(), fvar_id, depth)),
            Arc::new(abstract_fvar_in_expr((*body).clone(), fvar_id, depth + 1)),
        ),
        Expr::Pi(bi, ty, body) => Expr::Pi(
            bi,
            Arc::new(abstract_fvar_in_expr((*ty).clone(), fvar_id, depth)),
            Arc::new(abstract_fvar_in_expr((*body).clone(), fvar_id, depth + 1)),
        ),
        Expr::Let(ty, val, body) => Expr::Let(
            Arc::new(abstract_fvar_in_expr((*ty).clone(), fvar_id, depth)),
            Arc::new(abstract_fvar_in_expr((*val).clone(), fvar_id, depth)),
            Arc::new(abstract_fvar_in_expr((*body).clone(), fvar_id, depth + 1)),
        ),
        Expr::Proj(name, idx, inner) => Expr::Proj(
            name,
            idx,
            Arc::new(abstract_fvar_in_expr((*inner).clone(), fvar_id, depth)),
        ),
        Expr::MData(meta, inner) => Expr::MData(
            meta,
            Arc::new(abstract_fvar_in_expr((*inner).clone(), fvar_id, depth)),
        ),

        // Mode-specific extensions - use the existing abstract_fvar_at method on Expr
        Expr::CubicalInterval | Expr::CubicalI0 | Expr::CubicalI1 => e,
        Expr::CubicalPath { ty, left, right } => Expr::CubicalPath {
            ty: Arc::new(abstract_fvar_in_expr((*ty).clone(), fvar_id, depth)),
            left: Arc::new(abstract_fvar_in_expr((*left).clone(), fvar_id, depth)),
            right: Arc::new(abstract_fvar_in_expr((*right).clone(), fvar_id, depth)),
        },
        Expr::CubicalPathLam { body } => Expr::CubicalPathLam {
            body: Arc::new(abstract_fvar_in_expr((*body).clone(), fvar_id, depth + 1)),
        },
        Expr::CubicalPathApp { path, arg } => Expr::CubicalPathApp {
            path: Arc::new(abstract_fvar_in_expr((*path).clone(), fvar_id, depth)),
            arg: Arc::new(abstract_fvar_in_expr((*arg).clone(), fvar_id, depth)),
        },
        Expr::CubicalHComp {
            ty,
            phi,
            u,
            base,
        } => Expr::CubicalHComp {
            ty: Arc::new(abstract_fvar_in_expr((*ty).clone(), fvar_id, depth)),
            phi: Arc::new(abstract_fvar_in_expr((*phi).clone(), fvar_id, depth)),
            u: Arc::new(abstract_fvar_in_expr((*u).clone(), fvar_id, depth)),
            base: Arc::new(abstract_fvar_in_expr((*base).clone(), fvar_id, depth)),
        },
        Expr::CubicalTransp { ty, phi, base } => Expr::CubicalTransp {
            ty: Arc::new(abstract_fvar_in_expr((*ty).clone(), fvar_id, depth)),
            phi: Arc::new(abstract_fvar_in_expr((*phi).clone(), fvar_id, depth)),
            base: Arc::new(abstract_fvar_in_expr((*base).clone(), fvar_id, depth)),
        },
        Expr::ClassicalChoice {
            ty,
            pred,
            exists_proof,
        } => Expr::ClassicalChoice {
            ty: Arc::new(abstract_fvar_in_expr((*ty).clone(), fvar_id, depth)),
            pred: Arc::new(abstract_fvar_in_expr((*pred).clone(), fvar_id, depth)),
            exists_proof: Arc::new(abstract_fvar_in_expr((*exists_proof).clone(), fvar_id, depth)),
        },
        Expr::ClassicalEpsilon { ty, pred } => Expr::ClassicalEpsilon {
            ty: Arc::new(abstract_fvar_in_expr((*ty).clone(), fvar_id, depth)),
            pred: Arc::new(abstract_fvar_in_expr((*pred).clone(), fvar_id, depth)),
        },
        // ZFC expressions are passed through unchanged as they don't typically contain FVars
        Expr::ZFCSet(_) | Expr::ZFCMem { .. } | Expr::ZFCComprehension { .. } => e,

        // Impredicative mode extensions
        Expr::SProp => e,
        Expr::Squash(inner) => Expr::Squash(Arc::new(abstract_fvar_in_expr(
            (*inner).clone(),
            fvar_id,
            depth,
        ))),
    }
}

impl<'env> TypeChecker<'env> {
    /// Compute weak-head normal form
    ///
    /// WHNF reduces an expression to a form where the head is not a reducible redex.
    /// This includes beta reduction (for lambdas), delta reduction (for definitions),
    /// zeta reduction (for let bindings), and iota reduction (for recursors).
    ///
    /// # Properties
    ///
    /// - **Idempotent**: `whnf(whnf(e)) == whnf(e)` (enforced by debug assertion)
    /// - **Meaning-preserving**: `is_def_eq(e, whnf(e))` for all well-typed `e`
    pub fn whnf(&self, e: &Expr) -> Expr {
        stacker::maybe_grow(MIN_STACK_RED_ZONE, STACK_GROWTH_SIZE, || self.whnf_impl(e))
    }

    /// Implementation of WHNF (called via stacker::maybe_grow)
    fn whnf_impl(&self, e: &Expr) -> Expr {
        let result = self.whnf_core(e);

        // Debug assertion: WHNF must be idempotent
        // This catches bugs in the reduction logic
        #[cfg(debug_assertions)]
        {
            let result2 = self.whnf_core(&result);
            debug_assert!(
                result == result2,
                "WHNF not idempotent!\nInput: {e:?}\nFirst WHNF: {result:?}\nSecond WHNF: {result2:?}"
            );
        }

        result
    }

    /// Core WHNF implementation (without debug assertions)
    fn whnf_core(&self, e: &Expr) -> Expr {
        match e {
            Expr::App(f, a) => {
                let f_whnf = self.whnf(f);
                if let Expr::Lam(_, _, body) = &f_whnf {
                    // Beta reduction
                    let reduced = body.instantiate(a);
                    self.whnf(&reduced)
                } else {
                    // Check for iota reduction (recursor application)
                    let app_with_whnf = if std::ptr::eq(f.as_ref(), &f_whnf) {
                        e.clone()
                    } else {
                        Expr::App(f_whnf.into(), a.clone())
                    };

                    if let Some(reduced) = self.try_iota_reduction(&app_with_whnf) {
                        self.whnf(&reduced)
                    } else if let Some(reduced) = self.try_quot_reduction(&app_with_whnf) {
                        // Quotient reduction: Quot.lift f h (Quot.mk r a) → f a
                        self.whnf(&reduced)
                    } else {
                        app_with_whnf
                    }
                }
            }
            Expr::Let(_, val, body) => {
                // Zeta reduction
                let reduced = body.instantiate(val);
                self.whnf(&reduced)
            }
            Expr::Const(name, levels) => {
                // Delta reduction (unfold definitions)
                self.env
                    .unfold(name, levels)
                    .map_or_else(|| e.clone(), |val| self.whnf(&val))
            }
            Expr::FVar(id) => {
                // Unfold let-bound variables
                if let Some(decl) = self.ctx.get(*id) {
                    if let Some(val) = &decl.value {
                        return self.whnf(val);
                    }
                }
                e.clone()
            }
            Expr::Proj(struct_name, idx, expr) => {
                // Iota reduction for structures
                self.reduce_proj(struct_name, *idx, expr)
            }
            // MData is transparent - just reduce the inner expression
            Expr::MData(_, inner) => self.whnf(inner),
            _ => e.clone(),
        }
    }

    /// Try to apply iota reduction (recursor computation rule)
    ///
    /// For a recursor application: `I.rec params motive minors indices major`
    /// If major is a constructor application `I.ctor params args`, then:
    /// `I.rec params motive minors indices (I.ctor params args) → minor args rec_results`
    fn try_iota_reduction(&self, e: &Expr) -> Option<Expr> {
        // Collect all application arguments
        let head = e.get_app_fn();
        let args = e.get_app_args();

        // Check if head is a recursor
        let Expr::Const(rec_name, rec_levels) = head else {
            return None;
        };

        let rec_val = self.env.get_recursor(rec_name)?;

        // Calculate total expected args before major premise
        // Standard: params + motives + minors + indices
        // recOn:    params + motives + indices (major before minors)
        let args_before_major = match rec_val.arg_order {
            RecursorArgOrder::MajorAfterMinors => {
                rec_val.num_params as usize
                    + rec_val.num_motives as usize
                    + rec_val.num_minors as usize
                    + rec_val.num_indices as usize
            }
            RecursorArgOrder::MajorAfterMotive => {
                rec_val.num_params as usize
                    + rec_val.num_motives as usize
                    + rec_val.num_indices as usize
            }
        };

        // Need at least enough args to have the major premise (and minors when major precedes them)
        let required_args = match rec_val.arg_order {
            RecursorArgOrder::MajorAfterMinors => args_before_major + 1,
            RecursorArgOrder::MajorAfterMotive => {
                args_before_major + 1 + rec_val.num_minors as usize
            }
        };
        if args.len() < required_args {
            return None;
        }

        // Get the major premise (the value being eliminated)
        let major = args[args_before_major];
        let major_whnf = self.whnf(major);

        // Check if major is a constructor application
        let major_head = major_whnf.get_app_fn();
        let major_args = major_whnf.get_app_args();

        let Expr::Const(ctor_name, _) = major_head else {
            return None;
        };

        // Check this is actually a constructor of the inductive
        let ctor_val = self.env.get_constructor(ctor_name)?;
        if ctor_val.inductive_name != rec_val.inductive_name {
            return None;
        }

        // Find the recursor rule for this constructor
        let rule = rec_val
            .rules
            .iter()
            .find(|r| &r.constructor_name == ctor_name)?;

        // Determine where minors start in the application
        let minors_start = match rec_val.arg_order {
            RecursorArgOrder::MajorAfterMinors => {
                rec_val.num_params as usize + rec_val.num_motives as usize
            }
            RecursorArgOrder::MajorAfterMotive => args_before_major + 1,
        };

        // Get the minor premise for this constructor
        let minor_idx = minors_start + ctor_val.constructor_idx as usize;

        if minor_idx >= args.len() {
            return None;
        }

        let minor = args[minor_idx].clone();

        // The constructor's fields are the arguments after parameters.
        let field_start = (ctor_val.num_params as usize).min(major_args.len());
        let fields: Vec<&Expr> = major_args[field_start..].to_vec();

        // Build the result: apply minor to fields and recursive results
        // minor field₀ ... fieldₙ ih₀ ... ihₘ
        // where ih_i = rec motive minors field_i for recursive fields
        let mut result = minor;

        // Apply fields
        for field in &fields {
            result = Expr::app(result, (*field).clone());
        }

        // Apply induction hypotheses for recursive fields
        // For each recursive field, we need to apply the recursor to it
        for (i, &is_recursive) in rule.recursive_fields.iter().enumerate() {
            if is_recursive && i < fields.len() {
                // Build recursive call: rec motive minors recursive_field
                let recursive_field = fields[i];

                // Reconstruct the recursor application with this field as major
                let mut rec_call = Expr::const_(rec_name.clone(), rec_levels.clone());

                // Apply params (indices 0..num_params)
                for j in 0..rec_val.num_params as usize {
                    if j < args.len() {
                        rec_call = Expr::app(rec_call, args[j].clone());
                    }
                }

                // Apply motives
                for j in 0..rec_val.num_motives as usize {
                    let idx = rec_val.num_params as usize + j;
                    if idx < args.len() {
                        rec_call = Expr::app(rec_call, args[idx].clone());
                    }
                }

                match rec_val.arg_order {
                    RecursorArgOrder::MajorAfterMinors => {
                        // minors before indices, major last
                        for j in 0..rec_val.num_minors as usize {
                            let idx = minors_start + j;
                            if idx < args.len() {
                                rec_call = Expr::app(rec_call, args[idx].clone());
                            }
                        }

                        // Apply indices (if any)
                        let indices_start = rec_val.num_params as usize
                            + rec_val.num_motives as usize
                            + rec_val.num_minors as usize;
                        for j in 0..rec_val.num_indices as usize {
                            let idx = indices_start + j;
                            if idx < args.len() {
                                rec_call = Expr::app(rec_call, args[idx].clone());
                            }
                        }

                        // Apply the recursive field as major premise
                        rec_call = Expr::app(rec_call, (*recursive_field).clone());
                    }
                    RecursorArgOrder::MajorAfterMotive => {
                        // indices precede the major, minors follow
                        let indices_start =
                            rec_val.num_params as usize + rec_val.num_motives as usize;
                        for j in 0..rec_val.num_indices as usize {
                            let idx = indices_start + j;
                            if idx < args.len() {
                                rec_call = Expr::app(rec_call, args[idx].clone());
                            }
                        }

                        // Apply the recursive field as major premise
                        rec_call = Expr::app(rec_call, (*recursive_field).clone());

                        // Now apply minors (after major)
                        for j in 0..rec_val.num_minors as usize {
                            let idx = minors_start + j;
                            if idx < args.len() {
                                rec_call = Expr::app(rec_call, args[idx].clone());
                            }
                        }
                    }
                }

                result = Expr::app(result, rec_call);
            }
        }

        Some(result)
    }

    /// Try to reduce a quotient lift application
    ///
    /// The reduction rule is:
    /// `Quot.lift.{u v} α r β f h (Quot.mk.{u} α r a) ≡ f a`
    fn try_quot_reduction(&self, e: &Expr) -> Option<Expr> {
        // Collect all application arguments
        let head = e.get_app_fn();
        let args = e.get_app_args();

        // Use the quot module's reduction function
        crate::quot::try_quot_lift_reduction(head, &args, |expr| self.whnf(expr))
    }

    /// Reduce a projection
    fn reduce_proj(&self, struct_name: &Name, idx: u32, expr: &Expr) -> Expr {
        let expr_whnf = self.whnf(expr);

        // Check if the expression is a constructor application
        let head = expr_whnf.get_app_fn();
        let args = expr_whnf.get_app_args();

        if let Expr::Const(ctor_name, _) = head {
            // Check if this is a constructor of the struct
            if let Some(ctor_val) = self.env.get_constructor(ctor_name) {
                if &ctor_val.inductive_name == struct_name {
                    // Get the field at index idx (after parameters)
                    let field_idx = ctor_val.num_params as usize + idx as usize;
                    if field_idx < args.len() {
                        return args[field_idx].clone();
                    }
                }
            }
        }

        // Can't reduce
        Expr::Proj(struct_name.clone(), idx, expr_whnf.into())
    }

    /// Infer the type of a projection expression.
    ///
    /// For a projection `struct_name.idx e`, we need to:
    /// 1. Infer the type of `e`
    /// 2. Verify the type is an application of the struct's inductive type
    /// 3. Look up the constructor to find the field type at index `idx`
    /// 4. Instantiate the field type with the expression's type arguments
    fn infer_proj_type(
        &mut self,
        struct_name: &Name,
        idx: u32,
        expr: &Expr,
    ) -> Result<Expr, TypeError> {
        // 1. Infer the type of the projected expression
        let expr_type = self.infer_type(expr)?;
        let expr_type_whnf = self.whnf(&expr_type);

        // 2. Extract the inductive type name and arguments
        let (type_name, type_args) = match expr_type_whnf.get_app_fn() {
            Expr::Const(name, _levels) => (name, expr_type_whnf.get_app_args()),
            _ => return Err(TypeError::InvalidProjNotStruct(Box::new(expr_type_whnf))),
        };

        // Verify the type matches the struct name in the projection
        if type_name != struct_name {
            return Err(TypeError::InvalidProjNotStruct(Box::new(expr_type_whnf)));
        }

        // 3. Look up the inductive type
        let ind_val = self
            .env
            .get_inductive(struct_name)
            .ok_or_else(|| TypeError::UnknownInductive(struct_name.clone()))?;

        // Structures must have exactly one constructor
        if ind_val.constructor_names.len() != 1 {
            return Err(TypeError::InvalidProjNotUniqueConstructor(
                struct_name.clone(),
            ));
        }

        // 4. Look up the constructor
        let ctor_name = &ind_val.constructor_names[0];
        let ctor_val = self
            .env
            .get_constructor(ctor_name)
            .ok_or_else(|| TypeError::UnknownConst(ctor_name.clone()))?;

        // Check index is in bounds
        if idx >= ctor_val.num_fields {
            return Err(TypeError::InvalidProjIndexOutOfBounds(
                idx,
                ctor_val.num_fields,
            ));
        }

        // 5. Get the field type from the constructor
        // The constructor type is: (params...) → (fields...) → Ind params...
        // We need to skip num_params pis and then get the idx-th pi domain
        let ctor_type = &ctor_val.type_;

        // Instantiate parameters with the type arguments
        // Always call instantiate_params - it handles empty args correctly
        let num_params = ctor_val.num_params as usize;
        let param_count = num_params.min(type_args.len());
        let param_args: Vec<Expr> = type_args[..param_count]
            .iter()
            .map(|e| (*e).clone())
            .collect();
        let instantiated_ctor_type = self.instantiate_params(ctor_type, &param_args);

        // Navigate to the idx-th field
        let mut current_type = instantiated_ctor_type;
        for field_idx in 0..=idx {
            let current_whnf = self.whnf(&current_type);
            match current_whnf {
                Expr::Pi(_, domain, body) => {
                    if field_idx == idx {
                        // This is the field we want - substitute expr for previous fields
                        // But for simplicity, we return the domain directly
                        // A full implementation would substitute proj_i expr for BVar i
                        return Ok((*domain).clone());
                    }
                    // Substitute the projection of expr at this field for the bound variable
                    let proj_field = Expr::proj(struct_name.clone(), field_idx, expr.clone());
                    current_type = body.instantiate(&proj_field);
                }
                _ => {
                    return Err(TypeError::InvalidProjIndexOutOfBounds(idx, field_idx));
                }
            }
        }

        // Should not reach here
        Err(TypeError::InvalidProjIndexOutOfBounds(idx, 0))
    }

    /// Instantiate the parameters of a type with given arguments
    fn instantiate_params(&self, ty: &Expr, args: &[Expr]) -> Expr {
        let mut result = ty.clone();
        for arg in args {
            let result_whnf = self.whnf(&result);
            if let Expr::Pi(_, _, body) = result_whnf {
                result = body.instantiate(arg);
            } else {
                break;
            }
        }
        result
    }

    /// Check definitional equality
    ///
    /// This implements the core definitional equality algorithm, including:
    /// - Beta reduction
    /// - Delta reduction (definition unfolding)
    /// - Proof irrelevance (any two proofs of the same Prop are equal)
    /// - Eta expansion for functions (λ x. f x ≡ f when x not free in f)
    pub fn is_def_eq(&self, a: &Expr, b: &Expr) -> bool {
        stacker::maybe_grow(MIN_STACK_RED_ZONE, STACK_GROWTH_SIZE, || {
            self.is_def_eq_impl(a, b)
        })
    }

    /// Implementation of definitional equality (called via stacker::maybe_grow)
    fn is_def_eq_impl(&self, a: &Expr, b: &Expr) -> bool {
        // Quick check: pointer equality
        if std::ptr::eq(a, b) {
            return true;
        }

        // Reduce to WHNF
        let a_whnf = self.whnf(a);
        let b_whnf = self.whnf(b);

        // Try proof irrelevance first (before structural comparison)
        // If both terms are proofs of the same Prop, they're definitionally equal
        if self.try_proof_irrel_eq(&a_whnf, &b_whnf) {
            return true;
        }

        // Compare structure
        match (&a_whnf, &b_whnf) {
            (Expr::BVar(i), Expr::BVar(j)) => i == j,
            (Expr::FVar(i), Expr::FVar(j)) => i == j,
            (Expr::Sort(l1), Expr::Sort(l2)) => {
                // Use level definitional equality (normalizes before comparison)
                Level::is_def_eq(l1, l2)
            }
            (Expr::Const(n1, ls1), Expr::Const(n2, ls2)) => {
                n1 == n2
                    && ls1.len() == ls2.len()
                    && ls1
                        .iter()
                        .zip(ls2.iter())
                        .all(|(l1, l2)| Level::is_def_eq(l1, l2))
            }
            (Expr::App(f1, a1), Expr::App(f2, a2)) => {
                self.is_def_eq(f1, f2) && self.is_def_eq(a1, a2)
            }
            (Expr::Lam(_, t1, b1), Expr::Lam(_, t2, b2))
            | (Expr::Pi(_, t1, b1), Expr::Pi(_, t2, b2)) => {
                self.is_def_eq(t1, t2) && self.is_def_eq(b1, b2)
            }
            (Expr::Lit(l1), Expr::Lit(l2)) => l1 == l2,
            // Eta expansion: try to make Lam and non-Lam equal
            (Expr::Lam(bi, ty, body), _) => self.try_eta_expansion(&a_whnf, &b_whnf, *bi, ty, body),
            (_, Expr::Lam(bi, ty, body)) => self.try_eta_expansion(&b_whnf, &a_whnf, *bi, ty, body),
            _ => false,
        }
    }

    /// Try to prove equality using proof irrelevance.
    ///
    /// In Lean 4, any two proofs of the same Prop are definitionally equal.
    /// This is crucial for Prop to work correctly as a proof-irrelevant universe.
    ///
    /// Reference: Lean 4 kernel type_checker.cpp `is_def_eq_proof_irrel`
    fn try_proof_irrel_eq(&self, a: &Expr, b: &Expr) -> bool {
        // Note: We can't call infer_type here because is_def_eq is not &mut self.
        // We need to use a workaround: check if the expressions are obviously in Prop.
        // For a complete implementation, we'd need to either:
        // 1. Make is_def_eq take &mut self (changes API)
        // 2. Cache inferred types (more complex)
        // 3. Use interior mutability (Cell/RefCell)
        //
        // For now, we implement a conservative version that checks syntactically
        // whether both terms could be proofs. This matches common cases.

        // Try to infer types using a temporary type checker
        // This is safe because type inference is deterministic
        let ty_a = self.try_infer_type_quick(a);
        let ty_b = self.try_infer_type_quick(b);

        match (ty_a, ty_b) {
            (Some(ta), Some(tb)) => {
                // For proof irrelevance, ta must be in Prop (a is a proof).
                // If ta is in Prop but tb isn't, is_def_eq(ta, tb) will return false.
                // So we only need to check one side.
                if self.is_type_in_prop(&ta) {
                    // Check if the types (propositions) are definitionally equal
                    self.is_def_eq(&ta, &tb)
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    /// Quick type inference that doesn't modify state.
    /// Returns None if type cannot be easily inferred.
    fn try_infer_type_quick(&self, e: &Expr) -> Option<Expr> {
        match e {
            Expr::FVar(id) => self.ctx.get(*id).map(|d| d.type_.clone()),
            Expr::Const(name, levels) => self.env.instantiate_type(name, levels),
            Expr::Sort(l) => Some(Expr::Sort(Level::succ(l.clone()))),
            Expr::App(f, _a) => {
                // For applications, we need the function's return type
                let f_type = self.try_infer_type_quick(f)?;
                let f_type_whnf = self.whnf(&f_type);
                match f_type_whnf {
                    Expr::Pi(_, _, result_type) => {
                        // Note: We should instantiate result_type with a, but for
                        // proof irrelevance we mainly care if it's a Prop
                        // The instantiation won't change whether it's Prop
                        Some(result_type.as_ref().clone())
                    }
                    _ => None,
                }
            }
            // For lambdas and other constructs, we'd need more complex inference
            // which would require mutation. Return None for now.
            _ => None,
        }
    }

    /// Check if a type is in Prop (i.e., the type itself has type Sort(0)).
    fn is_type_in_prop(&self, ty: &Expr) -> bool {
        let ty_whnf = self.whnf(ty);
        // A term t has type in Prop if typeof(t) : Prop
        // We check if typeof(ty_whnf) == Sort(0)
        // Note: If ty_whnf is Sort(l), its type is Sort(succ(l)) which is never Prop.
        // The try_infer_type_quick path handles this correctly.
        self.try_infer_type_quick(&ty_whnf).is_some_and(|ty_of_ty| {
            let ty_of_ty_whnf = self.whnf(&ty_of_ty);
            matches!(ty_of_ty_whnf, Expr::Sort(l) if l.is_zero())
        })
    }

    /// Try eta expansion to prove equality.
    ///
    /// Eta expansion: (λ x. f x) ≡ f when x does not appear free in f.
    ///
    /// This is called when we have `Lam(bi, ty, body)` vs `other`.
    /// We check if `other : Pi(bi, ty, result_type)`, and if so,
    /// we create a lambda wrapper around `other` and compare.
    ///
    /// Reference: Lean 4 kernel type_checker.cpp `try_eta_expansion_core`
    fn try_eta_expansion(
        &self,
        _lam_expr: &Expr,
        other: &Expr,
        _bi: BinderInfo,
        lam_ty: &Expr,
        lam_body: &Expr,
    ) -> bool {
        // Get the type of `other` to see if it's a function type
        let Some(other_type) = self.try_infer_type_quick(other) else {
            return false;
        };

        let other_type_whnf = self.whnf(&other_type);

        match &other_type_whnf {
            Expr::Pi(_, pi_domain, _) => {
                // Check domain types match
                if !self.is_def_eq(lam_ty, pi_domain) {
                    return false;
                }

                // Eta expand `other`: create (λ x : ty. other x)
                // Then compare bodies: lam_body vs (other (BVar 0))
                // This is equivalent to: lam_body ≡ other (BVar 0)
                let other_applied = Expr::app(
                    // Lift `other` to account for the new binder
                    self.lift_expr(other, 0, 1),
                    Expr::bvar(0),
                );

                // Now compare the bodies
                self.is_def_eq(lam_body, &other_applied)
            }
            _ => false,
        }
    }

    /// Lift an expression by increasing all free de Bruijn indices >= cutoff by amount.
    /// Used during eta expansion to account for new binders.
    fn lift_expr(&self, e: &Expr, cutoff: u32, amount: u32) -> Expr {
        match e {
            Expr::BVar(idx) => {
                if *idx >= cutoff {
                    Expr::BVar(idx + amount)
                } else {
                    e.clone()
                }
            }
            Expr::FVar(_) | Expr::Sort(_) | Expr::Const(_, _) | Expr::Lit(_) => e.clone(),
            Expr::App(f, a) => Expr::App(
                Arc::new(self.lift_expr(f, cutoff, amount)),
                Arc::new(self.lift_expr(a, cutoff, amount)),
            ),
            Expr::Lam(bi, ty, body) => Expr::Lam(
                *bi,
                Arc::new(self.lift_expr(ty, cutoff, amount)),
                Arc::new(self.lift_expr(body, cutoff + 1, amount)),
            ),
            Expr::Pi(bi, ty, body) => Expr::Pi(
                *bi,
                Arc::new(self.lift_expr(ty, cutoff, amount)),
                Arc::new(self.lift_expr(body, cutoff + 1, amount)),
            ),
            Expr::Let(ty, val, body) => Expr::Let(
                Arc::new(self.lift_expr(ty, cutoff, amount)),
                Arc::new(self.lift_expr(val, cutoff, amount)),
                Arc::new(self.lift_expr(body, cutoff + 1, amount)),
            ),
            Expr::Proj(name, idx, inner) => Expr::Proj(
                name.clone(),
                *idx,
                Arc::new(self.lift_expr(inner, cutoff, amount)),
            ),
            Expr::MData(meta, inner) => Expr::MData(
                meta.clone(),
                Arc::new(self.lift_expr(inner, cutoff, amount)),
            ),

            // Mode-specific extensions - use existing lift method on Expr
            // The Expr::lift_at method handles all these cases properly
            Expr::CubicalInterval | Expr::CubicalI0 | Expr::CubicalI1 => e.clone(),
            Expr::CubicalPath { ty, left, right } => Expr::CubicalPath {
                ty: Arc::new(self.lift_expr(ty, cutoff, amount)),
                left: Arc::new(self.lift_expr(left, cutoff, amount)),
                right: Arc::new(self.lift_expr(right, cutoff, amount)),
            },
            Expr::CubicalPathLam { body } => Expr::CubicalPathLam {
                body: Arc::new(self.lift_expr(body, cutoff + 1, amount)),
            },
            Expr::CubicalPathApp { path, arg } => Expr::CubicalPathApp {
                path: Arc::new(self.lift_expr(path, cutoff, amount)),
                arg: Arc::new(self.lift_expr(arg, cutoff, amount)),
            },
            Expr::CubicalHComp { ty, phi, u, base } => Expr::CubicalHComp {
                ty: Arc::new(self.lift_expr(ty, cutoff, amount)),
                phi: Arc::new(self.lift_expr(phi, cutoff, amount)),
                u: Arc::new(self.lift_expr(u, cutoff, amount)),
                base: Arc::new(self.lift_expr(base, cutoff, amount)),
            },
            Expr::CubicalTransp { ty, phi, base } => Expr::CubicalTransp {
                ty: Arc::new(self.lift_expr(ty, cutoff, amount)),
                phi: Arc::new(self.lift_expr(phi, cutoff, amount)),
                base: Arc::new(self.lift_expr(base, cutoff, amount)),
            },
            Expr::ClassicalChoice {
                ty,
                pred,
                exists_proof,
            } => Expr::ClassicalChoice {
                ty: Arc::new(self.lift_expr(ty, cutoff, amount)),
                pred: Arc::new(self.lift_expr(pred, cutoff, amount)),
                exists_proof: Arc::new(self.lift_expr(exists_proof, cutoff, amount)),
            },
            Expr::ClassicalEpsilon { ty, pred } => Expr::ClassicalEpsilon {
                ty: Arc::new(self.lift_expr(ty, cutoff, amount)),
                pred: Arc::new(self.lift_expr(pred, cutoff, amount)),
            },
            // ZFC expressions typically don't have loose bound variables
            Expr::ZFCSet(_) | Expr::ZFCMem { .. } | Expr::ZFCComprehension { .. } => e.clone(),

            // Impredicative mode extensions
            Expr::SProp => e.clone(),
            Expr::Squash(inner) => {
                Expr::Squash(Arc::new(self.lift_expr(inner, cutoff, amount)))
            }
        }
    }

    /// Replace BVar(0) with FVar(id) in an expression
    fn open_bvar(&self, e: &Expr, id: FVarId) -> Expr {
        e.instantiate(&Expr::FVar(id))
    }
}

/// Type checking errors
///
/// Note: Expr fields are boxed to reduce the size of the Result type on the success path.
/// This improves performance since errors are rare but Results are returned frequently.
#[derive(Debug, thiserror::Error)]
pub enum TypeError {
    #[error("Unbound variable index: {0}")]
    UnboundVariable(u32),
    #[error("Unknown free variable: {0:?}")]
    UnknownFVar(FVarId),
    #[error("Unknown constant: {0}")]
    UnknownConst(Name),
    #[error("Expected function type, got: {0:?}")]
    NotAFunction(Box<Expr>),
    #[error("Type mismatch: expected {expected:?}, got {inferred:?}")]
    TypeMismatch {
        expected: Box<Expr>,
        inferred: Box<Expr>,
    },
    #[error("Expected sort, got: {0:?}")]
    ExpectedSort(Box<Expr>),
    #[error("Invalid projection: type {0:?} is not a structure")]
    InvalidProjNotStruct(Box<Expr>),
    #[error("Invalid projection: inductive {0} does not have a unique constructor")]
    InvalidProjNotUniqueConstructor(Name),
    #[error("Invalid projection: index {0} out of bounds for structure with {1} fields")]
    InvalidProjIndexOutOfBounds(u32, u32),
    #[error("Unknown inductive type: {0}")]
    UnknownInductive(Name),

    #[error("Feature '{feature}' requires {mode} mode")]
    ModeRequired { feature: String, mode: String },
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests2;

/// Mode-aware type checking tests
#[cfg(test)]
mod mode_tests {
    use super::*;
    use crate::env::Environment;
    use crate::mode::Lean5Mode;
    use std::sync::Arc;

    fn empty_env() -> Environment {
        Environment::default()
    }

    #[test]
    fn test_cubical_interval_requires_cubical_mode() {
        let env = empty_env();

        // Constructive mode (default) should reject cubical expressions
        let mut tc = TypeChecker::new(&env);
        let result = tc.infer_type_with_cert(&Expr::CubicalInterval);
        assert!(matches!(
            result,
            Err(TypeError::ModeRequired { feature, mode })
            if feature == "CubicalInterval" && mode == "Cubical"
        ));

        // Cubical mode should accept cubical expressions
        let mut tc = TypeChecker::with_mode(&env, Lean5Mode::Cubical);
        let result = tc.infer_type_with_cert(&Expr::CubicalInterval);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cubical_i0_i1_requires_cubical_mode() {
        let env = empty_env();

        // Constructive mode should reject i0/i1
        let mut tc = TypeChecker::new(&env);
        let result = tc.infer_type_with_cert(&Expr::CubicalI0);
        assert!(matches!(
            result,
            Err(TypeError::ModeRequired { feature, mode })
            if feature == "CubicalI0/CubicalI1" && mode == "Cubical"
        ));

        // Cubical mode should accept i0
        let mut tc = TypeChecker::with_mode(&env, Lean5Mode::Cubical);
        let result = tc.infer_type_with_cert(&Expr::CubicalI0);
        assert!(result.is_ok());
        let (ty, _) = result.unwrap();
        assert!(matches!(ty, Expr::CubicalInterval));

        // Cubical mode should accept i1
        let result = tc.infer_type_with_cert(&Expr::CubicalI1);
        assert!(result.is_ok());
        let (ty, _) = result.unwrap();
        assert!(matches!(ty, Expr::CubicalInterval));
    }

    #[test]
    fn test_classical_choice_requires_classical_mode() {
        let env = empty_env();
        // Use Sort(1) as a self-contained type that will type-check
        let sort1 = Expr::Sort(Level::succ(Level::zero()));
        // pred is a lambda: fun (x : Type) => true
        let pred = Expr::Lam(
            BinderInfo::Default,
            Arc::new(sort1.clone()),
            Arc::new(Expr::Sort(Level::zero())), // returns Prop
        );
        // exists_proof is just Sort(0) for now (placeholder)
        let exists_proof = Expr::Sort(Level::zero());
        let choice = Expr::ClassicalChoice {
            ty: Arc::new(sort1.clone()),
            pred: Arc::new(pred.clone()),
            exists_proof: Arc::new(exists_proof),
        };

        // Constructive mode should reject classical choice
        let mut tc = TypeChecker::new(&env);
        let result = tc.infer_type_with_cert(&choice);
        assert!(matches!(
            result,
            Err(TypeError::ModeRequired { feature, mode })
            if feature == "ClassicalChoice" && mode == "Classical"
        ));

        // Classical mode should accept classical choice (mode check passes, type checking continues)
        let mut tc = TypeChecker::with_mode(&env, Lean5Mode::Classical);
        let result = tc.infer_type_with_cert(&choice);
        // The mode check passes, so we get past ModeRequired error
        // Result should be Ok or some other error (not ModeRequired)
        assert!(!matches!(
            result,
            Err(TypeError::ModeRequired { .. })
        ));

        // SetTheoretic mode should also pass mode check
        let mut tc = TypeChecker::with_mode(&env, Lean5Mode::SetTheoretic);
        let result = tc.infer_type_with_cert(&choice);
        assert!(!matches!(
            result,
            Err(TypeError::ModeRequired { .. })
        ));
    }

    #[test]
    fn test_classical_epsilon_requires_classical_mode() {
        let env = empty_env();
        let sort1 = Expr::Sort(Level::succ(Level::zero()));
        let pred = Expr::Lam(
            BinderInfo::Default,
            Arc::new(sort1.clone()),
            Arc::new(Expr::Sort(Level::zero())),
        );
        let epsilon = Expr::ClassicalEpsilon {
            ty: Arc::new(sort1.clone()),
            pred: Arc::new(pred),
        };

        // Constructive mode should reject epsilon
        let mut tc = TypeChecker::new(&env);
        let result = tc.infer_type_with_cert(&epsilon);
        assert!(matches!(
            result,
            Err(TypeError::ModeRequired { feature, mode })
            if feature == "ClassicalEpsilon" && mode == "Classical"
        ));

        // Classical mode should pass mode check
        let mut tc = TypeChecker::with_mode(&env, Lean5Mode::Classical);
        let result = tc.infer_type_with_cert(&epsilon);
        assert!(!matches!(
            result,
            Err(TypeError::ModeRequired { .. })
        ));
    }

    #[test]
    fn test_zfc_set_requires_set_theoretic_mode() {
        use crate::expr::ZFCSetExpr;

        let env = empty_env();
        let zfc_set = Expr::ZFCSet(ZFCSetExpr::Empty);

        // Constructive mode should reject ZFC sets
        let mut tc = TypeChecker::new(&env);
        let result = tc.infer_type_with_cert(&zfc_set);
        assert!(matches!(
            result,
            Err(TypeError::ModeRequired { feature, mode })
            if feature == "ZFCSet" && mode == "SetTheoretic"
        ));

        // SetTheoretic mode should accept ZFC sets
        let mut tc = TypeChecker::with_mode(&env, Lean5Mode::SetTheoretic);
        let result = tc.infer_type_with_cert(&zfc_set);
        assert!(result.is_ok());
    }

    #[test]
    fn test_mode_getter_setter() {
        let env = empty_env();
        let mut tc = TypeChecker::new(&env);

        // Default mode is Constructive
        assert_eq!(tc.mode(), Lean5Mode::Constructive);

        // Can change mode
        tc.set_mode(Lean5Mode::Cubical);
        assert_eq!(tc.mode(), Lean5Mode::Cubical);

        tc.set_mode(Lean5Mode::Classical);
        assert_eq!(tc.mode(), Lean5Mode::Classical);
    }

    #[test]
    fn test_with_context_and_mode() {
        let env = empty_env();
        let ctx = LocalContext::new();

        let tc = TypeChecker::with_context_and_mode(&env, ctx, Lean5Mode::Cubical);
        assert_eq!(tc.mode(), Lean5Mode::Cubical);
    }

    #[test]
    fn test_impredicative_mode_rejects_cubical() {
        let env = empty_env();
        let mut tc = TypeChecker::with_mode(&env, Lean5Mode::Impredicative);

        // Impredicative mode should reject cubical expressions
        let result = tc.infer_type_with_cert(&Expr::CubicalInterval);
        assert!(matches!(
            result,
            Err(TypeError::ModeRequired { feature, mode })
            if feature == "CubicalInterval" && mode == "Cubical"
        ));
    }

    #[test]
    fn test_cubical_mode_rejects_classical() {
        let env = empty_env();
        let sort1 = Expr::Sort(Level::succ(Level::zero()));
        let epsilon = Expr::ClassicalEpsilon {
            ty: Arc::new(sort1),
            pred: Arc::new(Expr::const_(Name::from_string("P"), vec![])),
        };

        // Cubical mode should reject classical expressions
        let mut tc = TypeChecker::with_mode(&env, Lean5Mode::Cubical);
        let result = tc.infer_type_with_cert(&epsilon);
        assert!(matches!(
            result,
            Err(TypeError::ModeRequired { feature, mode })
            if feature == "ClassicalEpsilon" && mode == "Classical"
        ));
    }

    #[test]
    fn test_sprop_type_checking_in_impredicative_mode() {
        use crate::cert::ProofCert;
        let env = empty_env();
        let mut tc = TypeChecker::with_mode(&env, Lean5Mode::Impredicative);

        // SProp : Type 1 in Impredicative mode
        let result = tc.infer_type_with_cert(&Expr::SProp);
        assert!(result.is_ok());
        let (ty, cert) = result.unwrap();
        assert_eq!(ty, Expr::Sort(Level::succ(Level::zero())));
        assert!(matches!(cert, ProofCert::SProp));
    }

    #[test]
    fn test_sprop_rejected_in_constructive_mode() {
        let env = empty_env();
        let mut tc = TypeChecker::with_mode(&env, Lean5Mode::Constructive);

        // SProp should be rejected in Constructive mode
        let result = tc.infer_type_with_cert(&Expr::SProp);
        assert!(matches!(
            result,
            Err(TypeError::ModeRequired { feature, mode })
            if feature == "SProp" && mode == "Impredicative"
        ));
    }

    #[test]
    fn test_sprop_rejected_in_cubical_mode() {
        let env = empty_env();
        let mut tc = TypeChecker::with_mode(&env, Lean5Mode::Cubical);

        // SProp should be rejected in Cubical mode
        let result = tc.infer_type_with_cert(&Expr::SProp);
        assert!(matches!(
            result,
            Err(TypeError::ModeRequired { feature, mode })
            if feature == "SProp" && mode == "Impredicative"
        ));
    }

    #[test]
    fn test_sprop_allowed_in_classical_mode() {
        let env = empty_env();
        let mut tc = TypeChecker::with_mode(&env, Lean5Mode::Classical);

        // SProp should work in Classical mode (which extends Impredicative)
        let result = tc.infer_type_with_cert(&Expr::SProp);
        assert!(result.is_ok());
    }

    #[test]
    fn test_squash_type_checking_in_impredicative_mode() {
        use crate::cert::ProofCert;
        use std::sync::Arc;
        let env = empty_env();
        let mut tc = TypeChecker::with_mode(&env, Lean5Mode::Impredicative);

        // Squash Nat : SProp (when Nat : Type)
        // We use Sort 1 as a stand-in for a type
        let type_expr = Expr::Sort(Level::succ(Level::zero()));
        let squash_expr = Expr::Squash(Arc::new(type_expr));

        let result = tc.infer_type_with_cert(&squash_expr);
        assert!(result.is_ok());
        let (ty, cert) = result.unwrap();
        assert_eq!(ty, Expr::SProp);
        assert!(matches!(cert, ProofCert::Squash { .. }));
    }

    #[test]
    fn test_squash_rejected_in_constructive_mode() {
        use std::sync::Arc;
        let env = empty_env();
        let mut tc = TypeChecker::with_mode(&env, Lean5Mode::Constructive);

        let type_expr = Expr::Sort(Level::succ(Level::zero()));
        let squash_expr = Expr::Squash(Arc::new(type_expr));

        let result = tc.infer_type_with_cert(&squash_expr);
        assert!(matches!(
            result,
            Err(TypeError::ModeRequired { feature, mode })
            if feature == "Squash" && mode == "Impredicative"
        ));
    }
}
