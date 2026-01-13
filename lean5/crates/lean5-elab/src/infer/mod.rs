//! Type inference with metavariables
//!
//! Converts surface syntax to kernel expressions with:
//! - Named to de Bruijn conversion
//! - Metavariable creation for holes and implicit arguments
//! - Type inference
//! - Implicit argument insertion

use crate::instances::InstanceTable;
use crate::macro_integration::{expand_surface_macros, syntax_to_surface, MacroCtx};
use crate::unify::{MetaState, Unifier, UnifyResult};
use crate::ElabError;
use lean5_kernel::name::Name;
use lean5_kernel::{
    BinderInfo, CertError, CertVerifier, Environment, Expr, FVarId, Level, LocalContext, ProofCert,
    TypeChecker,
};
use lean5_macro::quotation::parse_quotation;
use lean5_parser::{
    LevelExpr, SurfaceArg, SurfaceBinder, SurfaceBinderInfo, SurfaceCtor, SurfaceDecl, SurfaceExpr,
    SurfaceField, SurfaceFieldAssign, SurfaceLit, SurfacePattern, UniverseExpr,
};
use std::collections::HashMap;

/// Elaboration context
pub struct ElabCtx<'a> {
    /// The kernel environment
    env: &'a Environment,
    /// Local bindings: name -> (fvar_id, type)
    locals: Vec<(String, FVarId, Expr)>,
    /// Universe parameter names
    universe_params: Vec<String>,
    /// Metavariable state (for unification)
    pub metas: MetaState,
    /// Next fresh free variable id
    next_fvar: u64,
    /// Next fresh universe parameter id
    next_universe: u64,
    /// Instance table for type class resolution
    instances: InstanceTable,
    /// Cache for instance resolution (tabled resolution)
    /// Maps normalized goal types to resolved instance expressions.
    /// This avoids re-resolving the same instance multiple times.
    instance_cache: HashMap<String, Expr>,
    /// Macro expansion context (built-ins + user-registered macro_rules)
    macro_ctx: MacroCtx,
}

impl<'a> ElabCtx<'a> {
    pub fn new(env: &'a Environment) -> Self {
        Self {
            env,
            locals: Vec::new(),
            universe_params: Vec::new(),
            metas: MetaState::new(),
            next_fvar: 0,
            next_universe: 0,
            instances: InstanceTable::new(),
            instance_cache: HashMap::new(),
            macro_ctx: MacroCtx::new(),
        }
    }

    /// Create with a pre-populated instance table
    pub fn with_instances(env: &'a Environment, instances: InstanceTable) -> Self {
        Self {
            env,
            locals: Vec::new(),
            universe_params: Vec::new(),
            metas: MetaState::new(),
            next_fvar: 0,
            next_universe: 0,
            instances,
            instance_cache: HashMap::new(),
            macro_ctx: MacroCtx::new(),
        }
    }

    /// Get mutable access to the instance table
    pub fn instances_mut(&mut self) -> &mut InstanceTable {
        &mut self.instances
    }

    /// Get read access to the instance table
    pub fn instances(&self) -> &InstanceTable {
        &self.instances
    }

    /// Get read access to the macro context
    pub fn macro_ctx(&self) -> &MacroCtx {
        &self.macro_ctx
    }

    /// Clear the instance resolution cache
    ///
    /// This should be called when the metavariable context changes significantly
    /// (e.g., after solving metavariables that might affect cached results).
    pub fn clear_instance_cache(&mut self) {
        self.instance_cache.clear();
    }

    /// Get instance cache statistics for debugging/profiling
    pub fn instance_cache_stats(&self) -> (usize, usize) {
        (self.instance_cache.len(), self.instance_cache.capacity())
    }

    /// Expand macros in a surface expression using the macro context.
    fn expand_macros(&mut self, surface: &SurfaceExpr) -> Result<SurfaceExpr, ElabError> {
        expand_surface_macros(&mut self.macro_ctx, surface)
            .map_err(|e| ElabError::MacroError(e.to_string()))
    }

    /// Normalize an expression for use as a cache key.
    ///
    /// This replaces metavariables with synthetic placeholders so that
    /// structurally similar goals (differing only in metavariable IDs)
    /// map to the same cache key.
    ///
    /// For example, `Add ?m1` and `Add ?m2` both normalize to `Add ?_0`.
    fn normalize_for_cache(&self, e: &Expr) -> String {
        use std::collections::HashMap as LocalMap;
        let mut meta_map: LocalMap<u64, usize> = LocalMap::new();
        let mut next_id = 0;

        fn normalize_expr(
            e: &Expr,
            meta_map: &mut LocalMap<u64, usize>,
            next_id: &mut usize,
        ) -> String {
            match e {
                Expr::BVar(idx) => format!("#{idx}"),
                Expr::FVar(fvar) => {
                    // Check if this is a metavariable (has high-bit tag)
                    if let Some(meta_id) = crate::unify::MetaState::from_fvar(*fvar) {
                        // Normalize metavariable IDs
                        let norm_id = *meta_map.entry(meta_id.0).or_insert_with(|| {
                            let id = *next_id;
                            *next_id += 1;
                            id
                        });
                        format!("?_{norm_id}")
                    } else {
                        // Regular free variable
                        format!("@{}", fvar.0)
                    }
                }
                Expr::Sort(lvl) => format!("Sort({})", normalize_level(lvl, meta_map, next_id)),
                Expr::Const(name, levels) => {
                    let lvls: Vec<_> = levels
                        .iter()
                        .map(|l| normalize_level(l, meta_map, next_id))
                        .collect();
                    if lvls.is_empty() {
                        format!("C:{name}")
                    } else {
                        format!("C:{}.[{}]", name, lvls.join(","))
                    }
                }
                Expr::App(f, arg) => {
                    format!(
                        "({} {})",
                        normalize_expr(f, meta_map, next_id),
                        normalize_expr(arg, meta_map, next_id)
                    )
                }
                Expr::Lam(bi, ty, body) => {
                    format!(
                        "(λ{:?} {} → {})",
                        bi,
                        normalize_expr(ty, meta_map, next_id),
                        normalize_expr(body, meta_map, next_id)
                    )
                }
                Expr::Pi(bi, ty, body) => {
                    format!(
                        "(Π{:?} {} → {})",
                        bi,
                        normalize_expr(ty, meta_map, next_id),
                        normalize_expr(body, meta_map, next_id)
                    )
                }
                Expr::Let(ty, val, body) => {
                    format!(
                        "(let {} := {} in {})",
                        normalize_expr(ty, meta_map, next_id),
                        normalize_expr(val, meta_map, next_id),
                        normalize_expr(body, meta_map, next_id)
                    )
                }
                Expr::Lit(lit) => format!("{lit:?}"),
                Expr::Proj(name, idx, e) => {
                    format!("{}.{}:{}", normalize_expr(e, meta_map, next_id), name, idx)
                }
                Expr::MData(_, inner) => {
                    // MData wraps another expression with metadata; normalize the inner
                    normalize_expr(inner, meta_map, next_id)
                }
                // Mode-specific expressions
                Expr::CubicalInterval => "CubicalI".to_string(),
                Expr::CubicalI0 => "I0".to_string(),
                Expr::CubicalI1 => "I1".to_string(),
                Expr::CubicalPath { .. } => "Path".to_string(),
                Expr::CubicalPathLam { .. } => "PathLam".to_string(),
                Expr::CubicalPathApp { .. } => "PathApp".to_string(),
                Expr::CubicalHComp { .. } => "HComp".to_string(),
                Expr::CubicalTransp { .. } => "Transp".to_string(),
                Expr::ClassicalChoice { .. } => "Choice".to_string(),
                Expr::ClassicalEpsilon { .. } => "Epsilon".to_string(),
                Expr::ZFCSet(_) => "ZFCSet".to_string(),
                Expr::ZFCMem { .. } => "ZFCMem".to_string(),
                Expr::ZFCComprehension { .. } => "ZFCComp".to_string(),
                Expr::SProp => "SProp".to_string(),
                Expr::Squash(inner) => {
                    format!("Squash({})", normalize_expr(inner, meta_map, next_id))
                }
            }
        }

        fn normalize_level(
            l: &Level,
            _meta_map: &mut LocalMap<u64, usize>,
            _next_id: &mut usize,
        ) -> String {
            // For levels, we use a simpler approach - just convert to string
            // Universe level metavariables are less common in instance resolution
            format!("{l:?}")
        }

        normalize_expr(e, &mut meta_map, &mut next_id)
    }

    /// Try to resolve an instance for a type class goal
    ///
    /// For example, given type `Add Nat`, this will search for a registered
    /// instance that implements `Add Nat`.
    ///
    /// Returns Some(instance_expr) if found, None otherwise.
    pub fn resolve_instance(&mut self, goal_ty: &Expr) -> Option<Expr> {
        self.resolve_instance_with_depth(goal_ty, 0)
    }

    fn resolve_instance_with_depth(&mut self, goal_ty: &Expr, depth: usize) -> Option<Expr> {
        use crate::instances::extract_class_app;

        const MAX_DEPTH: usize = 32;
        if depth > MAX_DEPTH {
            return None;
        }

        // Normalize the goal type with current metavariable assignments
        let goal_ty = self.whnf(goal_ty);
        let goal_ty = self.metas.instantiate(&goal_ty);

        // Generate cache key for this goal
        // We normalize the goal type so that structurally similar goals
        // (differing only in metavariable IDs) map to the same key.
        let cache_key = self.normalize_for_cache(&goal_ty);

        // Check cache for previously resolved instance
        // Note: We only use cached results if the goal is ground (no metavariables)
        // because metavariable-containing goals might resolve differently depending
        // on how those metavariables get solved later.
        let goal_is_ground = !self.has_metavars(&goal_ty);
        if goal_is_ground {
            if let Some(cached) = self.instance_cache.get(&cache_key) {
                return Some(cached.clone());
            }
        }

        // Extract the class name and arguments from the goal type
        let (class_name, goal_args) = extract_class_app(&goal_ty)?;

        // Check if this is a registered type class
        if !self.instances.is_class(&class_name) {
            return None;
        }

        // Get out-parameter and semi-out-parameter indices for this class
        let (out_params, _semi_out_params): (Vec<usize>, Vec<usize>) = self
            .instances
            .get_class(&class_name)
            .map(|info| (info.out_params.clone(), info.semi_out_params.clone()))
            .unwrap_or_default();

        // Note: semiOutParams are treated like regular parameters during unification
        // (they participate in Phase 1 bidirectional unification), but instances
        // promise to always fill them with concrete values. The only difference
        // from regular params is this "promise" - useful for tooling/error messages.

        // Clone instances to avoid borrow conflict with try_unify
        let instances: Vec<_> = self.instances.get_instances(&class_name).to_vec();

        // Try each instance in priority order
        for inst in instances {
            // Make a copy of the metavariable state to allow backtracking
            let metas_snapshot = self.metas.clone();

            // The instance expression and type (may contain implicit binders)
            let mut inst_expr = inst.expr.clone();
            let mut inst_type = self.whnf(&inst.type_);

            // Apply implicit parameters in the instance type (including dependent instances)
            let mut failed = false;
            while let Expr::Pi(bi, arg_ty, body_ty) = &inst_type {
                let instantiated_arg_ty = self.metas.instantiate(arg_ty);

                let arg = match bi {
                    BinderInfo::InstImplicit => {
                        // Resolve dependent instance argument
                        if let Some(resolved) =
                            self.resolve_instance_with_depth(&instantiated_arg_ty, depth + 1)
                        {
                            resolved
                        } else {
                            failed = true;
                            break;
                        }
                    }
                    // For other binders, create a metavariable to be solved by unification
                    _ => self.fresh_meta(instantiated_arg_ty.clone()),
                };

                // Apply the argument to the instance expression and type
                inst_expr = self.apply_instance_arg(inst_expr, &arg);
                inst_type = self.whnf(&self.metas.instantiate(&body_ty.instantiate(&arg)));
            }

            if failed {
                self.metas = metas_snapshot;
                continue;
            }

            // Extract class name and args from instance type after applying implicit binders
            if let Some((inst_class, inst_args)) = extract_class_app(&inst_type) {
                if inst_class != class_name {
                    self.metas = metas_snapshot;
                    continue;
                }

                // Try to unify the instance arguments with the goal arguments
                if inst_args.len() != goal_args.len() {
                    self.metas = metas_snapshot;
                    continue;
                }

                // Two-phase unification for out-parameters:
                // Phase 1: Unify non-out-parameters first (these must match)
                let mut unified = true;
                for (idx, (inst_arg, goal_arg)) in
                    inst_args.iter().zip(goal_args.iter()).enumerate()
                {
                    if !out_params.contains(&idx) {
                        // Non-out-parameter: must unify
                        if !self.try_unify(inst_arg, goal_arg) {
                            unified = false;
                            break;
                        }
                    }
                }

                if !unified {
                    self.metas = metas_snapshot;
                    continue;
                }

                // Phase 2: Unify out-parameters (these can be inferred from the instance)
                for (idx, (inst_arg, goal_arg)) in
                    inst_args.iter().zip(goal_args.iter()).enumerate()
                {
                    if out_params.contains(&idx) {
                        // Out-parameter: try to unify, direction is instance -> goal
                        if !self.try_unify(inst_arg, goal_arg) {
                            unified = false;
                            break;
                        }
                    }
                }

                if unified {
                    // Apply any metavariable substitutions and return the instance
                    let result = self.metas.instantiate(&inst_expr);

                    // Cache the result if the goal was ground
                    if goal_is_ground {
                        self.instance_cache
                            .insert(cache_key.clone(), result.clone());
                    }

                    return Some(result);
                }
            }

            // Unification failed, restore metavariable state
            self.metas = metas_snapshot;
        }

        None
    }

    /// Apply an argument to an instance expression, performing a simple
    /// beta-reduction when the expression is a lambda.
    fn apply_instance_arg(&self, func: Expr, arg: &Expr) -> Expr {
        match func {
            Expr::Lam(_, _, body) => body.instantiate(arg),
            _ => Expr::app(func, arg.clone()),
        }
    }

    /// Try to unify two expressions, returning true on success
    fn try_unify(&mut self, e1: &Expr, e2: &Expr) -> bool {
        let mut unifier = Unifier::new(&mut self.metas);
        matches!(unifier.unify(e1, e2), UnifyResult::Success)
    }

    /// Check if an expression contains any metavariables
    fn has_metavars(&self, e: &Expr) -> bool {
        match e {
            Expr::BVar(_) | Expr::Sort(_) | Expr::Lit(_) => false,
            Expr::FVar(fvar) => {
                // Check if this FVar is actually a metavariable
                MetaState::from_fvar(*fvar).is_some()
            }
            Expr::Const(_, levels) => levels.iter().any(|l| self.level_has_metavars(l)),
            Expr::App(f, arg) => self.has_metavars(f) || self.has_metavars(arg),
            Expr::Lam(_, ty, body) | Expr::Pi(_, ty, body) => {
                self.has_metavars(ty) || self.has_metavars(body)
            }
            Expr::Let(ty, val, body) => {
                self.has_metavars(ty) || self.has_metavars(val) || self.has_metavars(body)
            }
            Expr::Proj(_, _, e) => self.has_metavars(e),
            Expr::MData(_, inner) => self.has_metavars(inner),
            // Mode-specific expressions - conservatively say no metavars
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
            | Expr::SProp => false,
            Expr::Squash(inner) => self.has_metavars(inner),
        }
    }

    /// Check if a level contains any metavariables (universe level params)
    fn level_has_metavars(&self, l: &Level) -> bool {
        match l {
            // Note: We don't have level metavariables in our current implementation,
            // but if we add them, this would be where to check
            Level::Zero | Level::Param(_) => false,
            Level::Succ(l) => self.level_has_metavars(l),
            Level::Max(l1, l2) | Level::IMax(l1, l2) => {
                self.level_has_metavars(l1) || self.level_has_metavars(l2)
            }
        }
    }

    /// Set universe parameters for the current declaration
    #[must_use]
    pub fn with_universe_params(mut self, params: Vec<String>) -> Self {
        self.universe_params = params;
        self
    }

    /// Create a fresh free variable
    fn fresh_fvar(&mut self) -> FVarId {
        let id = FVarId(self.next_fvar);
        self.next_fvar += 1;
        id
    }

    /// Create a fresh metavariable
    pub fn fresh_meta(&mut self, ty: Expr) -> Expr {
        let id = self.metas.fresh(ty);
        Expr::FVar(MetaState::to_fvar(id))
    }

    /// Create a fresh universe parameter level
    ///
    /// This generates a new universe parameter name like `u_0`, `u_1`, etc.
    /// and returns a `Level::Param` with that name. The parameter is also
    /// added to `universe_params` so it's available for lookups.
    fn fresh_universe_param(&mut self) -> Level {
        let id = self.next_universe;
        self.next_universe += 1;
        let name = format!("u_{id}");
        self.universe_params.push(name.clone());
        Level::param(Name::from_string(&name))
    }

    /// Push a local binding
    fn push_local(&mut self, name: String, ty: Expr) -> FVarId {
        let fvar = self.fresh_fvar();
        self.locals.push((name, fvar, ty));
        fvar
    }

    /// Pop a local binding
    fn pop_local(&mut self) {
        self.locals.pop();
    }

    /// Build a LocalContext containing both locals and metavariables
    fn build_local_ctx(&self) -> LocalContext {
        let mut ctx = LocalContext::new();
        for (name, fvar, ty) in &self.locals {
            ctx.push_with_id(
                *fvar,
                Name::from_string(name),
                self.metas.instantiate(ty),
                BinderInfo::Default,
            );
        }

        for (meta_id, meta) in self.metas.iter() {
            let name = Name::from_string(&format!("?m{}", meta_id.0));
            ctx.push_with_id(
                MetaState::to_fvar(meta_id),
                name,
                self.metas.instantiate(&meta.ty),
                BinderInfo::Implicit,
            );
        }

        ctx
    }

    /// Look up a local by name
    fn lookup_local(&self, name: &str) -> Option<(FVarId, &Expr)> {
        // Search from innermost to outermost
        for (n, fvar, ty) in self.locals.iter().rev() {
            if n == name {
                return Some((*fvar, ty));
            }
        }
        None
    }

    /// Replace occurrences of a free variable with a constant applied to parameters.
    ///
    /// This is used when elaborating inductive types: the inductive type name is
    /// temporarily bound as a local during constructor elaboration, and then we need
    /// to replace references to it with the proper Const expression.
    ///
    /// When the fvar appears applied to arguments (e.g., `List α`), we check if
    /// those arguments are exactly the parameter fvars, and if so, replace with
    /// `Const(name) α` (keeping the original argument applications).
    fn replace_fvar_with_const(
        &self,
        expr: Expr,
        fvar_id: FVarId,
        const_name: &Name,
        _param_fvars: &[FVarId],
    ) -> Expr {
        match &expr {
            Expr::FVar(id) if *id == fvar_id => {
                // Direct reference to the inductive - replace with Const
                Expr::const_(const_name.clone(), vec![])
            }
            Expr::App(f, arg) => {
                // Check if function is (eventually) our fvar
                let new_f =
                    self.replace_fvar_with_const((**f).clone(), fvar_id, const_name, _param_fvars);
                let new_arg = self.replace_fvar_with_const(
                    (**arg).clone(),
                    fvar_id,
                    const_name,
                    _param_fvars,
                );
                Expr::app(new_f, new_arg)
            }
            Expr::Pi(bi, dom, cod) => {
                let new_dom = self.replace_fvar_with_const(
                    (**dom).clone(),
                    fvar_id,
                    const_name,
                    _param_fvars,
                );
                let new_cod = self.replace_fvar_with_const(
                    (**cod).clone(),
                    fvar_id,
                    const_name,
                    _param_fvars,
                );
                Expr::pi(*bi, new_dom, new_cod)
            }
            Expr::Lam(bi, dom, body) => {
                let new_dom = self.replace_fvar_with_const(
                    (**dom).clone(),
                    fvar_id,
                    const_name,
                    _param_fvars,
                );
                let new_body = self.replace_fvar_with_const(
                    (**body).clone(),
                    fvar_id,
                    const_name,
                    _param_fvars,
                );
                Expr::lam(*bi, new_dom, new_body)
            }
            Expr::Let(ty, val, body) => {
                let new_ty =
                    self.replace_fvar_with_const((**ty).clone(), fvar_id, const_name, _param_fvars);
                let new_val = self.replace_fvar_with_const(
                    (**val).clone(),
                    fvar_id,
                    const_name,
                    _param_fvars,
                );
                let new_body = self.replace_fvar_with_const(
                    (**body).clone(),
                    fvar_id,
                    const_name,
                    _param_fvars,
                );
                Expr::let_(new_ty, new_val, new_body)
            }
            // All other cases: no change needed
            _ => expr,
        }
    }

    /// Elaborate a surface expression to a kernel expression
    pub fn elaborate(&mut self, surface: &SurfaceExpr) -> Result<Expr, ElabError> {
        let expanded = self.expand_macros(surface)?;

        match &expanded {
            SurfaceExpr::Ident(_, name) => self.elab_ident(name),

            SurfaceExpr::Universe(_, univ) => self.elab_universe(univ),

            SurfaceExpr::App(_, func, args) => self.elab_app(func, args),

            // Pattern-matching lambda is elaborated the same way as regular lambda
            SurfaceExpr::Lambda(_, binders, body)
            | SurfaceExpr::PatternMatchLambda(_, binders, body) => self.elab_lambda(binders, body),

            SurfaceExpr::Pi(_, binders, body) => self.elab_pi(binders, body),

            SurfaceExpr::Arrow(_, from, to) => {
                let from_expr = self.elaborate(from)?;
                let to_expr = self.elaborate(to)?;
                Ok(Expr::arrow(from_expr, to_expr))
            }

            SurfaceExpr::Let(_, binder, val, body) => {
                // Elaborate type and value, avoiding double elaboration when type is inferred
                let (ty, val_expr) = if let Some(ty) = &binder.ty {
                    // Explicit type annotation provided
                    let ty_expr = self.elaborate(ty)?;
                    let val_expr = self.elaborate(val)?;
                    (ty_expr, val_expr)
                } else {
                    // Infer type from value - elaborate once and reuse
                    let val_expr = self.elaborate(val)?;
                    let ty = self.infer_type(&val_expr)?;
                    (ty, val_expr)
                };

                // Push local, elaborate body, then abstract
                let fvar = self.push_local(binder.name.clone(), ty.clone());
                let body_expr = self.elaborate(body)?;
                self.pop_local();

                // Abstract the fvar to a bvar
                let body_abs = body_expr.abstract_fvar(fvar);

                Ok(Expr::let_(ty, val_expr, body_abs))
            }

            SurfaceExpr::Lit(_, lit) => match lit {
                SurfaceLit::Nat(n) => Ok(Expr::nat_lit(*n)),
                SurfaceLit::String(s) => Ok(Expr::str_lit(s)),
            },

            SurfaceExpr::Paren(_, inner) => self.elaborate(inner),

            SurfaceExpr::Hole(_) => {
                // Create a fresh metavariable
                let ty_meta = self.fresh_meta(Expr::type_());
                let meta = self.fresh_meta(ty_meta);
                Ok(meta)
            }

            SurfaceExpr::Ascription(_, expr, ty) => {
                let ty_expr = self.elaborate(ty)?;
                let expr_val = self.elaborate(expr)?;

                // Infer the actual type of the expression
                let actual_ty = self.infer_type(&expr_val)?;

                // Unify the actual type with the expected type
                let mut unifier = Unifier::new(&mut self.metas);
                match unifier.unify(&actual_ty, &ty_expr) {
                    UnifyResult::Success => Ok(expr_val),
                    UnifyResult::Failure(msg) => Err(ElabError::TypeMismatch {
                        expected: format!("{ty_expr:?}"),
                        actual: format!("{actual_ty:?} ({msg})"),
                    }),
                    UnifyResult::Stuck => {
                        // Unification is stuck - can happen with unresolved metavariables
                        // For now, accept it but leave metavariables to be resolved later
                        Ok(expr_val)
                    }
                }
            }

            SurfaceExpr::If(_, cond, then_br, else_br) => {
                // Desugar to: Bool.casesOn (motive := fun _ => _) cond else_br then_br
                // For now, just elaborate the condition and branches
                let cond_expr = self.elaborate(cond)?;
                let then_expr = self.elaborate(then_br)?;
                let else_expr = self.elaborate(else_br)?;

                // Create if-then-else application using ite function
                let if_name = Name::from_string("ite");
                let ite = Expr::const_(if_name, vec![]);
                Ok(Expr::app(
                    Expr::app(Expr::app(ite, cond_expr), then_expr),
                    else_expr,
                ))
            }

            SurfaceExpr::Match(_, scrutinee, arms) => {
                // match e with | pat1 => body1 | pat2 => body2 | ...
                // Desugars to: T.casesOn (motive) e alt1 alt2 ...
                //
                // For simple cases with variable/wildcard patterns, we use let bindings.
                // For constructor patterns, we build casesOn applications.
                //
                // Currently supported:
                // - Single arm with Var/Wildcard pattern: let binding
                // - Multiple arms with Ctor patterns: casesOn
                // - Mixed patterns with fallback wildcard: casesOn with default
                if arms.is_empty() {
                    return Err(ElabError::NotImplemented("match with no arms".to_string()));
                }

                let scrutinee_expr = self.elaborate(scrutinee)?;
                let scrutinee_ty = self.infer_type(&scrutinee_expr)?;

                // Check for simple single-arm cases first
                if arms.len() == 1 {
                    let arm = &arms[0];
                    match &arm.pattern {
                        SurfacePattern::Var(name) => {
                            // match e with | x => body  ==  let x := e in body
                            let fvar = self.push_local(name.clone(), scrutinee_ty.clone());
                            let body_expr = self.elaborate(&arm.body)?;
                            self.pop_local();
                            let body_abs = body_expr.abstract_fvar(fvar);
                            return Ok(Expr::let_(scrutinee_ty, scrutinee_expr, body_abs));
                        }
                        SurfacePattern::Wildcard => {
                            // match e with | _ => body  ==  let _ := e in body
                            let body_expr = self.elaborate(&arm.body)?;
                            return Ok(Expr::let_(scrutinee_ty, scrutinee_expr, body_expr));
                        }
                        _ => {}
                    }
                }

                // For multiple arms or constructor patterns, build casesOn
                let type_name = self.get_type_name(&scrutinee_ty)?;
                let cases_on_name = Name::from_string(&format!("{type_name}.casesOn"));
                let cases_on = Expr::const_(cases_on_name, vec![]);

                // Elaborate the first arm body to determine the result type
                // We need to bind any pattern variables first
                let branch_ty = {
                    let first_arm = &arms[0];
                    match &first_arm.pattern {
                        SurfacePattern::Var(name) => {
                            let fvar = self.push_local(name.clone(), scrutinee_ty.clone());
                            let body = self.elaborate(&first_arm.body)?;
                            let ty = self.infer_type(&body)?;
                            self.pop_local();
                            let _ = fvar; // Used for binding, not needed in type
                            ty
                        }
                        SurfacePattern::Wildcard => {
                            let body = self.elaborate(&first_arm.body)?;
                            self.infer_type(&body)?
                        }
                        _ => {
                            // For constructor patterns, elaborate without special binding
                            // (constructor args would need their own types from the type definition)
                            let body = self.elaborate(&first_arm.body)?;
                            self.infer_type(&body)?
                        }
                    }
                };

                // Build motive: (fun _ : T => ResultType)
                let motive = Expr::lam(BinderInfo::Default, scrutinee_ty.clone(), branch_ty);

                // Build: casesOn motive scrutinee
                let mut result = Expr::app(Expr::app(cases_on, motive), scrutinee_expr);

                // Add case alternatives
                // For now, we handle arms in order, treating each as a constructor case
                for arm in arms {
                    let alt = match &arm.pattern {
                        SurfacePattern::Var(name) => {
                            // Variable binds the whole value - create a lambda
                            // Pattern: | x => body becomes (fun x : T => body)
                            let fvar = self.push_local(name.clone(), scrutinee_ty.clone());
                            let arm_body = self.elaborate(&arm.body)?;
                            self.pop_local();
                            let body_abs = arm_body.abstract_fvar(fvar);
                            Expr::lam(BinderInfo::Default, scrutinee_ty.clone(), body_abs)
                        }
                        SurfacePattern::Wildcard => {
                            // Wildcard is a catch-all - just elaborate body
                            self.elaborate(&arm.body)?
                        }
                        SurfacePattern::Ctor(_, sub_pats) => {
                            // Constructor pattern: wrap body in lambdas for args
                            let arm_body = self.elaborate(&arm.body)?;
                            if sub_pats.is_empty() {
                                arm_body
                            } else {
                                self.wrap_pattern_lambdas(sub_pats, arm_body)?
                            }
                        }
                        SurfacePattern::As(name, inner_pat) => {
                            // As pattern: bind name and also match inner
                            let fvar = self.push_local(name.clone(), scrutinee_ty.clone());
                            let arm_body = self.elaborate(&arm.body)?;
                            self.pop_local();
                            let body_abs = arm_body.abstract_fvar(fvar);
                            match inner_pat.as_ref() {
                                SurfacePattern::Ctor(_, sub_pats) if !sub_pats.is_empty() => {
                                    self.wrap_pattern_lambdas(sub_pats, body_abs)?
                                }
                                _ => Expr::lam(BinderInfo::Default, scrutinee_ty.clone(), body_abs),
                            }
                        }
                        _ => {
                            return Err(ElabError::NotImplemented(format!(
                                "match arm pattern: {:?}",
                                arm.pattern
                            )));
                        }
                    };
                    result = Expr::app(result, alt);
                }

                Ok(result)
            }

            SurfaceExpr::OutParam(_, inner) => {
                // outParam is just a marker for type class parameters
                // During normal elaboration, we just elaborate the inner type
                self.elaborate(inner)
            }

            SurfaceExpr::SemiOutParam(_, inner) => {
                // semiOutParam is also just a marker for type class parameters
                // During normal elaboration, we just elaborate the inner type
                self.elaborate(inner)
            }

            SurfaceExpr::Proj(_, expr, proj) => {
                let expr_val = self.elaborate(expr)?;
                let (struct_name, num_fields) = self.resolve_projection_target(&expr_val)?;

                match proj {
                    lean5_parser::Projection::Named(name) => {
                        let field_name = Name::from_string(name);
                        let idx = self
                            .env
                            .get_structure_field_index(&struct_name, &field_name)
                            .ok_or_else(|| ElabError::UnknownProjectionField {
                                struct_name: struct_name.clone(),
                                field: name.clone(),
                            })?;

                        Ok(Expr::proj(struct_name, idx, expr_val))
                    }
                    lean5_parser::Projection::Index(idx) => {
                        if *idx >= num_fields {
                            return Err(ElabError::ProjectionIndexOutOfBounds {
                                struct_name: struct_name.clone(),
                                idx: *idx,
                                field_count: num_fields,
                            });
                        }

                        Ok(Expr::proj(struct_name, *idx, expr_val))
                    }
                }
            }

            SurfaceExpr::UniverseInst(_, expr, _levels) => {
                // Universe instantiation: Foo.{u v}
                // For now, elaborate the inner expression and ignore the explicit universe levels
                // A full implementation would substitute the universe levels appropriately
                self.elaborate(expr)
            }

            SurfaceExpr::NamedArg(_, name, value) => {
                // Named argument: (name := expr)
                // This should typically appear inside an App, but if it appears standalone,
                // we just elaborate the value and ignore the name for now
                // The elaborator should handle named args in the App case
                let _ = name; // Name is used when this appears as an argument
                self.elaborate(value)
            }

            SurfaceExpr::SyntaxQuote(_, content) => {
                let quoted = parse_quotation(&format!("`{content}"))
                    .map_err(|e| ElabError::MacroError(e.to_string()))?;
                let surface = syntax_to_surface(&quoted.syntax).ok_or_else(|| {
                    ElabError::MacroError(
                        "could not convert syntax quotation to surface expression".into(),
                    )
                })?;
                self.elaborate(&surface)
            }

            SurfaceExpr::Explicit(_, inner) => {
                // Explicit application: @f
                // This disables implicit argument insertion for the inner expression.
                // A full implementation would set a flag in the elaborator state to
                // prevent auto-insertion of implicit arguments. For now, we simply
                // elaborate the inner expression - the semantic effect requires
                // integration with the application elaboration.
                self.elaborate(inner)
            }

            SurfaceExpr::LetRec(_, binder, val, body) => {
                // Recursive let binding: let rec f := v in e
                // For now, elaborate like a regular let but note that the value
                // may reference the binding itself (recursion).
                // A full implementation would use a fix combinator or similar.
                let (ty, val_expr) = if let Some(ty) = &binder.ty {
                    let ty_expr = self.elaborate(ty)?;
                    let val_expr = self.elaborate(val)?;
                    (ty_expr, val_expr)
                } else {
                    let val_expr = self.elaborate(val)?;
                    let ty = self.infer_type(&val_expr)?;
                    (ty, val_expr)
                };

                let fvar = self.push_local(binder.name.clone(), ty.clone());
                let body_expr = self.elaborate(body)?;
                self.pop_local();

                let body_abs = body_expr.abstract_fvar(fvar);
                Ok(Expr::let_(ty, val_expr, body_abs))
            }

            SurfaceExpr::IfLet(_, pat, scrutinee, then_br, else_br) => {
                // if let pat := scrutinee then then_br else else_br
                // Desugars to: match scrutinee with | pat => then_br | _ => else_br
                //
                // For now, we handle simple patterns directly:
                // - Var(x): let x := scrutinee in then_br (else_br is unreachable for Var)
                // - Wildcard: then_br (scrutinee evaluated for effects)
                // - Ctor patterns: construct casesOn application
                let scrutinee_expr = self.elaborate(scrutinee)?;

                match pat {
                    SurfacePattern::Var(name) => {
                        // Variable pattern always matches - bind scrutinee to name
                        // if let x := e then t else f  ==  let x := e in t
                        let scrutinee_ty = self.infer_type(&scrutinee_expr)?;
                        let fvar = self.push_local(name.clone(), scrutinee_ty.clone());
                        let then_expr = self.elaborate(then_br)?;
                        self.pop_local();
                        let body_abs = then_expr.abstract_fvar(fvar);
                        Ok(Expr::let_(scrutinee_ty, scrutinee_expr, body_abs))
                    }
                    SurfacePattern::Wildcard => {
                        // Wildcard always matches - evaluate scrutinee, return then
                        // if let _ := e then t else f  ==  let _ := e in t
                        let scrutinee_ty = self.infer_type(&scrutinee_expr)?;
                        let then_expr = self.elaborate(then_br)?;
                        // Create a let binding that ignores the value
                        Ok(Expr::let_(scrutinee_ty, scrutinee_expr, then_expr))
                    }
                    SurfacePattern::Ctor(ctor_name, sub_pats) => {
                        // Constructor pattern: need to check if scrutinee matches
                        // Generate: T.casesOn scrutinee (fun args => else_br) ... (fun args => then_br) ...
                        // For now, construct a simplified form using Option.casesOn-like pattern
                        let then_expr = self.elaborate(then_br)?;
                        let else_expr = self.elaborate(else_br)?;

                        // Create casesOn call structure
                        // T.casesOn (motive) scrutinee alt1 alt2 ...
                        let scrutinee_ty = self.infer_type(&scrutinee_expr)?;
                        let type_name = self.get_type_name(&scrutinee_ty)?;
                        let cases_on_name = Name::from_string(&format!("{type_name}.casesOn"));
                        let cases_on = Expr::const_(cases_on_name, vec![]);

                        // Build motive (returns type of branches)
                        let branch_ty = self.infer_type(&then_expr)?;
                        let motive =
                            Expr::lam(BinderInfo::Default, scrutinee_ty.clone(), branch_ty);

                        // For a constructor pattern like `some(x)` on `Option A`:
                        // We need to provide alternatives for each constructor in order
                        // For now, simplified: we use the ctor_name to determine which branch to use
                        let ctor_branch = if sub_pats.is_empty() {
                            then_expr.clone()
                        } else {
                            // Wrap in lambdas for constructor arguments
                            self.wrap_pattern_lambdas(sub_pats, then_expr)?
                        };

                        // Build casesOn application
                        // casesOn motive scrutinee <alternatives based on ctor position>
                        let result = Expr::app(
                            Expr::app(Expr::app(cases_on, motive), scrutinee_expr),
                            else_expr, // This is simplified - real impl needs proper ctor ordering
                        );
                        let result = Expr::app(result, ctor_branch);

                        // Mark with metadata indicating pattern source for debugging
                        let _ = ctor_name; // Note: used to determine which case to select
                        Ok(result)
                    }
                    _ => {
                        // Complex patterns (As, Or, Lit, NumeralAdd) - not yet supported
                        Err(ElabError::NotImplemented(format!(
                            "if-let with complex pattern: {pat:?}"
                        )))
                    }
                }
            }

            SurfaceExpr::IfDecidable(_, witness_name, prop, then_br, else_br) => {
                // if h : p then t else e
                // Desugars to: dite p (fun h : p => t) (fun h : ¬p => e)
                //
                // dite (Decidable If-Then-Else) has type:
                // dite : {α : Sort u} → (p : Prop) → [Decidable p] →
                //        (p → α) → (¬p → α) → α
                let prop_expr = self.elaborate(prop)?;

                // Create the then branch: (fun h : p => t)
                let then_fvar = self.push_local(witness_name.clone(), prop_expr.clone());
                let then_expr = self.elaborate(then_br)?;
                self.pop_local();
                let then_lambda = Expr::lam(
                    BinderInfo::Default,
                    prop_expr.clone(),
                    then_expr.abstract_fvar(then_fvar),
                );

                // Create the else branch: (fun h : ¬p => e)
                // ¬p = p → False = Pi(Default, p, False)
                let not_prop = Expr::pi(
                    BinderInfo::Default,
                    prop_expr.clone(),
                    Expr::const_(Name::from_string("False"), vec![]),
                );
                let else_fvar = self.push_local(witness_name.clone(), not_prop.clone());
                let else_expr = self.elaborate(else_br)?;
                self.pop_local();
                let else_lambda = Expr::lam(
                    BinderInfo::Default,
                    not_prop,
                    else_expr.abstract_fvar(else_fvar),
                );

                // Build: dite p then_lambda else_lambda
                // Note: The Decidable instance is resolved implicitly by type class resolution
                let dite = Expr::const_(Name::from_string("dite"), vec![]);
                let result = Expr::app(
                    Expr::app(Expr::app(dite, prop_expr), then_lambda),
                    else_lambda,
                );
                Ok(result)
            }
        }
    }

    fn elab_ident(&self, name: &str) -> Result<Expr, ElabError> {
        // First check locals
        if let Some((fvar, _ty)) = self.lookup_local(name) {
            return Ok(Expr::fvar(fvar));
        }

        // Then check constants in environment
        let const_name = Name::from_string(name);
        if let Some(info) = self.env.get_const(&const_name) {
            let levels: Vec<Level> = info
                .level_params
                .iter()
                .map(|_| Level::zero()) // Default to level 0 for now
                .collect();
            return Ok(Expr::const_(const_name, levels));
        }

        Err(ElabError::UnknownIdent(name.to_string()))
    }

    fn elab_universe(&mut self, univ: &UniverseExpr) -> Result<Expr, ElabError> {
        let level = match univ {
            UniverseExpr::Prop => Level::zero(),
            UniverseExpr::Type => Level::succ(Level::zero()),
            UniverseExpr::TypeLevel(level_expr) => {
                let l = self.elab_level(level_expr)?;
                Level::succ(l)
            }
            UniverseExpr::Sort(level_expr) => self.elab_level(level_expr)?,
            // Sort without explicit level: create a fresh universe parameter
            UniverseExpr::SortImplicit => self.fresh_universe_param(),
        };
        Ok(Expr::sort(level))
    }

    fn elab_level(&self, level: &LevelExpr) -> Result<Level, ElabError> {
        match level {
            LevelExpr::Lit(n) => {
                let mut l = Level::zero();
                for _ in 0..*n {
                    l = Level::succ(l);
                }
                Ok(l)
            }
            LevelExpr::Param(name) => {
                // Check if it's a known universe parameter
                if self.universe_params.contains(name) {
                    Ok(Level::param(Name::from_string(name)))
                } else {
                    Err(ElabError::UnknownIdent(format!("universe {name}")))
                }
            }
            LevelExpr::Succ(inner) => {
                let l = self.elab_level(inner)?;
                Ok(Level::succ(l))
            }
            LevelExpr::Max(l1, l2) => {
                let l1 = self.elab_level(l1)?;
                let l2 = self.elab_level(l2)?;
                Ok(Level::max(l1, l2))
            }
            LevelExpr::IMax(l1, l2) => {
                let l1 = self.elab_level(l1)?;
                let l2 = self.elab_level(l2)?;
                Ok(Level::imax(l1, l2))
            }
        }
    }

    fn elab_lambda(
        &mut self,
        binders: &[SurfaceBinder],
        body: &SurfaceExpr,
    ) -> Result<Expr, ElabError> {
        if binders.is_empty() {
            return self.elaborate(body);
        }

        let binder = &binders[0];
        let ty = if let Some(ty) = &binder.ty {
            self.elaborate(ty)?
        } else {
            // Create a fresh metavariable for the type
            self.fresh_meta(Expr::type_())
        };

        let bi = convert_binder_info(binder.info);
        let fvar = self.push_local(binder.name.clone(), ty.clone());

        let inner = self.elab_lambda(&binders[1..], body)?;

        self.pop_local();

        // Abstract the fvar to a bvar
        let inner_abs = inner.abstract_fvar(fvar);
        Ok(Expr::lam(bi, ty, inner_abs))
    }

    fn elab_pi(
        &mut self,
        binders: &[SurfaceBinder],
        body: &SurfaceExpr,
    ) -> Result<Expr, ElabError> {
        if binders.is_empty() {
            return self.elaborate(body);
        }

        let binder = &binders[0];
        let ty = if let Some(ty) = &binder.ty {
            self.elaborate(ty)?
        } else {
            // Pi types require explicit types
            return Err(ElabError::CannotInfer);
        };

        let bi = convert_binder_info(binder.info);
        let fvar = self.push_local(binder.name.clone(), ty.clone());

        let inner = self.elab_pi(&binders[1..], body)?;

        self.pop_local();

        // Abstract the fvar to a bvar
        let inner_abs = inner.abstract_fvar(fvar);
        Ok(Expr::pi(bi, ty, inner_abs))
    }

    /// Infer the type of an expression (delegating to kernel)
    fn infer_type(&self, expr: &Expr) -> Result<Expr, ElabError> {
        let mut tc = TypeChecker::with_context(self.env, self.build_local_ctx());
        let instantiated = self.metas.instantiate(expr);
        tc.infer_type(&instantiated)
            .map(|ty| self.metas.instantiate(&ty))
            .map_err(|e| ElabError::TypeMismatch {
                expected: "valid type".to_string(),
                actual: format!("{e:?}"),
            })
    }

    /// Compute weak-head normal form of an expression
    fn whnf(&self, expr: &Expr) -> Expr {
        let tc = TypeChecker::with_context(self.env, self.build_local_ctx());
        tc.whnf(&self.metas.instantiate(expr))
    }

    /// Resolve the structure name and field count for a projection target.
    /// Ensures the target type reduces to a single-constructor inductive.
    fn resolve_projection_target(&self, expr: &Expr) -> Result<(Name, u32), ElabError> {
        let expr_ty = self.infer_type(expr)?;
        let expr_ty_whnf = self.whnf(&expr_ty);

        let struct_name = match expr_ty_whnf.get_app_fn() {
            Expr::Const(name, _) => name.clone(),
            other => return Err(ElabError::InvalidProjectionTarget(format!("{other:?}"))),
        };

        let ind = self
            .env
            .get_inductive(&struct_name)
            .ok_or_else(|| ElabError::InvalidProjectionTarget(format!("{expr_ty_whnf:?}")))?;

        if ind.constructor_names.len() != 1 {
            return Err(ElabError::InvalidProjectionTarget(format!(
                "{expr_ty_whnf:?}"
            )));
        }

        let ctor_name = &ind.constructor_names[0];
        let ctor = self.env.get_constructor(ctor_name).ok_or_else(|| {
            ElabError::InvalidProjectionTarget(format!("missing constructor {ctor_name:?}"))
        })?;

        Ok((struct_name, ctor.num_fields))
    }

    /// Create a certificate verifier with the current local context pre-registered.
    ///
    /// This enables verification of elaborated expressions that contain free variables
    /// from the elaboration context. The verifier is initialized with all locals and
    /// metavariables from this context.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut ctx = ElabCtx::new(&env);
    /// let expr = ctx.elaborate(&surface)?;
    /// let (ty, cert) = ctx.infer_type_with_cert(&expr)?;
    /// let verifier = ctx.create_cert_verifier()?;
    /// verifier.verify(&cert, &expr)?;
    /// ```
    pub fn create_cert_verifier(&self) -> Result<CertVerifier<'a>, CertError> {
        let mut verifier = CertVerifier::new(self.env);
        verifier.register_local_context(&self.build_local_ctx())?;
        Ok(verifier)
    }

    /// Infer the type of an expression with a proof certificate.
    ///
    /// This is the certified variant of `infer_type` - it returns both the inferred
    /// type and a proof certificate that can be independently verified.
    ///
    /// The certificate can be verified using a `CertVerifier` created from
    /// `create_cert_verifier()`.
    pub fn infer_type_with_cert(&self, expr: &Expr) -> Result<(Expr, ProofCert), ElabError> {
        let mut tc = TypeChecker::with_context(self.env, self.build_local_ctx());
        let instantiated = self.metas.instantiate(expr);
        tc.infer_type_with_cert(&instantiated)
            .map(|(ty, cert)| (self.metas.instantiate(&ty), cert))
            .map_err(|e| ElabError::TypeMismatch {
                expected: "valid type".to_string(),
                actual: format!("{e:?}"),
            })
    }

    /// Elaborate and verify an expression with certificates.
    ///
    /// This combines elaboration, type inference, and certificate verification
    /// into a single operation. Returns the elaborated expression, its type,
    /// and the proof certificate.
    ///
    /// This is useful for verified elaboration pipelines where you want to ensure
    /// the elaborated expression type-checks correctly.
    pub fn elaborate_and_verify(
        &mut self,
        surface: &SurfaceExpr,
    ) -> Result<(Expr, Expr, ProofCert), ElabError> {
        let expr = self.elaborate(surface)?;
        let (ty, cert) = self.infer_type_with_cert(&expr)?;

        // Verify the certificate
        let mut verifier = self
            .create_cert_verifier()
            .map_err(|e| ElabError::TypeMismatch {
                expected: "valid certificate verifier".to_string(),
                actual: format!("{e:?}"),
            })?;

        verifier
            .verify(&cert, &expr)
            .map_err(|e| ElabError::TypeMismatch {
                expected: "certificate verification".to_string(),
                actual: format!("{e:?}"),
            })?;

        Ok((expr, ty, cert))
    }

    /// Check if a binder info requires implicit argument insertion
    fn is_implicit_binder(bi: BinderInfo) -> bool {
        matches!(
            bi,
            BinderInfo::Implicit | BinderInfo::StrictImplicit | BinderInfo::InstImplicit
        )
    }

    /// Insert implicit arguments for a function application.
    /// Returns the function with all implicit arguments applied, and the remaining function type.
    ///
    /// For InstImplicit binders `[inst : T]`, this attempts instance resolution.
    /// If resolution fails, falls back to creating a metavariable (which may be
    /// resolved later by unification).
    fn insert_implicit_args(&mut self, func: Expr, func_type: &Expr) -> (Expr, Expr) {
        let mut result = func;
        let mut ty = self.whnf(func_type);

        loop {
            match &ty {
                Expr::Pi(bi, arg_ty, body_ty) if Self::is_implicit_binder(*bi) => {
                    let arg_ty_inst = self.metas.instantiate(arg_ty);

                    // For InstImplicit, try instance resolution first
                    let arg = if *bi == BinderInfo::InstImplicit {
                        if let Some(inst) = self.resolve_instance(&arg_ty_inst) {
                            inst
                        } else {
                            // Fall back to metavariable if no instance found
                            self.fresh_meta(arg_ty_inst)
                        }
                    } else {
                        // For regular implicit/strict implicit, use metavariable
                        self.fresh_meta(arg_ty_inst)
                    };

                    result = Expr::app(result, arg.clone());
                    // Instantiate the body type with the argument
                    ty = self.whnf(&self.metas.instantiate(&body_ty.instantiate(&arg)));
                }
                _ => break,
            }
        }

        (result, ty)
    }

    /// Elaborate a function application with implicit argument insertion.
    ///
    /// For a function `f : {A : Type} → (x : A) → A` and call `f 42`:
    /// 1. Elaborate `f` to get its type
    /// 2. Insert metavariables for implicit arguments (A becomes ?m)
    /// 3. Elaborate explicit arguments and unify types
    fn elab_app(&mut self, func: &SurfaceExpr, args: &[SurfaceArg]) -> Result<Expr, ElabError> {
        // Elaborate the function
        let func_expr = self.elaborate(func)?;

        // Try to infer the function's type to know about implicit arguments
        // If we can't infer it (e.g., function is a metavariable), fall back to simple elaboration
        let func_type_result = self.infer_type(&func_expr);

        if let Ok(func_type) = func_type_result {
            // Insert leading implicit arguments
            let (mut result, mut current_type) = self.insert_implicit_args(func_expr, &func_type);

            // Process each explicit argument
            for arg in args {
                // Check the current type to see if we need more implicit args
                current_type = self.whnf(&current_type);

                // Extract binder info and types to avoid borrow issues
                let type_info = match &current_type {
                    Expr::Pi(bi, arg_ty, body_ty) => {
                        Some((*bi, arg_ty.as_ref().clone(), body_ty.as_ref().clone()))
                    }
                    _ => None,
                };

                if let Some((bi, expected_arg_ty, body_ty)) = type_info {
                    // If user provided an explicit argument but function expects implicit,
                    // keep inserting metavariables until we reach an explicit argument
                    let mut local_bi = bi;
                    let mut local_arg_ty = expected_arg_ty;
                    let mut local_body_ty = body_ty;

                    while !arg.explicit && Self::is_implicit_binder(local_bi) {
                        // Insert a metavariable for the implicit argument
                        let meta = self.fresh_meta(self.metas.instantiate(&local_arg_ty));
                        result = Expr::app(result, meta.clone());
                        current_type = self.metas.instantiate(&local_body_ty.instantiate(&meta));

                        // Insert any additional trailing implicit arguments
                        let (new_result, new_type) =
                            self.insert_implicit_args(result, &current_type);
                        result = new_result;
                        current_type = self.whnf(&new_type);

                        // Check if we need to continue inserting
                        if let Expr::Pi(next_bi, next_arg_ty, next_body_ty) = &current_type {
                            local_bi = *next_bi;
                            local_arg_ty = next_arg_ty.as_ref().clone();
                            local_body_ty = next_body_ty.as_ref().clone();
                        } else {
                            break;
                        }
                    }

                    // Now elaborate the actual argument
                    let arg_expr = self.elaborate(&arg.expr)?;
                    let arg_type = self.infer_type(&arg_expr)?;
                    let expected_arg_ty = self.metas.instantiate(&local_arg_ty);

                    match Unifier::new(&mut self.metas).unify(&arg_type, &expected_arg_ty) {
                        UnifyResult::Success => {}
                        UnifyResult::Failure(msg) => {
                            return Err(ElabError::TypeMismatch {
                                expected: format!("{expected_arg_ty:?}"),
                                actual: msg,
                            });
                        }
                        UnifyResult::Stuck => {
                            return Err(ElabError::CannotInfer);
                        }
                    }

                    result = Expr::app(result, arg_expr.clone());

                    // Update the type for the next iteration
                    // Need to get fresh body_ty since we may have consumed it
                    current_type = if let Expr::Pi(_, _, body) = &current_type {
                        self.metas.instantiate(&body.instantiate(&arg_expr))
                    } else {
                        current_type.clone() // Already not a Pi, just keep it
                    };

                    // Insert any trailing implicit arguments before the next explicit arg
                    let (new_result, new_type) = self.insert_implicit_args(result, &current_type);
                    result = new_result;
                    current_type = new_type;
                } else {
                    // Function type is not a Pi - just apply the argument anyway
                    // (type checking will catch errors later)
                    let arg_expr = self.elaborate(&arg.expr)?;
                    result = Expr::app(result, arg_expr);
                }
            }

            Ok(self.metas.instantiate(&result))
        } else {
            // Fallback: simple elaboration without implicit insertion
            let mut result = func_expr;
            for arg in args {
                let arg_expr = self.elaborate(&arg.expr)?;
                result = Expr::app(result, arg_expr);
            }
            Ok(self.metas.instantiate(&result))
        }
    }

    /// Elaborate a surface declaration to a kernel declaration
    pub fn elab_decl(&mut self, decl: &SurfaceDecl) -> Result<ElabResult, ElabError> {
        match decl {
            SurfaceDecl::Def {
                name,
                universe_params,
                binders,
                ty,
                val,
                ..
            } => {
                // Set universe params
                self.universe_params = universe_params.clone();

                // Elaborate binders as pi types around the type, lambdas around the value
                let (ty_expr, val_expr) = self.elab_def_body(binders, ty.as_deref(), val)?;

                Ok(ElabResult::Definition {
                    name: Name::from_string(name),
                    universe_params: universe_params
                        .iter()
                        .map(|s| Name::from_string(s))
                        .collect(),
                    ty: ty_expr,
                    val: val_expr,
                })
            }

            SurfaceDecl::Theorem {
                name,
                universe_params,
                binders,
                ty,
                proof,
                ..
            } => {
                self.universe_params = universe_params.clone();
                let (ty_expr, proof_expr) = self.elab_def_body(binders, Some(ty), proof)?;

                Ok(ElabResult::Theorem {
                    name: Name::from_string(name),
                    universe_params: universe_params
                        .iter()
                        .map(|s| Name::from_string(s))
                        .collect(),
                    ty: ty_expr,
                    proof: proof_expr,
                })
            }

            SurfaceDecl::Axiom {
                name,
                universe_params,
                binders,
                ty,
                ..
            } => {
                self.universe_params = universe_params.clone();
                let ty_expr = self.elab_axiom_type(binders, ty)?;

                Ok(ElabResult::Axiom {
                    name: Name::from_string(name),
                    universe_params: universe_params
                        .iter()
                        .map(|s| Name::from_string(s))
                        .collect(),
                    ty: ty_expr,
                })
            }

            SurfaceDecl::Inductive {
                name,
                universe_params,
                binders,
                ty,
                ctors,
                deriving,
                ..
            } => {
                self.universe_params = universe_params.clone();
                self.elab_inductive(name, universe_params, binders, ty, ctors, deriving)
            }

            SurfaceDecl::Structure {
                name,
                universe_params,
                binders,
                ty,
                fields,
                deriving,
                ..
            } => {
                self.universe_params = universe_params.clone();
                self.elab_structure(
                    name,
                    universe_params,
                    binders,
                    ty.as_deref(),
                    fields,
                    deriving,
                )
            }

            SurfaceDecl::Class {
                name,
                universe_params,
                binders,
                ty,
                fields,
                ..
            } => {
                // Classes are elaborated as structures, then registered as type classes
                // Classes don't have deriving clauses
                self.universe_params = universe_params.clone();
                let result = self.elab_structure(
                    name,
                    universe_params,
                    binders,
                    ty.as_deref(),
                    fields,
                    &[],
                )?;

                // Detect out-parameters by checking for outParam wrapper in binder types
                let out_params: Vec<usize> = binders
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, b)| {
                        if let Some(ty) = &b.ty {
                            if is_out_param_type(ty) {
                                return Some(idx);
                            }
                        }
                        None
                    })
                    .collect();

                // Detect semi-out-parameters by checking for semiOutParam wrapper
                let semi_out_params: Vec<usize> = binders
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, b)| {
                        if let Some(ty) = &b.ty {
                            if is_semi_out_param_type(ty) {
                                return Some(idx);
                            }
                        }
                        None
                    })
                    .collect();

                // Mark as a class in the instance table
                self.instances.register_class_full(
                    Name::from_string(name),
                    binders.len(),
                    out_params,
                    semi_out_params,
                );

                Ok(result)
            }

            SurfaceDecl::Instance {
                name,
                universe_params,
                binders,
                class_type,
                fields,
                priority,
                ..
            } => {
                self.universe_params = universe_params.clone();
                self.elab_instance(
                    name.as_deref(),
                    universe_params,
                    binders,
                    class_type,
                    fields,
                    *priority,
                )
            }

            // New declaration types - return skipped for now as they require more elaborate handling
            SurfaceDecl::Example { .. } => {
                // Example is an anonymous proof/definition, not stored
                Ok(ElabResult::Skipped)
            }
            SurfaceDecl::Import { .. } => {
                // Import statements are handled by the module system, not elaboration
                Ok(ElabResult::Skipped)
            }
            SurfaceDecl::Namespace { .. } => {
                // Namespaces are handled at the module level
                Ok(ElabResult::Skipped)
            }
            SurfaceDecl::Section { .. } => {
                // Sections are handled at the module level
                Ok(ElabResult::Skipped)
            }
            SurfaceDecl::UniverseDecl { .. } => {
                // Universe declarations just add names
                Ok(ElabResult::Skipped)
            }
            SurfaceDecl::Variable { .. } => {
                // Variable declarations add to context but don't produce output
                Ok(ElabResult::Skipped)
            }
            SurfaceDecl::Open { .. } => {
                // Open is a module-level operation
                Ok(ElabResult::Skipped)
            }
            SurfaceDecl::Check { .. } => {
                // #check is for interactive use
                Ok(ElabResult::Skipped)
            }
            SurfaceDecl::Eval { .. } => {
                // #eval is for interactive use
                Ok(ElabResult::Skipped)
            }
            SurfaceDecl::Print { .. } => {
                // #print is for interactive use
                Ok(ElabResult::Skipped)
            }
            SurfaceDecl::Mutual { .. } => {
                // Mutual recursion requires special handling
                Ok(ElabResult::Skipped)
            }
            SurfaceDecl::Syntax {
                name,
                precedence,
                pattern,
                category,
                ..
            } => {
                self.macro_ctx
                    .register_syntax(name.as_deref(), *precedence, pattern, category)
                    .map_err(|e| ElabError::MacroError(e.to_string()))?;
                Ok(ElabResult::Skipped)
            }
            SurfaceDecl::DeclareSyntaxCat { name, .. } => {
                self.macro_ctx.register_syntax_category(name);
                Ok(ElabResult::Skipped)
            }
            SurfaceDecl::Macro {
                pattern,
                category,
                expansion,
                ..
            } => {
                self.macro_ctx
                    .register_macro(pattern, category, expansion)
                    .map_err(|e| ElabError::MacroError(e.to_string()))?;
                Ok(ElabResult::Skipped)
            }
            SurfaceDecl::MacroRules { name, arms, .. } => {
                self.macro_ctx
                    .register_macro_rules(name.as_deref(), arms)
                    .map_err(|e| ElabError::MacroError(e.to_string()))?;
                Ok(ElabResult::Skipped)
            }
            SurfaceDecl::Notation {
                kind,
                precedence,
                pattern,
                expansion,
                ..
            } => {
                self.macro_ctx
                    .register_notation(*kind, *precedence, pattern, expansion)
                    .map_err(|e| ElabError::MacroError(e.to_string()))?;
                Ok(ElabResult::Skipped)
            }
            SurfaceDecl::Elab { .. } => {
                // Custom elaborators - not yet supported
                Ok(ElabResult::Skipped)
            }
            SurfaceDecl::Attribute { .. } => {
                // Attribute commands are post-processing
                Ok(ElabResult::Skipped)
            }
            SurfaceDecl::SetOption { .. } => {
                // Set option is for configuration
                Ok(ElabResult::Skipped)
            }
        }
    }

    /// Elaborate an instance declaration
    ///
    /// An instance provides an implementation of a type class for specific types.
    /// For example:
    /// ```text
    /// instance : Add Nat where
    ///   add := Nat.add
    /// ```
    ///
    /// This elaborates to a definition whose value is the class constructor
    /// applied to the field values, and registers the instance in the instance table.
    fn elab_instance(
        &mut self,
        name: Option<&str>,
        universe_params: &[String],
        binders: &[SurfaceBinder],
        class_type: &SurfaceExpr,
        fields: &[SurfaceFieldAssign],
        priority: Option<u32>,
    ) -> Result<ElabResult, ElabError> {
        use crate::instances::{extract_class_app, DEFAULT_PRIORITY};

        // Collect fvars for binders so we can abstract over them
        let mut binder_fvars = Vec::new();
        let mut binder_types = Vec::new();
        let mut binder_infos = Vec::new();

        // Elaborate binders (e.g., `[Add α]` for dependent instances)
        for binder in binders {
            let binder_ty = if let Some(t) = &binder.ty {
                self.elaborate(t)?
            } else {
                return Err(ElabError::CannotInfer);
            };
            let bi = convert_binder_info(binder.info);
            let fvar = self.push_local(binder.name.clone(), binder_ty.clone());
            binder_fvars.push(fvar);
            binder_types.push(binder_ty);
            binder_infos.push(bi);
        }

        // Elaborate the class type (e.g., `Add Nat`)
        let class_ty_expr = self.elaborate(class_type)?;

        // Extract the class name and arguments from the type
        let (class_name, _class_args) = extract_class_app(&class_ty_expr).ok_or_else(|| {
            ElabError::NotImplemented(format!(
                "instance class type must be a class application, got: {class_ty_expr:?}"
            ))
        })?;

        // Look up the class (which is a structure/inductive) to get field info
        let field_names = self.env.get_structure_field_names(&class_name)
            .ok_or_else(|| ElabError::NotImplemented(
                format!("class {class_name} not found in environment (must be declared as a class/structure first)")
            ))?
            .clone();

        // The constructor name is ClassName.mk
        let ctor_name = Name::from_string(&format!("{class_name}.mk"));

        // Check that we have all required fields
        let provided_fields: std::collections::HashSet<_> =
            fields.iter().map(|f| &f.name).collect();
        for expected in &field_names {
            let expected_str = expected.to_string();
            if !provided_fields.contains(&expected_str) {
                return Err(ElabError::NotImplemented(format!(
                    "missing field {expected_str} in instance for {class_name}"
                )));
            }
        }

        // Build the instance value by applying the constructor to field values
        // Order fields according to the class definition
        // Look up constructor to get its universe level parameters
        let ctor_levels: Vec<Level> = if let Some(info) = self.env.get_const(&ctor_name) {
            info.level_params
                .iter()
                .map(|_| Level::zero()) // Default to level 0, matching elab_ident behavior
                .collect()
        } else {
            vec![]
        };
        let mut instance_val = Expr::const_(ctor_name.clone(), ctor_levels);

        // First apply any type arguments from the class type
        // For `Add Nat`, we need to apply `Nat` to the constructor
        if let Some((_, class_args)) = extract_class_app(&class_ty_expr) {
            for arg in class_args {
                instance_val = Expr::app(instance_val, arg);
            }
        }

        // Then apply the field values in order
        for field_name in &field_names {
            let field_name_str = field_name.to_string();
            let field_assign = fields
                .iter()
                .find(|f| f.name == field_name_str)
                .ok_or_else(|| {
                    ElabError::NotImplemented(format!(
                        "missing field {field_name_str} in instance for {class_name}"
                    ))
                })?;

            let field_val = self.elaborate(&field_assign.val)?;
            instance_val = Expr::app(instance_val, field_val);
        }

        // Generate instance name if not provided
        let instance_name = if let Some(n) = name {
            Name::from_string(n)
        } else {
            // Auto-generate: inst<ClassName><TypeArg1>...
            let mut auto_name = format!("inst{class_name}");
            if let Some((_, class_args)) = extract_class_app(&class_ty_expr) {
                for arg in &class_args {
                    // Try to get a readable name from the argument
                    if let Expr::Const(n, _) = arg {
                        auto_name.push_str(&n.to_string());
                    }
                }
            }
            Name::from_string(&auto_name)
        };

        let priority = priority.unwrap_or(DEFAULT_PRIORITY);

        // Abstract over binders for both type and value
        let mut final_ty = class_ty_expr.clone();
        let mut final_val = instance_val;

        for i in (0..binders.len()).rev() {
            final_ty = final_ty.abstract_fvar(binder_fvars[i]);
            final_val = final_val.abstract_fvar(binder_fvars[i]);
            final_ty = Expr::pi(binder_infos[i], binder_types[i].clone(), final_ty);
            final_val = Expr::lam(binder_infos[i], binder_types[i].clone(), final_val);
        }

        // Pop binder fvars
        for _ in 0..binders.len() {
            self.pop_local();
        }

        // Register the instance
        self.instances.add_instance(
            instance_name.clone(),
            class_name.clone(),
            final_val.clone(),
            final_ty.clone(),
            priority,
        );

        Ok(ElabResult::Instance {
            name: instance_name,
            universe_params: universe_params
                .iter()
                .map(|s| Name::from_string(s))
                .collect(),
            class_name,
            ty: final_ty,
            val: final_val,
            priority,
        })
    }

    fn elab_def_body(
        &mut self,
        binders: &[SurfaceBinder],
        ty: Option<&SurfaceExpr>,
        val: &SurfaceExpr,
    ) -> Result<(Expr, Expr), ElabError> {
        if binders.is_empty() {
            let ty_expr = if let Some(ty) = ty {
                self.elaborate(ty)?
            } else {
                let val_expr = self.elaborate(val)?;
                self.infer_type(&val_expr)?
            };
            let val_expr = self.elaborate(val)?;
            return Ok((ty_expr, val_expr));
        }

        // Process first binder
        let binder = &binders[0];
        let binder_ty = if let Some(ty) = &binder.ty {
            self.elaborate(ty)?
        } else {
            self.fresh_meta(Expr::type_())
        };

        let bi = convert_binder_info(binder.info);
        let fvar = self.push_local(binder.name.clone(), binder_ty.clone());

        // Recursively process remaining binders
        let (inner_ty, inner_val) = self.elab_def_body(&binders[1..], ty, val)?;

        self.pop_local();

        // Abstract
        let ty_abs = inner_ty.abstract_fvar(fvar);
        let val_abs = inner_val.abstract_fvar(fvar);

        Ok((
            Expr::pi(bi, binder_ty.clone(), ty_abs),
            Expr::lam(bi, binder_ty, val_abs),
        ))
    }

    fn elab_axiom_type(
        &mut self,
        binders: &[SurfaceBinder],
        ty: &SurfaceExpr,
    ) -> Result<Expr, ElabError> {
        if binders.is_empty() {
            return self.elaborate(ty);
        }

        let binder = &binders[0];
        let binder_ty = if let Some(t) = &binder.ty {
            self.elaborate(t)?
        } else {
            return Err(ElabError::CannotInfer);
        };

        let bi = convert_binder_info(binder.info);
        let fvar = self.push_local(binder.name.clone(), binder_ty.clone());

        let inner_ty = self.elab_axiom_type(&binders[1..], ty)?;

        self.pop_local();
        let ty_abs = inner_ty.abstract_fvar(fvar);

        Ok(Expr::pi(bi, binder_ty, ty_abs))
    }

    /// Elaborate an inductive type declaration
    ///
    /// An inductive type has:
    /// - A name (e.g., `List`)
    /// - Universe parameters (e.g., `u`)
    /// - Parameters (e.g., `α : Type u`)
    /// - A result type (e.g., `Type u`)
    /// - Constructors (e.g., `nil`, `cons`)
    ///
    /// Example:
    /// ```text
    /// inductive List (α : Type u) : Type u
    /// | nil : List α
    /// | cons : α → List α → List α
    /// ```
    fn elab_inductive(
        &mut self,
        name: &str,
        universe_params: &[String],
        binders: &[SurfaceBinder],
        ty: &SurfaceExpr,
        ctors: &[SurfaceCtor],
        deriving: &[String],
    ) -> Result<ElabResult, ElabError> {
        let ind_name = Name::from_string(name);
        // SAFETY: Number of params bounded by practical limits (no inductive has billions of params)
        let num_params = u32::try_from(binders.len()).unwrap_or(u32::MAX);

        // Collect fvars for parameters so we can abstract over them
        let mut param_fvars = Vec::new();

        // Elaborate parameters
        for binder in binders {
            let binder_ty = if let Some(t) = &binder.ty {
                self.elaborate(t)?
            } else {
                return Err(ElabError::CannotInfer);
            };
            let fvar = self.push_local(binder.name.clone(), binder_ty);
            param_fvars.push(fvar);
        }

        // Elaborate the result type (e.g., Type, Type u, Prop)
        let result_ty = self.elaborate(ty)?;

        // Build the inductive type: (param1 : T1) → ... → (paramN : TN) → result_ty
        let mut ind_ty = result_ty.clone();
        for i in (0..binders.len()).rev() {
            let binder = &binders[i];
            let binder_ty = if let Some(t) = &binder.ty {
                self.elaborate(t)?
            } else {
                return Err(ElabError::CannotInfer);
            };
            ind_ty = ind_ty.abstract_fvar(param_fvars[i]);
            let bi = convert_binder_info(binder.info);
            ind_ty = Expr::pi(bi, binder_ty, ind_ty);
        }

        // Build the type of the inductive applied to parameters: IndName param1 ... paramN
        // This is the type that constructors return.
        let mut ind_applied = Expr::const_(ind_name.clone(), vec![]);
        for fvar in &param_fvars {
            ind_applied = Expr::app(ind_applied, Expr::fvar(*fvar));
        }

        // For recursive references in constructors, we need to bind the inductive name
        // temporarily. We'll use a local binding that will be resolved when the name
        // is looked up during constructor type elaboration.
        //
        // The type of the binding is the fully-applied inductive type (IndName α β ...),
        // but for name resolution purposes we store it as a local that represents
        // the inductive type constant.
        let ind_fvar = self.push_local(name.to_string(), ind_applied.clone());

        // Elaborate constructors
        let mut constructors = Vec::new();
        for ctor in ctors {
            let ctor_name = Name::from_string(&format!("{}.{}", name, ctor.name));

            // Elaborate constructor type (with params and inductive name in scope)
            let ctor_ty_raw = self.elaborate(&ctor.ty)?;

            // Replace references to the inductive fvar with the proper Const expression
            let ctor_ty_raw =
                self.replace_fvar_with_const(ctor_ty_raw, ind_fvar, &ind_name, &param_fvars);

            // Abstract constructor type over parameters
            let mut ctor_ty = ctor_ty_raw;
            for i in (0..binders.len()).rev() {
                let binder = &binders[i];
                ctor_ty = ctor_ty.abstract_fvar(param_fvars[i]);
                let binder_ty = if let Some(t) = &binder.ty {
                    self.elaborate(t)?
                } else {
                    return Err(ElabError::CannotInfer);
                };
                let bi = convert_binder_info(binder.info);
                ctor_ty = Expr::pi(bi, binder_ty, ctor_ty);
            }

            constructors.push((ctor_name, ctor_ty));
        }

        // Pop inductive fvar
        self.pop_local();

        // Pop param fvars
        for _ in 0..binders.len() {
            self.pop_local();
        }

        // Generate derived instances from deriving clause
        let derived_instances = self.generate_derived_instances_inductive(
            &ind_name,
            universe_params,
            binders,
            ctors,
            &ind_ty,
            deriving,
        );

        // Note: Recursors (rec, casesOn) are generated by the kernel during add_inductive.
        // They can be queried after registration via env.get_recursor("Type.rec").

        Ok(ElabResult::Inductive {
            name: ind_name,
            universe_params: universe_params
                .iter()
                .map(|s| Name::from_string(s))
                .collect(),
            num_params,
            ty: ind_ty,
            constructors,
            derived_instances,
        })
    }

    /// Generate derived instances for an inductive type
    ///
    /// For inductives, deriving requires pattern matching on constructors.
    /// E.g., for `inductive Bool | false | true deriving BEq`:
    /// ```text
    /// instance : BEq Bool where
    ///   beq a b := match a, b with
    ///     | Bool.false, Bool.false => true
    ///     | Bool.true, Bool.true => true
    ///     | _, _ => false
    /// ```
    fn generate_derived_instances_inductive(
        &mut self,
        ind_name: &Name,
        _universe_params: &[String],
        binders: &[SurfaceBinder],
        ctors: &[SurfaceCtor],
        _ind_ty: &Expr,
        deriving: &[String],
    ) -> Vec<DerivedInstance> {
        let mut instances = Vec::new();

        // Collect constructor names
        let ctor_names: Vec<Name> = ctors
            .iter()
            .map(|c| Name::from_string(&format!("{}.{}", ind_name, c.name)))
            .collect();

        for class_name in deriving {
            if let Some(instance) =
                self.derive_instance_inductive(ind_name, binders, ctors, &ctor_names, class_name)
            {
                instances.push(instance);
            }
        }

        instances
    }

    /// Derive a single instance for an inductive type
    fn derive_instance_inductive(
        &mut self,
        ind_name: &Name,
        binders: &[SurfaceBinder],
        ctors: &[SurfaceCtor],
        ctor_names: &[Name],
        class_name: &str,
    ) -> Option<DerivedInstance> {
        match class_name {
            "BEq" => Some(self.derive_beq_inductive(ind_name, binders, ctors, ctor_names)),
            "Repr" => Some(self.derive_repr_inductive(ind_name, binders, ctors, ctor_names)),
            "Hashable" => Some(self.derive_hashable_inductive(ind_name, binders, ctors, ctor_names)),
            "Inhabited" => self.derive_inhabited_inductive(ind_name, binders, ctors, ctor_names),
            "DecidableEq" => {
                Some(self.derive_decidable_eq_inductive(ind_name, binders, ctors, ctor_names))
            }
            _ => {
                // Unknown deriving class - silently skip
                None
            }
        }
    }

    /// Elaborate a structure declaration
    ///
    /// A structure is syntactic sugar for a single-constructor inductive with named fields.
    /// `structure Point where x : Nat  y : Nat`
    /// becomes:
    /// - Inductive `Point : Type` with constructor `Point.mk : Nat → Nat → Point`
    /// - Field names registered: ["x", "y"]
    fn elab_structure(
        &mut self,
        name: &str,
        universe_params: &[String],
        binders: &[SurfaceBinder],
        ty: Option<&SurfaceExpr>,
        fields: &[SurfaceField],
        deriving: &[String],
    ) -> Result<ElabResult, ElabError> {
        let struct_name = Name::from_string(name);
        let ctor_name = Name::from_string(&format!("{name}.mk"));
        // SAFETY: Number of params bounded by practical limits (no structure has billions of params)
        let num_params = u32::try_from(binders.len()).unwrap_or(u32::MAX);

        // Collect fvars for parameters so we can abstract over them
        let mut param_fvars = Vec::new();

        // Elaborate parameters
        for binder in binders {
            let binder_ty = if let Some(t) = &binder.ty {
                self.elaborate(t)?
            } else {
                return Err(ElabError::CannotInfer);
            };
            let fvar = self.push_local(binder.name.clone(), binder_ty);
            param_fvars.push(fvar);
        }

        // Structure result type (defaults to Type if not specified)
        let result_ty = if let Some(t) = ty {
            self.elaborate(t)?
        } else {
            Expr::type_()
        };

        // Build the structure type: (param1 : T1) → ... → (paramN : TN) → Type
        let mut struct_ty = result_ty.clone();
        for i in (0..binders.len()).rev() {
            let binder = &binders[i];
            let binder_ty = if let Some(t) = &binder.ty {
                self.elaborate(t)?
            } else {
                return Err(ElabError::CannotInfer);
            };
            struct_ty = struct_ty.abstract_fvar(param_fvars[i]);
            let bi = convert_binder_info(binder.info);
            struct_ty = Expr::pi(bi, binder_ty, struct_ty);
        }

        // Build the constructor type: fields → StructName params
        // First, elaborate field types (with params in scope)
        let mut field_types = Vec::new();
        let mut field_fvars = Vec::new();
        let mut field_names = Vec::new();

        for field in fields {
            let field_ty = self.elaborate(&field.ty)?;
            let fvar = self.push_local(field.name.clone(), field_ty.clone());
            field_types.push(field_ty);
            field_fvars.push(fvar);
            field_names.push(Name::from_string(&field.name));
        }

        // The return type of the constructor: StructName param1 ... paramN
        let mut ctor_result = Expr::const_(struct_name.clone(), vec![]);
        for fvar in &param_fvars {
            ctor_result = Expr::app(ctor_result, Expr::fvar(*fvar));
        }

        // Build constructor type: (field1 : T1) → ... → (fieldN : TN) → StructName params
        let mut ctor_ty = ctor_result.clone();
        for i in (0..fields.len()).rev() {
            ctor_ty = ctor_ty.abstract_fvar(field_fvars[i]);
            ctor_ty = Expr::pi(BinderInfo::Default, field_types[i].clone(), ctor_ty);
        }

        // Pop field fvars
        for _ in 0..fields.len() {
            self.pop_local();
        }

        // Generate projection functions
        // For each field i with type Ti, generate:
        //   StructName.fieldname : (params...) → StructName params → Ti
        //   StructName.fieldname = λ params s => s.i
        let mut projections = Vec::new();

        for (field_idx, field) in fields.iter().enumerate() {
            let proj_name = Name::from_string(&format!("{}.{}", name, field.name));

            // Re-elaborate field types to build the projection type with fresh scope
            // Push param fvars again for this projection
            let mut proj_param_fvars = Vec::new();
            for binder in binders {
                let binder_ty = if let Some(t) = &binder.ty {
                    self.elaborate(t)?
                } else {
                    return Err(ElabError::CannotInfer);
                };
                let fvar = self.push_local(binder.name.clone(), binder_ty);
                proj_param_fvars.push(fvar);
            }

            // Build struct type applied to params: StructName param1 ... paramN
            let mut struct_applied = Expr::const_(struct_name.clone(), vec![]);
            for fvar in &proj_param_fvars {
                struct_applied = Expr::app(struct_applied, Expr::fvar(*fvar));
            }

            // Push a local for the structure value FIRST
            // This is needed so that we can refer to it in field types
            let struct_fvar = self.push_local("self".to_string(), struct_applied.clone());

            // Build projection target type by re-elaborating field types
            // For dependent fields, we need earlier fields in scope, but they should
            // be projections of the struct value, not free variables.
            //
            // For field i with type Ti that may reference fields 0..i-1:
            // - Push earlier fields as locals whose VALUES are projections of struct_fvar
            // - When elaborating Ti, references to field j become s.j
            let mut earlier_field_fvars = Vec::new();
            for earlier_field in fields.iter().take(field_idx) {
                // Re-elaborate the earlier field's type (it may depend on even earlier fields)
                let earlier_ty = self.elaborate(&earlier_field.ty)?;
                // Push the field name as a local with its type
                let fvar = self.push_local(earlier_field.name.clone(), earlier_ty);
                earlier_field_fvars.push(fvar);
            }

            // Now elaborate the current field type with all earlier fields in scope
            let mut proj_field_ty = self.elaborate(&field.ty)?;

            // Substitute earlier field references with projections of struct_fvar
            // For each earlier field j, replace FVar(earlier_field_fvars[j]) with s.j
            for (j, &fvar) in earlier_field_fvars.iter().enumerate().rev() {
                // SAFETY: Field indices bounded by number of fields in structure
                let j_u32 = u32::try_from(j).unwrap_or(u32::MAX);
                let projection = Expr::proj(struct_name.clone(), j_u32, Expr::fvar(struct_fvar));
                proj_field_ty = proj_field_ty.subst_fvar(fvar, &projection);
            }

            // Pop the earlier field locals
            for _ in 0..field_idx {
                self.pop_local();
            }

            // Build projection value: Expr::proj(struct_name, field_idx, self)
            // SAFETY: Field index bounded by number of fields in structure
            let field_idx_u32 = u32::try_from(field_idx).unwrap_or(u32::MAX);
            let proj_body = Expr::proj(struct_name.clone(), field_idx_u32, Expr::fvar(struct_fvar));

            // Abstract over the struct value
            let proj_val_inner = proj_body.abstract_fvar(struct_fvar);
            let proj_val_lam =
                Expr::lam(BinderInfo::Default, struct_applied.clone(), proj_val_inner);

            // Build return type: StructName params → FieldType
            let proj_ty_inner = proj_field_ty.abstract_fvar(struct_fvar);
            let proj_ty_arrow = Expr::pi(BinderInfo::Default, struct_applied, proj_ty_inner);

            self.pop_local(); // pop struct_fvar

            // Abstract over params for both type and value
            let mut proj_ty = proj_ty_arrow;
            let mut proj_val = proj_val_lam;
            for i in (0..binders.len()).rev() {
                let binder = &binders[i];
                proj_ty = proj_ty.abstract_fvar(proj_param_fvars[i]);
                proj_val = proj_val.abstract_fvar(proj_param_fvars[i]);
                let binder_ty = if let Some(t) = &binder.ty {
                    self.elaborate(t)?
                } else {
                    return Err(ElabError::CannotInfer);
                };
                let bi = convert_binder_info(binder.info);
                proj_ty = Expr::pi(bi, binder_ty.clone(), proj_ty);
                proj_val = Expr::lam(bi, binder_ty, proj_val);
            }

            // Pop param fvars
            for _ in 0..binders.len() {
                self.pop_local();
            }

            projections.push((proj_name, proj_ty, proj_val));
        }

        // Abstract constructor type over parameters
        for i in (0..binders.len()).rev() {
            let binder = &binders[i];
            ctor_ty = ctor_ty.abstract_fvar(param_fvars[i]);
            let binder_ty = if let Some(t) = &binder.ty {
                self.elaborate(t)?
            } else {
                return Err(ElabError::CannotInfer);
            };
            let bi = convert_binder_info(binder.info);
            ctor_ty = Expr::pi(bi, binder_ty, ctor_ty);
        }

        // Pop param fvars
        for _ in 0..binders.len() {
            self.pop_local();
        }

        // Generate derived instances from deriving clause
        let derived_instances = self.generate_derived_instances(
            &struct_name,
            universe_params,
            binders,
            fields,
            &struct_ty,
            deriving,
        );

        Ok(ElabResult::Structure {
            name: struct_name,
            universe_params: universe_params
                .iter()
                .map(|s| Name::from_string(s))
                .collect(),
            num_params,
            ty: struct_ty,
            ctor_name,
            ctor_ty,
            field_names,
            projections,
            derived_instances,
        })
    }

    /// Generate derived instances for the given structure and deriving clauses
    fn generate_derived_instances(
        &mut self,
        struct_name: &Name,
        _universe_params: &[String],
        binders: &[SurfaceBinder],
        fields: &[SurfaceField],
        _struct_ty: &Expr,
        deriving: &[String],
    ) -> Vec<DerivedInstance> {
        let mut instances = Vec::new();

        // Collect field names for deriving handlers
        let field_names: Vec<Name> = fields.iter().map(|f| Name::from_string(&f.name)).collect();

        for class_name in deriving {
            if let Some(instance) =
                self.derive_instance(struct_name, binders, fields, &field_names, class_name)
            {
                instances.push(instance);
            }
        }

        instances
    }

    /// Derive a single type class instance for a structure
    ///
    /// Returns None if the class is not supported for deriving
    fn derive_instance(
        &mut self,
        struct_name: &Name,
        binders: &[SurfaceBinder],
        fields: &[SurfaceField],
        field_names: &[Name],
        class_name: &str,
    ) -> Option<DerivedInstance> {
        match class_name {
            "BEq" => Some(self.derive_beq(struct_name, binders, field_names)),
            "Repr" => Some(self.derive_repr(struct_name, binders, field_names)),
            "Hashable" => Some(self.derive_hashable(struct_name, binders, field_names)),
            "Inhabited" => Some(self.derive_inhabited(struct_name, binders, fields)),
            "DecidableEq" => Some(self.derive_decidable_eq(struct_name, binders, field_names)),
            _ => {
                // Unknown deriving class - silently skip for now
                // In a production implementation, this would error
                None
            }
        }
    }

    /// Build the parametric struct type applied to bound type variables
    ///
    /// For a structure like `structure Pair (α : Type) (β : Type)`,
    /// this builds `Pair α β` where α and β are bound variables at the given offset.
    ///
    /// `offset` is the de Bruijn offset for accessing the type parameters:
    /// - For instance type: offset = number of instance params (for [BEq α] etc.)
    /// - For function body: offset = number of lambdas before accessing params
    fn build_parametric_struct_type(
        &self,
        struct_name: &Name,
        num_params: usize,
        offset: usize,
    ) -> Expr {
        let mut result = Expr::const_(struct_name.clone(), vec![]);

        // Apply type parameter variables in order (from outermost to innermost)
        // With de Bruijn indices: if we have (α : Type) (β : Type), then
        // inside a body with `offset` additional binders:
        // - α is at index (offset + num_params - 1 - 0) = offset + num_params - 1
        // - β is at index (offset + num_params - 1 - 1) = offset + num_params - 2
        for i in 0..num_params {
            let var_idx = offset + num_params - 1 - i;
            // SAFETY: de Bruijn index bounded by context depth
            let var_idx_u32 = u32::try_from(var_idx).unwrap_or(u32::MAX);
            result = Expr::app(result, Expr::bvar(var_idx_u32));
        }

        result
    }

    /// Build the instance type with type parameter bindings and constraints
    ///
    /// For `BEq (Pair α β)` with params `(α : Type) (β : Type)`, this builds:
    /// `∀ (α : Type) (β : Type) [BEq α] [BEq β], BEq (Pair α β)`
    ///
    /// Returns: (instance_type, number_of_constraint_params)
    fn build_parametric_instance_type(
        &mut self,
        struct_name: &Name,
        binders: &[SurfaceBinder],
        class_name: &Name,
    ) -> (Expr, usize) {
        let num_params = binders.len();

        if num_params == 0 {
            // Non-parametric: just `Class StructName`
            let struct_type = Expr::const_(struct_name.clone(), vec![]);
            let instance_ty = Expr::app(Expr::const_(class_name.clone(), vec![]), struct_type);
            return (instance_ty, 0);
        }

        // For parametric structures, we need instance constraints for each type parameter
        // The number of constraint params depends on the class requirements.
        // For BEq/DecidableEq/Hashable: need constraint for each param
        // For Repr/Inhabited: also need constraint for each param
        let num_constraints = num_params;

        // Build the core type: Class (Struct α β ...)
        // At this point, we're inside num_params type binders + num_constraints constraint binders
        let struct_applied =
            self.build_parametric_struct_type(struct_name, num_params, num_constraints);
        let core_instance_ty = Expr::app(Expr::const_(class_name.clone(), vec![]), struct_applied);

        // Wrap with instance constraints: [Class α] → [Class β] → ...
        // These are applied in reverse order (innermost first)
        let mut result = core_instance_ty;
        for i in (0..num_params).rev() {
            // The constraint type: Class αᵢ
            // Inside constraint binder i, the type param αᵢ is at index:
            // (num_params - 1 - i) + (num_constraints - 1 - i) + 1 = num_params + num_constraints - 2*i - 1
            // Actually simpler: after the current constraint, remaining constraints + type params
            // Let's compute step by step for wrapping:
            // When wrapping constraint i (0-indexed from outside):
            // - We've already wrapped constraints (num_params-1) down to (i+1)
            // - Current constraint references param i
            // - Param i in the final type is at BVar(num_constraints + num_params - 1 - i) before any constraints
            // - After wrapping (num_params - 1 - i) constraints, param i is at BVar(num_params - 1 - i)

            // Actually, let's think of it differently:
            // Final structure: ∀α ∀β [BEq α] [BEq β], BEq (Pair α β)
            // Constraint [BEq α] references α which is 2 binders away (β and [BEq β] are between)
            // Wait, instance constraints are Pi types with BinderInfo::InstImplicit

            // For param at position i (0=α, 1=β in `(α : Type) (β : Type)`):
            // When building the Pi for constraint i, we're wrapping what comes after
            // Inside the body of this Pi, param i is at index (num_params + num_constraints - 1 - i) - current_depth
            // Let me use a cleaner approach: build from outside in

            // After wrapping i constraints (starting from i = num_params - 1 down to 0),
            // param i is accessed at BVar(num_params - 1 - i + something)
            // Let's just hard-code for now: constraint for param i references BVar(i) relative to type params
            // No wait, let me think more carefully.

            // Structure: Π(α:Type) Π(β:Type) Π[BEq α] Π[BEq β]. BEq (Pair α β)
            // In the body (BEq (Pair α β)):
            //   - [BEq β] is BVar(0)
            //   - [BEq α] is BVar(1)
            //   - β is BVar(2)
            //   - α is BVar(3)
            // So Pair α β = App(App(Pair, BVar(3)), BVar(2))

            // When wrapping [BEq β] (i = 1):
            //   - The constraint type is `BEq β`
            //   - At this point β is BVar(0) (just type params bound, no constraints yet)
            //   Wait no, I'm wrapping from inside out.

            // Let's build inside out:
            // Start with body at depth (num_params + num_constraints)
            // Then wrap each constraint from last to first

            // When wrapping constraint for param i (going from num_params-1 to 0):
            // Before this wrap, we have (num_params - 1 - i) constraints already wrapped
            // At this depth, param i is at BVar(num_params - 1 - i)
            // The constraint type should be: Class BVar(num_params - 1 - i)
            let param_offset = num_params - 1 - i;
            // SAFETY: de Bruijn index bounded by number of parameters
            let param_offset_u32 = u32::try_from(param_offset).unwrap_or(u32::MAX);
            let constraint_ty = Expr::app(
                Expr::const_(class_name.clone(), vec![]),
                Expr::bvar(param_offset_u32),
            );

            result = Expr::pi(BinderInfo::InstImplicit, constraint_ty, result);
        }

        // Wrap with type parameter bindings: (α : Type) → (β : Type) → ...
        // These are applied in reverse order (innermost first)
        for _i in (0..num_params).rev() {
            // Each type param has type `Type` (Sort 1)
            let type_sort = Expr::sort(Level::succ(Level::zero()));
            result = Expr::pi(BinderInfo::Implicit, type_sort, result);
        }

        (result, num_constraints)
    }

    /// Wrap an instance value with lambdas for type parameters and constraints
    ///
    /// For a parametric instance, the value needs to be a function taking
    /// the type parameters and their class instances as arguments.
    fn wrap_parametric_instance_value(
        &self,
        inner_val: Expr,
        num_params: usize,
        class_name: &Name,
    ) -> Expr {
        if num_params == 0 {
            return inner_val;
        }

        let mut result = inner_val;

        // Wrap with lambdas for instance constraints (innermost first = last param)
        for i in (0..num_params).rev() {
            // Constraint type references the param at appropriate offset
            let param_offset = num_params - 1 - i;
            // SAFETY: de Bruijn index bounded by number of parameters
            let param_offset_u32 = u32::try_from(param_offset).unwrap_or(u32::MAX);
            let constraint_ty = Expr::app(
                Expr::const_(class_name.clone(), vec![]),
                Expr::bvar(param_offset_u32),
            );
            result = Expr::lam(BinderInfo::InstImplicit, constraint_ty, result);
        }

        // Wrap with lambdas for type parameters (innermost first = last param)
        for _i in (0..num_params).rev() {
            let type_sort = Expr::sort(Level::succ(Level::zero()));
            result = Expr::lam(BinderInfo::Implicit, type_sort, result);
        }

        result
    }

    /// Derive BEq instance for a structure
    ///
    /// Generates: instance : BEq StructName where
    ///   beq := fun a b => a.field1 == b.field1 && a.field2 == b.field2 && ...
    ///
    /// For parametric structures like `structure Pair (α : Type) (β : Type)`:
    ///   instance [BEq α] [BEq β] : BEq (Pair α β) where ...
    ///
    /// The generated beq function compares each field pairwise and combines
    /// the results with Bool.and. For a structure with no fields, returns true.
    fn derive_beq(
        &mut self,
        struct_name: &Name,
        binders: &[SurfaceBinder],
        field_names: &[Name],
    ) -> DerivedInstance {
        let instance_name = Name::from_string(&format!("inst{struct_name}BEq"));
        let class_name = Name::from_string("BEq");
        let num_params = binders.len();

        // Build instance type with constraints for parametric structures
        let (instance_ty, _num_constraints) =
            self.build_parametric_instance_type(struct_name, binders, &class_name);

        // Build the parametric struct type for lambda annotations
        // Inside the a/b lambdas (at depth 2), params are further up
        // For parametric case: params are at indices (num_params * 2 + 1) and (num_params * 2)
        // But actually we need to rebuild the struct type at the right depth
        let struct_type = if num_params == 0 {
            Expr::const_(struct_name.clone(), vec![])
        } else {
            // Inside the instance lambdas (params + constraints), then inside a/b lambdas
            // The type annotation for a and b uses params at offset (num_params + 2)
            // because when we're at the 'a' lambda, we have:
            // - params at indices [num_params*2 + 1 .. num_params*2 + num_params]
            // - constraints at indices [num_params + 1 .. num_params * 2]
            // - 'a' at index 1, 'b' at index 0 (in the body)
            // For the type annotation: offset = num_params (constraints are between params and here)
            self.build_parametric_struct_type(struct_name, num_params, num_params)
        };

        // Build the beq function: fun a b => <comparisons>
        // The function body compares each field and ANDs the results together
        //
        // For struct Point { x : Nat, y : Nat }:
        //   beq := λ (a : Point) (b : Point) =>
        //            Bool.and (BEq.beq a.x b.x) (BEq.beq a.y b.y)
        //
        // We create the lambda structure but use metavariables for field comparisons
        // since full implementation requires resolving BEq instances for field types.

        // Create bound variable references for lambda parameters
        // In de Bruijn indexing: a = BVar(1), b = BVar(0) inside the body
        let a_ref = Expr::bvar(1);
        let b_ref = Expr::bvar(0);

        // Build the comparison body
        let body = if field_names.is_empty() {
            // No fields - return Bool.true
            Expr::const_(Name::from_string("Bool.true"), vec![])
        } else {
            // Build field comparisons: (a.field0 == b.field0) && (a.field1 == b.field1) && ...
            // We project each field and call BEq.beq on them

            let bool_and = Name::from_string("Bool.and");
            let beq_beq = Name::from_string("BEq.beq");

            // Start with comparison of first field
            let mut comparison = {
                let a_field = Expr::proj(struct_name.clone(), 0, a_ref.clone());
                let b_field = Expr::proj(struct_name.clone(), 0, b_ref.clone());
                // BEq.beq a.field0 b.field0
                Expr::app(
                    Expr::app(Expr::const_(beq_beq.clone(), vec![]), a_field),
                    b_field,
                )
            };

            // AND with remaining field comparisons
            for (idx, _field_name) in field_names.iter().enumerate().skip(1) {
                // SAFETY: Field index bounded by number of fields in structure
                let idx_u32 = u32::try_from(idx).unwrap_or(u32::MAX);
                let a_field = Expr::proj(struct_name.clone(), idx_u32, a_ref.clone());
                let b_field = Expr::proj(struct_name.clone(), idx_u32, b_ref.clone());

                // BEq.beq a.fieldN b.fieldN
                let field_cmp = Expr::app(
                    Expr::app(Expr::const_(beq_beq.clone(), vec![]), a_field),
                    b_field,
                );

                // Bool.and comparison field_cmp
                comparison = Expr::app(
                    Expr::app(Expr::const_(bool_and.clone(), vec![]), comparison),
                    field_cmp,
                );
            }

            comparison
        };

        // Build: λ (a : StructName) => λ (b : StructName) => body
        let inner_lam = Expr::lam(BinderInfo::Default, struct_type.clone(), body);
        let beq_func = Expr::lam(BinderInfo::Default, struct_type.clone(), inner_lam);

        // Build the instance value: BEq.mk beq_func
        // BEq.mk : (α → α → Bool) → BEq α
        let beq_mk = Name::from_string("BEq.mk");
        let core_instance_val = Expr::app(Expr::const_(beq_mk, vec![]), beq_func);

        // Wrap with lambdas for type parameters and constraints
        let instance_val =
            self.wrap_parametric_instance_value(core_instance_val, num_params, &class_name);

        DerivedInstance {
            name: instance_name,
            class_name,
            ty: instance_ty,
            val: instance_val,
            priority: 100,
        }
    }

    /// Derive Repr instance for a structure
    ///
    /// Generates: instance : Repr StructName where
    ///   reprPrec := fun s prec => "StructName { field1 := " ++ repr s.field1 ++ ", ... }"
    ///
    /// For parametric structures: instance [Repr α] [Repr β] : Repr (Pair α β) where ...
    ///
    /// The generated reprPrec function shows the structure name and all field values.
    fn derive_repr(
        &mut self,
        struct_name: &Name,
        binders: &[SurfaceBinder],
        field_names: &[Name],
    ) -> DerivedInstance {
        let instance_name = Name::from_string(&format!("inst{struct_name}Repr"));
        let class_name = Name::from_string("Repr");
        let num_params = binders.len();

        // Build instance type with constraints
        let (instance_ty, _num_constraints) =
            self.build_parametric_instance_type(struct_name, binders, &class_name);

        // Build parametric struct type for lambda annotations
        let struct_type = if num_params == 0 {
            Expr::const_(struct_name.clone(), vec![])
        } else {
            // At the s lambda, offset = num_params (for constraints)
            self.build_parametric_struct_type(struct_name, num_params, num_params)
        };

        // Build the reprPrec function: fun s prec => <representation>
        // Repr class has: reprPrec : α → Nat → Std.Format
        //
        // For struct Point { x : Nat, y : Nat }:
        //   reprPrec := λ (s : Point) (prec : Nat) =>
        //     Format.bracket "{ " (Format.joinSep [...fields...] ", ") " }"
        //
        // We build a simplified representation that just shows the structure
        // with field repr calls. Full implementation would use Format properly.

        // BVar(1) = s (the structure), BVar(0) = prec (precedence)
        let s_ref = Expr::bvar(1);
        let _prec_ref = Expr::bvar(0);

        // Build the format body
        // For simplicity, we build: Format.text "StructName { field1 := <repr>, ... }"
        // A real implementation would use proper Format combinators

        let body = if field_names.is_empty() {
            // No fields - just show structure name with empty braces
            // Format.text "StructName { }"
            let format_text = Name::from_string("Std.Format.text");
            let name_str = format!("{struct_name} {{ }}");
            Expr::app(Expr::const_(format_text, vec![]), Expr::str_lit(&name_str))
        } else {
            // Build a format showing all fields
            // We use Format.group with Format.nest for proper pretty printing
            //
            // Structure:
            //   Format.group (Format.nest 2 (Format.join [
            //     Format.text "StructName {",
            //     Format.line,
            //     field1 repr,
            //     ...,
            //     Format.text "}"
            //   ]))

            let format_text = Name::from_string("Std.Format.text");
            let format_append = Name::from_string("Std.Format.append");
            let repr_fn = Name::from_string("repr");

            // Start with "StructName { "
            let mut format = Expr::app(
                Expr::const_(format_text.clone(), vec![]),
                Expr::str_lit(format!("{struct_name} {{ ")),
            );

            // Add each field: "fieldName := " ++ repr s.field ++ ", "
            for (idx, field_name) in field_names.iter().enumerate() {
                // SAFETY: Field index bounded by number of fields in structure
                let idx_u32 = u32::try_from(idx).unwrap_or(u32::MAX);
                let s_field = Expr::proj(struct_name.clone(), idx_u32, s_ref.clone());

                // "fieldName := "
                let field_prefix = Expr::app(
                    Expr::const_(format_text.clone(), vec![]),
                    Expr::str_lit(format!("{field_name} := ")),
                );

                // repr s.field
                let field_repr = Expr::app(Expr::const_(repr_fn.clone(), vec![]), s_field);

                // Append field prefix
                format = Expr::app(
                    Expr::app(Expr::const_(format_append.clone(), vec![]), format),
                    field_prefix,
                );

                // Append field repr
                format = Expr::app(
                    Expr::app(Expr::const_(format_append.clone(), vec![]), format),
                    field_repr,
                );

                // Append separator (", " or " }")
                let separator = if idx < field_names.len() - 1 {
                    ", "
                } else {
                    " }"
                };
                let sep_format = Expr::app(
                    Expr::const_(format_text.clone(), vec![]),
                    Expr::str_lit(separator),
                );
                format = Expr::app(
                    Expr::app(Expr::const_(format_append.clone(), vec![]), format),
                    sep_format,
                );
            }

            format
        };

        // Build: λ (s : StructName) => λ (prec : Nat) => body
        let nat_type = Expr::const_(Name::from_string("Nat"), vec![]);
        let inner_lam = Expr::lam(BinderInfo::Default, nat_type, body);
        let repr_prec_func = Expr::lam(BinderInfo::Default, struct_type, inner_lam);

        // Build the instance value: Repr.mk reprPrec_func
        // Repr.mk : (α → Nat → Format) → Repr α
        let repr_mk = Name::from_string("Repr.mk");
        let core_instance_val = Expr::app(Expr::const_(repr_mk, vec![]), repr_prec_func);

        // Wrap with lambdas for type parameters and constraints
        let instance_val =
            self.wrap_parametric_instance_value(core_instance_val, num_params, &class_name);

        DerivedInstance {
            name: instance_name,
            class_name,
            ty: instance_ty,
            val: instance_val,
            priority: 100,
        }
    }

    /// Derive Hashable instance for a structure
    ///
    /// Generates: instance : Hashable StructName where
    ///   hash := fun s => mixHash (hash s.field1) (mixHash (hash s.field2) ...)
    ///
    /// For parametric structures: instance [Hashable α] [Hashable β] : Hashable (Pair α β) where ...
    ///
    /// The generated hash function hashes each field and mixes them together.
    fn derive_hashable(
        &mut self,
        struct_name: &Name,
        binders: &[SurfaceBinder],
        field_names: &[Name],
    ) -> DerivedInstance {
        let instance_name = Name::from_string(&format!("inst{struct_name}Hashable"));
        let class_name = Name::from_string("Hashable");
        let num_params = binders.len();

        // Build instance type with constraints
        let (instance_ty, _num_constraints) =
            self.build_parametric_instance_type(struct_name, binders, &class_name);

        // Build parametric struct type for lambda annotations
        let struct_type = if num_params == 0 {
            Expr::const_(struct_name.clone(), vec![])
        } else {
            self.build_parametric_struct_type(struct_name, num_params, num_params)
        };

        // Build the hash function: fun s => <hashing>
        // Hashable class has: hash : α → UInt64
        //
        // For struct Point { x : Nat, y : Nat }:
        //   hash := λ (s : Point) =>
        //     mixHash (hash s.x) (hash s.y)

        let s_ref = Expr::bvar(0);

        let body = if field_names.is_empty() {
            // No fields - return a constant hash (0)
            Expr::nat_lit(0)
        } else {
            let hash_fn = Name::from_string("hash");
            let mix_hash = Name::from_string("mixHash");

            // Start with hash of first field
            let first_field = Expr::proj(struct_name.clone(), 0, s_ref.clone());
            let mut hash_expr = Expr::app(Expr::const_(hash_fn.clone(), vec![]), first_field);

            // Mix with remaining field hashes
            for idx in 1..field_names.len() {
                // SAFETY: Field index bounded by number of fields in structure
                let idx_u32 = u32::try_from(idx).unwrap_or(u32::MAX);
                let field = Expr::proj(struct_name.clone(), idx_u32, s_ref.clone());
                let field_hash = Expr::app(Expr::const_(hash_fn.clone(), vec![]), field);

                // mixHash current_hash field_hash
                hash_expr = Expr::app(
                    Expr::app(Expr::const_(mix_hash.clone(), vec![]), hash_expr),
                    field_hash,
                );
            }

            hash_expr
        };

        // Build: λ (s : StructName) => body
        let hash_func = Expr::lam(BinderInfo::Default, struct_type, body);

        // Build the instance value: Hashable.mk hash_func
        // Hashable.mk : (α → UInt64) → Hashable α
        let hashable_mk = Name::from_string("Hashable.mk");
        let core_instance_val = Expr::app(Expr::const_(hashable_mk, vec![]), hash_func);

        // Wrap with lambdas for type parameters and constraints
        let instance_val =
            self.wrap_parametric_instance_value(core_instance_val, num_params, &class_name);

        DerivedInstance {
            name: instance_name,
            class_name,
            ty: instance_ty,
            val: instance_val,
            priority: 100,
        }
    }

    /// Derive Inhabited instance for a structure
    ///
    /// Generates: instance : Inhabited StructName where
    ///   default := StructName.mk (Inhabited.default field1) (Inhabited.default field2) ...
    ///
    /// For parametric structures: instance [Inhabited α] [Inhabited β] : Inhabited (Pair α β) where ...
    ///
    /// We construct the default value by applying the structure constructor to
    /// an `Inhabited.default` call for each field, introducing metavariables for
    /// the required field-level Inhabited instances.
    fn derive_inhabited(
        &mut self,
        struct_name: &Name,
        binders: &[SurfaceBinder],
        fields: &[SurfaceField],
    ) -> DerivedInstance {
        let instance_name = Name::from_string(&format!("inst{struct_name}Inhabited"));
        let class_name = Name::from_string("Inhabited");
        let num_params = binders.len();

        // Build instance type with constraints
        let (instance_ty, _num_constraints) =
            self.build_parametric_instance_type(struct_name, binders, &class_name);

        // Build the default value: StructName.mk (default field1) (default field2) ...
        let ctor_name = Name::from_string(&format!("{struct_name}.mk"));
        let mut ctor_app = Expr::const_(ctor_name, vec![]);

        // For parametric structures, the constructor needs type arguments
        // Apply the type parameters (as bound variables) to the constructor
        for i in 0..num_params {
            // Inside the instance lambdas, type params are at indices
            // [num_params * 2 - 1 .. num_params] (after constraints)
            let var_idx = num_params * 2 - 1 - i;
            // SAFETY: de Bruijn index bounded by context depth
            let var_idx_u32 = u32::try_from(var_idx).unwrap_or(u32::MAX);
            ctor_app = Expr::app(ctor_app, Expr::bvar(var_idx_u32));
        }

        if !fields.is_empty() {
            let inhabited_default = Name::from_string("Inhabited.default");

            for field in fields {
                // Elaborate the field type; if the environment lacks the name (e.g., Nat
                // not preloaded), fall back to a metavariable of type `Type`.
                let field_ty = match self.elaborate(&field.ty) {
                    Ok(ty) => ty,
                    Err(_) => self.fresh_meta(Expr::type_()),
                };

                // Instance: Inhabited field_ty
                let inhabited_field_ty =
                    Expr::app(Expr::const_(class_name.clone(), vec![]), field_ty.clone());
                let inhabited_meta = self.fresh_meta(inhabited_field_ty);

                // Inhabited.default {field_ty} (inst)
                let default_val = Expr::app(
                    Expr::app(Expr::const_(inhabited_default.clone(), vec![]), field_ty),
                    inhabited_meta,
                );

                ctor_app = Expr::app(ctor_app, default_val);
            }
        }

        // Build instance value: Inhabited.mk default_value
        let core_instance_val = Expr::app(
            Expr::const_(Name::from_string("Inhabited.mk"), vec![]),
            ctor_app,
        );

        // Wrap with lambdas for type parameters and constraints
        let instance_val =
            self.wrap_parametric_instance_value(core_instance_val, num_params, &class_name);

        DerivedInstance {
            name: instance_name,
            class_name,
            ty: instance_ty,
            val: instance_val,
            priority: 100,
        }
    }

    /// Derive DecidableEq instance for a structure
    ///
    /// Generates: `instance : DecidableEq StructName where`
    ///   `decEq := fun a b => ...` (field-aware decision procedure)
    ///
    /// For parametric structures: instance [DecidableEq α] [DecidableEq β] : DecidableEq (Pair α β) where ...
    ///
    /// For empty structs: Returns `Decidable.isTrue Eq.refl`
    /// For structs with fields: Builds a decision tree that compares fields using
    /// DecidableEq.decEq and combines results. Uses `decEq_of_beq` pattern where
    /// we compare fields and construct appropriate Decidable values.
    fn derive_decidable_eq(
        &mut self,
        struct_name: &Name,
        binders: &[SurfaceBinder],
        field_names: &[Name],
    ) -> DerivedInstance {
        let instance_name = Name::from_string(&format!("inst{struct_name}DecidableEq"));
        let class_name = Name::from_string("DecidableEq");
        let num_params = binders.len();

        // Build instance type with constraints
        let (instance_ty, _num_constraints) =
            self.build_parametric_instance_type(struct_name, binders, &class_name);

        // Build parametric struct type for lambda annotations
        let struct_type = if num_params == 0 {
            Expr::const_(struct_name.clone(), vec![])
        } else {
            self.build_parametric_struct_type(struct_name, num_params, num_params)
        };

        // Build decEq : StructName → StructName → Decidable (Eq _ _)
        // In de Bruijn indexing: a = BVar(1), b = BVar(0) inside the body
        let a_ref = Expr::bvar(1);
        let b_ref = Expr::bvar(0);

        let body = if field_names.is_empty() {
            // For empty struct: Decidable.isTrue Eq.refl
            // All empty structs of the same type are equal
            Expr::app(
                Expr::const_(Name::from_string("Decidable.isTrue"), vec![]),
                Expr::const_(Name::from_string("Eq.refl"), vec![]),
            )
        } else {
            // For struct with fields: build nested match/if structure
            // Pattern: match DecidableEq.decEq a.f0 b.f0 with
            //   | isTrue h0 => match DecidableEq.decEq a.f1 b.f1 with
            //     | isTrue h1 => isTrue (congrArg ...)
            //     | isFalse h1 => isFalse (...)
            //   | isFalse h0 => isFalse (...)
            //
            // Simplified approach: Use decidable_of_iff with BEq comparison
            // decEq a b := if BEq.beq a b then isTrue <proof> else isFalse <proof>
            //
            // We generate:
            //   match DecidableEq.decEq a.f0 b.f0, DecidableEq.decEq a.f1 b.f1, ... with
            //   | isTrue _, isTrue _, ... => isTrue <rfl_proof>
            //   | _, _, ... => isFalse <ne_proof>
            //
            // For simplicity, we use the decidableAnd pattern:
            // Build (DecidableEq.decEq a.f0 b.f0).and (DecidableEq.decEq a.f1 b.f1) ...
            // using And.decidable instances and a proof of equivalence.
            //
            // Actual approach: Create a comparison expression using DecidableEq.decEq
            // for each field and fold them together.

            let deceq_deceq = Name::from_string("DecidableEq.decEq");

            // Build comparison for first field
            let mut decision = {
                let a_field = Expr::proj(struct_name.clone(), 0, a_ref.clone());
                let b_field = Expr::proj(struct_name.clone(), 0, b_ref.clone());
                // DecidableEq.decEq a.field0 b.field0
                Expr::app(
                    Expr::app(Expr::const_(deceq_deceq.clone(), vec![]), a_field),
                    b_field,
                )
            };

            // For multiple fields, we need to combine the decisions.
            // In Lean, this would use Decidable.and or a match expression.
            // We'll use the decidableAnd pattern:
            //   instDecidableAnd : Decidable p → Decidable q → Decidable (p ∧ q)
            //
            // For struct equality: (a = b) ↔ (a.f0 = b.f0) ∧ (a.f1 = b.f1) ∧ ...
            // So we build: decidable_of_iff (And.decidable ...)
            //
            // Simpler encoding: fold field decisions with and_decidable
            let and_decidable = Name::from_string("instDecidableAnd");

            for (idx, _field_name) in field_names.iter().enumerate().skip(1) {
                // SAFETY: Field index bounded by number of fields in structure
                let idx_u32 = u32::try_from(idx).unwrap_or(u32::MAX);
                let a_field = Expr::proj(struct_name.clone(), idx_u32, a_ref.clone());
                let b_field = Expr::proj(struct_name.clone(), idx_u32, b_ref.clone());

                // DecidableEq.decEq a.fieldN b.fieldN
                let field_dec = Expr::app(
                    Expr::app(Expr::const_(deceq_deceq.clone(), vec![]), a_field),
                    b_field,
                );

                // Combine: instDecidableAnd decision field_dec
                decision = Expr::app(
                    Expr::app(Expr::const_(and_decidable.clone(), vec![]), decision),
                    field_dec,
                );
            }

            decision
        };

        // λ (a : Struct) (b : Struct) => body
        let inner_lam = Expr::lam(BinderInfo::Default, struct_type.clone(), body);
        let dec_eq_func = Expr::lam(BinderInfo::Default, struct_type.clone(), inner_lam);

        // Instance value: DecidableEq.mk dec_eq_func
        let core_instance_val = Expr::app(
            Expr::const_(Name::from_string("DecidableEq.mk"), vec![]),
            dec_eq_func,
        );

        // Wrap with lambdas for type parameters and constraints
        let instance_val =
            self.wrap_parametric_instance_value(core_instance_val, num_params, &class_name);

        DerivedInstance {
            name: instance_name,
            class_name,
            ty: instance_ty,
            val: instance_val,
            priority: 100,
        }
    }

    // ===========================================
    // Inductive deriving handlers
    // ===========================================

    /// Derive BEq for an inductive type
    ///
    /// For enumeration-like inductives (constructors with no arguments),
    /// generates a simple comparison based on constructor equality.
    ///
    /// For inductives with constructor arguments, we would need to compare
    /// arguments recursively. For now, we handle the simple enum case.
    fn derive_beq_inductive(
        &mut self,
        ind_name: &Name,
        binders: &[SurfaceBinder],
        _ctors: &[SurfaceCtor],
        ctor_names: &[Name],
    ) -> DerivedInstance {
        let instance_name = Name::from_string(&format!("inst{ind_name}BEq"));
        let class_name = Name::from_string("BEq");
        let num_params = binders.len();

        // Build instance type with constraints
        let (instance_ty, _num_constraints) =
            self.build_parametric_instance_type(ind_name, binders, &class_name);

        // Build parametric type for lambda annotations
        let ind_type = if num_params == 0 {
            Expr::const_(ind_name.clone(), vec![])
        } else {
            self.build_parametric_struct_type(ind_name, num_params, num_params)
        };

        // For simple enumerations, we can use the constructor index.
        // Generate: λ a b => ctorIdx a == ctorIdx b
        //
        // Since we don't have a direct ctorIdx function, we use a match approach:
        // For each constructor pair (ci, cj), return true if i == j, false otherwise.
        //
        // Generate BEq comparison using casesOn for multi-constructor inductives.
        // For enumerations: compare constructor tags using nested casesOn applications.
        // Pattern: casesOn a (casesOn b true false ...) (casesOn b false true ...) ...

        let body = if ctor_names.is_empty() {
            // No constructors - return true (vacuously)
            Expr::const_(Name::from_string("Bool.true"), vec![])
        } else if ctor_names.len() == 1 {
            // Single constructor - always equal
            Expr::const_(Name::from_string("Bool.true"), vec![])
        } else {
            // Multiple constructors - generate nested casesOn applications.
            // BEq.beq a b := IndName.casesOn a
            //   (IndName.casesOn b true false ...)   -- a = ctor0
            //   (IndName.casesOn b false true ...)   -- a = ctor1
            //   ...

            let rec_name = Name::from_string(&format!("{ind_name}.casesOn"));
            let bool_true = Expr::const_(Name::from_string("Bool.true"), vec![]);
            let bool_false = Expr::const_(Name::from_string("Bool.false"), vec![]);

            // Build the comparison for each constructor
            // For a with constructor i: b should also be constructor i
            let mut outer_cases = Vec::new();
            for i in 0..ctor_names.len() {
                // Case for b when a = ctor[i]
                let mut inner_cases = Vec::new();
                for j in 0..ctor_names.len() {
                    if i == j {
                        inner_cases.push(bool_true.clone());
                    } else {
                        inner_cases.push(bool_false.clone());
                    }
                }

                // Build: rec b case0 case1 ... caseN
                let mut b_match = Expr::app(
                    Expr::const_(rec_name.clone(), vec![]),
                    Expr::bvar(0), // b (innermost)
                );
                for case in inner_cases {
                    b_match = Expr::app(b_match, case);
                }
                outer_cases.push(b_match);
            }

            // Build: rec a (cases when a=ctor0) (cases when a=ctor1) ...
            let mut a_match = Expr::app(
                Expr::const_(rec_name, vec![]),
                Expr::bvar(1), // a (second from innermost)
            );
            for case in outer_cases {
                a_match = Expr::app(a_match, case);
            }

            a_match
        };

        // λ (a : Ind) (b : Ind) => body
        let inner_lam = Expr::lam(BinderInfo::Default, ind_type.clone(), body);
        let beq_func = Expr::lam(BinderInfo::Default, ind_type.clone(), inner_lam);

        // Instance value: BEq.mk beq_func
        let core_instance_val =
            Expr::app(Expr::const_(Name::from_string("BEq.mk"), vec![]), beq_func);

        // Wrap with lambdas for type parameters and constraints
        let instance_val =
            self.wrap_parametric_instance_value(core_instance_val, num_params, &class_name);

        DerivedInstance {
            name: instance_name,
            class_name,
            ty: instance_ty,
            val: instance_val,
            priority: 100,
        }
    }

    /// Derive Repr for an inductive type
    fn derive_repr_inductive(
        &mut self,
        ind_name: &Name,
        binders: &[SurfaceBinder],
        ctors: &[SurfaceCtor],
        ctor_names: &[Name],
    ) -> DerivedInstance {
        let instance_name = Name::from_string(&format!("inst{ind_name}Repr"));
        let class_name = Name::from_string("Repr");
        let num_params = binders.len();

        let (instance_ty, _num_constraints) =
            self.build_parametric_instance_type(ind_name, binders, &class_name);

        let ind_type = if num_params == 0 {
            Expr::const_(ind_name.clone(), vec![])
        } else {
            self.build_parametric_struct_type(ind_name, num_params, num_params)
        };

        // For Repr, generate reprPrec function that shows constructor name
        // λ (x : Ind) (prec : Nat) => match x with | ctor0 => "ctor0" | ...

        let body = if ctor_names.is_empty() {
            // No constructors - empty format
            Expr::const_(Name::from_string("Format.nil"), vec![])
        } else {
            // Generate casesOn that returns the constructor name as Format.text
            let rec_name = Name::from_string(&format!("{ind_name}.casesOn"));
            let mut result = Expr::app(
                Expr::const_(rec_name, vec![]),
                Expr::bvar(1), // x (second from innermost, after prec)
            );

            // For each constructor, append Format.text "CtorName"
            for (idx, _ctor_name) in ctor_names.iter().enumerate() {
                let ctor_str = &ctors[idx].name;
                let full_name = format!("{ind_name}.{ctor_str}");
                // Format.text "ctor_name"
                let format_text = Expr::app(
                    Expr::const_(Name::from_string("Format.text"), vec![]),
                    Expr::str_lit(full_name),
                );
                result = Expr::app(result, format_text);
            }

            result
        };

        // λ (x : Ind) (prec : Nat) => body
        let nat_type = Expr::const_(Name::from_string("Nat"), vec![]);
        let inner_lam = Expr::lam(BinderInfo::Default, nat_type, body);
        let repr_func = Expr::lam(BinderInfo::Default, ind_type.clone(), inner_lam);

        let core_instance_val = Expr::app(
            Expr::const_(Name::from_string("Repr.mk"), vec![]),
            repr_func,
        );

        let instance_val =
            self.wrap_parametric_instance_value(core_instance_val, num_params, &class_name);

        DerivedInstance {
            name: instance_name,
            class_name,
            ty: instance_ty,
            val: instance_val,
            priority: 100,
        }
    }

    /// Derive Hashable for an inductive type
    fn derive_hashable_inductive(
        &mut self,
        ind_name: &Name,
        binders: &[SurfaceBinder],
        _ctors: &[SurfaceCtor],
        ctor_names: &[Name],
    ) -> DerivedInstance {
        let instance_name = Name::from_string(&format!("inst{ind_name}Hashable"));
        let class_name = Name::from_string("Hashable");
        let num_params = binders.len();

        let (instance_ty, _num_constraints) =
            self.build_parametric_instance_type(ind_name, binders, &class_name);

        let ind_type = if num_params == 0 {
            Expr::const_(ind_name.clone(), vec![])
        } else {
            self.build_parametric_struct_type(ind_name, num_params, num_params)
        };

        // For Hashable, generate a hash function that returns constructor index
        // λ (x : Ind) => match x with | ctor0 => 0 | ctor1 => 1 | ...

        let body = if ctor_names.is_empty() {
            Expr::nat_lit(0)
        } else {
            let rec_name = Name::from_string(&format!("{ind_name}.casesOn"));
            let mut result = Expr::app(
                Expr::const_(rec_name, vec![]),
                Expr::bvar(0), // x (innermost)
            );

            for (idx, _) in ctor_names.iter().enumerate() {
                result = Expr::app(result, Expr::nat_lit(idx as u64));
            }

            result
        };

        // λ (x : Ind) => body
        let hash_func = Expr::lam(BinderInfo::Default, ind_type.clone(), body);

        let core_instance_val = Expr::app(
            Expr::const_(Name::from_string("Hashable.mk"), vec![]),
            hash_func,
        );

        let instance_val =
            self.wrap_parametric_instance_value(core_instance_val, num_params, &class_name);

        DerivedInstance {
            name: instance_name,
            class_name,
            ty: instance_ty,
            val: instance_val,
            priority: 100,
        }
    }

    /// Derive Inhabited for an inductive type
    ///
    /// Uses the first constructor as the default value.
    fn derive_inhabited_inductive(
        &mut self,
        ind_name: &Name,
        binders: &[SurfaceBinder],
        ctors: &[SurfaceCtor],
        ctor_names: &[Name],
    ) -> Option<DerivedInstance> {
        if ctor_names.is_empty() {
            // Can't derive Inhabited for an empty inductive
            return None;
        }

        let instance_name = Name::from_string(&format!("inst{ind_name}Inhabited"));
        let class_name = Name::from_string("Inhabited");
        let num_params = binders.len();

        let (instance_ty, _num_constraints) =
            self.build_parametric_instance_type(ind_name, binders, &class_name);

        // For Inhabited, use the first constructor
        // This works directly for nullary constructors.
        // For constructors with arguments, we'd need Inhabited instances for those types.

        // Check if first constructor is nullary by looking at its type
        let first_ctor = &ctors[0];
        let first_ctor_name = &ctor_names[0];

        // For simplicity, just use the constructor constant
        // This works for nullary constructors like Bool.false, Option.none, etc.
        let default_val = Expr::const_(first_ctor_name.clone(), vec![]);

        // Inhabited.mk default_val
        let core_instance_val = Expr::app(
            Expr::const_(Name::from_string("Inhabited.mk"), vec![]),
            default_val,
        );

        let instance_val =
            self.wrap_parametric_instance_value(core_instance_val, num_params, &class_name);

        // Store the first constructor name for potential debugging
        let _ = first_ctor;

        Some(DerivedInstance {
            name: instance_name,
            class_name,
            ty: instance_ty,
            val: instance_val,
            priority: 100,
        })
    }

    /// Derive DecidableEq for an inductive type
    fn derive_decidable_eq_inductive(
        &mut self,
        ind_name: &Name,
        binders: &[SurfaceBinder],
        _ctors: &[SurfaceCtor],
        ctor_names: &[Name],
    ) -> DerivedInstance {
        let instance_name = Name::from_string(&format!("inst{ind_name}DecidableEq"));
        let class_name = Name::from_string("DecidableEq");
        let num_params = binders.len();

        let (instance_ty, _num_constraints) =
            self.build_parametric_instance_type(ind_name, binders, &class_name);

        let ind_type = if num_params == 0 {
            Expr::const_(ind_name.clone(), vec![])
        } else {
            self.build_parametric_struct_type(ind_name, num_params, num_params)
        };

        // For DecidableEq on enumerations, we decide equality based on constructor.
        // Similar to BEq but returns Decidable instead of Bool.

        let body = if ctor_names.is_empty() {
            // No constructors - vacuously decidable
            Expr::app(
                Expr::const_(Name::from_string("Decidable.isTrue"), vec![]),
                Expr::const_(Name::from_string("Eq.refl"), vec![]),
            )
        } else if ctor_names.len() == 1 {
            // Single constructor - always equal
            Expr::app(
                Expr::const_(Name::from_string("Decidable.isTrue"), vec![]),
                Expr::const_(Name::from_string("Eq.refl"), vec![]),
            )
        } else {
            // For multiple constructors, use recursor-based matching.
            // This uses noConfusion to prove distinct constructors are unequal.
            //
            // For i ≠ j: The proof of ¬(Ctor_i = Ctor_j) is:
            //   λ h => IndName.noConfusion h
            // where noConfusion : {P : Sort u} → v1 = v2 → noConfusionType P v1 v2
            // When v1 and v2 are different constructors, noConfusionType P v1 v2 = P
            // So noConfusion specializes to (v1 = v2) → P, and with P = False
            // we get the required (v1 = v2) → False.

            let rec_name = Name::from_string(&format!("{ind_name}.casesOn"));
            let no_confusion_name = Name::from_string(&format!("{ind_name}.noConfusion"));
            let is_true = Expr::const_(Name::from_string("Decidable.isTrue"), vec![]);
            let is_false = Expr::const_(Name::from_string("Decidable.isFalse"), vec![]);
            let eq_refl = Expr::const_(Name::from_string("Eq.refl"), vec![]);

            // Build proof of ¬(a = b) using noConfusion: λ h => IndName.noConfusion h
            // This works when a and b are known to be different constructors.
            let no_confusion = Expr::const_(no_confusion_name, vec![]);
            let ne_proof = Expr::lam(
                BinderInfo::Default,
                // Type annotation for h: the equality type (simplified)
                Expr::const_(Name::from_string("Eq"), vec![]),
                // Body: noConfusion h (where h is bvar 0)
                Expr::app(no_confusion.clone(), Expr::bvar(0)),
            );

            let mut outer_cases = Vec::new();
            for i in 0..ctor_names.len() {
                let mut inner_cases = Vec::new();
                for j in 0..ctor_names.len() {
                    if i == j {
                        // Same constructor: isTrue Eq.refl
                        inner_cases.push(Expr::app(is_true.clone(), eq_refl.clone()));
                    } else {
                        // Different constructors: isFalse (λ h => noConfusion h)
                        inner_cases.push(Expr::app(is_false.clone(), ne_proof.clone()));
                    }
                }

                let mut b_match = Expr::app(Expr::const_(rec_name.clone(), vec![]), Expr::bvar(0));
                for case in inner_cases {
                    b_match = Expr::app(b_match, case);
                }
                outer_cases.push(b_match);
            }

            let mut a_match = Expr::app(Expr::const_(rec_name, vec![]), Expr::bvar(1));
            for case in outer_cases {
                a_match = Expr::app(a_match, case);
            }

            a_match
        };

        // λ (a : Ind) (b : Ind) => body
        let inner_lam = Expr::lam(BinderInfo::Default, ind_type.clone(), body);
        let dec_eq_func = Expr::lam(BinderInfo::Default, ind_type.clone(), inner_lam);

        let core_instance_val = Expr::app(
            Expr::const_(Name::from_string("DecidableEq.mk"), vec![]),
            dec_eq_func,
        );

        let instance_val =
            self.wrap_parametric_instance_value(core_instance_val, num_params, &class_name);

        DerivedInstance {
            name: instance_name,
            class_name,
            ty: instance_ty,
            val: instance_val,
            priority: 100,
        }
    }

    // Note: Recursor generation is now handled by the kernel in env.rs.
    // The kernel builds recursors (rec, casesOn) during add_inductive() and stores them
    // in the environment. They can be queried via env.get_recursor("Type.rec").

    /// Extract the type name from an expression (for casesOn lookup).
    /// Returns the base name of the type constructor.
    fn get_type_name(&self, ty: &Expr) -> Result<String, ElabError> {
        let ty = self.whnf(ty);
        match &ty {
            Expr::Const(name, _) => Ok(name.to_string()),
            Expr::App(func, _) => {
                // Recurse on the function to get the base type name
                // e.g., `Option Nat` -> `Option`
                self.get_type_name(func)
            }
            _ => Err(ElabError::NotImplemented(format!(
                "cannot extract type name from {ty:?}"
            ))),
        }
    }

    /// Wrap an expression in lambdas corresponding to pattern variables.
    /// Used to build case branches that bind pattern-matched arguments.
    fn wrap_pattern_lambdas(
        &mut self,
        pats: &[SurfacePattern],
        body: Expr,
    ) -> Result<Expr, ElabError> {
        // Build lambdas from right to left (innermost first)
        let mut result = body;
        for pat in pats.iter().rev() {
            match pat {
                SurfacePattern::Var(name) => {
                    // Bind this variable with a placeholder type
                    // In a full implementation, we'd look up the constructor's
                    // argument types to get the correct type here
                    let hole = self.fresh_meta(Expr::type_());
                    result = Expr::lam(BinderInfo::Default, hole, result);
                    // Note: the name is used for error messages but de Bruijn
                    // indices handle the actual binding
                    let _ = name;
                }
                SurfacePattern::Wildcard => {
                    // Wildcard binds but doesn't name the value
                    let hole = self.fresh_meta(Expr::type_());
                    result = Expr::lam(BinderInfo::Default, hole, result);
                }
                _ => {
                    return Err(ElabError::NotImplemented(format!(
                        "nested pattern in if-let: {pat:?}"
                    )));
                }
            }
        }
        Ok(result)
    }
}

/// A derived instance generated from a `deriving` clause
#[derive(Debug, Clone)]
pub struct DerivedInstance {
    /// Instance name (e.g., "instReprPoint")
    pub name: Name,
    /// Class name (e.g., "Repr")
    pub class_name: Name,
    /// Instance type (e.g., Repr Point)
    pub ty: Expr,
    /// Instance value
    pub val: Expr,
    /// Priority (default: 100)
    pub priority: u32,
}

// Note: Recursor information is provided by the kernel's RecursorVal type.
// See lean5_kernel::RecursorVal after calling env.add_inductive().

/// Result of elaboration
#[derive(Debug)]
pub enum ElabResult {
    Definition {
        name: Name,
        universe_params: Vec<Name>,
        ty: Expr,
        val: Expr,
    },
    Theorem {
        name: Name,
        universe_params: Vec<Name>,
        ty: Expr,
        proof: Expr,
    },
    Axiom {
        name: Name,
        universe_params: Vec<Name>,
        ty: Expr,
    },
    /// Inductive type declaration (multiple constructors)
    ///
    /// E.g.:
    /// ```text
    /// inductive List (α : Type) : Type
    /// | nil : List α
    /// | cons : α → List α → List α
    /// ```
    Inductive {
        /// Inductive type name
        name: Name,
        /// Universe parameters
        universe_params: Vec<Name>,
        /// Number of parameters
        num_params: u32,
        /// Inductive type
        ty: Expr,
        /// Constructors: (name, type)
        constructors: Vec<(Name, Expr)>,
        /// Derived type class instances
        derived_instances: Vec<DerivedInstance>,
        // Note: Recursors (rec, casesOn) are generated by the kernel during add_inductive
        // and can be queried via env.get_recursor("Type.rec") or env.get_recursor("Type.casesOn")
    },
    /// Structure declaration (single-constructor inductive with named fields)
    Structure {
        /// Structure name
        name: Name,
        /// Universe parameters
        universe_params: Vec<Name>,
        /// Number of parameters
        num_params: u32,
        /// Structure type
        ty: Expr,
        /// Constructor name
        ctor_name: Name,
        /// Constructor type (includes parameters and fields)
        ctor_ty: Expr,
        /// Field names (in order)
        field_names: Vec<Name>,
        /// Projection functions: (name, type, value) for each field
        /// E.g., for `structure Point where x : Nat  y : Nat`:
        /// - Point.x : Point → Nat, λ s => s.0
        /// - Point.y : Point → Nat, λ s => s.1
        projections: Vec<(Name, Expr, Expr)>,
        /// Derived type class instances
        /// E.g., for `deriving Repr, BEq`, contains generated instance definitions
        derived_instances: Vec<DerivedInstance>,
    },
    /// Type class instance declaration
    ///
    /// An instance provides an implementation of a type class for specific types.
    /// E.g., `instance : Add Nat where add := Nat.add`
    Instance {
        /// Instance name (auto-generated if not provided)
        name: Name,
        /// Universe parameters
        universe_params: Vec<Name>,
        /// The class name this instance implements
        class_name: Name,
        /// The instance type (e.g., `Add Nat`)
        ty: Expr,
        /// The instance value (structure constructor applied to field values)
        val: Expr,
        /// Instance priority (higher = tried first)
        priority: u32,
    },
    /// Skipped declaration (not yet implemented or module-level)
    Skipped,
}

fn convert_binder_info(info: SurfaceBinderInfo) -> BinderInfo {
    match info {
        SurfaceBinderInfo::Explicit => BinderInfo::Default,
        SurfaceBinderInfo::Implicit => BinderInfo::Implicit,
        SurfaceBinderInfo::StrictImplicit => BinderInfo::StrictImplicit,
        SurfaceBinderInfo::Instance => BinderInfo::InstImplicit,
    }
}

/// Check if a surface expression is an outParam wrapper
fn is_out_param_type(expr: &SurfaceExpr) -> bool {
    matches!(expr, SurfaceExpr::OutParam(_, _))
}

/// Check if a surface expression is a semiOutParam wrapper
fn is_semi_out_param_type(expr: &SurfaceExpr) -> bool {
    matches!(expr, SurfaceExpr::SemiOutParam(_, _))
}

#[cfg(test)]
mod tests;
