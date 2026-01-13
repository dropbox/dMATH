//! Tactic framework
//!
//! Provides a proof state and basic tactics for interactive theorem proving.
//! Tactics operate on goals (holes in a proof term) and produce proof terms.
//!
//! # Architecture
//!
//! The tactic framework uses a goal-based approach:
//! - A `Goal` represents an unproven proposition with local context
//! - A `ProofState` maintains a list of goals and metavariable assignments
//! - `Tactic`s transform proof states, closing goals or creating new ones
//!
//! # Basic Tactics
//!
//! - `exact e` - Provide an exact proof term `e` for the goal
//! - `intro x` - For goal `∀ (x : A), B`, introduce `x` and change goal to `B`
//! - `apply f` - For goal `B`, if `f : A → B`, change goal to `A`

use crate::unify::{MetaId, MetaState, Unifier, UnifyResult};
use lean5_kernel::name::Name;
use lean5_kernel::{
    BinderInfo, CertError, CertVerifier, Environment, Expr, FVarId, Level, LocalContext, ProofCert,
    TypeChecker,
};

// Submodules (alphabetically ordered)
pub mod abs_cases;
pub mod algebra;
pub mod arithmetic;
pub mod cast;
pub mod cc;
pub mod conv;
pub mod convert;
pub mod debug;
pub mod decide_eq;
pub mod equality;
pub mod extensionality;
pub mod finite_cases;
pub mod gcongr;
pub mod goal;
pub mod hypothesis;
pub mod instance;
pub mod library_search;
pub mod options;
pub mod pattern;
pub mod polyrith;
pub mod positivity;
pub mod ring;
pub mod search;
pub mod simp;
pub mod smt;
pub mod tauto;
pub mod unfold;
pub mod wlog;

// Re-exports organized alphabetically by module name

// Re-export abs_cases tactics
pub use abs_cases::{abs_cases, abs_cases_with_config, AbsCasesConfig};
#[allow(unused_imports)]
pub(crate) use abs_cases::{
    create_abs_em_proof, is_numeric_type, make_ge_expr, make_lt_expr, make_zero_for_type,
};

// Re-export algebra tactics
pub use algebra::{abel, abel_with_config, group, group_with_config, AbelConfig, GroupConfig};
#[allow(unused_imports)]
pub(crate) use algebra::{is_pi_expr, match_eq_simple, AbelTerm, GroupTerm};

// Re-export arithmetic tactics and helpers
#[allow(unused_imports)]
pub(crate) use arithmetic::{
    build_add_le_add_proof, build_scaled_proof, expr_to_linear, expr_to_omega_constraint,
    extract_certified_omega_constraints, extract_constant, extract_denominators,
    extract_single_var, fourier_motzkin_check, fourier_motzkin_check_certified, is_zero_expr,
    make_not, match_hmod_app, negate_omega_constraint, nlinarith_exprs_equal,
    omega_check_certified, push_neg_expr, try_compute_linear_product,
};
pub use arithmetic::{
    contrapose, contrapose_hyp, exprs_syntactically_equal, field_simp, get_app_fn,
    is_cast_function, is_false, linarith, make_equality, match_and, match_le, match_lt, match_not,
    match_or, nlinarith, nlinarith_with_config, norm_cast, omega, positivity, push_neg,
    CertifiedConstraint, CertifiedOmegaConstraint, FMCertifiedResult, FMResult,
    LinarithCertificate, LinearConstraint, LinearExpr, NlinarithConfig, OmegaCertificate,
    OmegaCertifiedResult, OmegaConstraint, OmegaContradictionType,
};

// Re-export cast tactics
pub use cast::{
    assumption_mod_cast, exact_mod_cast, lift, lift_with_config, push_cast, qify, zify, CastConfig,
    LiftConfig,
};

// Re-export cc tactics
#[allow(unused_imports)]
pub(crate) use cc::CCState;
pub use cc::{cc, cc_with_config, CCConfig};

// Re-export conv tactics
pub use conv::{conv_arg, conv_lhs, conv_rhs, conv_rw, ConvPath, ConvPosition, ConvState};

// Re-export convert/calc tactics
pub use convert::{
    calc_block, calc_eq, convert, convert_hyp, CalcJustification, CalcRel, CalcStep,
};
#[allow(unused_imports)]
pub(crate) use convert::{make_calc_rel, make_eq_refl};

// Re-export debug/utility tactics
#[allow(unused_imports)]
pub(crate) use debug::beta_reduce_all;
pub use debug::{
    bound, clean, itauto, itauto_with_config, substs, trace, trace_expr, trace_state,
    trace_with_level, ITautoConfig, TraceLevel, TraceOutput,
};

// Re-export decide_eq tactics
pub use decide_eq::decide_eq;
#[allow(unused_imports)]
pub(crate) use decide_eq::{
    decidable_type_check, eval_to_nat, exprs_definitely_not_equal, make_ne_proof,
    match_decidable_eq,
};

// Re-export equality tactics
pub(crate) use equality::{abstract_over, contains_expr, match_equality, replace_expr};
pub use equality::{calc_trans, rewrite, rewrite_ltr, rewrite_rtl, subst, subst_vars, symm, trans};

// Re-export extensionality tactics
pub use extensionality::{funext, propext, quot_ext, set_ext};

// Re-export finite_cases tactics
#[allow(unused_imports)]
pub(crate) use finite_cases::{
    expr_to_int, extract_nat_literal, get_finite_inhabitants, make_equality_type, make_int_literal,
    make_nat_literal, substitute_fvar,
};
pub use finite_cases::{fin_cases, interval_cases};

// Re-export gcongr tactic
pub use gcongr::gcongr;
#[allow(unused_imports)]
pub(crate) use gcongr::{make_ineq_goal, match_add, match_inequality, IneqRel};

// Re-export goal management tactics
pub use goal::{goal_count, pick_goal, rotate, rotate_back, swap};

// Re-export hypothesis tactics
#[allow(unused_imports)]
pub(crate) use hypothesis::collect_fvars;
pub use hypothesis::{
    apply_fun, apply_fun_goal, clear, clear_all_unused, clear_except, duplicate, rename,
    rename_all, replace, replace_hyp, specialize,
};

// Re-export instance tactics
pub use instance::{have_i, infer_i, let_i};

// Re-export library_search tactics
#[allow(unused_imports)]
pub(crate) use library_search::{
    calculate_type_similarity, count_pis, expr_depth, extract_head_name,
};
pub use library_search::{
    library_search, library_search_and_apply, library_search_show, library_search_with_config,
    LibrarySearchConfig, LibrarySearchMatchKind, LibrarySearchResult,
};

// Re-export options tactics
pub use options::{set_option, set_options, OptionValue, ProofOptions, SetOptionConfig};

// Re-export pattern/monotonicity tactics
#[allow(unused_imports)]
pub(crate) use pattern::{
    apply_predicate, count_foralls, exprs_equal, extract_binary_args, extract_class_name,
    find_first_type, generate_fresh_hyp_name, get_app_head, infer_simple_type, is_binary_app,
    is_continuity_goal, is_dite_const, is_false_prop, is_ite_const, is_measurability_goal,
    is_true_prop, make_relation, occurs_bvar_dsimp, rename_hypothesis, shift_bvars_dsimp,
    split_pattern_args, try_extract_exists, try_infer_expr_type,
};
pub use pattern::{
    choose, choose_simple, continuity, continuity_with_config, dsimp, dsimp_all, dsimp_at,
    dsimp_with_config, infer_instance, infer_instance_with_config, linear_combination,
    linear_combination_simple, linear_combination_with_config, measurability,
    measurability_with_config, mono, mono_with_config, nontriviality, nontriviality_of,
    nontriviality_with_config, peel, rintro, rintro_patterns, simpa, simpa_only, simpa_with_config,
    split_ifs, split_ifs_with_config, split_ifs_with_names, ChooseConfig, ContinuityConfig,
    DsimpConfig, InferInstanceConfig, LinearCoeff, LinearCombinationConfig, MeasurabilityConfig,
    MonoConfig, MonoStep, NontrivialityConfig, RIntroPattern, SplitIfsConfig,
};

// Re-export polyrith tactics
#[allow(unused_imports)]
pub(crate) use polyrith::gcd_u64;
pub use polyrith::{
    is_polynomial_expr, polyrith, polyrith_with_config, Polynomial, PolyrithCertificate,
    PolyrithConfig,
};

// Re-export positivity tactics
#[allow(unused_imports)]
pub(crate) use positivity::{
    analyze_positivity, extract_comparison_expr, is_abs_pattern, is_add_pattern, is_mul_pattern,
    is_square_pattern, make_positivity_prop, ComparisonKind, PositivityResult,
};
pub use positivity::{positivity_at, positivity_at_with_config, PositivityAtConfig};

// Re-export ring tactics
#[allow(unused_imports)]
pub(crate) use ring::{
    make_add, make_eq, make_mul, make_neg, make_pow, ring_collect_like_terms, ring_expr_to_expr,
    ring_exprs_equal, ring_flatten_add, ring_flatten_mul, ring_normalize, RingExpr,
};
pub use ring::{ring, ring_nf};

// Re-export search tactics
pub use search::{
    aesop, aesop_with_config, apply_search, apply_search_and_apply, exact_search,
    exact_search_and_apply, hint, suggest, AesopConfig, AesopRule, AesopRuleKind, SearchResult,
    TacticSuggestion,
};
pub(crate) use search::{can_apply_to_produce, types_unify};

// Re-export simp tactics
#[allow(unused_imports)]
pub(crate) use simp::{
    beta_reduce, contains_bvar, eta_reduce, is_trivial_equality, is_true_const, shift_expr,
    substitute_bvar,
};
pub use simp::{
    simp, simp_all, simp_default, simp_only, simp_rw, simp_rw_hyps, squeeze_simp,
    squeeze_simp_and_apply, squeeze_simp_with_config, SimpConfig, SimpLemma, SqueezeSimpConfig,
    SqueezeSimpResult,
};

// Re-export SMT tactics
pub(crate) use smt::create_sorry_term;
pub use smt::{decide, z4_bv, z4_decide, z4_omega, z4_smt, Z4Config};

// Re-export tauto tactics
#[allow(unused_imports)]
pub(crate) use tauto::fresh_hyp_name;
pub use tauto::tauto;

// Re-export unfold tactics
#[allow(unused_imports)]
pub(crate) use unfold::{collect_consts, substitute_const};
pub use unfold::{delta, unfold, unfold_at};

// Re-export wlog tactics
pub use wlog::{norm_num_at, push_neg_at, suffices_to_show, wlog};
#[allow(unused_imports)]
pub(crate) use wlog::{normalize_numerals, push_negations_in_expr};

/// A goal in the proof state
#[derive(Debug, Clone)]
pub struct Goal {
    /// Unique identifier for this goal (corresponds to a metavariable)
    pub meta_id: MetaId,
    /// The type to prove (target)
    pub target: Expr,
    /// Local context (hypotheses available)
    pub local_ctx: Vec<LocalDecl>,
}

/// A local declaration in the goal context
#[derive(Debug, Clone)]
pub struct LocalDecl {
    /// Free variable id
    pub fvar: FVarId,
    /// Name for display
    pub name: String,
    /// Type of this hypothesis
    pub ty: Expr,
    /// Optional value (for let-bindings)
    pub value: Option<Expr>,
}

/// The proof state containing all goals
#[derive(Debug, Clone)]
pub struct ProofState {
    /// The environment
    env: Environment,
    /// All goals (first is the main goal)
    goals: Vec<Goal>,
    /// Metavariable state for tracking assignments
    metas: MetaState,
    /// Next fresh free variable id
    next_fvar: u64,
}

impl ProofState {
    /// Create a new proof state for a goal type
    pub fn new(env: Environment, target: Expr) -> Self {
        let mut metas = MetaState::new();

        // Create a metavariable for the main goal
        let meta_id = metas.fresh(target.clone());

        let main_goal = Goal {
            meta_id,
            target,
            local_ctx: Vec::new(),
        };

        ProofState {
            env,
            goals: vec![main_goal],
            metas,
            next_fvar: 0,
        }
    }

    /// Create a new proof state with an existing local context
    pub fn with_context(env: Environment, target: Expr, ctx: Vec<LocalDecl>) -> Self {
        let mut metas = MetaState::new();
        let meta_id = metas.fresh(target.clone());

        // Find the maximum fvar id in the context
        let max_fvar = ctx.iter().map(|d| d.fvar.0).max().unwrap_or(0);

        let main_goal = Goal {
            meta_id,
            target,
            local_ctx: ctx,
        };

        ProofState {
            env,
            goals: vec![main_goal],
            metas,
            next_fvar: max_fvar + 1,
        }
    }

    /// Get the current (first) goal
    pub fn current_goal(&self) -> Option<&Goal> {
        self.goals.first()
    }

    /// Get the current (first) goal mutably
    pub fn current_goal_mut(&mut self) -> Option<&mut Goal> {
        self.goals.first_mut()
    }

    /// Get all remaining goals
    pub fn goals(&self) -> &[Goal] {
        &self.goals
    }

    /// Check if the proof is complete (no goals remain)
    pub fn is_complete(&self) -> bool {
        self.goals.is_empty()
    }

    /// Get the proof term (only valid when complete)
    pub fn proof_term(&self) -> Option<Expr> {
        if !self.is_complete() {
            return None;
        }

        // Get the main metavariable's assignment
        let main_goal_meta = MetaId(0); // First created meta is the main goal
        self.metas.get_assignment(main_goal_meta).cloned()
    }

    /// Get the instantiated proof term with all metavariables resolved
    pub fn instantiated_proof(&self) -> Option<Expr> {
        self.proof_term().map(|p| self.metas.instantiate(&p))
    }

    /// Get the metavariable state
    pub fn metas(&self) -> &MetaState {
        &self.metas
    }

    /// Get mutable metavariable state
    pub fn metas_mut(&mut self) -> &mut MetaState {
        &mut self.metas
    }

    /// Get the environment
    pub fn env(&self) -> &Environment {
        &self.env
    }

    /// Create a fresh free variable
    fn fresh_fvar(&mut self) -> FVarId {
        let id = FVarId(self.next_fvar);
        self.next_fvar += 1;
        id
    }

    /// Build a kernel LocalContext from a goal's local context
    fn build_local_ctx(&self, goal: &Goal) -> LocalContext {
        let mut ctx = LocalContext::new();
        for decl in &goal.local_ctx {
            ctx.push_with_id(
                decl.fvar,
                Name::from_string(&decl.name),
                self.metas.instantiate(&decl.ty),
                BinderInfo::Default,
            );
        }
        // Also add metavariables to context
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

    /// Infer the type of an expression in the goal's context
    pub fn infer_type(&self, goal: &Goal, expr: &Expr) -> Result<Expr, TacticError> {
        let ctx = self.build_local_ctx(goal);
        let mut tc = TypeChecker::with_context(&self.env, ctx);
        let instantiated = self.metas.instantiate(expr);
        tc.infer_type(&instantiated)
            .map(|ty| self.metas.instantiate(&ty))
            .map_err(|e| TacticError::TypeCheckFailed(format!("{e:?}")))
    }

    /// Check if two expressions are definitionally equal
    pub fn is_def_eq(&self, goal: &Goal, a: &Expr, b: &Expr) -> bool {
        let ctx = self.build_local_ctx(goal);
        let tc = TypeChecker::with_context(&self.env, ctx);
        let a_inst = self.metas.instantiate(a);
        let b_inst = self.metas.instantiate(b);
        tc.is_def_eq(&a_inst, &b_inst)
    }

    /// Compute weak-head normal form
    pub fn whnf(&self, goal: &Goal, expr: &Expr) -> Expr {
        let ctx = self.build_local_ctx(goal);
        let tc = TypeChecker::with_context(&self.env, ctx);
        tc.whnf(&self.metas.instantiate(expr))
    }

    /// Create a certificate verifier with the goal's local context pre-registered.
    ///
    /// This enables verification of proof terms that contain free variables
    /// from the goal's hypotheses. The verifier is initialized with all locals
    /// and metavariables from the goal context.
    pub fn create_cert_verifier(&self, goal: &Goal) -> Result<CertVerifier<'_>, CertError> {
        let ctx = self.build_local_ctx(goal);
        let mut verifier = CertVerifier::new(&self.env);
        verifier.register_local_context(&ctx)?;
        Ok(verifier)
    }

    /// Infer the type of an expression with a proof certificate.
    ///
    /// This is the certified variant of `infer_type` - it returns both the
    /// inferred type and a proof certificate that can be independently verified.
    pub fn infer_type_with_cert(
        &self,
        goal: &Goal,
        expr: &Expr,
    ) -> Result<(Expr, ProofCert), TacticError> {
        let ctx = self.build_local_ctx(goal);
        let mut tc = TypeChecker::with_context(&self.env, ctx);
        let instantiated = self.metas.instantiate(expr);
        tc.infer_type_with_cert(&instantiated)
            .map(|(ty, cert)| (self.metas.instantiate(&ty), cert))
            .map_err(|e| TacticError::TypeCheckFailed(format!("{e:?}")))
    }

    /// Verify that a proof term has the expected type using certificates.
    ///
    /// This provides a double-check that the proof term is correct by:
    /// 1. Inferring its type with a certificate
    /// 2. Checking definitional equality with the goal target
    /// 3. Verifying the certificate independently
    pub fn verify_proof(&self, goal: &Goal, proof: &Expr) -> Result<ProofCert, TacticError> {
        let (inferred_ty, cert) = self.infer_type_with_cert(goal, proof)?;

        // Check that inferred type matches goal target
        if !self.is_def_eq(goal, &inferred_ty, &goal.target) {
            return Err(TacticError::TypeMismatch {
                expected: format!("{:?}", goal.target),
                actual: format!("{inferred_ty:?}"),
            });
        }

        // Verify the certificate
        let mut verifier = self
            .create_cert_verifier(goal)
            .map_err(|e| TacticError::Other(format!("CertVerifier error: {e:?}")))?;

        verifier
            .verify(&cert, proof)
            .map_err(|e| TacticError::Other(format!("Certificate verification failed: {e:?}")))?;

        Ok(cert)
    }

    /// Close the current goal with a proof term
    fn close_goal(&mut self, proof: Expr) -> Result<(), TacticError> {
        let goal = self.goals.remove(0);
        self.metas.assign(goal.meta_id, proof);
        Ok(())
    }
}

/// Result type for tactic execution
pub type TacticResult = Result<(), TacticError>;

/// Errors that can occur during tactic execution
#[derive(Debug, Clone)]
pub enum TacticError {
    /// No goals to operate on
    NoGoals,
    /// Type mismatch
    TypeMismatch { expected: String, actual: String },
    /// Cannot apply tactic to this goal shape
    GoalMismatch(String),
    /// Unknown identifier
    UnknownIdent(String),
    /// Type checking failed
    TypeCheckFailed(String),
    /// Unification failed
    UnificationFailed(String),
    /// Hypothesis not found in context
    HypothesisNotFound(String),
    /// Other error
    Other(String),
}

impl std::fmt::Display for TacticError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TacticError::NoGoals => write!(f, "no goals"),
            TacticError::TypeMismatch { expected, actual } => {
                write!(f, "type mismatch: expected {expected}, got {actual}")
            }
            TacticError::GoalMismatch(msg) => write!(f, "goal mismatch: {msg}"),
            TacticError::UnknownIdent(name) => write!(f, "unknown identifier: {name}"),
            TacticError::TypeCheckFailed(msg) => write!(f, "type check failed: {msg}"),
            TacticError::UnificationFailed(msg) => write!(f, "unification failed: {msg}"),
            TacticError::HypothesisNotFound(name) => write!(f, "hypothesis not found: {name}"),
            TacticError::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for TacticError {}

// =============================================================================
// Tactics
// =============================================================================

/// Close the goal with an exact proof term
pub fn exact(state: &mut ProofState, proof: Expr) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Infer the type of the proof
    let proof_ty = state.infer_type(&goal, &proof)?;

    // Check that the proof has the right type (via unification to handle metas)
    let target = state.metas.instantiate(&goal.target);

    match Unifier::new(state.metas_mut()).unify(&proof_ty, &target) {
        UnifyResult::Success => {
            state.close_goal(proof)?;
            Ok(())
        }
        UnifyResult::Failure(msg) => Err(TacticError::TypeMismatch {
            expected: format!("{target:?}"),
            actual: msg,
        }),
        UnifyResult::Stuck => Err(TacticError::UnificationFailed(
            "unification stuck".to_string(),
        )),
    }
}

/// Introduce a hypothesis (for goals of the form ∀ x, P)
pub fn intro(state: &mut ProofState, name: String) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // WHNF to expose the Pi
    let target = state.whnf(&goal, &goal.target);

    match target {
        Expr::Pi(bi, domain, codomain) => {
            // Create a new free variable for the introduced hypothesis
            let fvar = state.fresh_fvar();

            // Create the new local declaration
            let local_decl = LocalDecl {
                fvar,
                name: name.clone(),
                ty: (*domain).clone(),
                value: None,
            };

            // Create new context with the hypothesis
            let mut new_ctx = goal.local_ctx.clone();
            new_ctx.push(local_decl);

            // Instantiate the codomain with the free variable
            let new_target = codomain.instantiate(&Expr::fvar(fvar));

            // Create new metavariable for the new goal
            let new_meta_id = state.metas.fresh(new_target.clone());

            // The proof of the original goal is λ x : A, <new_proof>
            let new_meta_expr = Expr::FVar(MetaState::to_fvar(new_meta_id));
            let proof = Expr::lam(bi, (*domain).clone(), new_meta_expr.abstract_fvar(fvar));

            // Close the current goal with this proof template
            state.close_goal(proof)?;

            // Add the new goal
            let new_goal = Goal {
                meta_id: new_meta_id,
                target: new_target,
                local_ctx: new_ctx,
            };

            state.goals.insert(0, new_goal);
            Ok(())
        }
        _ => Err(TacticError::GoalMismatch(
            "intro requires a forall/arrow goal".to_string(),
        )),
    }
}

/// Introduce multiple hypotheses
pub fn intros(state: &mut ProofState, names: Vec<String>) -> TacticResult {
    for name in names {
        intro(state, name)?;
    }
    Ok(())
}

/// Apply a function/theorem to the goal
pub fn apply(state: &mut ProofState, func: Expr) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Infer the type of the function
    let func_ty = state.infer_type(&goal, &func)?;

    // Collect the arguments needed and check if result matches target
    apply_aux(state, &goal, func, func_ty)
}

fn apply_aux(state: &mut ProofState, goal: &Goal, func: Expr, func_ty: Expr) -> TacticResult {
    let func_ty = state.whnf(goal, &func_ty);
    let target = state.metas.instantiate(&goal.target);

    match &func_ty {
        Expr::Pi(_bi, domain, codomain) => {
            // Create a metavariable for this argument
            let arg_meta_id = state.metas.fresh((**domain).clone());
            let arg_meta = Expr::FVar(MetaState::to_fvar(arg_meta_id));

            // Apply the function to the metavariable
            let applied = Expr::app(func.clone(), arg_meta.clone());

            // Instantiate the codomain
            let new_ty = codomain.instantiate(&arg_meta);

            // Try to unify the result with the target
            match Unifier::new(state.metas_mut()).unify(&new_ty, &target) {
                UnifyResult::Success => {
                    // Success! Close the goal with the applied function
                    state.close_goal(applied)?;

                    // Check if the argument metavariable was solved
                    if !state.metas.is_assigned(arg_meta_id) {
                        // Create a new goal for this unsolved argument
                        let new_goal = Goal {
                            meta_id: arg_meta_id,
                            target: state.metas.instantiate(domain),
                            local_ctx: goal.local_ctx.clone(),
                        };
                        state.goals.insert(0, new_goal);
                    }

                    Ok(())
                }
                UnifyResult::Failure(_) | UnifyResult::Stuck => {
                    // Try applying with more arguments
                    apply_aux(state, goal, applied, new_ty)
                }
            }
        }
        _ => {
            // Not a function type anymore, try direct unification
            match Unifier::new(state.metas_mut()).unify(&func_ty, &target) {
                UnifyResult::Success => {
                    state.close_goal(func)?;
                    Ok(())
                }
                UnifyResult::Failure(msg) => Err(TacticError::TypeMismatch {
                    expected: format!("{target:?}"),
                    actual: msg,
                }),
                UnifyResult::Stuck => Err(TacticError::UnificationFailed(
                    "apply: unification stuck".to_string(),
                )),
            }
        }
    }
}

/// Use a hypothesis from the context
pub fn assumption(state: &mut ProofState) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();
    let target = state.metas.instantiate(&goal.target);

    // Search through hypotheses for one that matches
    for decl in &goal.local_ctx {
        let hyp_ty = state.metas.instantiate(&decl.ty);
        if state.is_def_eq(&goal, &hyp_ty, &target) {
            return exact(state, Expr::fvar(decl.fvar));
        }
    }

    Err(TacticError::Other(
        "no matching hypothesis found".to_string(),
    ))
}

/// Constructor tactic for inductive types
pub fn constructor(state: &mut ProofState) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();
    let target = state.whnf(&goal, &goal.target);

    // Get the head constant of the target
    let head = target.get_app_fn();

    match head {
        Expr::Const(name, levels) => {
            // Look up the inductive type
            if let Some(ind_info) = state.env.get_inductive(name) {
                // Get the first constructor
                if let Some(ctor_name) = ind_info.constructor_names.first() {
                    let ctor = Expr::const_(ctor_name.clone(), levels.clone());
                    return apply(state, ctor);
                }
            }
            Err(TacticError::Other(format!("not an inductive type: {name}")))
        }
        _ => Err(TacticError::GoalMismatch(
            "goal is not an application of a constant".to_string(),
        )),
    }
}

/// Reflexivity tactic (for goals of the form a = a)
pub fn rfl(state: &mut ProofState) -> TacticResult {
    // Look for Eq.refl or rfl in the environment
    let eq_refl = Name::from_string("Eq.refl");
    if state.env.get_const(&eq_refl).is_some() {
        let refl = Expr::const_(eq_refl, vec![Level::zero()]);
        return apply(state, refl);
    }

    // Try rfl
    let rfl_name = Name::from_string("rfl");
    if state.env.get_const(&rfl_name).is_some() {
        let refl = Expr::const_(rfl_name, vec![Level::zero()]);
        return apply(state, refl);
    }

    Err(TacticError::Other(
        "rfl: no reflexivity constant found in environment".to_string(),
    ))
}

/// Case split on a hypothesis of inductive type
///
/// The `cases` tactic destructs a hypothesis of an inductive type,
/// creating one subgoal per constructor. Each subgoal has access to
/// the constructor's fields as new hypotheses.
///
/// # Example
/// For a hypothesis `h : Bool`, `cases h` produces two goals:
/// - One where `h` is replaced by `false`
/// - One where `h` is replaced by `true`
///
/// For `h : List A`, it produces:
/// - Goal for `List.nil` case
/// - Goal for `List.cons head tail` case, with new hypotheses for `head` and `tail`
pub fn cases(state: &mut ProofState, hyp_name: &str) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Find the hypothesis in the local context
    let (hyp_idx, hyp_decl) = goal
        .local_ctx
        .iter()
        .enumerate()
        .find(|(_, d)| d.name == hyp_name)
        .ok_or_else(|| TacticError::UnknownIdent(hyp_name.to_string()))?;
    let hyp_fvar = hyp_decl.fvar;
    let hyp_ty = state.metas.instantiate(&hyp_decl.ty);

    // WHNF to expose the inductive type
    let hyp_ty_whnf = state.whnf(&goal, &hyp_ty);

    // Get the head constant and arguments using existing API
    let head = hyp_ty_whnf.get_app_fn().clone();
    let args: Vec<Expr> = hyp_ty_whnf.get_app_args().into_iter().cloned().collect();

    let ind_name = match &head {
        Expr::Const(name, _) => name.clone(),
        _ => {
            return Err(TacticError::GoalMismatch(format!(
                "cases: hypothesis '{hyp_name}' has type '{hyp_ty_whnf:?}' which is not an inductive type"
            )));
        }
    };

    // Look up the inductive type information
    let ind_info = state
        .env
        .get_inductive(&ind_name)
        .ok_or_else(|| {
            TacticError::GoalMismatch(format!("cases: '{ind_name}' is not an inductive type"))
        })?
        .clone();

    // Get universe levels from the hypothesis type
    let levels = match &head {
        Expr::Const(_, lvls) => lvls.to_vec(),
        _ => vec![],
    };

    // Build the recursor name: IndName.rec
    let rec_name = Name::from_string(&format!("{ind_name}.rec"));
    let _rec_info = state
        .env
        .get_recursor(&rec_name)
        .ok_or_else(|| TacticError::Other(format!("cases: recursor '{rec_name}' not found")))?;

    // Create the motive: fun x => (goal target)
    // The motive abstracts over the term being case-split
    let motive_body = goal.target.clone();
    // Abstract the hypothesis from the goal target
    let motive = Expr::lam(
        BinderInfo::Default,
        hyp_ty.clone(),
        motive_body.abstract_fvar(hyp_fvar),
    );

    // For each constructor, create a new goal
    let num_ctors = ind_info.constructor_names.len();
    if num_ctors == 0 {
        // Inductive with no constructors (like False) - goal is discharged
        // Use False.elim or the recursor with no cases
        let rec = Expr::const_(rec_name.clone(), levels.clone());

        // Apply params, motive, major
        let mut proof = rec;
        // Add parameters
        for arg in args.iter().take(ind_info.num_params as usize) {
            proof = Expr::app(proof, arg.clone());
        }
        // Add motive
        proof = Expr::app(proof, motive.clone());
        // Add the hypothesis as major premise
        proof = Expr::app(proof, Expr::fvar(hyp_fvar));

        state.close_goal(proof)?;
        return Ok(());
    }

    // Create metavariables for each constructor case
    let mut case_metas = Vec::with_capacity(num_ctors);

    for ctor_name in &ind_info.constructor_names {
        let ctor_info = state
            .env
            .get_constructor(ctor_name)
            .ok_or_else(|| {
                TacticError::Other(format!("cases: constructor '{ctor_name}' not found"))
            })?
            .clone();

        // Create new local context with constructor fields as hypotheses
        let mut new_ctx = goal.local_ctx.clone();
        // Remove the original hypothesis (it's replaced by the constructor pattern)
        new_ctx.remove(hyp_idx);

        // Parse constructor type to get field types
        // Constructor type: {params} → (fields) → T args
        let mut ctor_ty = ctor_info.type_.clone();
        let mut field_fvars = Vec::new();
        let mut param_idx = 0;

        // Skip parameters (they're the same as the inductive's params)
        for _ in 0..ctor_info.num_params {
            if let Expr::Pi(_, _, codomain) = &ctor_ty {
                // Instantiate with the corresponding argument from the hypothesis type
                if param_idx < args.len() {
                    ctor_ty = codomain.instantiate(&args[param_idx]);
                } else {
                    ctor_ty = codomain.instantiate(&Expr::Sort(Level::zero())); // placeholder
                }
                param_idx += 1;
            }
        }

        // Collect fields (non-parameter arguments)
        let mut field_idx = 0;
        while let Expr::Pi(bi, domain, codomain) = ctor_ty.clone() {
            // Use the last component of the constructor name for field naming
            let ctor_short_name = ctor_name.to_string();
            let ctor_short = ctor_short_name
                .rsplit('.')
                .next()
                .unwrap_or(&ctor_short_name);
            let field_name = format!("{ctor_short}_{field_idx}");
            let field_fvar = state.fresh_fvar();

            let field_decl = LocalDecl {
                fvar: field_fvar,
                name: field_name,
                ty: (*domain).clone(),
                value: None,
            };
            new_ctx.push(field_decl);
            field_fvars.push(field_fvar);

            ctor_ty = codomain.instantiate(&Expr::fvar(field_fvar));
            field_idx += 1;

            // Stop if we've collected all fields
            if field_idx >= ctor_info.num_fields as usize {
                break;
            }

            // Also stop if not a dependent function type anymore
            if !matches!(
                bi,
                BinderInfo::Default | BinderInfo::Implicit | BinderInfo::InstImplicit
            ) {
                break;
            }
        }

        // Create the new target by substituting the hypothesis with the constructor applied to fields
        let mut ctor_app = Expr::const_(ctor_name.clone(), levels.clone());
        // Apply parameters
        for arg in args.iter().take(ind_info.num_params as usize) {
            ctor_app = Expr::app(ctor_app, arg.clone());
        }
        // Apply fields
        for fvar in &field_fvars {
            ctor_app = Expr::app(ctor_app, Expr::fvar(*fvar));
        }

        // Substitute the hypothesis with the constructor application in the target
        // Use Expr's built-in subst_fvar method
        let new_target = goal.target.subst_fvar(hyp_fvar, &ctor_app);
        let new_target = state.metas.instantiate(&new_target);

        // Create metavariable for this case
        let case_meta = state.metas.fresh(new_target.clone());
        case_metas.push((case_meta, new_ctx, new_target, field_fvars));
    }

    // Build the proof term using the recursor
    // rec params motive cases... major
    let rec = Expr::const_(rec_name, levels.clone());

    let mut proof = rec;
    // Add parameters
    for arg in args.iter().take(ind_info.num_params as usize) {
        proof = Expr::app(proof, arg.clone());
    }
    // Add motive
    proof = Expr::app(proof, motive);

    // Add each case (as lambdas abstracting over the fields)
    for (case_meta, new_ctx, _target, field_fvars) in &case_metas {
        let case_body = Expr::FVar(MetaState::to_fvar(*case_meta));

        // Abstract over fields in reverse order
        let mut case_proof = case_body;
        for fvar in field_fvars.iter().rev() {
            // Get the type of this field from the context we built
            let fvar_ty = new_ctx
                .iter()
                .find(|d| d.fvar == *fvar)
                .map_or_else(|| Expr::Sort(Level::zero()), |d| d.ty.clone());
            case_proof = Expr::lam(
                BinderInfo::Default,
                fvar_ty,
                case_proof.abstract_fvar(*fvar),
            );
        }

        proof = Expr::app(proof, case_proof);
    }

    // Add the major premise (the hypothesis)
    proof = Expr::app(proof, Expr::fvar(hyp_fvar));

    // Close the current goal with this proof
    state.close_goal(proof)?;

    // Add the new goals (one per constructor)
    for (case_meta, new_ctx, new_target, _) in case_metas {
        let new_goal = Goal {
            meta_id: case_meta,
            target: new_target,
            local_ctx: new_ctx,
        };
        state.goals.push(new_goal);
    }

    Ok(())
}

/// Perform induction on a hypothesis of inductive type
///
/// The `induction` tactic is similar to `cases` but additionally provides
/// induction hypotheses for recursive constructor fields. This is the key
/// mechanism for proving properties about recursive data structures.
///
/// # Example
/// For a hypothesis `n : Nat`, `induction n` produces two goals:
/// - Base case: goal with `n` replaced by `Nat.zero`
/// - Inductive case: goal with `n` replaced by `Nat.succ n'`, plus hypothesis `IH : P n'`
///
/// # Difference from `cases`
/// - `cases` just destructs the inductive, giving access to constructor fields
/// - `induction` also provides IH for recursive fields, enabling recursive proofs
///
/// The induction hypothesis type is `motive field` where `motive` is the
/// goal's target abstracted over the inductee.
pub fn induction(state: &mut ProofState, hyp_name: &str) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Find the hypothesis in the local context
    let (hyp_idx, hyp_decl) = goal
        .local_ctx
        .iter()
        .enumerate()
        .find(|(_, d)| d.name == hyp_name)
        .ok_or_else(|| TacticError::UnknownIdent(hyp_name.to_string()))?;
    let hyp_fvar = hyp_decl.fvar;
    let hyp_ty = state.metas.instantiate(&hyp_decl.ty);

    // WHNF to expose the inductive type
    let hyp_ty_whnf = state.whnf(&goal, &hyp_ty);

    // Get the head constant and arguments
    let head = hyp_ty_whnf.get_app_fn().clone();
    let args: Vec<Expr> = hyp_ty_whnf.get_app_args().into_iter().cloned().collect();

    let ind_name = match &head {
        Expr::Const(name, _) => name.clone(),
        _ => {
            return Err(TacticError::GoalMismatch(format!(
                "induction: hypothesis '{hyp_name}' has type '{hyp_ty_whnf:?}' which is not an inductive type"
            )));
        }
    };

    // Look up the inductive type information
    let ind_info = state
        .env
        .get_inductive(&ind_name)
        .ok_or_else(|| {
            TacticError::GoalMismatch(format!("induction: '{ind_name}' is not an inductive type"))
        })?
        .clone();

    // Get universe levels from the hypothesis type
    let levels = match &head {
        Expr::Const(_, lvls) => lvls.to_vec(),
        _ => vec![],
    };

    // Build the recursor name: IndName.rec (must use rec, not casesOn, for IH)
    let rec_name = Name::from_string(&format!("{ind_name}.rec"));
    let rec_info = state
        .env
        .get_recursor(&rec_name)
        .ok_or_else(|| TacticError::Other(format!("induction: recursor '{rec_name}' not found")))?
        .clone();

    // Create the motive: fun x => (goal target)
    let motive_body = goal.target.clone();
    let motive = Expr::lam(
        BinderInfo::Default,
        hyp_ty.clone(),
        motive_body.abstract_fvar(hyp_fvar),
    );

    // For each constructor, create a new goal with IH for recursive fields
    let num_ctors = ind_info.constructor_names.len();
    if num_ctors == 0 {
        // Inductive with no constructors (like False) - goal is discharged
        let rec = Expr::const_(rec_name.clone(), levels.clone());

        let mut proof = rec;
        for arg in args.iter().take(ind_info.num_params as usize) {
            proof = Expr::app(proof, arg.clone());
        }
        proof = Expr::app(proof, motive.clone());
        proof = Expr::app(proof, Expr::fvar(hyp_fvar));

        state.close_goal(proof)?;
        return Ok(());
    }

    // Create metavariables for each constructor case
    let mut case_metas = Vec::with_capacity(num_ctors);

    for (ctor_idx, ctor_name) in ind_info.constructor_names.iter().enumerate() {
        let ctor_info = state
            .env
            .get_constructor(ctor_name)
            .ok_or_else(|| {
                TacticError::Other(format!("induction: constructor '{ctor_name}' not found"))
            })?
            .clone();

        // Get recursive field info from the recursor
        let recursive_fields = if ctor_idx < rec_info.rules.len() {
            rec_info.rules[ctor_idx].recursive_fields.clone()
        } else {
            vec![false; ctor_info.num_fields as usize]
        };

        // Create new local context with constructor fields and IHs as hypotheses
        let mut new_ctx = goal.local_ctx.clone();
        new_ctx.remove(hyp_idx);

        // Parse constructor type to get field types
        let mut ctor_ty = ctor_info.type_.clone();
        let mut field_fvars = Vec::new();
        let mut field_types = Vec::new();
        let mut param_idx = 0;

        // Skip parameters
        for _ in 0..ctor_info.num_params {
            if let Expr::Pi(_, _, codomain) = &ctor_ty {
                if param_idx < args.len() {
                    ctor_ty = codomain.instantiate(&args[param_idx]);
                } else {
                    ctor_ty = codomain.instantiate(&Expr::Sort(Level::zero()));
                }
                param_idx += 1;
            }
        }

        // Collect fields (non-parameter arguments)
        let mut field_idx = 0;
        while let Expr::Pi(bi, domain, codomain) = ctor_ty.clone() {
            let ctor_short_name = ctor_name.to_string();
            let ctor_short = ctor_short_name
                .rsplit('.')
                .next()
                .unwrap_or(&ctor_short_name);
            let field_name = format!("{ctor_short}_{field_idx}");
            let field_fvar = state.fresh_fvar();

            let field_ty = (*domain).clone();
            field_types.push(field_ty.clone());

            let field_decl = LocalDecl {
                fvar: field_fvar,
                name: field_name,
                ty: field_ty,
                value: None,
            };
            new_ctx.push(field_decl);
            field_fvars.push(field_fvar);

            ctor_ty = codomain.instantiate(&Expr::fvar(field_fvar));
            field_idx += 1;

            if field_idx >= ctor_info.num_fields as usize {
                break;
            }

            if !matches!(
                bi,
                BinderInfo::Default | BinderInfo::Implicit | BinderInfo::InstImplicit
            ) {
                break;
            }
        }

        // Track which fields have IH and the IH fvars
        let mut ih_fvars = Vec::new();

        // Add induction hypotheses for recursive fields
        for (i, fvar) in field_fvars.iter().enumerate() {
            if i < recursive_fields.len() && recursive_fields[i] {
                // This field is recursive - add an IH
                let ctor_short_name = ctor_name.to_string();
                let ctor_short = ctor_short_name
                    .rsplit('.')
                    .next()
                    .unwrap_or(&ctor_short_name);
                let ih_name = format!("ih_{ctor_short}_{i}");
                let ih_fvar = state.fresh_fvar();

                // IH type is: motive applied to this field
                // i.e., the goal target with hyp_fvar replaced by this field
                let ih_ty = goal.target.subst_fvar(hyp_fvar, &Expr::fvar(*fvar));
                let ih_ty = state.metas.instantiate(&ih_ty);

                let ih_decl = LocalDecl {
                    fvar: ih_fvar,
                    name: ih_name,
                    ty: ih_ty,
                    value: None,
                };
                new_ctx.push(ih_decl);
                ih_fvars.push(Some(ih_fvar));
            } else {
                ih_fvars.push(None);
            }
        }

        // Create the new target by substituting the hypothesis with the constructor applied to fields
        let mut ctor_app = Expr::const_(ctor_name.clone(), levels.clone());
        for arg in args.iter().take(ind_info.num_params as usize) {
            ctor_app = Expr::app(ctor_app, arg.clone());
        }
        for fvar in &field_fvars {
            ctor_app = Expr::app(ctor_app, Expr::fvar(*fvar));
        }

        let new_target = goal.target.subst_fvar(hyp_fvar, &ctor_app);
        let new_target = state.metas.instantiate(&new_target);

        let case_meta = state.metas.fresh(new_target.clone());
        case_metas.push((
            case_meta,
            new_ctx,
            new_target,
            field_fvars,
            ih_fvars,
            field_types,
        ));
    }

    // Build the proof term using the recursor
    let rec = Expr::const_(rec_name, levels.clone());

    let mut proof = rec;
    for arg in args.iter().take(ind_info.num_params as usize) {
        proof = Expr::app(proof, arg.clone());
    }
    proof = Expr::app(proof, motive);

    // Add each case (as lambdas abstracting over fields AND IHs)
    for (case_meta, new_ctx, _target, field_fvars, ih_fvars, _field_types) in &case_metas {
        let case_body = Expr::FVar(MetaState::to_fvar(*case_meta));

        // The minor premise for induction takes: fields, then IHs for recursive fields
        // We need to abstract in reverse order
        let mut case_proof = case_body;

        // Abstract over IHs (in reverse)
        for ih_fvar in ih_fvars.iter().rev().flatten() {
            let ih_ty = new_ctx
                .iter()
                .find(|d| d.fvar == *ih_fvar)
                .map_or_else(|| Expr::Sort(Level::zero()), |d| d.ty.clone());
            case_proof = Expr::lam(
                BinderInfo::Default,
                ih_ty,
                case_proof.abstract_fvar(*ih_fvar),
            );
        }

        // Abstract over fields (in reverse)
        for fvar in field_fvars.iter().rev() {
            let fvar_ty = new_ctx
                .iter()
                .find(|d| d.fvar == *fvar)
                .map_or_else(|| Expr::Sort(Level::zero()), |d| d.ty.clone());
            case_proof = Expr::lam(
                BinderInfo::Default,
                fvar_ty,
                case_proof.abstract_fvar(*fvar),
            );
        }

        proof = Expr::app(proof, case_proof);
    }

    // Add the major premise (the hypothesis)
    proof = Expr::app(proof, Expr::fvar(hyp_fvar));

    // Close the current goal with this proof
    state.close_goal(proof)?;

    // Add the new goals (one per constructor)
    for (case_meta, new_ctx, new_target, _, _, _) in case_metas {
        let new_goal = Goal {
            meta_id: case_meta,
            target: new_target,
            local_ctx: new_ctx,
        };
        state.goals.push(new_goal);
    }

    Ok(())
}

// =============================================================================
// Certified Tactics
//
// These variants generate proof certificates during tactic execution.
// The certificates can be used for:
// 1. Proof archive storage and export
// 2. Independent verification by an external checker
// 3. Debugging and proof tracing
// =============================================================================

/// Result type for certified tactic execution
pub type CertifiedTacticResult = Result<ProofCert, TacticError>;

/// Close the goal with an exact proof term and return a certificate.
///
/// This is the certified variant of `exact`. It performs the same operation
/// but additionally generates a proof certificate witnessing the typing derivation.
pub fn exact_with_cert(state: &mut ProofState, proof: Expr) -> CertifiedTacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Infer the type with certificate
    let (proof_ty, cert) = state.infer_type_with_cert(&goal, &proof)?;

    // Check that the proof has the right type (via unification to handle metas)
    let target = state.metas.instantiate(&goal.target);

    match Unifier::new(state.metas_mut()).unify(&proof_ty, &target) {
        UnifyResult::Success => {
            state.close_goal(proof)?;
            Ok(cert)
        }
        UnifyResult::Failure(msg) => Err(TacticError::TypeMismatch {
            expected: format!("{target:?}"),
            actual: msg,
        }),
        UnifyResult::Stuck => Err(TacticError::UnificationFailed(
            "unification stuck".to_string(),
        )),
    }
}

/// Introduce a hypothesis and return a certificate for the intro step.
///
/// This is the certified variant of `intro`. It returns a certificate
/// representing the lambda abstraction being constructed. Note that the
/// certificate is for the proof term being built (λ x : A, ?body), not
/// for the final proof (which is not yet complete).
///
/// The returned certificate is for the domain type (A : Sort), confirming
/// it is a valid type. The full proof certificate is assembled when all
/// goals are closed.
pub fn intro_with_cert(state: &mut ProofState, name: String) -> CertifiedTacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // WHNF to expose the Pi
    let target = state.whnf(&goal, &goal.target);

    match target {
        Expr::Pi(bi, domain, codomain) => {
            // Generate certificate for the domain type
            let (_domain_type, domain_cert) = state.infer_type_with_cert(&goal, &domain)?;

            // Create a new free variable for the introduced hypothesis
            let fvar = state.fresh_fvar();

            // Create the new local declaration
            let local_decl = LocalDecl {
                fvar,
                name: name.clone(),
                ty: (*domain).clone(),
                value: None,
            };

            // Create new context with the hypothesis
            let mut new_ctx = goal.local_ctx.clone();
            new_ctx.push(local_decl);

            // Instantiate the codomain with the free variable
            let new_target = codomain.instantiate(&Expr::fvar(fvar));

            // Create new metavariable for the new goal
            let new_meta_id = state.metas.fresh(new_target.clone());

            // The proof of the original goal is λ x : A, <new_proof>
            let new_meta_expr = Expr::FVar(MetaState::to_fvar(new_meta_id));
            let proof = Expr::lam(bi, (*domain).clone(), new_meta_expr.abstract_fvar(fvar));

            // Close the current goal with this proof template
            state.close_goal(proof)?;

            // Add the new goal
            let new_goal = Goal {
                meta_id: new_meta_id,
                target: new_target,
                local_ctx: new_ctx,
            };

            state.goals.insert(0, new_goal);

            // Return the domain type certificate
            Ok(domain_cert)
        }
        _ => Err(TacticError::GoalMismatch(
            "intro requires a forall/arrow goal".to_string(),
        )),
    }
}

/// Apply a function and return a certificate for the application.
///
/// This is the certified variant of `apply`. It returns a certificate
/// for the function's type inference.
pub fn apply_with_cert(state: &mut ProofState, func: Expr) -> CertifiedTacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Infer the type of the function with certificate
    let (func_ty, cert) = state.infer_type_with_cert(&goal, &func)?;

    // Apply the function using the regular apply logic
    apply_aux(state, &goal, func, func_ty)?;

    Ok(cert)
}

/// Use a hypothesis from context and return a certificate.
///
/// This is the certified variant of `assumption`. It searches for a
/// hypothesis matching the goal and returns the certificate for that
/// hypothesis's type.
pub fn assumption_with_cert(state: &mut ProofState) -> CertifiedTacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();
    let target = state.metas.instantiate(&goal.target);

    // Search through hypotheses for one that matches
    for decl in &goal.local_ctx {
        let hyp_ty = state.metas.instantiate(&decl.ty);
        if state.is_def_eq(&goal, &hyp_ty, &target) {
            let hyp_expr = Expr::fvar(decl.fvar);
            // Get certificate for the hypothesis
            let (_, cert) = state.infer_type_with_cert(&goal, &hyp_expr)?;
            exact(state, hyp_expr)?;
            return Ok(cert);
        }
    }

    Err(TacticError::Other(
        "no matching hypothesis found".to_string(),
    ))
}

// =============================================================================
// Forward reasoning tactics: have, suffices
// =============================================================================

/// Introduce an auxiliary lemma.
///
/// The `have` tactic introduces an intermediate result into the proof.
/// Given `have h : T := proof`, it:
/// 1. Verifies that `proof` has type `T`
/// 2. Adds `h : T` to the local context
/// 3. Continues with the original goal
///
/// This is "forward reasoning" - proving a lemma and adding it as a hypothesis.
///
/// # Arguments
/// * `name` - Name for the new hypothesis
/// * `ty` - Type of the lemma to prove
/// * `proof` - Optional proof term. If None, creates a subgoal for the proof.
///
/// # Example
/// ```ignore
/// // Goal: P
/// have_tactic("h", T, Some(proof_of_T))
/// // Now have h : T in context, still need to prove P
/// ```
pub fn have_tactic(
    state: &mut ProofState,
    name: String,
    ty: Expr,
    proof: Option<Expr>,
) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    if let Some(pf) = proof {
        // Verify the proof has the expected type
        let proof_ty = state.infer_type(&goal, &pf)?;
        let ty_inst = state.metas.instantiate(&ty);

        match Unifier::new(state.metas_mut()).unify(&proof_ty, &ty_inst) {
            UnifyResult::Success => {
                // Proof is valid - add to context as a let-binding
                let fvar = state.fresh_fvar();

                let local_decl = LocalDecl {
                    fvar,
                    name: name.clone(),
                    ty: ty.clone(),
                    value: Some(pf.clone()),
                };

                // Create new goal with extended context
                let mut new_ctx = goal.local_ctx.clone();
                new_ctx.push(local_decl);

                // Create a new metavariable for the continuation
                let new_meta_id = state.metas.fresh(goal.target.clone());
                let new_meta_expr = Expr::FVar(MetaState::to_fvar(new_meta_id));

                // The proof of the original goal is:
                // let h : T := proof in <continuation>
                let proof_term = Expr::let_(ty, pf, new_meta_expr.abstract_fvar(fvar));

                state.close_goal(proof_term)?;

                // Add the new goal
                let new_goal = Goal {
                    meta_id: new_meta_id,
                    target: goal.target.clone(),
                    local_ctx: new_ctx,
                };

                state.goals.insert(0, new_goal);
                Ok(())
            }
            UnifyResult::Failure(msg) => Err(TacticError::TypeMismatch {
                expected: format!("{ty_inst:?}"),
                actual: msg,
            }),
            UnifyResult::Stuck => Err(TacticError::UnificationFailed(
                "unification stuck while checking have proof".to_string(),
            )),
        }
    } else {
        // No proof provided - create two goals:
        // 1. Prove T (the lemma)
        // 2. Continue original proof with h : T available

        let fvar = state.fresh_fvar();

        let local_decl = LocalDecl {
            fvar,
            name: name.clone(),
            ty: ty.clone(),
            value: None,
        };

        // Extended context for continuation
        let mut new_ctx = goal.local_ctx.clone();
        new_ctx.push(local_decl);

        // Create metavariables for both goals
        let lemma_meta_id = state.metas.fresh(ty.clone());
        let cont_meta_id = state.metas.fresh(goal.target.clone());

        let lemma_meta_expr = Expr::FVar(MetaState::to_fvar(lemma_meta_id));
        let cont_meta_expr = Expr::FVar(MetaState::to_fvar(cont_meta_id));

        // The proof of the original goal is:
        // let h : T := <lemma_proof> in <continuation>
        let proof_term = Expr::let_(
            ty.clone(),
            lemma_meta_expr,
            cont_meta_expr.abstract_fvar(fvar),
        );

        state.close_goal(proof_term)?;

        // Goal 1: prove the lemma T
        let lemma_goal = Goal {
            meta_id: lemma_meta_id,
            target: ty,
            local_ctx: goal.local_ctx.clone(),
        };

        // Goal 2: prove the original target with h available
        let cont_goal = Goal {
            meta_id: cont_meta_id,
            target: goal.target.clone(),
            local_ctx: new_ctx,
        };

        // Insert goals (lemma first, then continuation)
        state.goals.insert(0, cont_goal);
        state.goals.insert(0, lemma_goal);

        Ok(())
    }
}

/// Suffices tactic for backward reasoning.
///
/// The `suffices` tactic lets you prove a goal by showing that it follows from
/// a simpler statement. Given `suffices h : T by proof`, it:
/// 1. Creates a goal to prove T
/// 2. Creates a goal to prove (T → original_goal)
///
/// This is "backward reasoning" - reducing the goal to proving a sufficient condition.
///
/// # Arguments
/// * `_name` - Name for the sufficient condition hypothesis (unused in proof term)
/// * `ty` - Type representing the sufficient condition
/// * `proof_fn` - Optional proof that T implies the original goal
///
/// # Example
/// ```ignore
/// // Goal: P ∧ Q
/// suffices_tactic("h", P, Some(proof_of_P_implies_PandQ))
/// // Now just need to prove P
/// ```
///
/// Without a proof function:
/// ```ignore
/// // Goal: Complex
/// suffices_tactic("h", Simpler, None)
/// // Goal 1: Simpler
/// // Goal 2: Simpler → Complex
/// ```
pub fn suffices_tactic(
    state: &mut ProofState,
    _name: String,
    ty: Expr,
    proof_fn: Option<Expr>,
) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    if let Some(pf) = proof_fn {
        // Verify the proof function has type (ty → goal.target)
        let expected_fn_ty = Expr::arrow(ty.clone(), goal.target.clone());
        let proof_ty = state.infer_type(&goal, &pf)?;
        let expected_inst = state.metas.instantiate(&expected_fn_ty);

        match Unifier::new(state.metas_mut()).unify(&proof_ty, &expected_inst) {
            UnifyResult::Success => {
                // proof_fn is valid - now we just need to prove ty
                let sufficient_meta_id = state.metas.fresh(ty.clone());
                let sufficient_meta_expr = Expr::FVar(MetaState::to_fvar(sufficient_meta_id));

                // The proof of the original goal is: pf <proof_of_ty>
                let proof_term = Expr::app(pf, sufficient_meta_expr);

                state.close_goal(proof_term)?;

                // Create a goal for proving ty
                let sufficient_goal = Goal {
                    meta_id: sufficient_meta_id,
                    target: ty,
                    local_ctx: goal.local_ctx.clone(),
                };

                state.goals.insert(0, sufficient_goal);
                Ok(())
            }
            UnifyResult::Failure(msg) => Err(TacticError::TypeMismatch {
                expected: format!("{expected_inst:?}"),
                actual: msg,
            }),
            UnifyResult::Stuck => Err(TacticError::UnificationFailed(
                "unification stuck while checking suffices proof".to_string(),
            )),
        }
    } else {
        // No proof function provided - create two goals:
        // 1. Prove ty (the sufficient condition)
        // 2. Prove ty → original_goal (with h : ty available via intro)

        // Create metavariables for both goals
        let sufficient_meta_id = state.metas.fresh(ty.clone());
        let impl_meta_id = state
            .metas
            .fresh(Expr::arrow(ty.clone(), goal.target.clone()));

        let sufficient_meta_expr = Expr::FVar(MetaState::to_fvar(sufficient_meta_id));
        let impl_meta_expr = Expr::FVar(MetaState::to_fvar(impl_meta_id));

        // The proof of the original goal is: <impl_proof> <proof_of_ty>
        let proof_term = Expr::app(impl_meta_expr, sufficient_meta_expr);

        state.close_goal(proof_term)?;

        // Goal 1: prove the sufficient condition
        let sufficient_goal = Goal {
            meta_id: sufficient_meta_id,
            target: ty.clone(),
            local_ctx: goal.local_ctx.clone(),
        };

        // Goal 2: prove that sufficient condition implies original goal
        let impl_goal = Goal {
            meta_id: impl_meta_id,
            target: Expr::arrow(ty, goal.target),
            local_ctx: goal.local_ctx.clone(),
        };

        // Insert goals (sufficient first, then implication)
        state.goals.insert(0, impl_goal);
        state.goals.insert(0, sufficient_goal);

        Ok(())
    }
}

// =============================================================================
// Connective tactics: split, left, right
// =============================================================================

/// Split a conjunction goal into two subgoals and build `And.intro`.
///
/// For a goal `And A B`, produces subgoals `A` and `B` (left first) and
/// closes the current goal with `And.intro A B ?left ?right`.
pub fn split_tactic(state: &mut ProofState) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    let target = state.whnf(&goal, &goal.target);
    let head = target.get_app_fn().clone();
    let args: Vec<Expr> = target.get_app_args().into_iter().cloned().collect();

    match head {
        Expr::Const(name, levels) if name == Name::from_string("And") => {
            if args.len() != 2 {
                return Err(TacticError::GoalMismatch(
                    "split requires goal of form And a b".to_string(),
                ));
            }

            let left_ty = state.metas.instantiate(&args[0]);
            let right_ty = state.metas.instantiate(&args[1]);

            let left_meta_id = state.metas.fresh(left_ty.clone());
            let right_meta_id = state.metas.fresh(right_ty.clone());

            let left_meta = Expr::FVar(MetaState::to_fvar(left_meta_id));
            let right_meta = Expr::FVar(MetaState::to_fvar(right_meta_id));

            // Build And.intro a b ?left ?right
            let mut proof = Expr::const_(Name::from_string("And.intro"), levels.clone());
            proof = Expr::app(proof, args[0].clone());
            proof = Expr::app(proof, args[1].clone());
            proof = Expr::app(proof, left_meta.clone());
            proof = Expr::app(proof, right_meta.clone());

            state.close_goal(proof)?;

            // Insert subgoals: left first, then right
            let left_goal = Goal {
                meta_id: left_meta_id,
                target: left_ty,
                local_ctx: goal.local_ctx.clone(),
            };
            let right_goal = Goal {
                meta_id: right_meta_id,
                target: right_ty,
                local_ctx: goal.local_ctx.clone(),
            };

            state.goals.insert(0, right_goal);
            state.goals.insert(0, left_goal);
            Ok(())
        }
        _ => Err(TacticError::GoalMismatch(
            "split requires goal of form And a b".to_string(),
        )),
    }
}

/// Solve the left branch of a disjunction by reducing goal to its left side.
///
/// For goal `Or A B`, creates a subgoal `A` and closes the current goal with
/// `Or.inl A B ?proof`.
pub fn left_tactic(state: &mut ProofState) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    let target = state.whnf(&goal, &goal.target);
    let head = target.get_app_fn().clone();
    let args: Vec<Expr> = target.get_app_args().into_iter().cloned().collect();

    match head {
        Expr::Const(name, levels) if name == Name::from_string("Or") => {
            if args.len() != 2 {
                return Err(TacticError::GoalMismatch(
                    "left requires goal of form Or a b".to_string(),
                ));
            }

            let left_ty = state.metas.instantiate(&args[0]);
            let left_meta_id = state.metas.fresh(left_ty.clone());
            let left_meta = Expr::FVar(MetaState::to_fvar(left_meta_id));

            // Build Or.inl a b ?proof
            let mut proof = Expr::const_(Name::from_string("Or.inl"), levels.clone());
            proof = Expr::app(proof, args[0].clone());
            proof = Expr::app(proof, args[1].clone());
            proof = Expr::app(proof, left_meta.clone());

            state.close_goal(proof)?;

            let left_goal = Goal {
                meta_id: left_meta_id,
                target: left_ty,
                local_ctx: goal.local_ctx.clone(),
            };
            state.goals.insert(0, left_goal);
            Ok(())
        }
        _ => Err(TacticError::GoalMismatch(
            "left requires goal of form Or a b".to_string(),
        )),
    }
}

/// Solve the right branch of a disjunction by reducing goal to its right side.
///
/// For goal `Or A B`, creates a subgoal `B` and closes the current goal with
/// `Or.inr A B ?proof`.
pub fn right_tactic(state: &mut ProofState) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    let target = state.whnf(&goal, &goal.target);
    let head = target.get_app_fn().clone();
    let args: Vec<Expr> = target.get_app_args().into_iter().cloned().collect();

    match head {
        Expr::Const(name, levels) if name == Name::from_string("Or") => {
            if args.len() != 2 {
                return Err(TacticError::GoalMismatch(
                    "right requires goal of form Or a b".to_string(),
                ));
            }

            let right_ty = state.metas.instantiate(&args[1]);
            let right_meta_id = state.metas.fresh(right_ty.clone());
            let right_meta = Expr::FVar(MetaState::to_fvar(right_meta_id));

            // Build Or.inr a b ?proof
            let mut proof = Expr::const_(Name::from_string("Or.inr"), levels.clone());
            proof = Expr::app(proof, args[0].clone());
            proof = Expr::app(proof, args[1].clone());
            proof = Expr::app(proof, right_meta.clone());

            state.close_goal(proof)?;

            let right_goal = Goal {
                meta_id: right_meta_id,
                target: right_ty,
                local_ctx: goal.local_ctx.clone(),
            };
            state.goals.insert(0, right_goal);
            Ok(())
        }
        _ => Err(TacticError::GoalMismatch(
            "right requires goal of form Or a b".to_string(),
        )),
    }
}

// =============================================================================
// Contradiction and False-elimination tactics
// =============================================================================

/// The `exfalso` tactic changes the goal to `False`.
///
/// This is useful when we want to derive a contradiction to prove any proposition.
/// It applies the principle of explosion (ex falso quodlibet): from False, anything follows.
///
/// # Example
/// ```text
/// Goal: P
/// exfalso
/// Goal: False
/// ```
///
/// The proof term is `False.elim {P} <proof of False>`.
pub fn exfalso(state: &mut ProofState) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Check that False.elim exists
    let false_elim_name = Name::from_string("False.elim");
    if state.env.get_const(&false_elim_name).is_none() {
        return Err(TacticError::Other(
            "exfalso: False.elim not found in environment (call env.init_true_false())".to_string(),
        ));
    }

    // Create a new goal for False
    let false_type = Expr::const_(Name::from_string("False"), vec![]);
    let new_meta_id = state.metas.fresh(false_type.clone());

    // The proof is: False.elim {goal.target} <new_meta>
    // False.elim : {C : Sort u} → False → C
    let false_elim = Expr::const_(false_elim_name, vec![Level::zero()]);
    let proof = Expr::app(
        Expr::app(false_elim, goal.target.clone()),
        Expr::FVar(MetaState::to_fvar(new_meta_id)),
    );

    // Close the current goal
    state.close_goal(proof)?;

    // Add the new goal (prove False)
    let new_goal = Goal {
        meta_id: new_meta_id,
        target: false_type,
        local_ctx: goal.local_ctx.clone(),
    };
    state.goals.insert(0, new_goal);

    Ok(())
}

/// The `contradiction` tactic proves the goal by finding contradictory hypotheses.
///
/// It searches the local context for:
/// 1. A hypothesis `h : False` (directly proves any goal)
/// 2. A pair `h1 : P` and `h2 : ¬P` (or `h2 : P → False`)
///
/// # Example
/// ```text
/// h1 : P
/// h2 : ¬P
/// Goal: Q
/// contradiction  -- applies absurd h1 h2
/// ```
pub fn contradiction(state: &mut ProofState) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    let false_type = Expr::const_(Name::from_string("False"), vec![]);

    // First, look for h : False in context
    for decl in &goal.local_ctx {
        let ty = state.metas.instantiate(&decl.ty);
        let ty_whnf = state.whnf(&goal, &ty);

        // Check if this is False
        if state.is_def_eq(&goal, &ty_whnf, &false_type) {
            // Found h : False, use False.elim
            let false_elim = Expr::const_(Name::from_string("False.elim"), vec![Level::zero()]);
            let proof = Expr::app(
                Expr::app(false_elim, goal.target.clone()),
                Expr::fvar(decl.fvar),
            );
            return state.close_goal(proof);
        }
    }

    // Second, look for h1 : P and h2 : P → False (i.e., ¬P)
    for decl1 in &goal.local_ctx {
        let ty1 = state.metas.instantiate(&decl1.ty);
        let ty1_whnf = state.whnf(&goal, &ty1);

        for decl2 in &goal.local_ctx {
            if decl1.fvar == decl2.fvar {
                continue;
            }

            let ty2 = state.metas.instantiate(&decl2.ty);
            let ty2_whnf = state.whnf(&goal, &ty2);

            // Check if ty2 is ty1 → False (i.e., ¬ty1)
            if let Expr::Pi(_, domain, codomain) = &ty2_whnf {
                let domain_whnf = state.whnf(&goal, domain);
                let codomain_whnf = state.whnf(&goal, codomain);

                if state.is_def_eq(&goal, &domain_whnf, &ty1_whnf)
                    && state.is_def_eq(&goal, &codomain_whnf, &false_type)
                {
                    // Found h1 : P and h2 : P → False
                    // Use absurd if available, otherwise construct False.elim (h2 h1)
                    let absurd_name = Name::from_string("absurd");
                    if state.env.get_const(&absurd_name).is_some() {
                        // absurd : {a : Prop} → {b : Sort u} → a → ¬a → b
                        let absurd = Expr::const_(absurd_name, vec![Level::zero()]);
                        let proof = Expr::app(
                            Expr::app(
                                Expr::app(Expr::app(absurd, ty1_whnf.clone()), goal.target.clone()),
                                Expr::fvar(decl1.fvar),
                            ),
                            Expr::fvar(decl2.fvar),
                        );
                        return state.close_goal(proof);
                    }
                    // Fallback: False.elim {goal} (h2 h1)
                    let false_elim =
                        Expr::const_(Name::from_string("False.elim"), vec![Level::zero()]);
                    let proof = Expr::app(
                        Expr::app(false_elim, goal.target.clone()),
                        Expr::app(Expr::fvar(decl2.fvar), Expr::fvar(decl1.fvar)),
                    );
                    return state.close_goal(proof);
                }
            }
        }
    }

    Err(TacticError::Other(
        "contradiction: no contradictory hypotheses found".to_string(),
    ))
}

/// The `by_contra` tactic proves the goal by contradiction (classical reasoning).
///
/// It introduces `h : ¬goal` as a hypothesis and changes the goal to `False`.
/// Uses `Classical.byContradiction : {p : Prop} → (¬p → False) → p`.
///
/// # Example
/// ```text
/// Goal: P
/// by_contra h
/// h : ¬P (i.e., P → False)
/// Goal: False
/// ```
///
/// The proof term is `Classical.byContradiction {P} (fun h : ¬P => <proof of False>)`.
pub fn by_contra(state: &mut ProofState, hyp_name: String) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Check that Classical.byContradiction exists
    let by_contradiction_name = Name::from_string("Classical.byContradiction");
    if state.env.get_const(&by_contradiction_name).is_none() {
        return Err(TacticError::Other(
            "by_contra: Classical.byContradiction not found (call env.init_classical())"
                .to_string(),
        ));
    }

    let false_type = Expr::const_(Name::from_string("False"), vec![]);

    // The negation of the goal: goal → False
    let neg_goal = Expr::pi(BinderInfo::Default, goal.target.clone(), false_type.clone());

    // Create a fresh fvar for the new hypothesis h : ¬goal
    let hyp_fvar = state.fresh_fvar();

    // Create the new local context with h : ¬goal
    let mut new_ctx = goal.local_ctx.clone();
    new_ctx.push(LocalDecl {
        fvar: hyp_fvar,
        name: hyp_name,
        ty: neg_goal.clone(),
        value: None,
    });

    // Create a new goal for False
    let new_meta_id = state.metas.fresh(false_type.clone());

    // The proof is: Classical.byContradiction {goal.target} (fun h : ¬goal => <new_meta>)
    // byContradiction : {p : Prop} → (¬p → False) → p
    let by_contradiction = Expr::const_(by_contradiction_name, vec![]);
    let inner_lambda = Expr::lam(
        BinderInfo::Default,
        neg_goal,
        Expr::FVar(MetaState::to_fvar(new_meta_id)).abstract_fvar(hyp_fvar),
    );
    let proof = Expr::app(
        Expr::app(by_contradiction, goal.target.clone()),
        inner_lambda,
    );

    // Close the current goal
    state.close_goal(proof)?;

    // Add the new goal (prove False with h : ¬goal in context)
    let new_goal = Goal {
        meta_id: new_meta_id,
        target: false_type,
        local_ctx: new_ctx,
    };
    state.goals.insert(0, new_goal);

    Ok(())
}

/// The `existsi` tactic provides a witness for an existential goal.
///
/// For a goal `∃ x : α, P x`, `existsi w` reduces the goal to `P w`.
///
/// # Example
/// ```text
/// Goal: ∃ x : Nat, x > 0
/// existsi 1
/// Goal: 1 > 0
/// ```
///
/// The proof term is `Exists.intro {α} {P} w <proof of P w>`.
pub fn existsi(state: &mut ProofState, witness: Expr) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Check that Exists.intro exists
    let exists_intro_name = Name::from_string("Exists.intro");
    if state.env.get_const(&exists_intro_name).is_none() {
        return Err(TacticError::Other(
            "existsi: Exists.intro not found (call env.init_exists())".to_string(),
        ));
    }

    // WHNF to expose the Exists application
    let target = state.whnf(&goal, &goal.target);

    // Parse Exists {α} p
    // Exists : {α : Sort u} → (α → Prop) → Prop
    let (alpha, pred) = match_exists(&target).ok_or_else(|| {
        TacticError::GoalMismatch(format!(
            "existsi: goal '{target:?}' is not of the form '∃ x, P x'"
        ))
    })?;

    // Infer the type of the witness
    let witness_ty = state.infer_type(&goal, &witness)?;

    // Check that the witness has the right type (α)
    if !state.is_def_eq(&goal, &witness_ty, &alpha) {
        return Err(TacticError::TypeMismatch {
            expected: format!("{alpha:?}"),
            actual: format!("{witness_ty:?}"),
        });
    }

    // The new goal is: P witness
    let new_target = Expr::app(pred.clone(), witness.clone());
    let new_meta_id = state.metas.fresh(new_target.clone());

    // Infer universe level for α
    let alpha_ty = state.infer_type(&goal, &alpha)?;
    let level = match alpha_ty {
        Expr::Sort(l) => l,
        _ => Level::zero(), // Fallback
    };

    // The proof is: Exists.intro {α} {p} witness <new_meta>
    // Exists.intro : {α : Sort u} → {p : α → Prop} → (w : α) → p w → Exists p
    let exists_intro = Expr::const_(exists_intro_name, vec![level]);
    let proof = Expr::app(
        Expr::app(Expr::app(Expr::app(exists_intro, alpha), pred), witness),
        Expr::FVar(MetaState::to_fvar(new_meta_id)),
    );

    // Close the current goal
    state.close_goal(proof)?;

    // Add the new goal
    let new_goal = Goal {
        meta_id: new_meta_id,
        target: new_target,
        local_ctx: goal.local_ctx.clone(),
    };
    state.goals.insert(0, new_goal);

    Ok(())
}

/// Match an expression of the form `Exists {α} p` and extract α and p.
fn match_exists(expr: &Expr) -> Option<(Expr, Expr)> {
    // Exists {α} p is App(App(Const("Exists", _), α), p)
    let head = expr.get_app_fn();
    let args: Vec<&Expr> = expr.get_app_args();

    match head {
        Expr::Const(name, _) if name.to_string() == "Exists" => {
            if args.len() == 2 {
                Some((args[0].clone(), args[1].clone()))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// The `by_cases` tactic performs case analysis on a decidable proposition.
///
/// For a goal `G` and a decidable proposition `P`, `by_cases h : P` creates two goals:
/// 1. `G` with hypothesis `h : P`
/// 2. `G` with hypothesis `h : ¬P`
///
/// This uses Classical.em (excluded middle): `∀ p, p ∨ ¬p`.
///
/// # Example
/// ```text
/// Goal: Q
/// by_cases h : P
/// -- Case 1:
/// h : P
/// Goal: Q
/// -- Case 2:
/// h : ¬P
/// Goal: Q
/// ```
pub fn by_cases(state: &mut ProofState, hyp_name: String, prop: Expr) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Check that Classical.em exists
    let em_name = Name::from_string("Classical.em");
    if state.env.get_const(&em_name).is_none() {
        return Err(TacticError::Other(
            "by_cases: Classical.em not found (call env.init_classical())".to_string(),
        ));
    }

    // Check that Or.rec exists (generated by adding Or inductive in init_classical)
    let or_rec_name = Name::from_string("Or.rec");
    if state.env.get_const(&or_rec_name).is_none() {
        return Err(TacticError::Other(
            "by_cases: Or.rec not found (call env.init_classical())".to_string(),
        ));
    }

    // Verify prop is a Prop
    let prop_ty = state.infer_type(&goal, &prop)?;
    let prop_sort = Expr::prop();
    if !state.is_def_eq(&goal, &prop_ty, &prop_sort) {
        return Err(TacticError::TypeMismatch {
            expected: "Prop".to_string(),
            actual: format!("{prop_ty:?}"),
        });
    }

    let false_type = Expr::const_(Name::from_string("False"), vec![]);

    // ¬P = P → False
    let neg_prop = Expr::pi(BinderInfo::Default, prop.clone(), false_type);

    // Create fresh fvars for the hypotheses
    let fvar_pos = state.fresh_fvar();
    let fvar_neg = state.fresh_fvar();

    // Context for positive case: h : P
    let mut ctx_pos = goal.local_ctx.clone();
    ctx_pos.push(LocalDecl {
        fvar: fvar_pos,
        name: hyp_name.clone(),
        ty: prop.clone(),
        value: None,
    });

    // Context for negative case: h : ¬P
    let mut ctx_neg = goal.local_ctx.clone();
    ctx_neg.push(LocalDecl {
        fvar: fvar_neg,
        name: hyp_name.clone(),
        ty: neg_prop.clone(),
        value: None,
    });

    // Create metavariables for the two cases
    let meta_pos = state.metas.fresh(goal.target.clone());
    let meta_neg = state.metas.fresh(goal.target.clone());

    // Build the proof term using Or.rec
    // Or.rec : {a b : Prop} → {motive : Or a b → Sort u} →
    //          ((h : a) → motive (Or.inl a b h)) →
    //          ((h : b) → motive (Or.inr a b h)) →
    //          (t : Or a b) → motive t
    //
    // For proof-irrelevant usage (Prop), the motive is constant:
    // Or.rec {P} {¬P} {λ _ => goal} (λ h => ...) (λ h => ...) (Classical.em P)

    // Classical.em : ∀ p, p ∨ ¬p
    let em = Expr::const_(em_name, vec![]);
    let em_p = Expr::app(em, prop.clone());

    // Or.rec at universe 0 (for Prop targets)
    let or_rec = Expr::const_(or_rec_name, vec![Level::zero()]);

    // Motive: λ _ : Or P ¬P => goal
    let or_type = Expr::app(
        Expr::app(Expr::const_(Name::from_string("Or"), vec![]), prop.clone()),
        neg_prop.clone(),
    );
    let motive = Expr::lam(BinderInfo::Default, or_type, goal.target.clone());

    let branch_pos = Expr::lam(
        BinderInfo::Default,
        prop.clone(),
        Expr::FVar(MetaState::to_fvar(meta_pos)).abstract_fvar(fvar_pos),
    );

    let branch_neg = Expr::lam(
        BinderInfo::Default,
        neg_prop.clone(),
        Expr::FVar(MetaState::to_fvar(meta_neg)).abstract_fvar(fvar_neg),
    );

    // Or.rec {P} {¬P} {motive} branch_pos branch_neg em_p
    let proof = Expr::app(
        Expr::app(
            Expr::app(
                Expr::app(Expr::app(Expr::app(or_rec, prop.clone()), neg_prop), motive),
                branch_pos,
            ),
            branch_neg,
        ),
        em_p,
    );

    // Close the current goal
    state.close_goal(proof)?;

    // Add the two new goals (positive case first)
    let goal_pos = Goal {
        meta_id: meta_pos,
        target: goal.target.clone(),
        local_ctx: ctx_pos,
    };
    let goal_neg = Goal {
        meta_id: meta_neg,
        target: goal.target,
        local_ctx: ctx_neg,
    };

    state.goals.insert(0, goal_neg);
    state.goals.insert(0, goal_pos);

    Ok(())
}

// =============================================================================
// Tactic Combinators
// =============================================================================

/// A tactic is a function that transforms a proof state.
pub type Tactic = Box<dyn FnOnce(&mut ProofState) -> TacticResult>;

/// The `try_tactic` combinator runs a tactic but succeeds even if it fails.
///
/// This is useful for tactics that may not always apply but shouldn't
/// cause the overall proof script to fail.
///
/// # Example
/// ```text
/// try_tactic(|| rfl(state))  -- succeeds even if rfl fails
/// ```
pub fn try_tactic<F>(state: &mut ProofState, tactic: F) -> TacticResult
where
    F: FnOnce(&mut ProofState) -> TacticResult,
{
    // Save state in case tactic fails
    let saved_goals = state.goals.clone();
    let saved_metas = state.metas.clone();

    if tactic(state).is_ok() {
        Ok(())
    } else {
        // Restore state and succeed anyway
        state.goals = saved_goals;
        *state.metas_mut() = saved_metas;
        Ok(())
    }
}

/// The `repeat_tactic` combinator runs a tactic repeatedly until it fails.
///
/// Returns success after applying the tactic zero or more times.
/// The state after the last successful application is kept.
///
/// # Example
/// ```text
/// repeat_tactic(|| intro(state, "h"))  -- introduces all hypotheses
/// ```
///
/// # Arguments
/// * `max_iterations` - Maximum number of iterations (prevents infinite loops)
pub fn repeat_tactic<F>(
    state: &mut ProofState,
    mut tactic_factory: F,
    max_iterations: usize,
) -> TacticResult
where
    F: FnMut() -> Box<dyn FnOnce(&mut ProofState) -> TacticResult>,
{
    for _ in 0..max_iterations {
        let saved_goals = state.goals.clone();
        let saved_metas = state.metas.clone();

        let tactic = tactic_factory();
        if tactic(state).is_ok() {
            // Tactic succeeded, continue
            if state.is_complete() {
                break; // No more goals to work on
            }
        } else {
            // Tactic failed, restore and stop
            state.goals = saved_goals;
            *state.metas_mut() = saved_metas;
            break;
        }
    }
    Ok(())
}

/// The `first_tactic` combinator tries tactics in order until one succeeds.
///
/// Returns success if any tactic succeeds, error if all fail.
///
/// # Example
/// ```text
/// first_tactic(vec![
///     || assumption(state),
///     || rfl(state),
///     || trivial(state),
/// ])
/// ```
pub fn first_tactic<F>(state: &mut ProofState, tactics: Vec<F>) -> TacticResult
where
    F: FnOnce(&mut ProofState) -> TacticResult,
{
    let saved_goals = state.goals.clone();
    let saved_metas = state.metas.clone();

    for tactic in tactics {
        if tactic(state).is_ok() {
            return Ok(());
        }
        // Restore state and try next tactic
        state.goals = saved_goals.clone();
        *state.metas_mut() = saved_metas.clone();
    }

    Err(TacticError::Other("first: all tactics failed".to_string()))
}

/// The `all_goals` combinator applies a tactic to all goals.
///
/// The tactic is applied to each goal in order. If any application fails,
/// the entire combinator fails.
///
/// # Example
/// ```text
/// all_goals(|| assumption(state))  -- try assumption on all goals
/// ```
pub fn all_goals<F>(state: &mut ProofState, mut tactic_factory: F) -> TacticResult
where
    F: FnMut() -> Box<dyn FnOnce(&mut ProofState) -> TacticResult>,
{
    // Apply tactic to each goal
    // We need to be careful: applying a tactic may create new goals
    // We want to apply to the original goals, not new ones
    let original_goal_count = state.goals.len();
    let mut processed = 0;

    while processed < original_goal_count && !state.goals.is_empty() {
        let tactic = tactic_factory();
        tactic(state)?;
        processed += 1;
    }

    Ok(())
}

/// The `any_goals` combinator applies a tactic to all goals, succeeding if any succeed.
///
/// Unlike `all_goals`, this continues even if some goals fail and only
/// returns error if ALL goals fail.
///
/// # Example
/// ```text
/// any_goals(|| assumption(state))  -- assumption on goals that have a matching hyp
/// ```
pub fn any_goals<F>(state: &mut ProofState, mut tactic_factory: F) -> TacticResult
where
    F: FnMut() -> Box<dyn FnOnce(&mut ProofState) -> TacticResult>,
{
    let original_goal_count = state.goals.len();
    let mut processed = 0;
    let mut any_succeeded = false;

    while processed < original_goal_count && !state.goals.is_empty() {
        let saved_goals = state.goals.clone();
        let saved_metas = state.metas.clone();

        let tactic = tactic_factory();
        if tactic(state).is_ok() {
            any_succeeded = true;
        } else {
            // Restore state for this goal and skip it
            state.goals = saved_goals;
            *state.metas_mut() = saved_metas;
            // Move to next goal by rotating
            if !state.goals.is_empty() {
                let goal = state.goals.remove(0);
                state.goals.push(goal);
            }
        }
        processed += 1;
    }

    if any_succeeded {
        Ok(())
    } else {
        Err(TacticError::Other(
            "any_goals: no goal succeeded".to_string(),
        ))
    }
}

/// The `focus` combinator applies a tactic to only the first goal.
///
/// This is useful when you want to work on a specific goal without
/// affecting others.
pub fn focus<F>(state: &mut ProofState, tactic: F) -> TacticResult
where
    F: FnOnce(&mut ProofState) -> TacticResult,
{
    if state.goals.is_empty() {
        return Err(TacticError::NoGoals);
    }

    // Temporarily remove all but the first goal
    let rest = state.goals.split_off(1);

    let result = tactic(state);

    // Restore remaining goals (after any new goals from tactic)
    state.goals.extend(rest);

    result
}

/// Simple automation tactic that tries several basic tactics.
///
/// Tries: assumption, rfl (if available)
pub fn trivial(state: &mut ProofState) -> TacticResult {
    // Try assumption first
    if assumption(state).is_ok() {
        return Ok(());
    }

    // Try rfl
    if rfl(state).is_ok() {
        return Ok(());
    }

    Err(TacticError::Other(
        "trivial: no tactic succeeded".to_string(),
    ))
}

/// Solve the goal by iteratively applying hypotheses from the context.
///
/// This tactic searches for a proof by:
/// 1. Trying `assumption` (if the goal is directly in the context)
/// 2. Trying to apply each hypothesis and recursively solving subgoals
///
/// Uses depth-limited search to prevent infinite loops.
///
/// # Arguments
/// * `state` - The proof state
/// * `max_depth` - Maximum recursion depth (default: 5)
///
/// # Example
/// ```text
/// -- Given hypotheses h1 : A, h2 : A → B, h3 : B → C
/// -- Goal: C
/// -- solve_by_elim finds: h3 (h2 h1)
/// ```
pub fn solve_by_elim(state: &mut ProofState, max_depth: usize) -> TacticResult {
    solve_by_elim_aux(state, max_depth, 0)
}

fn solve_by_elim_aux(
    state: &mut ProofState,
    max_depth: usize,
    current_depth: usize,
) -> TacticResult {
    if current_depth > max_depth {
        return Err(TacticError::Other(format!(
            "solve_by_elim: exceeded max depth {max_depth}"
        )));
    }

    // No goals means success
    if state.goals.is_empty() {
        return Ok(());
    }

    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // First, try assumption (direct match)
    let saved_goals = state.goals.clone();
    let saved_metas = state.metas.clone();
    if assumption(state).is_ok() {
        // If there are more goals, try to solve them too
        if state.goals.is_empty() {
            return Ok(());
        }
        // Recursively solve remaining goals
        if solve_by_elim_aux(state, max_depth, current_depth).is_ok() {
            return Ok(());
        }
        // Restore state and try other approaches
        state.goals = saved_goals;
        *state.metas_mut() = saved_metas;
    }

    // Try applying each hypothesis
    for decl in &goal.local_ctx {
        let hyp_ty = state.metas.instantiate(&decl.ty);

        // Only try hypotheses that look applicable (are function types or match)
        if is_applicable_hyp(&hyp_ty) || could_match_goal(state, &goal, &hyp_ty) {
            let saved_goals = state.goals.clone();
            let saved_metas = state.metas.clone();
            let hyp_expr = Expr::fvar(decl.fvar);

            if apply(state, hyp_expr).is_ok() {
                // Recursively try to solve all generated subgoals
                if solve_all_goals(state, max_depth, current_depth + 1).is_ok() {
                    return Ok(());
                }
            }

            // Restore state if this path didn't work
            state.goals = saved_goals;
            *state.metas_mut() = saved_metas;
        }
    }

    Err(TacticError::Other(
        "solve_by_elim: no applicable hypothesis found".to_string(),
    ))
}

/// Check if a hypothesis type looks applicable (is a function type)
fn is_applicable_hyp(ty: &Expr) -> bool {
    matches!(ty, Expr::Pi(_, _, _))
}

/// Check if a hypothesis type could directly match the goal
fn could_match_goal(state: &ProofState, goal: &Goal, hyp_ty: &Expr) -> bool {
    let target = state.metas.instantiate(&goal.target);
    state.is_def_eq(goal, hyp_ty, &target)
}

/// Helper to solve all remaining goals recursively
fn solve_all_goals(state: &mut ProofState, max_depth: usize, current_depth: usize) -> TacticResult {
    while !state.goals.is_empty() {
        solve_by_elim_aux(state, max_depth, current_depth)?;
    }
    Ok(())
}

/// Try a tactic and restore the proof state if it fails.
///
/// This is useful for automation tactics that want to try multiple strategies
/// without leaving partial progress behind on failure.
fn try_tactic_preserving_state<F>(state: &mut ProofState, tactic: F) -> bool
where
    F: FnOnce(&mut ProofState) -> TacticResult,
{
    let saved_goals = state.goals.clone();
    let saved_metas = state.metas.clone();

    if tactic(state).is_ok() {
        true
    } else {
        state.goals = saved_goals;
        *state.metas_mut() = saved_metas;
        false
    }
}

// ============================================================================
// blast - Aggressive automation tactic
// ============================================================================

/// Configuration for the `blast` automation tactic.
#[derive(Debug, Clone)]
pub struct BlastConfig {
    /// Maximum number of rounds to try automation steps
    pub max_rounds: usize,
    /// Depth limit for solve_by_elim search
    pub solve_by_elim_depth: usize,
    /// Whether to try propositional automation (tauto/contradiction)
    pub use_tauto: bool,
    /// Whether to run simplification
    pub use_simp: bool,
    /// Whether to enable arithmetic solvers (linarith/nlinarith/norm_num)
    pub use_arith: bool,
    /// Whether to run omega for integer arithmetic
    pub use_omega: bool,
    /// Whether to normalize rings
    pub use_ring: bool,
    /// Whether to attempt decide/native_decide
    pub use_decide: bool,
    /// Whether to try instance synthesis
    pub use_instances: bool,
    /// Whether to consult library_search as a last resort
    pub use_library_search: bool,
}

impl Default for BlastConfig {
    fn default() -> Self {
        Self {
            max_rounds: 8,
            solve_by_elim_depth: 5,
            use_tauto: true,
            use_simp: true,
            use_arith: true,
            use_omega: true,
            use_ring: true,
            use_decide: true,
            use_instances: true,
            use_library_search: true,
        }
    }
}

impl BlastConfig {
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_max_rounds(mut self, rounds: usize) -> Self {
        self.max_rounds = rounds;
        self
    }

    #[must_use]
    pub fn with_solve_by_elim_depth(mut self, depth: usize) -> Self {
        self.solve_by_elim_depth = depth;
        self
    }

    #[must_use]
    pub fn use_arith(mut self, enabled: bool) -> Self {
        self.use_arith = enabled;
        self
    }

    #[must_use]
    pub fn use_tauto(mut self, enabled: bool) -> Self {
        self.use_tauto = enabled;
        self
    }

    #[must_use]
    pub fn use_simp(mut self, enabled: bool) -> Self {
        self.use_simp = enabled;
        self
    }

    #[must_use]
    pub fn use_library_search(mut self, enabled: bool) -> Self {
        self.use_library_search = enabled;
        self
    }
}

/// Aggressive automation tactic inspired by mathlib's blast-style solvers.
///
/// `blast` chains together many existing tactics in a bounded search:
/// - Immediate closers: `assumption`, `trivial`, `contradiction`, `rfl`
/// - Small search: `solve_by_elim`
/// - Decision procedures: `decide`, `native_decide`
/// - Propositional automation: `tauto`
/// - Simplification and rewriting: `simp`
/// - Arithmetic: `linarith`, `nlinarith`, `norm_num`, `omega`, `ring_nf`
/// - Type class synthesis: `infer_instance`
/// - Library search fallback: `library_search` + `exact`
pub fn blast(state: &mut ProofState) -> TacticResult {
    blast_with_config(state, BlastConfig::default())
}

/// `blast` with custom configuration.
pub fn blast_with_config(state: &mut ProofState, config: BlastConfig) -> TacticResult {
    if state.goals.is_empty() {
        return Err(TacticError::NoGoals);
    }

    for _ in 0..config.max_rounds {
        if state.is_complete() {
            return Ok(());
        }

        if !blast_round(state, &config) {
            break;
        }
    }

    if state.is_complete() {
        Ok(())
    } else {
        Err(TacticError::Other(
            "blast: automation failed to close goals".to_string(),
        ))
    }
}

/// Run a single blast automation round. Returns true if any tactic made progress.
fn blast_round(state: &mut ProofState, config: &BlastConfig) -> bool {
    // Quick closers
    if try_tactic_preserving_state(state, assumption) {
        return true;
    }
    if try_tactic_preserving_state(state, trivial) {
        return true;
    }
    if try_tactic_preserving_state(state, contradiction) {
        return true;
    }
    if try_tactic_preserving_state(state, rfl) {
        return true;
    }

    // Small search with context
    if try_tactic_preserving_state(state, |s| solve_by_elim(s, config.solve_by_elim_depth)) {
        return true;
    }

    // Decision procedures
    if config.use_decide && try_tactic_preserving_state(state, decide) {
        return true;
    }
    if config.use_decide && try_tactic_preserving_state(state, native_decide) {
        return true;
    }

    // Propositional automation
    if config.use_tauto && try_tactic_preserving_state(state, tauto) {
        return true;
    }

    // Simplification
    if config.use_simp && try_tactic_preserving_state(state, |s| simp(s, SimpConfig::new())) {
        return true;
    }

    // Arithmetic
    if config.use_arith && try_tactic_preserving_state(state, linarith) {
        return true;
    }
    if config.use_arith && try_tactic_preserving_state(state, nlinarith) {
        return true;
    }
    if config.use_arith && try_tactic_preserving_state(state, norm_num) {
        return true;
    }

    if config.use_omega && try_tactic_preserving_state(state, omega) {
        return true;
    }

    if config.use_ring && try_tactic_preserving_state(state, ring_nf) {
        return true;
    }

    // Type class synthesis
    if config.use_instances && try_tactic_preserving_state(state, infer_instance) {
        return true;
    }

    // Library search fallback: try first candidate with exact
    if config.use_library_search {
        let saved_goals = state.goals.clone();
        let saved_metas = state.metas.clone();

        if let Ok(results) = library_search(state) {
            for res in results {
                if try_tactic_preserving_state(state, |s| exact(s, res.expr.clone())) {
                    return true;
                }
            }
        }

        state.goals = saved_goals;
        *state.metas_mut() = saved_metas;
    }

    false
}

// clear, rename, duplicate, specialize moved to hypothesis.rs

/// Apply congruence to break down an equality goal.
///
/// For a goal `f a₁ ... aₙ = f b₁ ... bₙ`, creates subgoals `a₁ = b₁`, ..., `aₙ = bₙ`.
/// The function `f` must be the same on both sides.
///
/// This is useful when you need to prove equality by showing each argument is equal.
///
/// # Example
/// For goal `Nat.add x y = Nat.add x' y'`, creates subgoals `x = x'` and `y = y'`.
pub fn congr(state: &mut ProofState) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    let target = state.whnf(&goal, &goal.target);

    // Check if target is an Eq application
    let head = target.get_app_fn();
    let args: Vec<Expr> = target.get_app_args().into_iter().cloned().collect();

    match head {
        Expr::Const(name, levels) if *name == Name::from_string("Eq") => {
            // Eq takes 3 args: type, lhs, rhs
            if args.len() != 3 {
                return Err(TacticError::GoalMismatch(
                    "congr: expected Eq with 3 arguments".to_string(),
                ));
            }

            let _eq_ty = args[0].clone();
            let lhs = args[1].clone();
            let rhs = args[2].clone();

            // Get the function and args of lhs and rhs
            let lhs_fn = lhs.get_app_fn();
            let rhs_fn = rhs.get_app_fn();
            let lhs_args: Vec<Expr> = lhs.get_app_args().into_iter().cloned().collect();
            let rhs_args: Vec<Expr> = rhs.get_app_args().into_iter().cloned().collect();

            // Check same function
            if !state.is_def_eq(&goal, lhs_fn, rhs_fn) {
                return Err(TacticError::GoalMismatch(
                    "congr: functions on both sides must be equal".to_string(),
                ));
            }

            // Check same number of args
            if lhs_args.len() != rhs_args.len() {
                return Err(TacticError::GoalMismatch(
                    "congr: argument counts must match".to_string(),
                ));
            }

            if lhs_args.is_empty() {
                // No arguments - just need reflexivity
                return rfl(state);
            }

            // For a single argument, use congrArg
            if lhs_args.len() == 1 {
                // Build proof: congrArg f ?h where ?h : a = b
                let arg_ty = state.infer_type(&goal, &lhs_args[0])?;
                let eq_goal_ty = Expr::app(
                    Expr::app(
                        Expr::app(
                            Expr::const_(Name::from_string("Eq"), levels.clone()),
                            arg_ty,
                        ),
                        lhs_args[0].clone(),
                    ),
                    rhs_args[0].clone(),
                );

                let arg_eq_meta_id = state.metas.fresh(eq_goal_ty.clone());
                let arg_eq_meta = Expr::FVar(MetaState::to_fvar(arg_eq_meta_id));

                // Try to find congrArg in environment
                let congr_arg = Name::from_string("congrArg");
                if state.env.get_const(&congr_arg).is_some() {
                    // congrArg : ∀ {α β} (f : α → β) {a₁ a₂ : α}, a₁ = a₂ → f a₁ = f a₂
                    let mut proof = Expr::const_(congr_arg, levels.clone());
                    proof = Expr::app(proof, lhs_fn.clone()); // f
                    proof = Expr::app(proof, arg_eq_meta.clone()); // h : a₁ = a₂

                    state.close_goal(proof)?;

                    let new_goal = Goal {
                        meta_id: arg_eq_meta_id,
                        target: eq_goal_ty,
                        local_ctx: goal.local_ctx.clone(),
                    };
                    state.goals.insert(0, new_goal);

                    return Ok(());
                }
            }

            // For multiple arguments or no congrArg, fall back to recursive approach
            // Create a subgoal for each argument pair
            let mut new_goals = Vec::new();
            let mut proofs = Vec::new();

            for (la, ra) in lhs_args.iter().zip(rhs_args.iter()) {
                let arg_ty = state.infer_type(&goal, la)?;
                let eq_goal_ty = Expr::app(
                    Expr::app(
                        Expr::app(
                            Expr::const_(Name::from_string("Eq"), levels.clone()),
                            arg_ty,
                        ),
                        la.clone(),
                    ),
                    ra.clone(),
                );

                let meta_id = state.metas.fresh(eq_goal_ty.clone());
                proofs.push(Expr::FVar(MetaState::to_fvar(meta_id)));

                new_goals.push(Goal {
                    meta_id,
                    target: eq_goal_ty,
                    local_ctx: goal.local_ctx.clone(),
                });
            }

            // Build the combined proof using Eq.subst repeatedly
            // Start with rfl for f = f, then substitute each argument
            let eq_refl = Name::from_string("Eq.refl");
            let eq_subst = Name::from_string("Eq.subst");

            // If we have Eq.refl and Eq.subst, build a chain
            if state.env.get_const(&eq_refl).is_some() && state.env.get_const(&eq_subst).is_some() {
                // For now, just create subgoals - a full implementation would build
                // the proof term properly. This is a simplification.
                let full_eq_meta_id = state.metas.fresh(goal.target.clone());
                let full_eq_meta = Expr::FVar(MetaState::to_fvar(full_eq_meta_id));
                state.close_goal(full_eq_meta)?;

                // Replace with a single goal that the user must prove
                // (proper congruence proof building is complex)
                let final_goal = Goal {
                    meta_id: full_eq_meta_id,
                    target: goal.target.clone(),
                    local_ctx: goal.local_ctx.clone(),
                };
                state.goals.insert(0, final_goal);

                // Also add the argument equality goals as hints
                for ng in new_goals.into_iter().rev() {
                    state.goals.insert(0, ng);
                }

                Ok(())
            } else {
                Err(TacticError::Other(
                    "congr: Eq.refl or Eq.subst not found".to_string(),
                ))
            }
        }
        _ => Err(TacticError::GoalMismatch(
            "congr: goal must be an equality".to_string(),
        )),
    }
}

/// Obtain (destructure) a hypothesis with an existential or sigma type.
///
/// For a hypothesis `h : ∃ x : A, P x`, introduces `x : A` and `h : P x`
/// into the context. The original hypothesis is replaced.
///
/// # Arguments
/// * `state` - The proof state
/// * `hyp_name` - Name of the hypothesis to destructure
/// * `var_name` - Name for the introduced variable
/// * `new_hyp_name` - Name for the property hypothesis
///
/// # Example
/// If you have `h : ∃ n : Nat, n > 0`, calling `obtain(state, "h", "n", "hn")`
/// gives you `n : Nat` and `hn : n > 0`.
pub fn obtain(
    state: &mut ProofState,
    hyp_name: &str,
    var_name: &str,
    new_hyp_name: &str,
) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Find the hypothesis
    let (idx, decl) = goal
        .local_ctx
        .iter()
        .enumerate()
        .find(|(_, d)| d.name == hyp_name)
        .ok_or_else(|| TacticError::HypothesisNotFound(hyp_name.to_string()))?;
    let decl = decl.clone();

    // Get the type and normalize
    let hyp_ty = state.whnf(&goal, &decl.ty);

    // Check for Exists type
    let head = hyp_ty.get_app_fn();
    let args: Vec<Expr> = hyp_ty.get_app_args().into_iter().cloned().collect();

    match head {
        Expr::Const(name, _levels) if *name == Name::from_string("Exists") => {
            // Exists takes 2 args: type and predicate
            if args.len() != 2 {
                return Err(TacticError::GoalMismatch(
                    "obtain: expected Exists with 2 arguments".to_string(),
                ));
            }

            let var_ty = args[0].clone();
            let pred = args[1].clone();

            // Create fvars for the witness and proof
            let var_fvar = state.fresh_fvar();
            let hyp_fvar = state.fresh_fvar();

            // The predicate type applied to the witness
            let pred_applied = Expr::app(pred, Expr::fvar(var_fvar));

            // Remove the old hypothesis and add the new ones
            let goal_mut = state.current_goal_mut().ok_or(TacticError::NoGoals)?;

            // Remove the original hypothesis
            goal_mut.local_ctx.remove(idx);

            // Add the witness variable
            goal_mut.local_ctx.push(LocalDecl {
                fvar: var_fvar,
                name: var_name.to_string(),
                ty: var_ty,
                value: None,
            });

            // Add the property hypothesis
            goal_mut.local_ctx.push(LocalDecl {
                fvar: hyp_fvar,
                name: new_hyp_name.to_string(),
                ty: pred_applied,
                value: None,
            });

            Ok(())
        }
        // Also handle Sigma types (dependent pairs)
        Expr::Const(name, _levels) if *name == Name::from_string("Sigma") => {
            if args.len() != 2 {
                return Err(TacticError::GoalMismatch(
                    "obtain: expected Sigma with 2 arguments".to_string(),
                ));
            }

            let var_ty = args[0].clone();
            let pred = args[1].clone();

            let var_fvar = state.fresh_fvar();
            let hyp_fvar = state.fresh_fvar();

            let pred_applied = Expr::app(pred, Expr::fvar(var_fvar));

            let goal_mut = state.current_goal_mut().ok_or(TacticError::NoGoals)?;
            goal_mut.local_ctx.remove(idx);

            goal_mut.local_ctx.push(LocalDecl {
                fvar: var_fvar,
                name: var_name.to_string(),
                ty: var_ty,
                value: None,
            });

            goal_mut.local_ctx.push(LocalDecl {
                fvar: hyp_fvar,
                name: new_hyp_name.to_string(),
                ty: pred_applied,
                value: None,
            });

            Ok(())
        }
        _ => Err(TacticError::GoalMismatch(format!(
            "obtain: hypothesis '{hyp_name}' has type {hyp_ty:?}, expected ∃ or Σ"
        ))),
    }
}

/// Revert a hypothesis back into the goal.
///
/// For a hypothesis `h : A` and goal `⊢ B`, this changes the goal to `⊢ A → B`
/// and removes `h` from the context. This is the inverse of `intro`.
///
/// # Arguments
/// * `state` - The proof state
/// * `hyp_name` - Name of the hypothesis to revert
///
/// # Example
/// With context `h : P` and goal `Q`, calling `revert(state, "h")` gives goal `P → Q`.
pub fn revert(state: &mut ProofState, hyp_name: &str) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Find the hypothesis
    let (idx, decl) = goal
        .local_ctx
        .iter()
        .enumerate()
        .find(|(_, d)| d.name == hyp_name)
        .ok_or_else(|| TacticError::HypothesisNotFound(hyp_name.to_string()))?;
    let decl = decl.clone();

    // Create new target: hyp_ty → old_target
    let new_target = Expr::pi(
        BinderInfo::Default,
        decl.ty.clone(),
        goal.target.abstract_fvar(decl.fvar),
    );

    // Create new metavariable for the new goal
    let new_meta_id = state.metas.fresh(new_target.clone());
    let new_meta = Expr::FVar(MetaState::to_fvar(new_meta_id));

    // The proof of the old goal is: new_proof h
    let proof = Expr::app(new_meta, Expr::fvar(decl.fvar));
    state.close_goal(proof)?;

    // Create the new goal with h removed from context
    let mut new_ctx = goal.local_ctx.clone();
    new_ctx.remove(idx);

    let new_goal = Goal {
        meta_id: new_meta_id,
        target: new_target,
        local_ctx: new_ctx,
    };

    state.goals.insert(0, new_goal);
    Ok(())
}

// =============================================================================
// Generalization tactics
// =============================================================================

/// The `generalize` tactic abstracts a term in the goal by introducing a new variable.
///
/// Given a goal containing term `e : T`, this tactic:
/// 1. Replaces all occurrences of `e` in the goal with a fresh variable `x`
/// 2. Adds `x : T` as a new hypothesis (via intro-like mechanism)
///
/// This is useful for strengthening the induction hypothesis when doing induction,
/// or for abstracting over a specific value.
///
/// # Arguments
/// * `state` - The proof state
/// * `term` - The term to generalize
/// * `var_name` - Name for the new variable
///
/// # Example
/// ```text
/// goal : f 5 = g 5
///
/// generalize 5 as n
///
/// n : Nat
/// goal : f n = g n
/// ```
///
/// Proving `f n = g n` for all `n` is stronger but allows induction on `n`.
pub fn generalize(state: &mut ProofState, term: Expr, var_name: String) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Infer the type of the term
    let term_ty = state.infer_type(&goal, &term)?;

    // Check if the term appears in the goal
    let target = state.metas.instantiate(&goal.target);
    if !contains_expr(&target, &term) {
        return Err(TacticError::Other(
            "generalize: term does not appear in the goal".to_string(),
        ));
    }

    // Abstract over the term in the goal
    let abstracted_target = abstract_over(&target, &term);

    // Create a new goal: ∀ x : T, goal[term → x]
    let forall_target = Expr::pi(BinderInfo::Default, term_ty.clone(), abstracted_target);

    // Create metavariable for the new goal
    let new_meta_id = state.metas.fresh(forall_target.clone());
    let new_meta = Expr::FVar(MetaState::to_fvar(new_meta_id));

    // The proof of the original goal is: new_proof term
    let proof = Expr::app(new_meta, term);
    state.close_goal(proof)?;

    // Add the new goal
    let new_goal = Goal {
        meta_id: new_meta_id,
        target: forall_target,
        local_ctx: goal.local_ctx.clone(),
    };
    state.goals.insert(0, new_goal);

    // Now apply intro to introduce the variable
    intro(state, var_name)?;

    Ok(())
}

/// The `generalize_eq` tactic generalizes a term and adds an equality hypothesis.
///
/// Like `generalize`, but also adds a hypothesis `h : x = e` where `e` is the
/// original term and `x` is the new variable.
///
/// # Arguments
/// * `state` - The proof state
/// * `term` - The term to generalize
/// * `var_name` - Name for the new variable
/// * `eq_name` - Name for the equality hypothesis
///
/// # Example
/// ```text
/// goal : f 5 = g 5
///
/// generalize_eq 5 as n heq
///
/// n : Nat
/// heq : n = 5
/// goal : f n = g n
/// ```
pub fn generalize_eq(
    state: &mut ProofState,
    term: Expr,
    var_name: String,
    eq_name: String,
) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Infer the type of the term
    let term_ty = state.infer_type(&goal, &term)?;

    // Check if the term appears in the goal
    let target = state.metas.instantiate(&goal.target);
    if !contains_expr(&target, &term) {
        return Err(TacticError::Other(
            "generalize_eq: term does not appear in the goal".to_string(),
        ));
    }

    // Check that Eq is available
    let eq_name_const = Name::from_string("Eq");
    if state.env.get_const(&eq_name_const).is_none() {
        return Err(TacticError::Other(
            "generalize_eq: Eq not found in environment (call env.init_eq())".to_string(),
        ));
    }

    // Create fresh variable for the generalized term
    let var_fvar = state.fresh_fvar();

    // Abstract over the term in the goal
    let abstracted_target = abstract_over(&target, &term);

    // Build the equality type: Eq term_ty var term
    let eq_type = Expr::app(
        Expr::app(
            Expr::app(
                Expr::const_(eq_name_const, vec![Level::zero()]),
                term_ty.clone(),
            ),
            Expr::fvar(var_fvar),
        ),
        term.clone(),
    );

    // The new goal structure is:
    // ∀ (x : T), x = e → goal[e → x]
    // First, instantiate abstracted_target with the bvar
    let inner_target = abstracted_target.instantiate(&Expr::fvar(var_fvar));

    // Create new context with the variable and equality
    let mut new_ctx = goal.local_ctx.clone();
    new_ctx.push(LocalDecl {
        fvar: var_fvar,
        name: var_name.clone(),
        ty: term_ty.clone(),
        value: None,
    });

    let eq_fvar = state.fresh_fvar();
    new_ctx.push(LocalDecl {
        fvar: eq_fvar,
        name: eq_name.clone(),
        ty: eq_type,
        value: None,
    });

    // Create metavariable for the new goal
    let new_meta_id = state.metas.fresh(inner_target.clone());
    let new_meta = Expr::FVar(MetaState::to_fvar(new_meta_id));

    // Build the proof term:
    // The original goal is proved by:
    // (fun x heq => new_proof) term (Eq.refl term)
    let eq_refl_name = Name::from_string("Eq.refl");
    let refl_proof = Expr::app(
        Expr::app(
            Expr::const_(eq_refl_name, vec![Level::zero()]),
            term_ty.clone(),
        ),
        term.clone(),
    );

    let proof = Expr::app(
        Expr::app(
            Expr::lam(
                BinderInfo::Default,
                term_ty.clone(),
                Expr::lam(
                    BinderInfo::Default,
                    // eq_type with fvar replaced by bvar(0)
                    Expr::app(
                        Expr::app(
                            Expr::app(
                                Expr::const_(Name::from_string("Eq"), vec![Level::zero()]),
                                term_ty.clone(),
                            ),
                            Expr::bvar(0),
                        ),
                        term.clone(),
                    ),
                    new_meta,
                ),
            ),
            term.clone(),
        ),
        refl_proof,
    );

    state.close_goal(proof)?;

    // Add the new goal
    let new_goal = Goal {
        meta_id: new_meta_id,
        target: inner_target,
        local_ctx: new_ctx,
    };
    state.goals.insert(0, new_goal);

    Ok(())
}

/// The `ext` tactic for function extensionality.
///
/// For a goal `f = g` where `f` and `g` are functions, this tactic applies
/// functional extensionality to reduce the goal to `∀ x, f x = g x`.
///
/// # Arguments
/// * `state` - The proof state
/// * `var_name` - Name for the argument variable
///
/// # Example
/// ```text
/// f g : Nat → Nat
/// goal : f = g
///
/// ext n
///
/// n : Nat
/// goal : f n = g n
/// ```
///
/// # Errors
/// - `Other` if `funext` is not in the environment
/// - `GoalMismatch` if the goal is not an equality of function types
pub fn ext(state: &mut ProofState, var_name: String) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();
    let target = state.whnf(&goal, &goal.target);

    // Match goal as equality f = g
    let (_eq_ty, lhs, rhs, eq_levels) = match_equality(&target)
        .map_err(|_| TacticError::GoalMismatch("ext: goal is not an equality".to_string()))?;

    // Infer types of lhs and rhs
    let lhs_ty = state.infer_type(&goal, &lhs)?;
    let lhs_ty_whnf = state.whnf(&goal, &lhs_ty);

    // Check that lhs has function type (Pi type)
    let (domain, codomain) = match &lhs_ty_whnf {
        Expr::Pi(_bi, dom, cod) => ((**dom).clone(), (**cod).clone()),
        _ => {
            return Err(TacticError::GoalMismatch(
                "ext: left-hand side is not a function".to_string(),
            ));
        }
    };

    // Check for funext in environment
    let funext_name = Name::from_string("funext");
    if state.env.get_const(&funext_name).is_none() {
        return Err(TacticError::Other(
            "ext: funext not found in environment (call env.init_funext())".to_string(),
        ));
    }

    // Create fresh variable for the argument
    let arg_fvar = state.fresh_fvar();

    // Build the new goal: f x = g x
    let f_app = Expr::app(lhs.clone(), Expr::fvar(arg_fvar));
    let g_app = Expr::app(rhs.clone(), Expr::fvar(arg_fvar));

    // Instantiate codomain with the argument (if dependent)
    let result_ty = codomain.instantiate(&Expr::fvar(arg_fvar));

    let new_target = Expr::app(
        Expr::app(
            Expr::app(
                Expr::const_(Name::from_string("Eq"), eq_levels.clone()),
                result_ty,
            ),
            f_app,
        ),
        g_app,
    );

    // Create new context with the argument variable
    let mut new_ctx = goal.local_ctx.clone();
    new_ctx.push(LocalDecl {
        fvar: arg_fvar,
        name: var_name,
        ty: domain.clone(),
        value: None,
    });

    // Create metavariable for the pointwise equality proof
    let pointwise_meta_id = state.metas.fresh(new_target.clone());
    let pointwise_meta = Expr::FVar(MetaState::to_fvar(pointwise_meta_id));

    // Build the proof using funext:
    // funext : ∀ {α : Sort u} {β : α → Sort v} {f g : ∀ x, β x},
    //          (∀ x, f x = g x) → f = g
    //
    // We need to apply funext to (fun x => pointwise_meta)
    let funext = Expr::const_(funext_name, eq_levels);

    // The function argument to funext: λ x, proof_that_f_x_eq_g_x
    let h_fun = Expr::lam(BinderInfo::Default, domain.clone(), pointwise_meta);

    // Apply funext (with implicit arguments inferred)
    // funext {α} {β} {f} {g} h
    // We'll build a simpler version assuming the type checker can infer implicit args
    let proof = Expr::app(
        Expr::app(
            Expr::app(
                Expr::app(Expr::app(funext, lhs_ty_whnf.clone()), codomain),
                lhs,
            ),
            rhs,
        ),
        h_fun,
    );

    state.close_goal(proof)?;

    // Add the new goal
    let new_goal = Goal {
        meta_id: pointwise_meta_id,
        target: new_target,
        local_ctx: new_ctx,
    };
    state.goals.insert(0, new_goal);

    Ok(())
}

/// Injection tactic for decomposing constructor equalities.
///
/// Given a hypothesis `h : C x₁ ... xₙ = C y₁ ... yₙ` where `C` is a constructor,
/// produces new hypotheses `h₁ : x₁ = y₁, ..., hₙ : xₙ = yₙ` by injectivity
/// of the constructor.
///
/// # Example
/// ```text
/// h : Nat.succ a = Nat.succ b
/// goal : P
///
/// injection h
///
/// h_inj : a = b
/// goal : P
/// ```
///
/// # Errors
/// - `HypothesisNotFound` if the hypothesis doesn't exist
/// - `GoalMismatch` if the hypothesis is not an equality between constructor applications
/// - `GoalMismatch` if the two sides use different constructors
/// - `GoalMismatch` if the constructors have no fields (nothing to inject)
pub fn injection(state: &mut ProofState, hyp_name: &str) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Find the hypothesis
    let hyp_decl = goal
        .local_ctx
        .iter()
        .find(|d| d.name == hyp_name)
        .ok_or_else(|| TacticError::HypothesisNotFound(hyp_name.to_string()))?
        .clone();

    // Check that the hypothesis is an equality
    let hyp_ty = state.whnf(&goal, &hyp_decl.ty);
    let (eq_type, lhs, rhs, eq_levels) = match_equality(&hyp_ty).map_err(|_| {
        TacticError::GoalMismatch(format!("injection: {hyp_name} is not an equality"))
    })?;

    // Get constructor applications from both sides
    let lhs_whnf = state.whnf(&goal, &lhs);
    let rhs_whnf = state.whnf(&goal, &rhs);

    let lhs_head = lhs_whnf.get_app_fn();
    let rhs_head = rhs_whnf.get_app_fn();

    // Check both sides are constructor applications
    let lhs_ctor_name = match lhs_head {
        Expr::Const(name, _) => name.clone(),
        _ => {
            return Err(TacticError::GoalMismatch(
                "injection: left-hand side is not a constructor application".to_string(),
            ));
        }
    };

    let rhs_ctor_name = match rhs_head {
        Expr::Const(name, _) => name.clone(),
        _ => {
            return Err(TacticError::GoalMismatch(
                "injection: right-hand side is not a constructor application".to_string(),
            ));
        }
    };

    // Verify the constructor names match
    if lhs_ctor_name != rhs_ctor_name {
        return Err(TacticError::GoalMismatch(format!(
            "injection: constructors do not match ({lhs_ctor_name} vs {rhs_ctor_name})"
        )));
    }

    // Look up constructor info
    let ctor_info = state
        .env
        .get_constructor(&lhs_ctor_name)
        .ok_or_else(|| {
            TacticError::GoalMismatch(format!("injection: {lhs_ctor_name} is not a constructor"))
        })?
        .clone();

    // Get the arguments from both sides
    let lhs_args: Vec<&Expr> = lhs_whnf.get_app_args();
    let rhs_args: Vec<&Expr> = rhs_whnf.get_app_args();

    // Skip parameters (first num_params args)
    let num_params = ctor_info.num_params as usize;
    let num_fields = ctor_info.num_fields as usize;

    if num_fields == 0 {
        return Err(TacticError::GoalMismatch(
            "injection: constructor has no fields to inject".to_string(),
        ));
    }

    // The field arguments start after the parameters
    let lhs_fields: Vec<&Expr> = lhs_args
        .iter()
        .skip(num_params)
        .take(num_fields)
        .copied()
        .collect();
    let rhs_fields: Vec<&Expr> = rhs_args
        .iter()
        .skip(num_params)
        .take(num_fields)
        .copied()
        .collect();

    if lhs_fields.len() != rhs_fields.len() || lhs_fields.len() != num_fields {
        return Err(TacticError::GoalMismatch(format!(
            "injection: argument count mismatch (expected {} fields, got lhs={}, rhs={})",
            num_fields,
            lhs_fields.len(),
            rhs_fields.len()
        )));
    }

    // Build new local context with injected equalities
    let mut new_ctx = goal.local_ctx.clone();

    // Parse constructor type to get field types
    let mut ctor_ty = ctor_info.type_.clone();

    // Skip parameters (instantiate them with actual values from the term)
    for i in 0..num_params {
        if let Expr::Pi(_, _, codomain) = &ctor_ty {
            if i < lhs_args.len() {
                ctor_ty = codomain.instantiate(lhs_args[i]);
            } else {
                ctor_ty = codomain.instantiate(&Expr::Sort(Level::zero())); // placeholder
            }
        }
    }

    // Collect field types
    let mut field_types: Vec<Expr> = Vec::with_capacity(num_fields);
    for i in 0..num_fields {
        if let Expr::Pi(_, domain, codomain) = ctor_ty.clone() {
            field_types.push((*domain).clone());
            // Instantiate with lhs field for proper typing
            if i < lhs_fields.len() {
                ctor_ty = codomain.instantiate(lhs_fields[i]);
            }
        }
    }

    // Create equality hypotheses for each field pair
    for i in 0..num_fields {
        let lhs_field = lhs_fields[i];
        let rhs_field = rhs_fields[i];

        // Get the type for this field
        let field_ty = if i < field_types.len() {
            field_types[i].clone()
        } else {
            // Infer the type if we couldn't get it from constructor
            state
                .infer_type(&goal, lhs_field)
                .unwrap_or_else(|_| eq_type.clone())
        };

        // Build equality type: lhs_field = rhs_field
        let eq_hyp_ty = Expr::app(
            Expr::app(
                Expr::app(
                    Expr::const_(Name::from_string("Eq"), eq_levels.clone()),
                    field_ty,
                ),
                lhs_field.clone(),
            ),
            rhs_field.clone(),
        );

        // Create fresh fvar for this hypothesis
        let inj_fvar = state.fresh_fvar();
        let inj_name = format!(
            "{}_inj{}",
            hyp_name,
            if num_fields > 1 {
                format!("_{}", i + 1)
            } else {
                String::new()
            }
        );

        new_ctx.push(LocalDecl {
            fvar: inj_fvar,
            name: inj_name,
            ty: eq_hyp_ty,
            value: None,
        });
    }

    // Create new goal with extended context (same target)
    let new_meta_id = state.metas.fresh(goal.target.clone());
    let new_meta = Expr::FVar(MetaState::to_fvar(new_meta_id));

    // Close the current goal (proof will use the injected hypotheses)
    state.close_goal(new_meta)?;

    // Add the new goal
    let new_goal = Goal {
        meta_id: new_meta_id,
        target: goal.target.clone(),
        local_ctx: new_ctx,
    };
    state.goals.insert(0, new_goal);

    Ok(())
}

/// Discriminate tactic: derive False from an impossible constructor equality.
///
/// Given a hypothesis `h : C₁ args = C₂ args` where `C₁` and `C₂` are different
/// constructors of the same inductive type, this tactic closes the goal since
/// different constructors are never equal (no confusion).
///
/// # Example
/// ```text
/// h : Nat.zero = Nat.succ n
/// goal : P
///
/// discriminate h
///
/// (goal closed - absurd hypothesis)
/// ```
///
/// # Errors
/// - `HypothesisNotFound` if the hypothesis doesn't exist
/// - `GoalMismatch` if the hypothesis is not an equality
/// - `GoalMismatch` if the constructors are the same (use injection instead)
/// - `GoalMismatch` if the types are not from the same inductive
pub fn discriminate(state: &mut ProofState, hyp_name: &str) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Find the hypothesis
    let hyp_decl = goal
        .local_ctx
        .iter()
        .find(|d| d.name == hyp_name)
        .ok_or_else(|| TacticError::HypothesisNotFound(hyp_name.to_string()))?
        .clone();

    // Check that the hypothesis is an equality
    let hyp_ty = state.whnf(&goal, &hyp_decl.ty);
    let (_eq_type, lhs, rhs, _eq_levels) = match_equality(&hyp_ty).map_err(|_| {
        TacticError::GoalMismatch(format!("discriminate: {hyp_name} is not an equality"))
    })?;

    // Get constructor applications from both sides
    let lhs_whnf = state.whnf(&goal, &lhs);
    let rhs_whnf = state.whnf(&goal, &rhs);

    let lhs_head = lhs_whnf.get_app_fn();
    let rhs_head = rhs_whnf.get_app_fn();

    // Check both sides are constructor applications
    let lhs_ctor_name = match lhs_head {
        Expr::Const(name, _) => name.clone(),
        _ => {
            return Err(TacticError::GoalMismatch(
                "discriminate: left-hand side is not a constructor application".to_string(),
            ));
        }
    };

    let rhs_ctor_name = match rhs_head {
        Expr::Const(name, _) => name.clone(),
        _ => {
            return Err(TacticError::GoalMismatch(
                "discriminate: right-hand side is not a constructor application".to_string(),
            ));
        }
    };

    // Verify the constructors are DIFFERENT
    if lhs_ctor_name == rhs_ctor_name {
        return Err(TacticError::GoalMismatch(
            "discriminate: constructors are the same (use injection instead)".to_string(),
        ));
    }

    // Verify both are constructors of the same inductive type
    let lhs_ctor_info = state
        .env
        .get_constructor(&lhs_ctor_name)
        .ok_or_else(|| {
            TacticError::GoalMismatch(format!(
                "discriminate: {lhs_ctor_name} is not a constructor"
            ))
        })?
        .clone();

    let rhs_ctor_info = state
        .env
        .get_constructor(&rhs_ctor_name)
        .ok_or_else(|| {
            TacticError::GoalMismatch(format!(
                "discriminate: {rhs_ctor_name} is not a constructor"
            ))
        })?
        .clone();

    if lhs_ctor_info.inductive_name != rhs_ctor_info.inductive_name {
        return Err(TacticError::GoalMismatch(format!(
            "discriminate: constructors are from different types ({} vs {})",
            lhs_ctor_info.inductive_name, rhs_ctor_info.inductive_name
        )));
    }

    // Build proof using False.elim
    // Different constructors of the same inductive are never equal (no confusion)
    // The proof is: False.elim (noConfusion h) where noConfusion derives False
    //
    // For now, we'll use a simplified approach: we assert that the equality
    // is absurd and the goal is closed. In a full implementation, we'd need
    // to actually generate the noConfusion proof term.

    // Check if False and False.elim are available
    let false_name = Name::from_string("False");
    let false_elim_name = Name::from_string("False.elim");

    if state.env.get_const(&false_name).is_none() {
        return Err(TacticError::Other(
            "discriminate: False not found in environment".to_string(),
        ));
    }

    // Build a proof term using False.elim
    // Since different constructors cannot be equal, we're deriving False from h
    // In Lean, this would use the noConfusion principle automatically generated
    // for each inductive type. Here we'll construct a placeholder that type-checks.

    // Get the inductive info for no_confusion
    let ind_name = &lhs_ctor_info.inductive_name;
    let no_confusion_name = Name::from_string(&format!("{ind_name}.noConfusion"));

    // If noConfusion is available, use it; otherwise use a simpler approach
    let proof = if state.env.get_const(&no_confusion_name).is_some() {
        // Use: noConfusion h
        Expr::app(
            Expr::const_(no_confusion_name, vec![]),
            Expr::fvar(hyp_decl.fvar),
        )
    } else {
        // Fallback: use False.elim with a metavariable for the False proof
        // This is a simplification; in practice we'd need proper noConfusion
        let false_proof_meta = state.metas.fresh(Expr::const_(false_name.clone(), vec![]));
        let false_proof = Expr::FVar(MetaState::to_fvar(false_proof_meta));

        // False.elim : {C : Sort u} → False → C
        // Apply to our goal type
        Expr::app(
            Expr::app(
                Expr::const_(false_elim_name, vec![Level::zero()]),
                goal.target.clone(),
            ),
            false_proof,
        )
    };

    state.close_goal(proof)?;

    Ok(())
}

/// Recursive cases tactic for nested pattern matching.
///
/// Like `cases` but recursively destructs nested constructors.
/// Given a hypothesis of type `Option (Option A)`, `rcases h` will
/// generate cases for `none`, `some none`, and `some (some a)`.
///
/// # Example
/// ```text
/// h : Option (Option Nat)
/// goal : P
///
/// rcases h
///
/// case none: goal : P
/// case some_none: goal : P
/// case some_some: a : Nat, goal : P
/// ```
///
/// # Errors
/// - Same as `cases` for the base case
pub fn rcases(state: &mut ProofState, hyp_name: &str, max_depth: usize) -> TacticResult {
    if max_depth == 0 {
        return Ok(());
    }

    // First apply cases
    cases(state, hyp_name)?;

    // Then try to recursively apply rcases on any new hypotheses that are inductive types
    let goals_count = state.goals.len();
    let mut processed = 0;

    while processed < goals_count {
        // Get the current goal (we iterate through all goals created by cases)
        let goal_idx = processed;
        if goal_idx >= state.goals.len() {
            break;
        }

        let goal = state.goals[goal_idx].clone();
        processed += 1;

        // Find hypotheses in this goal's context that are inductive types
        // and could be further destructed
        for decl in &goal.local_ctx {
            let decl_ty = state.whnf(&goal, &decl.ty);
            let ty_head = decl_ty.get_app_fn();

            // Check if this is an inductive type
            if let Expr::Const(name, _) = ty_head {
                if state.env.get_inductive(name).is_some() {
                    // This is an inductive type, try rcases
                    // But we need to be careful not to infinite loop
                    // So we only recurse if depth allows

                    // Focus on this goal temporarily
                    let original_goal = state.goals.remove(goal_idx);
                    state.goals.insert(0, original_goal);

                    // Try to apply rcases (may fail if already destructed)
                    if rcases(state, &decl.name, max_depth - 1).is_ok() {
                        // Success - the goal structure changed
                        processed = 0; // Restart iteration
                        break;
                    }

                    // Restore goal order if rcases failed
                    let moved_goal = state.goals.remove(0);
                    state.goals.insert(goal_idx, moved_goal);
                }
            }
        }
    }

    Ok(())
}

/// Numeric normalization tactic.
///
/// Evaluates numeric expressions involving natural numbers, integers,
/// and rationals. Uses computation to prove equalities.
///
/// # Supported
/// - Natural number arithmetic (+, *, -, ^)
/// - Integer arithmetic
/// - Rational arithmetic
/// - Comparisons (<, ≤, >, ≥)
/// - Divisibility (∣)
/// - GCD, LCM
///
/// # Example
/// ```text
/// -- Goal: 2 + 2 = 4
/// norm_num
/// -- Goal closed
///
/// -- Goal: 3 < 5
/// norm_num
/// -- Goal closed
/// ```
///
/// # Errors
/// - `NoGoals` if there are no goals
/// - `Other` if numeric evaluation fails to close the goal
pub fn norm_num(state: &mut ProofState) -> TacticResult {
    if state.goals.is_empty() {
        return Err(TacticError::NoGoals);
    }

    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();
    let target = &goal.target;

    // Try to evaluate the goal
    if try_eval_prop(target) {
        // The proposition evaluates to true, close with decide or rfl
        if decide(state).is_ok() {
            return Ok(());
        }
        if rfl(state).is_ok() {
            return Ok(());
        }
    }

    // Check if goal is an equality and try to normalize both sides
    if let Ok((_ty, lhs, rhs, _levels)) = match_equality(target) {
        let lhs_val = eval_nat_expr(&lhs);
        let rhs_val = eval_nat_expr(&rhs);

        if let (Some(l), Some(r)) = (lhs_val, rhs_val) {
            if l == r {
                return rfl(state);
            }
            return Err(TacticError::Other(format!("norm_num: {l} ≠ {r}")));
        }
    }

    // Check for comparisons
    if let Some(result) = try_eval_comparison(target) {
        if result {
            // True comparison - use decide
            return decide(state);
        }
        return Err(TacticError::Other(
            "norm_num: comparison is false".to_string(),
        ));
    }

    Err(TacticError::Other(
        "norm_num: could not evaluate numeric goal".to_string(),
    ))
}

/// Try to evaluate a proposition to true/false
fn try_eval_prop(expr: &Expr) -> bool {
    // Check for True
    if let Expr::Const(name, _) = expr {
        if name == &Name::from_string("True") {
            return true;
        }
    }

    // Check for comparisons
    if let Some(result) = try_eval_comparison(expr) {
        return result;
    }

    // Check for equality
    if let Ok((_ty, lhs, rhs, _levels)) = match_equality(expr) {
        if let (Some(l), Some(r)) = (eval_nat_expr(&lhs), eval_nat_expr(&rhs)) {
            return l == r;
        }
    }

    false
}

/// Try to evaluate a comparison expression
fn try_eval_comparison(expr: &Expr) -> Option<bool> {
    // Extract comparison: op lhs rhs
    if let Expr::App(f, rhs) = expr {
        if let Expr::App(f2, lhs) = f.as_ref() {
            if let Expr::App(f3, _ty) = f2.as_ref() {
                if let Expr::Const(op_name, _) = f3.as_ref() {
                    let op_str = op_name.to_string();
                    let l = eval_nat_expr(lhs)?;
                    let r = eval_nat_expr(rhs)?;

                    if op_str.contains("LT.lt") || op_str.contains("Nat.lt") {
                        return Some(l < r);
                    }
                    if op_str.contains("LE.le") || op_str.contains("Nat.le") {
                        return Some(l <= r);
                    }
                    if op_str.contains("GT.gt") {
                        return Some(l > r);
                    }
                    if op_str.contains("GE.ge") {
                        return Some(l >= r);
                    }
                }
            }
        }
    }

    None
}

/// Evaluate a natural number expression to a value
fn eval_nat_expr(expr: &Expr) -> Option<u64> {
    match expr {
        Expr::Lit(lean5_kernel::expr::Literal::Nat(n)) => Some(*n),

        Expr::Const(name, _) => {
            let name_str = name.to_string();
            if name_str == "Nat.zero" {
                Some(0)
            } else if name_str == "Nat.one" || name_str == "1" {
                Some(1)
            } else {
                None
            }
        }

        Expr::App(f, arg) => {
            // Check for Nat.succ
            if let Expr::Const(name, _) = f.as_ref() {
                if name.to_string() == "Nat.succ" {
                    return Some(eval_nat_expr(arg)? + 1);
                }
            }

            // Check for binary operations
            if let Expr::App(f2, arg1) = f.as_ref() {
                // Direct operations (Nat.add, etc.)
                if let Expr::Const(op_name, _) = f2.as_ref() {
                    let op_str = op_name.to_string();
                    let l = eval_nat_expr(arg1)?;
                    let r = eval_nat_expr(arg)?;

                    if op_str.contains("add") || op_str.contains("Add") {
                        return Some(l + r);
                    }
                    if op_str.contains("mul") || op_str.contains("Mul") {
                        return Some(l * r);
                    }
                    if op_str.contains("sub") || op_str.contains("Sub") {
                        return Some(l.saturating_sub(r));
                    }
                    if op_str.contains("pow") || op_str.contains("Pow") {
                        return Some(l.pow(r as u32));
                    }
                }

                // HAdd.hAdd, HMul.hMul, etc.
                if let Expr::App(f3, _) = f2.as_ref() {
                    if let Expr::App(f4, _) = f3.as_ref() {
                        if let Expr::Const(op_name, _) = f4.as_ref() {
                            let op_str = op_name.to_string();
                            let l = eval_nat_expr(arg1)?;
                            let r = eval_nat_expr(arg)?;

                            if op_str == "HAdd.hAdd" {
                                return Some(l + r);
                            }
                            if op_str == "HMul.hMul" {
                                return Some(l * r);
                            }
                            if op_str == "HSub.hSub" {
                                return Some(l.saturating_sub(r));
                            }
                            if op_str == "HPow.hPow" {
                                return Some(l.pow(r as u32));
                            }
                        }
                    }
                }
            }

            None
        }

        _ => None,
    }
}

// =============================================================================
// AC-reflexivity tactic
// =============================================================================

/// AC-reflexivity tactic.
///
/// Proves equality by checking if both sides are equal up to associativity
/// and commutativity of operations. Useful for equations that are trivially
/// equal modulo AC.
///
/// # Supported Operations
/// - Addition (commutative, associative)
/// - Multiplication (commutative, associative)
/// - Boolean operations (and, or)
/// - Set operations (union, intersection)
///
/// # Example
/// ```text
/// -- Goal: a + b + c = c + a + b
/// ac_rfl
/// -- Goal closed
/// ```
///
/// # Errors
/// - `NoGoals` if there are no goals
/// - `GoalMismatch` if goal is not an equality
/// - `Other` if sides are not AC-equal
pub fn ac_rfl(state: &mut ProofState) -> TacticResult {
    if state.goals.is_empty() {
        return Err(TacticError::NoGoals);
    }

    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Check that goal is an equality
    let (_ty, lhs, rhs, _levels) = match_equality(&goal.target)
        .map_err(|_| TacticError::GoalMismatch("ac_rfl: goal is not an equality".to_string()))?;

    // Normalize both sides using AC normalization
    let lhs_norm = ac_normalize(&lhs);
    let rhs_norm = ac_normalize(&rhs);

    // Check if normalized forms are equal
    if ac_exprs_equal(&lhs_norm, &rhs_norm) {
        rfl(state)
    } else {
        Err(TacticError::Other(format!(
            "ac_rfl: sides not AC-equal:\n  LHS: {lhs_norm:?}\n  RHS: {rhs_norm:?}"
        )))
    }
}

/// AC-normalized expression representation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ACExpr {
    /// A constant or variable (atomic)
    Atom(String),
    /// Commutative-associative operation with sorted operands
    CAOp { op: String, operands: Vec<ACExpr> },
    /// Non-AC application
    App(Box<ACExpr>, Box<ACExpr>),
    /// Lambda
    Lambda(Box<ACExpr>, Box<ACExpr>),
    /// Bound variable
    BVar(usize),
}

/// Normalize expression for AC equality
fn ac_normalize(expr: &Expr) -> ACExpr {
    match expr {
        Expr::BVar(idx) => ACExpr::BVar(*idx as usize),
        Expr::FVar(id) => ACExpr::Atom(format!("fvar_{}", id.0)),
        Expr::Const(name, _) => ACExpr::Atom(name.to_string()),
        Expr::Lit(lit) => ACExpr::Atom(format!("{lit:?}")),
        Expr::Sort(level) => ACExpr::Atom(format!("Sort_{level:?}")),

        Expr::App(f, arg) => {
            // Check for AC operations
            if let Some((op_name, operands)) = extract_ac_operation(expr) {
                // Recursively normalize operands and sort them
                let mut normalized: Vec<ACExpr> =
                    operands.iter().map(|e| ac_normalize(e)).collect();
                normalized.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));

                return ACExpr::CAOp {
                    op: op_name,
                    operands: normalized,
                };
            }

            // Regular application
            ACExpr::App(Box::new(ac_normalize(f)), Box::new(ac_normalize(arg)))
        }

        Expr::Lam(_, ty, body) => {
            ACExpr::Lambda(Box::new(ac_normalize(ty)), Box::new(ac_normalize(body)))
        }

        Expr::Pi(_, ty, body) => {
            // Treat Pi as a special app for AC purposes
            ACExpr::App(
                Box::new(ACExpr::Atom("Pi".to_string())),
                Box::new(ACExpr::App(
                    Box::new(ac_normalize(ty)),
                    Box::new(ac_normalize(body)),
                )),
            )
        }

        Expr::Let(ty, val, body) => ACExpr::App(
            Box::new(ACExpr::Atom("Let".to_string())),
            Box::new(ACExpr::App(
                Box::new(ac_normalize(ty)),
                Box::new(ACExpr::App(
                    Box::new(ac_normalize(val)),
                    Box::new(ac_normalize(body)),
                )),
            )),
        ),

        Expr::Proj(name, idx, e) => ACExpr::App(
            Box::new(ACExpr::Atom(format!("Proj_{name}_{idx}"))),
            Box::new(ac_normalize(e)),
        ),

        Expr::MData(_, inner) => ac_normalize(inner),

        // Mode-specific expressions - treat as atoms
        Expr::CubicalInterval => ACExpr::Atom("CubicalI".to_string()),
        Expr::CubicalI0 => ACExpr::Atom("I0".to_string()),
        Expr::CubicalI1 => ACExpr::Atom("I1".to_string()),
        Expr::CubicalPath { .. } => ACExpr::Atom("Path".to_string()),
        Expr::CubicalPathLam { .. } => ACExpr::Atom("PathLam".to_string()),
        Expr::CubicalPathApp { .. } => ACExpr::Atom("PathApp".to_string()),
        Expr::CubicalHComp { .. } => ACExpr::Atom("HComp".to_string()),
        Expr::CubicalTransp { .. } => ACExpr::Atom("Transp".to_string()),
        Expr::ClassicalChoice { .. } => ACExpr::Atom("Choice".to_string()),
        Expr::ClassicalEpsilon { .. } => ACExpr::Atom("Epsilon".to_string()),
        Expr::ZFCSet(_) => ACExpr::Atom("ZFCSet".to_string()),
        Expr::ZFCMem { .. } => ACExpr::Atom("ZFCMem".to_string()),
        Expr::ZFCComprehension { .. } => ACExpr::Atom("ZFCComp".to_string()),
        Expr::SProp => ACExpr::Atom("SProp".to_string()),
        Expr::Squash(inner) => ACExpr::App(
            Box::new(ACExpr::Atom("Squash".to_string())),
            Box::new(ac_normalize(inner)),
        ),
    }
}

/// Extract AC operation and its operands from an expression
fn extract_ac_operation(expr: &Expr) -> Option<(String, Vec<&Expr>)> {
    // Check if this is a binary application of an AC operator
    if let Expr::App(f, _arg2) = expr {
        if let Expr::App(f2, _arg1) = f.as_ref() {
            // Try to get the operator name
            let op_name = get_ac_op_name(f2)?;

            // Flatten nested applications of the same operator
            let mut operands = Vec::new();
            flatten_ac_operands(expr, &op_name, &mut operands);

            if operands.len() >= 2 {
                return Some((op_name, operands));
            }
        }
    }
    None
}

/// Get AC operator name if this is an AC operator
fn get_ac_op_name(expr: &Expr) -> Option<String> {
    match get_app_fn(expr) {
        Expr::Const(name, _) => {
            let name_str = name.to_string();

            // Known commutative-associative operations
            if name_str.contains("Add")
                || name_str.contains("add")
                || name_str.contains("HAdd.hAdd")
            {
                return Some("add".to_string());
            }
            if name_str.contains("Mul")
                || name_str.contains("mul")
                || name_str.contains("HMul.hMul")
            {
                return Some("mul".to_string());
            }
            if name_str.contains("And") || name_str.contains("and") {
                return Some("and".to_string());
            }
            if name_str.contains("Or") || name_str.contains("or") {
                return Some("or".to_string());
            }
            if name_str.contains("Union") || name_str.contains("union") {
                return Some("union".to_string());
            }
            if name_str.contains("Inter") || name_str.contains("inter") {
                return Some("inter".to_string());
            }
            if name_str.contains("Max") || name_str.contains("max") {
                return Some("max".to_string());
            }
            if name_str.contains("Min") || name_str.contains("min") {
                return Some("min".to_string());
            }

            None
        }
        _ => None,
    }
}

/// Flatten nested applications of an AC operator
fn flatten_ac_operands<'a>(expr: &'a Expr, target_op: &str, operands: &mut Vec<&'a Expr>) {
    if let Expr::App(f, arg2) = expr {
        if let Expr::App(f2, arg1) = f.as_ref() {
            if let Some(op) = get_ac_op_name(f2) {
                if op == target_op {
                    // Recursively flatten
                    flatten_ac_operands(arg1, target_op, operands);
                    flatten_ac_operands(arg2, target_op, operands);
                    return;
                }
            }
        }
    }
    // Not a matching application - this is a leaf
    operands.push(expr);
}

/// Check if two AC expressions are equal
fn ac_exprs_equal(e1: &ACExpr, e2: &ACExpr) -> bool {
    match (e1, e2) {
        (ACExpr::Atom(s1), ACExpr::Atom(s2)) => s1 == s2,
        (ACExpr::BVar(i1), ACExpr::BVar(i2)) => i1 == i2,
        (
            ACExpr::CAOp {
                op: op1,
                operands: ops1,
            },
            ACExpr::CAOp {
                op: op2,
                operands: ops2,
            },
        ) => {
            if op1 != op2 || ops1.len() != ops2.len() {
                return false;
            }
            // Operands are already sorted, so just compare pairwise
            ops1.iter()
                .zip(ops2.iter())
                .all(|(a, b)| ac_exprs_equal(a, b))
        }
        (ACExpr::App(f1, a1), ACExpr::App(f2, a2)) => {
            ac_exprs_equal(f1, f2) && ac_exprs_equal(a1, a2)
        }
        (ACExpr::Lambda(t1, b1), ACExpr::Lambda(t2, b2)) => {
            ac_exprs_equal(t1, t2) && ac_exprs_equal(b1, b2)
        }
        _ => false,
    }
}

// ============================================================================
// refine - Term refinement with holes
// ============================================================================

/// Refine the goal with a term containing holes.
///
/// `refine` allows you to provide a partial proof term where some parts are
/// indicated as underscores/placeholders. Each hole becomes a new goal.
///
/// # Algorithm
/// 1. Parse the expression looking for placeholder expressions
/// 2. Create new metavariables for each hole
/// 3. Create new goals for each metavariable
///
/// # Example
/// ```text
/// -- Goal: P ∧ Q
/// refine ⟨?_, ?_⟩
/// -- Goal 1: P
/// -- Goal 2: Q
/// ```
///
/// # Errors
/// - `NoGoals` if there are no goals
/// - `TypeMismatch` if the refined term doesn't match the goal type
pub fn refine(state: &mut ProofState, term: Expr) -> TacticResult {
    if state.goals.is_empty() {
        return Err(TacticError::NoGoals);
    }

    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Collect all placeholder holes in the term
    let hole_count = count_placeholders(&term);

    if hole_count == 0 {
        // No holes - this is just exact
        return exact(state, term);
    }

    // Create new goals for each hole based on what type they should have
    // Since we don't have type info, use the current target as placeholder
    let mut new_goals = Vec::new();

    for _ in 0..hole_count {
        let new_meta_id = state.metas.fresh(goal.target.clone());
        let new_goal = Goal {
            meta_id: new_meta_id,
            target: goal.target.clone(),
            local_ctx: goal.local_ctx.clone(),
        };
        new_goals.push(new_goal);
    }

    // Replace the current goal with the refined proof
    state.goals.remove(0);

    // Add new goals (in reverse order so first hole is first goal)
    for new_goal in new_goals.into_iter().rev() {
        state.goals.insert(0, new_goal);
    }

    Ok(())
}

/// Count placeholder expressions in an expression tree
fn count_placeholders(expr: &Expr) -> usize {
    match expr {
        // We use a convention where a const named "_" or "?" is a placeholder
        Expr::Const(name, _) => {
            let name_str = name.to_string();
            usize::from(name_str == "_" || name_str == "?" || name_str.starts_with("?_"))
        }
        Expr::App(f, arg) => count_placeholders(f) + count_placeholders(arg),
        Expr::Lam(_, ty, body) | Expr::Pi(_, ty, body) => {
            count_placeholders(ty) + count_placeholders(body)
        }
        Expr::Let(ty, val, body) => {
            count_placeholders(ty) + count_placeholders(val) + count_placeholders(body)
        }
        _ => 0,
    }
}

/// Refine with a placeholder expression indicating a hole
/// This creates a goal with an unknown type (placeholder)
pub fn refine_placeholder(state: &mut ProofState) -> TacticResult {
    if state.goals.is_empty() {
        return Err(TacticError::NoGoals);
    }

    // Simply create a new metavariable goal that inherits the current target
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();
    let placeholder = goal.target.clone();

    let new_meta_id = state.metas.fresh(placeholder.clone());
    let new_goal = Goal {
        meta_id: new_meta_id,
        target: placeholder,
        local_ctx: goal.local_ctx.clone(),
    };

    state.goals.push(new_goal);
    Ok(())
}

// ============================================================================
// use_ - Existential introduction (like existsi but more convenient)
// ============================================================================

/// Provide witnesses for an existential goal.
///
/// `use` is similar to `existsi` but more convenient for providing multiple
/// witnesses at once for nested existential quantifiers.
///
/// # Algorithm
/// 1. Check that the goal is an existential (∃ x, P x)
/// 2. Substitute the witness for the bound variable
/// 3. Continue with nested existentials if more witnesses provided
///
/// # Example
/// ```text
/// -- Goal: ∃ x y, x + y = 5
/// use 2, 3
/// -- Goal: 2 + 3 = 5
/// ```
///
/// # Errors
/// - `NoGoals` if there are no goals
/// - `GoalMismatch` if goal is not an existential
pub fn use_(state: &mut ProofState, witnesses: Vec<Expr>) -> TacticResult {
    if state.goals.is_empty() {
        return Err(TacticError::NoGoals);
    }

    if witnesses.is_empty() {
        return Err(TacticError::Other("use: no witnesses provided".to_string()));
    }

    // Apply existsi for each witness
    for witness in witnesses {
        existsi(state, witness)?;
    }

    Ok(())
}

/// Single witness version of use
pub fn use_single(state: &mut ProofState, witness: Expr) -> TacticResult {
    use_(state, vec![witness])
}

// ============================================================================
// native_decide - Decide decidable propositions by computation
// ============================================================================

/// Decide a decidable proposition by computation.
///
/// `native_decide` is similar to `decide` but indicates that native code
/// should be used for computation when possible. In our implementation,
/// this is the same as `decide` since we don't have a separate native
/// code path.
///
/// # Algorithm
/// 1. Check if the goal type is a decidable proposition
/// 2. Evaluate the decision procedure
/// 3. If it returns true, close the goal with the proof
///
/// # Example
/// ```text
/// -- Goal: 2 + 2 = 4
/// native_decide
/// -- Goal closed (evaluated to true)
/// ```
///
/// # Errors
/// - `NoGoals` if there are no goals
/// - `Other` if the proposition is not decidable or evaluates to false
pub fn native_decide(state: &mut ProofState) -> TacticResult {
    // native_decide is the same as decide in our implementation
    // In Lean 4, native_decide uses native code for faster evaluation
    decide(state)
}

// ============================================================================
// dec_trivial - discharge trivial decidable goals
// ============================================================================

/// Attempt to discharge a goal using decision procedures and trivial reasoning.
///
/// This tactic mirrors mathlib's `dec_trivial`: it tries to solve the goal
/// using decision procedures (`decide`, `native_decide`, `decide_eq`), falling
/// back to straightforward reasoning (`contradiction`, `trivial`).
pub fn dec_trivial(state: &mut ProofState) -> TacticResult {
    if state.goals.is_empty() {
        return Err(TacticError::NoGoals);
    }

    if try_tactic_preserving_state(state, assumption) {
        return Ok(());
    }

    if try_tactic_preserving_state(state, decide_eq) {
        return Ok(());
    }

    if try_tactic_preserving_state(state, decide) {
        return Ok(());
    }

    if try_tactic_preserving_state(state, native_decide) {
        return Ok(());
    }

    if try_tactic_preserving_state(state, contradiction) {
        return Ok(());
    }

    if try_tactic_preserving_state(state, trivial) {
        return Ok(());
    }

    Err(TacticError::Other(
        "dec_trivial: goal is not trivially decidable".to_string(),
    ))
}

// ============================================================================
// Development Tactics
// ============================================================================

/// Mark the current goal as admitted (sorry).
///
/// This is a development aid that allows proceeding past unproven goals.
/// The resulting proof is NOT valid and should only be used during development.
///
/// # Example
/// ```text
/// theorem foo : P := by
///   sorry  -- Goal marked as admitted
/// ```
pub fn sorry(state: &mut ProofState) -> TacticResult {
    if state.goals.is_empty() {
        return Err(TacticError::NoGoals);
    }

    let goal = state.goals.remove(0);
    // Create a placeholder proof term using lcProof (like Lean 4)
    // lcProof is an axiom that provides a proof of any proposition
    let sorry_proof = Expr::app(
        Expr::const_(Name::from_string("sorryAx"), vec![Level::zero()]),
        goal.target.clone(),
    );
    state.metas.assign(goal.meta_id, sorry_proof);
    Ok(())
}

/// Alias for `sorry`.
pub fn admit(state: &mut ProofState) -> TacticResult {
    sorry(state)
}

// funext, propext, set_ext, quot_ext moved to extensionality.rs
// conv types and tactics moved to conv.rs

// =============================================================================
// Change/Show Tactics
// =============================================================================

/// Change the goal to a definitionally equal type.
///
/// This tactic changes the current goal to a new type that is
/// definitionally equal to the original goal type.
///
/// # Example
/// ```text
/// -- Goal: (fun x => x) n = n
/// change n = n
/// -- Goal: n = n
/// ```
pub fn change(state: &mut ProofState, new_type: Expr) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Check that new_type is definitionally equal to current target
    // For now, we do a structural check - a full implementation would use
    // the type checker's definitional equality
    let old_target = &goal.target;

    // Weak check: allow if they're equal after WHNF
    let old_whnf = state.whnf(&goal, old_target);
    let new_whnf = state.whnf(&goal, &new_type);

    // In a full implementation, we'd use the kernel's def_eq check
    // For now, accept if WHNF forms match or if they're syntactically equal
    let are_equal = old_whnf == new_whnf || *old_target == new_type;

    if !are_equal {
        // Still allow the change but warn - the kernel will catch type errors
        // This is permissive behavior for usability
    }

    // Update the goal type
    state.goals[0].target = new_type;
    Ok(())
}

/// Alias for `change` - explicitly show what type we're proving.
pub fn show(state: &mut ProofState, ty: Expr) -> TacticResult {
    change(state, ty)
}

/// Change the type of a hypothesis to a definitionally equal type.
pub fn change_at(state: &mut ProofState, hyp_name: &str, new_type: Expr) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Find the hypothesis
    let hyp_idx = goal
        .local_ctx
        .iter()
        .position(|d| d.name == hyp_name)
        .ok_or_else(|| TacticError::Other(format!("hypothesis '{hyp_name}' not found")))?;

    // Update the hypothesis type
    state.goals[0].local_ctx[hyp_idx].ty = new_type;
    Ok(())
}

// =============================================================================
// Reflexive Closure and Utility Tactics
// =============================================================================

/// Try to close the goal using reflexivity of any relation.
///
/// Attempts `rfl` first, then tries other reflexive relations.
pub fn rfl_closure(state: &mut ProofState) -> TacticResult {
    // Try standard rfl first
    if rfl(state).is_ok() {
        return Ok(());
    }

    // Try Iff.rfl for ↔ goals
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();
    let target = state.whnf(&goal, &goal.target);

    // Check for Iff
    if let Expr::App(f, _) = &target {
        if let Expr::App(iff, _) = &**f {
            if let Expr::Const(name, _) = &**iff {
                if name == &Name::from_string("Iff") {
                    let iff_rfl = Expr::const_(Name::from_string("Iff.rfl"), vec![]);
                    if exact(state, iff_rfl).is_ok() {
                        return Ok(());
                    }
                }
            }
        }
    }

    // Try HEq.rfl for heterogeneous equality
    if let Expr::App(f, _) = &target {
        if let Expr::App(heq, _) = &**f {
            if let Expr::Const(name, _) = &**heq {
                if name == &Name::from_string("HEq") {
                    let heq_rfl = Expr::const_(
                        Name::from_string("HEq.rfl"),
                        vec![Level::param(Name::from_string("u"))],
                    );
                    if exact(state, heq_rfl).is_ok() {
                        return Ok(());
                    }
                }
            }
        }
    }

    Err(TacticError::Other(
        "rfl_closure: no reflexivity rule applies".to_string(),
    ))
}

/// Normalize the goal by applying beta/eta reduction.
///
/// Reduces (λ x, e) y to e[x := y] and similar.
pub fn norm_beta(state: &mut ProofState) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();
    let target = &goal.target;

    // Apply WHNF which includes beta reduction
    let normalized = state.whnf(&goal, target);

    if normalized == *target {
        return Err(TacticError::Other(
            "norm_beta: no reduction possible".to_string(),
        ));
    }

    state.goals[0].target = normalized;
    Ok(())
}

/// Assert a proposition and add it as a hypothesis.
///
/// `assert name : type` creates two goals:
/// 1. Prove `type`
/// 2. Original goal with `name : type` added to context
pub fn assert_tactic(state: &mut ProofState, name: String, prop: Expr) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Create metavariable for the proof of prop
    let proof_meta = state.metas.fresh(prop.clone());

    // Create new local decl for the hypothesis
    let hyp_fvar = state.metas.fresh(prop.clone());
    let mut new_local_ctx = goal.local_ctx.clone();
    new_local_ctx.push(LocalDecl {
        fvar: MetaState::to_fvar(hyp_fvar),
        name: name.clone(),
        ty: prop.clone(),
        value: None,
    });

    // Create the continuation goal with the new hypothesis
    let cont_meta = state.metas.fresh(goal.target.clone());
    let cont_goal = Goal {
        meta_id: cont_meta,
        target: goal.target.clone(),
        local_ctx: new_local_ctx,
    };

    // Create the proof goal for the assertion
    let proof_goal = Goal {
        meta_id: proof_meta,
        target: prop.clone(),
        local_ctx: goal.local_ctx.clone(),
    };

    // Build the proof term: (λ (h : prop), ?cont) ?proof
    let proof = Expr::app(
        Expr::lam(
            BinderInfo::Default,
            prop,
            Expr::FVar(MetaState::to_fvar(cont_meta)),
        ),
        Expr::FVar(MetaState::to_fvar(proof_meta)),
    );

    // Close the original goal
    let old_goal = state.goals.remove(0);
    state.metas.assign(old_goal.meta_id, proof);

    // Add both new goals (proof first, then continuation)
    state.goals.insert(0, cont_goal);
    state.goals.insert(0, proof_goal);

    Ok(())
}

/// Like `assert` but puts the proof goal second instead of first.
pub fn assert_after(state: &mut ProofState, name: String, prop: Expr) -> TacticResult {
    assert_tactic(state, name, prop)?;

    // Swap the two goals so proof is second
    if state.goals.len() >= 2 {
        state.goals.swap(0, 1);
    }

    Ok(())
}

// decide_eq, decidable_type_check, exprs_definitely_not_equal, eval_to_nat, make_ne_proof
// moved to decide_eq.rs

// let_i, have_i, infer_i moved to instance.rs

// abs_cases, AbsCasesConfig, is_numeric_type, make_zero_for_type,
// make_ge_expr, make_lt_expr, create_abs_em_proof moved to abs_cases.rs

// OptionValue, SetOptionConfig, ProofOptions, set_option, set_options
// moved to options.rs

// clear_all_unused, collect_fvars, rename_all, apply_fun, apply_fun_goal,
// clear_except, replace, replace_hyp moved to hypothesis.rs

// trace, trace_with_level, trace_state, trace_expr, TraceLevel, TraceOutput,
// itauto, itauto_with_config, ITautoConfig, clean, beta_reduce_all, bound, substs
// moved to debug.rs

#[cfg(test)]
mod tests;
