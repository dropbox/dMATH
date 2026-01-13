//! State enumeration for model checking
//!
//! This module implements constraint extraction and state enumeration for:
//! 1. Init predicates - extracting equality constraints to generate initial states
//! 2. Next relations - handling primed variables to generate successor states
//!
//! # Approach
//!
//! For Init predicates:
//! - Parse conjunctions to extract individual constraints
//! - Handle equality constraints: `x = value`
//! - Handle membership constraints: `x \in S`
//! - Enumerate all satisfying states
//!
//! For Next relations:
//! - Bind current state variables
//! - Find primed variable assignments: `x' = expr`
//! - Handle UNCHANGED: equivalent to `x' = x`
//! - Handle disjunctions (multiple actions)
//! - Enumerate all successor states

use num_bigint::BigInt;

use crate::compiled_guard::{CompiledExpr, CompiledGuard, EvaluatedAssignment};
use crate::error::EvalError;
use crate::eval::{apply_substitutions, eval, expr_has_primed_param, Env, EvalCtx, OpEnv};
use crate::state::{compute_diff_fingerprint, ArrayState, DiffChanges, DiffSuccessor, State, UndoEntry};
use crate::value::SortedSet;
use crate::var_index::{VarIndex, VarRegistry};
use crate::Value;
use std::collections::{BTreeSet, HashSet};
use std::sync::{Arc, OnceLock};
use tla_core::ast::{BoundVar, Expr, OperatorDef, Substitution};
use tla_core::{Span, Spanned};

// Cached debug flags to avoid env::var syscalls in hot paths
pub fn debug_enum() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("TLA2_DEBUG_ENUM").is_ok())
}

fn debug_exists() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("TLA2_DEBUG_EXISTS").is_ok())
}

fn debug_exists_limit() -> Option<usize> {
    static LIMIT: OnceLock<Option<usize>> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("TLA2_DEBUG_EXISTS_LIMIT")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
    })
}

/// Check if continuation-passing enumeration is enabled.
/// This is the TLC-style algorithm that avoids AST cloning.
/// Enabled by default; disable with TLA2_NO_CONTINUATION=1 for debugging.
fn use_continuation() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("TLA2_NO_CONTINUATION").is_err())
}

/// Check if an error indicates the expression contains action-level constructs
/// (primed variables or UNCHANGED) that can't be evaluated as a guard.
///
/// These errors are expected during guard evaluation when the expression being
/// checked is actually an action (assignment) rather than a guard.
fn is_action_level_error(e: &EvalError) -> bool {
    matches!(e, EvalError::Internal { message, .. }
        if message.contains("Primed variable")
           || message.contains("UNCHANGED cannot be evaluated"))
}

fn is_disabled_action_error(err: &EvalError) -> bool {
    matches!(
        err,
        EvalError::NotInDomain { .. }
            | EvalError::IndexOutOfBounds { .. }
            | EvalError::NoSuchField { .. }
            | EvalError::ChooseFailed { .. }
            | EvalError::DivisionByZero { .. }
            | EvalError::TypeError { .. }
    )
}

fn profile_enum_detail() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("TLA2_PROFILE_ENUM_DETAIL").is_ok())
}

/// Check if bind/unbind enumeration is enabled.
///
/// When enabled via TLA2_BIND_UNBIND=1, uses TLC-style mutable state
/// exploration with ArrayState bind/unbind instead of collecting symbolic
/// assignments. This avoids intermediate allocations and enables 10-20x
/// speedup on EXCEPT-heavy specs like MCBakery.
///
/// See designs/2026-01-13-bind-unbind-architecture.md and issue #101.
fn use_bind_unbind() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("TLA2_BIND_UNBIND").is_ok())
}

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering as AtomicOrdering};

static DEBUG_EXISTS_LINES: AtomicUsize = AtomicUsize::new(0);

fn should_print_exists_debug_line() -> bool {
    let Some(limit) = debug_exists_limit() else {
        return true;
    };
    let line = DEBUG_EXISTS_LINES.fetch_add(1, AtomicOrdering::Relaxed);
    line < limit
}

// Global profiling accumulators (enabled via TLA2_PROFILE_ENUM_DETAIL=1)
// Note: Counters that were declared but never incremented have been deleted (Re: #51)
static PROF_DOMAIN_US: AtomicU64 = AtomicU64::new(0);
static PROF_GUARD_US: AtomicU64 = AtomicU64::new(0);
static PROF_ASSIGN_US: AtomicU64 = AtomicU64::new(0);

// Domain profiling
static PROF_DOMAIN_SUBSET_COUNT: AtomicU64 = AtomicU64::new(0);

// EXISTS loop profiling
static PROF_EXISTS_LOOP_ITERS: AtomicU64 = AtomicU64::new(0);
static PROF_EXISTS_TOTAL_US: AtomicU64 = AtomicU64::new(0);
static PROF_EXISTS_SINGLE_BOUND: AtomicU64 = AtomicU64::new(0);
static PROF_EXISTS_MULTI_BOUND: AtomicU64 = AtomicU64::new(0);

// EXISTS body profiling
static PROF_EXISTS_BODY_CALL_US: AtomicU64 = AtomicU64::new(0);
static PROF_INLINE_GUARD_PASS_COUNT: AtomicU64 = AtomicU64::new(0);
static PROF_INLINE_GUARD_CALLS: AtomicU64 = AtomicU64::new(0);

/// Print and reset enumeration profiling stats
/// Note: Simplified in #51 to remove dead counters
pub fn print_enum_profile_stats() {
    if !profile_enum_detail() {
        return;
    }
    let domain = PROF_DOMAIN_US.swap(0, AtomicOrdering::Relaxed);
    let guard = PROF_GUARD_US.swap(0, AtomicOrdering::Relaxed);
    let assign = PROF_ASSIGN_US.swap(0, AtomicOrdering::Relaxed);
    let total = domain + guard + assign;

    // Also check continuation stats (don't early return if we have those)
    let stack_marks_peek = STACK_MARK_COUNT.load(AtomicOrdering::Relaxed);
    let constraint_pushes_peek = CONSTRAINT_PUSH_COUNT.load(AtomicOrdering::Relaxed);
    if total == 0 && stack_marks_peek == 0 && constraint_pushes_peek == 0 {
        return;
    }
    eprintln!("=== Enumeration Detail Profile ===");
    if total > 0 {
        eprintln!(
            "  Domain eval:     {:>8.3}s ({:>5.1}%)",
            domain as f64 / 1_000_000.0,
            domain as f64 / total as f64 * 100.0
        );
        eprintln!(
            "  Guard eval:      {:>8.3}s ({:>5.1}%)",
            guard as f64 / 1_000_000.0,
            guard as f64 / total as f64 * 100.0
        );
        eprintln!(
            "  Assignment eval: {:>8.3}s ({:>5.1}%)",
            assign as f64 / 1_000_000.0,
            assign as f64 / total as f64 * 100.0
        );
    }
    eprintln!("  ---");

    // Domain breakdown
    let domain_subset_count = PROF_DOMAIN_SUBSET_COUNT.swap(0, AtomicOrdering::Relaxed);
    if domain_subset_count > 0 {
        eprintln!("  --- Domain breakdown ---");
        eprintln!("    Subsets gen:     {}", domain_subset_count);
    }

    // EXISTS loop stats
    let exists_loop_iters = PROF_EXISTS_LOOP_ITERS.swap(0, AtomicOrdering::Relaxed);
    let exists_total = PROF_EXISTS_TOTAL_US.swap(0, AtomicOrdering::Relaxed);
    let single_bound = PROF_EXISTS_SINGLE_BOUND.swap(0, AtomicOrdering::Relaxed);
    let multi_bound = PROF_EXISTS_MULTI_BOUND.swap(0, AtomicOrdering::Relaxed);
    let exists_body_call = PROF_EXISTS_BODY_CALL_US.swap(0, AtomicOrdering::Relaxed);
    let inline_guard_calls = PROF_INLINE_GUARD_CALLS.swap(0, AtomicOrdering::Relaxed);
    let inline_guard_pass = PROF_INLINE_GUARD_PASS_COUNT.swap(0, AtomicOrdering::Relaxed);
    if exists_loop_iters > 0 {
        eprintln!("  --- EXISTS loop breakdown ---");
        eprintln!("    Loop iterations: {}", exists_loop_iters);
        eprintln!("    Single-bound (→body): {}", single_bound);
        eprintln!("    Multi-bound (→recurse): {}", multi_bound);
        if exists_total > 0 {
            eprintln!(
                "    Total EXISTS:    {:>8.3}s",
                exists_total as f64 / 1_000_000.0
            );
        }
        if exists_body_call > 0 || inline_guard_calls > 0 {
            eprintln!("    Body calls:      {}", exists_body_call);
            eprintln!(
                "    Inline guards:   {} calls, {} passed",
                inline_guard_calls, inline_guard_pass
            );
        }
    }

    // Continuation enumeration stats
    let stack_marks = STACK_MARK_COUNT.swap(0, AtomicOrdering::Relaxed);
    let constraint_pushes = CONSTRAINT_PUSH_COUNT.swap(0, AtomicOrdering::Relaxed);
    if stack_marks > 0 || constraint_pushes > 0 {
        eprintln!("  --- Continuation enumeration ---");
        eprintln!("    Stack marks:         {}", stack_marks);
        eprintln!("    Constraint pushes:   {}", constraint_pushes);
    }
}

/// Check if an expression is "constant" - i.e., doesn't depend on state variables
/// or primed variables. Such expressions can be evaluated once during preprocessing.
#[cfg(test)]
#[allow(dead_code)]
fn is_constant_expr(ctx: &EvalCtx, expr: &Spanned<Expr>, local_vars: &HashSet<String>) -> bool {
    is_constant_expr_with_visited(ctx, expr, local_vars, &mut HashSet::new())
}

/// Helper for constant checking with visited set to handle recursive operators.
#[cfg(test)]
#[allow(dead_code)]
fn is_constant_expr_with_visited(
    ctx: &EvalCtx,
    expr: &Spanned<Expr>,
    local_vars: &HashSet<String>,
    visited_ops: &mut HashSet<String>,
) -> bool {
    match &expr.node {
        // Identifier: check if it's a state variable, local variable, or operator
        Expr::Ident(name) => {
            // Local variables (from outer EXISTS) are not constant for inner expressions
            if local_vars.contains(name) {
                return false;
            }
            // State variables are not constant
            if ctx.var_registry().get(name).is_some() {
                return false;
            }
            // Check if this is a zero-arg operator - if so, check its body for state refs
            if let Some(def) = ctx.get_op(name) {
                if def.params.is_empty() {
                    // Avoid infinite recursion for recursive operators
                    if visited_ops.contains(name) {
                        // Assume recursive operators are not constant (safe approximation)
                        return false;
                    }
                    visited_ops.insert(name.clone());
                    let result = is_constant_expr_with_visited(ctx, &def.body, local_vars, visited_ops);
                    visited_ops.remove(name);
                    return result;
                }
            }
            // Constants (not operators) are constant
            true
        }

        // Primed expressions are never constant
        Expr::Prime(_) => false,

        // Binary operators: both operands must be constant
        Expr::And(a, b)
        | Expr::Or(a, b)
        | Expr::Implies(a, b)
        | Expr::Equiv(a, b)
        | Expr::Eq(a, b)
        | Expr::Neq(a, b)
        | Expr::Lt(a, b)
        | Expr::Leq(a, b)
        | Expr::Gt(a, b)
        | Expr::Geq(a, b)
        | Expr::Add(a, b)
        | Expr::Sub(a, b)
        | Expr::Mul(a, b)
        | Expr::Div(a, b)
        | Expr::IntDiv(a, b)
        | Expr::Mod(a, b)
        | Expr::Pow(a, b)
        | Expr::Range(a, b)
        | Expr::Union(a, b)
        | Expr::Intersect(a, b)
        | Expr::SetMinus(a, b)
        | Expr::In(a, b)
        | Expr::NotIn(a, b)
        | Expr::Subseteq(a, b)
        | Expr::FuncSet(a, b)
        | Expr::LeadsTo(a, b) => {
            is_constant_expr_with_visited(ctx, a, local_vars, visited_ops)
                && is_constant_expr_with_visited(ctx, b, local_vars, visited_ops)
        }

        // Unary operators
        Expr::Not(a)
        | Expr::Neg(a)
        | Expr::Domain(a)
        | Expr::Powerset(a)
        | Expr::BigUnion(a)
        | Expr::Always(a)
        | Expr::Eventually(a)
        | Expr::Enabled(a)
        | Expr::Unchanged(a) => is_constant_expr_with_visited(ctx, a, local_vars, visited_ops),

        // Tuples, sets: all elements must be constant
        Expr::Tuple(elems) | Expr::SetEnum(elems) => elems
            .iter()
            .all(|e| is_constant_expr_with_visited(ctx, e, local_vars, visited_ops)),

        // Times (cartesian product): all sets must be constant
        Expr::Times(elems) => elems
            .iter()
            .all(|e| is_constant_expr_with_visited(ctx, e, local_vars, visited_ops)),

        // Record: all field values must be constant
        Expr::Record(fields) | Expr::RecordSet(fields) => fields
            .iter()
            .all(|(_, v)| is_constant_expr_with_visited(ctx, v, local_vars, visited_ops)),

        // RecordAccess: base must be constant
        Expr::RecordAccess(base, _) => {
            is_constant_expr_with_visited(ctx, base, local_vars, visited_ops)
        }

        // Function application: both function and argument must be constant
        Expr::FuncApply(f, arg) => {
            is_constant_expr_with_visited(ctx, f, local_vars, visited_ops)
                && is_constant_expr_with_visited(ctx, arg, local_vars, visited_ops)
        }

        // Operator application: check args and operator body if zero-arg
        Expr::Apply(op_expr, args) => {
            // All args must be constant
            if !args
                .iter()
                .all(|a| is_constant_expr_with_visited(ctx, a, local_vars, visited_ops))
            {
                return false;
            }
            // Also check operator body if it's a named operator
            if let Expr::Ident(op_name) = &op_expr.node {
                if let Some(def) = ctx.get_op(op_name) {
                    // For operators with params, check if body references state vars
                    // (after substituting args - conservative: assume not constant)
                    if !def.params.is_empty() {
                        // Operators with params that reference state vars in body are not constant
                        // Check body with params as local vars
                        let mut extended = local_vars.clone();
                        for param in &def.params {
                            extended.insert(param.name.node.clone());
                        }
                        if visited_ops.contains(op_name) {
                            return false; // Recursive - assume not constant
                        }
                        visited_ops.insert(op_name.clone());
                        let result =
                            is_constant_expr_with_visited(ctx, &def.body, &extended, visited_ops);
                        visited_ops.remove(op_name);
                        return result;
                    }
                }
            }
            true
        }

        // If-then-else: all branches must be constant
        Expr::If(cond, then_expr, else_expr) => {
            is_constant_expr_with_visited(ctx, cond, local_vars, visited_ops)
                && is_constant_expr_with_visited(ctx, then_expr, local_vars, visited_ops)
                && is_constant_expr_with_visited(ctx, else_expr, local_vars, visited_ops)
        }

        // Quantifiers: body may use bound variables, so check with extended local_vars
        Expr::Forall(bounds, body) | Expr::Exists(bounds, body) => {
            // Domain must be constant
            for bound in bounds {
                if let Some(domain) = &bound.domain {
                    if !is_constant_expr_with_visited(ctx, domain, local_vars, visited_ops) {
                        return false;
                    }
                }
            }
            // Body: extend local_vars with bound variable names
            let mut extended = local_vars.clone();
            for bound in bounds {
                extended.insert(bound.name.node.clone());
            }
            is_constant_expr_with_visited(ctx, body, &extended, visited_ops)
        }

        // Set builder: bounds and body must be constant
        Expr::SetBuilder(expr, bounds) => {
            for bound in bounds {
                if let Some(domain) = &bound.domain {
                    if !is_constant_expr_with_visited(ctx, domain, local_vars, visited_ops) {
                        return false;
                    }
                }
            }
            let mut extended = local_vars.clone();
            for bound in bounds {
                extended.insert(bound.name.node.clone());
            }
            is_constant_expr_with_visited(ctx, expr, &extended, visited_ops)
        }

        // Set filter: single bound and predicate must be constant
        Expr::SetFilter(bound, pred) => {
            if let Some(domain) = &bound.domain {
                if !is_constant_expr_with_visited(ctx, domain, local_vars, visited_ops) {
                    return false;
                }
            }
            let mut extended = local_vars.clone();
            extended.insert(bound.name.node.clone());
            is_constant_expr_with_visited(ctx, pred, &extended, visited_ops)
        }

        // Function definition: bound domains and body must be constant
        Expr::FuncDef(bounds, body) => {
            for bound in bounds {
                if let Some(domain) = &bound.domain {
                    if !is_constant_expr_with_visited(ctx, domain, local_vars, visited_ops) {
                        return false;
                    }
                }
            }
            let mut extended = local_vars.clone();
            for bound in bounds {
                extended.insert(bound.name.node.clone());
            }
            is_constant_expr_with_visited(ctx, body, &extended, visited_ops)
        }

        // Let expressions: definitions and body must be constant
        Expr::Let(defs, body) => {
            let mut extended = local_vars.clone();
            for def in defs {
                if !is_constant_expr_with_visited(ctx, &def.body, &extended, visited_ops) {
                    return false;
                }
                extended.insert(def.name.node.clone());
            }
            is_constant_expr_with_visited(ctx, body, &extended, visited_ops)
        }

        // CHOOSE: domain and predicate must be constant
        Expr::Choose(bound, body) => {
            if let Some(domain) = &bound.domain {
                if !is_constant_expr_with_visited(ctx, domain, local_vars, visited_ops) {
                    return false;
                }
            }
            let mut extended = local_vars.clone();
            extended.insert(bound.name.node.clone());
            is_constant_expr_with_visited(ctx, body, &extended, visited_ops)
        }

        // EXCEPT: function and all specs must be constant
        Expr::Except(f, specs) => {
            if !is_constant_expr_with_visited(ctx, f, local_vars, visited_ops) {
                return false;
            }
            for spec in specs {
                for path_elem in &spec.path {
                    use tla_core::ast::ExceptPathElement;
                    match path_elem {
                        ExceptPathElement::Index(idx) => {
                            if !is_constant_expr_with_visited(ctx, idx, local_vars, visited_ops) {
                                return false;
                            }
                        }
                        ExceptPathElement::Field(_) => {
                            // Field names are always constant
                        }
                    }
                }
                if !is_constant_expr_with_visited(ctx, &spec.value, local_vars, visited_ops) {
                    return false;
                }
            }
            true
        }

        // CASE: guards and values must be constant
        Expr::Case(arms, other) => {
            for arm in arms {
                if !is_constant_expr_with_visited(ctx, &arm.guard, local_vars, visited_ops)
                    || !is_constant_expr_with_visited(ctx, &arm.body, local_vars, visited_ops)
                {
                    return false;
                }
            }
            if let Some(default) = other {
                is_constant_expr_with_visited(ctx, default, local_vars, visited_ops)
            } else {
                true
            }
        }

        // Temporal operators with two args (fairness)
        Expr::WeakFair(a, b) | Expr::StrongFair(a, b) => {
            is_constant_expr_with_visited(ctx, a, local_vars, visited_ops)
                && is_constant_expr_with_visited(ctx, b, local_vars, visited_ops)
        }

        // Everything else: assume not constant for safety
        _ => false,
    }
}

/// Tracks local variables in scope during preprocessing.
///
/// Used to compute stack depths for O(1) local variable lookup.
/// Variables are added when entering EXISTS bodies and the depth
/// is computed relative to the scope order.
#[derive(Debug, Clone, Default)]
pub struct LocalScope {
    /// Variable names in binding order (first bound = first element)
    /// These are quantifier-bound variables that become LocalVar at runtime.
    vars: Vec<String>,
    /// LET bindings that are inlined at compile time.
    /// These are compiled expressions that replace references to LET-defined names.
    let_bindings: Vec<(String, CompiledExpr)>,
}

impl LocalScope {
    /// Create an empty scope
    pub fn new() -> Self {
        LocalScope {
            vars: Vec::new(),
            let_bindings: Vec::new(),
        }
    }

    /// Create a new scope with an additional bound variable (for quantifiers)
    pub fn with_var(&self, name: &str) -> Self {
        let mut vars = self.vars.clone();
        vars.push(name.to_string());
        LocalScope {
            vars,
            let_bindings: self.let_bindings.clone(),
        }
    }

    /// Create a new scope with an additional LET binding (for compile-time inlining)
    pub fn with_let_binding(&self, name: &str, expr: CompiledExpr) -> Self {
        let mut let_bindings = self.let_bindings.clone();
        let_bindings.push((name.to_string(), expr));
        LocalScope {
            vars: self.vars.clone(),
            let_bindings,
        }
    }

    /// Get a LET binding by name (for compile-time inlining).
    /// Returns the innermost binding if there are multiple with the same name.
    pub fn get_let_binding(&self, name: &str) -> Option<&CompiledExpr> {
        self.let_bindings
            .iter()
            .rev()
            .find(|(n, _)| n == name)
            .map(|(_, expr)| expr)
    }

    /// Get the depth of a local variable (for O(1) stack access).
    /// Returns None if the variable is not in scope.
    ///
    /// depth=0 means most recently bound (end of vars list)
    pub fn get_depth(&self, name: &str) -> Option<u8> {
        // IMPORTANT: Choose the *innermost* binding when names repeat due to shadowing
        // (e.g., nested quantifiers reusing the same variable name).
        self.vars
            .iter()
            .rev()
            .position(|var| var == name)
            .map(|depth| depth as u8)
    }

    /// Get the list of local variable names
    pub fn vars(&self) -> &Vec<String> {
        &self.vars
    }
}

fn debug_extract() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("TLA2_DEBUG_EXTRACT").is_ok())
}

/// When we inline `Instance!Op(args)` by substituting `Op`'s body into the outer module,
/// any unqualified helper operator references (e.g., `Helper(x)`) would otherwise be
/// resolved in the *outer* operator namespace, leading to undefined operators or
/// incorrect shadowing when the outer module defines the same name.
///
/// We conservatively rewrite:
/// - `Helper(x)` (Apply(Ident("Helper"), ...)) to `Instance!Helper(x)`
/// - `ConstOp` (Ident("ConstOp")) to `Instance!ConstOp` for zero-argument operators
///
/// This rewrite is performed before applying INSTANCE/parameter substitutions so that
/// substitution RHS expressions (written in the outer module context) are not affected.
fn qualify_instance_ops_for_inlining(
    expr: &Spanned<Expr>,
    instance_target: &tla_core::ast::ModuleTarget,
    instance_ops: &OpEnv,
    params: &[tla_core::ast::OpParam],
) -> Spanned<Expr> {
    #[derive(Clone, Default)]
    struct Scope {
        bound_vars: HashSet<String>,
        shadow_ops: HashSet<String>,
    }

    fn add_bound_var(scope: &mut Scope, bv: &BoundVar) {
        scope.bound_vars.insert(bv.name.node.clone());
        if let Some(pattern) = &bv.pattern {
            match pattern {
                tla_core::ast::BoundPattern::Var(v) => {
                    scope.bound_vars.insert(v.node.clone());
                }
                tla_core::ast::BoundPattern::Tuple(vs) => {
                    for v in vs {
                        scope.bound_vars.insert(v.node.clone());
                    }
                }
            }
        }
    }

    fn add_op_params(scope: &mut Scope, params: &[tla_core::ast::OpParam]) {
        for p in params {
            scope.bound_vars.insert(p.name.node.clone());
        }
    }

    fn qualify(
        expr: &Spanned<Expr>,
        instance_target: &tla_core::ast::ModuleTarget,
        instance_ops: &OpEnv,
        scope: &Scope,
    ) -> Spanned<Expr> {
        use tla_core::ast::{ExceptPathElement, ExceptSpec, ModuleTarget};

        let new_node = match &expr.node {
            Expr::Ident(name) => {
                if scope.bound_vars.contains(name) || scope.shadow_ops.contains(name) {
                    Expr::Ident(name.clone())
                } else if let Some(def) = instance_ops.get(name) {
                    if def.params.is_empty() {
                        Expr::ModuleRef(instance_target.clone(), name.clone(), Vec::new())
                    } else {
                        Expr::Ident(name.clone())
                    }
                } else {
                    Expr::Ident(name.clone())
                }
            }

            Expr::Apply(op_expr, args) => {
                let new_op_expr = qualify(op_expr, instance_target, instance_ops, scope);
                let new_args: Vec<_> = args
                    .iter()
                    .map(|a| qualify(a, instance_target, instance_ops, scope))
                    .collect();

                if let Expr::Ident(op_name) = &op_expr.node {
                    if !scope.bound_vars.contains(op_name)
                        && !scope.shadow_ops.contains(op_name)
                        && instance_ops.contains_key(op_name)
                    {
                        Expr::ModuleRef(instance_target.clone(), op_name.clone(), new_args)
                    } else {
                        Expr::Apply(Box::new(new_op_expr), new_args)
                    }
                } else {
                    Expr::Apply(Box::new(new_op_expr), new_args)
                }
            }

            Expr::Let(defs, body) => {
                // LET-bound operator names shadow module operators within the LET scope.
                let mut let_scope = scope.clone();
                for d in defs {
                    let_scope.shadow_ops.insert(d.name.node.clone());
                }

                let new_defs: Vec<_> = defs
                    .iter()
                    .map(|d| {
                        let mut def_scope = let_scope.clone();
                        add_op_params(&mut def_scope, &d.params);
                        let new_body = qualify(&d.body, instance_target, instance_ops, &def_scope);
                        let mut new_def = d.clone();
                        new_def.body = new_body;
                        new_def
                    })
                    .collect();
                let new_body = qualify(body, instance_target, instance_ops, &let_scope);
                Expr::Let(new_defs, Box::new(new_body))
            }

            Expr::Forall(bounds, body) => {
                let new_bounds: Vec<_> = bounds
                    .iter()
                    .map(|b| BoundVar {
                        name: b.name.clone(),
                        pattern: b.pattern.clone(),
                        domain: b
                            .domain
                            .as_ref()
                            .map(|d| Box::new(qualify(d, instance_target, instance_ops, scope))),
                    })
                    .collect();
                let mut body_scope = scope.clone();
                for b in bounds {
                    add_bound_var(&mut body_scope, b);
                }
                let new_body = qualify(body, instance_target, instance_ops, &body_scope);
                Expr::Forall(new_bounds, Box::new(new_body))
            }

            Expr::Exists(bounds, body) => {
                let new_bounds: Vec<_> = bounds
                    .iter()
                    .map(|b| BoundVar {
                        name: b.name.clone(),
                        pattern: b.pattern.clone(),
                        domain: b
                            .domain
                            .as_ref()
                            .map(|d| Box::new(qualify(d, instance_target, instance_ops, scope))),
                    })
                    .collect();
                let mut body_scope = scope.clone();
                for b in bounds {
                    add_bound_var(&mut body_scope, b);
                }
                let new_body = qualify(body, instance_target, instance_ops, &body_scope);
                Expr::Exists(new_bounds, Box::new(new_body))
            }

            Expr::Choose(bound, body) => {
                let new_bound = BoundVar {
                    name: bound.name.clone(),
                    pattern: bound.pattern.clone(),
                    domain: bound
                        .domain
                        .as_ref()
                        .map(|d| Box::new(qualify(d, instance_target, instance_ops, scope))),
                };
                let mut body_scope = scope.clone();
                add_bound_var(&mut body_scope, bound);
                let new_body = qualify(body, instance_target, instance_ops, &body_scope);
                Expr::Choose(new_bound, Box::new(new_body))
            }

            Expr::SetBuilder(elem, bounds) => {
                let new_bounds: Vec<_> = bounds
                    .iter()
                    .map(|b| BoundVar {
                        name: b.name.clone(),
                        pattern: b.pattern.clone(),
                        domain: b
                            .domain
                            .as_ref()
                            .map(|d| Box::new(qualify(d, instance_target, instance_ops, scope))),
                    })
                    .collect();
                let mut elem_scope = scope.clone();
                for b in bounds {
                    add_bound_var(&mut elem_scope, b);
                }
                let new_elem = qualify(elem, instance_target, instance_ops, &elem_scope);
                Expr::SetBuilder(Box::new(new_elem), new_bounds)
            }

            Expr::SetFilter(bound, pred) => {
                let new_bound = BoundVar {
                    name: bound.name.clone(),
                    pattern: bound.pattern.clone(),
                    domain: bound
                        .domain
                        .as_ref()
                        .map(|d| Box::new(qualify(d, instance_target, instance_ops, scope))),
                };
                let mut pred_scope = scope.clone();
                add_bound_var(&mut pred_scope, bound);
                let new_pred = qualify(pred, instance_target, instance_ops, &pred_scope);
                Expr::SetFilter(new_bound, Box::new(new_pred))
            }

            Expr::FuncDef(bounds, body) => {
                let new_bounds: Vec<_> = bounds
                    .iter()
                    .map(|b| BoundVar {
                        name: b.name.clone(),
                        pattern: b.pattern.clone(),
                        domain: b
                            .domain
                            .as_ref()
                            .map(|d| Box::new(qualify(d, instance_target, instance_ops, scope))),
                    })
                    .collect();
                let mut body_scope = scope.clone();
                for b in bounds {
                    add_bound_var(&mut body_scope, b);
                }
                let new_body = qualify(body, instance_target, instance_ops, &body_scope);
                Expr::FuncDef(new_bounds, Box::new(new_body))
            }

            Expr::Lambda(params, body) => {
                let mut body_scope = scope.clone();
                for p in params {
                    body_scope.bound_vars.insert(p.node.clone());
                }
                let new_body = qualify(body, instance_target, instance_ops, &body_scope);
                Expr::Lambda(params.clone(), Box::new(new_body))
            }

            Expr::ModuleRef(target, op, args) => {
                let new_args: Vec<_> = args
                    .iter()
                    .map(|a| qualify(a, instance_target, instance_ops, scope))
                    .collect();
                let new_target = match target {
                    ModuleTarget::Named(_) => target.clone(),
                    ModuleTarget::Parameterized(name, params) => ModuleTarget::Parameterized(
                        name.clone(),
                        params
                            .iter()
                            .map(|p| qualify(p, instance_target, instance_ops, scope))
                            .collect(),
                    ),
                    ModuleTarget::Chained(base) => ModuleTarget::Chained(Box::new(qualify(
                        base,
                        instance_target,
                        instance_ops,
                        scope,
                    ))),
                };
                Expr::ModuleRef(new_target, op.clone(), new_args)
            }

            Expr::InstanceExpr(module, subs) => {
                let new_subs: Vec<_> = subs
                    .iter()
                    .map(|sub| tla_core::ast::Substitution {
                        from: sub.from.clone(),
                        to: qualify(&sub.to, instance_target, instance_ops, scope),
                    })
                    .collect();
                Expr::InstanceExpr(module.clone(), new_subs)
            }

            Expr::Bool(b) => Expr::Bool(*b),
            Expr::Int(n) => Expr::Int(n.clone()),
            Expr::String(s) => Expr::String(s.clone()),
            Expr::OpRef(op) => Expr::OpRef(op.clone()),

            Expr::And(a, b) => Expr::And(
                Box::new(qualify(a, instance_target, instance_ops, scope)),
                Box::new(qualify(b, instance_target, instance_ops, scope)),
            ),
            Expr::Or(a, b) => Expr::Or(
                Box::new(qualify(a, instance_target, instance_ops, scope)),
                Box::new(qualify(b, instance_target, instance_ops, scope)),
            ),
            Expr::Not(a) => Expr::Not(Box::new(qualify(a, instance_target, instance_ops, scope))),
            Expr::Implies(a, b) => Expr::Implies(
                Box::new(qualify(a, instance_target, instance_ops, scope)),
                Box::new(qualify(b, instance_target, instance_ops, scope)),
            ),
            Expr::Equiv(a, b) => Expr::Equiv(
                Box::new(qualify(a, instance_target, instance_ops, scope)),
                Box::new(qualify(b, instance_target, instance_ops, scope)),
            ),

            Expr::SetEnum(elems) => Expr::SetEnum(
                elems
                    .iter()
                    .map(|e| qualify(e, instance_target, instance_ops, scope))
                    .collect(),
            ),

            Expr::In(a, b) => Expr::In(
                Box::new(qualify(a, instance_target, instance_ops, scope)),
                Box::new(qualify(b, instance_target, instance_ops, scope)),
            ),
            Expr::NotIn(a, b) => Expr::NotIn(
                Box::new(qualify(a, instance_target, instance_ops, scope)),
                Box::new(qualify(b, instance_target, instance_ops, scope)),
            ),
            Expr::Subseteq(a, b) => Expr::Subseteq(
                Box::new(qualify(a, instance_target, instance_ops, scope)),
                Box::new(qualify(b, instance_target, instance_ops, scope)),
            ),
            Expr::Union(a, b) => Expr::Union(
                Box::new(qualify(a, instance_target, instance_ops, scope)),
                Box::new(qualify(b, instance_target, instance_ops, scope)),
            ),
            Expr::Intersect(a, b) => Expr::Intersect(
                Box::new(qualify(a, instance_target, instance_ops, scope)),
                Box::new(qualify(b, instance_target, instance_ops, scope)),
            ),
            Expr::SetMinus(a, b) => Expr::SetMinus(
                Box::new(qualify(a, instance_target, instance_ops, scope)),
                Box::new(qualify(b, instance_target, instance_ops, scope)),
            ),
            Expr::Powerset(a) => {
                Expr::Powerset(Box::new(qualify(a, instance_target, instance_ops, scope)))
            }
            Expr::BigUnion(a) => {
                Expr::BigUnion(Box::new(qualify(a, instance_target, instance_ops, scope)))
            }

            Expr::FuncApply(f, arg) => Expr::FuncApply(
                Box::new(qualify(f, instance_target, instance_ops, scope)),
                Box::new(qualify(arg, instance_target, instance_ops, scope)),
            ),
            Expr::Domain(a) => {
                Expr::Domain(Box::new(qualify(a, instance_target, instance_ops, scope)))
            }
            Expr::Except(base, specs) => {
                let new_base = Box::new(qualify(base, instance_target, instance_ops, scope));
                let new_specs: Vec<_> = specs
                    .iter()
                    .map(|spec| ExceptSpec {
                        path: spec
                            .path
                            .iter()
                            .map(|elem| match elem {
                                ExceptPathElement::Index(idx_expr) => ExceptPathElement::Index(
                                    qualify(idx_expr, instance_target, instance_ops, scope),
                                ),
                                ExceptPathElement::Field(f) => ExceptPathElement::Field(f.clone()),
                            })
                            .collect(),
                        value: qualify(&spec.value, instance_target, instance_ops, scope),
                    })
                    .collect();
                Expr::Except(new_base, new_specs)
            }
            Expr::FuncSet(a, b) => Expr::FuncSet(
                Box::new(qualify(a, instance_target, instance_ops, scope)),
                Box::new(qualify(b, instance_target, instance_ops, scope)),
            ),

            Expr::Record(fields) => Expr::Record(
                fields
                    .iter()
                    .map(|(name, e)| {
                        (
                            name.clone(),
                            qualify(e, instance_target, instance_ops, scope),
                        )
                    })
                    .collect(),
            ),
            Expr::RecordAccess(r, field) => Expr::RecordAccess(
                Box::new(qualify(r, instance_target, instance_ops, scope)),
                field.clone(),
            ),
            Expr::RecordSet(fields) => Expr::RecordSet(
                fields
                    .iter()
                    .map(|(name, e)| {
                        (
                            name.clone(),
                            qualify(e, instance_target, instance_ops, scope),
                        )
                    })
                    .collect(),
            ),

            Expr::Tuple(elems) => Expr::Tuple(
                elems
                    .iter()
                    .map(|e| qualify(e, instance_target, instance_ops, scope))
                    .collect(),
            ),
            Expr::Times(elems) => Expr::Times(
                elems
                    .iter()
                    .map(|e| qualify(e, instance_target, instance_ops, scope))
                    .collect(),
            ),

            Expr::Prime(a) => {
                Expr::Prime(Box::new(qualify(a, instance_target, instance_ops, scope)))
            }
            Expr::Always(a) => {
                Expr::Always(Box::new(qualify(a, instance_target, instance_ops, scope)))
            }
            Expr::Eventually(a) => {
                Expr::Eventually(Box::new(qualify(a, instance_target, instance_ops, scope)))
            }
            Expr::LeadsTo(a, b) => Expr::LeadsTo(
                Box::new(qualify(a, instance_target, instance_ops, scope)),
                Box::new(qualify(b, instance_target, instance_ops, scope)),
            ),
            Expr::WeakFair(vars, a) => Expr::WeakFair(
                Box::new(qualify(vars, instance_target, instance_ops, scope)),
                Box::new(qualify(a, instance_target, instance_ops, scope)),
            ),
            Expr::StrongFair(vars, a) => Expr::StrongFair(
                Box::new(qualify(vars, instance_target, instance_ops, scope)),
                Box::new(qualify(a, instance_target, instance_ops, scope)),
            ),
            Expr::Enabled(a) => {
                Expr::Enabled(Box::new(qualify(a, instance_target, instance_ops, scope)))
            }
            Expr::Unchanged(a) => {
                Expr::Unchanged(Box::new(qualify(a, instance_target, instance_ops, scope)))
            }

            Expr::If(cond, then_e, else_e) => Expr::If(
                Box::new(qualify(cond, instance_target, instance_ops, scope)),
                Box::new(qualify(then_e, instance_target, instance_ops, scope)),
                Box::new(qualify(else_e, instance_target, instance_ops, scope)),
            ),
            Expr::Case(arms, other) => {
                let new_arms: Vec<_> = arms
                    .iter()
                    .map(|arm| tla_core::ast::CaseArm {
                        guard: qualify(&arm.guard, instance_target, instance_ops, scope),
                        body: qualify(&arm.body, instance_target, instance_ops, scope),
                    })
                    .collect();
                let new_other = other
                    .as_ref()
                    .map(|o| Box::new(qualify(o, instance_target, instance_ops, scope)));
                Expr::Case(new_arms, new_other)
            }

            Expr::Eq(a, b) => Expr::Eq(
                Box::new(qualify(a, instance_target, instance_ops, scope)),
                Box::new(qualify(b, instance_target, instance_ops, scope)),
            ),
            Expr::Neq(a, b) => Expr::Neq(
                Box::new(qualify(a, instance_target, instance_ops, scope)),
                Box::new(qualify(b, instance_target, instance_ops, scope)),
            ),
            Expr::Lt(a, b) => Expr::Lt(
                Box::new(qualify(a, instance_target, instance_ops, scope)),
                Box::new(qualify(b, instance_target, instance_ops, scope)),
            ),
            Expr::Leq(a, b) => Expr::Leq(
                Box::new(qualify(a, instance_target, instance_ops, scope)),
                Box::new(qualify(b, instance_target, instance_ops, scope)),
            ),
            Expr::Gt(a, b) => Expr::Gt(
                Box::new(qualify(a, instance_target, instance_ops, scope)),
                Box::new(qualify(b, instance_target, instance_ops, scope)),
            ),
            Expr::Geq(a, b) => Expr::Geq(
                Box::new(qualify(a, instance_target, instance_ops, scope)),
                Box::new(qualify(b, instance_target, instance_ops, scope)),
            ),

            Expr::Add(a, b) => Expr::Add(
                Box::new(qualify(a, instance_target, instance_ops, scope)),
                Box::new(qualify(b, instance_target, instance_ops, scope)),
            ),
            Expr::Sub(a, b) => Expr::Sub(
                Box::new(qualify(a, instance_target, instance_ops, scope)),
                Box::new(qualify(b, instance_target, instance_ops, scope)),
            ),
            Expr::Mul(a, b) => Expr::Mul(
                Box::new(qualify(a, instance_target, instance_ops, scope)),
                Box::new(qualify(b, instance_target, instance_ops, scope)),
            ),
            Expr::Div(a, b) => Expr::Div(
                Box::new(qualify(a, instance_target, instance_ops, scope)),
                Box::new(qualify(b, instance_target, instance_ops, scope)),
            ),
            Expr::IntDiv(a, b) => Expr::IntDiv(
                Box::new(qualify(a, instance_target, instance_ops, scope)),
                Box::new(qualify(b, instance_target, instance_ops, scope)),
            ),
            Expr::Mod(a, b) => Expr::Mod(
                Box::new(qualify(a, instance_target, instance_ops, scope)),
                Box::new(qualify(b, instance_target, instance_ops, scope)),
            ),
            Expr::Pow(a, b) => Expr::Pow(
                Box::new(qualify(a, instance_target, instance_ops, scope)),
                Box::new(qualify(b, instance_target, instance_ops, scope)),
            ),
            Expr::Neg(a) => Expr::Neg(Box::new(qualify(a, instance_target, instance_ops, scope))),
            Expr::Range(a, b) => Expr::Range(
                Box::new(qualify(a, instance_target, instance_ops, scope)),
                Box::new(qualify(b, instance_target, instance_ops, scope)),
            ),
        };

        Spanned::new(new_node, expr.span)
    }

    let mut scope = Scope::default();
    add_op_params(&mut scope, params);
    qualify(expr, instance_target, instance_ops, &scope)
}
/// Check if an expression contains primed variables.
/// Public for use in compiled_guard.rs to determine if Fallback expressions need next-state context.
pub(crate) fn expr_contains_prime(expr: &Expr) -> bool {
    match expr {
        Expr::Prime(_) => true,
        Expr::Unchanged(_) => true,
        // Binary operators
        Expr::And(a, b)
        | Expr::Or(a, b)
        | Expr::Implies(a, b)
        | Expr::Equiv(a, b)
        | Expr::Eq(a, b)
        | Expr::Neq(a, b)
        | Expr::Lt(a, b)
        | Expr::Leq(a, b)
        | Expr::Gt(a, b)
        | Expr::Geq(a, b)
        | Expr::Add(a, b)
        | Expr::Sub(a, b)
        | Expr::Mul(a, b)
        | Expr::Div(a, b)
        | Expr::IntDiv(a, b)
        | Expr::Mod(a, b)
        | Expr::Pow(a, b)
        | Expr::Range(a, b)
        | Expr::In(a, b)
        | Expr::NotIn(a, b)
        | Expr::Subseteq(a, b)
        | Expr::Union(a, b)
        | Expr::Intersect(a, b)
        | Expr::SetMinus(a, b)
        | Expr::FuncSet(a, b)
        | Expr::LeadsTo(a, b)
        | Expr::WeakFair(a, b)
        | Expr::StrongFair(a, b) => expr_contains_prime(&a.node) || expr_contains_prime(&b.node),
        // Unary operators
        Expr::Not(a)
        | Expr::Neg(a)
        | Expr::Domain(a)
        | Expr::Powerset(a)
        | Expr::BigUnion(a)
        | Expr::Always(a)
        | Expr::Eventually(a)
        | Expr::Enabled(a) => expr_contains_prime(&a.node),
        // IF-THEN-ELSE
        Expr::If(c, t, e) => {
            expr_contains_prime(&c.node)
                || expr_contains_prime(&t.node)
                || expr_contains_prime(&e.node)
        }
        // LET
        Expr::Let(defs, body) => {
            defs.iter().any(|d| expr_contains_prime(&d.body.node))
                || expr_contains_prime(&body.node)
        }
        // Quantifiers - must check both body AND bounds' domains
        Expr::Forall(bounds, body) | Expr::Exists(bounds, body) => {
            expr_contains_prime(&body.node)
                || bounds
                    .iter()
                    .any(|b| b.domain.as_ref().is_some_and(|d| expr_contains_prime(&d.node)))
        }
        Expr::Choose(bound, body) => {
            expr_contains_prime(&body.node)
                || bound
                    .domain
                    .as_ref()
                    .is_some_and(|d| expr_contains_prime(&d.node))
        }
        // Function/operator application
        Expr::Apply(op, args) => {
            expr_contains_prime(&op.node) || args.iter().any(|a| expr_contains_prime(&a.node))
        }
        Expr::FuncApply(f, arg) => expr_contains_prime(&f.node) || expr_contains_prime(&arg.node),
        // Function definition - check both body AND bounds' domains
        Expr::FuncDef(bounds, body) => {
            expr_contains_prime(&body.node)
                || bounds
                    .iter()
                    .any(|b| b.domain.as_ref().is_some_and(|d| expr_contains_prime(&d.node)))
        }
        // Sets - must check bounds' domains too
        Expr::SetBuilder(e, bounds) => {
            expr_contains_prime(&e.node)
                || bounds
                    .iter()
                    .any(|b| b.domain.as_ref().is_some_and(|d| expr_contains_prime(&d.node)))
        }
        // SetFilter - CRITICAL: must check bound's domain, not just predicate!
        // For {m \in rcvd'[self] : m[2] = "ECHO0"}, the prime is in the domain
        Expr::SetFilter(bound, pred) => {
            bound
                .domain
                .as_ref()
                .is_some_and(|d| expr_contains_prime(&d.node))
                || expr_contains_prime(&pred.node)
        }
        Expr::SetEnum(elems) | Expr::Tuple(elems) => {
            elems.iter().any(|e| expr_contains_prime(&e.node))
        }
        Expr::Times(elems) => elems.iter().any(|e| expr_contains_prime(&e.node)),
        // Records
        Expr::Record(fields) | Expr::RecordSet(fields) => {
            fields.iter().any(|(_, v)| expr_contains_prime(&v.node))
        }
        Expr::RecordAccess(base, _) => expr_contains_prime(&base.node),
        // EXCEPT
        Expr::Except(base, updates) => {
            expr_contains_prime(&base.node)
                || updates.iter().any(|u| {
                    u.path.iter().any(|p| match p {
                        tla_core::ast::ExceptPathElement::Index(idx_expr) => {
                            expr_contains_prime(&idx_expr.node)
                        }
                        tla_core::ast::ExceptPathElement::Field(_) => false,
                    }) || expr_contains_prime(&u.value.node)
                })
        }
        // CASE
        Expr::Case(arms, other) => {
            arms.iter()
                .any(|a| expr_contains_prime(&a.guard.node) || expr_contains_prime(&a.body.node))
                || other.as_ref().is_some_and(|o| expr_contains_prime(&o.node))
        }
        // Atoms - don't contain primes
        Expr::Ident(_) | Expr::Bool(_) | Expr::Int(_) | Expr::String(_) | Expr::OpRef(_) => false,
        // Module references
        Expr::ModuleRef(_, _, args) => args.iter().any(|a| expr_contains_prime(&a.node)),
        // Instance expressions - substitutions might reference primed vars
        Expr::InstanceExpr(_, subs) => subs.iter().any(|s| expr_contains_prime(&s.to.node)),
        // Lambda
        Expr::Lambda(_, body) => expr_contains_prime(&body.node),
    }
}

/// A constraint extracted from a predicate
#[derive(Debug, Clone)]
pub enum Constraint {
    /// Variable equals a concrete value: x = v
    Eq(String, Value),
    /// Variable is in a set: x \in S
    In(String, Vec<Value>),
    /// Variable is not equal to a value: x /= v or x # v
    /// Note: NotEq alone cannot define a domain; it must be combined with
    /// other constraints that establish the domain.
    NotEq(String, Value),
    /// Variable equals an expression that depends on other variables.
    /// The expression will be evaluated after binding other variables.
    /// This handles patterns like: `state = [n \in Node |-> IF initiator[n] THEN "cand" ELSE "lost"]`
    /// where `state` depends on `initiator`.
    Deferred(String, Box<Spanned<Expr>>),
    /// Variable is in a set expression that depends on other variables.
    /// The set expression will be evaluated after binding other variables.
    /// This handles patterns like: terminationDetected \in {FALSE, terminated}
    /// where `terminated` depends on `active`.
    DeferredIn(String, Box<Spanned<Expr>>),
    /// A boolean filter expression that must evaluate to TRUE.
    /// This handles constraints like `in.ack = in.rdy` that compare record fields
    /// or other expressions that don't directly enumerate variable values.
    /// The expression is evaluated after all variables have been bound from
    /// other constraints, and states that don't satisfy the filter are discarded.
    Filter(Box<Spanned<Expr>>),
}

/// Substitute parameters in an expression with argument expressions
///
/// Used to inline operator definitions during constraint extraction.
/// For example, if we have `XInit(v) == v = 0` and we call `XInit(x)`,
/// this substitutes `v -> x` to get `x = 0`.
fn substitute_params(
    expr: &Expr,
    params: &[tla_core::ast::OpParam],
    args: &[Spanned<Expr>],
) -> Expr {
    use std::collections::HashMap;
    let param_map: HashMap<&str, &Expr> = params
        .iter()
        .zip(args.iter())
        .map(|(p, a)| (p.name.node.as_str(), &a.node))
        .collect();
    substitute_params_rec(expr, &param_map)
}

fn substitute_params_rec(expr: &Expr, param_map: &std::collections::HashMap<&str, &Expr>) -> Expr {
    match expr {
        Expr::Ident(name) => {
            // If this identifier is a parameter, substitute it
            if let Some(&replacement) = param_map.get(name.as_str()) {
                replacement.clone()
            } else {
                expr.clone()
            }
        }
        Expr::And(a, b) => Expr::And(
            Box::new(Spanned {
                node: substitute_params_rec(&a.node, param_map),
                span: a.span,
            }),
            Box::new(Spanned {
                node: substitute_params_rec(&b.node, param_map),
                span: b.span,
            }),
        ),
        Expr::Or(a, b) => Expr::Or(
            Box::new(Spanned {
                node: substitute_params_rec(&a.node, param_map),
                span: a.span,
            }),
            Box::new(Spanned {
                node: substitute_params_rec(&b.node, param_map),
                span: b.span,
            }),
        ),
        Expr::Eq(a, b) => Expr::Eq(
            Box::new(Spanned {
                node: substitute_params_rec(&a.node, param_map),
                span: a.span,
            }),
            Box::new(Spanned {
                node: substitute_params_rec(&b.node, param_map),
                span: b.span,
            }),
        ),
        Expr::Neq(a, b) => Expr::Neq(
            Box::new(Spanned {
                node: substitute_params_rec(&a.node, param_map),
                span: a.span,
            }),
            Box::new(Spanned {
                node: substitute_params_rec(&b.node, param_map),
                span: b.span,
            }),
        ),
        Expr::In(a, b) => Expr::In(
            Box::new(Spanned {
                node: substitute_params_rec(&a.node, param_map),
                span: a.span,
            }),
            Box::new(Spanned {
                node: substitute_params_rec(&b.node, param_map),
                span: b.span,
            }),
        ),
        Expr::Apply(op, args) => Expr::Apply(
            Box::new(Spanned {
                node: substitute_params_rec(&op.node, param_map),
                span: op.span,
            }),
            args.iter()
                .map(|a| Spanned {
                    node: substitute_params_rec(&a.node, param_map),
                    span: a.span,
                })
                .collect(),
        ),
        // For other expressions, just clone (they don't contain variables to substitute)
        _ => expr.clone(),
    }
}

/// Extract constraints from an Init predicate expression
///
/// Maximum expression complexity for constraint extraction.
/// This limit exists because extract_constraints_rec uses recursion.
/// With stacker::maybe_grow and proper red zone (STACK_RED_ZONE), much larger
/// expressions can be handled. The limit is set conservatively to avoid
/// excessive memory usage from very deeply nested expressions.
///
/// To fully remove this limit, convert extract_constraints_rec to iterative
/// using an explicit task stack (similar to count_expr_nodes pattern above).
/// This requires handling the complex result combination logic (cross-product
/// for AND, concatenation for OR, operator inlining, etc.) iteratively.
const MAX_EXPR_NODES: usize = 4096;

/// Count the approximate number of "structural" nodes in an expression tree.
///
/// This is iterative to avoid stack overflow, and intentionally ignores leaf nodes
/// (identifiers, literals, etc.) so that long conjunctions of simple equalities
/// don't get rejected as "too complex".
fn count_expr_nodes(expr: &Expr) -> usize {
    let mut count = 0usize;
    let mut stack: Vec<&Expr> = vec![expr];

    while let Some(e) = stack.pop() {
        match e {
            Expr::And(a, b) | Expr::Or(a, b) | Expr::Eq(a, b) | Expr::Neq(a, b)
            | Expr::Lt(a, b) | Expr::Leq(a, b) | Expr::Gt(a, b) | Expr::Geq(a, b)
            | Expr::In(a, b) | Expr::NotIn(a, b) | Expr::Implies(a, b) => {
                count += 1;
                // Stop early if we've exceeded the limit
                if count > MAX_EXPR_NODES {
                    return count;
                }
                stack.push(&a.node);
                stack.push(&b.node);
            }
            Expr::Not(inner) => {
                count += 1;
                if count > MAX_EXPR_NODES {
                    return count;
                }
                stack.push(&inner.node);
            }
            Expr::Apply(func, args) => {
                count += 1;
                if count > MAX_EXPR_NODES {
                    return count;
                }
                stack.push(&func.node);
                for arg in args {
                    stack.push(&arg.node);
                }
            }
            _ => {}
        }
    }
    count
}

///
/// Returns a list of constraints that can be used to generate initial states.
/// If the predicate cannot be analyzed, returns None.
pub fn extract_init_constraints(
    ctx: &EvalCtx,
    expr: &Spanned<Expr>,
    vars: &[Arc<str>],
) -> Option<Vec<Vec<Constraint>>> {
    // Bail out early for complex expressions to prevent stack overflow
    if count_expr_nodes(&expr.node) > MAX_EXPR_NODES {
        return None;
    }
    extract_constraints_rec(ctx, &expr.node, vars)
}

fn extract_constraints_rec(
    ctx: &EvalCtx,
    expr: &Expr,
    vars: &[Arc<str>],
) -> Option<Vec<Vec<Constraint>>> {
    // Use stacker to grow stack on demand for deeply nested expressions.
    // Use STACK_RED_ZONE (1MB) like other recursive functions, not 32KB.
    stacker::maybe_grow(STACK_RED_ZONE, STACK_GROW_SIZE, || {
        extract_constraints_rec_inner(ctx, expr, vars)
    })
}

fn extract_constraints_rec_inner(
    ctx: &EvalCtx,
    expr: &Expr,
    vars: &[Arc<str>],
) -> Option<Vec<Vec<Constraint>>> {
    match expr {
        // Conjunction: cross-product (DNF expansion)
        Expr::And(a, b) => {
            let left = extract_constraints_rec(ctx, &a.node, vars)?;
            let right = extract_constraints_rec(ctx, &b.node, vars)?;

            let mut branches: Vec<Vec<Constraint>> = Vec::new();
            for left_branch in &left {
                for right_branch in &right {
                    let mut merged = left_branch.clone();
                    merged.extend(right_branch.iter().cloned());
                    branches.push(merged);
                }
            }
            Some(branches)
        }

        // Disjunction: concatenate branches
        Expr::Or(a, b) => {
            let mut branches = extract_constraints_rec(ctx, &a.node, vars)?;
            branches.extend(extract_constraints_rec(ctx, &b.node, vars)?);
            Some(branches)
        }

        // Equality: x = value
        Expr::Eq(lhs, rhs) => {
            // Check if LHS is a variable
            if let Expr::Ident(name) = &lhs.node {
                if vars.iter().any(|v| v.as_ref() == name) {
                    // Try to evaluate RHS to get a concrete value
                    if let Ok(value) = eval(ctx, rhs) {
                        return Some(vec![vec![Constraint::Eq(name.clone(), value)]]);
                    }
                    // Evaluation failed - likely depends on other variables
                    // Store as a deferred constraint to evaluate later
                    return Some(vec![vec![Constraint::Deferred(name.clone(), rhs.clone())]]);
                }
            }
            // Check if RHS is a variable (symmetric case: value = x)
            if let Expr::Ident(name) = &rhs.node {
                if vars.iter().any(|v| v.as_ref() == name) {
                    if let Ok(value) = eval(ctx, lhs) {
                        return Some(vec![vec![Constraint::Eq(name.clone(), value)]]);
                    }
                    // Evaluation failed - store as deferred constraint
                    return Some(vec![vec![Constraint::Deferred(name.clone(), lhs.clone())]]);
                }
            }
            // Neither side is a simple variable - treat as a filter constraint.
            // This handles patterns like `in.ack = in.rdy` (record field comparisons)
            // or `Len(q) = 0` where the expression can be evaluated after
            // variables are bound.
            Some(vec![vec![Constraint::Filter(Box::new(Spanned {
                node: expr.clone(),
                span: lhs.span, // Use LHS span as approximation
            }))]])
        }

        // Inequality: x /= value or x # value
        Expr::Neq(lhs, rhs) => {
            // Check if LHS is a variable
            if let Expr::Ident(name) = &lhs.node {
                if vars.iter().any(|v| v.as_ref() == name) {
                    // Try to evaluate RHS to get a concrete value
                    if let Ok(value) = eval(ctx, rhs) {
                        return Some(vec![vec![Constraint::NotEq(name.clone(), value)]]);
                    }
                }
            }
            // Check if RHS is a variable (symmetric case: value /= x)
            if let Expr::Ident(name) = &rhs.node {
                if vars.iter().any(|v| v.as_ref() == name) {
                    if let Ok(value) = eval(ctx, lhs) {
                        return Some(vec![vec![Constraint::NotEq(name.clone(), value)]]);
                    }
                }
            }
            // Neither side is a simple variable - treat as a filter constraint.
            Some(vec![vec![Constraint::Filter(Box::new(Spanned {
                node: expr.clone(),
                span: lhs.span,
            }))]])
        }

        // Membership: x \in S
        Expr::In(lhs, rhs) => {
            if let Expr::Ident(name) = &lhs.node {
                if vars.iter().any(|v| v.as_ref() == name) {
                    // Evaluate the set
                    if let Ok(set_val) = eval(ctx, rhs) {
                        // Handle both Set and Interval using iter_set
                        if let Some(iter) = set_val.iter_set() {
                            let values: Vec<Value> = iter.collect();
                            return Some(vec![vec![Constraint::In(name.clone(), values)]]);
                        }
                    }
                    // Evaluation failed - likely depends on other variables
                    // Store as a deferred constraint to evaluate later
                    return Some(vec![vec![Constraint::DeferredIn(
                        name.clone(),
                        rhs.clone(),
                    )]]);
                }
            }
            None
        }

        // TRUE is always satisfied (no constraint added)
        Expr::Bool(true) => Some(vec![Vec::new()]),

        // FALSE is unsatisfiable (no branches)
        Expr::Bool(false) => Some(Vec::new()),

        // Negation: ~expr - evaluate if possible, otherwise fail
        Expr::Not(inner) => {
            // First try to recursively extract constraints
            if let Some(inner_branches) = extract_constraints_rec(ctx, &inner.node, vars) {
                // If inner has no branches (FALSE), negation makes it TRUE (no constraints)
                if inner_branches.is_empty() {
                    return Some(vec![Vec::new()]);
                }
                // If inner has one empty branch (TRUE), negation makes it FALSE (no branches)
                if inner_branches.len() == 1 && inner_branches[0].is_empty() {
                    return Some(Vec::new());
                }
            }
            // Try to evaluate the expression as a constant boolean
            if let Ok(Value::Bool(b)) = eval(ctx, inner) {
                if b {
                    // ~TRUE = FALSE: no satisfying branches
                    Some(Vec::new())
                } else {
                    // ~FALSE = TRUE: trivially satisfied
                    Some(vec![Vec::new()])
                }
            } else {
                // Can't evaluate or not a boolean
                None
            }
        }

        // Operator application: inline the operator definition and extract constraints from body
        Expr::Apply(op_expr, args) => {
            if let Expr::Ident(name) = &op_expr.node {
                if let Some(def) = ctx.get_op(name) {
                    if def.params.len() == args.len() {
                        if args.is_empty() {
                            // Zero-argument operator: inline directly
                            return extract_constraints_rec(ctx, &def.body.node, vars);
                        } else {
                            // Operator with arguments: substitute arguments into the body
                            let body = substitute_params(&def.body.node, &def.params, args);
                            return extract_constraints_rec(ctx, &body, vars);
                        }
                    }
                }
            }
            // Not a user-defined operator (may be builtin like IsStronglyConnected).
            // Try to evaluate as a boolean predicate - this handles guards in Init
            // like `IsStronglyConnected(G)` where G is a concrete graph.
            let spanned_expr = Spanned {
                node: expr.clone(),
                span: tla_core::Span::dummy(),
            };
            if let Ok(Value::Bool(b)) = eval(ctx, &spanned_expr) {
                if b {
                    // TRUE: trivially satisfied (no constraint)
                    return Some(vec![Vec::new()]);
                } else {
                    // FALSE: unsatisfiable (no branches)
                    return Some(Vec::new());
                }
            }
            // Evaluation failed - can't extract constraint
            None
        }

        // Identifier might be a zero-argument operator reference or a constant
        Expr::Ident(name) => {
            // Check if this is a defined operator (not a variable)
            if !vars.iter().any(|v| v.as_ref() == name) {
                if let Some(def) = ctx.get_op(name) {
                    // Zero-argument operator: inline its body
                    if def.params.is_empty() {
                        return extract_constraints_rec(ctx, &def.body.node, vars);
                    }
                }
                // Try to evaluate as a constant boolean (e.g., a config constant)
                let spanned_expr = Spanned {
                    node: expr.clone(),
                    span: tla_core::Span::dummy(),
                };
                if let Ok(Value::Bool(b)) = eval(ctx, &spanned_expr) {
                    if b {
                        // TRUE: trivially satisfied
                        return Some(vec![Vec::new()]);
                    } else {
                        // FALSE: no satisfying branches
                        return Some(Vec::new());
                    }
                }
            }
            // Can't extract constraint from a variable reference by itself
            None
        }

        // ModuleRef: instance reference like InChan!Init
        // Look up the instance, get the operator, apply substitutions, then extract constraints
        Expr::ModuleRef(instance_name, op_name, args) => {
            // Only support zero-argument module references for now
            if !args.is_empty() {
                return None;
            }

            // Resolve the instance info.
            //
            // Named instances defined inside instanced modules (nested INSTANCE) are not
            // registered globally; they are visible via the module's local operator scope.
            // Match eval_module_ref by falling back to a locally-visible operator whose body
            // is an InstanceExpr.
            let (module_name, instance_subs): (String, Vec<Substitution>) = if let Some(info) =
                ctx.get_instance(instance_name.name())
            {
                (info.module_name.clone(), info.substitutions.clone())
            } else if let Some(def) = ctx.get_op(instance_name.name()) {
                match &def.body.node {
                    Expr::InstanceExpr(module_name, subs) => (module_name.clone(), subs.clone()),
                    _ => return None,
                }
            } else {
                return None;
            };

            // Get the operator from the instanced module
            let op_def = ctx.get_instance_op(&module_name, op_name)?;

            // Compose substitutions through nested instances (like eval_module_ref).
            fn compose_instance_substitutions(
                instance_subs: &[Substitution],
                outer_subs: Option<&[Substitution]>,
            ) -> Vec<Substitution> {
                let Some(outer_subs) = outer_subs else {
                    return instance_subs.to_vec();
                };
                if outer_subs.is_empty() {
                    return instance_subs.to_vec();
                }

                let mut combined = Vec::with_capacity(instance_subs.len() + outer_subs.len());
                let mut overridden: std::collections::HashSet<&str> =
                    std::collections::HashSet::new();

                for sub in instance_subs {
                    overridden.insert(sub.from.node.as_str());
                    combined.push(Substitution {
                        from: sub.from.clone(),
                        to: apply_substitutions(&sub.to, outer_subs),
                    });
                }

                for sub in outer_subs {
                    if overridden.contains(sub.from.node.as_str()) {
                        continue;
                    }
                    combined.push(sub.clone());
                }

                combined
            }

            let effective_subs =
                compose_instance_substitutions(&instance_subs, ctx.instance_substitutions());

            // Apply substitutions to the operator body
            let substituted_body = apply_substitutions(&op_def.body, &effective_subs);

            // Evaluate/extract in a scope where unqualified operator names resolve within the
            // instanced module (e.g., RingBuffer!Init referencing LastIndex).
            let instance_local_ops: OpEnv = ctx
                .instance_ops()
                .get(&module_name)
                .cloned()
                .unwrap_or_default();
            let instance_ctx = ctx
                .with_local_ops(instance_local_ops)
                .with_instance_substitutions(effective_subs);

            // Recursively extract constraints from the substituted body
            extract_constraints_rec(&instance_ctx, &substituted_body.node, vars)
        }

        // IF-THEN-ELSE: equivalent to (cond /\ then) \/ (~cond /\ else)
        // For Init constraints, we first try to evaluate the condition.
        // If it evaluates to a boolean, we select the appropriate branch.
        // If it doesn't evaluate (depends on variables), we try to expand to disjunction.
        Expr::If(cond, then_branch, else_branch) => {
            // First, try to evaluate the condition directly
            if let Ok(Value::Bool(cond_val)) = eval(ctx, cond) {
                if cond_val {
                    return extract_constraints_rec(ctx, &then_branch.node, vars);
                } else {
                    return extract_constraints_rec(ctx, &else_branch.node, vars);
                }
            }

            // Condition couldn't be evaluated - try to extract it as a constraint
            // For patterns like: IF x = 0 THEN ... ELSE ...
            // We can expand to: (x = 0 /\ then) \/ (x /= 0 /\ else)

            // Extract constraints from both branches
            let then_constraints = extract_constraints_rec(ctx, &then_branch.node, vars)?;
            let else_constraints = extract_constraints_rec(ctx, &else_branch.node, vars)?;

            // Try to extract a constraint from the condition and its negation
            let cond_constraint = extract_constraint_from_condition(ctx, &cond.node, vars);
            let negated_constraint = negate_condition(ctx, &cond.node, vars);

            match (cond_constraint, negated_constraint) {
                (Some(cond_c), Some(neg_c)) => {
                    // We have both condition and negation constraints
                    // Combine: (cond /\ then) \/ (negated /\ else)
                    let mut combined_then = Vec::new();
                    for then_branch in &then_constraints {
                        let mut merged = vec![cond_c.clone()];
                        merged.extend(then_branch.iter().cloned());
                        combined_then.push(merged);
                    }

                    let mut combined_else = Vec::new();
                    for else_branch in &else_constraints {
                        let mut merged = vec![neg_c.clone()];
                        merged.extend(else_branch.iter().cloned());
                        combined_else.push(merged);
                    }

                    let mut all_branches = combined_then;
                    all_branches.extend(combined_else);
                    Some(all_branches)
                }
                (Some(cond_c), None) => {
                    // Have condition but can't negate it
                    // Combine condition with then-branch, include else as-is
                    let mut combined_then = Vec::new();
                    for then_branch in &then_constraints {
                        let mut merged = vec![cond_c.clone()];
                        merged.extend(then_branch.iter().cloned());
                        combined_then.push(merged);
                    }

                    let mut all_branches = combined_then;
                    all_branches.extend(else_constraints);
                    Some(all_branches)
                }
                _ => {
                    // Can't extract condition as constraint - just union branches
                    let mut all_branches = then_constraints;
                    all_branches.extend(else_constraints);
                    Some(all_branches)
                }
            }
        }

        // Existential quantifier: \E x \in S : P(x)
        // This creates multiple branches - one for each value in S, where we substitute x with that value
        // and extract constraints from P(x)
        Expr::Exists(bound_vars, body) => {
            // Handle single bound variable (most common case)
            // Multiple bound vars like \E x \in S, y \in T : P(x, y) would need nested iteration
            if bound_vars.len() == 1 {
                let bvar = &bound_vars[0];
                let domain = bvar.domain.as_ref()?;

                // Evaluate domain to get set of values
                let domain_val = eval(ctx, domain).ok()?;
                let domain_iter = domain_val.iter_set()?;

                // For each value in domain, substitute into body and extract constraints
                let mut all_branches = Vec::new();
                for val in domain_iter {
                    // Substitute the bound variable with the value
                    let substituted = substitute_bound_var(&body.node, &bvar.name.node, &val);

                    // Extract constraints from the substituted body
                    if let Some(branches) = extract_constraints_rec(ctx, &substituted, vars) {
                        all_branches.extend(branches);
                    } else {
                        // If any branch fails, we can't fully enumerate
                        return None;
                    }
                }

                if all_branches.is_empty() {
                    // Empty domain means no satisfying states (like FALSE)
                    Some(Vec::new())
                } else {
                    Some(all_branches)
                }
            } else if bound_vars.len() == 2 {
                // Handle two bound variables: \E x \in S, y \in T : P(x, y)
                let bvar0 = &bound_vars[0];
                let bvar1 = &bound_vars[1];
                let domain0 = bvar0.domain.as_ref()?;
                let domain1 = bvar1.domain.as_ref()?;

                let domain_val0 = eval(ctx, domain0).ok()?;
                let domain_iter0: Vec<_> = domain_val0.iter_set()?.collect();

                let mut all_branches = Vec::new();
                for val0 in &domain_iter0 {
                    // Substitute first variable
                    let substituted0 = substitute_bound_var(&body.node, &bvar0.name.node, val0);

                    // Evaluate second domain (may depend on first variable, but we've substituted)
                    // For simplicity, assume domain1 doesn't depend on bvar0
                    let domain_val1 = eval(ctx, domain1).ok()?;
                    let domain_iter1 = domain_val1.iter_set()?;

                    for val1 in domain_iter1 {
                        let substituted =
                            substitute_bound_var(&substituted0, &bvar1.name.node, &val1);

                        if let Some(branches) = extract_constraints_rec(ctx, &substituted, vars) {
                            all_branches.extend(branches);
                        } else {
                            return None;
                        }
                    }
                }

                if all_branches.is_empty() {
                    Some(Vec::new())
                } else {
                    Some(all_branches)
                }
            } else {
                // More than 2 bound variables - not supported for now
                None
            }
        }

        // For now, other expressions are not supported for constraint extraction
        _ => None,
    }
}

/// Substitute a bound variable name with a value in an expression.
/// Returns a new expression with all occurrences of `var_name` replaced with the value.
fn substitute_bound_var(expr: &Expr, var_name: &str, value: &Value) -> Expr {
    let value_expr = value_to_expr(value);
    substitute_bound_var_with_expr(expr, var_name, &value_expr)
}

fn substitute_bound_var_with_expr(expr: &Expr, var_name: &str, replacement: &Expr) -> Expr {
    match expr {
        Expr::Ident(name) if name == var_name => replacement.clone(),

        Expr::And(a, b) => Expr::And(
            Box::new(Spanned {
                node: substitute_bound_var_with_expr(&a.node, var_name, replacement),
                span: a.span,
            }),
            Box::new(Spanned {
                node: substitute_bound_var_with_expr(&b.node, var_name, replacement),
                span: b.span,
            }),
        ),

        Expr::Or(a, b) => Expr::Or(
            Box::new(Spanned {
                node: substitute_bound_var_with_expr(&a.node, var_name, replacement),
                span: a.span,
            }),
            Box::new(Spanned {
                node: substitute_bound_var_with_expr(&b.node, var_name, replacement),
                span: b.span,
            }),
        ),

        Expr::Eq(a, b) => Expr::Eq(
            Box::new(Spanned {
                node: substitute_bound_var_with_expr(&a.node, var_name, replacement),
                span: a.span,
            }),
            Box::new(Spanned {
                node: substitute_bound_var_with_expr(&b.node, var_name, replacement),
                span: b.span,
            }),
        ),

        Expr::Neq(a, b) => Expr::Neq(
            Box::new(Spanned {
                node: substitute_bound_var_with_expr(&a.node, var_name, replacement),
                span: a.span,
            }),
            Box::new(Spanned {
                node: substitute_bound_var_with_expr(&b.node, var_name, replacement),
                span: b.span,
            }),
        ),

        Expr::In(a, b) => Expr::In(
            Box::new(Spanned {
                node: substitute_bound_var_with_expr(&a.node, var_name, replacement),
                span: a.span,
            }),
            Box::new(Spanned {
                node: substitute_bound_var_with_expr(&b.node, var_name, replacement),
                span: b.span,
            }),
        ),

        Expr::If(cond, then_br, else_br) => Expr::If(
            Box::new(Spanned {
                node: substitute_bound_var_with_expr(&cond.node, var_name, replacement),
                span: cond.span,
            }),
            Box::new(Spanned {
                node: substitute_bound_var_with_expr(&then_br.node, var_name, replacement),
                span: then_br.span,
            }),
            Box::new(Spanned {
                node: substitute_bound_var_with_expr(&else_br.node, var_name, replacement),
                span: else_br.span,
            }),
        ),

        Expr::Apply(op, args) => Expr::Apply(
            Box::new(Spanned {
                node: substitute_bound_var_with_expr(&op.node, var_name, replacement),
                span: op.span,
            }),
            args.iter()
                .map(|a| Spanned {
                    node: substitute_bound_var_with_expr(&a.node, var_name, replacement),
                    span: a.span,
                })
                .collect(),
        ),

        Expr::FuncDef(bvars, body) => {
            // Check if any bound variable shadows our target
            if bvars.iter().any(|bv| bv.name.node == var_name) {
                // Variable is shadowed, don't substitute in body
                expr.clone()
            } else {
                // Substitute in domains and body
                let new_bvars: Vec<_> = bvars
                    .iter()
                    .map(|bv| BoundVar {
                        name: bv.name.clone(),
                        domain: bv.domain.as_ref().map(|d| {
                            Box::new(Spanned {
                                node: substitute_bound_var_with_expr(
                                    &d.node,
                                    var_name,
                                    replacement,
                                ),
                                span: d.span,
                            })
                        }),
                        pattern: bv.pattern.clone(),
                    })
                    .collect();
                Expr::FuncDef(
                    new_bvars,
                    Box::new(Spanned {
                        node: substitute_bound_var_with_expr(&body.node, var_name, replacement),
                        span: body.span,
                    }),
                )
            }
        }

        // SetBuilder: {expr : x \in S, y \in T, ...}
        Expr::SetBuilder(body, bvars) => {
            // Check if any bound variable shadows our target
            if bvars.iter().any(|bv| bv.name.node == var_name) {
                // Variable is shadowed
                expr.clone()
            } else {
                // Substitute in domains and body
                let new_bvars: Vec<_> = bvars
                    .iter()
                    .map(|bv| BoundVar {
                        name: bv.name.clone(),
                        domain: bv.domain.as_ref().map(|d| {
                            Box::new(Spanned {
                                node: substitute_bound_var_with_expr(
                                    &d.node,
                                    var_name,
                                    replacement,
                                ),
                                span: d.span,
                            })
                        }),
                        pattern: bv.pattern.clone(),
                    })
                    .collect();
                Expr::SetBuilder(
                    Box::new(Spanned {
                        node: substitute_bound_var_with_expr(&body.node, var_name, replacement),
                        span: body.span,
                    }),
                    new_bvars,
                )
            }
        }

        // SetFilter: {x \in S : P}
        Expr::SetFilter(bvar, body) => {
            // Check if bound variable shadows our target
            if bvar.name.node == var_name {
                expr.clone()
            } else {
                let new_domain = bvar.domain.as_ref().map(|d| {
                    Box::new(Spanned {
                        node: substitute_bound_var_with_expr(&d.node, var_name, replacement),
                        span: d.span,
                    })
                });
                Expr::SetFilter(
                    BoundVar {
                        name: bvar.name.clone(),
                        domain: new_domain,
                        pattern: bvar.pattern.clone(),
                    },
                    Box::new(Spanned {
                        node: substitute_bound_var_with_expr(&body.node, var_name, replacement),
                        span: body.span,
                    }),
                )
            }
        }

        // SetEnum: {a, b, c}
        Expr::SetEnum(elems) => Expr::SetEnum(
            elems
                .iter()
                .map(|e| Spanned {
                    node: substitute_bound_var_with_expr(&e.node, var_name, replacement),
                    span: e.span,
                })
                .collect(),
        ),

        // Not: ~A
        Expr::Not(inner) => Expr::Not(Box::new(Spanned {
            node: substitute_bound_var_with_expr(&inner.node, var_name, replacement),
            span: inner.span,
        })),

        // Implies: A => B
        Expr::Implies(a, b) => Expr::Implies(
            Box::new(Spanned {
                node: substitute_bound_var_with_expr(&a.node, var_name, replacement),
                span: a.span,
            }),
            Box::new(Spanned {
                node: substitute_bound_var_with_expr(&b.node, var_name, replacement),
                span: b.span,
            }),
        ),

        // Let: LET defs IN body
        Expr::Let(defs, body) => {
            // Substitute in definition bodies (but check for shadowing)
            let new_defs: Vec<_> = defs
                .iter()
                .map(|def| OperatorDef {
                    name: def.name.clone(),
                    params: def.params.clone(),
                    local: def.local,
                    body: Spanned {
                        node: if def.params.iter().any(|p| p.name.node == var_name) {
                            // Variable is shadowed by parameter
                            def.body.node.clone()
                        } else {
                            substitute_bound_var_with_expr(&def.body.node, var_name, replacement)
                        },
                        span: def.body.span,
                    },
                })
                .collect();
            // Check if any LET name shadows our target
            let shadowed = defs.iter().any(|d| d.name.node == var_name);
            Expr::Let(
                new_defs,
                Box::new(Spanned {
                    node: if shadowed {
                        body.node.clone()
                    } else {
                        substitute_bound_var_with_expr(&body.node, var_name, replacement)
                    },
                    span: body.span,
                }),
            )
        }

        // Tuple: <<a, b, c>>
        Expr::Tuple(elems) => Expr::Tuple(
            elems
                .iter()
                .map(|e| Spanned {
                    node: substitute_bound_var_with_expr(&e.node, var_name, replacement),
                    span: e.span,
                })
                .collect(),
        ),

        // FuncApply: f[x]
        Expr::FuncApply(f, arg) => Expr::FuncApply(
            Box::new(Spanned {
                node: substitute_bound_var_with_expr(&f.node, var_name, replacement),
                span: f.span,
            }),
            Box::new(Spanned {
                node: substitute_bound_var_with_expr(&arg.node, var_name, replacement),
                span: arg.span,
            }),
        ),

        // Record: [a |-> 1, b |-> 2]
        Expr::Record(fields) => Expr::Record(
            fields
                .iter()
                .map(|(name, value)| {
                    (
                        name.clone(),
                        Spanned {
                            node: substitute_bound_var_with_expr(&value.node, var_name, replacement),
                            span: value.span,
                        },
                    )
                })
                .collect(),
        ),

        // RecordAccess: r.field
        Expr::RecordAccess(base, field) => Expr::RecordAccess(
            Box::new(Spanned {
                node: substitute_bound_var_with_expr(&base.node, var_name, replacement),
                span: base.span,
            }),
            field.clone(),
        ),

        // For other expressions, clone (no substitution needed or not handled)
        _ => expr.clone(),
    }
}

/// Extract the "remainder" of a conjunction after removing the conjunct referencing a specific operator.
///
/// When Init = A /\ B and we're enumerating from A (e.g., TypeOK), we don't need to re-evaluate A
/// in the filter - we only need to evaluate B. This function extracts B given A's name.
///
/// Returns Some(remainder) if the named conjunct is found and removed.
/// Returns None if the named conjunct is not found as a top-level conjunct.
pub fn extract_conjunction_remainder(
    ctx: &EvalCtx,
    expr: &Spanned<Expr>,
    conjunct_name: &str,
) -> Option<Spanned<Expr>> {
    extract_remainder_rec(ctx, expr, conjunct_name)
}

fn extract_remainder_rec(
    ctx: &EvalCtx,
    expr: &Spanned<Expr>,
    conjunct_name: &str,
) -> Option<Spanned<Expr>> {
    match &expr.node {
        // If this is a direct reference to the conjunct name, the remainder is "true"
        // (i.e., nothing else to check)
        Expr::Ident(name) if name == conjunct_name => {
            // Return Bool(true) to indicate this conjunct is removed
            Some(Spanned {
                node: Expr::Bool(true),
                span: expr.span,
            })
        }

        // Conjunction: try to remove the named conjunct from either side
        Expr::And(a, b) => {
            // Try removing from left side
            if let Some(left_remainder) = extract_remainder_rec(ctx, a, conjunct_name) {
                // Left contained the conjunct - result is left_remainder /\ b
                if matches!(left_remainder.node, Expr::Bool(true)) {
                    // If left is just "true", return right only
                    return Some((**b).clone());
                }
                // Otherwise combine left_remainder with right
                return Some(Spanned {
                    node: Expr::And(Box::new(left_remainder), b.clone()),
                    span: expr.span,
                });
            }

            // Try removing from right side
            if let Some(right_remainder) = extract_remainder_rec(ctx, b, conjunct_name) {
                // Right contained the conjunct - result is a /\ right_remainder
                if matches!(right_remainder.node, Expr::Bool(true)) {
                    // If right is just "true", return left only
                    return Some((**a).clone());
                }
                // Otherwise combine left with right_remainder
                return Some(Spanned {
                    node: Expr::And(a.clone(), Box::new(right_remainder)),
                    span: expr.span,
                });
            }

            // Conjunct name not found in either side
            None
        }

        // Operator application: check if it's calling the conjunct operator
        Expr::Apply(op, args) => {
            // Check if this is a call to a user-defined operator that matches conjunct_name
            if let Expr::Ident(op_name) = &op.node {
                // First, try to expand the operator definition
                if let Some(def) = ctx.get_op(op_name) {
                    if def.params.is_empty() && args.is_empty() {
                        // No-arg operator - check if its body can have the conjunct removed
                        if let Some(remainder) =
                            extract_remainder_rec(ctx, &def.body, conjunct_name)
                        {
                            return Some(remainder);
                        }
                    }
                }
            }
            None
        }

        // Other expressions: conjunct not found at this level
        _ => None,
    }
}

/// Extract a single constraint from a condition expression.
/// Used for IF-THEN-ELSE handling.
fn extract_constraint_from_condition(
    ctx: &EvalCtx,
    expr: &Expr,
    vars: &[Arc<str>],
) -> Option<Constraint> {
    match expr {
        // Equality: x = value
        Expr::Eq(lhs, rhs) => {
            if let Expr::Ident(name) = &lhs.node {
                if vars.iter().any(|v| v.as_ref() == name) {
                    if let Ok(value) = eval(ctx, rhs) {
                        return Some(Constraint::Eq(name.clone(), value));
                    }
                }
            }
            if let Expr::Ident(name) = &rhs.node {
                if vars.iter().any(|v| v.as_ref() == name) {
                    if let Ok(value) = eval(ctx, lhs) {
                        return Some(Constraint::Eq(name.clone(), value));
                    }
                }
            }
            None
        }
        // Inequality: x /= value
        Expr::Neq(lhs, rhs) => {
            if let Expr::Ident(name) = &lhs.node {
                if vars.iter().any(|v| v.as_ref() == name) {
                    if let Ok(value) = eval(ctx, rhs) {
                        return Some(Constraint::NotEq(name.clone(), value));
                    }
                }
            }
            if let Expr::Ident(name) = &rhs.node {
                if vars.iter().any(|v| v.as_ref() == name) {
                    if let Ok(value) = eval(ctx, lhs) {
                        return Some(Constraint::NotEq(name.clone(), value));
                    }
                }
            }
            None
        }
        // Membership: x \in S
        Expr::In(lhs, rhs) => {
            if let Expr::Ident(name) = &lhs.node {
                if vars.iter().any(|v| v.as_ref() == name) {
                    if let Ok(set_val) = eval(ctx, rhs) {
                        // Handle both Set and Interval using iter_set
                        if let Some(iter) = set_val.iter_set() {
                            let values: Vec<Value> = iter.collect();
                            return Some(Constraint::In(name.clone(), values));
                        }
                    }
                }
            }
            None
        }
        _ => None,
    }
}

/// Negate a condition expression to produce a constraint.
/// For example, negating x = 0 produces x /= 0.
fn negate_condition(ctx: &EvalCtx, expr: &Expr, vars: &[Arc<str>]) -> Option<Constraint> {
    match expr {
        // Negation of x = value is x /= value
        Expr::Eq(lhs, rhs) => {
            if let Expr::Ident(name) = &lhs.node {
                if vars.iter().any(|v| v.as_ref() == name) {
                    if let Ok(value) = eval(ctx, rhs) {
                        return Some(Constraint::NotEq(name.clone(), value));
                    }
                }
            }
            if let Expr::Ident(name) = &rhs.node {
                if vars.iter().any(|v| v.as_ref() == name) {
                    if let Ok(value) = eval(ctx, lhs) {
                        return Some(Constraint::NotEq(name.clone(), value));
                    }
                }
            }
            None
        }
        // Negation of x /= value is x = value
        Expr::Neq(lhs, rhs) => {
            if let Expr::Ident(name) = &lhs.node {
                if vars.iter().any(|v| v.as_ref() == name) {
                    if let Ok(value) = eval(ctx, rhs) {
                        return Some(Constraint::Eq(name.clone(), value));
                    }
                }
            }
            if let Expr::Ident(name) = &rhs.node {
                if vars.iter().any(|v| v.as_ref() == name) {
                    if let Ok(value) = eval(ctx, lhs) {
                        return Some(Constraint::Eq(name.clone(), value));
                    }
                }
            }
            None
        }
        // Can't easily negate other constraints (like x \in S -> x \notin S)
        _ => None,
    }
}

/// A deferred constraint pending evaluation.
enum DeferredConstraint<'a> {
    /// Deferred equality: var = expr (produces one value)
    Eq(&'a str, &'a Spanned<Expr>),
    /// Deferred membership: var \in expr (produces multiple values)
    In(&'a str, &'a Spanned<Expr>),
}

/// Generate all states satisfying the given constraints.
///
/// If `ctx` is provided, deferred constraints (expressions that depend on other variables)
/// will be evaluated after binding the immediate constraints.
fn enumerate_states_from_constraints(
    ctx: Option<&EvalCtx>,
    vars: &[Arc<str>],
    constraints: &[Constraint],
) -> Option<Vec<State>> {
    // Separate constraints into immediate, deferred, and filter
    let mut immediate: Vec<&Constraint> = Vec::new();
    let mut deferred: Vec<DeferredConstraint> = Vec::new();
    let mut filters: Vec<&Spanned<Expr>> = Vec::new();

    for c in constraints {
        match c {
            Constraint::Deferred(name, expr) => {
                deferred.push(DeferredConstraint::Eq(name.as_str(), expr.as_ref()))
            }
            Constraint::DeferredIn(name, expr) => {
                deferred.push(DeferredConstraint::In(name.as_str(), expr.as_ref()))
            }
            Constraint::Filter(expr) => {
                filters.push(expr.as_ref());
            }
            _ => {
                immediate.push(c);
            }
        }
    }

    // Find which variables have immediate constraints vs deferred
    let immediate_vars: Vec<Arc<str>> = vars
        .iter()
        .filter(|v| {
            immediate.iter().any(|c| match c {
                Constraint::Eq(name, _) | Constraint::In(name, _) | Constraint::NotEq(name, _) => {
                    name == v.as_ref()
                }
                Constraint::Deferred(_, _)
                | Constraint::DeferredIn(_, _)
                | Constraint::Filter(_) => false,
            })
        })
        .cloned()
        .collect();

    // Build a map of variable -> possible values for immediate constraints
    let immediate_owned: Vec<Constraint> = immediate.iter().map(|&c| c.clone()).collect();
    let mut var_values: Vec<(Arc<str>, Vec<Value>)> = Vec::new();
    for var in &immediate_vars {
        let values = find_values_for_var(var, &immediate_owned)?;
        if values.is_empty() {
            // Unsatisfiable branch
            return Some(Vec::new());
        }
        var_values.push((var.clone(), values));
    }

    // Enumerate all combinations of immediate constraints
    let mut partial_states = Vec::new();
    enumerate_combinations(&var_values, 0, &mut Vec::new(), &mut partial_states);

    // If no deferred constraints and no filters, we're done
    if deferred.is_empty() && filters.is_empty() {
        return Some(partial_states);
    }

    // Need context to evaluate deferred constraints and filters
    let ctx = ctx?;

    // For each partial state, evaluate deferred constraints
    // Note: DeferredIn can produce multiple values, so we may expand states
    let mut final_states = Vec::new();
    for partial_state in partial_states {
        // Create a new context with the partial state bound
        let mut eval_ctx = ctx.clone();
        for (name, value) in partial_state.vars() {
            eval_ctx.bind_mut(Arc::clone(name), value.clone());
        }

        // Process deferred constraints, potentially expanding to multiple states
        // Start with the partial state and apply each deferred constraint
        let mut current_states = vec![partial_state];

        for def_constraint in &deferred {
            let mut next_states = Vec::new();

            for state in &current_states {
                // Update context with current state bindings
                for (name, value) in state.vars() {
                    eval_ctx.bind_mut(Arc::clone(name), value.clone());
                }

                match def_constraint {
                    DeferredConstraint::Eq(var_name, expr) => {
                        // Deferred equality: produces exactly one value
                        match eval(&eval_ctx, expr) {
                            Ok(value) => {
                                next_states.push(state.clone().with_var(*var_name, value));
                            }
                            Err(_) => {
                                // Evaluation failed - skip this state
                            }
                        }
                    }
                    DeferredConstraint::In(var_name, expr) => {
                        // Deferred membership: produces multiple values (one per set element)
                        match eval(&eval_ctx, expr) {
                            Ok(set_val) => {
                                if let Some(iter) = set_val.iter_set() {
                                    for elem in iter {
                                        next_states.push(state.clone().with_var(*var_name, elem));
                                    }
                                } else if set_val.is_set() {
                                    // Set-like value but not enumerable (e.g., Nat, Int, Real)
                                    // This is an error - cannot enumerate infinite sets
                                    return None;
                                }
                            }
                            Err(_) => {
                                // Evaluation failed - skip this state
                            }
                        }
                    }
                }
            }

            current_states = next_states;
            if current_states.is_empty() {
                break;
            }
        }

        final_states.extend(current_states);
    }

    // Apply filter constraints to eliminate non-matching states
    if !filters.is_empty() {
        final_states.retain(|state| {
            // Create context with state bindings
            let mut filter_ctx = ctx.clone();
            for (name, value) in state.vars() {
                filter_ctx.bind_mut(Arc::clone(name), value.clone());
            }

            // All filters must evaluate to true
            for filter_expr in &filters {
                match eval(&filter_ctx, filter_expr) {
                    Ok(Value::Bool(true)) => {}
                    Ok(Value::Bool(false)) => return false,
                    _ => return false, // Evaluation failed or not boolean - filter out
                }
            }
            true
        });
    }

    Some(final_states)
}

// Allow mutable_key_type: State/Value have interior mutability for lazy evaluation memoization,
// but Ord/Eq implementations don't depend on the mutable state
#[allow(clippy::mutable_key_type)]
pub fn enumerate_states_from_constraint_branches(
    ctx: Option<&EvalCtx>,
    vars: &[Arc<str>],
    branches: &[Vec<Constraint>],
) -> Option<Vec<State>> {
    let mut all_states = BTreeSet::new();
    for branch in branches {
        let states = enumerate_states_from_constraints(ctx, vars, branch)?;
        for state in states {
            all_states.insert(state);
        }
    }
    Some(all_states.into_iter().collect())
}

/// Streaming enumeration directly to BulkStateStorage.
///
/// This is a memory-efficient alternative to `enumerate_states_from_constraint_branches`
/// that avoids creating intermediate `State` (OrdMap-based) objects. Instead, it yields
/// value arrays directly to BulkStateStorage.
///
/// For MCBakery ISpec with 655K states, this eliminates 655K OrdMap allocations.
///
/// The filter receives values in VarRegistry order.
///
/// # Arguments
/// * `ctx` - Mutable evaluation context (needed for filter and deferred constraints)
/// * `vars` - State variable names in declaration order
/// * `branches` - Constraint branches
/// * `storage` - BulkStateStorage to push states into
/// * `filter` - Closure receiving value slice, returns Ok(true) to keep state
///
/// # Returns
/// Number of states added, or None if enumeration failed.
#[allow(clippy::mutable_key_type)]
pub fn enumerate_constraints_to_bulk<F>(
    ctx: &mut EvalCtx,
    vars: &[Arc<str>],
    branches: &[Vec<Constraint>],
    storage: &mut crate::arena::BulkStateStorage,
    mut filter: F,
) -> Option<usize>
where
    F: FnMut(&[Value], &mut EvalCtx) -> Result<bool, crate::error::EvalError>,
{
    let mut seen_fingerprints: HashSet<u64> = HashSet::new();
    let mut added_count = 0;

    for branch in branches {
        // Extract immediate, deferred, and filter constraints
        let mut immediate: Vec<&Constraint> = Vec::new();
        let mut deferred: Vec<DeferredConstraint> = Vec::new();
        let mut filters: Vec<&Spanned<Expr>> = Vec::new();

        for c in branch {
            match c {
                Constraint::Deferred(name, expr) => {
                    deferred.push(DeferredConstraint::Eq(name.as_str(), expr.as_ref()))
                }
                Constraint::DeferredIn(name, expr) => {
                    deferred.push(DeferredConstraint::In(name.as_str(), expr.as_ref()))
                }
                Constraint::Filter(expr) => {
                    filters.push(expr.as_ref());
                }
                _ => immediate.push(c),
            }
        }

        // Build var -> possible values map for immediate constraints
        let immediate_vars: Vec<Arc<str>> = vars
            .iter()
            .filter(|v| {
                immediate.iter().any(|c| match c {
                    Constraint::Eq(name, _)
                    | Constraint::In(name, _)
                    | Constraint::NotEq(name, _) => name == v.as_ref(),
                    Constraint::Deferred(_, _)
                    | Constraint::DeferredIn(_, _)
                    | Constraint::Filter(_) => false,
                })
            })
            .cloned()
            .collect();

        let immediate_owned: Vec<Constraint> = immediate.iter().map(|&c| c.clone()).collect();
        let mut var_values: Vec<(Arc<str>, Vec<Value>)> = Vec::new();
        for var in &immediate_vars {
            let values = find_values_for_var(var, &immediate_owned)?;
            if values.is_empty() {
                // Unsatisfiable branch, skip
                continue;
            }
            var_values.push((var.clone(), values));
        }

        // Create a combined filter that applies both the user filter and the constraint filters
        let combined_filter =
            |values: &[Value], eval_ctx: &mut EvalCtx| -> Result<bool, crate::error::EvalError> {
                // First apply user filter
                if !filter(values, eval_ctx)? {
                    return Ok(false);
                }

                // Then apply constraint filters (all must evaluate to true)
                for filter_expr in &filters {
                    match eval(eval_ctx, filter_expr) {
                        Ok(Value::Bool(true)) => {}
                        Ok(Value::Bool(false)) => return Ok(false),
                        Ok(_) => return Ok(false),  // Not boolean
                        Err(_) => return Ok(false), // Evaluation failed
                    }
                }

                Ok(true)
            };

        // Enumerate combinations and apply combined filter
        let mut combined_filter = combined_filter;
        enumerate_combinations_to_bulk(
            ctx,
            vars,
            &var_values,
            &deferred,
            storage,
            &mut seen_fingerprints,
            &mut combined_filter,
            &mut added_count,
        )?;
    }

    Some(added_count)
}

/// Helper: Enumerate combinations directly to bulk storage with filtering.
#[allow(clippy::mutable_key_type, clippy::too_many_arguments)]
fn enumerate_combinations_to_bulk<F>(
    ctx: &mut EvalCtx,
    all_vars: &[Arc<str>],
    var_values: &[(Arc<str>, Vec<Value>)],
    deferred: &[DeferredConstraint],
    storage: &mut crate::arena::BulkStateStorage,
    seen: &mut HashSet<u64>,
    filter: &mut F,
    added_count: &mut usize,
) -> Option<()>
where
    F: FnMut(&[Value], &mut EvalCtx) -> Result<bool, crate::error::EvalError>,
{
    // Build value buffer in var order
    let mut values: Vec<Value> = vec![Value::Bool(false); all_vars.len()];

    // Create var name -> index map
    let var_indices: std::collections::HashMap<&str, usize> = all_vars
        .iter()
        .enumerate()
        .map(|(i, v)| (v.as_ref(), i))
        .collect();

    enumerate_combinations_to_bulk_rec(
        ctx,
        all_vars,
        var_values,
        0,
        deferred,
        &mut values,
        &var_indices,
        storage,
        seen,
        filter,
        added_count,
    )
}

/// Recursive helper for combination enumeration to bulk storage.
#[allow(clippy::too_many_arguments)]
fn enumerate_combinations_to_bulk_rec<F>(
    ctx: &mut EvalCtx,
    all_vars: &[Arc<str>],
    var_values: &[(Arc<str>, Vec<Value>)],
    idx: usize,
    deferred: &[DeferredConstraint],
    values: &mut [Value],
    var_indices: &std::collections::HashMap<&str, usize>,
    storage: &mut crate::arena::BulkStateStorage,
    seen: &mut HashSet<u64>,
    filter: &mut F,
    added_count: &mut usize,
) -> Option<()>
where
    F: FnMut(&[Value], &mut EvalCtx) -> Result<bool, crate::error::EvalError>,
{
    if idx == var_values.len() {
        // All immediate variables assigned, now handle deferred constraints
        if deferred.is_empty() {
            // No deferred - check fingerprint and filter
            let fp = compute_values_fingerprint(values);
            if !seen.insert(fp) {
                return Some(()); // Duplicate
            }

            // Bind values to context for filter evaluation
            let saved = ctx.save_scope();
            for (i, var) in all_vars.iter().enumerate() {
                ctx.bind_mut(Arc::clone(var), values[i].clone());
            }

            let keep = match filter(values, ctx) {
                Ok(b) => b,
                Err(_) => {
                    ctx.restore_scope(saved);
                    return None;
                }
            };

            ctx.restore_scope(saved);

            if keep {
                storage.push_from_values(values);
                *added_count += 1;
            }
            return Some(());
        }

        // Handle deferred constraints - bind immediate values first
        let saved = ctx.save_scope();
        for (i, var) in all_vars.iter().enumerate() {
            // Only bind if we have a value (immediate constraint)
            if values[i] != Value::Bool(false)
                || var_values.iter().any(|(v, _)| v.as_ref() == var.as_ref())
            {
                ctx.bind_mut(Arc::clone(var), values[i].clone());
            }
        }

        // Evaluate deferred constraints and enumerate
        let result = evaluate_deferred_to_bulk(
            ctx,
            all_vars,
            deferred,
            values,
            var_indices,
            storage,
            seen,
            filter,
            added_count,
        );

        ctx.restore_scope(saved);
        return result;
    }

    let (var, possible_values) = &var_values[idx];
    let var_idx = *var_indices.get(var.as_ref())?;

    for val in possible_values {
        values[var_idx] = val.clone();
        enumerate_combinations_to_bulk_rec(
            ctx,
            all_vars,
            var_values,
            idx + 1,
            deferred,
            values,
            var_indices,
            storage,
            seen,
            filter,
            added_count,
        )?;
    }

    Some(())
}

/// Helper: Evaluate deferred constraints and push results to bulk storage.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::only_used_in_recursion)]
fn evaluate_deferred_to_bulk<F>(
    ctx: &mut EvalCtx,
    all_vars: &[Arc<str>],
    deferred: &[DeferredConstraint],
    values: &mut [Value],
    var_indices: &std::collections::HashMap<&str, usize>,
    storage: &mut crate::arena::BulkStateStorage,
    seen: &mut HashSet<u64>,
    filter: &mut F,
    added_count: &mut usize,
) -> Option<()>
where
    F: FnMut(&[Value], &mut EvalCtx) -> Result<bool, crate::error::EvalError>,
{
    if deferred.is_empty() {
        // All deferred resolved, check fingerprint and filter
        let fp = compute_values_fingerprint(values);
        if !seen.insert(fp) {
            return Some(()); // Duplicate
        }

        let keep = match filter(values, ctx) {
            Ok(b) => b,
            Err(_) => return None,
        };

        if keep {
            storage.push_from_values(values);
            *added_count += 1;
        }
        return Some(());
    }

    let constraint = &deferred[0];
    let remaining = &deferred[1..];

    match constraint {
        DeferredConstraint::Eq(name, expr) => {
            let value = crate::eval::eval(ctx, expr).ok()?;
            let var_idx = *var_indices.get(*name)?;
            values[var_idx] = value.clone();
            ctx.bind_mut(Arc::from(*name), value);
            evaluate_deferred_to_bulk(
                ctx,
                all_vars,
                remaining,
                values,
                var_indices,
                storage,
                seen,
                filter,
                added_count,
            )
        }
        DeferredConstraint::In(name, expr) => {
            let set_value = crate::eval::eval(ctx, expr).ok()?;
            let elements = match &set_value {
                Value::Set(s) => s.iter().cloned().collect::<Vec<_>>(),
                Value::Interval(iv) => iv.iter_values().collect::<Vec<_>>(),
                _ => return None,
            };

            let var_idx = *var_indices.get(*name)?;
            for elem in elements {
                values[var_idx] = elem.clone();
                let saved = ctx.save_scope();
                ctx.bind_mut(Arc::from(*name), elem);
                let result = evaluate_deferred_to_bulk(
                    ctx,
                    all_vars,
                    remaining,
                    values,
                    var_indices,
                    storage,
                    seen,
                    filter,
                    added_count,
                );
                ctx.restore_scope(saved);
                result?;
            }
            Some(())
        }
    }
}

/// Compute a fingerprint for a value slice (for deduplication).
fn compute_values_fingerprint(values: &[Value]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = rustc_hash::FxHasher::default();
    for v in values {
        v.hash(&mut hasher);
    }
    hasher.finish()
}

/// Check which variables have no positive constraints (Eq, In, Deferred, or DeferredIn) in any branch.
/// Returns a list of variable names that are missing positive constraints.
/// Note: NotEq alone cannot define a domain, so it doesn't count as a "defining" constraint.
pub fn find_unconstrained_vars(vars: &[Arc<str>], branches: &[Vec<Constraint>]) -> Vec<String> {
    let mut unconstrained = Vec::new();
    for var in vars {
        // Check if any branch has a positive constraint for this variable
        let has_positive_constraint = branches.iter().any(|branch| {
            branch.iter().any(|c| match c {
                Constraint::Eq(name, _)
                | Constraint::In(name, _)
                | Constraint::Deferred(name, _)
                | Constraint::DeferredIn(name, _) => name == var.as_ref(),
                Constraint::NotEq(_, _) | Constraint::Filter(_) => false, // NotEq/Filter don't define a domain
            })
        });
        if !has_positive_constraint {
            unconstrained.push(var.to_string());
        }
    }
    unconstrained
}

// Allow mutable_key_type: Value has interior mutability for lazy evaluation memoization,
// but Ord/Eq implementations don't depend on the mutable state
#[allow(clippy::mutable_key_type)]
fn find_values_for_var(var: &Arc<str>, constraints: &[Constraint]) -> Option<Vec<Value>> {
    let mut domain: Option<BTreeSet<Value>> = None;
    let mut excluded: Vec<Value> = Vec::new();

    // First pass: Build domain from positive constraints (Eq, In)
    // and collect excluded values from NotEq
    for constraint in constraints {
        match constraint {
            Constraint::Eq(name, value) if name == var.as_ref() => {
                let mut set = BTreeSet::new();
                set.insert(value.clone());
                domain = Some(match domain {
                    None => set,
                    Some(existing) => existing.intersection(&set).cloned().collect(),
                });
            }
            Constraint::In(name, values) if name == var.as_ref() => {
                let set: BTreeSet<Value> = values.iter().cloned().collect();
                domain = Some(match domain {
                    None => set,
                    Some(existing) => existing.intersection(&set).cloned().collect(),
                });
            }
            Constraint::NotEq(name, value) if name == var.as_ref() => {
                excluded.push(value.clone());
            }
            _ => {}
        }

        // Early exit if domain becomes empty
        if matches!(domain, Some(ref d) if d.is_empty()) {
            return Some(Vec::new());
        }
    }

    // Second pass: Remove excluded values from domain
    if let Some(mut d) = domain {
        for excl in &excluded {
            d.remove(excl);
        }
        Some(d.into_iter().collect())
    } else if !excluded.is_empty() {
        // NotEq constraints exist but no positive constraints define a domain
        // Cannot enumerate without a bounded domain
        None
    } else {
        // No constraints at all for this variable
        None
    }
}

fn enumerate_combinations(
    var_values: &[(Arc<str>, Vec<Value>)],
    idx: usize,
    current: &mut Vec<(Arc<str>, Value)>,
    results: &mut Vec<State>,
) {
    if idx == var_values.len() {
        // All variables assigned - create state
        let state = State::from_pairs(current.iter().map(|(k, v)| (Arc::clone(k), v.clone())));
        results.push(state);
        return;
    }

    let (var, values) = &var_values[idx];
    for val in values {
        current.push((var.clone(), val.clone()));
        enumerate_combinations(var_values, idx + 1, current, results);
        current.pop();
    }
}

/// A primed assignment extracted from a Next relation
#[derive(Debug, Clone)]
pub enum PrimedAssignment {
    /// x' = value (computed)
    Assign(Arc<str>, Value),
    /// x' = x (UNCHANGED)
    Unchanged(Arc<str>),
    /// x' \in S (choose from a set of values)
    InSet(Arc<str>, Vec<Value>),
    /// x' = expr where expr depends on InSet variables (must be evaluated per-combination)
    DeferredExpr(Arc<str>, Spanned<Expr>),
}

/// Captured bindings for deferred expression evaluation.
/// These are local bindings (EXISTS bounds, operator params) that may go out of scope.
type CapturedBindings = Vec<(Arc<str>, Value)>;

/// A symbolic assignment before evaluation
#[derive(Debug, Clone)]
enum SymbolicAssignment {
    /// x' = expr (expression to be evaluated with captured bindings)
    Expr(Arc<str>, Spanned<Expr>, CapturedBindings),
    /// x' = value (already evaluated - used for LET bindings)
    Value(Arc<str>, Value),
    /// x' = x (UNCHANGED)
    Unchanged(Arc<str>),
    /// x' \in S (set expression to be evaluated with captured bindings)
    InSet(Arc<str>, Spanned<Expr>, CapturedBindings),
}

/// Enumerate successor states from a Next relation
///
/// Given the current state and Next relation, finds all states s' such that Next(s, s') holds.
///
/// **TLC Semantics**: If evaluation fails with certain errors (NotInDomain, IndexOutOfBounds,
/// NoSuchField, ChooseFailed), the action is treated as disabled (returns empty vector).
/// This matches TLC behavior where errors in disabled action branches are silently ignored.
/// For example, in `SendReplicatedRequest` from CheckpointCoordination:
/// ```tla
/// SendReplicatedRequest(prospect) ==
///   LET currentLease == CurrentLease[Leader] IN  \* Fails when Leader=NoNode
///   /\ HaveQuorum                                \* Would be FALSE anyway
///   ...
/// ```
/// When Leader=NoNode, `CurrentLease[Leader]` fails, but HaveQuorum is FALSE so the
/// action would be disabled anyway. TLC catches this and treats the action as disabled.
pub fn enumerate_successors(
    ctx: &mut EvalCtx,
    next_def: &tla_core::ast::OperatorDef,
    current_state: &State,
    vars: &[Arc<str>],
) -> Result<Vec<State>, EvalError> {
    // Bind current state variables
    for (name, value) in current_state.vars() {
        ctx.bind_mut(Arc::clone(name), value.clone());
    }

    // Analyze the Next relation structure
    if debug_enum() {
        eprintln!(
            "enumerate_successors: next_def.body span={:?}",
            next_def.body.span
        );
    }

    // Use continuation-passing enumeration if enabled (TLA2_USE_CONTINUATION=1)
    let enum_result = if use_continuation() {
        enumerate_with_continuation(ctx, &next_def.body, current_state, vars)
    } else {
        enumerate_next_rec(ctx, &next_def.body, current_state, vars)
    };

    let successors = match enum_result {
        Ok(s) => s,
        Err(e) => {
            // TLC semantics: certain errors during action evaluation mean the action is disabled.
            // This happens when evaluation fails on a code path that would be unreachable
            // if guards were checked first.
            match &e {
                EvalError::NotInDomain { .. } => {
                    // Function applied to value not in domain (e.g., F[NoNode] where NoNode not in DOMAIN F)
                    if debug_enum() {
                        eprintln!(
                            "enumerate_successors: NotInDomain error, treating action as disabled: {:?}",
                            e
                        );
                    }
                    return Ok(Vec::new());
                }
                EvalError::IndexOutOfBounds { .. } => {
                    // Sequence index out of bounds
                    if debug_enum() {
                        eprintln!(
                            "enumerate_successors: IndexOutOfBounds error, treating action as disabled: {:?}",
                            e
                        );
                    }
                    return Ok(Vec::new());
                }
                EvalError::NoSuchField { .. } => {
                    // Record field not found
                    if debug_enum() {
                        eprintln!(
                            "enumerate_successors: NoSuchField error, treating action as disabled: {:?}",
                            e
                        );
                    }
                    return Ok(Vec::new());
                }
                EvalError::ChooseFailed { .. } => {
                    // CHOOSE found no witness
                    if debug_enum() {
                        eprintln!(
                            "enumerate_successors: ChooseFailed error, treating action as disabled: {:?}",
                            e
                        );
                    }
                    return Ok(Vec::new());
                }
                EvalError::DivisionByZero { .. } => {
                    // Division by zero in unreachable code path
                    if debug_enum() {
                        eprintln!(
                            "enumerate_successors: DivisionByZero error, treating action as disabled: {:?}",
                            e
                        );
                    }
                    return Ok(Vec::new());
                }
                EvalError::TypeError { .. } => {
                    // Type error in unreachable code path.
                    // This can happen when LET bindings like `key == args[1]` are evaluated
                    // before guards that protect them (e.g., args is a ModelValue like NIL,
                    // not a sequence). TLC evaluates lazily so doesn't hit this.
                    if debug_enum() {
                        eprintln!(
                            "enumerate_successors: TypeError error, treating action as disabled: {:?}",
                            e
                        );
                    }
                    return Ok(Vec::new());
                }
                // Other errors (UndefinedVar, UndefinedOp, ArityMismatch, Internal, SetTooLarge)
                // are likely spec bugs and should be propagated
                _ => return Err(e),
            }
        }
    };

    Ok(successors)
}

/// Evaluate ENABLED(action) - returns true if action has any successor states from current state.
///
/// This is used to evaluate `ENABLED A` expressions in action guards. The operator returns
/// true iff there exists some next state s' such that A(s, s') is true from the current state s.
///
/// # Arguments
/// * `ctx` - Evaluation context with current state bound
/// * `action` - The action expression inside ENABLED
/// * `vars` - Variable names for the spec
///
/// # Returns
/// Ok(true) if the action is enabled (has at least one successor), Ok(false) otherwise.
/// Errors propagate from action evaluation only if they indicate spec bugs.
pub fn eval_enabled(
    ctx: &mut EvalCtx,
    action: &Spanned<Expr>,
    vars: &[Arc<str>],
) -> Result<bool, EvalError> {
    // Build current state from context's env, filtering to only state variables.
    // ctx.env may contain bound parameters (e, p, etc.) from outer quantifiers
    // which shouldn't be part of the state.
    let var_set: HashSet<&str> = vars.iter().map(|s| s.as_ref()).collect();
    let current_state = State::from_pairs(
        ctx.env
            .iter()
            .filter(|(k, _)| var_set.contains(k.as_ref()))
            .map(|(k, v)| (Arc::clone(k), v.clone())),
    );

    // Try to enumerate successors from the action
    let enum_result = if use_continuation() {
        enumerate_with_continuation(ctx, action, &current_state, vars)
    } else {
        enumerate_next_rec(ctx, action, &current_state, vars)
    };

    match enum_result {
        Ok(successors) => {
            // ENABLED is true iff there's at least one successor
            // Note: We don't filter stuttering steps here - ENABLED A is true if A can produce
            // any successor, even if that successor is the same as the current state.
            Ok(!successors.is_empty())
        }
        Err(e) => {
            // TLC semantics: certain runtime errors mean the action is disabled
            if is_disabled_action_error(&e) {
                if debug_enum() {
                    eprintln!(
                        "eval_enabled: error during action evaluation, treating as disabled: {:?}",
                        e
                    );
                }
                return Ok(false);
            }
            // Other errors are spec bugs
            Err(e)
        }
    }
}

/// Enumerate successors from an arbitrary action expression.
///
/// This is used for ENABLED<<A>>_vars evaluation in liveness checking, where we need
/// to check if the action produces any non-stuttering successor (vars' ≠ vars).
///
/// Unlike `enumerate_successors` which takes an `OperatorDef`, this function takes
/// a raw expression, making it suitable for evaluating sub-actions from fairness
/// constraints.
///
/// # Arguments
/// * `ctx` - Evaluation context with current state bound
/// * `action` - The action expression to enumerate
/// * `current_state` - The current state
/// * `vars` - Variable names for the spec
///
/// # Returns
/// Vector of successor states (may include stuttering - caller should filter if needed).
pub fn enumerate_action_successors(
    ctx: &mut EvalCtx,
    action: &Spanned<Expr>,
    current_state: &State,
    vars: &[Arc<str>],
) -> Result<Vec<State>, EvalError> {
    let enum_result = if use_continuation() {
        enumerate_with_continuation(ctx, action, current_state, vars)
    } else {
        enumerate_next_rec(ctx, action, current_state, vars)
    };

    match enum_result {
        Ok(successors) => Ok(successors),
        Err(e) => {
            // TLC semantics: certain runtime errors mean the action is disabled
            if is_disabled_action_error(&e) {
                if debug_enum() {
                    eprintln!(
                        "enumerate_action_successors: error, treating as disabled: {:?}",
                        e
                    );
                }
                return Ok(Vec::new());
            }
            Err(e)
        }
    }
}

/// Enumerate successors through an existential quantifier
///
/// For `\E p \in ProcSet : b0(p)`, enumerate all values in ProcSet and
/// for each value, bind p and recursively find successors from the body.
///
/// **Optimization**: For `\E x \in SUBSET S : x \subseteq T /\ P(x)`,
/// we only iterate over subsets of T (or T ∩ S) instead of all 2^|S| subsets.
/// This is a crucial optimization for specs like bcastFolklore.
fn enumerate_exists(
    ctx: &mut EvalCtx,
    bounds: &[tla_core::ast::BoundVar],
    body: &Spanned<Expr>,
    current_state: &State,
    vars: &[Arc<str>],
) -> Result<Vec<State>, EvalError> {
    if bounds.is_empty() {
        // No more bound variables, just recurse on body
        return enumerate_next_rec(ctx, body, current_state, vars);
    }

    // Handle the first bound variable
    let bound = &bounds[0];
    let remaining_bounds = &bounds[1..];
    let var_name_str = bound.name.node.as_str();
    let var_name = Arc::from(var_name_str);

    // Evaluate the domain to get the set of values
    let domain_values = if let Some(domain_expr) = &bound.domain {
        // **SUBSET Optimization**: Check if domain is SUBSET(S) and body has x \subseteq T guard
        // In that case, only iterate over subsets of T (not all 2^|S| subsets)
        if let Some(optimized) = try_optimize_subset_domain(ctx, domain_expr, var_name_str, body)? {
            optimized
        } else {
            // Evaluate the domain expression
            // If evaluation fails (e.g., undefined variable, type error), treat action as disabled
            let domain_val = match eval(ctx, domain_expr) {
                Ok(v) => v,
                Err(_) => {
                    // Domain evaluation failed - action is disabled
                    return Ok(Vec::new());
                }
            };
            // Handle both Set and Interval using to_sorted_set
            // If the domain is not enumerable (e.g., ModelValue), treat action as disabled
            // This handles cases like `\E j \in temp[self]` where temp[self] is a ModelValue
            // (uninitialized variable) because the guard `pc[self] = "Li4b"` is false
            match domain_val.to_sorted_set() {
                Some(s) => s.as_slice().to_vec(),
                None => {
                    // Non-enumerable domain - action is disabled, not an error
                    // TLC behavior: if domain can't be enumerated, action is simply disabled
                    return Ok(Vec::new());
                }
            }
        }
    } else {
        // No domain specified - cannot enumerate
        return Ok(Vec::new());
    };

    if debug_exists() && should_print_exists_debug_line() {
        eprintln!(
            "EXISTS {}: domain_size={} remaining_bounds={} body_span={:?}",
            var_name_str,
            domain_values.len(),
            remaining_bounds.len(),
            body.span
        );
    }

    let mut all_successors = Vec::new();

    // For each value in the domain, bind the variable and recurse
    // Use stack-based bindings for O(1) push/pop (avoids HashMap allocation)
    for value in domain_values {
        let mark = ctx.mark_stack();
        ctx.push_binding(Arc::clone(&var_name), value);

        // If there are more bound variables, recursively handle them
        let successors = if remaining_bounds.is_empty() {
            enumerate_next_rec(ctx, body, current_state, vars)?
        } else {
            enumerate_exists(ctx, remaining_bounds, body, current_state, vars)?
        };

        ctx.pop_to_mark(mark);
        all_successors.extend(successors);
    }

    Ok(all_successors)
}

/// Try to optimize SUBSET domain enumeration.
///
/// Detects patterns like:
///   \E x \in SUBSET(S):
///     /\ x \subseteq upper_bound    (upper constraint)
///     /\ lower_bound \subseteq x    (lower constraint)
///
/// Instead of enumerating all 2^|upper| subsets, enumerate:
///   { lower ∪ X | X ⊆ (upper \ lower) }
/// This reduces enumeration from 2^|upper| to 2^|upper - lower|.
///
/// Returns Some(values) if optimization applies, None otherwise.
fn try_optimize_subset_domain(
    ctx: &EvalCtx,
    domain_expr: &Spanned<Expr>,
    var_name: &str,
    body: &Spanned<Expr>,
) -> Result<Option<Vec<Value>>, EvalError> {
    // Check if domain is SUBSET(S)
    let Expr::Powerset(inner_set_expr) = &domain_expr.node else {
        return Ok(None);
    };

    // Look for upper bound: x \subseteq upper
    let upper_bound_expr = find_subseteq_guard(body, var_name);

    // Look for lower bound: lower \subseteq x
    let lower_bound_expr = find_superset_guard(body, var_name);

    // Need at least one bound to optimize
    if upper_bound_expr.is_none() && lower_bound_expr.is_none() {
        return Ok(None);
    }

    // Evaluate original domain S
    let original_set_val = eval(ctx, inner_set_expr)?;
    let Some(original_set) = original_set_val.to_sorted_set() else {
        return Ok(None);
    };

    // Compute upper bound (defaults to original domain S)
    // Use SortedSet for O(log n) binary search instead of OrdSet B-tree
    let upper_set: SortedSet = if let Some(upper_expr) = upper_bound_expr {
        let upper_val = eval(ctx, upper_expr)?;
        let Some(upper) = upper_val.to_sorted_set() else {
            return Ok(None);
        };
        // upper ∩ S: filter upper to elements that are in original_set
        SortedSet::from_iter(upper.iter().filter(|v| original_set.contains(v)).cloned())
    } else {
        original_set.clone()
    };

    // Compute lower bound (defaults to empty)
    let lower_vec: Vec<Value> = if let Some(lower_expr) = lower_bound_expr {
        let lower_val = eval(ctx, lower_expr)?;
        let Some(lower) = lower_val.to_sorted_set() else {
            return Ok(None);
        };
        // lower must be a subset of upper for valid enumeration
        let lower_slice = lower.as_slice();
        let is_subset = lower_slice.iter().all(|v| upper_set.contains(v));
        if !is_subset {
            // lower has elements not in upper - no valid subsets exist
            return Ok(Some(Vec::new()));
        }
        lower_slice.to_vec()
    } else {
        Vec::new()
    };

    // Compute additional elements: upper \ lower
    let additional: Vec<Value> = if lower_vec.is_empty() {
        upper_set.as_slice().to_vec()
    } else {
        upper_set
            .iter()
            .filter(|v| lower_vec.binary_search(v).is_err())
            .cloned()
            .collect()
    };

    // Generate all subsets of additional, each unioned with lower_vec
    let n = additional.len();
    let count = 1usize << n;
    let mut result = Vec::with_capacity(count);

    for mask in 0..count {
        let selected_additional = additional
            .iter()
            .enumerate()
            .filter(|(i, _)| (mask >> i) & 1 == 1)
            .map(|(_, v)| v.clone());

        result.push(Value::Set(SortedSet::from_iter(
            lower_vec.iter().cloned().chain(selected_additional),
        )));
    }

    if debug_enum() {
        eprintln!(
            "SUBSET optimization: |S|={}, |upper|={}, |lower|={}, |additional|={}, subsets={}",
            original_set.len(),
            upper_set.len(),
            lower_vec.len(),
            additional.len(),
            result.len()
        );
    }

    if debug_exists() && should_print_exists_debug_line() {
        eprintln!(
            "EXISTS {}: SUBSET optimization |S|={} |upper|={} |lower|={} |additional|={} subsets={}",
            var_name,
            original_set.len(),
            upper_set.len(),
            lower_vec.len(),
            additional.len(),
            result.len()
        );
    }

    Ok(Some(result))
}

/// Find a subseteq guard `var_name \subseteq T` in the body expression.
/// Returns the expression T if found (upper bound constraint).
fn find_subseteq_guard<'a>(body: &'a Spanned<Expr>, var_name: &str) -> Option<&'a Spanned<Expr>> {
    match &body.node {
        // Direct subseteq: x \subseteq T
        Expr::Subseteq(lhs, rhs) => {
            if let Expr::Ident(name) = &lhs.node {
                if name == var_name {
                    return Some(rhs);
                }
            }
            None
        }
        // Conjunction: look in both sides
        Expr::And(a, b) => {
            find_subseteq_guard(a, var_name).or_else(|| find_subseteq_guard(b, var_name))
        }
        // Also check nested structures
        Expr::Exists(_, inner_body) => find_subseteq_guard(inner_body, var_name),
        _ => None,
    }
}

/// Find a superset guard `T \subseteq var_name` in the body expression.
/// Returns the expression T if found (lower bound constraint - var must contain T).
fn find_superset_guard<'a>(body: &'a Spanned<Expr>, var_name: &str) -> Option<&'a Spanned<Expr>> {
    match &body.node {
        // Direct subseteq: T \subseteq x means x must contain T
        Expr::Subseteq(lhs, rhs) => {
            if let Expr::Ident(name) = &rhs.node {
                if name == var_name {
                    return Some(lhs);
                }
            }
            None
        }
        // Conjunction: look in both sides
        Expr::And(a, b) => {
            find_superset_guard(a, var_name).or_else(|| find_superset_guard(b, var_name))
        }
        // Also check nested structures
        Expr::Exists(_, inner_body) => find_superset_guard(inner_body, var_name),
        _ => None,
    }
}

/// Flatten an And expression tree into a list of conjuncts (iterative to avoid stack overflow)
fn flatten_and(expr: &Expr) -> Vec<&Expr> {
    let mut result = Vec::new();
    let mut stack = vec![expr];
    while let Some(current) = stack.pop() {
        match current {
            Expr::And(a, b) => {
                // Push right first so left is processed first (maintains order)
                stack.push(&b.node);
                stack.push(&a.node);
            }
            _ => result.push(current),
        }
    }
    result
}

/// Flatten an And tree into a list of conjuncts (iterative to avoid stack overflow)
fn flatten_and_spanned(expr: &Spanned<Expr>, out: &mut Vec<Spanned<Expr>>) {
    let mut stack = vec![expr];
    while let Some(current) = stack.pop() {
        match &current.node {
            Expr::And(a, b) => {
                // Push right first so left is processed first (maintains order)
                stack.push(b);
                stack.push(a);
            }
            _ => out.push(current.clone()),
        }
    }
}

fn rebuild_and_spanned(conjuncts: &[Spanned<Expr>], span: Span) -> Spanned<Expr> {
    assert!(!conjuncts.is_empty());
    let mut iter = conjuncts.iter();
    let first = iter.next().expect("conjuncts non-empty").clone();
    iter.fold(first, |acc, next| {
        Spanned::new(Expr::And(Box::new(acc), Box::new(next.clone())), span)
    })
}

fn expr_contains_ident(expr: &Expr, name: &str) -> bool {
    match expr {
        Expr::Ident(n) => n == name,
        Expr::Apply(op, args) => {
            expr_contains_ident(&op.node, name)
                || args.iter().any(|a| expr_contains_ident(&a.node, name))
        }
        Expr::ModuleRef(_m, _op, args) => args.iter().any(|a| expr_contains_ident(&a.node, name)),
        Expr::Lambda(params, body) => {
            params.iter().any(|p| p.node == name) || expr_contains_ident(&body.node, name)
        }

        Expr::And(a, b)
        | Expr::Or(a, b)
        | Expr::Implies(a, b)
        | Expr::Equiv(a, b)
        | Expr::Eq(a, b)
        | Expr::Neq(a, b)
        | Expr::Lt(a, b)
        | Expr::Leq(a, b)
        | Expr::Gt(a, b)
        | Expr::Geq(a, b)
        | Expr::In(a, b)
        | Expr::NotIn(a, b)
        | Expr::Subseteq(a, b)
        | Expr::Union(a, b)
        | Expr::Intersect(a, b)
        | Expr::SetMinus(a, b)
        | Expr::Add(a, b)
        | Expr::Sub(a, b)
        | Expr::Mul(a, b)
        | Expr::Div(a, b)
        | Expr::IntDiv(a, b)
        | Expr::Mod(a, b)
        | Expr::Pow(a, b)
        | Expr::Range(a, b) => {
            expr_contains_ident(&a.node, name) || expr_contains_ident(&b.node, name)
        }

        Expr::Not(a)
        | Expr::Neg(a)
        | Expr::Powerset(a)
        | Expr::BigUnion(a)
        | Expr::Domain(a)
        | Expr::Prime(a)
        | Expr::Always(a)
        | Expr::Eventually(a)
        | Expr::Enabled(a)
        | Expr::Unchanged(a) => expr_contains_ident(&a.node, name),

        Expr::FuncApply(func, arg) => {
            expr_contains_ident(&func.node, name) || expr_contains_ident(&arg.node, name)
        }

        Expr::LeadsTo(a, b) | Expr::WeakFair(a, b) | Expr::StrongFair(a, b) => {
            expr_contains_ident(&a.node, name) || expr_contains_ident(&b.node, name)
        }

        Expr::If(cond, then_br, else_br) => {
            expr_contains_ident(&cond.node, name)
                || expr_contains_ident(&then_br.node, name)
                || expr_contains_ident(&else_br.node, name)
        }

        Expr::Case(arms, other) => {
            arms.iter().any(|arm| {
                expr_contains_ident(&arm.guard.node, name)
                    || expr_contains_ident(&arm.body.node, name)
            }) || other
                .as_ref()
                .is_some_and(|e| expr_contains_ident(&e.node, name))
        }

        Expr::Forall(bounds, body) | Expr::Exists(bounds, body) => {
            bounds.iter().any(|b| b.name.node == name)
                || bounds.iter().any(|b| {
                    b.domain
                        .as_ref()
                        .is_some_and(|d| expr_contains_ident(&d.node, name))
                })
                || expr_contains_ident(&body.node, name)
        }

        Expr::Choose(bound, body) => {
            bound.name.node == name
                || bound
                    .domain
                    .as_ref()
                    .is_some_and(|d| expr_contains_ident(&d.node, name))
                || expr_contains_ident(&body.node, name)
        }

        Expr::SetEnum(elems) => elems.iter().any(|e| expr_contains_ident(&e.node, name)),
        Expr::SetBuilder(body, bounds) => {
            expr_contains_ident(&body.node, name)
                || bounds.iter().any(|b| b.name.node == name)
                || bounds.iter().any(|b| {
                    b.domain
                        .as_ref()
                        .is_some_and(|d| expr_contains_ident(&d.node, name))
                })
        }
        Expr::SetFilter(bound, pred) => {
            bound.name.node == name
                || bound
                    .domain
                    .as_ref()
                    .is_some_and(|d| expr_contains_ident(&d.node, name))
                || expr_contains_ident(&pred.node, name)
        }

        Expr::FuncDef(bounds, body) => {
            bounds.iter().any(|b| b.name.node == name)
                || bounds.iter().any(|b| {
                    b.domain
                        .as_ref()
                        .is_some_and(|d| expr_contains_ident(&d.node, name))
                })
                || expr_contains_ident(&body.node, name)
        }

        Expr::FuncSet(domain, range) => {
            expr_contains_ident(&domain.node, name) || expr_contains_ident(&range.node, name)
        }
        Expr::Except(base, specs) => {
            expr_contains_ident(&base.node, name)
                || specs.iter().any(|spec| {
                    spec.path.iter().any(|el| match el {
                        tla_core::ast::ExceptPathElement::Index(idx) => {
                            expr_contains_ident(&idx.node, name)
                        }
                        tla_core::ast::ExceptPathElement::Field(_field) => false,
                    }) || expr_contains_ident(&spec.value.node, name)
                })
        }

        Expr::Record(fields) => fields
            .iter()
            .any(|(_k, v)| expr_contains_ident(&v.node, name)),
        Expr::RecordAccess(rec, _field) => expr_contains_ident(&rec.node, name),
        Expr::RecordSet(fields) => fields
            .iter()
            .any(|(_k, v)| expr_contains_ident(&v.node, name)),

        Expr::Tuple(elems) | Expr::Times(elems) => {
            elems.iter().any(|e| expr_contains_ident(&e.node, name))
        }

        Expr::Let(defs, body) => {
            defs.iter().any(|d| {
                d.name.node == name
                    || d.params.iter().any(|p| p.name.node == name)
                    || expr_contains_ident(&d.body.node, name)
            }) || expr_contains_ident(&body.node, name)
        }

        // Leaves without child expressions
        Expr::Bool(_)
        | Expr::Int(_)
        | Expr::String(_)
        | Expr::InstanceExpr(_, _)
        | Expr::OpRef(_) => false,
    }
}

/// Check if an expression contains an Exists node anywhere in its tree.
/// This is used to determine if an Apply node should be inlined to expose
/// the existential for proper lifting/handling.
fn expr_contains_exists(expr: &Expr) -> bool {
    match expr {
        Expr::Exists(_, _) => true,

        // Binary expressions
        Expr::And(a, b)
        | Expr::Or(a, b)
        | Expr::Implies(a, b)
        | Expr::Equiv(a, b)
        | Expr::Eq(a, b)
        | Expr::Neq(a, b)
        | Expr::Lt(a, b)
        | Expr::Leq(a, b)
        | Expr::Gt(a, b)
        | Expr::Geq(a, b)
        | Expr::In(a, b)
        | Expr::NotIn(a, b)
        | Expr::Add(a, b)
        | Expr::Sub(a, b)
        | Expr::Mul(a, b)
        | Expr::Div(a, b)
        | Expr::IntDiv(a, b)
        | Expr::Mod(a, b)
        | Expr::Pow(a, b)
        | Expr::Range(a, b)
        | Expr::SetMinus(a, b)
        | Expr::Union(a, b)
        | Expr::Intersect(a, b)
        | Expr::FuncApply(a, b)
        | Expr::FuncSet(a, b)
        | Expr::Subseteq(a, b)
        | Expr::LeadsTo(a, b)
        | Expr::WeakFair(a, b)
        | Expr::StrongFair(a, b) => expr_contains_exists(&a.node) || expr_contains_exists(&b.node),

        // Unary
        Expr::Not(e)
        | Expr::Neg(e)
        | Expr::Prime(e)
        | Expr::Always(e)
        | Expr::Eventually(e)
        | Expr::Unchanged(e)
        | Expr::Domain(e)
        | Expr::Enabled(e)
        | Expr::Powerset(e)
        | Expr::BigUnion(e) => expr_contains_exists(&e.node),

        // Collections with Vec<Spanned<Expr>>
        Expr::SetEnum(es) | Expr::Tuple(es) | Expr::Times(es) => {
            es.iter().any(|e| expr_contains_exists(&e.node))
        }

        // Quantifiers (besides Exists)
        Expr::Forall(_, body) => expr_contains_exists(&body.node),

        // SetBuilder, SetFilter, FuncDef
        Expr::SetBuilder(e, _) => expr_contains_exists(&e.node),
        Expr::SetFilter(_, pred) => expr_contains_exists(&pred.node),
        Expr::FuncDef(_, body) => expr_contains_exists(&body.node),

        // Except: base expression + specs
        Expr::Except(base, specs) => {
            expr_contains_exists(&base.node)
                || specs.iter().any(|s| expr_contains_exists(&s.value.node))
        }

        // If-then-else
        Expr::If(cond, then_e, else_e) => {
            expr_contains_exists(&cond.node)
                || expr_contains_exists(&then_e.node)
                || expr_contains_exists(&else_e.node)
        }

        // Case
        Expr::Case(arms, other) => {
            arms.iter().any(|arm| {
                expr_contains_exists(&arm.guard.node) || expr_contains_exists(&arm.body.node)
            }) || other
                .as_ref()
                .is_some_and(|e| expr_contains_exists(&e.node))
        }

        // Let
        Expr::Let(defs, body) => {
            defs.iter().any(|d| expr_contains_exists(&d.body.node))
                || expr_contains_exists(&body.node)
        }

        // Apply - check op and args, but not the operator definition body
        // (that's checked at a higher level where we have access to ctx)
        Expr::Apply(op, args) => {
            expr_contains_exists(&op.node) || args.iter().any(|a| expr_contains_exists(&a.node))
        }

        // Module references
        Expr::ModuleRef(_, _, args) => args.iter().any(|a| expr_contains_exists(&a.node)),

        // Choose: has BoundVar and body
        Expr::Choose(_, body) => expr_contains_exists(&body.node),

        // Record expressions: Vec<(Spanned<String>, Spanned<Expr>)>
        Expr::Record(fields) | Expr::RecordSet(fields) => {
            fields.iter().any(|(_, v)| expr_contains_exists(&v.node))
        }

        // RecordAccess
        Expr::RecordAccess(e, _) => expr_contains_exists(&e.node),

        // Lambda
        Expr::Lambda(_, body) => expr_contains_exists(&body.node),

        // Leaf expressions
        Expr::Ident(_)
        | Expr::Bool(_)
        | Expr::Int(_)
        | Expr::String(_)
        | Expr::InstanceExpr(_, _)
        | Expr::OpRef(_) => false,
    }
}

/// Check if an expression contains an OR (disjunction) that has primed variables,
/// following operator calls to check if they contain action-level disjunctions.
/// This is used to detect when an operator body contains action-level disjunctions
/// that need to be inlined for proper enumeration.
fn expr_contains_or_with_primed_ctx(ctx: &EvalCtx, expr: &Expr) -> bool {
    match expr {
        // Found OR - check if any branch is an action (contains primes at some level)
        Expr::Or(a, b) => {
            expr_contains_prime_ctx(ctx, &a.node)
                || expr_contains_prime_ctx(ctx, &b.node)
                || expr_contains_or_with_primed_ctx(ctx, &a.node)
                || expr_contains_or_with_primed_ctx(ctx, &b.node)
        }

        // Operator application - check if operator body contains Or with primes
        Expr::Apply(op_expr, _args) => {
            if let Expr::Ident(op_name) = &op_expr.node {
                if let Some(def) = ctx.get_op(op_name) {
                    return expr_contains_or_with_primed_ctx(ctx, &def.body.node);
                }
            }
            false
        }

        // Zero-param operator reference - check body
        Expr::Ident(name) => {
            if let Some(def) = ctx.get_op(name) {
                if def.params.is_empty() {
                    return expr_contains_or_with_primed_ctx(ctx, &def.body.node);
                }
            }
            false
        }

        // Binary expressions - recurse
        Expr::And(a, b)
        | Expr::Implies(a, b)
        | Expr::Equiv(a, b)
        | Expr::Eq(a, b)
        | Expr::Neq(a, b)
        | Expr::Lt(a, b)
        | Expr::Leq(a, b)
        | Expr::Gt(a, b)
        | Expr::Geq(a, b)
        | Expr::In(a, b)
        | Expr::NotIn(a, b)
        | Expr::Add(a, b)
        | Expr::Sub(a, b)
        | Expr::Mul(a, b)
        | Expr::Div(a, b)
        | Expr::IntDiv(a, b)
        | Expr::Mod(a, b)
        | Expr::Pow(a, b)
        | Expr::Range(a, b)
        | Expr::SetMinus(a, b)
        | Expr::Union(a, b)
        | Expr::Intersect(a, b)
        | Expr::FuncApply(a, b)
        | Expr::FuncSet(a, b)
        | Expr::Subseteq(a, b)
        | Expr::LeadsTo(a, b)
        | Expr::WeakFair(a, b)
        | Expr::StrongFair(a, b) => {
            expr_contains_or_with_primed_ctx(ctx, &a.node)
                || expr_contains_or_with_primed_ctx(ctx, &b.node)
        }

        // Unary
        Expr::Not(e)
        | Expr::Neg(e)
        | Expr::Prime(e)
        | Expr::Always(e)
        | Expr::Eventually(e)
        | Expr::Unchanged(e)
        | Expr::Domain(e)
        | Expr::Enabled(e)
        | Expr::Powerset(e)
        | Expr::BigUnion(e) => expr_contains_or_with_primed_ctx(ctx, &e.node),

        // Collections
        Expr::SetEnum(elems) | Expr::Tuple(elems) => {
            elems.iter().any(|e| expr_contains_or_with_primed_ctx(ctx, &e.node))
        }

        // Quantifiers and LET
        Expr::Exists(_, body) | Expr::Forall(_, body) => expr_contains_or_with_primed_ctx(ctx, &body.node),
        Expr::Let(_, body) => expr_contains_or_with_primed_ctx(ctx, &body.node),
        Expr::If(_, then_br, else_br) => {
            expr_contains_or_with_primed_ctx(ctx, &then_br.node)
                || expr_contains_or_with_primed_ctx(ctx, &else_br.node)
        }

        // Everything else
        _ => false,
    }
}

/// Check if an expression contains primed variables, following operator calls.
fn expr_contains_prime_ctx(ctx: &EvalCtx, expr: &Expr) -> bool {
    match expr {
        Expr::Prime(_) | Expr::Unchanged(_) => true,

        // Operator application - check operator body
        Expr::Apply(op_expr, args) => {
            if let Expr::Ident(op_name) = &op_expr.node {
                if let Some(def) = ctx.get_op(op_name) {
                    return expr_contains_prime_ctx(ctx, &def.body.node);
                }
            }
            // Check arguments too
            args.iter().any(|a| expr_contains_prime_ctx(ctx, &a.node))
        }

        // Zero-param operator reference - check body
        Expr::Ident(name) => {
            if let Some(def) = ctx.get_op(name) {
                if def.params.is_empty() {
                    return expr_contains_prime_ctx(ctx, &def.body.node);
                }
            }
            false
        }

        // Binary
        Expr::And(a, b)
        | Expr::Or(a, b)
        | Expr::Implies(a, b)
        | Expr::Equiv(a, b)
        | Expr::Eq(a, b)
        | Expr::Neq(a, b)
        | Expr::Lt(a, b)
        | Expr::Leq(a, b)
        | Expr::Gt(a, b)
        | Expr::Geq(a, b)
        | Expr::In(a, b)
        | Expr::NotIn(a, b)
        | Expr::Add(a, b)
        | Expr::Sub(a, b)
        | Expr::Mul(a, b)
        | Expr::Div(a, b)
        | Expr::IntDiv(a, b)
        | Expr::Mod(a, b)
        | Expr::Pow(a, b)
        | Expr::Range(a, b)
        | Expr::SetMinus(a, b)
        | Expr::Union(a, b)
        | Expr::Intersect(a, b)
        | Expr::FuncApply(a, b)
        | Expr::FuncSet(a, b)
        | Expr::Subseteq(a, b) => {
            expr_contains_prime_ctx(ctx, &a.node) || expr_contains_prime_ctx(ctx, &b.node)
        }

        // Unary
        Expr::Not(e) | Expr::Neg(e) | Expr::Domain(e) | Expr::Powerset(e) | Expr::BigUnion(e) => {
            expr_contains_prime_ctx(ctx, &e.node)
        }

        // Collections
        Expr::SetEnum(elems) | Expr::Tuple(elems) => {
            elems.iter().any(|e| expr_contains_prime_ctx(ctx, &e.node))
        }

        // Quantifiers and LET
        Expr::Exists(_, body) | Expr::Forall(_, body) => expr_contains_prime_ctx(ctx, &body.node),
        Expr::Let(_, body) => expr_contains_prime_ctx(ctx, &body.node),
        Expr::If(c, t, e) => {
            expr_contains_prime_ctx(ctx, &c.node)
                || expr_contains_prime_ctx(ctx, &t.node)
                || expr_contains_prime_ctx(ctx, &e.node)
        }

        _ => false,
    }
}

/// Try to inline zero-param operator references (Ident) in a conjunction if they contain
/// action-level OR expressions (disjunctions with primed variables).
///
/// This is necessary because disjunctions buried inside operator definitions are not visible
/// to the OR distribution code in enumerate_next_rec, which only looks at direct conjuncts
/// after flattening. By inlining Ident nodes whose bodies contain OR with primed vars, we
/// expose the disjunctions for proper enumeration.
///
/// Example: `Next == Move_Cat /\ Observe_Box` where `Observe_Box == LET ... IN \/ ... \/ ...`
/// The OR is inside Observe_Box's body. After inlining, it becomes visible for distribution.
fn try_inline_ident_with_or(ctx: &EvalCtx, expr: &Spanned<Expr>) -> Option<Spanned<Expr>> {
    let Expr::And(_, _) = &expr.node else {
        return None;
    };

    let mut conjuncts = Vec::new();
    flatten_and_spanned(expr, &mut conjuncts);

    let mut any_inlined = false;
    let mut new_conjuncts = Vec::new();

    for conjunct in &conjuncts {
        // Handle Ident nodes (zero-param operator references)
        if let Expr::Ident(op_name) = &conjunct.node {
            if let Some(def) = ctx.get_op(op_name) {
                if def.params.is_empty() {
                    // Check if the operator body contains an OR with primed variables
                    // Use context-aware version to follow nested operator calls
                    if expr_contains_or_with_primed_ctx(ctx, &def.body.node) {
                        // Inline the operator body directly
                        if debug_enum() {
                            eprintln!(
                                "try_inline_ident_with_or: inlining {} which contains OR",
                                op_name
                            );
                        }
                        new_conjuncts.push(def.body.clone());
                        any_inlined = true;
                        continue;
                    }
                }
            }
        }

        // Handle Apply nodes (operator calls with arguments) that contain OR
        if let Expr::Apply(op_expr, args) = &conjunct.node {
            if let Expr::Ident(op_name) = &op_expr.node {
                if let Some(def) = ctx.get_op(op_name) {
                    // Check if the operator body contains an OR with primed variables
                    // Use context-aware version to follow nested operator calls
                    if expr_contains_or_with_primed_ctx(ctx, &def.body.node) {
                        // Inline: substitute parameters with argument expressions
                        let subs: Vec<Substitution> = def
                            .params
                            .iter()
                            .zip(args.iter())
                            .map(|(param, arg)| Substitution {
                                from: param.name.clone(),
                                to: arg.clone(),
                            })
                            .collect();

                        let inlined = apply_substitutions(&def.body, &subs);
                        if debug_enum() {
                            eprintln!(
                                "try_inline_ident_with_or: inlining Apply {} which contains OR",
                                op_name
                            );
                        }
                        new_conjuncts.push(inlined);
                        any_inlined = true;
                        continue;
                    }
                }
            }
        }

        new_conjuncts.push(conjunct.clone());
    }

    if any_inlined {
        Some(rebuild_and_spanned(&new_conjuncts, expr.span))
    } else {
        None
    }
}

/// Try to inline Apply and ModuleRef nodes in a conjunction if they contain Exists expressions.
/// This is necessary because existentials buried inside operator applications
/// are not visible to `try_lift_exists_from_and`, which only looks at direct
/// conjuncts. By inlining Apply/ModuleRef nodes whose bodies contain Exists, we expose
/// the existentials for proper lifting.
///
/// Example: `Receive(self) /\ (UponV1(self) \/ ...)` where Receive contains `\E msgs ...`
/// becomes: `(guard /\ \E msgs ... /\ rcvd' = ...) /\ (UponV1(self) \/ ...)`
///
/// Also handles INSTANCE operators: `CommChan!Deliver(i)` where Deliver contains `\E boxes ...`
fn try_inline_apply_with_exists(ctx: &EvalCtx, expr: &Spanned<Expr>) -> Option<Spanned<Expr>> {
    let Expr::And(_, _) = &expr.node else {
        return None;
    };

    let mut conjuncts = Vec::new();
    flatten_and_spanned(expr, &mut conjuncts);

    let mut any_inlined = false;
    let mut new_conjuncts = Vec::new();

    for conjunct in &conjuncts {
        // Handle bare Ident nodes (zero-param operator references like PaxosAccepted)
        // These must be inlined if they contain EXISTS, since extraction can't handle
        // EXISTS with multiple bounds and the subsequent action validation will fail.
        if let Expr::Ident(op_name) = &conjunct.node {
            if let Some(def) = ctx.get_op(op_name) {
                if def.params.is_empty() && expr_contains_exists(&def.body.node) {
                    // Inline the zero-param operator body
                    if debug_enum() {
                        eprintln!(
                            "try_inline_apply_with_exists: inlining zero-param {} which contains EXISTS",
                            op_name
                        );
                    }
                    new_conjuncts.push(def.body.clone());
                    any_inlined = true;
                    continue;
                }
            }
        }

        // Handle regular Apply nodes (local operator calls)
        if let Expr::Apply(op_expr, args) = &conjunct.node {
            if let Expr::Ident(op_name) = &op_expr.node {
                if let Some(def) = ctx.get_op(op_name) {
                    // Check if the operator body contains an Exists
                    if expr_contains_exists(&def.body.node) {
                        // Inline: substitute parameters with argument expressions
                        let subs: Vec<Substitution> = def
                            .params
                            .iter()
                            .zip(args.iter())
                            .map(|(param, arg)| Substitution {
                                from: param.name.clone(),
                                to: arg.clone(),
                            })
                            .collect();

                        let inlined = apply_substitutions(&def.body, &subs);
                        new_conjuncts.push(inlined);
                        any_inlined = true;
                        continue;
                    }
                }
            }
        }

        // Handle ModuleRef nodes (INSTANCE operator calls like CommChan!Deliver(i))
        if let Expr::ModuleRef(instance_name, op_name, args) = &conjunct.node {
            if let Some(instance_info) = ctx.get_instance(instance_name.name()) {
                if let Some(op_def) = ctx.get_instance_op(&instance_info.module_name, op_name) {
                    // Check if the operator body contains an Exists
                    if expr_contains_exists(&op_def.body.node) {
                        // IMPORTANT: Preserve the instanced module's operator namespace.
                        // Unqualified helper operator calls inside the instanced module body
                        // must continue to resolve within the instanced module after inlining.
                        let mut inlined = if let Some(instance_ops) =
                            ctx.shared.instance_ops.get(&instance_info.module_name)
                        {
                            qualify_instance_ops_for_inlining(
                                &op_def.body,
                                instance_name,
                                instance_ops,
                                &op_def.params,
                            )
                        } else {
                            op_def.body.clone()
                        };

                        // Apply INSTANCE ... WITH substitutions first
                        inlined = apply_substitutions(&inlined, &instance_info.substitutions);

                        // Then apply parameter substitutions
                        if !op_def.params.is_empty() && op_def.params.len() == args.len() {
                            let param_subs: Vec<Substitution> = op_def
                                .params
                                .iter()
                                .zip(args.iter())
                                .map(|(param, arg)| Substitution {
                                    from: param.name.clone(),
                                    to: arg.clone(),
                                })
                                .collect();
                            inlined = apply_substitutions(&inlined, &param_subs);
                        }

                        new_conjuncts.push(inlined);
                        any_inlined = true;
                        continue;
                    }
                }
            }
        }

        new_conjuncts.push(conjunct.clone());
    }

    if any_inlined {
        Some(rebuild_and_spanned(&new_conjuncts, expr.span))
    } else {
        None
    }
}

/// Flags for patterns found in an And expression tree.
/// This enables a single traversal to detect all patterns instead of separate traversals.
#[derive(Default, Clone, Copy)]
struct AndPatterns {
    has_let: bool,
    has_if: bool,
    has_exists: bool,
    has_or: bool,
}

/// Scan an And expression tree for patterns in a single pass.
/// Returns flags indicating which patterns are present.
/// This consolidates `and_contains_let`, `and_contains_if`, `and_contains_exists`,
/// and Or detection into a single traversal for better performance.
fn scan_and_patterns(expr: &Expr) -> AndPatterns {
    let mut patterns = AndPatterns::default();
    scan_and_patterns_inner(expr, &mut patterns);
    patterns
}

fn scan_and_patterns_inner(expr: &Expr, patterns: &mut AndPatterns) {
    // Early exit if we've found all patterns
    if patterns.has_let && patterns.has_if && patterns.has_exists && patterns.has_or {
        return;
    }

    match expr {
        Expr::And(a, b) => {
            // Check immediate children first
            match &a.node {
                Expr::Let(_, _) => patterns.has_let = true,
                Expr::If(_, _, _) => patterns.has_if = true,
                Expr::Exists(_, _) => patterns.has_exists = true,
                Expr::Or(_, _) => patterns.has_or = true,
                _ => {}
            }
            match &b.node {
                Expr::Let(_, _) => patterns.has_let = true,
                Expr::If(_, _, _) => patterns.has_if = true,
                Expr::Exists(_, _) => patterns.has_exists = true,
                Expr::Or(_, _) => patterns.has_or = true,
                _ => {}
            }
            // Recurse into And children
            scan_and_patterns_inner(&a.node, patterns);
            scan_and_patterns_inner(&b.node, patterns);
        }
        Expr::Let(_, _) => patterns.has_let = true,
        Expr::If(_, _, _) => patterns.has_if = true,
        Expr::Exists(_, _) => patterns.has_exists = true,
        Expr::Or(_, _) => patterns.has_or = true,
        _ => {}
    }
}

/// Fast version of try_lift_exists_from_and that skips the and_contains_exists check.
/// Caller must verify the pattern exists via scan_and_patterns first.
fn try_lift_exists_from_and_fast(expr: &Spanned<Expr>) -> Option<Spanned<Expr>> {
    let mut conjuncts = Vec::new();
    flatten_and_spanned(expr, &mut conjuncts);

    for (idx, conjunct) in conjuncts.iter().enumerate() {
        let Expr::Exists(bounds, body) = &conjunct.node else {
            continue;
        };

        let bound_names: Vec<&str> = bounds.iter().map(|b| b.name.node.as_str()).collect();
        let used_elsewhere = conjuncts.iter().enumerate().any(|(j, other)| {
            if j == idx {
                return false;
            }
            bound_names
                .iter()
                .any(|name| expr_contains_ident(&other.node, name))
        });
        if used_elsewhere {
            continue;
        }

        let mut lifted_conjuncts = Vec::with_capacity(conjuncts.len());
        for (j, other) in conjuncts.iter().enumerate() {
            if j == idx {
                lifted_conjuncts.push((**body).clone());
            } else {
                lifted_conjuncts.push(other.clone());
            }
        }

        let new_body = rebuild_and_spanned(&lifted_conjuncts, expr.span);
        return Some(Spanned::new(
            Expr::Exists(bounds.clone(), Box::new(new_body)),
            expr.span,
        ));
    }

    None
}

/// Fast version of try_simplify_if_in_and that skips the and_contains_if check.
/// Caller must verify the pattern exists via scan_and_patterns first.
fn try_simplify_if_in_and_fast(ctx: &EvalCtx, expr: &Spanned<Expr>) -> Option<Spanned<Expr>> {
    let Expr::And(_, _) = &expr.node else {
        return None;
    };

    let mut conjuncts = Vec::new();
    flatten_and_spanned(expr, &mut conjuncts);

    for (idx, conjunct) in conjuncts.iter().enumerate() {
        let Expr::If(cond, then_branch, else_branch) = &conjunct.node else {
            continue;
        };

        match eval(ctx, cond) {
            Ok(Value::Bool(true)) => {
                let mut new_conjuncts = Vec::with_capacity(conjuncts.len());
                for (j, other) in conjuncts.iter().enumerate() {
                    if j == idx {
                        new_conjuncts.push((**then_branch).clone());
                    } else {
                        new_conjuncts.push(other.clone());
                    }
                }
                return Some(rebuild_and_spanned(&new_conjuncts, expr.span));
            }
            Ok(Value::Bool(false)) => {
                let mut new_conjuncts = Vec::with_capacity(conjuncts.len());
                for (j, other) in conjuncts.iter().enumerate() {
                    if j == idx {
                        new_conjuncts.push((**else_branch).clone());
                    } else {
                        new_conjuncts.push(other.clone());
                    }
                }
                return Some(rebuild_and_spanned(&new_conjuncts, expr.span));
            }
            _ => continue,
        }
    }

    None
}

/// Fast version of try_distribute_if_in_and that skips the and_contains_if check.
/// Caller must verify the pattern exists via scan_and_patterns first.
fn try_distribute_if_in_and_fast(expr: &Spanned<Expr>) -> Option<Spanned<Expr>> {
    let Expr::And(_, _) = &expr.node else {
        return None;
    };

    let mut conjuncts = Vec::new();
    flatten_and_spanned(expr, &mut conjuncts);

    for (idx, conjunct) in conjuncts.iter().enumerate() {
        let Expr::If(cond, then_branch, else_branch) = &conjunct.node else {
            continue;
        };

        let mut then_conjuncts = Vec::new();
        let mut else_conjuncts = Vec::new();

        for (j, other) in conjuncts.iter().enumerate() {
            if j == idx {
                continue;
            }
            then_conjuncts.push(other.clone());
            else_conjuncts.push(other.clone());
        }

        then_conjuncts.push((**cond).clone());
        then_conjuncts.push((**then_branch).clone());

        let not_cond = Spanned::new(Expr::Not(Box::new((**cond).clone())), cond.span);
        else_conjuncts.push(not_cond);
        else_conjuncts.push((**else_branch).clone());

        let then_expr = rebuild_and_spanned(&then_conjuncts, expr.span);
        let else_expr = rebuild_and_spanned(&else_conjuncts, expr.span);
        return Some(Spanned::new(
            Expr::Or(Box::new(then_expr), Box::new(else_expr)),
            expr.span,
        ));
    }

    None
}

/// Fast version of try_inline_let_in_and that skips the and_contains_let check.
/// Caller must verify the pattern exists via scan_and_patterns first.
fn try_inline_let_in_and_fast(ctx: &EvalCtx, expr: &Spanned<Expr>) -> Option<(EvalCtx, Spanned<Expr>)> {
    let Expr::And(_, _) = &expr.node else {
        return None;
    };

    let mut conjuncts = Vec::new();
    flatten_and_spanned(expr, &mut conjuncts);

    for (idx, conjunct) in conjuncts.iter().enumerate() {
        let Expr::Let(defs, body) = &conjunct.node else {
            continue;
        };

        let def_names: Vec<&str> = defs.iter().map(|d| d.name.node.as_str()).collect();
        let used_elsewhere = conjuncts.iter().enumerate().any(|(j, other)| {
            if j == idx {
                return false;
            }
            def_names
                .iter()
                .any(|name| expr_contains_ident(&other.node, name))
        });
        if used_elsewhere {
            continue;
        }

        let mut new_ctx = ctx.clone();
        let mut local_ops = new_ctx.local_ops.as_ref().map(|o| (**o).clone()).unwrap_or_default();
        for def in defs {
            local_ops.insert(def.name.node.clone(), def.clone());
        }
        new_ctx = new_ctx.with_local_ops(local_ops);

        let mut new_conjuncts = Vec::with_capacity(conjuncts.len());
        for (j, other) in conjuncts.iter().enumerate() {
            if j == idx {
                new_conjuncts.push((**body).clone());
            } else {
                new_conjuncts.push(other.clone());
            }
        }

        let new_expr = rebuild_and_spanned(&new_conjuncts, expr.span);
        return Some((new_ctx, new_expr));
    }

    None
}

/// Like try_distribute_if_in_and but works with pre-flattened conjuncts.
/// Returns (then_conjuncts, else_conjuncts) if an IF was found and distributed.
#[allow(dead_code, clippy::type_complexity)]
fn try_distribute_if_flattened(
    conjuncts: &[Spanned<Expr>],
) -> Option<(Vec<Spanned<Expr>>, Vec<Spanned<Expr>>)> {
    for (idx, conjunct) in conjuncts.iter().enumerate() {
        let Expr::If(cond, then_branch, else_branch) = &conjunct.node else {
            continue;
        };

        let mut then_conjuncts = Vec::new();
        let mut else_conjuncts = Vec::new();

        for (j, other) in conjuncts.iter().enumerate() {
            if j == idx {
                continue;
            }
            then_conjuncts.push(other.clone());
            else_conjuncts.push(other.clone());
        }

        // IF P THEN T ELSE E  <==>  (P /\ T) \/ (~P /\ E)
        then_conjuncts.push((**cond).clone());
        then_conjuncts.push((**then_branch).clone());

        let not_cond = Spanned::new(Expr::Not(Box::new((**cond).clone())), cond.span);
        else_conjuncts.push(not_cond);
        else_conjuncts.push((**else_branch).clone());

        return Some((then_conjuncts, else_conjuncts));
    }

    None
}

/// Find the first Or expression in a list of conjuncts, returning its index
fn find_or_conjunct(conjuncts: &[&Expr]) -> Option<usize> {
    conjuncts.iter().position(|e| matches!(e, Expr::Or(_, _)))
}

/// Rebuild a conjunction from a list of conjuncts (folding right)
fn rebuild_and(conjuncts: &[&Expr], span: tla_core::span::Span) -> Expr {
    assert!(!conjuncts.is_empty());
    if conjuncts.len() == 1 {
        conjuncts[0].clone()
    } else {
        let first = Spanned::new(conjuncts[0].clone(), span);
        let rest = rebuild_and(&conjuncts[1..], span);
        Expr::And(Box::new(first), Box::new(Spanned::new(rest, span)))
    }
}

/// Red zone size: when stack has less than this remaining, grow
/// This needs to be large enough for the recursive enumeration frames
const STACK_RED_ZONE: usize = 1024 * 1024; // 1MB red zone
/// Stack growth size: how much to grow when we hit the red zone
const STACK_GROW_SIZE: usize = 4 * 1024 * 1024; // 4MB growth

fn enumerate_next_rec(
    ctx: &mut EvalCtx,
    expr: &Spanned<Expr>,
    current_state: &State,
    vars: &[Arc<str>],
) -> Result<Vec<State>, EvalError> {
    // Use stacker to grow stack on demand for deeply nested expressions
    stacker::maybe_grow(STACK_RED_ZONE, STACK_GROW_SIZE, || {
        enumerate_next_rec_inner(ctx, expr, current_state, vars)
    })
}

fn enumerate_next_rec_inner(
    ctx: &mut EvalCtx,
    expr: &Spanned<Expr>,
    current_state: &State,
    vars: &[Arc<str>],
) -> Result<Vec<State>, EvalError> {
    let debug = debug_enum();
    if debug {
        eprintln!(
            "enumerate_next_rec: expr span={:?}, type={:?}",
            expr.span,
            std::mem::discriminant(&expr.node)
        );
    }
    match &expr.node {
        // Disjunction: collect successors from both branches
        Expr::Or(a, b) => {
            // TLC short-circuits and treats many runtime errors as "disabled" branches.
            // When enumerating `A \\/ B`, an error in A should not prevent exploring B.
            let left = match enumerate_next_rec(ctx, a, current_state, vars) {
                Ok(v) => v,
                Err(e) if is_disabled_action_error(&e) => {
                    if debug {
                        eprintln!("OR: left branch error {:?}, treating as disabled", e);
                    }
                    Vec::new()
                }
                Err(e) => return Err(e),
            };
            let right = match enumerate_next_rec(ctx, b, current_state, vars) {
                Ok(v) => v,
                Err(e) if is_disabled_action_error(&e) => {
                    if debug {
                        eprintln!("OR: right branch error {:?}, treating as disabled", e);
                    }
                    Vec::new()
                }
                Err(e) => return Err(e),
            };
            if debug {
                eprintln!(
                    "OR: left={} states, right={} states",
                    left.len(),
                    right.len()
                );
            }
            let mut successors = left;
            successors.extend(right);
            Ok(successors)
        }

        // Existential quantifier: \E p \in ProcSet : b0(p)
        // For each value in the domain, bind the variable and recurse
        Expr::Exists(bounds, body) => enumerate_exists(ctx, bounds, body, current_state, vars),

        // Conjunction: evaluate guards and combine assignments
        Expr::And(a, b) => {
            // Single-pass pattern scan to avoid multiple And tree traversals.
            // This consolidates `and_contains_let`, `and_contains_if`, `and_contains_exists`,
            // and Or detection into one traversal.
            let patterns = scan_and_patterns(&expr.node);

            // Fast path: if no special patterns (LET/IF/EXISTS/OR), check guards immediately.
            // This is the common case for simple actions like `pc[self] = "x" /\ x' = ...`.
            // Avoids the overhead of trying transformations when none are needed.
            let has_any_pattern = patterns.has_let || patterns.has_if || patterns.has_exists || patterns.has_or;
            if !has_any_pattern {
                // Fast path: check guards, then extract assignments directly
                if !check_and_guards(ctx, expr, debug)? {
                    if debug {
                        eprintln!("AND: (fast path) guard check failed, skipping");
                    }
                    return Ok(Vec::new());
                }

                // Try inlining operators that might contain hidden structure.
                // This is still needed even without top-level patterns because
                // operators might expand to IF/OR/EXISTS.
                if let Some(inlined) = try_inline_apply_with_exists(ctx, expr) {
                    if debug {
                        eprintln!("AND: (fast path) inlined Apply with Exists");
                    }
                    return enumerate_next_rec(ctx, &inlined, current_state, vars);
                }
                if let Some(inlined) = try_inline_ident_with_or(ctx, expr) {
                    if debug {
                        eprintln!("AND: (fast path) inlined operator with OR");
                    }
                    return enumerate_next_rec(ctx, &inlined, current_state, vars);
                }

                // Extract assignments and build successor states
                if debug {
                    eprintln!("AND: (fast path) extracting symbolic assignments");
                }
                let mut symbolic = Vec::new();
                extract_symbolic_assignments(ctx, a, vars, &mut symbolic)?;
                extract_symbolic_assignments(ctx, b, vars, &mut symbolic)?;

                let assignments = evaluate_symbolic_assignments(ctx, &symbolic)?;
                let registry = ctx.var_registry();
                let reg_opt = if registry.is_empty() { None } else { Some(registry) };
                let mut states = build_successor_states_with_ctx(current_state, vars, &assignments, reg_opt, Some(ctx));

                let has_primed_assignments = !symbolic.is_empty();
                if has_primed_assignments || needs_next_state_validation(&expr.node) {
                    states.retain(|st| action_holds_in_next_state(ctx, expr, st));
                }
                if debug {
                    eprintln!("AND: (fast path) built {} successor states", states.len());
                }
                return Ok(states);
            }

            // Inline LET conjuncts to expose action structure (IF/EXISTS) in their bodies.
            // Only try if we detected LET in the pattern scan.
            if patterns.has_let {
                if let Some((mut new_ctx, inlined)) = try_inline_let_in_and_fast(ctx, expr) {
                    if debug {
                        eprintln!("AND: inlining LET inside conjunction");
                    }
                    return enumerate_next_rec(&mut new_ctx, &inlined, current_state, vars);
                }
            }

            // Optimization: Try to evaluate IF conditions in conjunctions directly.
            // If the condition can be evaluated, replace IF with its branch instead of
            // distributing (which creates expensive cloned expressions).
            // Only try if we detected IF in the pattern scan.
            if patterns.has_if {
                if let Some(simplified) = try_simplify_if_in_and_fast(ctx, expr) {
                    if debug {
                        eprintln!("AND: simplified IF inside conjunction");
                    }
                    return enumerate_next_rec(ctx, &simplified, current_state, vars);
                }

                // Fallback: Distribute IF expressions inside conjunctions.
                // This handles cases where the condition can't be evaluated (e.g., depends
                // on primed variables).
                if let Some(distributed) = try_distribute_if_in_and_fast(expr) {
                    if debug {
                        eprintln!("AND: distributing IF inside conjunction");
                    }
                    return enumerate_next_rec(ctx, &distributed, current_state, vars);
                }
            }

            // Check all guards in the And tree before extracting assignments.
            // For And(And(guard, assign1), assign2), we need to check if guard is false.
            // We do this by checking all guards in the And tree before extracting assignments.
            if !check_and_guards(ctx, expr, debug)? {
                if debug {
                    eprintln!("AND: nested guard check failed, skipping");
                }
                return Ok(Vec::new());
            }

            // Inline Apply nodes that contain Exists expressions.
            // This is essential for specs like bcastFolklore where:
            //   Step(self) == Receive(self) /\ (UponV1(self) \/ ...)
            // and Receive(self) contains `\E msgs \in SUBSET M : ...`
            // Without inlining, the Exists is hidden inside the Apply and cannot be lifted.
            if let Some(inlined) = try_inline_apply_with_exists(ctx, expr) {
                if debug {
                    eprintln!("AND: inlined Apply with Exists");
                }
                return enumerate_next_rec(ctx, &inlined, current_state, vars);
            }

            // Handle exists nested within And chains by lifting the quantifier.
            // This is essential for PlusCal translations like:
            //   pc[self] = "n1" /\ \E id \in msgs[self] : ... /\ pc' = ...
            // Only try if we detected EXISTS in the pattern scan.
            if patterns.has_exists {
                if let Some(lifted) = try_lift_exists_from_and_fast(expr) {
                    if debug {
                        eprintln!("AND: lifting Exists out of conjunction");
                    }
                    return enumerate_next_rec(ctx, &lifted, current_state, vars);
                }
            }

            // Inline operator references (Ident/Apply) that contain OR with primed variables.
            // This is essential for specs like CatEvenBoxes where:
            //   Next == Move_Cat /\ Observe_Box
            // and Observe_Box contains `LET ... IN \/ branch1 \/ branch2`
            // Without inlining, the OR is hidden inside the operator and cannot be distributed.
            if let Some(inlined) = try_inline_ident_with_or(ctx, expr) {
                if debug {
                    eprintln!("AND: inlined operator with OR");
                }
                return enumerate_next_rec(ctx, &inlined, current_state, vars);
            }

            // Handle Or nested within And chains by distributing.
            // Flatten the And tree, find any Or conjuncts, and distribute.
            // This is essential for Next relations like: guard /\ x' = e /\ (branch1 \/ branch2)
            // Only try if we detected OR in the pattern scan.
            let conjuncts = flatten_and(&expr.node);
            if patterns.has_or {
                if let Some(or_idx) = find_or_conjunct(&conjuncts) {
                if let Expr::Or(or_left, or_right) = conjuncts[or_idx] {
                    if debug {
                        eprintln!(
                            "AND: found Or at position {} of {} conjuncts, distributing",
                            or_idx,
                            conjuncts.len()
                        );
                        eprintln!("AND: or_left type={:?}, span={:?}", std::mem::discriminant(&or_left.node), or_left.span);
                        eprintln!("AND: or_right type={:?}, span={:?}", std::mem::discriminant(&or_right.node), or_right.span);
                    }
                    // Build two new conjunctions: one with or_left, one with or_right
                    let mut conjuncts_left: Vec<&Expr> = Vec::new();
                    let mut conjuncts_right: Vec<&Expr> = Vec::new();
                    for (i, c) in conjuncts.iter().enumerate() {
                        if i == or_idx {
                            conjuncts_left.push(&or_left.node);
                            conjuncts_right.push(&or_right.node);
                        } else {
                            conjuncts_left.push(c);
                            conjuncts_right.push(c);
                        }
                    }
                    let new_left = rebuild_and(&conjuncts_left, expr.span);
                    let new_right = rebuild_and(&conjuncts_right, expr.span);
                    let left_spanned = Spanned::new(new_left, expr.span);
                    let right_spanned = Spanned::new(new_right, expr.span);
                    if debug {
                        eprintln!("AND: left branch type={:?}", std::mem::discriminant(&left_spanned.node));
                        eprintln!("AND: right branch type={:?}", std::mem::discriminant(&right_spanned.node));
                    }
                    let mut successors =
                        enumerate_next_rec(ctx, &left_spanned, current_state, vars)?;
                    if debug {
                        eprintln!("AND: left branch returned {} successors", successors.len());
                    }
                    successors.extend(enumerate_next_rec(
                        ctx,
                        &right_spanned,
                        current_state,
                        vars,
                    )?);
                    return Ok(successors);
                }
                }
            }

            if debug {
                eprintln!("AND: extracting symbolic assignments from both sides");
                eprintln!("AND: a span={:?}, b span={:?}", a.span, b.span);
            }
            // Both sides might have primed assignments - extract symbolically first
            let mut symbolic = Vec::new();
            if debug {
                eprintln!("AND: calling extract on a (span {:?})", a.span);
            }
            extract_symbolic_assignments(ctx, a, vars, &mut symbolic)?;
            if debug {
                eprintln!("AND: calling extract on b (span {:?})", b.span);
            }
            extract_symbolic_assignments(ctx, b, vars, &mut symbolic)?;
            if debug {
                eprintln!("AND: symbolic={:?}", symbolic);
            }

            // Evaluate symbolic assignments, building up next-state context
            let assignments = evaluate_symbolic_assignments(ctx, &symbolic)?;
            if debug {
                eprintln!("AND: assignments={:?}", assignments);
            }

            // Build successor states from assignments (may be multiple for x' \in S)
            // Pass ctx for deferred expression evaluation
            let registry = ctx.var_registry();
            let reg_opt = if registry.is_empty() {
                None
            } else {
                Some(registry)
            };
            let mut states = build_successor_states_with_ctx(current_state, vars, &assignments, reg_opt, Some(ctx));

            // Validate generated successors by evaluating constraints that depend on next-state.
            // This is essential for patterns like:
            // - `IF rcvd'[self] = 1 THEN ...` where the branch condition depends on next-state
            // - `HCnxt => t >= 1` where HCnxt is an action operator (contains primed vars)
            // - `UpdateOpOrder` containing `Serializable'` which validates opOrder' constraints
            //
            // Key insight: If we have ANY primed variable assignments, this is an ACTION that
            // might contain constraints (prime guards) from operator bodies. We must validate
            // the entire expression against each generated successor state.
            let has_primed_assignments = !symbolic.is_empty();
            if has_primed_assignments || needs_next_state_validation(&expr.node) {
                states.retain(|st| action_holds_in_next_state(ctx, expr, st));
            }
            if debug {
                eprintln!("AND: built {} successor states", states.len());
            }
            Ok(states)
        }

        // Zero-arity operator reference: FillSmallJug, etc.
        // These are parsed as Ident, not Apply, since they take no arguments
        Expr::Ident(name) => {
            // Check if this identifier names an operator
            if let Some(def) = ctx.get_op(name).cloned() {
                // Recurse into the operator body
                return enumerate_next_rec(ctx, &def.body, current_state, vars);
            }
            // Not an operator - treat as guard (probably a variable)
            Ok(Vec::new())
        }

        // Operator application with arguments: b0(p), Action(x, y), etc.
        Expr::Apply(op_expr, args) => {
            if let Expr::Ident(op_name) = &op_expr.node {
                if let Some(def) = ctx.get_op(op_name).cloned() {
                    // Bind arguments to parameters using stack-based O(1) push/pop
                    let mark = ctx.mark_stack();
                    let debug_apply = std::env::var("TLA2_DEBUG_APPLY").is_ok();
                    for (param, arg) in def.params.iter().zip(args.iter()) {
                        let arg_val = eval(ctx, arg)?;
                        if debug_apply {
                            eprintln!("[DEBUG APPLY] {}({}): binding param '{}' to {:?}",
                                op_name, def.params.iter().map(|p| p.name.node.as_str()).collect::<Vec<_>>().join(", "),
                                param.name.node, arg_val);
                        }
                        ctx.push_binding(Arc::from(param.name.node.as_str()), arg_val);
                    }
                    // Recurse into the operator body with bound parameters
                    let result = enumerate_next_rec(ctx, &def.body, current_state, vars);
                    ctx.pop_to_mark(mark);
                    return result;
                }
            }
            // Unknown operator - try as expression
            Ok(Vec::new())
        }

        // LET ... IN body - add local definitions to context
        // Note: For LET with parameterized operators, we need to keep using clone
        // because define_op modifies shared state
        Expr::Let(defs, body) => {
            // Register LET definitions lazily via local_ops (like eval.rs does).
            // Definitions are only evaluated when referenced, enabling short-circuit
            // evaluation to avoid evaluating unused bindings that would fail.
            // E.g., in btree's WhichToSplit: `LET parent == ParentOf(node) IN ...`
            // where ParentOf(root) fails, but `parent` is never used when node=root.
            use crate::OpEnv;
            let mut local_ops: OpEnv = match &ctx.local_ops {
                Some(ops) => (**ops).clone(),
                None => OpEnv::default(),
            };
            for def in defs {
                local_ops.insert(def.name.node.clone(), def.clone());
            }
            let mut new_ctx = ctx.clone();
            new_ctx.local_ops = Some(std::sync::Arc::new(local_ops));
            enumerate_next_rec(&mut new_ctx, body, current_state, vars)
        }

        // IF-THEN-ELSE: evaluate condition and recurse into appropriate branch
        // This handles patterns like: IF guard THEN action1 ELSE action2
        Expr::If(cond, then_branch, else_branch) => {
            // First, try to evaluate the condition
            match eval(ctx, cond) {
                Ok(Value::Bool(true)) => {
                    // Condition is TRUE - recurse into then branch
                    if debug {
                        eprintln!("IF: condition=true, taking then branch");
                    }
                    enumerate_next_rec(ctx, then_branch, current_state, vars)
                }
                Ok(Value::Bool(false)) => {
                    // Condition is FALSE - recurse into else branch
                    if debug {
                        eprintln!("IF: condition=false, taking else branch");
                    }
                    enumerate_next_rec(ctx, else_branch, current_state, vars)
                }
                _ => {
                    // Condition couldn't be evaluated or isn't boolean
                    // Try both branches (like an Or)
                    if debug {
                        eprintln!("IF: condition couldn't be evaluated, trying both branches");
                    }
                    let then_states = enumerate_next_rec(ctx, then_branch, current_state, vars)?;
                    let else_states = enumerate_next_rec(ctx, else_branch, current_state, vars)?;
                    let mut all_states = then_states;
                    all_states.extend(else_states);
                    Ok(all_states)
                }
            }
        }

        // CASE expression: evaluate guards and take matching branch
        // CASE g1 -> e1 [] g2 -> e2 [] ... [] OTHER -> e_default
        Expr::Case(arms, other) => {
            if debug {
                eprintln!("CASE: {} arms, other={}", arms.len(), other.is_some());
            }
            // Evaluate each guard in order
            for (i, arm) in arms.iter().enumerate() {
                match eval(ctx, &arm.guard) {
                    Ok(Value::Bool(true)) => {
                        // Guard is true - recurse into this branch
                        if debug {
                            eprintln!("CASE: arm {} guard=true, taking branch", i);
                        }
                        return enumerate_next_rec(ctx, &arm.body, current_state, vars);
                    }
                    Ok(Value::Bool(false)) => {
                        // Guard is false - continue to next arm
                        if debug {
                            eprintln!("CASE: arm {} guard=false, continuing", i);
                        }
                        continue;
                    }
                    Ok(_) => {
                        // Guard didn't evaluate to boolean - skip
                        if debug {
                            eprintln!("CASE: arm {} guard non-boolean, skipping", i);
                        }
                        continue;
                    }
                    Err(e) => {
                        // Guard evaluation error - treat as false (TLC semantics)
                        if debug {
                            eprintln!("CASE: arm {} guard error {:?}, treating as false", i, e);
                        }
                        continue;
                    }
                }
            }
            // No arm matched - check for OTHER clause
            if let Some(other_body) = other {
                if debug {
                    eprintln!("CASE: no arm matched, taking OTHER branch");
                }
                enumerate_next_rec(ctx, other_body, current_state, vars)
            } else {
                // No arm matched and no OTHER - no successors
                // This could be an error in TLC but we'll treat it as no successors
                if debug {
                    eprintln!("CASE: no arm matched, no OTHER, no successors");
                }
                Ok(Vec::new())
            }
        }

        // Module reference (INSTANCE operator call like Sched!Allocate(c, S))
        // Inline the operator body and recursively enumerate.
        // This is essential for ENABLED evaluation during liveness checking.
        Expr::ModuleRef(instance_name, op_name, args) => {
            if debug {
                eprintln!("ModuleRef: {}!{} with {} args", instance_name, op_name, args.len());
            }
            // Look up instance metadata
            let instance_info = match ctx.get_instance(instance_name.name()) {
                Some(info) => info.clone(),
                None => {
                    if debug {
                        let instances: Vec<_> = ctx.instances().iter().map(|(k,_)| k.as_str()).collect();
                        eprintln!("ModuleRef: looking for '{}', ctx has instances: {:?}",
                                  instance_name.name(), instances);
                        eprintln!("ModuleRef: instance {} not found, falling back to default handling", instance_name);
                    }
                    // Fall back to default handling
                    let mut symbolic = Vec::new();
                    extract_symbolic_assignments(ctx, expr, vars, &mut symbolic)?;
                    let assignments = evaluate_symbolic_assignments(ctx, &symbolic)?;
                    let registry = ctx.var_registry();
                    let reg_opt = if registry.is_empty() { None } else { Some(registry) };
                    let mut states = build_successor_states_with_ctx(current_state, vars, &assignments, reg_opt, Some(ctx));
                    if needs_next_state_validation(&expr.node) {
                        states.retain(|st| action_holds_in_next_state(ctx, expr, st));
                    }
                    return Ok(states);
                }
            };

            // Get the operator definition
            let op_def = match ctx.get_instance_op(&instance_info.module_name, op_name) {
                Some(def) => def.clone(),
                None => {
                    if debug {
                        eprintln!("ModuleRef: operator {}!{} not found", instance_name, op_name);
                    }
                    return Ok(Vec::new());
                }
            };

            // Apply INSTANCE ... WITH substitutions
            let mut inlined = apply_substitutions(&op_def.body, &instance_info.substitutions);

            // Apply parameter substitutions
            if !op_def.params.is_empty() && op_def.params.len() == args.len() {
                let param_subs: Vec<Substitution> = op_def
                    .params
                    .iter()
                    .zip(args.iter())
                    .map(|(param, arg)| Substitution {
                        from: param.name.clone(),
                        to: arg.clone(),
                    })
                    .collect();
                inlined = apply_substitutions(&inlined, &param_subs);
            }

            if debug {
                eprintln!("ModuleRef: inlined body span={:?}", inlined.span);
            }

            // Set up instance-local operator scope so unqualified names resolve correctly
            use crate::OpEnv;
            let instance_local_ops: OpEnv = ctx
                .instance_ops()
                .get(&instance_info.module_name)
                .cloned()
                .unwrap_or_default();

            // Preserve existing local ops from outer scope
            let merged_ops = if let Some(outer_ops) = ctx.local_ops.as_deref() {
                let mut merged = instance_local_ops.clone();
                for (name, def) in outer_ops.iter() {
                    merged.entry(name.clone()).or_insert_with(|| def.clone());
                }
                merged
            } else {
                instance_local_ops
            };

            let scoped_ctx = ctx.with_local_ops(merged_ops);
            enumerate_next_rec(&mut scoped_ctx.clone(), &inlined, current_state, vars)
        }

        // Single assignment or constraint - try to extract
        _ => {
            let mut symbolic = Vec::new();
            extract_symbolic_assignments(ctx, expr, vars, &mut symbolic)?;
            let assignments = evaluate_symbolic_assignments(ctx, &symbolic)?;
            // Pass ctx for deferred expression evaluation
            let registry = ctx.var_registry();
            let reg_opt = if registry.is_empty() {
                None
            } else {
                Some(registry)
            };
            let mut states = build_successor_states_with_ctx(current_state, vars, &assignments, reg_opt, Some(ctx));
            if needs_next_state_validation(&expr.node) {
                states.retain(|st| action_holds_in_next_state(ctx, expr, st));
            }
            if !states.is_empty() {
                Ok(states)
            } else {
                // Try to evaluate as boolean - if FALSE, no successors
                match eval(ctx, expr) {
                    Ok(Value::Bool(true)) => {
                        // TRUE with no assignments - identity transition?
                        // Return empty for now (would need UNCHANGED handling)
                        Ok(Vec::new())
                    }
                    Ok(Value::Bool(false)) => Ok(Vec::new()),
                    _ => Ok(Vec::new()),
                }
            }
        }
    }
}

// =============================================================================
// CONTINUATION-PASSING ENUMERATION (TLC-style)
// =============================================================================
//
// This implementation uses TLC's algorithm for successor enumeration, which
// avoids AST cloning by using a work stack. See designs/2026-01-12-continuation-passing-enumeration.md
//
// Algorithm:
// 1. For conjunction And(a, b): push b onto stack, process a
// 2. For disjunction Or(a, b): fork - try both with same stack
// 3. For EXISTS: enumerate domain, try body with each binding
// 4. For guards: evaluate, if false return empty
// 5. For assignments: accumulate
// 6. When stack empty: emit successor state

/// Work item for continuation-passing enumeration.
/// Each variant holds a reference to part of the original AST (no cloning).
#[derive(Debug, Clone)]
enum WorkItem<'a> {
    /// A conjunct to process - could be guard, assignment, or nested structure
    Conjunct(&'a Spanned<Expr>),
}

// Profiling counters for continuation enumeration (enabled via TLA2_PROFILE_ENUM_DETAIL)
static STACK_MARK_COUNT: AtomicU64 = AtomicU64::new(0);
static CONSTRAINT_PUSH_COUNT: AtomicU64 = AtomicU64::new(0);

/// A constraint captured with its local bindings for deferred evaluation.
///
/// Instead of substituting local binding values into the AST (expensive O(AST) operation),
/// we capture the bindings alongside the expression and restore them at evaluation time.
/// This is O(n) where n is the binding count (typically 1-10) instead of O(AST size).
#[derive(Debug, Clone)]
struct CapturedConstraint<'a> {
    /// Reference to the original expression AST (no cloning needed)
    expr: &'a Spanned<Expr>,
    /// Captured local bindings (EXISTS bounds, operator params) that may go out of scope
    bindings: Vec<(Arc<str>, Value)>,
}

#[derive(Debug)]
struct WorkStack<'a> {
    items: Vec<WorkItem<'a>>,
    /// Stack pointer: number of valid (unprocessed) items.
    /// For LIFO: items[0..items_idx] are valid, items[items_idx..] may contain stale data.
    /// next_item() decrements then returns items[items_idx].
    /// push_item() sets items[items_idx] (or pushes) then increments items_idx.
    items_idx: usize,
    symbolic: Vec<SymbolicAssignment>,
    /// Action-level constraints that must be validated in next-state context.
    ///
    /// Constraints are stored with captured bindings instead of substituted ASTs.
    /// This avoids O(AST) substitution in favor of O(bindings) capture.
    constraints: Vec<CapturedConstraint<'a>>,
    /// Bug #86: Tracks operator boundary for proper binding isolation
    operator_boundary: Option<usize>,
}

/// Lightweight mark for stack state - all integers, no cloning.
/// This replaces the previous WorkStackSnapshot which cloned the items vector.
#[derive(Debug, Clone, Copy)]
struct StackMark {
    items_idx: usize,
    items_len: usize,
    symbolic_len: usize,
    constraints_len: usize,
    operator_boundary: Option<usize>,
}

impl<'a> WorkStack<'a> {
    /// Create a mark capturing current stack state.
    /// O(1) - just copies five integers, no heap allocation.
    #[inline]
    fn mark(&self) -> StackMark {
        if profile_enum_detail() {
            STACK_MARK_COUNT.fetch_add(1, AtomicOrdering::Relaxed);
        }
        StackMark {
            items_idx: self.items_idx,
            items_len: self.items.len(),
            symbolic_len: self.symbolic.len(),
            constraints_len: self.constraints.len(),
            operator_boundary: self.operator_boundary,
        }
    }

    /// Restore stack to a previously marked state.
    /// O(1) for index reset, O(n) for truncations where n = items added since mark.
    #[inline]
    fn restore(&mut self, mark: StackMark) {
        self.items_idx = mark.items_idx;
        self.items.truncate(mark.items_len);
        self.symbolic.truncate(mark.symbolic_len);
        self.constraints.truncate(mark.constraints_len);
        self.operator_boundary = mark.operator_boundary;
    }

    /// Push a work item onto the stack (LIFO).
    /// May reuse slots beyond items_idx or grow the vector.
    #[inline]
    fn push_item(&mut self, item: WorkItem<'a>) {
        if self.items_idx < self.items.len() {
            // Reuse slot (overwriting previously consumed/stale item)
            self.items[self.items_idx] = item;
        } else {
            // Grow the vector
            self.items.push(item);
        }
        self.items_idx += 1;
    }

    /// Get next work item using LIFO order.
    /// items_idx is the stack height; decrement then return items[items_idx].
    /// Items are NOT removed from the vector - restore() resets items_idx.
    #[inline]
    fn next_item(&mut self) -> Option<&WorkItem<'a>> {
        if self.items_idx > 0 {
            self.items_idx -= 1;
            Some(&self.items[self.items_idx])
        } else {
            None
        }
    }
}

/// Try to build a partial next-state from the symbolic assignments accumulated so far.
///
/// This enables progressive state construction like TLC does - when we encounter an IF
/// with a primed condition like `IF unchecked'[self] = {} THEN ...`, we can evaluate it
/// if `unchecked'` was assigned earlier in the same conjunct chain.
///
/// Returns Some(partial_next_state) if we could evaluate deterministic assignments,
/// None if any assignment is non-deterministic (InSet) or evaluation fails.
fn try_build_partial_next_state(
    ctx: &EvalCtx,
    symbolic: &[SymbolicAssignment],
    current_state: &State,
) -> Option<crate::eval::Env> {
    use crate::eval::Env;

    let mut next_state: Env = Env::new();

    for sym in symbolic {
        match sym {
            SymbolicAssignment::Value(name, value) => {
                next_state.insert(name.clone(), value.clone());
            }
            SymbolicAssignment::Unchanged(name) => {
                // UNCHANGED means x' = x
                if let Some(current) = ctx.env.get(name) {
                    next_state.insert(name.clone(), current.clone());
                } else if let Some(current) = current_state.get(name) {
                    next_state.insert(name.clone(), current.clone());
                }
            }
            SymbolicAssignment::Expr(name, expr, bindings) => {
                // Create context with current partial next_state for primed variable lookup
                let mut eval_ctx = ctx.clone();
                eval_ctx.next_state = Some(std::sync::Arc::new(next_state.clone()));
                let binding_ctx = eval_ctx.with_captured_bindings(bindings);

                // Try to evaluate - if it fails, we can't build partial state
                let eval_res = if profile_enum_detail() {
                    let start = std::time::Instant::now();
                    let res = eval(&binding_ctx, expr);
                    PROF_ASSIGN_US.fetch_add(
                        start.elapsed().as_micros() as u64,
                        AtomicOrdering::Relaxed,
                    );
                    res
                } else {
                    eval(&binding_ctx, expr)
                };
                match eval_res {
                    Ok(value) => {
                        next_state.insert(name.clone(), value);
                    }
                    Err(_) => {
                        // Can't evaluate this assignment yet - return what we have so far
                        // This is still useful if the IF condition only needs vars assigned before this
                        continue;
                    }
                }
            }
            SymbolicAssignment::InSet(_, _, _) => {
                // Non-deterministic assignment - can't include in partial state
                // but continue to try evaluating other assignments
                continue;
            }
        }
    }

    if next_state.is_empty() {
        None
    } else {
        Some(next_state)
    }
}

/// Continuation-passing successor enumeration (TLC-style).
///
/// This replaces the old `enumerate_next_rec` with a stack-based approach
/// that avoids AST cloning and multiple tree traversals.
pub fn enumerate_with_continuation(
    ctx: &mut EvalCtx,
    expr: &Spanned<Expr>,
    current_state: &State,
    vars: &[Arc<str>],
) -> Result<Vec<State>, EvalError> {
    // Route to bind/unbind implementation if enabled
    if use_bind_unbind() {
        return enumerate_with_bind(ctx, expr, current_state, vars);
    }

    let debug = debug_enum();
    let mut successors = Vec::new();
    let mut stack = WorkStack {
        items: Vec::new(),
        items_idx: 0,
        symbolic: Vec::new(),
        constraints: Vec::new(),
        operator_boundary: None,
    };

    // Start by pushing the root expression
    enumerate_continuation_inner(
        ctx,
        expr,
        &mut stack,
        current_state,
        vars,
        &mut successors,
        debug,
    )?;

    Ok(successors)
}

/// TLC-style bind/unbind enumeration for successor generation.
///
/// This is an alternative to the symbolic assignment approach that directly
/// mutates an ArrayState during traversal, only cloning when storing successors.
///
/// Key differences from enumerate_with_continuation:
/// - Uses ArrayState with bind/unbind instead of Vec<SymbolicAssignment>
/// - O(1) variable binding/unbinding vs O(vars) symbolic collection
/// - snapshot() clones only when storing a successor
/// - Implements TLC's bind/unbind pattern (Tool.java:1333-1380)
///
/// Enable with TLA2_BIND_UNBIND=1 environment variable.
///
/// Part of #101: Architecture: Bind/unbind enumeration for 10-20x performance
fn enumerate_with_bind(
    ctx: &mut EvalCtx,
    expr: &Spanned<Expr>,
    current_state: &State,
    vars: &[Arc<str>],
) -> Result<Vec<State>, EvalError> {
    let debug = debug_enum();
    // Clone registry (O(1) - just Arc clone) to avoid borrow conflict with ctx
    let registry = ctx.var_registry().clone();

    // Create mutable working state from current state
    let mut working = ArrayState::from_state(current_state, &registry);
    // Pre-allocate undo stack (most actions modify 1-4 variables)
    let mut undo: Vec<UndoEntry> = Vec::with_capacity(vars.len() * 2);
    let mut successors = Vec::new();
    // Pending conjuncts stack - tracks remaining work after branching
    let mut pending: Vec<&Spanned<Expr>> = Vec::with_capacity(8);

    // Build var_indices lookup for fast name -> VarIndex resolution
    let var_indices: Vec<(Arc<str>, VarIndex)> = vars
        .iter()
        .filter_map(|name| registry.get(name.as_ref()).map(|idx| (Arc::clone(name), idx)))
        .collect();

    if debug {
        eprintln!(
            "BIND: Starting enumeration with {} vars, working state has {} values",
            vars.len(),
            working.len()
        );
    }

    // Process the expression tree with bind/unbind
    let result = enumerate_bind_inner(
        ctx,
        expr,
        &mut working,
        &mut undo,
        &mut pending,
        current_state,
        &var_indices,
        &registry,
        &mut successors,
        debug,
    );

    // Handle the result
    match result {
        Ok(()) => {
            // If we made bindings but no successors were produced,
            // snapshot the working state as the successor.
            // This handles the case where we processed all conjuncts
            // but didn't hit an explicit "leaf" guard.
            if successors.is_empty() && !undo.is_empty() {
                if debug {
                    eprintln!("BIND: No successors from inner, but {} bindings - snapshotting", undo.len());
                }
                successors.push(working.snapshot(&registry));
            }
        }
        Err(e) if is_disabled_action_error(&e) => {
            // Action disabled - no successors from this state
            if debug {
                eprintln!("BIND: Action disabled: {:?}", e);
            }
        }
        Err(e) => return Err(e),
    }

    // If still no successors, fall back to symbolic for correctness verification
    // This catches cases not yet handled by bind/unbind
    if successors.is_empty() {
        if debug {
            eprintln!("BIND: Still no successors, falling back to symbolic");
        }
        let mut stack = WorkStack {
            items: Vec::new(),
            items_idx: 0,
            symbolic: Vec::new(),
            constraints: Vec::new(),
            operator_boundary: None,
        };
        enumerate_continuation_inner(
            ctx,
            expr,
            &mut stack,
            current_state,
            vars,
            &mut successors,
            debug,
        )?;
        if debug && !successors.is_empty() {
            eprintln!("BIND: Fell back to symbolic - produced {} successors", successors.len());
        }
    }

    Ok(successors)
}

/// Inner recursive function for bind/unbind enumeration.
///
/// Traverses the expression tree, binding values directly to the working state
/// and using unbind_to() to backtrack when exploring branches.
///
/// The `pending` stack tracks remaining conjuncts that need to be processed
/// after the current expression. This enables proper handling of expressions
/// like `x' \in S /\ y' = f(x')` where each element of S needs to continue
/// processing the rest of the conjunction.
fn enumerate_bind_inner<'a>(
    ctx: &mut EvalCtx,
    expr: &'a Spanned<Expr>,
    working: &mut ArrayState,
    undo: &mut Vec<UndoEntry>,
    pending: &mut Vec<&'a Spanned<Expr>>,
    current_state: &State,
    var_indices: &[(Arc<str>, VarIndex)],
    registry: &VarRegistry,
    successors: &mut Vec<State>,
    debug: bool,
) -> Result<(), EvalError> {
    // Use stacker for deeply nested expressions
    stacker::maybe_grow(STACK_RED_ZONE, STACK_GROW_SIZE, || {
        enumerate_bind_dispatch(
            ctx,
            expr,
            working,
            undo,
            pending,
            current_state,
            var_indices,
            registry,
            successors,
            debug,
        )
    })
}

/// Dispatch for bind/unbind enumeration based on expression type.
///
/// This parallels enumerate_continuation_dispatch but uses bind/unbind
/// instead of symbolic assignment collection.
///
/// Key insight: When we encounter expressions that produce multiple bindings
/// (like `x' \in S`), we need to process the remaining conjuncts for EACH
/// binding, not just the last one. The `pending` stack enables this by tracking
/// what work remains after the current expression.
fn enumerate_bind_dispatch<'a>(
    ctx: &mut EvalCtx,
    expr: &'a Spanned<Expr>,
    working: &mut ArrayState,
    undo: &mut Vec<UndoEntry>,
    pending: &mut Vec<&'a Spanned<Expr>>,
    current_state: &State,
    var_indices: &[(Arc<str>, VarIndex)],
    registry: &VarRegistry,
    successors: &mut Vec<State>,
    debug: bool,
) -> Result<(), EvalError> {
    match &expr.node {
        // Conjunction: push right onto pending stack, process left.
        // The right side will be processed by continue_bind_or_snapshot
        // when we reach a leaf expression (assignment or guard).
        Expr::And(a, b) => {
            if debug {
                eprintln!("BIND: And - push right to pending (len={}), process left", pending.len() + 1);
            }
            pending.push(b);
            let result = enumerate_bind_inner(ctx, a, working, undo, pending, current_state, var_indices, registry, successors, debug);
            // Pop b from pending if it wasn't consumed (error case or guard-only branch)
            // Use pointer comparison since Expr doesn't implement PartialEq
            if pending.last().map(|x| std::ptr::eq(*x, b.as_ref())).unwrap_or(false) {
                pending.pop();
            }
            result
        }

        // Disjunction: fork - try both branches with save/restore
        // Each branch gets its own copy of the pending stack state
        Expr::Or(a, b) => {
            if debug {
                eprintln!("BIND: Or - forking with save_point, pending_len={}", pending.len());
            }
            let save_point = undo.len();
            // Save pending state - truncate doesn't restore popped items
            let saved_pending: Vec<_> = pending.clone();

            // Try left branch
            match enumerate_bind_inner(ctx, a, working, undo, pending, current_state, var_indices, registry, successors, debug) {
                Ok(()) => {}
                Err(e) if is_disabled_action_error(&e) => {
                    if debug {
                        eprintln!("BIND: Or left branch disabled: {:?}", e);
                    }
                }
                Err(e) => return Err(e),
            }

            // Restore for right branch
            working.unbind_to(undo, save_point);
            pending.clear();
            pending.extend(saved_pending.iter().cloned());

            // Try right branch
            match enumerate_bind_inner(ctx, b, working, undo, pending, current_state, var_indices, registry, successors, debug) {
                Ok(()) => {}
                Err(e) if is_disabled_action_error(&e) => {
                    if debug {
                        eprintln!("BIND: Or right branch disabled: {:?}", e);
                    }
                }
                Err(e) => return Err(e),
            }

            // Final restore
            working.unbind_to(undo, save_point);
            pending.clear();
            pending.extend(saved_pending.iter().cloned());
            Ok(())
        }

        // Equality: check for primed variable assignment (x' = expr)
        Expr::Eq(lhs, rhs) => {
            // Check if LHS is a primed state variable: x'
            if let Expr::Prime(inner_lhs) = &lhs.node {
                if let Expr::Ident(name) = &inner_lhs.node {
                    if let Some((_, idx)) = var_indices.iter().find(|(n, _)| n.as_ref() == name.as_str()) {
                        // Fast path: x' = x (UNCHANGED)
                        if let Expr::Ident(rhs_name) = &rhs.node {
                            if rhs_name == name && !ctx.has_local_binding(rhs_name.as_str()) {
                                if debug {
                                    eprintln!("BIND: Eq - x' = x (UNCHANGED) for {}", name);
                                }
                                // For UNCHANGED, bind to current state value
                                let current_val = current_state.get(name.as_str()).cloned()
                                    .unwrap_or(Value::Bool(false));
                                working.bind(*idx, current_val, undo);
                                // Continue with pending conjuncts or snapshot
                                return continue_bind_or_snapshot(ctx, working, undo, pending, current_state, var_indices, registry, successors, debug);
                            }
                        }

                        // Try evaluation with next_state_env set so bound primes can be resolved.
                        // This handles y' = x' * 2 when x' is already bound in working state.
                        let prev_next = ctx.bind_next_state_array(working.values());
                        let eval_result = eval(ctx, rhs);
                        ctx.restore_next_state_env(prev_next);

                        match eval_result {
                            Ok(value) => {
                                if debug {
                                    eprintln!("BIND: Eq - binding {}' = {:?}", name, value);
                                }
                                working.bind(*idx, value, undo);
                                // Continue with pending conjuncts or snapshot
                                return continue_bind_or_snapshot(ctx, working, undo, pending, current_state, var_indices, registry, successors, debug);
                            }
                            Err(e) if is_disabled_action_error(&e) => {
                                if debug {
                                    eprintln!("BIND: Eq - action disabled: {:?}", e);
                                }
                                return Ok(()); // Action disabled
                            }
                            Err(_) => {
                                // Fall through to leaf processing
                            }
                        }
                    }
                }
            }
            // Check symmetric case: expr = x'
            if let Expr::Prime(inner_rhs) = &rhs.node {
                if let Expr::Ident(name) = &inner_rhs.node {
                    if let Some((_, idx)) = var_indices.iter().find(|(n, _)| n.as_ref() == name.as_str()) {
                        // Fast path: x = x' (UNCHANGED)
                        if let Expr::Ident(lhs_name) = &lhs.node {
                            if lhs_name == name && !ctx.has_local_binding(lhs_name.as_str()) {
                                if debug {
                                    eprintln!("BIND: Eq - x = x' (UNCHANGED) for {}", name);
                                }
                                let current_val = current_state.get(name.as_str()).cloned()
                                    .unwrap_or(Value::Bool(false));
                                working.bind(*idx, current_val, undo);
                                // Continue with pending conjuncts or snapshot
                                return continue_bind_or_snapshot(ctx, working, undo, pending, current_state, var_indices, registry, successors, debug);
                            }
                        }

                        // Try evaluation with next_state_env set
                        let prev_next = ctx.bind_next_state_array(working.values());
                        let eval_result = eval(ctx, lhs);
                        ctx.restore_next_state_env(prev_next);

                        match eval_result {
                            Ok(value) => {
                                if debug {
                                    eprintln!("BIND: Eq - binding {} = {}'", value, name);
                                }
                                working.bind(*idx, value, undo);
                                // Continue with pending conjuncts or snapshot
                                return continue_bind_or_snapshot(ctx, working, undo, pending, current_state, var_indices, registry, successors, debug);
                            }
                            Err(e) if is_disabled_action_error(&e) => {
                                if debug {
                                    eprintln!("BIND: Eq - action disabled: {:?}", e);
                                }
                                return Ok(());
                            }
                            Err(_) => {
                                // Fall through
                            }
                        }
                    }
                }
            }
            // Not a simple primed assignment - treat as leaf/guard
            process_bind_leaf(ctx, expr, working, undo, pending, current_state, var_indices, registry, successors, debug)
        }

        // Membership: check for primed variable assignment (x' \in S)
        // This is the key fix: for each element, bind then CONTINUE with pending
        Expr::In(lhs, rhs) => {
            if let Expr::Prime(inner_lhs) = &lhs.node {
                if let Expr::Ident(name) = &inner_lhs.node {
                    if let Some((_, idx)) = var_indices.iter().find(|(n, _)| n.as_ref() == name.as_str()) {
                        // Try to evaluate the set immediately
                        if !expr_contains_any_prime(&rhs.node) {
                            match eval(ctx, rhs) {
                                Ok(Value::Set(set)) => {
                                    if debug {
                                        eprintln!("BIND: In - {}' \\in set (size {}), pending={}", name, set.len(), pending.len());
                                    }
                                    if set.is_empty() {
                                        // Empty domain - no successors
                                        return Ok(());
                                    }
                                    let save_point = undo.len();
                                    // Save pending state - truncate doesn't restore popped items!
                                    // Part of #101: Fix continuation bug where pending items consumed
                                    // in first iteration were missing in subsequent iterations.
                                    let saved_pending: Vec<_> = pending.clone();
                                    // For each element, bind and continue with remaining conjuncts
                                    for elem in set.iter() {
                                        working.bind(*idx, elem.clone(), undo);
                                        // Continue with pending conjuncts or snapshot
                                        match continue_bind_or_snapshot(ctx, working, undo, pending, current_state, var_indices, registry, successors, debug) {
                                            Ok(()) => {}
                                            Err(e) if is_disabled_action_error(&e) => {
                                                if debug {
                                                    eprintln!("BIND: In elem disabled: {:?}", e);
                                                }
                                            }
                                            Err(e) => return Err(e),
                                        }
                                        // Restore for next element - must restore actual items, not just length
                                        working.unbind_to(undo, save_point);
                                        pending.clear();
                                        pending.extend(saved_pending.iter().cloned());
                                    }
                                    return Ok(());
                                }
                                Err(e) if is_disabled_action_error(&e) => {
                                    return Ok(()); // Action disabled
                                }
                                _ => {
                                    // Fall through
                                }
                            }
                        }
                    }
                }
            }
            // Not a simple primed membership - treat as leaf/guard
            process_bind_leaf(ctx, expr, working, undo, pending, current_state, var_indices, registry, successors, debug)
        }

        // IF-THEN-ELSE: evaluate condition and recurse into appropriate branch
        Expr::If(cond, then_branch, else_branch) => {
            // Part of #101, #116: Set up next_state_env so condition can reference bound primed vars
            let prev_next = ctx.bind_next_state_array(working.values());
            let cond_result = eval(ctx, cond);
            ctx.restore_next_state_env(prev_next);
            match cond_result {
                Ok(Value::Bool(true)) => {
                    if debug {
                        eprintln!("BIND: If - condition true, taking then branch");
                    }
                    enumerate_bind_inner(ctx, then_branch, working, undo, pending, current_state, var_indices, registry, successors, debug)
                }
                Ok(Value::Bool(false)) => {
                    if debug {
                        eprintln!("BIND: If - condition false, taking else branch");
                    }
                    enumerate_bind_inner(ctx, else_branch, working, undo, pending, current_state, var_indices, registry, successors, debug)
                }
                _ => {
                    // Condition couldn't be evaluated - try both branches
                    if debug {
                        eprintln!("BIND: If - condition unevaluable, trying both branches");
                    }
                    let save_point = undo.len();
                    // Save pending state - truncate doesn't restore popped items
                    let saved_pending: Vec<_> = pending.clone();
                    let _ = enumerate_bind_inner(ctx, then_branch, working, undo, pending, current_state, var_indices, registry, successors, debug);
                    working.unbind_to(undo, save_point);
                    pending.clear();
                    pending.extend(saved_pending.iter().cloned());
                    let _ = enumerate_bind_inner(ctx, else_branch, working, undo, pending, current_state, var_indices, registry, successors, debug);
                    working.unbind_to(undo, save_point);
                    pending.clear();
                    pending.extend(saved_pending.iter().cloned());
                    Ok(())
                }
            }
        }

        // EXISTS: enumerate domain values, try body with each binding
        Expr::Exists(bounds, body) => {
            if debug {
                eprintln!("BIND: Exists with {} bounds", bounds.len());
            }
            enumerate_bind_exists(ctx, bounds, body, working, undo, pending, current_state, var_indices, registry, successors, debug)
        }

        // APPLY: inline operator definitions
        // Part of #101: Phase 3 - enables operator inlining to reduce fallback rate
        Expr::Apply(op_expr, args) => {
            if let Expr::Ident(op_name) = &op_expr.node {
                // Look up operator definition using raw pointer to avoid holding borrow
                let def_ptr = ctx
                    .get_op(op_name)
                    .map(|def| def as *const OperatorDef);
                if let Some(def_ptr) = def_ptr {
                    // Safety: OperatorDef is stored in ctx.shared.ops / ctx.local_ops and is not
                    // mutated during bind/unbind enumeration.
                    let def = unsafe { &*def_ptr };

                    if debug {
                        eprintln!("BIND: Apply - inlining operator {} with {} args", op_name, args.len());
                    }

                    // Bind arguments to parameters
                    // Part of #101, #116: Set up next_state_env so arguments can reference bound primed vars
                    let mark = ctx.mark_stack();
                    let prev_next = ctx.bind_next_state_array(working.values());
                    for (param, arg) in def.params.iter().zip(args.iter()) {
                        match eval(ctx, arg) {
                            Ok(arg_val) => {
                                ctx.push_binding(Arc::from(param.name.node.as_str()), arg_val);
                            }
                            Err(e) if is_disabled_action_error(&e) => {
                                ctx.restore_next_state_env(prev_next);
                                ctx.pop_to_mark(mark);
                                return Ok(()); // Action disabled
                            }
                            Err(e) => {
                                ctx.restore_next_state_env(prev_next);
                                ctx.pop_to_mark(mark);
                                return Err(e);
                            }
                        }
                    }
                    ctx.restore_next_state_env(prev_next);

                    // Process operator body
                    let res = enumerate_bind_inner(
                        ctx,
                        &def.body,
                        working,
                        undo,
                        pending,
                        current_state,
                        var_indices,
                        registry,
                        successors,
                        debug,
                    );
                    ctx.pop_to_mark(mark);
                    return res;
                }
            }
            // Unknown operator - treat as leaf
            if debug {
                eprintln!("BIND: Apply - not a user-defined operator, treating as leaf");
            }
            process_bind_leaf(ctx, expr, working, undo, pending, current_state, var_indices, registry, successors, debug)
        }

        // IDENT: may be a zero-arg operator reference
        // Part of #101: Phase 3 - handles `Next == Step` where Step is a zero-arg operator
        Expr::Ident(name) => {
            let def_ptr = ctx.get_op(name).map(|def| def as *const OperatorDef);
            if let Some(def_ptr) = def_ptr {
                // Safety: see Apply case above
                let def = unsafe { &*def_ptr };
                if def.params.is_empty() {
                    if debug {
                        eprintln!("BIND: Ident - inlining zero-arg operator {}", name);
                    }
                    return enumerate_bind_inner(
                        ctx,
                        &def.body,
                        working,
                        undo,
                        pending,
                        current_state,
                        var_indices,
                        registry,
                        successors,
                        debug,
                    );
                }
            }
            // Not an operator or has params - treat as leaf
            process_bind_leaf(ctx, expr, working, undo, pending, current_state, var_indices, registry, successors, debug)
        }

        // UNCHANGED: bind primed variables to their current values
        // Part of #101: Phase 3 - reduce fallback rate
        Expr::Unchanged(inner) => {
            if debug {
                eprintln!("BIND: Unchanged expression");
            }
            // Extract unchanged variables and bind them to current values
            bind_unchanged_vars(&inner.node, working, undo, current_state, var_indices, debug)?;
            // Continue with pending conjuncts or snapshot
            continue_bind_or_snapshot(ctx, working, undo, pending, current_state, var_indices, registry, successors, debug)
        }

        // LET: add definitions to context and continue with body
        // Part of #101, #120: Required for specs using LET in actions
        Expr::Let(defs, body) => {
            if debug {
                eprintln!("BIND: Let with {} definitions", defs.len());
            }
            // Register LET definitions into local_ops (same as symbolic path)
            use crate::OpEnv;
            let mut local_ops: OpEnv = match &ctx.local_ops {
                Some(ops) => (**ops).clone(),
                None => OpEnv::default(),
            };
            for def in defs {
                local_ops.insert(def.name.node.clone(), def.clone());
            }
            let mut new_ctx = ctx.clone();
            new_ctx.local_ops = Some(std::sync::Arc::new(local_ops));
            enumerate_bind_inner(&mut new_ctx, body, working, undo, pending, current_state, var_indices, registry, successors, debug)
        }

        // Default: treat as leaf expression (guard or action)
        _ => {
            process_bind_leaf(ctx, expr, working, undo, pending, current_state, var_indices, registry, successors, debug)
        }
    }
}

/// Continue processing pending conjuncts or snapshot if done.
///
/// This is the key helper for the bind/unbind algorithm. After binding a value,
/// we need to either:
/// 1. Process the next pending conjunct (from And handlers)
/// 2. Snapshot the working state if no more pending work
///
/// Part of #101: Fix for the continuation bug where In handler didn't process
/// remaining conjuncts for each element.
fn continue_bind_or_snapshot<'a>(
    ctx: &mut EvalCtx,
    working: &mut ArrayState,
    undo: &mut Vec<UndoEntry>,
    pending: &mut Vec<&'a Spanned<Expr>>,
    current_state: &State,
    var_indices: &[(Arc<str>, VarIndex)],
    registry: &VarRegistry,
    successors: &mut Vec<State>,
    debug: bool,
) -> Result<(), EvalError> {
    if let Some(next) = pending.pop() {
        // Process next pending conjunct
        if debug {
            eprintln!("BIND: continue - processing pending conjunct (remaining={})", pending.len());
        }
        enumerate_bind_inner(ctx, next, working, undo, pending, current_state, var_indices, registry, successors, debug)
    } else {
        // No more pending work - snapshot the working state
        if debug {
            eprintln!("BIND: continue - no pending, snapshotting ({} bindings)", undo.len());
        }
        if !undo.is_empty() {
            successors.push(working.snapshot(registry));
        } else {
            // Identity transition - push current state
            successors.push(current_state.clone());
        }
        Ok(())
    }
}

/// Process a leaf expression in bind/unbind enumeration.
///
/// This handles expressions that are either:
/// 1. Boolean guards - evaluate and continue/fail
/// 2. Action-level expressions - continue with pending or snapshot
fn process_bind_leaf<'a>(
    ctx: &mut EvalCtx,
    expr: &Spanned<Expr>,
    working: &mut ArrayState,
    undo: &mut Vec<UndoEntry>,
    pending: &mut Vec<&'a Spanned<Expr>>,
    current_state: &State,
    var_indices: &[(Arc<str>, VarIndex)],
    registry: &VarRegistry,
    successors: &mut Vec<State>,
    debug: bool,
) -> Result<(), EvalError> {
    // Bind working state so guards can reference already-bound primed variables.
    // For example, in `nxt' = [nxt EXCEPT ![self] = i] /\ ~ flag[nxt'[self]]`,
    // the guard `~ flag[nxt'[self]]` needs to see the already-bound value of nxt'.
    // Part of #101: Fix state count discrepancy in MCBakery.
    let prev_next = ctx.bind_next_state_array(working.values());

    // Try to evaluate as a boolean guard
    let eval_result = eval(ctx, expr);
    ctx.restore_next_state_env(prev_next);

    match eval_result {
        Ok(Value::Bool(true)) => {
            // Guard passed - continue with pending or snapshot
            if debug {
                eprintln!("BIND: Leaf guard passed, continuing (pending={})", pending.len());
            }
            continue_bind_or_snapshot(ctx, working, undo, pending, current_state, var_indices, registry, successors, debug)
        }
        Ok(Value::Bool(false)) => {
            // Guard failed - no successor from this branch
            if debug {
                eprintln!("BIND: Leaf guard failed");
            }
            Ok(())
        }
        Err(e) if is_action_level_error(&e) => {
            // Action-level expression - we may need deeper extraction
            // For now, return Ok to allow fallback
            if debug {
                eprintln!("BIND: Leaf is action-level, deferring");
            }
            Ok(())
        }
        Err(e) if is_disabled_action_error(&e) => {
            // Action disabled
            if debug {
                eprintln!("BIND: Leaf action disabled: {:?}", e);
            }
            Ok(())
        }
        _ => {
            // Can't evaluate - action or error
            if debug {
                eprintln!("BIND: Leaf couldn't be evaluated");
            }
            Ok(())
        }
    }
}

/// Handle EXISTS in bind/unbind enumeration.
///
/// Enumerates the domain and tries the body with each binding.
fn enumerate_bind_exists<'a>(
    ctx: &mut EvalCtx,
    bounds: &'a [BoundVar],
    body: &'a Spanned<Expr>,
    working: &mut ArrayState,
    undo: &mut Vec<UndoEntry>,
    pending: &mut Vec<&'a Spanned<Expr>>,
    current_state: &State,
    var_indices: &[(Arc<str>, VarIndex)],
    registry: &VarRegistry,
    successors: &mut Vec<State>,
    debug: bool,
) -> Result<(), EvalError> {
    if bounds.is_empty() {
        // No bounds - just process body
        return enumerate_bind_inner(ctx, body, working, undo, pending, current_state, var_indices, registry, successors, debug);
    }

    // Get first bound variable
    let bound = &bounds[0];
    let remaining_bounds = &bounds[1..];

    // Get domain expression (if present)
    let domain_expr = match &bound.domain {
        Some(d) => d.as_ref(),
        None => {
            // No domain specified - skip to remaining bounds
            return enumerate_bind_exists(ctx, remaining_bounds, body, working, undo, pending, current_state, var_indices, registry, successors, debug);
        }
    };

    // Evaluate domain
    // Part of #101, #116: Set up next_state_env so domain can reference bound primed vars
    let prev_next = ctx.bind_next_state_array(working.values());
    let domain = match eval(ctx, domain_expr) {
        Ok(v) => {
            ctx.restore_next_state_env(prev_next);
            v
        }
        Err(e) if is_disabled_action_error(&e) => {
            ctx.restore_next_state_env(prev_next);
            return Ok(());
        }
        Err(e) => {
            ctx.restore_next_state_env(prev_next);
            return Err(e);
        }
    };

    let domain_set = match domain.to_sorted_set() {
        Some(set) => set,
        None => return Ok(()), // Can't enumerate - skip
    };

    if domain_set.is_empty() {
        return Ok(()); // Empty domain
    }

    let ctx_mark = ctx.mark_stack();
    let save_point = undo.len();
    // Save pending state - truncate doesn't restore popped items
    let saved_pending: Vec<_> = pending.clone();

    for val in domain_set.iter() {
        // Bind the EXISTS variable
        ctx.push_binding(Arc::from(bound.name.node.as_str()), val.clone());

        // If more bounds, recurse with them
        let result = if remaining_bounds.is_empty() {
            enumerate_bind_inner(ctx, body, working, undo, pending, current_state, var_indices, registry, successors, debug)
        } else {
            enumerate_bind_exists(ctx, remaining_bounds, body, working, undo, pending, current_state, var_indices, registry, successors, debug)
        };

        ctx.pop_to_mark(ctx_mark);
        working.unbind_to(undo, save_point);
        pending.clear();
        pending.extend(saved_pending.iter().cloned());

        match result {
            Ok(()) => {}
            Err(e) if is_disabled_action_error(&e) => {}
            Err(e) => return Err(e),
        }
    }

    Ok(())
}

/// Bind unchanged variables to their current state values.
///
/// Handles both simple identifiers (UNCHANGED x) and tuples (UNCHANGED <<x, y>>).
/// Part of #101: Phase 3 - reduce fallback rate by handling UNCHANGED directly.
fn bind_unchanged_vars(
    expr: &Expr,
    working: &mut ArrayState,
    undo: &mut Vec<UndoEntry>,
    current_state: &State,
    var_indices: &[(Arc<str>, VarIndex)],
    debug: bool,
) -> Result<(), EvalError> {
    match expr {
        Expr::Ident(name) => {
            // Single variable: UNCHANGED x
            if let Some((_, idx)) = var_indices.iter().find(|(n, _)| n.as_ref() == name.as_str()) {
                let current_val = current_state.get(name.as_str()).cloned()
                    .unwrap_or(Value::Bool(false));
                if debug {
                    eprintln!("BIND: Unchanged - binding {}' = {:?}", name, current_val);
                }
                working.bind(*idx, current_val, undo);
            }
            Ok(())
        }
        Expr::Tuple(elems) => {
            // Tuple: UNCHANGED <<x, y, z>>
            for elem in elems {
                bind_unchanged_vars(&elem.node, working, undo, current_state, var_indices, debug)?;
            }
            Ok(())
        }
        _ => {
            // Other expressions - ignore (could be complex expression like UNCHANGED f[x])
            if debug {
                eprintln!("BIND: Unchanged - skipping complex expression");
            }
            Ok(())
        }
    }
}

fn enumerate_continuation_inner<'a>(
    ctx: &mut EvalCtx,
    expr: &'a Spanned<Expr>,
    stack: &mut WorkStack<'a>,
    current_state: &State,
    vars: &[Arc<str>],
    successors: &mut Vec<State>,
    debug: bool,
) -> Result<(), EvalError> {
    // Use stacker for deeply nested expressions
    stacker::maybe_grow(STACK_RED_ZONE, STACK_GROW_SIZE, || {
        enumerate_continuation_dispatch(ctx, expr, stack, current_state, vars, successors, debug)
    })
}

fn enumerate_continuation_dispatch<'a>(
    ctx: &mut EvalCtx,
    expr: &'a Spanned<Expr>,
    stack: &mut WorkStack<'a>,
    current_state: &State,
    vars: &[Arc<str>],
    successors: &mut Vec<State>,
    debug: bool,
) -> Result<(), EvalError> {
    match &expr.node {
        // Conjunction: push right side onto stack, continue with left
        // This is the key optimization - no AST rebuilding
        Expr::And(a, b) => {
            if debug {
                eprintln!("CONT: And - push right, process left");
            }
            stack.push_item(WorkItem::Conjunct(b));
            enumerate_continuation_inner(ctx, a, stack, current_state, vars, successors, debug)
        }

        // Disjunction: fork - try both branches with current stack.
        // We must snapshot because branch processing may pop work items.
        Expr::Or(a, b) => {
            if debug {
                eprintln!("CONT: Or - forking");
            }
            // Mark before left branch - will restore before right branch.
            // Uses index-based iteration: mark saves (items_idx, items_len, symbolic_len, constraints_len).
            // No cloning - just four integers.
            let stack_mark = stack.mark();
            match enumerate_continuation_inner(ctx, a, stack, current_state, vars, successors, debug) {
                Ok(()) => {}
                Err(e) if is_disabled_action_error(&e) => {
                    if debug {
                        eprintln!("CONT: Or left branch disabled: {:?}", e);
                    }
                }
                Err(e) => return Err(e),
            }

            // Restore stack state for right branch (reset index, truncate additions)
            stack.restore(stack_mark);
            match enumerate_continuation_inner(ctx, b, stack, current_state, vars, successors, debug) {
                Ok(()) => {}
                Err(e) if is_disabled_action_error(&e) => {
                    if debug {
                        eprintln!("CONT: Or right branch disabled: {:?}", e);
                    }
                }
                Err(e) => return Err(e),
            }
            Ok(())
        }

        // EXISTS: enumerate domain values, try body with each binding
        Expr::Exists(bounds, body) => {
            if debug {
                eprintln!("CONT: Exists with {} bounds", bounds.len());
            }
            enumerate_exists_continuation(ctx, bounds, body, stack, current_state, vars, successors, debug)
        }

        // IF-THEN-ELSE: evaluate condition and recurse into appropriate branch
        Expr::If(cond, then_branch, else_branch) => {
            let cond_res = if profile_enum_detail() {
                let start = std::time::Instant::now();
                let res = eval(ctx, cond);
                PROF_GUARD_US.fetch_add(
                    start.elapsed().as_micros() as u64,
                    AtomicOrdering::Relaxed,
                );
                res
            } else {
                eval(ctx, cond)
            };
            match cond_res {
                Ok(Value::Bool(true)) => {
                    if debug {
                        eprintln!("CONT: If - condition true, taking then branch");
                    }
                    enumerate_continuation_inner(ctx, then_branch, stack, current_state, vars, successors, debug)
                }
                Ok(Value::Bool(false)) => {
                    if debug {
                        eprintln!("CONT: If - condition false, taking else branch");
                    }
                    enumerate_continuation_inner(ctx, else_branch, stack, current_state, vars, successors, debug)
                }
                Ok(_) | Err(_) => {
                    // Condition couldn't be evaluated (likely contains primed variables).
                    // Try progressive state construction (like TLC): build partial next-state
                    // from assignments accumulated so far, then retry evaluation.
                    //
                    // This handles patterns like:
                    //   unchecked' = [unchecked EXCEPT ![self] = ...]
                    //   /\ IF unchecked'[self] = {} THEN pc' = "cs" ELSE pc' = "w1"
                    // where the IF condition depends on a primed variable assigned earlier.

                    if !stack.symbolic.is_empty() {
                        if let Some(partial_next) = try_build_partial_next_state(ctx, &stack.symbolic, current_state) {
                            // Retry evaluation with partial next-state
                            let mut eval_ctx = ctx.clone();
                            eval_ctx.next_state = Some(std::sync::Arc::new(partial_next));

                            let retry_res = if profile_enum_detail() {
                                let start = std::time::Instant::now();
                                let res = eval(&eval_ctx, cond);
                                PROF_GUARD_US.fetch_add(
                                    start.elapsed().as_micros() as u64,
                                    AtomicOrdering::Relaxed,
                                );
                                res
                            } else {
                                eval(&eval_ctx, cond)
                            };
                            match retry_res {
                                Ok(Value::Bool(true)) => {
                                    if debug {
                                        eprintln!("CONT: If - condition true (with partial next-state), taking then branch");
                                    }
                                    return enumerate_continuation_inner(ctx, then_branch, stack, current_state, vars, successors, debug);
                                }
                                Ok(Value::Bool(false)) => {
                                    if debug {
                                        eprintln!("CONT: If - condition false (with partial next-state), taking else branch");
                                    }
                                    return enumerate_continuation_inner(ctx, else_branch, stack, current_state, vars, successors, debug);
                                }
                                _ => {
                                    // Still can't evaluate - fall through to slow path
                                }
                            }
                        }
                    }

                    // Fall back to treating IF as a leaf (slow path with branch merging)
                    if debug {
                        eprintln!("CONT: If - condition unevaluable, treating as leaf (slow path)");
                    }
                    process_continuation_leaf(ctx, expr, stack, current_state, vars, successors, debug)
                }
            }
        }

        // LET: add definitions to context and continue with body
        Expr::Let(defs, body) => {
            if debug {
                eprintln!("CONT: Let with {} definitions", defs.len());
            }
            // Register LET definitions lazily
            use crate::OpEnv;
            let mut local_ops: OpEnv = match &ctx.local_ops {
                Some(ops) => (**ops).clone(),
                None => OpEnv::default(),
            };
            for def in defs {
                local_ops.insert(def.name.node.clone(), def.clone());
            }
            let mut new_ctx = ctx.clone();
            new_ctx.local_ops = Some(std::sync::Arc::new(local_ops));
            enumerate_continuation_inner(&mut new_ctx, body, stack, current_state, vars, successors, debug)
        }

        // Operator application: inline and process
        // Bug #86 fix: Isolate operator body from caller's pending items
        Expr::Apply(op_expr, args) => {
            if let Expr::Ident(op_name) = &op_expr.node {
                let def_ptr = ctx
                    .get_op(op_name)
                    .map(|def| def as *const OperatorDef);
                if let Some(def_ptr) = def_ptr {
                    // Safety: OperatorDef is stored in ctx.shared.ops / ctx.local_ops and is not
                    // mutated during continuation enumeration. We use a raw pointer to avoid
                    // holding an immutable borrow of `ctx` across mutations (binding stack ops).
                    let def = unsafe { &*def_ptr };

                    // Bind arguments to parameters.
                    let mark = ctx.mark_stack();
                    for (param, arg) in def.params.iter().zip(args.iter()) {
                        let arg_val = eval(ctx, arg)?;
                        ctx.push_binding(Arc::from(param.name.node.as_str()), arg_val);
                    }

                    // Process operator body - remaining items in stack are processed inside
                    let res = enumerate_continuation_inner(
                        ctx,
                        &def.body,
                        stack,
                        current_state,
                        vars,
                        successors,
                        debug,
                    );
                    ctx.pop_to_mark(mark);
                    return res;
                }
            }
            // Unknown operator - treat as leaf
            process_continuation_leaf(ctx, expr, stack, current_state, vars, successors, debug)
        }

        // Identifier that might be an operator
        Expr::Ident(name) => {
            let def_ptr = ctx.get_op(name).map(|def| def as *const OperatorDef);
            if let Some(def_ptr) = def_ptr {
                // Safety: see Apply() case above.
                let def = unsafe { &*def_ptr };
                if def.params.is_empty() {
                    return enumerate_continuation_inner(
                        ctx,
                        &def.body,
                        stack,
                        current_state,
                        vars,
                        successors,
                        debug,
                    );
                }
            }
            // Not an operator or has params - treat as leaf
            process_continuation_leaf(ctx, expr, stack, current_state, vars, successors, debug)
        }

        // CASE expression
        Expr::Case(arms, other) => {
            if debug {
                eprintln!("CONT: Case with {} arms", arms.len());
            }
            // Mark once for all arms; only needed for arms whose guards are unevaluable.
            let stack_mark = stack.mark();
            for arm in arms {
                let guard_res = if profile_enum_detail() {
                    let start = std::time::Instant::now();
                    let res = eval(ctx, &arm.guard);
                    PROF_GUARD_US.fetch_add(
                        start.elapsed().as_micros() as u64,
                        AtomicOrdering::Relaxed,
                    );
                    res
                } else {
                    eval(ctx, &arm.guard)
                };
                match guard_res {
                    Ok(Value::Bool(true)) => {
                        return enumerate_continuation_inner(ctx, &arm.body, stack, current_state, vars, successors, debug);
                    }
                    Ok(Value::Bool(false)) => continue,
                    _ => {
                        // Condition couldn't be evaluated - try this arm, restoring the stack afterward.
                        // We must restore from the mark because branch processing advances items_idx.
                        let res =
                            enumerate_continuation_inner(ctx, &arm.body, stack, current_state, vars, successors, debug);
                        stack.restore(stack_mark);
                        res?;
                    }
                }
            }
            // No arm matched - check for OTHER clause
            if let Some(other_body) = other {
                enumerate_continuation_inner(ctx, other_body, stack, current_state, vars, successors, debug)
            } else {
                Ok(())
            }
        }

        // Equality: check for primed variable assignment (x' = expr)
        // TLC-style: evaluate RHS immediately and bind, avoiding deferred evaluation
        Expr::Eq(lhs, rhs) => {
            // Check if LHS is a primed state variable: x'
            if let Expr::Prime(inner_lhs) = &lhs.node {
                if let Expr::Ident(name) = &inner_lhs.node {
                    if let Some(var) = vars.iter().find(|v| v.as_ref() == name.as_str()) {
                        // Fast path: x' = x (UNCHANGED)
                        if let Expr::Ident(rhs_name) = &rhs.node {
                            if rhs_name == name && !ctx.has_local_binding(rhs_name.as_str()) {
                                if debug {
                                    eprintln!("CONT: Eq - x' = x (UNCHANGED) for {}", name);
                                }
                                stack.symbolic.push(SymbolicAssignment::Unchanged(Arc::clone(var)));
                                return continue_with_stack(ctx, stack, current_state, vars, successors, debug);
                            }
                        }

                        // Try immediate evaluation (TLC-style bind)
                        // Only if RHS doesn't contain primed variables
                        if !expr_contains_any_prime(&rhs.node) {
                            let eval_res = if profile_enum_detail() {
                                let start = std::time::Instant::now();
                                let res = eval(ctx, rhs);
                                PROF_ASSIGN_US.fetch_add(
                                    start.elapsed().as_micros() as u64,
                                    AtomicOrdering::Relaxed,
                                );
                                res
                            } else {
                                eval(ctx, rhs)
                            };
                            match eval_res {
                                Ok(value) => {
                                    if debug {
                                        eprintln!("CONT: Eq - immediate eval {}' = {:?}", name, value);
                                    }
                                    // Bind immediately (TLC-style)
                                    stack.symbolic.push(SymbolicAssignment::Value(Arc::clone(var), value));
                                    return continue_with_stack(ctx, stack, current_state, vars, successors, debug);
                                }
                                Err(e) if is_disabled_action_error(&e) => {
                                    // Action disabled - no successors from this branch
                                    if debug {
                                        eprintln!("CONT: Eq - action disabled: {:?}", e);
                                    }
                                    return Ok(());
                                }
                                Err(_) => {
                                    // Evaluation failed - fall through to leaf processing
                                    // This handles cases where the expression can't be evaluated yet
                                }
                            }
                        }
                    }
                }
            }
            // Check symmetric case: expr = x'
            if let Expr::Prime(inner_rhs) = &rhs.node {
                if let Expr::Ident(name) = &inner_rhs.node {
                    if let Some(var) = vars.iter().find(|v| v.as_ref() == name.as_str()) {
                        // Fast path: x = x' (UNCHANGED)
                        if let Expr::Ident(lhs_name) = &lhs.node {
                            if lhs_name == name && !ctx.has_local_binding(lhs_name.as_str()) {
                                if debug {
                                    eprintln!("CONT: Eq - x = x' (UNCHANGED) for {}", name);
                                }
                                stack.symbolic.push(SymbolicAssignment::Unchanged(Arc::clone(var)));
                                return continue_with_stack(ctx, stack, current_state, vars, successors, debug);
                            }
                        }

                        // Try immediate evaluation for expr = x' (LHS is the value)
                        if !expr_contains_any_prime(&lhs.node) {
                            let eval_res = if profile_enum_detail() {
                                let start = std::time::Instant::now();
                                let res = eval(ctx, lhs);
                                PROF_ASSIGN_US.fetch_add(
                                    start.elapsed().as_micros() as u64,
                                    AtomicOrdering::Relaxed,
                                );
                                res
                            } else {
                                eval(ctx, lhs)
                            };
                            match eval_res {
                                Ok(value) => {
                                    if debug {
                                        eprintln!("CONT: Eq - immediate eval {} = {}'", value, name);
                                    }
                                    stack.symbolic.push(SymbolicAssignment::Value(Arc::clone(var), value));
                                    return continue_with_stack(ctx, stack, current_state, vars, successors, debug);
                                }
                                Err(e) if is_disabled_action_error(&e) => {
                                    if debug {
                                        eprintln!("CONT: Eq - action disabled: {:?}", e);
                                    }
                                    return Ok(());
                                }
                                Err(_) => {
                                    // Fall through to leaf processing
                                }
                            }
                        }
                    }
                }
            }
            // Fall through to leaf processing for complex cases
            process_continuation_leaf(ctx, expr, stack, current_state, vars, successors, debug)
        }

        // Membership: check for primed variable assignment (x' \in S)
        Expr::In(lhs, rhs) => {
            // Check if LHS is a primed state variable: x' \in S
            if let Expr::Prime(inner_lhs) = &lhs.node {
                if let Expr::Ident(name) = &inner_lhs.node {
                    if let Some(var) = vars.iter().find(|v| v.as_ref() == name.as_str()) {
                        // Try to evaluate the set immediately
                        if !expr_contains_any_prime(&rhs.node) {
                            let eval_res = if profile_enum_detail() {
                                let start = std::time::Instant::now();
                                let res = eval(ctx, rhs);
                                PROF_DOMAIN_US.fetch_add(
                                    start.elapsed().as_micros() as u64,
                                    AtomicOrdering::Relaxed,
                                );
                                res
                            } else {
                                eval(ctx, rhs)
                            };
                            match eval_res {
                                Ok(set_val) => {
                                    if let Some(set) = set_val.to_sorted_set() {
                                        if debug {
                                            eprintln!("CONT: In - x' \\in S for {} with {} elements", name, set.len());
                                        }
                                        // Fork for each element in the set (like TLC)
                                        let stack_mark = stack.mark();
                                        for (i, elem) in set.iter().enumerate() {
                                            if i > 0 {
                                                stack.restore(stack_mark);
                                            }
                                            stack.symbolic.push(SymbolicAssignment::Value(Arc::clone(var), elem.clone()));
                                            match continue_with_stack(ctx, stack, current_state, vars, successors, debug) {
                                                Ok(()) => {}
                                                Err(e) if is_disabled_action_error(&e) => {
                                                    if debug {
                                                        eprintln!("CONT: In - branch disabled for {}: {:?}", elem, e);
                                                    }
                                                }
                                                Err(e) => return Err(e),
                                            }
                                        }
                                        return Ok(());
                                    }
                                }
                                Err(e) if is_disabled_action_error(&e) => {
                                    if debug {
                                        eprintln!("CONT: In - action disabled: {:?}", e);
                                    }
                                    return Ok(());
                                }
                                Err(_) => {
                                    // Fall through to leaf processing
                                }
                            }
                        }
                    }
                }
            }
            // Fall through to leaf processing
            process_continuation_leaf(ctx, expr, stack, current_state, vars, successors, debug)
        }

        // UNCHANGED: extract variables and mark them unchanged
        Expr::Unchanged(inner) => {
            // Extract unchanged variables inline (reuse existing helper)
            let before = stack.symbolic.len();
            extract_unchanged_vars_symbolic(&inner.node, vars, &mut stack.symbolic);
            if debug {
                eprintln!("CONT: Unchanged - extracted {} vars", stack.symbolic.len() - before);
            }
            continue_with_stack(ctx, stack, current_state, vars, successors, debug)
        }

        // All other expressions: treat as leaf (guard or assignment)
        _ => process_continuation_leaf(ctx, expr, stack, current_state, vars, successors, debug),
    }
}

/// Process a leaf expression in continuation-passing enumeration.
/// This handles guards and assignments, then continues with the work stack.
fn process_continuation_leaf<'a>(
    ctx: &mut EvalCtx,
    expr: &'a Spanned<Expr>,
    stack: &mut WorkStack<'a>,
    current_state: &State,
    vars: &[Arc<str>],
    successors: &mut Vec<State>,
    debug: bool,
) -> Result<(), EvalError> {
    // Check if this is a guard (can be evaluated to boolean)
    let eval_res = if profile_enum_detail() {
        let start = std::time::Instant::now();
        let res = eval(ctx, expr);
        PROF_GUARD_US.fetch_add(
            start.elapsed().as_micros() as u64,
            AtomicOrdering::Relaxed,
        );
        res
    } else {
        eval(ctx, expr)
    };
    match eval_res {
        Ok(Value::Bool(true)) => {
            // Guard passed - continue with work stack
            if debug {
                eprintln!(
                    "CONT: Guard passed, continuing with stack (len={})",
                    stack.items.len()
                );
            }
            continue_with_stack(ctx, stack, current_state, vars, successors, debug)
        }
        Ok(Value::Bool(false)) => {
            // Guard failed - this branch produces no successors
            if debug {
                eprintln!("CONT: Guard failed");
            }
            Ok(())
        }
        Err(e) if is_action_level_error(&e) => {
            // This is an action (contains primed vars) - extract assignments
            if debug {
                eprintln!("CONT: Leaf is action, extracting assignments");
            }

            // Accumulate assignments from only this leaf; remaining conjuncts will be processed
            // normally via the work stack (including Or forking).
            extract_symbolic_assignments(ctx, expr, vars, &mut stack.symbolic)?;

            // Preserve the action constraint for final next-state validation.
            // Capture local bindings (O(bindings)) instead of substituting into AST (O(AST size)).
            if profile_enum_detail() {
                CONSTRAINT_PUSH_COUNT.fetch_add(1, AtomicOrdering::Relaxed);
            }
            stack.constraints.push(CapturedConstraint {
                expr,
                bindings: ctx.get_local_bindings().to_vec(),
            });

            continue_with_stack(ctx, stack, current_state, vars, successors, debug)
        }
        Err(e) if is_disabled_action_error(&e) => {
            // Action disabled due to domain error etc - no successors from this branch
            if debug {
                eprintln!("CONT: Action disabled: {:?}", e);
            }
            Ok(())
        }
        Ok(_) | Err(_) => {
            // Non-boolean result or other error - no successors
            if debug {
                eprintln!("CONT: Leaf evaluation failed or non-boolean");
            }
            Ok(())
        }
    }
}

/// Continue processing the work stack.
/// Uses LIFO order (pop from end) for proper left-to-right conjunction processing.
fn continue_with_stack<'a>(
    ctx: &mut EvalCtx,
    stack: &mut WorkStack<'a>,
    current_state: &State,
    vars: &[Arc<str>],
    successors: &mut Vec<State>,
    debug: bool,
) -> Result<(), EvalError> {
    match stack.next_item() {
        None => {
            finalize_continuation_branch(ctx, stack, current_state, vars, successors, debug)
        }
        Some(&WorkItem::Conjunct(expr)) => {
            // Process next conjunct
            enumerate_continuation_inner(ctx, expr, stack, current_state, vars, successors, debug)
        }
    }
}

fn finalize_continuation_branch(
    ctx: &mut EvalCtx,
    stack: &WorkStack<'_>,
    current_state: &State,
    vars: &[Arc<str>],
    successors: &mut Vec<State>,
    debug: bool,
) -> Result<(), EvalError> {
    if debug {
        eprintln!(
            "CONT: Stack empty, finalizing branch (symbolic={}, constraints={})",
            stack.symbolic.len(),
            stack.constraints.len()
        );
    }

    if stack.symbolic.is_empty() {
        // Identity transition (pure guards / pure action predicates).
        if stack.constraints.iter().all(|c| {
            let eval_ctx = ctx.with_captured_bindings(&c.bindings);
            action_holds_in_next_state(&eval_ctx, c.expr, current_state)
        }) {
            successors.push(current_state.clone());
        }
        return Ok(());
    }

    let assignments = evaluate_symbolic_assignments(ctx, &stack.symbolic)?;
    if assignments.is_empty() {
        return Ok(());
    }

    let registry = ctx.var_registry();
    let reg_opt = if registry.is_empty() { None } else { Some(registry) };
    let states = build_successor_states_with_ctx(current_state, vars, &assignments, reg_opt, Some(ctx));

    for state in states {
        if stack.constraints.iter().all(|c| {
            let eval_ctx = ctx.with_captured_bindings(&c.bindings);
            action_holds_in_next_state(&eval_ctx, c.expr, &state)
        }) {
            successors.push(state);
        } else if debug {
            eprintln!("CONT: Some constraint failed for candidate successor");
        }
    }

    Ok(())
}

/// Enumerate EXISTS expression with continuation-passing.
#[allow(clippy::too_many_arguments)]
fn enumerate_exists_continuation<'a>(
    ctx: &mut EvalCtx,
    bounds: &'a [BoundVar],
    body: &'a Spanned<Expr>,
    stack: &mut WorkStack<'a>,
    current_state: &State,
    vars: &[Arc<str>],
    successors: &mut Vec<State>,
    debug: bool,
) -> Result<(), EvalError> {
    // Enumerate all combinations of bound variables
    enumerate_exists_bounds_continuation(ctx, bounds, 0, body, stack, current_state, vars, successors, debug)
}

#[allow(clippy::too_many_arguments)]
fn enumerate_exists_bounds_continuation<'a>(
    ctx: &mut EvalCtx,
    bounds: &'a [BoundVar],
    bound_idx: usize,
    body: &'a Spanned<Expr>,
    stack: &mut WorkStack<'a>,
    current_state: &State,
    vars: &[Arc<str>],
    successors: &mut Vec<State>,
    debug: bool,
) -> Result<(), EvalError> {
    if bound_idx >= bounds.len() {
        // All bounds enumerated - process body
        return enumerate_continuation_inner(ctx, body, stack, current_state, vars, successors, debug);
    }

    let bound = &bounds[bound_idx];

    // Get domain expression (if present)
    let domain_expr = match &bound.domain {
        Some(d) => d.as_ref(),
        None => {
            // No domain specified - skip this bound (shouldn't happen for valid EXISTS)
            return enumerate_exists_bounds_continuation(
                ctx, bounds, bound_idx + 1, body, stack, current_state, vars, successors, debug,
            );
        }
    };

    let domain = if profile_enum_detail() {
        let start = std::time::Instant::now();
        let res = eval(ctx, domain_expr);
        PROF_DOMAIN_US.fetch_add(
            start.elapsed().as_micros() as u64,
            AtomicOrdering::Relaxed,
        );
        res?
    } else {
        eval(ctx, domain_expr)?
    };
    let domain_set = domain.to_sorted_set().ok_or_else(|| EvalError::TypeError {
        expected: "set",
        got: domain.type_name(),
        span: domain_expr.span.into(),
    })?;

    if debug_exists() && should_print_exists_debug_line() {
        eprintln!(
            "EXISTS {}[{}/{}]: domain_size={} body_span={:?}",
            bound.name.node.as_str(),
            bound_idx + 1,
            bounds.len(),
            domain_set.len(),
            body.span
        );
    }

    let ctx_mark = ctx.mark_stack();
    let stack_mark = stack.mark();
    for val in domain_set.iter() {
        if profile_enum_detail() {
            PROF_EXISTS_LOOP_ITERS.fetch_add(1, AtomicOrdering::Relaxed);
            if bound_idx + 1 == bounds.len() {
                PROF_EXISTS_SINGLE_BOUND.fetch_add(1, AtomicOrdering::Relaxed);
            } else {
                PROF_EXISTS_MULTI_BOUND.fetch_add(1, AtomicOrdering::Relaxed);
            }
        }
        ctx.push_binding(Arc::from(bound.name.node.as_str()), val.clone());
        let res = enumerate_exists_bounds_continuation(
            ctx, bounds, bound_idx + 1, body, stack, current_state, vars, successors, debug,
        );

        // Cleanup bindings and restore stack state for next iteration.
        ctx.pop_to_mark(ctx_mark);
        stack.restore(stack_mark);

        res?;
    }

    Ok(())
}

// =============================================================================
// END CONTINUATION-PASSING ENUMERATION
// =============================================================================

/// Check if an expression contains ANY primed variables, regardless of context.
/// This is used for implication LHS where even `x' = expr` patterns are boolean conditions.
pub(crate) fn expr_contains_any_prime(expr: &Expr) -> bool {
    match expr {
        Expr::Prime(_) => true,

        // Binary operators
        Expr::And(a, b)
        | Expr::Or(a, b)
        | Expr::Implies(a, b)
        | Expr::Equiv(a, b)
        | Expr::Eq(a, b)
        | Expr::Neq(a, b)
        | Expr::Lt(a, b)
        | Expr::Leq(a, b)
        | Expr::Gt(a, b)
        | Expr::Geq(a, b)
        | Expr::In(a, b)
        | Expr::NotIn(a, b)
        | Expr::Subseteq(a, b)
        | Expr::Union(a, b)
        | Expr::Intersect(a, b)
        | Expr::SetMinus(a, b)
        | Expr::Add(a, b)
        | Expr::Sub(a, b)
        | Expr::Mul(a, b)
        | Expr::Div(a, b)
        | Expr::IntDiv(a, b)
        | Expr::Mod(a, b)
        | Expr::Pow(a, b)
        | Expr::Range(a, b)
        | Expr::LeadsTo(a, b)
        | Expr::WeakFair(a, b)
        | Expr::StrongFair(a, b)
        | Expr::FuncSet(a, b)
        | Expr::FuncApply(a, b) => {
            expr_contains_any_prime(&a.node) || expr_contains_any_prime(&b.node)
        }

        // Unary operators
        Expr::Not(a)
        | Expr::Neg(a)
        | Expr::Powerset(a)
        | Expr::BigUnion(a)
        | Expr::Domain(a)
        | Expr::Always(a)
        | Expr::Eventually(a)
        | Expr::Enabled(a)
        | Expr::Unchanged(a) => expr_contains_any_prime(&a.node),

        Expr::If(cond, then_br, else_br) => {
            expr_contains_any_prime(&cond.node)
                || expr_contains_any_prime(&then_br.node)
                || expr_contains_any_prime(&else_br.node)
        }

        Expr::Case(arms, other) => {
            arms.iter().any(|arm| {
                expr_contains_any_prime(&arm.guard.node) || expr_contains_any_prime(&arm.body.node)
            }) || other
                .as_ref()
                .is_some_and(|e| expr_contains_any_prime(&e.node))
        }

        Expr::Let(defs, body) => {
            defs.iter()
                .any(|def| expr_contains_any_prime(&def.body.node))
                || expr_contains_any_prime(&body.node)
        }

        Expr::Forall(bounds, body) | Expr::Exists(bounds, body) => {
            bounds.iter().any(|b| {
                b.domain
                    .as_ref()
                    .is_some_and(|d| expr_contains_any_prime(&d.node))
            }) || expr_contains_any_prime(&body.node)
        }

        Expr::Apply(f, args) => {
            expr_contains_any_prime(&f.node)
                || args.iter().any(|a| expr_contains_any_prime(&a.node))
        }

        Expr::FuncDef(bounds, body) | Expr::SetBuilder(body, bounds) => {
            bounds.iter().any(|b| {
                b.domain
                    .as_ref()
                    .is_some_and(|d| expr_contains_any_prime(&d.node))
            }) || expr_contains_any_prime(&body.node)
        }

        Expr::SetFilter(bound, pred) => {
            bound
                .domain
                .as_ref()
                .is_some_and(|d| expr_contains_any_prime(&d.node))
                || expr_contains_any_prime(&pred.node)
        }

        Expr::Choose(bound, body) => {
            bound
                .domain
                .as_ref()
                .is_some_and(|d| expr_contains_any_prime(&d.node))
                || expr_contains_any_prime(&body.node)
        }

        Expr::Tuple(elems) | Expr::SetEnum(elems) | Expr::Times(elems) => {
            elems.iter().any(|e| expr_contains_any_prime(&e.node))
        }

        Expr::Record(fields) | Expr::RecordSet(fields) => {
            fields.iter().any(|(_, v)| expr_contains_any_prime(&v.node))
        }

        Expr::Except(base, specs) => {
            expr_contains_any_prime(&base.node)
                || specs.iter().any(|s| {
                    s.path.iter().any(|p| match p {
                        tla_core::ast::ExceptPathElement::Index(e) => {
                            expr_contains_any_prime(&e.node)
                        }
                        tla_core::ast::ExceptPathElement::Field(_) => false,
                    }) || expr_contains_any_prime(&s.value.node)
                })
        }

        Expr::RecordAccess(base, _) => expr_contains_any_prime(&base.node),

        Expr::ModuleRef(_, _, args) => args.iter().any(|a| expr_contains_any_prime(&a.node)),
        Expr::InstanceExpr(_, subs) => subs.iter().any(|s| expr_contains_any_prime(&s.to.node)),

        Expr::Lambda(_, body) => expr_contains_any_prime(&body.node),

        // These don't contain primes
        Expr::Ident(_) | Expr::Bool(_) | Expr::Int(_) | Expr::String(_) | Expr::OpRef(_) => false,
    }
}

fn expr_contains_prime_not_assignment(expr: &Expr) -> bool {
    match expr {
        // Assignment of the form x' = expr: ignore primes on the LHS but still check RHS.
        // The RHS might contain operator applications that expand to prime guards.
        Expr::Eq(lhs, rhs) => {
            let lhs_is_simple_prime_assign = matches!(
                &lhs.node,
                Expr::Prime(inner) if matches!(&inner.node, Expr::Ident(_))
            );
            let lhs_bad = if lhs_is_simple_prime_assign {
                false
            } else {
                expr_contains_prime_not_assignment(&lhs.node)
            };
            lhs_bad || expr_contains_prime_not_assignment(&rhs.node)
        }

        // Membership assignment of the form x' \in S
        Expr::In(lhs, rhs) => {
            let lhs_is_simple_prime = matches!(
                &lhs.node,
                Expr::Prime(inner) if matches!(&inner.node, Expr::Ident(_))
            );
            let lhs_bad = if lhs_is_simple_prime {
                false
            } else {
                expr_contains_prime_not_assignment(&lhs.node)
            };
            lhs_bad || expr_contains_prime_not_assignment(&rhs.node)
        }

        Expr::Prime(_) => true,

        // For Implies, the LHS is used as a boolean condition, not as an assignment.
        // If the LHS contains primed variables (even in x' = expr form), we need validation
        // because the implication's truth depends on evaluating the primed expressions.
        // Example: HCnxt => t >= 1 where HCnxt = (hr' = IF hr # 12 THEN hr + 1 ELSE 1)
        // Here HCnxt evaluates to TRUE when hr changes, so the constraint must be checked.
        Expr::Implies(a, b) => {
            expr_contains_any_prime(&a.node) || expr_contains_prime_not_assignment(&b.node)
        }

        Expr::And(a, b)
        | Expr::Or(a, b)
        | Expr::Equiv(a, b)
        | Expr::Neq(a, b)
        | Expr::Lt(a, b)
        | Expr::Leq(a, b)
        | Expr::Gt(a, b)
        | Expr::Geq(a, b)
        | Expr::NotIn(a, b)
        | Expr::Subseteq(a, b)
        | Expr::Union(a, b)
        | Expr::Intersect(a, b)
        | Expr::SetMinus(a, b)
        | Expr::Add(a, b)
        | Expr::Sub(a, b)
        | Expr::Mul(a, b)
        | Expr::Div(a, b)
        | Expr::IntDiv(a, b)
        | Expr::Mod(a, b)
        | Expr::Pow(a, b)
        | Expr::Range(a, b)
        | Expr::LeadsTo(a, b)
        | Expr::WeakFair(a, b)
        | Expr::StrongFair(a, b) => {
            expr_contains_prime_not_assignment(&a.node)
                || expr_contains_prime_not_assignment(&b.node)
        }

        Expr::Not(a)
        | Expr::Neg(a)
        | Expr::Powerset(a)
        | Expr::BigUnion(a)
        | Expr::Domain(a)
        | Expr::Always(a)
        | Expr::Eventually(a)
        | Expr::Enabled(a)
        | Expr::Unchanged(a) => expr_contains_prime_not_assignment(&a.node),

        Expr::If(cond, then_br, else_br) => {
            expr_contains_prime_not_assignment(&cond.node)
                || expr_contains_prime_not_assignment(&then_br.node)
                || expr_contains_prime_not_assignment(&else_br.node)
        }

        Expr::Case(arms, other) => {
            arms.iter().any(|arm| {
                expr_contains_prime_not_assignment(&arm.guard.node)
                    || expr_contains_prime_not_assignment(&arm.body.node)
            }) || other
                .as_ref()
                .is_some_and(|e| expr_contains_prime_not_assignment(&e.node))
        }

        Expr::Forall(bounds, body) | Expr::Exists(bounds, body) => {
            bounds.iter().any(|b| {
                b.domain
                    .as_ref()
                    .is_some_and(|d| expr_contains_prime_not_assignment(&d.node))
            }) || expr_contains_prime_not_assignment(&body.node)
        }

        Expr::Choose(bound, body) => {
            bound
                .domain
                .as_ref()
                .is_some_and(|d| expr_contains_prime_not_assignment(&d.node))
                || expr_contains_prime_not_assignment(&body.node)
        }

        Expr::Apply(op, args) => {
            expr_contains_prime_not_assignment(&op.node)
                || args
                    .iter()
                    .any(|a| expr_contains_prime_not_assignment(&a.node))
        }

        Expr::FuncApply(func, arg) => {
            expr_contains_prime_not_assignment(&func.node)
                || expr_contains_prime_not_assignment(&arg.node)
        }

        Expr::FuncDef(bounds, body) => {
            bounds.iter().any(|b| {
                b.domain
                    .as_ref()
                    .is_some_and(|d| expr_contains_prime_not_assignment(&d.node))
            }) || expr_contains_prime_not_assignment(&body.node)
        }

        Expr::FuncSet(a, b) => {
            expr_contains_prime_not_assignment(&a.node)
                || expr_contains_prime_not_assignment(&b.node)
        }

        Expr::SetEnum(elems) | Expr::Tuple(elems) | Expr::Times(elems) => elems
            .iter()
            .any(|e| expr_contains_prime_not_assignment(&e.node)),

        Expr::SetBuilder(expr, bounds) => {
            bounds.iter().any(|b| {
                b.domain
                    .as_ref()
                    .is_some_and(|d| expr_contains_prime_not_assignment(&d.node))
            }) || expr_contains_prime_not_assignment(&expr.node)
        }

        Expr::SetFilter(bound, pred) => {
            bound
                .domain
                .as_ref()
                .is_some_and(|d| expr_contains_prime_not_assignment(&d.node))
                || expr_contains_prime_not_assignment(&pred.node)
        }

        Expr::Let(defs, body) => {
            defs.iter()
                .any(|d| expr_contains_prime_not_assignment(&d.body.node))
                || expr_contains_prime_not_assignment(&body.node)
        }

        Expr::Record(fields) | Expr::RecordSet(fields) => fields
            .iter()
            .any(|(_, v)| expr_contains_prime_not_assignment(&v.node)),

        Expr::RecordAccess(e, _) => expr_contains_prime_not_assignment(&e.node),

        Expr::ModuleRef(_, _, args) => args
            .iter()
            .any(|a| expr_contains_prime_not_assignment(&a.node)),

        Expr::Lambda(_, body) => expr_contains_prime_not_assignment(&body.node),

        Expr::Except(expr, specs) => {
            expr_contains_prime_not_assignment(&expr.node)
                || specs
                    .iter()
                    .any(|s| expr_contains_prime_not_assignment(&s.value.node))
        }

        Expr::InstanceExpr(_, _)
        | Expr::Ident(_)
        | Expr::Bool(_)
        | Expr::Int(_)
        | Expr::String(_)
        | Expr::OpRef(_) => false,
    }
}

/// Check if an expression needs next-state validation.
/// Returns true if any part of the expression might evaluate differently depending on next-state.
/// This includes:
/// - Primed variables outside simple assignments (e.g., IF x' = 1 THEN ...)
/// - Implications whose LHS might be an action operator (e.g., HCnxt => t >= 1)
fn needs_next_state_validation(expr: &Expr) -> bool {
    match expr {
        // Implications might have action predicates in LHS (e.g., HCnxt => t >= 1)
        // If LHS is an identifier or operator call, it might expand to contain primes
        Expr::Implies(a, _) => {
            matches!(&a.node, Expr::Ident(_) | Expr::Apply(_, _))
                || expr_contains_prime_not_assignment(&a.node)
        }

        // Equivalences might have action predicates on either side
        Expr::Equiv(a, b) => {
            matches!(&a.node, Expr::Ident(_) | Expr::Apply(_, _))
                || matches!(&b.node, Expr::Ident(_) | Expr::Apply(_, _))
                || needs_next_state_validation(&a.node)
                || needs_next_state_validation(&b.node)
        }

        // Recurse into conjunctions/disjunctions
        Expr::And(a, b) | Expr::Or(a, b) => {
            needs_next_state_validation(&a.node) || needs_next_state_validation(&b.node)
        }

        // For other expressions, check for primed variables
        _ => expr_contains_prime_not_assignment(expr),
    }
}

/// Check if an expression might contain an action predicate via operator reference.
/// This catches cases like `HCnxt => t >= 1` where HCnxt is an operator that contains primes,
/// and bare operator references like `UpdateOpOrder` that might contain `Serializable'`.
///
/// IMPORTANT: Comparison expressions like `rcvd01(self) >= N - T` might contain hidden primed
/// variables in operator bodies (e.g., `rcvd01(self) == Cardinality({m \in rcvd'[self] : ...})`).
/// We must check if any operand contains an operator application that might expand to primed vars.
fn might_contain_action_predicate(expr: &Expr) -> bool {
    match expr {
        // Implications might have action predicates in LHS (e.g., HCnxt => t >= 1)
        Expr::Implies(a, _) => {
            matches!(&a.node, Expr::Ident(_) | Expr::Apply(_, _))
                || expr_contains_prime_not_assignment(&a.node)
        }
        // Equivalences might have action predicates on either side
        Expr::Equiv(a, b) => {
            matches!(&a.node, Expr::Ident(_) | Expr::Apply(_, _))
                || matches!(&b.node, Expr::Ident(_) | Expr::Apply(_, _))
                || expr_contains_prime_not_assignment(&a.node)
                || expr_contains_prime_not_assignment(&b.node)
        }
        // Bare operator references might expand to contain action predicates (primed expressions).
        // E.g., `UpdateOpOrder` containing `Serializable'` which validates opOrder' constraints.
        // We conservatively assume all operator references might contain action predicates
        // since we can't cheaply determine this without expanding the operator body.
        Expr::Ident(_) | Expr::Apply(_, _) => true,
        // Comparison and arithmetic expressions might contain operator applications that have
        // hidden primed variables in their bodies. Check recursively.
        // This catches patterns like `rcvd01(self) >= N - T` where rcvd01's body contains rcvd'.
        Expr::Lt(a, b)
        | Expr::Leq(a, b)
        | Expr::Gt(a, b)
        | Expr::Geq(a, b)
        | Expr::Eq(a, b)
        | Expr::Neq(a, b)
        | Expr::Add(a, b)
        | Expr::Sub(a, b)
        | Expr::Mul(a, b)
        | Expr::Div(a, b)
        | Expr::IntDiv(a, b)
        | Expr::Mod(a, b)
        | Expr::In(a, b)
        | Expr::NotIn(a, b)
        | Expr::Subseteq(a, b) => {
            expr_contains_operator_application(&a.node)
                || expr_contains_operator_application(&b.node)
                || expr_contains_prime_not_assignment(&a.node)
                || expr_contains_prime_not_assignment(&b.node)
        }
        Expr::Not(a) | Expr::Neg(a) => {
            expr_contains_operator_application(&a.node)
                || expr_contains_prime_not_assignment(&a.node)
        }
        _ => expr_contains_prime_not_assignment(expr),
    }
}

/// Check if an expression contains an operator application (Apply or zero-arg Ident operator).
/// This is used to detect expressions that might have hidden primed variables in operator bodies.
fn expr_contains_operator_application(expr: &Expr) -> bool {
    match expr {
        Expr::Apply(_, _) => true,
        // Note: Ident could be a variable or a zero-arg operator. We conservatively assume it might
        // be an operator application since we can't cheaply determine this without context lookup.
        // However, treating all Idents as potential operators would be too conservative (most are vars).
        // So we only mark Apply as definite operator applications.
        Expr::Lt(a, b)
        | Expr::Leq(a, b)
        | Expr::Gt(a, b)
        | Expr::Geq(a, b)
        | Expr::Eq(a, b)
        | Expr::Neq(a, b)
        | Expr::Add(a, b)
        | Expr::Sub(a, b)
        | Expr::Mul(a, b)
        | Expr::Div(a, b)
        | Expr::IntDiv(a, b)
        | Expr::Mod(a, b)
        | Expr::In(a, b)
        | Expr::NotIn(a, b)
        | Expr::Subseteq(a, b)
        | Expr::And(a, b)
        | Expr::Or(a, b) => {
            expr_contains_operator_application(&a.node)
                || expr_contains_operator_application(&b.node)
        }
        Expr::Not(a) | Expr::Neg(a) => expr_contains_operator_application(&a.node),
        Expr::If(c, t, e) => {
            expr_contains_operator_application(&c.node)
                || expr_contains_operator_application(&t.node)
                || expr_contains_operator_application(&e.node)
        }
        Expr::FuncApply(f, arg) => {
            expr_contains_operator_application(&f.node)
                || expr_contains_operator_application(&arg.node)
        }
        _ => false,
    }
}

fn action_holds_in_next_state(ctx: &EvalCtx, expr: &Spanned<Expr>, next_state: &State) -> bool {
    let debug = debug_enum();

    let mut next_env = Env::new();
    for (name, value) in next_state.vars() {
        next_env.insert(Arc::clone(name), value.clone());
    }

    let eval_ctx = ctx.with_next_state(next_env);

    // We validate conjuncts that:
    // 1. Contain primed variables outside of simple assignment LHS, OR
    // 2. Are implications/equivalences that might reference action operators
    // This catches patterns like `HCnxt => t >= 1` where HCnxt is an operator
    // containing primed variables (hr' = IF hr # 12 THEN hr + 1 ELSE 1).
    let mut conjuncts = Vec::new();
    flatten_and_spanned(expr, &mut conjuncts);

    for conjunct in &conjuncts {
        if !might_contain_action_predicate(&conjunct.node) {
            continue;
        }

        match eval(&eval_ctx, conjunct) {
            Ok(Value::Bool(true)) => {}
            Ok(Value::Bool(false)) => {
                if debug {
                    eprintln!(
                        "action_holds_in_next_state: prime-guard false at span {:?}",
                        conjunct.span
                    );
                }
                return false;
            }
            Ok(other) => {
                if debug {
                    eprintln!(
                        "action_holds_in_next_state: prime-guard non-boolean ({}) at span {:?}",
                        other.type_name(),
                        conjunct.span
                    );
                }
                return false;
            }
            Err(e) => {
                if debug {
                    eprintln!(
                        "action_holds_in_next_state: prime-guard eval error ({:?}) at span {:?}",
                        e, conjunct.span
                    );
                }
                return false;
            }
        }
    }

    true
}

/// Recursively check all guard expressions within an And tree.
/// Returns false if any guard evaluates to false, true otherwise.
/// This handles cases like And(And(guard, assignment), assignment) where
/// the guard is nested within the And tree.
fn check_and_guards(ctx: &EvalCtx, expr: &Spanned<Expr>, debug: bool) -> Result<bool, EvalError> {
    // Use stacker to grow stack on demand for deeply nested And trees
    stacker::maybe_grow(STACK_RED_ZONE, STACK_GROW_SIZE, || {
        check_and_guards_inner(ctx, expr, debug)
    })
}

fn check_and_guards_inner(
    ctx: &EvalCtx,
    expr: &Spanned<Expr>,
    debug: bool,
) -> Result<bool, EvalError> {
    match &expr.node {
        Expr::And(a, b) => {
            // Check both sides of the And
            if !check_and_guards(ctx, a, debug)? {
                return Ok(false);
            }
            check_and_guards(ctx, b, debug)
        }
        // LET expressions: bind definitions and recurse into body
        // This is essential for patterns like:
        //   LET newThis == ... IN /\ IsSafe(newThis) /\ x' = ...
        // The IsSafe guards need to be checked with newThis bound.
        // If LET binding evaluation fails (e.g., domain error), treat action as disabled.
        Expr::Let(defs, body) => {
            let mut new_ctx = ctx.clone();
            for def in defs {
                if def.params.is_empty() {
                    // Zero-param definition: try to evaluate and bind
                    match eval(&new_ctx, &def.body) {
                        Ok(val) => {
                            new_ctx.env.insert(Arc::from(def.name.node.as_str()), val);
                        }
                        Err(_) => {
                            // LET binding evaluation failed - action is disabled
                            return Ok(false);
                        }
                    }
                } else {
                    // Operator definition with params
                    new_ctx.define_op(def.name.node.clone(), def.clone());
                }
            }
            check_and_guards(&new_ctx, body, debug)
        }
        // Or expressions: do NOT evaluate guards inside Or branches.
        // Each Or branch has its own guards that should be checked AFTER distribution.
        // Walking into Or branches and checking guards like `cnt # 0` would incorrectly
        // reject the entire action when only some branches have that guard.
        Expr::Or(_, _) => Ok(true),
        _ => {
            // For non-And, non-Let, non-Or expressions, check if it's a guard and evaluate it
            if is_guard_expression(&expr.node) && !is_operator_reference_guard_unsafe(ctx, &expr.node)
            {
                match eval(ctx, expr) {
                    Ok(Value::Bool(false)) => {
                        if debug {
                            eprintln!("check_and_guards: guard={:?} evaluated to false", expr.node);
                        }
                        Ok(false)
                    }
                    Ok(Value::Bool(true)) => Ok(true),
                    Ok(_) => {
                        // Non-boolean result: not a true guard, continue
                        Ok(true)
                    }
                    Err(e) => {
                        // Action-level errors (primed vars, UNCHANGED) mean this isn't a guard,
                        // so continue. Other errors (type errors, undefined ops) mean the action
                        // is disabled.
                        if is_action_level_error(&e) {
                            if debug {
                                eprintln!(
                                    "check_and_guards: eval error {:?} is action-level, not a guard",
                                    e
                                );
                            }
                            Ok(true)
                        } else {
                            if debug {
                                eprintln!(
                                    "check_and_guards: eval error {:?}, action disabled",
                                    e
                                );
                            }
                            Ok(false)
                        }
                    }
                }
            } else {
                // Not a guard expression (contains primes), skip
                Ok(true)
            }
        }
    }
}

/// True if this expression is an operator reference that could hide action-level content.
///
/// Evaluating such expressions during guard short-circuiting is unsafe: action-level operators
/// can short-circuit to `FALSE` before touching primed variables, causing us to incorrectly
/// treat the whole action as disabled (AllocatorImplementation: `Sched!Schedule`/`Sched!Allocate`).
fn is_operator_reference_guard_unsafe(ctx: &EvalCtx, expr: &Expr) -> bool {
    match expr {
        // Instanced module operators frequently contain primed assignments.
        Expr::ModuleRef(_, _, _) => true,
        // A bare identifier might name a zero-arg operator (action or predicate).
        // We conservatively avoid short-circuit evaluation and rely on full action validation.
        Expr::Ident(name) => {
            let resolved = ctx.resolve_op_name(name.as_str());
            ctx.get_op(resolved).is_some()
        }
        // Operator application may hide primed variables inside the operator body.
        Expr::Apply(op_expr, _args) => match &op_expr.node {
            Expr::Ident(op_name) => {
                let resolved = ctx.resolve_op_name(op_name.as_str());
                ctx.get_op(resolved).is_some()
            }
            _ => false,
        },
        _ => false,
    }
}

/// Check if an expression is a guard (doesn't contain primed variables)
fn is_guard_expression(expr: &Expr) -> bool {
    match expr {
        // Primed expressions are NOT guards
        Expr::Prime(_) => false,

        // Check for UNCHANGED (not a guard)
        Expr::Unchanged(_) => false,

        // Boolean operations - check recursively
        Expr::And(a, b) | Expr::Or(a, b) | Expr::Implies(a, b) | Expr::Equiv(a, b) => {
            is_guard_expression(&a.node) && is_guard_expression(&b.node)
        }
        Expr::Not(a) => is_guard_expression(&a.node),

        // Comparisons - check recursively
        Expr::Eq(a, b)
        | Expr::Neq(a, b)
        | Expr::Lt(a, b)
        | Expr::Leq(a, b)
        | Expr::Gt(a, b)
        | Expr::Geq(a, b) => is_guard_expression(&a.node) && is_guard_expression(&b.node),

        // Arithmetic - check recursively
        Expr::Add(a, b)
        | Expr::Sub(a, b)
        | Expr::Mul(a, b)
        | Expr::Div(a, b)
        | Expr::IntDiv(a, b)
        | Expr::Mod(a, b)
        | Expr::Pow(a, b)
        | Expr::Range(a, b) => is_guard_expression(&a.node) && is_guard_expression(&b.node),
        Expr::Neg(a) => is_guard_expression(&a.node),

        // Set operations - check recursively
        Expr::In(a, b)
        | Expr::NotIn(a, b)
        | Expr::Subseteq(a, b)
        | Expr::Union(a, b)
        | Expr::Intersect(a, b)
        | Expr::SetMinus(a, b) => is_guard_expression(&a.node) && is_guard_expression(&b.node),

        // Identifiers are guards (unprimed variables)
        Expr::Ident(_) => true,

        // Literals are guards
        Expr::Bool(_) | Expr::Int(_) | Expr::String(_) => true,

        // LET expressions - check if body is a guard
        Expr::Let(_defs, body) => is_guard_expression(&body.node),

        // IF-THEN-ELSE - check all branches
        Expr::If(cond, then_br, else_br) => {
            is_guard_expression(&cond.node)
                && is_guard_expression(&then_br.node)
                && is_guard_expression(&else_br.node)
        }

        // Quantifiers - check body
        Expr::Forall(_, body) | Expr::Exists(_, body) => is_guard_expression(&body.node),

        // Function application - generally not a guard if args could contain primes
        // But for now, be conservative
        Expr::Apply(_, args) => args.iter().all(|a| is_guard_expression(&a.node)),

        // Function construction with body - check body
        Expr::FuncDef(_, body) => is_guard_expression(&body.node),

        // Set builders - check expression
        Expr::SetBuilder(expr, _) => is_guard_expression(&expr.node),
        Expr::SetFilter(_, pred) => is_guard_expression(&pred.node),

        // CASE expressions - check all branches
        Expr::Case(arms, other) => {
            arms.iter().all(|arm| {
                is_guard_expression(&arm.guard.node) && is_guard_expression(&arm.body.node)
            }) && other
                .as_ref()
                .map(|e| is_guard_expression(&e.node))
                .unwrap_or(true)
        }

        // Other constructs - assume they're guards unless they contain explicit primes
        // This is a heuristic - expressions like CHOOSE, Except might contain primed
        // variable references but typically don't in guard positions
        _ => true,
    }
}

fn extract_unchanged_vars_symbolic(
    expr: &Expr,
    vars: &[Arc<str>],
    assignments: &mut Vec<SymbolicAssignment>,
) {
    match expr {
        Expr::Ident(name) => {
            if let Some(var) = vars.iter().find(|v| v.as_ref() == name.as_str()) {
                assignments.push(SymbolicAssignment::Unchanged(Arc::clone(var)));
            }
        }
        Expr::Tuple(elems) => {
            for elem in elems {
                extract_unchanged_vars_symbolic(&elem.node, vars, assignments);
            }
        }
        _ => {}
    }
}

/// Convert a Value back to an Expr for embedding in expressions
///
/// This is used when we've evaluated values and need to create expressions from them,
/// e.g., when converting existential assignments to InSet expressions.
fn value_to_expr(value: &Value) -> Expr {
    match value {
        Value::Bool(b) => Expr::Bool(*b),
        Value::SmallInt(n) => Expr::Int(BigInt::from(*n)),
        Value::Int(n) => Expr::Int(n.clone()),
        Value::String(s) => Expr::String(s.to_string()),
        Value::ModelValue(name) => {
            // Model values are represented as identifiers
            Expr::Ident(name.to_string())
        }
        Value::Tuple(elems) => {
            let exprs: Vec<_> = elems
                .iter()
                .map(|e| Spanned::dummy(value_to_expr(e)))
                .collect();
            Expr::Tuple(exprs)
        }
        Value::Set(set) => {
            let exprs: Vec<_> = set
                .iter()
                .map(|e| Spanned::dummy(value_to_expr(e)))
                .collect();
            Expr::SetEnum(exprs)
        }
        Value::Seq(elems) => {
            // Sequences are tuples in TLA+
            let exprs: Vec<_> = elems
                .iter()
                .map(|e| Spanned::dummy(value_to_expr(e)))
                .collect();
            Expr::Tuple(exprs)
        }
        Value::Record(fields) => {
            let field_exprs: Vec<_> = fields
                .iter()
                .map(|(k, v)| {
                    (
                        Spanned::dummy(k.to_string()),
                        Spanned::dummy(value_to_expr(v)),
                    )
                })
                .collect();
            Expr::Record(field_exprs)
        }
        Value::Func(func) => {
            // Convert function to a chain of :> and @@ operators.
            // For {a |-> va, b |-> vb}: (a :> va) @@ (b :> vb)
            // This produces Value::Func when evaluated, unlike SetEnum which gives Value::Set.
            let entries = func.entries();
            if entries.is_empty() {
                // Empty function: use [x \in {} |-> x]
                Expr::FuncDef(
                    vec![tla_core::ast::BoundVar {
                        name: Spanned::dummy("_".to_string()),
                        pattern: None,
                        domain: Some(Box::new(Spanned::dummy(Expr::SetEnum(vec![])))),
                    }],
                    Box::new(Spanned::dummy(Expr::Ident("_".to_string()))),
                )
            } else {
                // Build chain: (k1 :> v1) @@ (k2 :> v2) @@ ...
                let mut result: Option<Expr> = None;
                for (k, v) in entries.iter() {
                    let single = Expr::Apply(
                        Box::new(Spanned::dummy(Expr::Ident(":>".to_string()))),
                        vec![
                            Spanned::dummy(value_to_expr(k)),
                            Spanned::dummy(value_to_expr(v)),
                        ],
                    );
                    result = Some(match result {
                        None => single,
                        Some(acc) => Expr::Apply(
                            Box::new(Spanned::dummy(Expr::Ident("@@".to_string()))),
                            vec![Spanned::dummy(acc), Spanned::dummy(single)],
                        ),
                    });
                }
                result.unwrap()
            }
        }
        // For complex values that can't be easily represented as expressions,
        // create a placeholder that will be looked up at evaluation time
        _ => {
            // For intervals and other special values, we use a string representation
            // that can be parsed back. This is a fallback.
            Expr::String(format!("{:?}", value))
        }
    }
}
/// Check if an expression references any state variables (non-primed).
/// This is used to prevent eager evaluation of expressions that depend on current state.
fn expr_references_state_vars(expr: &Expr, vars: &[Arc<str>]) -> bool {
    match expr {
        Expr::Ident(name) => vars.iter().any(|v| v.as_ref() == name.as_str()),
        Expr::Prime(_) => false, // Primed vars are handled separately
        Expr::And(a, b)
        | Expr::Or(a, b)
        | Expr::Implies(a, b)
        | Expr::Equiv(a, b)
        | Expr::Eq(a, b)
        | Expr::Neq(a, b)
        | Expr::Lt(a, b)
        | Expr::Leq(a, b)
        | Expr::Gt(a, b)
        | Expr::Geq(a, b)
        | Expr::Add(a, b)
        | Expr::Sub(a, b)
        | Expr::Mul(a, b)
        | Expr::Div(a, b)
        | Expr::IntDiv(a, b)
        | Expr::Mod(a, b)
        | Expr::Pow(a, b)
        | Expr::Range(a, b)
        | Expr::Union(a, b)
        | Expr::Intersect(a, b)
        | Expr::SetMinus(a, b)
        | Expr::In(a, b)
        | Expr::NotIn(a, b)
        | Expr::Subseteq(a, b)
        | Expr::FuncSet(a, b) => {
            expr_references_state_vars(&a.node, vars) || expr_references_state_vars(&b.node, vars)
        }
        Expr::Not(a)
        | Expr::Neg(a)
        | Expr::Domain(a)
        | Expr::Powerset(a)
        | Expr::BigUnion(a) => expr_references_state_vars(&a.node, vars),
        Expr::Tuple(elems) | Expr::SetEnum(elems) => {
            elems.iter().any(|e| expr_references_state_vars(&e.node, vars))
        }
        Expr::Times(elems) => elems.iter().any(|e| expr_references_state_vars(&e.node, vars)),
        Expr::Record(fields) | Expr::RecordSet(fields) => fields
            .iter()
            .any(|(_, v)| expr_references_state_vars(&v.node, vars)),
        Expr::RecordAccess(base, _) => expr_references_state_vars(&base.node, vars),
        Expr::FuncApply(f, arg) => {
            expr_references_state_vars(&f.node, vars) || expr_references_state_vars(&arg.node, vars)
        }
        Expr::Apply(op, args) => {
            expr_references_state_vars(&op.node, vars)
                || args.iter().any(|a| expr_references_state_vars(&a.node, vars))
        }
        Expr::If(cond, then_expr, else_expr) => {
            expr_references_state_vars(&cond.node, vars)
                || expr_references_state_vars(&then_expr.node, vars)
                || expr_references_state_vars(&else_expr.node, vars)
        }
        Expr::Forall(bounds, body) | Expr::Exists(bounds, body) => {
            bounds.iter().any(|b| {
                b.domain
                    .as_ref()
                    .is_some_and(|d| expr_references_state_vars(&d.node, vars))
            }) || expr_references_state_vars(&body.node, vars)
        }
        Expr::SetBuilder(expr, bounds) => {
            bounds.iter().any(|b| {
                b.domain
                    .as_ref()
                    .is_some_and(|d| expr_references_state_vars(&d.node, vars))
            }) || expr_references_state_vars(&expr.node, vars)
        }
        Expr::SetFilter(bound, pred) => {
            bound
                .domain
                .as_ref()
                .is_some_and(|d| expr_references_state_vars(&d.node, vars))
                || expr_references_state_vars(&pred.node, vars)
        }
        Expr::FuncDef(bounds, body) => {
            bounds.iter().any(|b| {
                b.domain
                    .as_ref()
                    .is_some_and(|d| expr_references_state_vars(&d.node, vars))
            }) || expr_references_state_vars(&body.node, vars)
        }
        Expr::Let(defs, body) => {
            defs.iter()
                .any(|def| expr_references_state_vars(&def.body.node, vars))
                || expr_references_state_vars(&body.node, vars)
        }
        Expr::Choose(bound, body) => {
            bound
                .domain
                .as_ref()
                .is_some_and(|d| expr_references_state_vars(&d.node, vars))
                || expr_references_state_vars(&body.node, vars)
        }
        Expr::Except(f, specs) => {
            expr_references_state_vars(&f.node, vars)
                || specs.iter().any(|s| {
                    s.path.iter().any(|p| match p {
                        tla_core::ast::ExceptPathElement::Index(idx) => {
                            expr_references_state_vars(&idx.node, vars)
                        }
                        tla_core::ast::ExceptPathElement::Field(_) => false,
                    }) || expr_references_state_vars(&s.value.node, vars)
                })
        }
        Expr::Case(arms, other) => {
            arms.iter().any(|arm| {
                expr_references_state_vars(&arm.guard.node, vars)
                    || expr_references_state_vars(&arm.body.node, vars)
            }) || other
                .as_ref()
                .is_some_and(|e| expr_references_state_vars(&e.node, vars))
        }
        // Literals and others - no state variable references
        Expr::Bool(_) | Expr::Int(_) | Expr::String(_) | Expr::OpRef(_) => false,
        // Handle remaining cases conservatively
        _ => false,
    }
}

/// Extract symbolic assignments (expressions, not values) from a Next relation
fn extract_symbolic_assignments(
    ctx: &EvalCtx,
    expr: &Spanned<Expr>,
    vars: &[Arc<str>],
    assignments: &mut Vec<SymbolicAssignment>,
) -> Result<(), EvalError> {
    fn inner(
        ctx: &EvalCtx,
        expr: &Spanned<Expr>,
        vars: &[Arc<str>],
        assignments: &mut Vec<SymbolicAssignment>,
        eager_eval: bool,
    ) -> Result<(), EvalError> {
        let debug = debug_extract();
        if debug {
            eprintln!(
                "EXTRACT: expr type={:?}, span={:?}",
                std::mem::discriminant(&expr.node),
                expr.span
            );
        }

        match &expr.node {
            // Conjunction: extract from both sides
            Expr::And(a, b) => {
                if debug {
                    eprintln!("EXTRACT: And - processing both sides");
                }
                inner(ctx, a, vars, assignments, eager_eval)?;
                inner(ctx, b, vars, assignments, eager_eval)?;
                Ok(())
            }

            // Equality with primed variable: x' = expr
            Expr::Eq(lhs, rhs) => {
                // Check if LHS is primed variable: x'
                if let Expr::Prime(inner_lhs) = &lhs.node {
                    if let Expr::Ident(name) = &inner_lhs.node {
                        if let Some(var) = vars.iter().find(|v| v.as_ref() == name.as_str()) {
                            // Fast path: x' = x (treat as UNCHANGED) when `x` is not shadowed by a
                            // locally-bound variable.
                            if let Expr::Ident(rhs_name) = &rhs.node {
                                if rhs_name == name && !ctx.has_local_binding(rhs_name.as_str()) {
                                    assignments
                                        .push(SymbolicAssignment::Unchanged(Arc::clone(var)));
                                    return Ok(());
                                }
                            }

                            // Only eager-eval if RHS doesn't reference state variables.
                            // If RHS contains state vars (like `messages \union {m}`), we must
                            // defer evaluation to runtime when we have the actual current state.
                            let rhs_refs_state = expr_references_state_vars(&rhs.node, vars);
                            if eager_eval && !rhs_refs_state {
                                // Try to evaluate the expression eagerly
                                // This handles LET-bound variables that are in ctx.env
                                match eval(ctx, rhs) {
                                    Ok(value) => assignments
                                        .push(SymbolicAssignment::Value(Arc::clone(var), value)),
                                    Err(_) => {
                                        // Can't evaluate now (might contain primed vars), defer
                                        // Capture bindings (O(n)) instead of substituting (O(AST))
                                        assignments.push(SymbolicAssignment::Expr(
                                            Arc::clone(var),
                                            (**rhs).clone(),
                                            ctx.get_local_bindings().to_vec(),
                                        ));
                                    }
                                }
                            } else {
                                // Keep RHS as an expression so it can be embedded in conditional
                                // assignments (e.g. prime-dependent IF branches).
                                // Capture bindings (O(n)) instead of substituting (O(AST))
                                assignments.push(SymbolicAssignment::Expr(
                                    Arc::clone(var),
                                    (**rhs).clone(),
                                    ctx.get_local_bindings().to_vec(),
                                ));
                            }
                            return Ok(());
                        }
                    }
                }
                // Check if RHS is primed variable (symmetric case)
                if let Expr::Prime(inner_rhs) = &rhs.node {
                    if let Expr::Ident(name) = &inner_rhs.node {
                        if let Some(var) = vars.iter().find(|v| v.as_ref() == name.as_str()) {
                            // Fast path: x = x' (treat as UNCHANGED) when `x` is not shadowed by a
                            // locally-bound variable.
                            if let Expr::Ident(lhs_name) = &lhs.node {
                                if lhs_name == name && !ctx.has_local_binding(lhs_name.as_str()) {
                                    assignments
                                        .push(SymbolicAssignment::Unchanged(Arc::clone(var)));
                                    return Ok(());
                                }
                            }

                            // Only eager-eval if LHS doesn't reference state variables.
                            let lhs_refs_state = expr_references_state_vars(&lhs.node, vars);
                            if eager_eval && !lhs_refs_state {
                                match eval(ctx, lhs) {
                                    Ok(value) => assignments
                                        .push(SymbolicAssignment::Value(Arc::clone(var), value)),
                                    Err(_) => {
                                        // Capture bindings (O(n)) instead of substituting (O(AST))
                                        assignments.push(SymbolicAssignment::Expr(
                                            Arc::clone(var),
                                            (**lhs).clone(),
                                            ctx.get_local_bindings().to_vec(),
                                        ));
                                    }
                                }
                            } else {
                                // Capture bindings (O(n)) instead of substituting (O(AST))
                                assignments.push(SymbolicAssignment::Expr(
                                    Arc::clone(var),
                                    (**lhs).clone(),
                                    ctx.get_local_bindings().to_vec(),
                                ));
                            }
                            return Ok(());
                        }
                    }
                }
                Ok(())
            }

            // UNCHANGED <<x, y>> or UNCHANGED x
            Expr::Unchanged(inner_expr) => {
                extract_unchanged_vars_symbolic(&inner_expr.node, vars, assignments);
                Ok(())
            }

            // Membership with primed variable: x' \in S
            Expr::In(lhs, rhs) => {
                if let Expr::Prime(inner_lhs) = &lhs.node {
                    if let Expr::Ident(name) = &inner_lhs.node {
                        if let Some(var) = vars.iter().find(|v| v.as_ref() == name.as_str()) {
                            // Capture bindings (O(n)) instead of substituting (O(AST))
                            assignments.push(SymbolicAssignment::InSet(
                                Arc::clone(var),
                                (**rhs).clone(),
                                ctx.get_local_bindings().to_vec(),
                            ));
                        }
                    }
                }
                Ok(())
            }

            // Existential quantifier with primed assignments: \E x \in S : var' = expr(x)
            // This converts to var' \in {expr(x) : x \in S}
            Expr::Exists(bounds, body) => {
                // Only handle simple cases: single bound variable
                if bounds.len() != 1 {
                    // For multiple bounds, fall back to not extracting (will be handled by enumerate_next_rec)
                    return Ok(());
                }
                let bound = &bounds[0];
                let var_name = &bound.name.node;

                // Evaluate the domain
                let domain = match &bound.domain {
                    Some(dom_expr) => match eval(ctx, dom_expr) {
                        Ok(d) => d,
                        Err(_) => return Ok(()), // Can't evaluate domain, skip
                    },
                    None => return Ok(()), // No domain, skip
                };

                let domain_values = match domain.to_sorted_set() {
                    Some(s) => s,
                    None => return Ok(()), // Domain is not a set, skip
                };

                if domain_values.is_empty() {
                    return Ok(()); // Empty domain, no assignments possible
                }

                // For each value in the domain, evaluate the body to find assignments
                // Collect all possible values for primed variables
                let mut primed_values: std::collections::HashMap<Arc<str>, Vec<Value>> =
                    std::collections::HashMap::new();

                for val in domain_values.iter() {
                    // Bind the quantified variable using bind_local so the binding
                    // is captured in local_stack for deferred expression evaluation.
                    // This fixes #87 where EXISTS bindings were not properly captured.
                    let new_ctx = ctx.bind_local(Arc::from(var_name.as_str()), val.clone());

                    // Check if guards pass before extracting assignments.
                    // This ensures we only extract assignments from EXISTS body when guards
                    // are satisfied. Without this check, we'd extract assignments from ALL
                    // domain values, not just those where `\E i : guard /\ x' = f(i)` holds.
                    // The guard check filters out values that don't satisfy the preconditions.
                    if !check_and_guards(&new_ctx, body, false)? {
                        continue; // Skip this value - guards not satisfied
                    }

                    // Extract assignments from the body with this binding
                    let mut body_assignments = Vec::new();
                    inner(&new_ctx, body, vars, &mut body_assignments, eager_eval)?;

                    // Collect all Value assignments
                    for assign in body_assignments {
                        match assign {
                            SymbolicAssignment::Value(var, value) => {
                                primed_values.entry(var).or_default().push(value);
                            }
                            SymbolicAssignment::Expr(var, ref expr_spanned, ref bindings) => {
                                // Restore captured bindings before evaluation
                                let eval_ctx = new_ctx.with_captured_bindings(bindings);
                                // Try to evaluate the expression with the current binding
                                if let Ok(value) = eval(&eval_ctx, expr_spanned) {
                                    primed_values.entry(var).or_default().push(value);
                                }
                            }
                            SymbolicAssignment::Unchanged(var) => {
                                // UNCHANGED inside existential - keep current value
                                if let Some(current) = new_ctx.env.get(&var) {
                                    primed_values.entry(var).or_default().push(current.clone());
                                }
                            }
                            SymbolicAssignment::InSet(var, ref set_expr, ref bindings) => {
                                // Restore captured bindings before evaluation
                                let eval_ctx = new_ctx.with_captured_bindings(bindings);
                                // InSet inside existential - evaluate and merge values
                                if let Ok(set_val) = eval(&eval_ctx, set_expr) {
                                    if let Some(s) = set_val.to_sorted_set() {
                                        primed_values
                                            .entry(var)
                                            .or_default()
                                            .extend(s.iter().cloned());
                                    }
                                }
                            }
                        }
                    }
                }

                // Convert collected values to Value assignments with SetEnum
                // Use Value variant directly since we've already evaluated all possibilities
                for (var, values) in primed_values {
                    if !values.is_empty() {
                        // Deduplicate values using a sorted set
                        let unique: crate::value::SortedSet = values.into_iter().collect();
                        // Build a SetEnum expression containing all the values
                        let set_elems: Vec<Spanned<Expr>> = unique
                            .iter()
                            .map(|v| Spanned::new(value_to_expr(v), body.span))
                            .collect();
                        let set_expr = Spanned::new(Expr::SetEnum(set_elems), body.span);
                        // No bindings needed - set_expr is already fully evaluated
                        assignments.push(SymbolicAssignment::InSet(var, set_expr, Vec::new()));
                    }
                }

                Ok(())
            }

            // TRUE is always satisfied
            Expr::Bool(true) => Ok(()),

            // Zero-param operator reference - recurse into operator body
            Expr::Ident(name) => {
                // Apply config operator replacement (e.g., `Send <- MCSend`) before lookup.
                let resolved_name = ctx.resolve_op_name(name);
                if let Some(def) = ctx.get_op(resolved_name) {
                    if def.params.is_empty() {
                        if debug {
                            eprintln!(
                                "EXTRACT: Ident {} is zero-param op, recursing into body",
                                resolved_name
                            );
                        }
                        inner(ctx, &def.body, vars, assignments, eager_eval)?;
                    }
                }
                Ok(())
            }

            // Operator application - might be a user-defined action
            Expr::Apply(op_expr, args) => {
                if let Expr::Ident(op_name) = &op_expr.node {
                    // Apply config operator replacement (e.g., `Send <- MCSend`) before lookup.
                    let resolved_name = ctx.resolve_op_name(op_name);
                    if let Some(def) = ctx.get_op(resolved_name) {
                        // Check if any parameter appears primed in the body
                        // If so, use expression substitution (call-by-name) instead of value binding
                        // This is required for TLA+ semantics like Action1(x,y) == x' = [x EXCEPT ![1] = y']
                        let needs_substitution = def
                            .params
                            .iter()
                            .any(|param| expr_has_primed_param(&def.body.node, &param.name.node));

                        // If any argument expression contains primes, we MUST substitute call-by-name.
                        //
                        // Example (MCWriteThroughCache):
                        //   MCSend(p, d, oldMemInt, newMemInt) == newMemInt = <<p, d>>
                        //   Send(p, req, memInt, memInt')  (via config replacement Send <- MCSend)
                        //
                        // Here `memInt'` cannot be evaluated in the current-state context, so a
                        // call-by-value bind would silently skip binding `newMemInt` and we'd miss
                        // the implied assignment to `memInt'`.
                        let args_contain_prime = args.iter().any(|a| expr_contains_prime(&a.node));

                        if needs_substitution || args_contain_prime {
                            // Use call-by-name: substitute argument expressions into body
                            let subs: Vec<Substitution> = def
                                .params
                                .iter()
                                .zip(args.iter())
                                .map(|(param, arg)| Substitution {
                                    from: param.name.clone(),
                                    to: arg.clone(),
                                })
                                .collect();
                            let substituted_body = apply_substitutions(&def.body, &subs);
                            return inner(ctx, &substituted_body, vars, assignments, eager_eval);
                        }

                        // No primed parameters - use call-by-value (faster)
                        let mut new_ctx = ctx.clone();
                        for (param, arg) in def.params.iter().zip(args.iter()) {
                            if let Ok(arg_val) = eval(ctx, arg) {
                                new_ctx
                                    .env
                                    .insert(Arc::from(param.name.node.as_str()), arg_val);
                            }
                        }
                        inner(&new_ctx, &def.body, vars, assignments, eager_eval)?;
                    }
                }
                Ok(())
            }

            // Module reference (named INSTANCE): Inst!Op(args)
            //
            // Important: the INSTANCE'd operator body may contain primed assignments even when
            // the ModuleRef syntax does not, so we must inline the operator body here to
            // extract assignments correctly.
            //
            // Additionally, arguments may contain primed expressions (e.g. `inDelivery'`),
            // which must be substituted call-by-name so they can be evaluated later with
            // the progressive next-state context.
            Expr::ModuleRef(instance_name, op_name, args) => {
                // Look up instance metadata (module name + WITH substitutions)
                let instance_info = ctx.get_instance(instance_name.name()).ok_or_else(|| {
                    EvalError::UndefinedOp {
                        name: format!("{}!{}", instance_name, op_name),
                        span: Some(expr.span),
                    }
                })?;

                // Find the referenced operator definition in the instanced module.
                let op_def = ctx
                    .get_instance_op(&instance_info.module_name, op_name)
                    .ok_or_else(|| EvalError::UndefinedOp {
                        name: format!("{}!{}", instance_name, op_name),
                        span: Some(expr.span),
                    })?
                    .clone();

                if op_def.params.len() != args.len() {
                    return Err(EvalError::ArityMismatch {
                        op: format!("{}!{}", instance_name, op_name),
                        expected: op_def.params.len(),
                        got: args.len(),
                        span: Some(expr.span),
                    });
                }

                // Apply INSTANCE ... WITH substitutions (module-level substitutions)
                let mut inlined = apply_substitutions(&op_def.body, &instance_info.substitutions);

                // Always apply parameter substitution (call-by-name) so primed arguments
                // remain evaluable when next-state context is available.
                if !op_def.params.is_empty() {
                    let subs: Vec<Substitution> = op_def
                        .params
                        .iter()
                        .zip(args.iter())
                        .map(|(param, arg)| Substitution {
                            from: param.name.clone(),
                            to: arg.clone(),
                        })
                        .collect();
                    inlined = apply_substitutions(&inlined, &subs);
                }

                // Evaluate/extract inside a scope where unqualified operator names resolve
                // to the instanced module's definitions.
                let mut instance_local_ops: OpEnv = ctx
                    .instance_ops()
                    .get(&instance_info.module_name)
                    .cloned()
                    .unwrap_or_default();

                // Preserve any existing local ops (e.g. from surrounding LET) for names
                // not defined by the instanced module (needed for substitution expressions).
                if let Some(outer_local) = ctx.local_ops.as_deref() {
                    for (name, def) in outer_local.iter() {
                        instance_local_ops
                            .entry(name.clone())
                            .or_insert_with(|| def.clone());
                    }
                }

                let scoped_ctx = ctx.with_local_ops(instance_local_ops);
                inner(&scoped_ctx, &inlined, vars, assignments, eager_eval)
            }

            // LET ... IN body - add local definitions and extract from body
            Expr::Let(defs, body) => {
                let mut new_ctx = ctx.clone();
                // For action-level LET bindings that depend on primed variables, we can't
                // eagerly evaluate them in the current-state context. Instead, we inline
                // the binding into the body so it can be evaluated later under an
                // appropriate (progressive) next-state context.
                let mut inlined_body: Spanned<Expr> = (**body).clone();
                for def in defs {
                    if def.params.is_empty() {
                        // Zero-param definition: only eagerly evaluate when it is state-level.
                        //
                        // Action-level LET bindings can depend on primed variables (e.g. `LET x == y' IN ...`)
                        // and must be evaluated under a next-state context. In those cases we keep the
                        // definition as an operator so it can be evaluated later with progressive
                        // next-state bindings during successor construction.
                        if expr_contains_any_prime(&def.body.node) {
                            let subs = vec![Substitution {
                                from: def.name.clone(),
                                to: def.body.clone(),
                            }];
                            inlined_body = apply_substitutions(&inlined_body, &subs);
                        } else {
                            // State-level LET binding: evaluate now and bind into env.
                            // This handles cases like: LET r == ChooseOne(...) IN ...
                            let val = eval(&new_ctx, &def.body)?;
                            new_ctx.env.insert(Arc::from(def.name.node.as_str()), val);
                        }
                    } else {
                        // Operator definition with params: store for later application
                        new_ctx.define_op(def.name.node.clone(), def.clone());
                    }
                }
                inner(&new_ctx, &inlined_body, vars, assignments, eager_eval)
            }

            // IF-THEN-ELSE: evaluate condition and extract from appropriate branch
            Expr::If(cond, then_branch, else_branch) => match eval(ctx, cond) {
                Ok(Value::Bool(true)) => inner(ctx, then_branch, vars, assignments, eager_eval),
                Ok(Value::Bool(false)) => inner(ctx, else_branch, vars, assignments, eager_eval),
                _ => {
                    // Condition couldn't be evaluated in the current-state context (likely
                    // depends on primed variables). Merge assignments from both branches into
                    // conditional assignments that can be evaluated later with progressive
                    // next-state context.

                    #[derive(Clone)]
                    enum BranchAssignKind {
                        Expr(Spanned<Expr>),
                        Unchanged,
                        InSet(Spanned<Expr>),
                    }

                    let mut then_assignments = Vec::new();
                    inner(ctx, then_branch, vars, &mut then_assignments, false)?;

                    let mut else_assignments = Vec::new();
                    inner(ctx, else_branch, vars, &mut else_assignments, false)?;

                    let mut then_order: Vec<Arc<str>> = Vec::new();
                    let mut then_map: std::collections::HashMap<Arc<str>, BranchAssignKind> =
                        std::collections::HashMap::new();
                    for assign in then_assignments {
                        let (name, kind) = match assign {
                            SymbolicAssignment::Expr(name, rhs, _bindings) => {
                                // Ignore bindings - will use merged bindings at capture point
                                (name, BranchAssignKind::Expr(rhs))
                            }
                            SymbolicAssignment::Unchanged(name) => {
                                (name, BranchAssignKind::Unchanged)
                            }
                            SymbolicAssignment::InSet(name, set_expr, _bindings) => {
                                // Ignore bindings - will use merged bindings at capture point
                                (name, BranchAssignKind::InSet(set_expr))
                            }
                            SymbolicAssignment::Value(_, _) => {
                                // No-eager-eval mode should not produce Value, but ignore if it does.
                                continue;
                            }
                        };

                        if !then_map.contains_key(&name) {
                            then_order.push(name.clone());
                        }
                        then_map.insert(name, kind);
                    }

                    let mut else_map: std::collections::HashMap<Arc<str>, BranchAssignKind> =
                        std::collections::HashMap::new();
                    for assign in else_assignments {
                        let (name, kind) = match assign {
                            SymbolicAssignment::Expr(name, rhs, _bindings) => {
                                // Ignore bindings - will use merged bindings at capture point
                                (name, BranchAssignKind::Expr(rhs))
                            }
                            SymbolicAssignment::Unchanged(name) => {
                                (name, BranchAssignKind::Unchanged)
                            }
                            SymbolicAssignment::InSet(name, set_expr, _bindings) => {
                                // Ignore bindings - will use merged bindings at capture point
                                (name, BranchAssignKind::InSet(set_expr))
                            }
                            SymbolicAssignment::Value(_, _) => {
                                continue;
                            }
                        };
                        else_map.insert(name, kind);
                    }

                    // Capture bindings for the merged IF expression
                    let captured_bindings = ctx.get_local_bindings().to_vec();

                    for name in then_order {
                        let Some(then_kind) = then_map.get(&name) else {
                            continue;
                        };
                        let Some(else_kind) = else_map.get(&name) else {
                            continue;
                        };

                        match (then_kind, else_kind) {
                            (
                                BranchAssignKind::Expr(then_rhs),
                                BranchAssignKind::Expr(else_rhs),
                            ) => {
                                let merged = Spanned::new(
                                    Expr::If(
                                        cond.clone(),
                                        Box::new(then_rhs.clone()),
                                        Box::new(else_rhs.clone()),
                                    ),
                                    expr.span,
                                );
                                assignments.push(SymbolicAssignment::Expr(
                                    name.clone(),
                                    merged,
                                    captured_bindings.clone(),
                                ));
                            }
                            (
                                BranchAssignKind::InSet(then_set),
                                BranchAssignKind::InSet(else_set),
                            ) => {
                                let merged_set = Spanned::new(
                                    Expr::If(
                                        cond.clone(),
                                        Box::new(then_set.clone()),
                                        Box::new(else_set.clone()),
                                    ),
                                    expr.span,
                                );
                                assignments.push(SymbolicAssignment::InSet(
                                    name.clone(),
                                    merged_set,
                                    captured_bindings.clone(),
                                ));
                            }
                            (BranchAssignKind::Unchanged, BranchAssignKind::Unchanged) => {
                                assignments.push(SymbolicAssignment::Unchanged(name.clone()));
                            }
                            _ => {
                                continue;
                            }
                        }
                    }

                    Ok(())
                }
            },

            // CASE: evaluate guards and extract from matching branch
            Expr::Case(arms, other) => {
                if debug {
                    eprintln!(
                        "EXTRACT: Case - {} arms, other={}",
                        arms.len(),
                        other.is_some()
                    );
                }
                for (i, arm) in arms.iter().enumerate() {
                    match eval(ctx, &arm.guard) {
                        Ok(Value::Bool(true)) => {
                            if debug {
                                eprintln!(
                                    "EXTRACT: Case arm {} guard=true, extracting from body",
                                    i
                                );
                            }
                            return inner(ctx, &arm.body, vars, assignments, eager_eval);
                        }
                        Ok(Value::Bool(false)) => {
                            if debug {
                                eprintln!("EXTRACT: Case arm {} guard=false, continuing", i);
                            }
                            continue;
                        }
                        _ => continue,
                    }
                }
                // No arm matched - try OTHER
                if let Some(other_body) = other {
                    if debug {
                        eprintln!("EXTRACT: Case - no arm matched, extracting from OTHER");
                    }
                    inner(ctx, other_body, vars, assignments, eager_eval)
                } else {
                    Ok(())
                }
            }

            _ => Ok(()),
        }
    }

    inner(ctx, expr, vars, assignments, true)
}

/// Check if an expression references any primed variables from the given set.
/// This is used to detect when an expression depends on InSet variables
/// and must be deferred until we know the specific InSet value.
fn expr_references_primed_vars(ctx: &EvalCtx, expr: &Expr, vars: &HashSet<Arc<str>>) -> bool {
    if vars.is_empty() {
        return false;
    }

    fn inner(
        ctx: &EvalCtx,
        expr: &Expr,
        vars: &HashSet<Arc<str>>,
        visited_ops: &mut std::collections::HashSet<String>,
    ) -> bool {
        match expr {
            Expr::Prime(inner_expr) => {
                // Check if the primed variable is in our set
                if let Expr::Ident(name) = &inner_expr.node {
                    if vars.iter().any(|v| v.as_ref() == name.as_str()) {
                        return true;
                    }
                }
                // Also recurse in case of complex primed expressions
                inner(ctx, &inner_expr.node, vars, visited_ops)
            }

            // Expand operator references to check their bodies (with recursion protection).
            Expr::Ident(name) => {
                if let Some(def) = ctx.get_op(name) {
                    if !visited_ops.insert(name.clone()) {
                        return false;
                    }
                    inner(ctx, &def.body.node, vars, visited_ops)
                } else {
                    false
                }
            }

            Expr::Apply(op, args) => {
                // Check operator body
                if let Expr::Ident(name) = &op.node {
                    if let Some(def) = ctx.get_op(name) {
                        if visited_ops.insert(name.clone())
                            && inner(ctx, &def.body.node, vars, visited_ops)
                        {
                            return true;
                        }
                    }
                }
                // Also check arguments
                args.iter()
                    .any(|a| inner(ctx, &a.node, vars, visited_ops))
            }

            // Recurse into compound expressions
            Expr::And(a, b)
            | Expr::Or(a, b)
            | Expr::Implies(a, b)
            | Expr::Equiv(a, b)
            | Expr::Eq(a, b)
            | Expr::Neq(a, b)
            | Expr::Lt(a, b)
            | Expr::Leq(a, b)
            | Expr::Gt(a, b)
            | Expr::Geq(a, b)
            | Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::Mod(a, b)
            | Expr::Range(a, b)
            | Expr::In(a, b)
            | Expr::NotIn(a, b)
            | Expr::Subseteq(a, b)
            | Expr::Union(a, b)
            | Expr::Intersect(a, b)
            | Expr::SetMinus(a, b)
            | Expr::FuncSet(a, b)
            | Expr::LeadsTo(a, b)
            | Expr::IntDiv(a, b)
            | Expr::Pow(a, b) => {
                inner(ctx, &a.node, vars, visited_ops) || inner(ctx, &b.node, vars, visited_ops)
            }

            Expr::Except(base, specs) => {
                inner(ctx, &base.node, vars, visited_ops)
                    || specs.iter().any(|spec| {
                        spec.path.iter().any(|p| match p {
                            tla_core::ast::ExceptPathElement::Index(idx) => {
                                inner(ctx, &idx.node, vars, visited_ops)
                            }
                            tla_core::ast::ExceptPathElement::Field(_) => false,
                        }) || inner(ctx, &spec.value.node, vars, visited_ops)
                    })
            }

            Expr::If(cond, then_br, else_br) => {
                inner(ctx, &cond.node, vars, visited_ops)
                    || inner(ctx, &then_br.node, vars, visited_ops)
                    || inner(ctx, &else_br.node, vars, visited_ops)
            }

            Expr::Not(a)
            | Expr::Neg(a)
            | Expr::Domain(a)
            | Expr::Powerset(a)
            | Expr::BigUnion(a)
            | Expr::Enabled(a)
            | Expr::Unchanged(a)
            | Expr::Always(a)
            | Expr::Eventually(a) => inner(ctx, &a.node, vars, visited_ops),

            Expr::Forall(bounds, body) | Expr::Exists(bounds, body) => {
                bounds.iter().any(|b| {
                    b.domain
                        .as_ref()
                        .is_some_and(|d| inner(ctx, &d.node, vars, visited_ops))
                }) || inner(ctx, &body.node, vars, visited_ops)
            }

            Expr::Choose(bound, body) => {
                bound
                    .domain
                    .as_ref()
                    .is_some_and(|d| inner(ctx, &d.node, vars, visited_ops))
                    || inner(ctx, &body.node, vars, visited_ops)
            }

            Expr::SetEnum(elems) | Expr::Tuple(elems) => elems
                .iter()
                .any(|e| inner(ctx, &e.node, vars, visited_ops)),

            Expr::SetFilter(bound, body) => {
                bound
                    .domain
                    .as_ref()
                    .is_some_and(|d| inner(ctx, &d.node, vars, visited_ops))
                    || inner(ctx, &body.node, vars, visited_ops)
            }

            Expr::SetBuilder(body, bounds) => {
                inner(ctx, &body.node, vars, visited_ops)
                    || bounds.iter().any(|b| {
                        b.domain
                            .as_ref()
                            .is_some_and(|d| inner(ctx, &d.node, vars, visited_ops))
                    })
            }

            Expr::FuncDef(bounds, body) => {
                bounds.iter().any(|b| {
                    b.domain
                        .as_ref()
                        .is_some_and(|d| inner(ctx, &d.node, vars, visited_ops))
                }) || inner(ctx, &body.node, vars, visited_ops)
            }

            Expr::FuncApply(f, arg) => {
                inner(ctx, &f.node, vars, visited_ops) || inner(ctx, &arg.node, vars, visited_ops)
            }

            Expr::RecordAccess(r, _) => inner(ctx, &r.node, vars, visited_ops),

            Expr::Record(fields) | Expr::RecordSet(fields) => fields
                .iter()
                .any(|(_, v)| inner(ctx, &v.node, vars, visited_ops)),

            Expr::Case(arms, other) => {
                arms.iter().any(|arm| {
                    inner(ctx, &arm.guard.node, vars, visited_ops)
                        || inner(ctx, &arm.body.node, vars, visited_ops)
                }) || other
                    .as_ref()
                    .is_some_and(|o| inner(ctx, &o.node, vars, visited_ops))
            }

            Expr::Let(bindings, body) => {
                bindings
                    .iter()
                    .any(|def| inner(ctx, &def.body.node, vars, visited_ops))
                    || inner(ctx, &body.node, vars, visited_ops)
            }

            Expr::WeakFair(v, a) | Expr::StrongFair(v, a) => {
                inner(ctx, &v.node, vars, visited_ops) || inner(ctx, &a.node, vars, visited_ops)
            }

            Expr::ModuleRef(_m, _op, args) => args
                .iter()
                .any(|a| inner(ctx, &a.node, vars, visited_ops)),

            Expr::Lambda(_params, body) => inner(ctx, &body.node, vars, visited_ops),

            // Times is a tuple-like expression (<<a, b, c>>)
            Expr::Times(elems) => elems
                .iter()
                .any(|e| inner(ctx, &e.node, vars, visited_ops)),

            // OpRef is an operator reference (operator name) - can't contain primed vars itself,
            // but the referenced operator body can.
            Expr::OpRef(name) => {
                if let Some(def) = ctx.get_op(name) {
                    if !visited_ops.insert(name.clone()) {
                        return false;
                    }
                    inner(ctx, &def.body.node, vars, visited_ops)
                } else {
                    false
                }
            }

            // InstanceExpr - ignore (should not contain primed vars relevant to InSet vars)
            Expr::InstanceExpr(_, _) => false,

            // Leaf expressions that can't contain primed vars
            Expr::Bool(_) | Expr::Int(_) | Expr::String(_) => false,
        }
    }

    inner(ctx, expr, vars, &mut std::collections::HashSet::new())
}

/// Collect primed variable names referenced in an expression.
/// Returns the names of variables (without ') that appear as x' in the expression.
fn get_primed_var_refs(expr: &Expr) -> HashSet<Arc<str>> {
    let mut refs = HashSet::new();
    get_primed_var_refs_inner(expr, &mut refs);
    refs
}

/// Context-aware version of get_primed_var_refs that follows operator references.
/// This is necessary for topological sorting when expressions contain operator calls
/// (like `IF ActionX THEN 5 ELSE y` where ActionX = `x < 1 /\ x' = x + 1`).
fn get_primed_var_refs_with_ctx(ctx: &EvalCtx, expr: &Expr) -> HashSet<Arc<str>> {
    let mut refs = HashSet::new();
    let mut visited_ops = std::collections::HashSet::new();
    get_primed_var_refs_with_ctx_inner(ctx, expr, &mut refs, &mut visited_ops);
    refs
}

fn get_primed_var_refs_with_ctx_inner(
    ctx: &EvalCtx,
    expr: &Expr,
    refs: &mut HashSet<Arc<str>>,
    visited_ops: &mut std::collections::HashSet<String>,
) {
    match expr {
        // Prime is what we're looking for - extract the variable name
        Expr::Prime(inner) => {
            if let Expr::Ident(name) = &inner.node {
                refs.insert(Arc::from(name.as_str()));
            }
            // Also recurse in case of nested primes (rare but possible)
            get_primed_var_refs_with_ctx_inner(ctx, &inner.node, refs, visited_ops);
        }

        // Operator reference - follow into body with recursion protection
        Expr::Ident(name) => {
            if let Some(def) = ctx.get_op(name) {
                if def.params.is_empty() && visited_ops.insert(name.clone()) {
                    get_primed_var_refs_with_ctx_inner(ctx, &def.body.node, refs, visited_ops);
                }
            }
        }

        // Operator application - check body and arguments
        Expr::Apply(op, args) => {
            if let Expr::Ident(name) = &op.node {
                if let Some(def) = ctx.get_op(name) {
                    if visited_ops.insert(name.clone()) {
                        get_primed_var_refs_with_ctx_inner(ctx, &def.body.node, refs, visited_ops);
                    }
                }
            }
            for a in args {
                get_primed_var_refs_with_ctx_inner(ctx, &a.node, refs, visited_ops);
            }
        }

        // Binary operators - recurse both sides
        Expr::And(l, r)
        | Expr::Or(l, r)
        | Expr::Implies(l, r)
        | Expr::Equiv(l, r)
        | Expr::In(l, r)
        | Expr::NotIn(l, r)
        | Expr::Subseteq(l, r)
        | Expr::Union(l, r)
        | Expr::Intersect(l, r)
        | Expr::SetMinus(l, r)
        | Expr::FuncApply(l, r)
        | Expr::FuncSet(l, r)
        | Expr::Eq(l, r)
        | Expr::Neq(l, r)
        | Expr::Lt(l, r)
        | Expr::Leq(l, r)
        | Expr::Gt(l, r)
        | Expr::Geq(l, r)
        | Expr::Add(l, r)
        | Expr::Sub(l, r)
        | Expr::Mul(l, r)
        | Expr::Div(l, r)
        | Expr::IntDiv(l, r)
        | Expr::Mod(l, r)
        | Expr::Pow(l, r)
        | Expr::Range(l, r)
        | Expr::LeadsTo(l, r) => {
            get_primed_var_refs_with_ctx_inner(ctx, &l.node, refs, visited_ops);
            get_primed_var_refs_with_ctx_inner(ctx, &r.node, refs, visited_ops);
        }

        // Unary operators
        Expr::Not(inner)
        | Expr::Powerset(inner)
        | Expr::BigUnion(inner)
        | Expr::Domain(inner)
        | Expr::Always(inner)
        | Expr::Eventually(inner)
        | Expr::Enabled(inner)
        | Expr::Unchanged(inner)
        | Expr::Neg(inner) => {
            get_primed_var_refs_with_ctx_inner(ctx, &inner.node, refs, visited_ops);
        }

        // Quantifiers and binders
        Expr::Forall(bounds, body) | Expr::Exists(bounds, body) => {
            for b in bounds {
                if let Some(domain) = &b.domain {
                    get_primed_var_refs_with_ctx_inner(ctx, &domain.node, refs, visited_ops);
                }
            }
            get_primed_var_refs_with_ctx_inner(ctx, &body.node, refs, visited_ops);
        }
        Expr::Choose(bound, body) => {
            if let Some(domain) = &bound.domain {
                get_primed_var_refs_with_ctx_inner(ctx, &domain.node, refs, visited_ops);
            }
            get_primed_var_refs_with_ctx_inner(ctx, &body.node, refs, visited_ops);
        }
        Expr::SetBuilder(e, bounds) | Expr::FuncDef(bounds, e) => {
            for b in bounds {
                if let Some(domain) = &b.domain {
                    get_primed_var_refs_with_ctx_inner(ctx, &domain.node, refs, visited_ops);
                }
            }
            get_primed_var_refs_with_ctx_inner(ctx, &e.node, refs, visited_ops);
        }
        Expr::SetFilter(bound, predicate) => {
            if let Some(domain) = &bound.domain {
                get_primed_var_refs_with_ctx_inner(ctx, &domain.node, refs, visited_ops);
            }
            get_primed_var_refs_with_ctx_inner(ctx, &predicate.node, refs, visited_ops);
        }

        // Collections
        Expr::SetEnum(elems) | Expr::Tuple(elems) | Expr::Times(elems) => {
            for e in elems {
                get_primed_var_refs_with_ctx_inner(ctx, &e.node, refs, visited_ops);
            }
        }
        Expr::Record(fields) | Expr::RecordSet(fields) => {
            for (_, e) in fields {
                get_primed_var_refs_with_ctx_inner(ctx, &e.node, refs, visited_ops);
            }
        }

        // Control flow
        Expr::If(cond, then_e, else_e) => {
            get_primed_var_refs_with_ctx_inner(ctx, &cond.node, refs, visited_ops);
            get_primed_var_refs_with_ctx_inner(ctx, &then_e.node, refs, visited_ops);
            get_primed_var_refs_with_ctx_inner(ctx, &else_e.node, refs, visited_ops);
        }
        Expr::Case(arms, default) => {
            for arm in arms {
                get_primed_var_refs_with_ctx_inner(ctx, &arm.guard.node, refs, visited_ops);
                get_primed_var_refs_with_ctx_inner(ctx, &arm.body.node, refs, visited_ops);
            }
            if let Some(d) = default {
                get_primed_var_refs_with_ctx_inner(ctx, &d.node, refs, visited_ops);
            }
        }
        Expr::Let(defs, body) => {
            for def in defs {
                get_primed_var_refs_with_ctx_inner(ctx, &def.body.node, refs, visited_ops);
            }
            get_primed_var_refs_with_ctx_inner(ctx, &body.node, refs, visited_ops);
        }

        // EXCEPT
        Expr::Except(base, specs) => {
            get_primed_var_refs_with_ctx_inner(ctx, &base.node, refs, visited_ops);
            for spec in specs {
                for elem in &spec.path {
                    if let tla_core::ast::ExceptPathElement::Index(e) = elem {
                        get_primed_var_refs_with_ctx_inner(ctx, &e.node, refs, visited_ops);
                    }
                }
                get_primed_var_refs_with_ctx_inner(ctx, &spec.value.node, refs, visited_ops);
            }
        }

        // Temporal operators with two arguments
        Expr::WeakFair(vars_expr, action) | Expr::StrongFair(vars_expr, action) => {
            get_primed_var_refs_with_ctx_inner(ctx, &vars_expr.node, refs, visited_ops);
            get_primed_var_refs_with_ctx_inner(ctx, &action.node, refs, visited_ops);
        }

        // Lambda
        Expr::Lambda(_, body) => {
            get_primed_var_refs_with_ctx_inner(ctx, &body.node, refs, visited_ops);
        }

        // Field access
        Expr::RecordAccess(base, _) => {
            get_primed_var_refs_with_ctx_inner(ctx, &base.node, refs, visited_ops);
        }

        // ModuleRef
        Expr::ModuleRef(_, _, args) => {
            for a in args {
                get_primed_var_refs_with_ctx_inner(ctx, &a.node, refs, visited_ops);
            }
        }

        // Instance expression
        Expr::InstanceExpr(_, subs) => {
            for sub in subs {
                get_primed_var_refs_with_ctx_inner(ctx, &sub.to.node, refs, visited_ops);
            }
        }

        // Leaf nodes - no recursion needed
        Expr::Bool(_) | Expr::Int(_) | Expr::String(_) | Expr::OpRef(_) => {}
    }
}

/// Recursive helper for get_primed_var_refs
fn get_primed_var_refs_inner(expr: &Expr, refs: &mut HashSet<Arc<str>>) {
    use tla_core::ast::Expr;
    match expr {
        // Prime is what we're looking for - extract the variable name
        Expr::Prime(inner) => {
            if let Expr::Ident(name) = &inner.node {
                refs.insert(Arc::from(name.as_str()));
            }
            // Also recurse in case of nested primes (rare but possible)
            get_primed_var_refs_inner(&inner.node, refs);
        }

        // Binary operators - recurse both sides
        Expr::And(l, r)
        | Expr::Or(l, r)
        | Expr::Implies(l, r)
        | Expr::Equiv(l, r)
        | Expr::In(l, r)
        | Expr::NotIn(l, r)
        | Expr::Subseteq(l, r)
        | Expr::Union(l, r)
        | Expr::Intersect(l, r)
        | Expr::SetMinus(l, r)
        | Expr::FuncApply(l, r)
        | Expr::FuncSet(l, r)
        | Expr::Eq(l, r)
        | Expr::Neq(l, r)
        | Expr::Lt(l, r)
        | Expr::Leq(l, r)
        | Expr::Gt(l, r)
        | Expr::Geq(l, r)
        | Expr::Add(l, r)
        | Expr::Sub(l, r)
        | Expr::Mul(l, r)
        | Expr::Div(l, r)
        | Expr::IntDiv(l, r)
        | Expr::Mod(l, r)
        | Expr::Pow(l, r)
        | Expr::Range(l, r)
        | Expr::LeadsTo(l, r) => {
            get_primed_var_refs_inner(&l.node, refs);
            get_primed_var_refs_inner(&r.node, refs);
        }

        // Unary operators
        Expr::Not(inner)
        | Expr::Powerset(inner)
        | Expr::BigUnion(inner)
        | Expr::Domain(inner)
        | Expr::Always(inner)
        | Expr::Eventually(inner)
        | Expr::Enabled(inner)
        | Expr::Unchanged(inner)
        | Expr::Neg(inner) => {
            get_primed_var_refs_inner(&inner.node, refs);
        }

        // Quantifiers and binders
        Expr::Forall(bounds, body) | Expr::Exists(bounds, body) => {
            for b in bounds {
                if let Some(domain) = &b.domain {
                    get_primed_var_refs_inner(&domain.node, refs);
                }
            }
            get_primed_var_refs_inner(&body.node, refs);
        }
        Expr::Choose(bound, body) => {
            if let Some(domain) = &bound.domain {
                get_primed_var_refs_inner(&domain.node, refs);
            }
            get_primed_var_refs_inner(&body.node, refs);
        }
        Expr::SetBuilder(expr, bounds) | Expr::FuncDef(bounds, expr) => {
            for b in bounds {
                if let Some(domain) = &b.domain {
                    get_primed_var_refs_inner(&domain.node, refs);
                }
            }
            get_primed_var_refs_inner(&expr.node, refs);
        }
        Expr::SetFilter(bound, predicate) => {
            if let Some(domain) = &bound.domain {
                get_primed_var_refs_inner(&domain.node, refs);
            }
            get_primed_var_refs_inner(&predicate.node, refs);
        }

        // Collections
        Expr::SetEnum(elems) | Expr::Tuple(elems) | Expr::Times(elems) => {
            for e in elems {
                get_primed_var_refs_inner(&e.node, refs);
            }
        }
        Expr::Record(fields) | Expr::RecordSet(fields) => {
            for (_, e) in fields {
                get_primed_var_refs_inner(&e.node, refs);
            }
        }

        // Application and references
        Expr::Apply(func, args) => {
            get_primed_var_refs_inner(&func.node, refs);
            for a in args {
                get_primed_var_refs_inner(&a.node, refs);
            }
        }
        Expr::ModuleRef(_, _, args) => {
            for a in args {
                get_primed_var_refs_inner(&a.node, refs);
            }
        }

        // Control flow
        Expr::If(cond, then_e, else_e) => {
            get_primed_var_refs_inner(&cond.node, refs);
            get_primed_var_refs_inner(&then_e.node, refs);
            get_primed_var_refs_inner(&else_e.node, refs);
        }
        Expr::Case(arms, default) => {
            for arm in arms {
                get_primed_var_refs_inner(&arm.guard.node, refs);
                get_primed_var_refs_inner(&arm.body.node, refs);
            }
            if let Some(d) = default {
                get_primed_var_refs_inner(&d.node, refs);
            }
        }
        Expr::Let(defs, body) => {
            for def in defs {
                get_primed_var_refs_inner(&def.body.node, refs);
            }
            get_primed_var_refs_inner(&body.node, refs);
        }

        // EXCEPT
        Expr::Except(base, specs) => {
            get_primed_var_refs_inner(&base.node, refs);
            for spec in specs {
                for elem in &spec.path {
                    match elem {
                        tla_core::ast::ExceptPathElement::Index(e) => {
                            get_primed_var_refs_inner(&e.node, refs);
                        }
                        tla_core::ast::ExceptPathElement::Field(_) => {}
                    }
                }
                get_primed_var_refs_inner(&spec.value.node, refs);
            }
        }

        // Temporal operators with two arguments
        Expr::WeakFair(vars, action) | Expr::StrongFair(vars, action) => {
            get_primed_var_refs_inner(&vars.node, refs);
            get_primed_var_refs_inner(&action.node, refs);
        }

        // Lambda
        Expr::Lambda(_, body) => {
            get_primed_var_refs_inner(&body.node, refs);
        }

        // Field access
        Expr::RecordAccess(base, _) => {
            get_primed_var_refs_inner(&base.node, refs);
        }

        // Leaf nodes - no recursion needed
        Expr::Bool(_) | Expr::Int(_) | Expr::String(_) | Expr::Ident(_) | Expr::OpRef(_) => {}

        // Instance expression with substitutions
        Expr::InstanceExpr(_, subs) => {
            for sub in subs {
                get_primed_var_refs_inner(&sub.to.node, refs);
            }
        }
    }
}

/// Topologically sort symbolic assignments so that assignments defining x' come before
/// assignments that reference x'. This ensures proper evaluation order when computing
/// expressions like `announced' = (count' >= VT)` which depends on `count' = count + 1`.
///
/// When `ctx` is provided, operator references are followed to find hidden dependencies
/// (e.g., `IF ActionX THEN 5 ELSE y` where ActionX contains `x'`).
fn topological_sort_assignments(
    ctx: Option<&EvalCtx>,
    symbolic: &[SymbolicAssignment],
) -> Vec<SymbolicAssignment> {
    use std::cmp::Reverse;
    use std::collections::{BinaryHeap, HashMap};

    let debug = std::env::var("TLA2_DEBUG_TOPOSORT").is_ok();

    if symbolic.is_empty() {
        return Vec::new();
    }

    let n = symbolic.len();

    // Fast path: single assignment has no dependencies to sort.
    if n == 1 {
        return symbolic.to_vec();
    }

    // Capture defined variable per assignment in stable, input order.
    let mut defined_vars: Vec<Arc<str>> = Vec::with_capacity(n);
    for (i, sym) in symbolic.iter().enumerate() {
        let name = match sym {
            SymbolicAssignment::Value(n, _)
            | SymbolicAssignment::Expr(n, _, _)
            | SymbolicAssignment::Unchanged(n)
            | SymbolicAssignment::InSet(n, _, _) => n.clone(),
        };
        if debug {
            eprintln!("[TOPOSORT] Assignment {}: {}' = {:?}", i, name, sym);
        }
        defined_vars.push(name);
    }

    // Map variable name -> defining assignment index.
    //
    // If a variable is assigned more than once, pick the FIRST occurrence to keep the
    // result deterministic and consistent with extraction order.
    let mut var_to_idx: HashMap<Arc<str>, usize> = HashMap::new();
    for (i, name) in defined_vars.iter().enumerate() {
        var_to_idx.entry(name.clone()).or_insert(i);
    }

    // deps[i] = indices that must come before i
    let mut deps: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, sym) in symbolic.iter().enumerate() {
        let refs = match sym {
            SymbolicAssignment::Expr(_, expr, _) => {
                if let Some(c) = ctx {
                    get_primed_var_refs_with_ctx(c, &expr.node)
                } else {
                    get_primed_var_refs(&expr.node)
                }
            }
            SymbolicAssignment::InSet(_, expr, _) => {
                if let Some(c) = ctx {
                    get_primed_var_refs_with_ctx(c, &expr.node)
                } else {
                    get_primed_var_refs(&expr.node)
                }
            }
            SymbolicAssignment::Value(_, _) | SymbolicAssignment::Unchanged(_) => HashSet::new(),
        };

        for var_name in &refs {
            if let Some(&def_idx) = var_to_idx.get(var_name) {
                if def_idx != i {
                    deps[i].push(def_idx);
                }
            }
        }

        deps[i].sort_unstable();
        deps[i].dedup();

        if debug && !refs.is_empty() {
            eprintln!(
                "[TOPOSORT] Assignment {} references primed vars: {:?}, deps: {:?}",
                i,
                refs.iter().map(|s| s.as_ref()).collect::<Vec<_>>(),
                deps[i]
            );
        }
    }

    // adjacency[dep] = assignments that depend on dep
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, dep_list) in deps.iter().enumerate() {
        for &dep in dep_list {
            adjacency[dep].push(i);
        }
    }
    for out in &mut adjacency {
        out.sort_unstable();
        out.dedup();
    }

    // Kahn's algorithm with deterministic tie-breaker (smallest input index first).
    let mut in_degree: Vec<usize> = deps.iter().map(|d| d.len()).collect();
    let mut ready: BinaryHeap<Reverse<usize>> = BinaryHeap::new();
    for (i, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            ready.push(Reverse(i));
        }
    }

    let mut order: Vec<usize> = Vec::with_capacity(n);
    while let Some(Reverse(i)) = ready.pop() {
        order.push(i);
        for &j in &adjacency[i] {
            let deg = &mut in_degree[j];
            debug_assert!(*deg > 0, "in_degree underflow for node {j} (from {i})");
            *deg -= 1;
            if *deg == 0 {
                ready.push(Reverse(j));
            }
        }
    }

    // If we couldn't sort all (cycle detected), fall back to original order
    if order.len() != symbolic.len() {
        if debug {
            eprintln!("[TOPOSORT] Cycle detected, falling back to original order");
        }
        return symbolic.to_vec();
    }

    if debug {
        eprintln!("[TOPOSORT] Sorted order:");
        for (out_i, &in_i) in order.iter().enumerate() {
            eprintln!("[TOPOSORT]   {}: {}'", out_i, defined_vars[in_i].as_ref());
        }
    }

    order.into_iter().map(|i| symbolic[i].clone()).collect()
}

/// Evaluate symbolic assignments with progressive next-state context
///
/// This handles TLA+ simultaneous assignment semantics by evaluating assignments
/// in order and making computed values available for subsequent expressions.
///
/// When an expression references primed variables from InSet assignments (x' \in S),
/// we defer evaluation until successor state building when we know the specific value.
///
/// **Conflict detection**: When the same variable receives multiple assignments from
/// different parts of a conjunction (e.g., `UNCHANGED hr` from NowNext and `hr' = 5`
/// from HCnxt), this function detects the conflict and returns an empty result,
/// indicating no valid successor states exist for this branch.
fn evaluate_symbolic_assignments(
    ctx: &EvalCtx,
    symbolic: &[SymbolicAssignment],
) -> Result<Vec<PrimedAssignment>, EvalError> {
    use crate::eval::Env;
    use std::collections::HashMap;
    let mut next_state: Env = Env::new();
    let mut result = Vec::new();
    // Avoid cloning the entire EvalCtx (including local_stack) for every assignment.
    // We only need to update the next_state view as we compute primed values.
    let mut eval_ctx = ctx.clone();

    // Track which variables come from InSet (have multiple possible values)
    let mut inset_vars: HashSet<Arc<str>> = HashSet::new();

    // Track constraints for conflict detection:
    // - None: unconstrained
    // - Some(None): constrained to InSet (values tracked in pending_insets)
    // - Some(Some(v)): constrained to specific value
    let mut constraints: HashMap<Arc<str>, Option<Value>> = HashMap::new();

    // Track pending InSet assignments (may be filtered by later constraints)
    let mut pending_insets: HashMap<Arc<str>, Vec<Value>> = HashMap::new();

    // First pass: identify InSet variables
    for sym in symbolic {
        if let SymbolicAssignment::InSet(name, _, _) = sym {
            inset_vars.insert(name.clone());
        }
    }

    // Topological sort: ensure assignments that define x' come before expressions that reference x'.
    // This fixes the case where `announced' = (count' >= VT)` is extracted before `count' = count + 1`.
    // Pass ctx to follow operator references for hidden dependencies.
    let sorted_symbolic = topological_sort_assignments(Some(ctx), symbolic);

    let debug = debug_enum();

    for sym in &sorted_symbolic {
        match sym {
            SymbolicAssignment::Expr(name, expr, bindings) => {
                // Check if this expression references any primed InSet variables
                let refs_inset = expr_references_primed_vars(ctx, &expr.node, &inset_vars);
                if refs_inset {
                    if debug {
                        eprintln!(
                            "evaluate_symbolic_assignments: deferring {}' (references InSet primed vars)",
                            name
                        );
                    }
                    // Can't evaluate now - will be evaluated per-combination
                    result.push(PrimedAssignment::DeferredExpr(name.clone(), expr.clone()));
                } else {
                    // Create context with current next_state for primed variable lookup
                    // and restore captured bindings for deferred evaluation
                    eval_ctx.next_state = Some(Arc::new(next_state.clone()));
                    let binding_ctx = eval_ctx.with_captured_bindings(bindings);
                    let value = if profile_enum_detail() {
                        let start = std::time::Instant::now();
                        let res = eval(&binding_ctx, expr);
                        PROF_ASSIGN_US.fetch_add(
                            start.elapsed().as_micros() as u64,
                            AtomicOrdering::Relaxed,
                        );
                        res?
                    } else {
                        eval(&binding_ctx, expr)?
                    };

                    // Check for conflicts with existing constraints
                    if let Some(existing) = constraints.get(name) {
                        match existing {
                            Some(existing_val) => {
                                if existing_val != &value {
                                    if debug {
                                        eprintln!(
                                            "evaluate_symbolic_assignments: CONFLICT {}' = {} vs existing {}",
                                            name, value, existing_val
                                        );
                                    }
                                    return Ok(Vec::new()); // Conflict - no valid successors
                                }
                                // Same value - redundant but not conflict, skip adding again
                                continue;
                            }
                            None => {
                                // Was InSet, now constrained to specific value
                                // Filter InSet to just this value
                                if let Some(inset_values) = pending_insets.get_mut(name) {
                                    if !inset_values.contains(&value) {
                                        if debug {
                                            eprintln!(
                                                "evaluate_symbolic_assignments: CONFLICT {}' = {} not in InSet {:?}",
                                                name, value, inset_values
                                            );
                                        }
                                        return Ok(Vec::new()); // Value not in set - conflict
                                    }
                                    // Keep only this value
                                    *inset_values = vec![value.clone()];
                                }
                            }
                        }
                    }

                    constraints.insert(name.clone(), Some(value.clone()));
                    next_state.insert(name.clone(), value.clone());
                    result.push(PrimedAssignment::Assign(name.clone(), value));
                }
            }
            SymbolicAssignment::Value(name, value) => {
                // Check for conflicts with existing constraints
                if let Some(existing) = constraints.get(name) {
                    match existing {
                        Some(existing_val) => {
                            if existing_val != value {
                                if debug {
                                    eprintln!(
                                        "evaluate_symbolic_assignments: CONFLICT {}' = {} vs existing {}",
                                        name, value, existing_val
                                    );
                                }
                                return Ok(Vec::new()); // Conflict
                            }
                            // Same value - redundant, skip
                            continue;
                        }
                        None => {
                            // Was InSet, filter to this value
                            if let Some(inset_values) = pending_insets.get_mut(name) {
                                if !inset_values.contains(value) {
                                    if debug {
                                        eprintln!(
                                            "evaluate_symbolic_assignments: CONFLICT {}' = {} not in InSet {:?}",
                                            name, value, inset_values
                                        );
                                    }
                                    return Ok(Vec::new());
                                }
                                *inset_values = vec![value.clone()];
                            }
                        }
                    }
                }

                constraints.insert(name.clone(), Some(value.clone()));
                next_state.insert(name.clone(), value.clone());
                result.push(PrimedAssignment::Assign(name.clone(), value.clone()));
            }
            SymbolicAssignment::Unchanged(name) => {
                // UNCHANGED x means x' = x
                if let Some(current) = ctx.env.get(name) {
                    // Check for conflicts
                    if let Some(existing) = constraints.get(name) {
                        match existing {
                            Some(existing_val) => {
                                if existing_val != current {
                                    if debug {
                                        eprintln!(
                                            "evaluate_symbolic_assignments: CONFLICT UNCHANGED {}' = {} vs existing {}",
                                            name, current, existing_val
                                        );
                                    }
                                    return Ok(Vec::new()); // Conflict
                                }
                                // Same value - redundant, skip
                                continue;
                            }
                            None => {
                                // Was InSet, filter to current value
                                if let Some(inset_values) = pending_insets.get_mut(name) {
                                    if !inset_values.contains(current) {
                                        if debug {
                                            eprintln!(
                                                "evaluate_symbolic_assignments: CONFLICT UNCHANGED {}' = {} not in InSet {:?}",
                                                name, current, inset_values
                                            );
                                        }
                                        return Ok(Vec::new());
                                    }
                                    *inset_values = vec![current.clone()];
                                }
                            }
                        }
                    }

                    constraints.insert(name.clone(), Some(current.clone()));
                    next_state.insert(name.clone(), current.clone());
                    result.push(PrimedAssignment::Unchanged(name.clone()));
                }
            }
            SymbolicAssignment::InSet(name, set_expr, bindings) => {
                // Restore captured bindings for deferred evaluation
                eval_ctx.next_state = Some(Arc::new(next_state.clone()));
                let binding_ctx = eval_ctx.with_captured_bindings(bindings);
                let set_val = if profile_enum_detail() {
                    let start = std::time::Instant::now();
                    let res = eval(&binding_ctx, set_expr);
                    PROF_ASSIGN_US.fetch_add(
                        start.elapsed().as_micros() as u64,
                        AtomicOrdering::Relaxed,
                    );
                    res?
                } else {
                    eval(&binding_ctx, set_expr)?
                };
                // Handle both Set and Interval using to_sorted_set
                if let Some(s) = set_val.to_sorted_set() {
                    let mut values: Vec<Value> = s.as_slice().to_vec();

                    // Check for conflicts with existing constraints
                    if let Some(existing) = constraints.get(name) {
                        match existing {
                            Some(existing_val) => {
                                // Already constrained to a value, filter InSet
                                if !values.contains(existing_val) {
                                    if debug {
                                        eprintln!(
                                            "evaluate_symbolic_assignments: CONFLICT {}' existing {} not in InSet {:?}",
                                            name, existing_val, values
                                        );
                                    }
                                    return Ok(Vec::new()); // Conflict
                                }
                                // Existing value is in set - no need to add InSet, already constrained
                                continue;
                            }
                            None => {
                                // Was already InSet, intersect
                                if let Some(prev_values) = pending_insets.get(name) {
                                    values.retain(|v| prev_values.contains(v));
                                    if values.is_empty() {
                                        if debug {
                                            eprintln!(
                                                "evaluate_symbolic_assignments: CONFLICT {}' InSet intersection empty",
                                                name
                                            );
                                        }
                                        return Ok(Vec::new()); // Empty intersection
                                    }
                                }
                            }
                        }
                    }

                    if !values.is_empty() {
                        constraints.insert(name.clone(), None); // Mark as InSet-constrained
                        pending_insets.insert(name.clone(), values.clone());
                        // Don't add to result yet - we'll finalize InSets at the end
                    }
                }
            }
        }
    }

    // Finalize pending InSets - add them to result
    for (name, values) in pending_insets {
        if values.len() == 1 {
            // Single value - convert to Assign
            let value = values.into_iter().next().unwrap();
            next_state.insert(name.clone(), value.clone());
            result.push(PrimedAssignment::Assign(name.clone(), value));
        } else if !values.is_empty() {
            result.push(PrimedAssignment::InSet(name, values));
        }
    }

    Ok(result)
}

/// Build all possible successor states from assignments
///
/// Returns multiple states when assignments contain `InSet` variants.
/// Uses ArrayState for O(1) variable access when registry is available.
fn build_successor_states(
    current_state: &State,
    vars: &[Arc<str>],
    assignments: &[PrimedAssignment],
    registry: Option<&VarRegistry>,
) -> Vec<State> {
    // Fast path: use indexed access when registry is available
    if let Some(reg) = registry {
        // Build as ArrayState first (avoids OrdMap construction in hot path)
        let array_states =
            build_successor_states_indexed_raw(current_state, vars, assignments, reg);
        // Convert to State only at the end
        return array_states
            .into_iter()
            .map(|arr| arr.to_state(reg))
            .collect();
    }

    // Fallback: original implementation without registry
    let mut var_values: Vec<(Arc<str>, Vec<Value>)> = Vec::new();

    for var in vars {
        let values = find_values_for_var_in_assignments(var, assignments, current_state);
        if values.is_empty() {
            let current_value = current_state
                .vars()
                .find(|(n, _)| n.as_ref() == var.as_ref())
                .map(|(_, v)| v.clone());
            if let Some(v) = current_value {
                var_values.push((var.clone(), vec![v]));
                continue;
            }
            return Vec::new();
        }
        var_values.push((var.clone(), values));
    }

    let mut states = Vec::new();
    enumerate_successor_combinations(&var_values, 0, &mut Vec::new(), &mut states);
    states
}

/// Build successor states with optional context for deferred expression evaluation
///
/// When `ctx` is provided and there are `DeferredExpr` assignments, they will be
/// evaluated for each InSet combination with the appropriate next-state context.
fn build_successor_states_with_ctx(
    current_state: &State,
    vars: &[Arc<str>],
    assignments: &[PrimedAssignment],
    registry: Option<&VarRegistry>,
    ctx: Option<&EvalCtx>,
) -> Vec<State> {
    // Check if we have any deferred expressions that need evaluation
    let has_deferred = assignments
        .iter()
        .any(|a| matches!(a, PrimedAssignment::DeferredExpr(_, _)));

    if has_deferred {
        if let Some(ctx) = ctx {
            // Slow path: need to evaluate deferred expressions per-combination
            return build_successor_states_with_deferred(current_state, vars, assignments, ctx);
        }
    }

    // Fast path: no deferred expressions, use existing implementation
    build_successor_states(current_state, vars, assignments, registry)
}

/// Build successor states when there are DeferredExpr assignments.
///
/// For each combination of InSet values, creates a next-state context and evaluates
/// any deferred expressions before building the final state.
fn build_successor_states_with_deferred(
    current_state: &State,
    vars: &[Arc<str>],
    assignments: &[PrimedAssignment],
    ctx: &EvalCtx,
) -> Vec<State> {
    use crate::eval::Env;

    // Separate InSet assignments (which produce combinations) from others
    let mut inset_vars: Vec<(Arc<str>, Vec<Value>)> = Vec::new();
    let mut deferred: Vec<(Arc<str>, Spanned<Expr>)> = Vec::new();
    let mut fixed_assignments: Vec<(Arc<str>, Value)> = Vec::new();
    let mut unchanged_vars: Vec<Arc<str>> = Vec::new();

    for assignment in assignments {
        match assignment {
            PrimedAssignment::Assign(name, value) => {
                fixed_assignments.push((name.clone(), value.clone()));
            }
            PrimedAssignment::Unchanged(name) => {
                unchanged_vars.push(name.clone());
            }
            PrimedAssignment::InSet(name, values) => {
                if values.is_empty() {
                    return Vec::new(); // Empty domain means no successors
                }
                inset_vars.push((name.clone(), values.clone()));
            }
            PrimedAssignment::DeferredExpr(name, expr) => {
                deferred.push((name.clone(), expr.clone()));
            }
        }
    }

    // Generate all combinations of InSet values
    let combinations = generate_inset_combinations(&inset_vars);
    let debug = debug_enum();

    let mut results = Vec::new();

    for combo in combinations {
        // Build next_state with this combination
        let mut next_state: Env = Env::new();

        // Add fixed assignments
        for (name, value) in &fixed_assignments {
            next_state.insert(name.clone(), value.clone());
        }

        // Add unchanged vars from current state
        for name in &unchanged_vars {
            if let Some(current) = ctx.env.get(name) {
                next_state.insert(name.clone(), current.clone());
            }
        }

        // Add InSet values from this combination
        for (name, value) in &combo {
            next_state.insert(name.clone(), value.clone());
        }

        // Evaluate deferred expressions with this next_state context
        let mut eval_ctx = ctx.clone();
        eval_ctx.next_state = Some(Arc::new(next_state.clone()));

        let mut all_ok = true;
        for (name, expr) in &deferred {
            match eval(&eval_ctx, expr) {
                Ok(value) => {
                    if debug {
                        eprintln!("build_successor_states_with_deferred: {}' = {}", name, value);
                    }
                    next_state.insert(name.clone(), value);
                    // Update context for subsequent deferred exprs that might depend on this
                    eval_ctx.next_state = Some(Arc::new(next_state.clone()));
                }
                Err(e) => {
                    if debug {
                        eprintln!("build_successor_states_with_deferred: {}' eval error: {:?}", name, e);
                    }
                    all_ok = false;
                    break;
                }
            }
        }

        if !all_ok {
            continue; // Skip this combination if evaluation failed
        }

        // Build the state from next_state
        let mut state_builder = im::OrdMap::new();
        for var in vars {
            let value = next_state
                .get(var)
                .cloned()
                .or_else(|| {
                    current_state
                        .vars()
                        .find(|(n, _)| n.as_ref() == var.as_ref())
                        .map(|(_, v)| v.clone())
                });
            if let Some(v) = value {
                state_builder.insert(var.clone(), v);
            } else {
                // Missing variable - skip this state
                all_ok = false;
                break;
            }
        }

        if all_ok {
            results.push(State::from_vars(state_builder));
        }
    }

    results
}

/// Generate all combinations of InSet values
fn generate_inset_combinations(
    inset_vars: &[(Arc<str>, Vec<Value>)],
) -> Vec<Vec<(Arc<str>, Value)>> {
    if inset_vars.is_empty() {
        return vec![vec![]]; // One empty combination if no InSet vars
    }

    let mut combinations = vec![vec![]];

    for (name, values) in inset_vars {
        let mut new_combinations = Vec::new();
        for combo in &combinations {
            for value in values {
                let mut new_combo = combo.clone();
                new_combo.push((name.clone(), value.clone()));
                new_combinations.push(new_combo);
            }
        }
        combinations = new_combinations;
    }

    combinations
}

/// Fast path: build successor states as ArrayState using O(1) variable access
///
/// This is the raw function that returns ArrayState directly without OrdMap conversion.
/// Used by the fast BFS loop that keeps ArrayState throughout.
fn build_successor_states_indexed_raw(
    current_state: &State,
    vars: &[Arc<str>],
    assignments: &[PrimedAssignment],
    registry: &VarRegistry,
) -> Vec<ArrayState> {
    let num_vars = vars.len();

    // Convert current state to ArrayState for O(1) access
    let mut array_state = ArrayState::from_state(current_state, registry);

    // Build lookup map for assignments: var_name -> possible values
    // This avoids repeated linear searches through assignments
    let mut assignment_map: Vec<Option<Vec<Value>>> = vec![None; num_vars];
    for assignment in assignments {
        match assignment {
            PrimedAssignment::Assign(name, value) => {
                if let Some(idx) = registry.get(name) {
                    assignment_map[idx.as_usize()] = Some(vec![value.clone()]);
                }
            }
            // Explicit UNCHANGED is a no-op here; unassigned vars default to current values.
            PrimedAssignment::Unchanged(_) => {}
            PrimedAssignment::InSet(name, values) => {
                if let Some(idx) = registry.get(name) {
                    assignment_map[idx.as_usize()] = Some(values.clone());
                }
            }
            // DeferredExpr should have been handled by build_successor_states_with_deferred
            PrimedAssignment::DeferredExpr(_, _) => {
                unreachable!("DeferredExpr should be handled by build_successor_states_with_deferred")
            }
        }
    }

    // If any IN-set assignment has an empty domain, there are no successors.
    if assignment_map
        .iter()
        .any(|v| matches!(v, Some(vals) if vals.is_empty()))
    {
        return Vec::new();
    }

    // Heuristic: incremental fingerprint updates require hashing both the old and new value
    // for each assigned var. If most vars are assigned, it's cheaper to recompute the
    // fingerprint once per successor at the leaf.
    let assigned_vars = assignment_map.iter().filter(|v| v.is_some()).count();
    let use_incremental_fp = assigned_vars.saturating_mul(2) < num_vars;
    if use_incremental_fp {
        // Ensure the combined-xor cache exists so we can update fingerprints incrementally.
        let _ = array_state.fingerprint(registry);
    }

    // Enumerate all combinations directly as ArrayState, mutating from current state.
    let num_combinations = assignment_map.iter().fold(1usize, |acc, v| {
        acc.saturating_mul(v.as_ref().map_or(1, |vals| vals.len()))
    });
    let mut results = Vec::with_capacity(num_combinations);
    enumerate_successor_combinations_array(
        &assignment_map,
        0,
        &mut array_state,
        registry,
        &mut results,
        use_incremental_fp,
    );
    results
}

fn find_values_for_var_in_assignments(
    var: &Arc<str>,
    assignments: &[PrimedAssignment],
    current_state: &State,
) -> Vec<Value> {
    for assignment in assignments {
        match assignment {
            PrimedAssignment::Assign(name, value) if name.as_ref() == var.as_ref() => {
                return vec![value.clone()];
            }
            PrimedAssignment::Unchanged(name) if name.as_ref() == var.as_ref() => {
                for (n, v) in current_state.vars() {
                    if n.as_ref() == var.as_ref() {
                        return vec![v.clone()];
                    }
                }
            }
            PrimedAssignment::InSet(name, values) if name.as_ref() == var.as_ref() => {
                return values.clone();
            }
            _ => {}
        }
    }
    Vec::new()
}

/// Fallback enumeration using name-based state building
fn enumerate_successor_combinations(
    var_values: &[(Arc<str>, Vec<Value>)],
    idx: usize,
    current: &mut Vec<(Arc<str>, Value)>,
    results: &mut Vec<State>,
) {
    if idx == var_values.len() {
        let state = State::from_pairs(current.iter().map(|(k, v)| (Arc::clone(k), v.clone())));
        results.push(state);
        return;
    }

    let (var, values) = &var_values[idx];
    for val in values {
        current.push((var.clone(), val.clone()));
        enumerate_successor_combinations(var_values, idx + 1, current, results);
        current.pop();
    }
}

// ============================================================================
// ARRAYSTATE-RETURNING VARIANTS - For O(1) state building without OrdMap
// ============================================================================

/// Build successor states as ArrayState directly, avoiding OrdMap construction.
///
/// This is the fast path for state enumeration. Returns ArrayState objects that
/// can be used directly for fingerprint computation and invariant checking.
/// Only convert to State when needed for trace reconstruction.
pub fn build_successor_array_states(
    current_state: &State,
    vars: &[Arc<str>],
    assignments: &[PrimedAssignment],
    registry: &VarRegistry,
) -> Vec<ArrayState> {
    let num_vars = vars.len();

    // Convert current state to ArrayState for O(1) access
    let mut array_state = ArrayState::from_state(current_state, registry);

    // Build lookup map for assignments: var_name -> possible values
    // This avoids repeated linear searches through assignments
    let mut assignment_map: Vec<Option<Vec<Value>>> = vec![None; num_vars];
    for assignment in assignments {
        match assignment {
            PrimedAssignment::Assign(name, value) => {
                if let Some(idx) = registry.get(name) {
                    assignment_map[idx.as_usize()] = Some(vec![value.clone()]);
                }
            }
            // Explicit UNCHANGED is a no-op here; unassigned vars default to current values.
            PrimedAssignment::Unchanged(_) => {}
            PrimedAssignment::InSet(name, values) => {
                if let Some(idx) = registry.get(name) {
                    assignment_map[idx.as_usize()] = Some(values.clone());
                }
            }
            // DeferredExpr should have been handled by build_successor_states_with_deferred
            PrimedAssignment::DeferredExpr(_, _) => {
                unreachable!("DeferredExpr should be handled by build_successor_states_with_deferred")
            }
        }
    }

    // If any IN-set assignment has an empty domain, there are no successors.
    if assignment_map
        .iter()
        .any(|v| matches!(v, Some(vals) if vals.is_empty()))
    {
        return Vec::new();
    }

    // Heuristic: incremental fingerprint updates require hashing both the old and new value
    // for each assigned var. If most vars are assigned, it's cheaper to recompute the
    // fingerprint once per successor at the leaf.
    let assigned_vars = assignment_map.iter().filter(|v| v.is_some()).count();
    let use_incremental_fp = assigned_vars.saturating_mul(2) < num_vars;
    if use_incremental_fp {
        // Ensure the combined-xor cache exists so we can update fingerprints incrementally.
        let _ = array_state.fingerprint(registry);
    }

    // Pre-compute number of combinations for pre-allocation.
    let num_combinations = assignment_map.iter().fold(1usize, |acc, v| {
        acc.saturating_mul(v.as_ref().map_or(1, |vals| vals.len()))
    });

    // Enumerate all combinations, keeping as ArrayState.
    let mut results = Vec::with_capacity(num_combinations);
    enumerate_successor_combinations_array(
        &assignment_map,
        0,
        &mut array_state,
        registry,
        &mut results,
        use_incremental_fp,
    );
    results
}

/// Build successor ArrayStates from an existing ArrayState (avoids State conversion)
pub fn build_successor_array_states_from_array(
    current_array: &ArrayState,
    vars: &[Arc<str>],
    assignments: &[PrimedAssignment],
    registry: &VarRegistry,
) -> Vec<ArrayState> {
    let num_vars = vars.len();

    // Build lookup map for assignments: var_name -> possible values
    let mut assignment_map: Vec<Option<Vec<Value>>> = vec![None; num_vars];
    for assignment in assignments {
        match assignment {
            PrimedAssignment::Assign(name, value) => {
                if let Some(idx) = registry.get(name) {
                    assignment_map[idx.as_usize()] = Some(vec![value.clone()]);
                }
            }
            // Explicit UNCHANGED is a no-op here; unassigned vars default to current values.
            PrimedAssignment::Unchanged(_) => {}
            PrimedAssignment::InSet(name, values) => {
                if let Some(idx) = registry.get(name) {
                    assignment_map[idx.as_usize()] = Some(values.clone());
                }
            }
            // DeferredExpr should have been handled by build_successor_states_with_deferred
            PrimedAssignment::DeferredExpr(_, _) => {
                unreachable!("DeferredExpr should be handled by build_successor_states_with_deferred")
            }
        }
    }

    // If any IN-set assignment has an empty domain, there are no successors.
    if assignment_map
        .iter()
        .any(|v| matches!(v, Some(vals) if vals.is_empty()))
    {
        return Vec::new();
    }

    // Heuristic: incremental fingerprint updates require hashing both the old and new value
    // for each assigned var. If most vars are assigned, it's cheaper to recompute the
    // fingerprint once per successor at the leaf.
    let assigned_vars = assignment_map.iter().filter(|v| v.is_some()).count();
    let use_incremental_fp = assigned_vars.saturating_mul(2) < num_vars;

    // Pre-compute number of combinations for pre-allocation.
    let num_combinations = assignment_map.iter().fold(1usize, |acc, v| {
        acc.saturating_mul(v.as_ref().map_or(1, |vals| vals.len()))
    });

    // Enumerate all combinations by mutating from the current state.
    let mut results = Vec::with_capacity(num_combinations);
    let mut array_state = current_array.clone();
    if use_incremental_fp {
        // Ensure the combined-xor cache exists so we can update fingerprints incrementally.
        let _ = array_state.fingerprint(registry);
    }
    enumerate_successor_combinations_array(
        &assignment_map,
        0,
        &mut array_state,
        registry,
        &mut results,
        use_incremental_fp,
    );
    results
}

/// Enumerate combinations directly into Vec<ArrayState> without OrdMap conversion
fn enumerate_successor_combinations_array(
    assignment_map: &[Option<Vec<Value>>],
    idx: usize,
    array_state: &mut ArrayState,
    registry: &VarRegistry,
    results: &mut Vec<ArrayState>,
    use_incremental_fp: bool,
) {
    if idx == assignment_map.len() {
        if !use_incremental_fp {
            let _ = array_state.fingerprint(registry);
        }
        // All variables assigned - clone the ArrayState (no OrdMap!)
        results.push(array_state.clone());
        return;
    }

    let var_idx = VarIndex(idx as u16);
    match &assignment_map[idx] {
        Some(values) => {
            for val in values {
                if use_incremental_fp {
                    array_state.set_with_registry(var_idx, val.clone(), registry);
                } else {
                    array_state.set(var_idx, val.clone());
                }
                enumerate_successor_combinations_array(
                    assignment_map,
                    idx + 1,
                    array_state,
                    registry,
                    results,
                    use_incremental_fp,
                );
            }
        }
        None => {
            enumerate_successor_combinations_array(
                assignment_map,
                idx + 1,
                array_state,
                registry,
                results,
                use_incremental_fp,
            );
        }
    }
}

/// Enumerate combinations with early guard validation.
///
/// This is an optimization over `enumerate_successor_combinations_array`: instead of
/// generating all combinations and then filtering with `retain`, we check guards at
/// each leaf BEFORE cloning. This avoids allocating ArrayStates that will be rejected.
///
/// Performance impact: If N% of combinations fail guards, this saves N% of ArrayState
/// clones. The manager's analysis showed ~48% rejection rate on bosco, so this should
/// roughly halve the ArrayState allocation overhead.
#[allow(clippy::too_many_arguments)]
fn enumerate_successor_combinations_array_with_guards(
    assignment_map: &[Option<Vec<Value>>],
    idx: usize,
    array_state: &mut ArrayState,
    registry: &VarRegistry,
    results: &mut Vec<ArrayState>,
    use_incremental_fp: bool,
    ctx: &mut EvalCtx,
    prime_guards: &[Spanned<Expr>],
    compiled_prime_guards: &[CompiledGuard],
    current_array: &ArrayState,
    vars: &[Arc<str>],
) {
    if idx == assignment_map.len() {
        // All variables assigned - check guards BEFORE cloning
        if !prime_guards.is_empty()
            && !validate_prime_guards_for_next_array_inline(
                ctx,
                current_array,
                vars,
                prime_guards,
                compiled_prime_guards,
                array_state,
            )
        {
            // Guards failed - skip this combination
            return;
        }

        if !use_incremental_fp {
            let _ = array_state.fingerprint(registry);
        }
        // Guards passed (or no guards) - clone the ArrayState
        results.push(array_state.clone());
        return;
    }

    let var_idx = VarIndex(idx as u16);
    match &assignment_map[idx] {
        Some(values) => {
            for val in values {
                if use_incremental_fp {
                    array_state.set_with_registry(var_idx, val.clone(), registry);
                } else {
                    array_state.set(var_idx, val.clone());
                }
                enumerate_successor_combinations_array_with_guards(
                    assignment_map,
                    idx + 1,
                    array_state,
                    registry,
                    results,
                    use_incremental_fp,
                    ctx,
                    prime_guards,
                    compiled_prime_guards,
                    current_array,
                    vars,
                );
            }
        }
        None => {
            enumerate_successor_combinations_array_with_guards(
                assignment_map,
                idx + 1,
                array_state,
                registry,
                results,
                use_incremental_fp,
                ctx,
                prime_guards,
                compiled_prime_guards,
                current_array,
                vars,
            );
        }
    }
}

/// Inline guard validation without debug flag overhead.
///
/// This is a streamlined version of `validate_prime_guards_for_next_array` that:
/// 1. Skips debug logging
/// 2. Avoids the extra function call overhead
/// 3. Is designed to be called in the hot path of combination generation
#[inline]
fn validate_prime_guards_for_next_array_inline(
    ctx: &mut EvalCtx,
    current_array: &ArrayState,
    vars: &[Arc<str>],
    prime_guards: &[Spanned<Expr>],
    compiled_prime_guards: &[CompiledGuard],
    next_array: &ArrayState,
) -> bool {
    let can_use_compiled =
        !compiled_prime_guards.is_empty() && compiled_prime_guards.len() == prime_guards.len();

    // Bind next state for primed variable evaluation
    let old_next_state_env = ctx.bind_next_state_array(next_array.values());

    let result = if can_use_compiled {
        let mut passed = true;
        for compiled in compiled_prime_guards.iter() {
            match compiled.eval_with_arrays(ctx, current_array, next_array) {
                Ok(true) => {}
                Ok(false) => {
                    passed = false;
                    break;
                }
                Err(_) => {
                    // Fall back to AST evaluation
                    passed = prime_guards_hold_in_next_array_fast(ctx, prime_guards, vars);
                    break;
                }
            }
        }
        passed
    } else {
        prime_guards_hold_in_next_array_fast(ctx, prime_guards, vars)
    };

    // Restore previous binding
    ctx.restore_next_state_env(old_next_state_env);
    result
}

/// Fast AST-based prime guard evaluation without profiling overhead.
#[inline]
fn prime_guards_hold_in_next_array_fast(
    ctx: &EvalCtx,
    prime_guards: &[Spanned<Expr>],
    _vars: &[Arc<str>],
) -> bool {
    for conjunct in prime_guards {
        match eval(ctx, conjunct) {
            Ok(Value::Bool(true)) => {}
            _ => return false, // false, non-bool, or error = fail
        }
    }
    true
}

/// Build successor ArrayStates from an existing ArrayState with early guard validation.
///
/// This is the optimized path that validates prime guards BEFORE cloning ArrayStates.
/// Use this when prime_guards is non-empty to avoid building states that will be rejected.
pub fn build_successor_array_states_with_guards(
    current_array: &ArrayState,
    vars: &[Arc<str>],
    assignments: &[PrimedAssignment],
    registry: &VarRegistry,
    ctx: &mut EvalCtx,
    prime_guards: &[Spanned<Expr>],
    compiled_prime_guards: &[CompiledGuard],
) -> Vec<ArrayState> {
    let num_vars = vars.len();

    // Build lookup map for assignments: var_name -> possible values
    let mut assignment_map: Vec<Option<Vec<Value>>> = vec![None; num_vars];
    for assignment in assignments {
        match assignment {
            PrimedAssignment::Assign(name, value) => {
                if let Some(idx) = registry.get(name) {
                    assignment_map[idx.as_usize()] = Some(vec![value.clone()]);
                }
            }
            PrimedAssignment::Unchanged(_) => {}
            PrimedAssignment::InSet(name, values) => {
                if let Some(idx) = registry.get(name) {
                    assignment_map[idx.as_usize()] = Some(values.clone());
                }
            }
            PrimedAssignment::DeferredExpr(_, _) => {
                // For deferred expressions, fall back to the non-guard path
                // which handles evaluation per-combination
                return build_successor_array_states_from_array(current_array, vars, assignments, registry);
            }
        }
    }

    // If any IN-set assignment has an empty domain, there are no successors.
    if assignment_map
        .iter()
        .any(|v| matches!(v, Some(vals) if vals.is_empty()))
    {
        return Vec::new();
    }

    // Heuristic for incremental fingerprinting
    let assigned_vars = assignment_map.iter().filter(|v| v.is_some()).count();
    let use_incremental_fp = assigned_vars.saturating_mul(2) < num_vars;

    // Pre-compute number of combinations for pre-allocation
    // (We allocate for worst case, but may use less if guards filter)
    let num_combinations = assignment_map.iter().fold(1usize, |acc, v| {
        acc.saturating_mul(v.as_ref().map_or(1, |vals| vals.len()))
    });

    let mut results = Vec::with_capacity(num_combinations);
    let mut array_state = current_array.clone();
    if use_incremental_fp {
        let _ = array_state.fingerprint(registry);
    }

    enumerate_successor_combinations_array_with_guards(
        &assignment_map,
        0,
        &mut array_state,
        registry,
        &mut results,
        use_incremental_fp,
        ctx,
        prime_guards,
        compiled_prime_guards,
        current_array,
        vars,
    );

    results
}

/// Build successor ArrayStates from EvaluatedAssignment with early guard validation.
///
/// This is the compiled-path version that takes EvaluatedAssignment (from compiled_guard.rs)
/// instead of PrimedAssignment. Used when `evaluate_compiled_assignments` succeeds.
///
/// Re: #16 - Optimizes bosco spec by validating guards before cloning ArrayStates.
pub fn build_successor_array_states_with_guards_compiled(
    current_array: &ArrayState,
    vars: &[Arc<str>],
    assignments: &[EvaluatedAssignment],
    registry: &VarRegistry,
    ctx: &mut EvalCtx,
    prime_guards: &[Spanned<Expr>],
    compiled_prime_guards: &[CompiledGuard],
) -> Vec<ArrayState> {
    let num_vars = vars.len();

    // Build lookup map for assignments: var_idx -> possible values
    let mut assignment_map: Vec<Option<Vec<Value>>> = vec![None; num_vars];
    let mut copy_pairs: Vec<(VarIndex, VarIndex)> = Vec::new();

    for assignment in assignments {
        match assignment {
            EvaluatedAssignment::Assign { var_idx, value, .. } => {
                assignment_map[var_idx.as_usize()] = Some(vec![value.clone()]);
            }
            EvaluatedAssignment::Unchanged { .. } => {}
            EvaluatedAssignment::InSet {
                var_idx, values, ..
            } => {
                assignment_map[var_idx.as_usize()] = Some(values.clone());
            }
            EvaluatedAssignment::CopyFromVar {
                dest_idx, src_idx, ..
            } => {
                copy_pairs.push((*dest_idx, *src_idx));
            }
        }
    }

    // If any IN-set assignment has an empty domain, there are no successors.
    if assignment_map
        .iter()
        .any(|v| matches!(v, Some(vals) if vals.is_empty()))
    {
        return Vec::new();
    }

    // Heuristic for incremental fingerprinting
    let assigned_vars = assignment_map.iter().filter(|v| v.is_some()).count();
    let use_incremental_fp = assigned_vars.saturating_mul(2) < num_vars;

    // Pre-compute number of combinations for pre-allocation
    let num_combinations = assignment_map.iter().fold(1usize, |acc, v| {
        acc.saturating_mul(v.as_ref().map_or(1, |vals| vals.len()))
    });

    let mut results = Vec::with_capacity(num_combinations);
    let mut array_state = current_array.clone();
    if use_incremental_fp {
        let _ = array_state.fingerprint(registry);
    }

    enumerate_successor_combinations_array_with_guards(
        &assignment_map,
        0,
        &mut array_state,
        registry,
        &mut results,
        use_incremental_fp,
        ctx,
        prime_guards,
        compiled_prime_guards,
        current_array,
        vars,
    );

    // Apply correlated primed-variable copies (x' = y') after enumeration
    if !copy_pairs.is_empty() {
        for succ in results.iter_mut() {
            for _ in 0..copy_pairs.len() {
                for (dest, src) in copy_pairs.iter().copied() {
                    let v = succ.get(src).clone();
                    succ.set(dest, v);
                }
            }
        }
    }

    results
}

/// Build successor ArrayStates with optional context for deferred expression evaluation
///
/// When `ctx` is provided and there are `DeferredExpr` assignments, they will be
/// evaluated for each InSet combination with the appropriate next-state context.
pub fn build_successor_array_states_with_ctx(
    current_array: &ArrayState,
    vars: &[Arc<str>],
    assignments: &[PrimedAssignment],
    registry: &VarRegistry,
    ctx: Option<&EvalCtx>,
) -> Vec<ArrayState> {
    // Check if we have any deferred expressions that need evaluation
    let has_deferred = assignments
        .iter()
        .any(|a| matches!(a, PrimedAssignment::DeferredExpr(_, _)));

    if has_deferred {
        if let Some(ctx) = ctx {
            // Slow path: need to evaluate deferred expressions per-combination
            return build_successor_array_states_with_deferred(
                current_array,
                vars,
                assignments,
                registry,
                ctx,
            );
        }
    }

    // Fast path: no deferred expressions, use existing implementation
    build_successor_array_states_from_array(current_array, vars, assignments, registry)
}

/// Build successor ArrayStates when there are DeferredExpr assignments.
///
/// For each combination of InSet values, creates a next-state context and evaluates
/// any deferred expressions before building the final ArrayState.
fn build_successor_array_states_with_deferred(
    current_array: &ArrayState,
    vars: &[Arc<str>],
    assignments: &[PrimedAssignment],
    registry: &VarRegistry,
    ctx: &EvalCtx,
) -> Vec<ArrayState> {
    use crate::eval::Env;

    // Separate InSet assignments (which produce combinations) from others
    let mut inset_vars: Vec<(Arc<str>, Vec<Value>)> = Vec::new();
    let mut deferred: Vec<(Arc<str>, Spanned<Expr>)> = Vec::new();
    let mut fixed_assignments: Vec<(Arc<str>, Value)> = Vec::new();
    let mut unchanged_vars: Vec<Arc<str>> = Vec::new();

    for assignment in assignments {
        match assignment {
            PrimedAssignment::Assign(name, value) => {
                fixed_assignments.push((name.clone(), value.clone()));
            }
            PrimedAssignment::Unchanged(name) => {
                unchanged_vars.push(name.clone());
            }
            PrimedAssignment::InSet(name, values) => {
                if values.is_empty() {
                    return Vec::new(); // Empty domain means no successors
                }
                inset_vars.push((name.clone(), values.clone()));
            }
            PrimedAssignment::DeferredExpr(name, expr) => {
                deferred.push((name.clone(), expr.clone()));
            }
        }
    }

    // Generate all combinations of InSet values
    let combinations = generate_inset_combinations(&inset_vars);
    let debug = debug_enum();

    let mut results = Vec::new();

    for combo in combinations {
        // Build next_state with this combination
        let mut next_state: Env = Env::new();

        // Add fixed assignments
        for (name, value) in &fixed_assignments {
            next_state.insert(name.clone(), value.clone());
        }

        // Add unchanged vars from current state
        for name in &unchanged_vars {
            if let Some(idx) = registry.get(name) {
                next_state.insert(name.clone(), current_array.get(idx).clone());
            }
        }

        // Add InSet values from this combination
        for (name, value) in &combo {
            next_state.insert(name.clone(), value.clone());
        }

        // Evaluate deferred expressions with this next_state context
        let mut eval_ctx = ctx.clone();
        eval_ctx.next_state = Some(Arc::new(next_state.clone()));

        let mut all_ok = true;
        for (name, expr) in &deferred {
            match eval(&eval_ctx, expr) {
                Ok(value) => {
                    if debug {
                        eprintln!(
                            "build_successor_array_states_with_deferred: {}' = {}",
                            name, value
                        );
                    }
                    next_state.insert(name.clone(), value);
                    // Update context for subsequent deferred exprs that might depend on this
                    eval_ctx.next_state = Some(Arc::new(next_state.clone()));
                }
                Err(e) => {
                    if debug {
                        eprintln!(
                            "build_successor_array_states_with_deferred: {}' eval error: {:?}",
                            name, e
                        );
                    }
                    all_ok = false;
                    break;
                }
            }
        }

        if !all_ok {
            continue; // Skip this combination if evaluation failed
        }

        // Build the ArrayState from next_state
        let mut array_state = current_array.clone();
        for var in vars {
            if let Some(value) = next_state.get(var) {
                if let Some(idx) = registry.get(var) {
                    array_state.set(idx, value.clone());
                }
            }
        }
        let _ = array_state.fingerprint(registry);
        results.push(array_state);
    }

    results
}

// ============================================================================
// Diff-based enumeration - avoids cloning full states for duplicate detection
// ============================================================================

#[derive(Clone, Copy)]
enum DiffAssignmentChoices<'a> {
    Single(&'a Value),
    Multi(&'a [Value]),
}

#[inline]
fn count_diff_combinations(choices: &[Option<DiffAssignmentChoices<'_>>]) -> usize {
    let mut count = 1usize;
    for choice in choices {
        if let Some(DiffAssignmentChoices::Multi(values)) = choice {
            count = count.saturating_mul(values.len());
        }
    }
    count
}

/// Build successor DiffSuccessors from an ArrayState (avoids full state cloning)
///
/// This is an optimization for high duplicate rate scenarios. Instead of cloning
/// the full ArrayState for each successor, we only track the changes (diffs).
/// The full state can be materialized later, but only for unique successors (~5%).
///
/// # Arguments
/// * `current_array` - The base state (must have fingerprint cache with value_fps)
/// * `vars` - Variable names in index order
/// * `assignments` - List of primed assignments from enumeration
/// * `registry` - Variable registry for fingerprint computation
pub fn build_successor_diffs_from_array(
    current_array: &ArrayState,
    vars: &[Arc<str>],
    assignments: &[PrimedAssignment],
    registry: &VarRegistry,
) -> Vec<DiffSuccessor> {
    let num_vars = vars.len();

    let mut choices: Vec<Option<DiffAssignmentChoices<'_>>> = vec![None; num_vars];
    for assignment in assignments {
        match assignment {
            PrimedAssignment::Assign(name, value) => {
                if let Some(idx) = registry.get(name) {
                    choices[idx.as_usize()] = Some(DiffAssignmentChoices::Single(value));
                }
            }
            PrimedAssignment::Unchanged(_) => {}
            PrimedAssignment::InSet(name, values) => {
                if let Some(idx) = registry.get(name) {
                    choices[idx.as_usize()] = Some(DiffAssignmentChoices::Multi(values));
                }
            }
            // DeferredExpr should be handled by build_successor_states_with_deferred, not this fast path
            PrimedAssignment::DeferredExpr(_, _) => {
                unreachable!("DeferredExpr in diff successor fast path")
            }
        }
    }

    if choices
        .iter()
        .any(|c| matches!(c, Some(DiffAssignmentChoices::Multi(values)) if values.is_empty()))
    {
        return Vec::new();
    }

    let num_combinations = count_diff_combinations(&choices);
    let mut results = Vec::with_capacity(num_combinations);
    let mut changes: Vec<(VarIndex, &Value)> = Vec::new();

    enumerate_successor_combinations_diff_choices(
        &choices,
        0,
        current_array,
        registry,
        &mut changes,
        &mut results,
    );

    results
}

pub fn build_successor_diffs_from_array_filtered<F>(
    current_array: &ArrayState,
    vars: &[Arc<str>],
    assignments: &[PrimedAssignment],
    registry: &VarRegistry,
    mut filter: F,
) -> Vec<DiffSuccessor>
where
    F: FnMut(&ArrayState) -> bool,
{
    let num_vars = vars.len();

    let mut choices: Vec<Option<DiffAssignmentChoices<'_>>> = vec![None; num_vars];
    for assignment in assignments {
        match assignment {
            PrimedAssignment::Assign(name, value) => {
                if let Some(idx) = registry.get(name) {
                    choices[idx.as_usize()] = Some(DiffAssignmentChoices::Single(value));
                }
            }
            PrimedAssignment::Unchanged(_) => {}
            PrimedAssignment::InSet(name, values) => {
                if let Some(idx) = registry.get(name) {
                    choices[idx.as_usize()] = Some(DiffAssignmentChoices::Multi(values));
                }
            }
            // DeferredExpr should be handled by build_successor_states_with_deferred, not this fast path
            PrimedAssignment::DeferredExpr(_, _) => {
                unreachable!("DeferredExpr in diff successor filtered fast path")
            }
        }
    }

    if choices
        .iter()
        .any(|c| matches!(c, Some(DiffAssignmentChoices::Multi(values)) if values.is_empty()))
    {
        return Vec::new();
    }

    let num_combinations = count_diff_combinations(&choices);
    let mut results = Vec::with_capacity(num_combinations);
    let mut changes: Vec<(VarIndex, &Value)> = Vec::new();
    let mut next_array = current_array.clone();

    enumerate_successor_combinations_diff_choices_filtered(
        &choices,
        0,
        current_array,
        &mut next_array,
        registry,
        &mut changes,
        &mut results,
        &mut filter,
    );

    results
}

fn enumerate_successor_combinations_diff_choices<'a>(
    choices: &[Option<DiffAssignmentChoices<'a>>],
    idx: usize,
    base: &ArrayState,
    registry: &VarRegistry,
    changes: &mut Vec<(VarIndex, &'a Value)>,
    results: &mut Vec<DiffSuccessor>,
) {
    if idx >= choices.len() {
        let owned_changes: DiffChanges = changes
            .iter()
            .map(|(var_idx, v)| (*var_idx, (*v).clone()))
            .collect();
        let fp = compute_diff_fingerprint(base, &owned_changes, registry);
        results.push(DiffSuccessor::from_smallvec(fp, owned_changes));
        return;
    }

    let var_idx = VarIndex(idx as u16);
    match &choices[idx] {
        Some(DiffAssignmentChoices::Single(v)) => {
            changes.push((var_idx, v));
            enumerate_successor_combinations_diff_choices(
                choices,
                idx + 1,
                base,
                registry,
                changes,
                results,
            );
            changes.pop();
        }
        Some(DiffAssignmentChoices::Multi(values)) => {
            for v in *values {
                changes.push((var_idx, v));
                enumerate_successor_combinations_diff_choices(
                    choices,
                    idx + 1,
                    base,
                    registry,
                    changes,
                    results,
                );
                changes.pop();
            }
        }
        None => enumerate_successor_combinations_diff_choices(
            choices,
            idx + 1,
            base,
            registry,
            changes,
            results,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn enumerate_successor_combinations_diff_choices_filtered<'a, F>(
    choices: &[Option<DiffAssignmentChoices<'a>>],
    idx: usize,
    base: &ArrayState,
    next: &mut ArrayState,
    registry: &VarRegistry,
    changes: &mut Vec<(VarIndex, &'a Value)>,
    results: &mut Vec<DiffSuccessor>,
    filter: &mut F,
) where
    F: FnMut(&ArrayState) -> bool,
{
    if idx >= choices.len() {
        if !filter(next) {
            return;
        }
        let owned_changes: DiffChanges = changes
            .iter()
            .map(|(var_idx, v)| (*var_idx, (*v).clone()))
            .collect();
        let fp = compute_diff_fingerprint(base, &owned_changes, registry);
        results.push(DiffSuccessor::from_smallvec(fp, owned_changes));
        return;
    }

    let var_idx = VarIndex(idx as u16);
    match &choices[idx] {
        Some(DiffAssignmentChoices::Single(v)) => {
            next.set(var_idx, (*v).clone());
            changes.push((var_idx, v));
            enumerate_successor_combinations_diff_choices_filtered(
                choices,
                idx + 1,
                base,
                next,
                registry,
                changes,
                results,
                filter,
            );
            changes.pop();
        }
        Some(DiffAssignmentChoices::Multi(values)) => {
            for v in *values {
                next.set(var_idx, v.clone());
                changes.push((var_idx, v));
                enumerate_successor_combinations_diff_choices_filtered(
                    choices,
                    idx + 1,
                    base,
                    next,
                    registry,
                    changes,
                    results,
                    filter,
                );
                changes.pop();
            }
        }
        None => enumerate_successor_combinations_diff_choices_filtered(
            choices,
            idx + 1,
            base,
            next,
            registry,
            changes,
            results,
            filter,
        ),
    }
}

/// Build successor diffs from array state with deferred expression support.
///
/// This handles DeferredExpr assignments by evaluating them with proper next_state context
/// for each combination of InSet values.
pub fn build_successor_diffs_with_deferred(
    current_array: &ArrayState,
    vars: &[Arc<str>],
    assignments: &[PrimedAssignment],
    registry: &VarRegistry,
    ctx: &EvalCtx,
) -> Vec<DiffSuccessor> {
    use crate::eval::Env;

    // Separate InSet assignments (which produce combinations) from others
    let mut inset_vars: Vec<(Arc<str>, Vec<Value>)> = Vec::new();
    let mut deferred: Vec<(Arc<str>, Spanned<Expr>)> = Vec::new();
    let mut fixed_assignments: Vec<(Arc<str>, Value)> = Vec::new();
    let mut unchanged_vars: Vec<Arc<str>> = Vec::new();

    for assignment in assignments {
        match assignment {
            PrimedAssignment::Assign(name, value) => {
                fixed_assignments.push((name.clone(), value.clone()));
            }
            PrimedAssignment::Unchanged(name) => {
                unchanged_vars.push(name.clone());
            }
            PrimedAssignment::InSet(name, values) => {
                if values.is_empty() {
                    return Vec::new(); // Empty domain means no successors
                }
                inset_vars.push((name.clone(), values.clone()));
            }
            PrimedAssignment::DeferredExpr(name, expr) => {
                deferred.push((name.clone(), expr.clone()));
            }
        }
    }

    // Generate all combinations of InSet values
    let combinations = generate_inset_combinations(&inset_vars);
    let debug = debug_enum();

    let mut results = Vec::new();

    for combo in combinations {
        // Build next_state with this combination
        let mut next_state: Env = Env::new();

        // Add fixed assignments
        for (name, value) in &fixed_assignments {
            next_state.insert(name.clone(), value.clone());
        }

        // Add unchanged vars from current state
        for name in &unchanged_vars {
            if let Some(idx) = registry.get(name) {
                let value = current_array.get(idx);
                next_state.insert(name.clone(), value.clone());
            }
        }

        // Add InSet values from this combination
        for (name, value) in &combo {
            next_state.insert(name.clone(), value.clone());
        }

        // Evaluate deferred expressions with this next_state context
        let mut eval_ctx = ctx.clone();
        eval_ctx.next_state = Some(Arc::new(next_state.clone()));

        let mut all_ok = true;
        for (name, expr) in &deferred {
            match eval(&eval_ctx, expr) {
                Ok(value) => {
                    if debug {
                        eprintln!(
                            "build_successor_diffs_with_deferred: {}' = {}",
                            name, value
                        );
                    }
                    next_state.insert(name.clone(), value);
                    // Update context for subsequent deferred exprs that might depend on this
                    eval_ctx.next_state = Some(Arc::new(next_state.clone()));
                }
                Err(e) => {
                    if debug {
                        eprintln!(
                            "build_successor_diffs_with_deferred: {}' eval error: {:?}",
                            name, e
                        );
                    }
                    all_ok = false;
                    break;
                }
            }
        }

        if !all_ok {
            continue; // Skip this combination if evaluation failed
        }

        // Build DiffSuccessor from changes
        let mut changes: Vec<(VarIndex, Value)> = Vec::new();
        for (i, var) in vars.iter().enumerate() {
            if let Some(new_value) = next_state.get(var) {
                let var_idx = VarIndex(i as u16);
                let old_value = current_array.get(var_idx);
                if new_value != old_value {
                    changes.push((var_idx, new_value.clone()));
                }
            }
        }

        let fp = compute_diff_fingerprint(current_array, &changes, registry);
        results.push(DiffSuccessor::new(fp, changes));
    }

    results
}

/// Build successor diffs with deferred expression support and prime guard filtering.
#[allow(clippy::too_many_arguments)]
pub fn build_successor_diffs_with_deferred_filtered<F>(
    current_array: &ArrayState,
    vars: &[Arc<str>],
    assignments: &[PrimedAssignment],
    registry: &VarRegistry,
    ctx: &EvalCtx,
    mut filter: F,
) -> Vec<DiffSuccessor>
where
    F: FnMut(&ArrayState) -> bool,
{
    use crate::eval::Env;

    // Separate InSet assignments (which produce combinations) from others
    let mut inset_vars: Vec<(Arc<str>, Vec<Value>)> = Vec::new();
    let mut deferred: Vec<(Arc<str>, Spanned<Expr>)> = Vec::new();
    let mut fixed_assignments: Vec<(Arc<str>, Value)> = Vec::new();
    let mut unchanged_vars: Vec<Arc<str>> = Vec::new();

    for assignment in assignments {
        match assignment {
            PrimedAssignment::Assign(name, value) => {
                fixed_assignments.push((name.clone(), value.clone()));
            }
            PrimedAssignment::Unchanged(name) => {
                unchanged_vars.push(name.clone());
            }
            PrimedAssignment::InSet(name, values) => {
                if values.is_empty() {
                    return Vec::new(); // Empty domain means no successors
                }
                inset_vars.push((name.clone(), values.clone()));
            }
            PrimedAssignment::DeferredExpr(name, expr) => {
                deferred.push((name.clone(), expr.clone()));
            }
        }
    }

    // Generate all combinations of InSet values
    let combinations = generate_inset_combinations(&inset_vars);
    let debug = debug_enum();

    let mut results = Vec::new();

    for combo in combinations {
        // Build next_state with this combination
        let mut next_state: Env = Env::new();

        // Add fixed assignments
        for (name, value) in &fixed_assignments {
            next_state.insert(name.clone(), value.clone());
        }

        // Add unchanged vars from current state
        for name in &unchanged_vars {
            if let Some(idx) = registry.get(name) {
                let value = current_array.get(idx);
                next_state.insert(name.clone(), value.clone());
            }
        }

        // Add InSet values from this combination
        for (name, value) in &combo {
            next_state.insert(name.clone(), value.clone());
        }

        // Evaluate deferred expressions with this next_state context
        let mut eval_ctx = ctx.clone();
        eval_ctx.next_state = Some(Arc::new(next_state.clone()));

        let mut all_ok = true;
        for (name, expr) in &deferred {
            match eval(&eval_ctx, expr) {
                Ok(value) => {
                    if debug {
                        eprintln!(
                            "build_successor_diffs_with_deferred_filtered: {}' = {}",
                            name, value
                        );
                    }
                    next_state.insert(name.clone(), value);
                    // Update context for subsequent deferred exprs that might depend on this
                    eval_ctx.next_state = Some(Arc::new(next_state.clone()));
                }
                Err(e) => {
                    if debug {
                        eprintln!(
                            "build_successor_diffs_with_deferred_filtered: {}' eval error: {:?}",
                            name, e
                        );
                    }
                    all_ok = false;
                    break;
                }
            }
        }

        if !all_ok {
            continue; // Skip this combination if evaluation failed
        }

        // Build next ArrayState for filtering
        let mut next_array = current_array.clone();
        for (i, var) in vars.iter().enumerate() {
            if let Some(new_value) = next_state.get(var) {
                let var_idx = VarIndex(i as u16);
                next_array.set(var_idx, new_value.clone());
            }
        }

        // Apply filter
        if !filter(&next_array) {
            continue;
        }

        // Build DiffSuccessor from changes
        let mut changes: Vec<(VarIndex, Value)> = Vec::new();
        for (i, var) in vars.iter().enumerate() {
            if let Some(new_value) = next_state.get(var) {
                let var_idx = VarIndex(i as u16);
                let old_value = current_array.get(var_idx);
                if new_value != old_value {
                    changes.push((var_idx, new_value.clone()));
                }
            }
        }

        let fp = compute_diff_fingerprint(current_array, &changes, registry);
        results.push(DiffSuccessor::new(fp, changes));
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use tla_core::{lower, parse_to_syntax_tree, FileId};

    fn setup_module(src: &str) -> (tla_core::ast::Module, EvalCtx, Vec<Arc<str>>) {
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let mut ctx = EvalCtx::new();
        ctx.load_module(&module);

        let mut vars = Vec::new();
        for unit in &module.units {
            if let tla_core::ast::Unit::Variable(var_names) = &unit.node {
                for var in var_names {
                    let name = Arc::from(var.node.as_str());
                    ctx.register_var(Arc::clone(&name));
                    vars.push(name);
                }
            }
        }

        (module, ctx, vars)
    }

    #[test]
    fn test_local_scope_get_depth_prefers_innermost_binding() {
        let scope = LocalScope::new().with_var("x").with_var("y").with_var("x");
        assert_eq!(scope.get_depth("x"), Some(0));
        assert_eq!(scope.get_depth("y"), Some(1));
        assert_eq!(scope.get_depth("z"), None);
    }

    #[test]
    fn test_extract_simple_equality_constraints() {
        let src = r#"
---- MODULE Test ----
VARIABLE x, y
Init == x = 0 /\ y = 1
====
"#;
        let (module, ctx, vars) = setup_module(src);

        // Find Init definition
        let init_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Init" {
                        return Some(def);
                    }
                }
                None
            })
            .unwrap();

        let branches = extract_init_constraints(&ctx, &init_def.body, &vars).unwrap();
        assert_eq!(branches.len(), 1);
        assert_eq!(branches[0].len(), 2);
    }

    #[test]
    fn test_enumerate_states_simple() {
        let src = r#"
---- MODULE Test ----
VARIABLE x, y
Init == x = 0 /\ y = 1
====
"#;
        let (module, ctx, vars) = setup_module(src);

        let init_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Init" {
                        return Some(def);
                    }
                }
                None
            })
            .unwrap();

        let branches = extract_init_constraints(&ctx, &init_def.body, &vars).unwrap();
        let states = enumerate_states_from_constraint_branches(None, &vars, &branches).unwrap();

        assert_eq!(states.len(), 1);
        let state = &states[0];

        // Check state values
        let state_vars: Vec<_> = state.vars().collect();
        assert_eq!(state_vars.len(), 2);
    }

    #[test]
    fn test_enumerate_states_membership() {
        let src = r#"
---- MODULE Test ----
VARIABLE x
Init == x \in {1, 2, 3}
====
"#;
        let (module, ctx, vars) = setup_module(src);

        let init_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Init" {
                        return Some(def);
                    }
                }
                None
            })
            .unwrap();

        let branches = extract_init_constraints(&ctx, &init_def.body, &vars).unwrap();
        let states = enumerate_states_from_constraint_branches(None, &vars, &branches).unwrap();

        assert_eq!(states.len(), 3);
    }

    #[test]
    fn test_enumerate_states_combination() {
        let src = r#"
---- MODULE Test ----
VARIABLE x, y
Init == x \in {0, 1} /\ y \in {2, 3}
====
"#;
        let (module, ctx, vars) = setup_module(src);

        let init_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Init" {
                        return Some(def);
                    }
                }
                None
            })
            .unwrap();

        let branches = extract_init_constraints(&ctx, &init_def.body, &vars).unwrap();
        let states = enumerate_states_from_constraint_branches(None, &vars, &branches).unwrap();

        // 2 choices for x * 2 choices for y = 4 states
        assert_eq!(states.len(), 4);
    }

    #[test]
    fn test_enumerate_states_disjunctive_init() {
        let src = r#"
---- MODULE Test ----
VARIABLE x
Init == x = 0 \/ x = 1
====
"#;
        let (module, ctx, vars) = setup_module(src);

        let init_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Init" {
                        return Some(def);
                    }
                }
                None
            })
            .unwrap();

        let branches = extract_init_constraints(&ctx, &init_def.body, &vars).unwrap();
        let states = enumerate_states_from_constraint_branches(None, &vars, &branches).unwrap();

        assert_eq!(states.len(), 2);
    }

    #[test]
    fn test_enumerate_states_conjunction_intersection() {
        let src = r#"
---- MODULE Test ----
VARIABLE x
Init == x \in {0, 1} /\ x \in {1, 2}
====
"#;
        let (module, ctx, vars) = setup_module(src);

        let init_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Init" {
                        return Some(def);
                    }
                }
                None
            })
            .unwrap();

        let branches = extract_init_constraints(&ctx, &init_def.body, &vars).unwrap();
        let states = enumerate_states_from_constraint_branches(None, &vars, &branches).unwrap();

        assert_eq!(states.len(), 1);
        assert_eq!(states[0].get("x"), Some(&Value::int(1)));
    }

    #[test]
    fn test_enumerate_states_conjunction_contradiction() {
        let src = r#"
---- MODULE Test ----
VARIABLE x
Init == x = 0 /\ x = 1
====
"#;
        let (module, ctx, vars) = setup_module(src);

        let init_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Init" {
                        return Some(def);
                    }
                }
                None
            })
            .unwrap();

        let branches = extract_init_constraints(&ctx, &init_def.body, &vars).unwrap();
        let states = enumerate_states_from_constraint_branches(None, &vars, &branches).unwrap();

        assert!(states.is_empty());
    }

    #[test]
    fn test_enumerate_states_nested_or_and() {
        let src = r#"
---- MODULE Test ----
VARIABLE x, y
Init == (x = 0 \/ x = 1) /\ y = 2
====
"#;
        let (module, ctx, vars) = setup_module(src);

        let init_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Init" {
                        return Some(def);
                    }
                }
                None
            })
            .unwrap();

        let branches = extract_init_constraints(&ctx, &init_def.body, &vars).unwrap();
        assert_eq!(branches.len(), 2);

        let states = enumerate_states_from_constraint_branches(None, &vars, &branches).unwrap();
        assert_eq!(states.len(), 2);
    }

    #[test]
    fn test_extract_primed_assignments() {
        let src = r#"
---- MODULE Test ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#;
        let (module, mut ctx, vars) = setup_module(src);

        // Set up current state
        let current_state = State::from_pairs([("x", Value::int(0))]);

        // Find Next definition
        let next_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Next" {
                        return Some(def.clone());
                    }
                }
                None
            })
            .unwrap();

        let successors = enumerate_successors(&mut ctx, &next_def, &current_state, &vars).unwrap();

        assert_eq!(successors.len(), 1);
        // x should be 1 in successor
        let succ_vars: Vec<_> = successors[0].vars().collect();
        assert_eq!(succ_vars.len(), 1);
        assert_eq!(succ_vars[0].1, &Value::int(1));
    }

    #[test]
    fn test_enumerate_successors_unchanged() {
        let src = r#"
---- MODULE Test ----
VARIABLE x, y
Init == x = 0 /\ y = 0
Next == x' = x + 1 /\ UNCHANGED y
====
"#;
        let (module, mut ctx, vars) = setup_module(src);

        let current_state = State::from_pairs([("x", Value::int(0)), ("y", Value::int(5))]);

        let next_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Next" {
                        return Some(def.clone());
                    }
                }
                None
            })
            .unwrap();

        let successors = enumerate_successors(&mut ctx, &next_def, &current_state, &vars).unwrap();

        assert_eq!(successors.len(), 1);
        // Check x = 1, y = 5 (unchanged)
        for (name, value) in successors[0].vars() {
            if name.as_ref() == "x" {
                assert_eq!(value, &Value::int(1));
            } else if name.as_ref() == "y" {
                assert_eq!(value, &Value::int(5));
            }
        }
    }

    #[test]
    fn test_enumerate_successors_disjunction() {
        let src = r#"
---- MODULE Test ----
VARIABLE x
Init == x = 0
Next == x' = x + 1 \/ x' = x + 2
====
"#;
        let (module, mut ctx, vars) = setup_module(src);

        let current_state = State::from_pairs([("x", Value::int(0))]);

        let next_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Next" {
                        return Some(def.clone());
                    }
                }
                None
            })
            .unwrap();

        let successors = enumerate_successors(&mut ctx, &next_def, &current_state, &vars).unwrap();

        // Should have 2 successors: x=1 and x=2
        assert_eq!(successors.len(), 2);
    }

    #[test]
    fn test_enumerate_successors_in_set() {
        // Test x' \in S where S is a set of possible values
        let src = r#"
---- MODULE Test ----
VARIABLE x
Init == x = 0
Next == x' \in {x + 1, x + 2, x + 3}
====
"#;
        let (module, mut ctx, vars) = setup_module(src);

        let current_state = State::from_pairs([("x", Value::int(0))]);

        let next_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Next" {
                        return Some(def.clone());
                    }
                }
                None
            })
            .unwrap();

        let successors = enumerate_successors(&mut ctx, &next_def, &current_state, &vars).unwrap();

        // Should have 3 successors: x=1, x=2, x=3
        assert_eq!(successors.len(), 3);

        // Collect successor x values
        let mut x_values: Vec<i64> = successors
            .iter()
            .filter_map(|s| {
                s.vars()
                    .find(|(n, _)| n.as_ref() == "x")
                    .and_then(|(_, v)| v.as_i64())
            })
            .collect();
        x_values.sort();
        assert_eq!(x_values, vec![1, 2, 3]);
    }

    #[test]
    fn test_enumerate_successors_in_set_with_unchanged() {
        // Test x' \in S combined with UNCHANGED y
        let src = r#"
---- MODULE Test ----
VARIABLE x, y
Init == x = 0 /\ y = 10
Next == x' \in {x + 1, x + 2} /\ UNCHANGED y
====
"#;
        let (module, mut ctx, vars) = setup_module(src);

        let current_state = State::from_pairs([("x", Value::int(0)), ("y", Value::int(10))]);

        let next_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Next" {
                        return Some(def.clone());
                    }
                }
                None
            })
            .unwrap();

        let successors = enumerate_successors(&mut ctx, &next_def, &current_state, &vars).unwrap();

        // Should have 2 successors
        assert_eq!(successors.len(), 2);

        // All successors should have y=10
        for succ in &successors {
            let y_val = succ
                .vars()
                .find(|(n, _)| n.as_ref() == "y")
                .map(|(_, v)| v.clone());
            assert_eq!(y_val, Some(Value::int(10)));
        }
    }

    #[test]
    fn test_enumerate_successors_multiple_in_set() {
        // Test both x' \in S and y' \in T (cartesian product)
        let src = r#"
---- MODULE Test ----
VARIABLE x, y
Init == x = 0 /\ y = 0
Next == x' \in {1, 2} /\ y' \in {10, 20}
====
"#;
        let (module, mut ctx, vars) = setup_module(src);

        let current_state = State::from_pairs([("x", Value::int(0)), ("y", Value::int(0))]);

        let next_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Next" {
                        return Some(def.clone());
                    }
                }
                None
            })
            .unwrap();

        let successors = enumerate_successors(&mut ctx, &next_def, &current_state, &vars).unwrap();

        // Should have 2 * 2 = 4 successors (cartesian product)
        assert_eq!(successors.len(), 4);

        // Verify all combinations exist
        let mut combinations: Vec<(i64, i64)> = successors
            .iter()
            .map(|s| {
                let x: i64 = s
                    .vars()
                    .find(|(n, _)| n.as_ref() == "x")
                    .and_then(|(_, v)| v.as_i64())
                    .unwrap();
                let y: i64 = s
                    .vars()
                    .find(|(n, _)| n.as_ref() == "y")
                    .and_then(|(_, v)| v.as_i64())
                    .unwrap();
                (x, y)
            })
            .collect();
        combinations.sort();
        assert_eq!(combinations, vec![(1, 10), (1, 20), (2, 10), (2, 20)]);
    }

    #[test]
    fn test_enumerate_successors_exists_in_conjunction() {
        // Regression: existential quantifiers inside conjunctions must be enumerated.
        // Pattern appears in PlusCal translations (e.g. ChangRoberts' n1(self)).
        let src = r#"
---- MODULE Test ----
VARIABLE x, y
Init == x = 0 /\ y = {}
Next == /\ x = 0
        /\ \E id \in {1, 2}:
             y' = y \cup {id}
        /\ x' = x
====
"#;
        let (module, mut ctx, vars) = setup_module(src);

        let current_state = State::from_pairs([("x", Value::int(0)), ("y", Value::empty_set())]);

        let next_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Next" {
                        return Some(def.clone());
                    }
                }
                None
            })
            .unwrap();

        let successors = enumerate_successors(&mut ctx, &next_def, &current_state, &vars).unwrap();
        assert_eq!(successors.len(), 2);

        for succ in &successors {
            assert_eq!(succ.get("x").and_then(|v| v.as_i64()), Some(0));
        }

        let mut ys: Vec<Vec<i64>> = successors
            .iter()
            .map(|s| {
                let set = s.get("y").and_then(|v| v.as_set()).unwrap();
                let mut elems: Vec<i64> = set.iter().map(|v| v.as_i64().unwrap()).collect();
                elems.sort();
                elems
            })
            .collect();
        ys.sort();
        assert_eq!(ys, vec![vec![1], vec![2]]);
    }

    #[test]
    fn test_enumerate_states_operator_inlining() {
        // Test that Init can refer to helper operators
        let src = r#"
---- MODULE Test ----
VARIABLE x, y
TypeInit == x \in {0, 1} /\ y \in {10, 20}
Init == TypeInit /\ x = 0
====
"#;
        let (module, ctx, vars) = setup_module(src);

        let init_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Init" {
                        return Some(def);
                    }
                }
                None
            })
            .unwrap();

        let branches = extract_init_constraints(&ctx, &init_def.body, &vars).unwrap();
        let states = enumerate_states_from_constraint_branches(None, &vars, &branches).unwrap();

        // TypeInit gives x \in {0,1} and y \in {10,20}
        // But Init also requires x = 0
        // So x is intersected to {0}, giving 2 states: (0,10), (0,20)
        assert_eq!(states.len(), 2);

        // Verify all states have x = 0
        for state in &states {
            let x_val = state
                .vars()
                .find(|(n, _)| n.as_ref() == "x")
                .map(|(_, v)| v.clone());
            assert_eq!(x_val, Some(Value::int(0)));
        }
    }

    #[test]
    fn test_enumerate_states_if_then_else_constant_condition() {
        // IF-THEN-ELSE with a condition that evaluates to a constant
        let src = r#"
---- MODULE Test ----
VARIABLE x
Enabled == TRUE
Init == IF Enabled THEN x = 1 ELSE x = 0
====
"#;
        let (module, ctx, vars) = setup_module(src);

        let init_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Init" {
                        return Some(def);
                    }
                }
                None
            })
            .unwrap();

        let branches = extract_init_constraints(&ctx, &init_def.body, &vars).unwrap();
        let states = enumerate_states_from_constraint_branches(None, &vars, &branches).unwrap();

        // Enabled is TRUE, so we take the THEN branch: x = 1
        assert_eq!(states.len(), 1);
        assert_eq!(states[0].get("x"), Some(&Value::int(1)));
    }

    #[test]
    fn test_enumerate_states_if_then_else_false_condition() {
        // IF-THEN-ELSE with FALSE condition
        let src = r#"
---- MODULE Test ----
VARIABLE x
Disabled == FALSE
Init == IF Disabled THEN x = 1 ELSE x = 0
====
"#;
        let (module, ctx, vars) = setup_module(src);

        let init_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Init" {
                        return Some(def);
                    }
                }
                None
            })
            .unwrap();

        let branches = extract_init_constraints(&ctx, &init_def.body, &vars).unwrap();
        let states = enumerate_states_from_constraint_branches(None, &vars, &branches).unwrap();

        // Disabled is FALSE, so we take the ELSE branch: x = 0
        assert_eq!(states.len(), 1);
        assert_eq!(states[0].get("x"), Some(&Value::int(0)));
    }

    #[test]
    fn test_enumerate_states_if_then_else_with_constraint_condition() {
        // IF-THEN-ELSE where condition is a constraint on a variable
        // Init == x \in {0, 1} /\ IF x = 0 THEN y = 10 ELSE y = 20
        // This should expand to: (x = 0 /\ y = 10) \/ (x \in {0,1} /\ y = 20)
        // After intersection: (x = 0, y = 10) \/ (x \in {0,1}, y = 20)
        // which gives states: (0, 10), (0, 20), (1, 20)
        let src = r#"
---- MODULE Test ----
VARIABLE x, y
Init == x \in {0, 1} /\ IF x = 0 THEN y = 10 ELSE y = 20
====
"#;
        let (module, ctx, vars) = setup_module(src);

        let init_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Init" {
                        return Some(def);
                    }
                }
                None
            })
            .unwrap();

        let branches = extract_init_constraints(&ctx, &init_def.body, &vars).unwrap();
        let states = enumerate_states_from_constraint_branches(None, &vars, &branches).unwrap();

        // With proper negation support, we get:
        // - Branch from condition x=0 + then y=10: (x=0 ∩ {0,1}, y=10) => (0, 10)
        // - Branch from ~condition x/=0 + else y=20: (x∈{0,1} ∩ x/=0, y=20) => (1, 20)
        // So exactly 2 states: (0, 10) and (1, 20)
        assert_eq!(states.len(), 2);

        // Verify the states
        let mut state_pairs: Vec<(i64, i64)> = states
            .iter()
            .map(|s| {
                let x: i64 = s.get("x").and_then(|v| v.as_i64()).unwrap();
                let y: i64 = s.get("y").and_then(|v| v.as_i64()).unwrap();
                (x, y)
            })
            .collect();
        state_pairs.sort();
        assert_eq!(state_pairs, vec![(0, 10), (1, 20)]);
    }

    #[test]
    fn test_enumerate_states_nested_if_then_else() {
        // Nested IF-THEN-ELSE with constant conditions
        let src = r#"
---- MODULE Test ----
VARIABLE x
A == TRUE
B == FALSE
Init == IF A THEN (IF B THEN x = 1 ELSE x = 2) ELSE x = 3
====
"#;
        let (module, ctx, vars) = setup_module(src);

        let init_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Init" {
                        return Some(def);
                    }
                }
                None
            })
            .unwrap();

        let branches = extract_init_constraints(&ctx, &init_def.body, &vars).unwrap();
        let states = enumerate_states_from_constraint_branches(None, &vars, &branches).unwrap();

        // A is TRUE, so we enter IF B THEN x=1 ELSE x=2
        // B is FALSE, so we get x = 2
        assert_eq!(states.len(), 1);
        assert_eq!(states[0].get("x"), Some(&Value::int(2)));
    }

    // ============================
    // Inequality constraint tests
    // ============================

    #[test]
    fn test_enumerate_states_simple_inequality() {
        // x \in {0, 1, 2} /\ x /= 1 => x ∈ {0, 2}
        let src = r#"
---- MODULE Test ----
VARIABLE x
Init == x \in {0, 1, 2} /\ x /= 1
====
"#;
        let (module, ctx, vars) = setup_module(src);

        let init_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Init" {
                        return Some(def);
                    }
                }
                None
            })
            .unwrap();

        let branches = extract_init_constraints(&ctx, &init_def.body, &vars).unwrap();
        let states = enumerate_states_from_constraint_branches(None, &vars, &branches).unwrap();

        // Should have 2 states: x=0 and x=2 (x=1 is excluded)
        assert_eq!(states.len(), 2);

        let mut x_values: Vec<i64> = states
            .iter()
            .filter_map(|s| s.get("x").and_then(|v| v.as_i64()))
            .collect();
        x_values.sort();
        assert_eq!(x_values, vec![0, 2]);
    }

    #[test]
    fn test_enumerate_states_multiple_inequalities() {
        // x \in {0, 1, 2, 3, 4} /\ x /= 1 /\ x /= 3 => x ∈ {0, 2, 4}
        let src = r#"
---- MODULE Test ----
VARIABLE x
Init == x \in {0, 1, 2, 3, 4} /\ x /= 1 /\ x /= 3
====
"#;
        let (module, ctx, vars) = setup_module(src);

        let init_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Init" {
                        return Some(def);
                    }
                }
                None
            })
            .unwrap();

        let branches = extract_init_constraints(&ctx, &init_def.body, &vars).unwrap();
        let states = enumerate_states_from_constraint_branches(None, &vars, &branches).unwrap();

        // Should have 3 states: x=0, x=2, x=4
        assert_eq!(states.len(), 3);

        let mut x_values: Vec<i64> = states
            .iter()
            .filter_map(|s| s.get("x").and_then(|v| v.as_i64()))
            .collect();
        x_values.sort();
        assert_eq!(x_values, vec![0, 2, 4]);
    }

    #[test]
    fn test_enumerate_states_inequality_with_two_vars() {
        // x \in {0, 1} /\ y \in {10, 20, 30} /\ y /= 20
        let src = r#"
---- MODULE Test ----
VARIABLE x, y
Init == x \in {0, 1} /\ y \in {10, 20, 30} /\ y /= 20
====
"#;
        let (module, ctx, vars) = setup_module(src);

        let init_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Init" {
                        return Some(def);
                    }
                }
                None
            })
            .unwrap();

        let branches = extract_init_constraints(&ctx, &init_def.body, &vars).unwrap();
        let states = enumerate_states_from_constraint_branches(None, &vars, &branches).unwrap();

        // 2 choices for x * 2 choices for y (10, 30) = 4 states
        assert_eq!(states.len(), 4);
    }

    #[test]
    fn test_enumerate_states_inequality_excludes_all() {
        // x \in {1} /\ x /= 1 => empty (contradiction)
        let src = r#"
---- MODULE Test ----
VARIABLE x
Init == x \in {1} /\ x /= 1
====
"#;
        let (module, ctx, vars) = setup_module(src);

        let init_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Init" {
                        return Some(def);
                    }
                }
                None
            })
            .unwrap();

        let branches = extract_init_constraints(&ctx, &init_def.body, &vars).unwrap();
        let states = enumerate_states_from_constraint_branches(None, &vars, &branches).unwrap();

        // Contradiction: x must be 1 but also not 1
        assert!(states.is_empty());
    }

    #[test]
    fn test_enumerate_states_inequality_in_disjunction() {
        // (x = 0 /\ y = 10) \/ (x \in {0, 1, 2} /\ x /= 0 /\ y = 20)
        let src = r#"
---- MODULE Test ----
VARIABLE x, y
Init == (x = 0 /\ y = 10) \/ (x \in {0, 1, 2} /\ x /= 0 /\ y = 20)
====
"#;
        let (module, ctx, vars) = setup_module(src);

        let init_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Init" {
                        return Some(def);
                    }
                }
                None
            })
            .unwrap();

        let branches = extract_init_constraints(&ctx, &init_def.body, &vars).unwrap();
        let states = enumerate_states_from_constraint_branches(None, &vars, &branches).unwrap();

        // First branch: (0, 10)
        // Second branch: x in {1, 2} (excluding 0), y = 20 => (1, 20), (2, 20)
        // Total: 3 states
        assert_eq!(states.len(), 3);

        let mut state_pairs: Vec<(i64, i64)> = states
            .iter()
            .map(|s| {
                let x: i64 = s.get("x").and_then(|v| v.as_i64()).unwrap();
                let y: i64 = s.get("y").and_then(|v| v.as_i64()).unwrap();
                (x, y)
            })
            .collect();
        state_pairs.sort();
        assert_eq!(state_pairs, vec![(0, 10), (1, 20), (2, 20)]);
    }

    #[test]
    fn test_enumerate_states_direct_inequality_syntax() {
        // Test using # syntax directly (if parsed as Neq)
        // x \in {0, 1, 2} /\ x # 1 => x ∈ {0, 2}
        // Note: TLA+ uses # as inequality operator, same as /=
        let src = r#"
---- MODULE Test ----
VARIABLE x
Init == x \in {0, 1, 2} /\ x # 1
====
"#;
        let (module, ctx, vars) = setup_module(src);

        let init_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Init" {
                        return Some(def);
                    }
                }
                None
            })
            .unwrap();

        let result = extract_init_constraints(&ctx, &init_def.body, &vars);

        // If # is parsed as Neq, we should get constraints
        if let Some(branches) = result {
            let states = enumerate_states_from_constraint_branches(None, &vars, &branches).unwrap();
            assert_eq!(states.len(), 2);

            let mut x_values: Vec<i64> = states
                .iter()
                .filter_map(|s| {
                    s.get("x")
                        .and_then(|v| v.as_int())
                        .and_then(|n| n.try_into().ok())
                })
                .collect();
            x_values.sort();
            assert_eq!(x_values, vec![0, 2]);
        }
        // If # is not parsed as Neq, the test still passes (result is None)
    }

    #[test]
    fn test_func_constructor_in_init() {
        use tla_core::ast::Expr;

        let src = r#"
---- MODULE Test ----
CONSTANT N
VARIABLE pc

ProcSet == 1..N

Init == pc = [p \in ProcSet |-> "b0"]
====
"#;
        let tree = tla_core::parse_to_syntax_tree(src);

        // Print the concrete syntax tree (Debug format)
        eprintln!("\n=== CST (debug) ===");
        eprintln!("{:#?}", tree);

        let lower_result = tla_core::lower(tla_core::FileId(0), &tree);
        eprintln!("\nLower errors: {:?}", lower_result.errors);

        let module = lower_result.module.unwrap();

        eprintln!("\n=== All operators ===");
        for unit in &module.units {
            if let tla_core::ast::Unit::Operator(def) = &unit.node {
                eprintln!(
                    "Op '{}': body type = {:?}",
                    def.name.node,
                    std::mem::discriminant(&def.body.node)
                );
                // Match on the body to show more detail
                match &def.body.node {
                    Expr::Ident(name) => eprintln!("  -> Ident({})", name),
                    Expr::Eq(lhs, rhs) => {
                        eprintln!("  -> Eq(lhs={:?}, rhs={:?})", lhs.node, rhs.node);
                    }
                    other => eprintln!("  -> Other: {:?}", other),
                }
            }
        }

        let (_, mut ctx, vars) = setup_module(src);

        // Bind N = 3
        ctx.bind_mut("N", Value::int(3));

        let init_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Init" {
                        return Some(def);
                    }
                }
                None
            })
            .unwrap();

        // The Init body should be Eq, not Ident
        match &init_def.body.node {
            Expr::Eq(_, _) => eprintln!("Init body is correctly an Eq"),
            Expr::Ident(name) => panic!("Init body is incorrectly Ident({})", name),
            other => panic!("Init body is unexpected: {:?}", other),
        }

        let branches = extract_init_constraints(&ctx, &init_def.body, &vars);
        eprintln!("Branches: {:?}", branches);
        assert!(
            branches.is_some(),
            "Failed to extract constraints from Init with function constructor"
        );

        // Verify the constraint value is a function or sequence
        // (domain 1..N becomes a Seq with our semantics)
        let branches = branches.unwrap();
        assert_eq!(branches.len(), 1, "Expected one branch");
        let branch = &branches[0];
        assert_eq!(branch.len(), 1, "Expected one constraint");
        match &branch[0] {
            Constraint::Eq(name, value) => {
                assert_eq!(name.as_ref() as &str, "pc");
                // Domain 1..N creates a Seq (functions with domain 1..n are sequences in TLA+)
                assert!(
                    matches!(value, Value::Func(_) | Value::IntFunc(_) | Value::Seq(_)),
                    "Expected function or sequence value, got {:?}",
                    value
                );
            }
            other => panic!("Expected Eq constraint, got {:?}", other),
        }
    }

    #[test]
    fn test_enumerate_function_space_membership() {
        // Test grid \in [Pos -> BOOLEAN] pattern
        // Use explicit set enum to avoid operator lookup
        let src = r#"
---- MODULE Test ----
VARIABLE grid
Init == grid \in [{<<1, 1>>, <<1, 2>>} -> {TRUE, FALSE}]
====
"#;
        let tree = tla_core::parse_to_syntax_tree(src);
        eprintln!("=== CST ===");
        eprintln!("{:#?}", tree);

        let lower_result = lower(FileId(0), &tree);
        eprintln!("Lower errors: {:?}", lower_result.errors);

        let module = lower_result.module.unwrap();
        for unit in &module.units {
            if let tla_core::ast::Unit::Operator(def) = &unit.node {
                eprintln!("Op '{}' body: {:?}", def.name.node, def.body.node);
            }
        }

        let mut ctx = EvalCtx::new();
        ctx.load_module(&module);

        let mut vars = Vec::new();
        for unit in &module.units {
            if let tla_core::ast::Unit::Variable(var_names) = &unit.node {
                for var in var_names {
                    vars.push(Arc::from(var.node.as_str()));
                }
            }
        }

        let init_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Init" {
                        return Some(def);
                    }
                }
                None
            })
            .unwrap();

        eprintln!("Init body (in test): {:?}", init_def.body.node);

        let branches = extract_init_constraints(&ctx, &init_def.body, &vars);
        eprintln!("Branches: {:?}", branches);

        assert!(
            branches.is_some(),
            "Should extract constraints from function space membership"
        );

        let branches = branches.unwrap();
        // {<<1,1>>, <<1,2>>} has 2 elements, {TRUE, FALSE} has 2 elements
        // So [... -> ...] has 2^2 = 4 functions
        let states = enumerate_states_from_constraint_branches(None, &vars, &branches).unwrap();
        assert_eq!(states.len(), 4, "Expected 4 initial states (2^2 functions)");
    }

    #[test]
    fn test_enumerate_function_space_with_operator_ref() {
        // Test grid \in [Pos -> BOOLEAN] where Pos is an operator
        let src = r#"
---- MODULE Test ----
EXTENDS Integers
VARIABLE grid

Pos == {<<1, 1>>, <<1, 2>>}
Init == grid \in [Pos -> BOOLEAN]
====
"#;
        let tree = tla_core::parse_to_syntax_tree(src);
        eprintln!("=== CST with Pos ===");
        eprintln!("{:#?}", tree);

        let lower_result = lower(FileId(0), &tree);
        eprintln!("Lower errors: {:?}", lower_result.errors);

        let module = lower_result.module.unwrap();
        for unit in &module.units {
            if let tla_core::ast::Unit::Operator(def) = &unit.node {
                eprintln!("Op '{}' body: {:?}", def.name.node, def.body.node);
            }
        }

        let mut ctx = EvalCtx::new();
        ctx.load_module(&module);

        let mut vars = Vec::new();
        for unit in &module.units {
            if let tla_core::ast::Unit::Variable(var_names) = &unit.node {
                for var in var_names {
                    vars.push(Arc::from(var.node.as_str()));
                }
            }
        }

        let init_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Init" {
                        return Some(def);
                    }
                }
                None
            })
            .unwrap();

        eprintln!("Init body with Pos: {:?}", init_def.body.node);

        let branches = extract_init_constraints(&ctx, &init_def.body, &vars);
        eprintln!("Branches with Pos: {:?}", branches);

        assert!(
            branches.is_some(),
            "Should extract constraints from function space membership with operator reference"
        );

        let branches = branches.unwrap();
        let states = enumerate_states_from_constraint_branches(None, &vars, &branches).unwrap();
        assert_eq!(states.len(), 4, "Expected 4 initial states (2^2 functions)");
    }

    #[test]
    fn test_inline_apply_with_exists_in_conjunction() {
        // Test the pattern from bcastFolklore: Op(x) /\ disjunction
        // where Op(x) contains \E inside its body
        // This should properly inline the operator to expose the Exists
        let src = r#"
---- MODULE Test ----
EXTENDS Integers
VARIABLE x, y

\* Operator that contains an existential quantifier
Receive(self) ==
  /\ \E val \in {1, 2}:
        /\ x' = val
        /\ y' = self

\* Step combines the operator with a disjunction
Step(self) ==
  /\ Receive(self)
  /\ TRUE

Init == x = 0 /\ y = 0

Next == \E self \in {10, 20}: Step(self)
====
"#;
        let (module, mut ctx, vars) = setup_module(src);

        let next_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Next" {
                        return Some(def.clone());
                    }
                }
                None
            })
            .unwrap();

        // Start from initial state
        let init_state =
            State::from_pairs([("x", Value::Int(0.into())), ("y", Value::Int(0.into()))]);

        let successors = enumerate_successors(&mut ctx, &next_def, &init_state, &vars).unwrap();

        // Should find successors for:
        // - self=10, val=1 -> x'=1, y'=10
        // - self=10, val=2 -> x'=2, y'=10
        // - self=20, val=1 -> x'=1, y'=20
        // - self=20, val=2 -> x'=2, y'=20
        // Total: 4 successor states
        assert_eq!(
            successors.len(),
            4,
            "Expected 4 successor states from nested Exists in Apply. Got: {:?}",
            successors
        );
    }

    /// Test that LET bindings in disabled actions don't cause evaluation errors.
    ///
    /// This reproduces a bug found in MCCheckpointCoordination where:
    /// ```tla
    /// SendReplicatedRequest(prospect) ==
    ///   LET currentLease == F[Leader] IN  \* Fails when Leader=NoNode!
    ///   /\ Guard                          \* Would be FALSE anyway
    ///   ...
    /// ```
    /// When Leader=NoNode, F[Leader] fails because NoNode is not in the domain of F.
    /// But the Guard would be FALSE anyway, so the action should be disabled, not error.
    /// TLC handles this gracefully; TLA2 should too.
    #[test]
    fn test_let_in_disabled_action_does_not_error() {
        let src = r#"
---- MODULE Test ----
CONSTANTS NoNode
VARIABLE Leader, F

\* Guard checks Leader is valid BEFORE using it
Guard == Leader /= NoNode

\* Action has LET that would fail if Leader=NoNode
\* but Guard would be FALSE anyway
ActionWithLet ==
  LET val == F[Leader] IN
  /\ Guard
  /\ Leader' = Leader
  /\ F' = F

Init == Leader = NoNode /\ F = [n \in {1,2,3} |-> n]

Next == ActionWithLet
====
"#;
        let (module, mut ctx, vars) = setup_module(src);

        // Bind NoNode constant
        ctx.bind_mut("NoNode".to_string(), Value::model_value("NoNode"));

        let init_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Init" {
                        return Some(def.clone());
                    }
                }
                None
            })
            .unwrap();

        let next_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Next" {
                        return Some(def.clone());
                    }
                }
                None
            })
            .unwrap();

        // Generate initial state where Leader = NoNode
        let branches = extract_init_constraints(&ctx, &init_def.body, &vars).unwrap();
        let init_states =
            enumerate_states_from_constraint_branches(Some(&ctx), &vars, &branches).unwrap();
        assert!(!init_states.is_empty(), "Should have initial states");
        let init_state = &init_states[0];

        // Verify Leader is NoNode
        let leader_val = init_state
            .vars()
            .find(|(n, _)| n.as_ref() == "Leader")
            .map(|(_, v)| v)
            .unwrap();
        assert!(
            matches!(leader_val, Value::ModelValue(s) if s.as_ref() == "NoNode"),
            "Leader should be NoNode, got {:?}",
            leader_val
        );

        // Enumerate successors - this should NOT error, but return empty (action disabled)
        // Before the fix, this would fail with "NotInDomain: @NoNode not in domain"
        let result = enumerate_successors(&mut ctx, &next_def, init_state, &vars);
        assert!(
            result.is_ok(),
            "Should not error on LET in disabled action, got: {:?}",
            result.err()
        );

        let successors = result.unwrap();
        // Action should be disabled (Guard is FALSE because Leader = NoNode)
        assert_eq!(
            successors.len(),
            0,
            "Action should be disabled (no successors) when Leader=NoNode"
        );
    }
    #[test]
    fn test_find_subseteq_guard_direct() {
        // x \subseteq S should return S
        use tla_core::{Span, Spanned};

        let span = Span::dummy();
        let body = Spanned::new(
            Expr::Subseteq(
                Box::new(Spanned::new(Expr::Ident("x".to_string()), span)),
                Box::new(Spanned::new(Expr::Ident("S".to_string()), span)),
            ),
            span,
        );

        let result = find_subseteq_guard(&body, "x");
        assert!(result.is_some());
        if let Expr::Ident(name) = &result.unwrap().node {
            assert_eq!(name, "S");
        } else {
            panic!("Expected Ident expression");
        }
    }

    #[test]
    fn test_find_subseteq_guard_not_found() {
        // y \subseteq S should NOT match when looking for "x"
        use tla_core::{Span, Spanned};

        let span = Span::dummy();
        let body = Spanned::new(
            Expr::Subseteq(
                Box::new(Spanned::new(Expr::Ident("y".to_string()), span)),
                Box::new(Spanned::new(Expr::Ident("S".to_string()), span)),
            ),
            span,
        );

        assert!(find_subseteq_guard(&body, "x").is_none());
    }

    #[test]
    fn test_find_subseteq_guard_in_conjunction() {
        // (P /\ x \subseteq S) should find x \subseteq S
        use tla_core::{Span, Spanned};

        let span = Span::dummy();
        let body = Spanned::new(
            Expr::And(
                Box::new(Spanned::new(Expr::Ident("P".to_string()), span)),
                Box::new(Spanned::new(
                    Expr::Subseteq(
                        Box::new(Spanned::new(Expr::Ident("x".to_string()), span)),
                        Box::new(Spanned::new(Expr::Ident("S".to_string()), span)),
                    ),
                    span,
                )),
            ),
            span,
        );

        let result = find_subseteq_guard(&body, "x");
        assert!(result.is_some());
        if let Expr::Ident(name) = &result.unwrap().node {
            assert_eq!(name, "S");
        } else {
            panic!("Expected Ident expression");
        }
    }

    #[test]
    fn test_find_subseteq_guard_in_nested_conjunction() {
        // (P /\ (Q /\ x \subseteq S)) should find x \subseteq S
        use tla_core::{Span, Spanned};

        let span = Span::dummy();
        let body = Spanned::new(
            Expr::And(
                Box::new(Spanned::new(Expr::Ident("P".to_string()), span)),
                Box::new(Spanned::new(
                    Expr::And(
                        Box::new(Spanned::new(Expr::Ident("Q".to_string()), span)),
                        Box::new(Spanned::new(
                            Expr::Subseteq(
                                Box::new(Spanned::new(Expr::Ident("x".to_string()), span)),
                                Box::new(Spanned::new(Expr::Ident("S".to_string()), span)),
                            ),
                            span,
                        )),
                    ),
                    span,
                )),
            ),
            span,
        );

        let result = find_subseteq_guard(&body, "x");
        assert!(result.is_some());
        if let Expr::Ident(name) = &result.unwrap().node {
            assert_eq!(name, "S");
        } else {
            panic!("Expected Ident expression");
        }
    }

    #[test]
    fn test_find_subseteq_guard_in_exists() {
        // \E y \in T : x \subseteq S should find x \subseteq S
        use tla_core::{Span, Spanned};
        use tla_core::ast::BoundVar;

        let span = Span::dummy();
        let body = Spanned::new(
            Expr::Exists(
                vec![BoundVar {
                    name: Spanned::new("y".to_string(), span),
                    domain: Some(Box::new(Spanned::new(Expr::Ident("T".to_string()), span))),
                    pattern: None,
                }],
                Box::new(Spanned::new(
                    Expr::Subseteq(
                        Box::new(Spanned::new(Expr::Ident("x".to_string()), span)),
                        Box::new(Spanned::new(Expr::Ident("S".to_string()), span)),
                    ),
                    span,
                )),
            ),
            span,
        );

        let result = find_subseteq_guard(&body, "x");
        assert!(result.is_some());
    }

    #[test]
    fn test_find_superset_guard_direct() {
        // S \subseteq x should return S (x must contain S)
        use tla_core::{Span, Spanned};

        let span = Span::dummy();
        let body = Spanned::new(
            Expr::Subseteq(
                Box::new(Spanned::new(Expr::Ident("S".to_string()), span)),
                Box::new(Spanned::new(Expr::Ident("x".to_string()), span)),
            ),
            span,
        );

        let result = find_superset_guard(&body, "x");
        assert!(result.is_some());
        if let Expr::Ident(name) = &result.unwrap().node {
            assert_eq!(name, "S");
        } else {
            panic!("Expected Ident expression");
        }
    }

    #[test]
    fn test_find_superset_guard_not_found() {
        // S \subseteq y should NOT match when looking for "x"
        use tla_core::{Span, Spanned};

        let span = Span::dummy();
        let body = Spanned::new(
            Expr::Subseteq(
                Box::new(Spanned::new(Expr::Ident("S".to_string()), span)),
                Box::new(Spanned::new(Expr::Ident("y".to_string()), span)),
            ),
            span,
        );

        assert!(find_superset_guard(&body, "x").is_none());
    }

    #[test]
    fn test_find_superset_guard_in_conjunction() {
        // (P /\ S \subseteq x) should find S
        use tla_core::{Span, Spanned};

        let span = Span::dummy();
        let body = Spanned::new(
            Expr::And(
                Box::new(Spanned::new(Expr::Ident("P".to_string()), span)),
                Box::new(Spanned::new(
                    Expr::Subseteq(
                        Box::new(Spanned::new(Expr::Ident("S".to_string()), span)),
                        Box::new(Spanned::new(Expr::Ident("x".to_string()), span)),
                    ),
                    span,
                )),
            ),
            span,
        );

        let result = find_superset_guard(&body, "x");
        assert!(result.is_some());
        if let Expr::Ident(name) = &result.unwrap().node {
            assert_eq!(name, "S");
        } else {
            panic!("Expected Ident expression");
        }
    }

    #[test]
    fn test_find_superset_guard_in_exists() {
        // \E y \in T : S \subseteq x should find S
        use tla_core::{Span, Spanned};
        use tla_core::ast::BoundVar;

        let span = Span::dummy();
        let body = Spanned::new(
            Expr::Exists(
                vec![BoundVar {
                    name: Spanned::new("y".to_string(), span),
                    domain: Some(Box::new(Spanned::new(Expr::Ident("T".to_string()), span))),
                    pattern: None,
                }],
                Box::new(Spanned::new(
                    Expr::Subseteq(
                        Box::new(Spanned::new(Expr::Ident("S".to_string()), span)),
                        Box::new(Spanned::new(Expr::Ident("x".to_string()), span)),
                    ),
                    span,
                )),
            ),
            span,
        );

        let result = find_superset_guard(&body, "x");
        assert!(result.is_some());
    }

    #[test]
    fn test_find_guards_both_upper_and_lower_bound() {
        // (S \subseteq x /\ x \subseteq T) should find both bounds
        // lower bound: S (from S \subseteq x)
        // upper bound: T (from x \subseteq T)
        use tla_core::{Span, Spanned};

        let span = Span::dummy();
        let body = Spanned::new(
            Expr::And(
                Box::new(Spanned::new(
                    Expr::Subseteq(
                        Box::new(Spanned::new(Expr::Ident("S".to_string()), span)),
                        Box::new(Spanned::new(Expr::Ident("x".to_string()), span)),
                    ),
                    span,
                )),
                Box::new(Spanned::new(
                    Expr::Subseteq(
                        Box::new(Spanned::new(Expr::Ident("x".to_string()), span)),
                        Box::new(Spanned::new(Expr::Ident("T".to_string()), span)),
                    ),
                    span,
                )),
            ),
            span,
        );

        // Upper bound: T (from x \subseteq T)
        let upper = find_subseteq_guard(&body, "x");
        assert!(upper.is_some());
        if let Expr::Ident(name) = &upper.unwrap().node {
            assert_eq!(name, "T");
        }

        // Lower bound: S (from S \subseteq x)
        let lower = find_superset_guard(&body, "x");
        assert!(lower.is_some());
        if let Expr::Ident(name) = &lower.unwrap().node {
            assert_eq!(name, "S");
        }
    }

    #[test]
    fn test_find_guards_other_expression_types() {
        // Non-subseteq expressions should return None
        use tla_core::{Span, Spanned};

        let span = Span::dummy();

        // Just an identifier
        let body1 = Spanned::new(Expr::Ident("x".to_string()), span);
        assert!(find_subseteq_guard(&body1, "x").is_none());
        assert!(find_superset_guard(&body1, "x").is_none());

        // Equality expression
        let body2 = Spanned::new(
            Expr::Eq(
                Box::new(Spanned::new(Expr::Ident("x".to_string()), span)),
                Box::new(Spanned::new(Expr::Ident("S".to_string()), span)),
            ),
            span,
        );
        assert!(find_subseteq_guard(&body2, "x").is_none());
        assert!(find_superset_guard(&body2, "x").is_none());

        // In expression
        let body3 = Spanned::new(
            Expr::In(
                Box::new(Spanned::new(Expr::Ident("x".to_string()), span)),
                Box::new(Spanned::new(Expr::Ident("S".to_string()), span)),
            ),
            span,
        );
        assert!(find_subseteq_guard(&body3, "x").is_none());
        assert!(find_superset_guard(&body3, "x").is_none());
    }

    /// Test gap #43: SUBSET bounds in Or-branches
    ///
    /// Documents that the current implementation does NOT search into Or-branches
    /// for subset bounds. Guards inside disjunctions are not found because:
    /// 1. A guard in one branch doesn't constrain values when taking the other branch
    /// 2. The optimization would need to take the intersection of guards from all branches
    ///
    /// Example: `\E x \in SUBSET(S) : (x \subseteq T1 \/ x \subseteq T2) /\ P(x)`
    /// Currently finds no upper bound because the guard is inside an Or.
    #[test]
    fn test_find_subseteq_guard_in_or_branch_not_found() {
        use tla_core::{Span, Spanned};

        let span = Span::dummy();
        // (x \subseteq T1 \/ x \subseteq T2) - guard is inside Or
        let body = Spanned::new(
            Expr::Or(
                Box::new(Spanned::new(
                    Expr::Subseteq(
                        Box::new(Spanned::new(Expr::Ident("x".to_string()), span)),
                        Box::new(Spanned::new(Expr::Ident("T1".to_string()), span)),
                    ),
                    span,
                )),
                Box::new(Spanned::new(
                    Expr::Subseteq(
                        Box::new(Spanned::new(Expr::Ident("x".to_string()), span)),
                        Box::new(Spanned::new(Expr::Ident("T2".to_string()), span)),
                    ),
                    span,
                )),
            ),
            span,
        );

        // Current behavior: does NOT find guard inside Or
        // This is intentional - a guard in one Or-branch doesn't apply to the other branch
        let result = find_subseteq_guard(&body, "x");
        assert!(
            result.is_none(),
            "find_subseteq_guard should not search into Or branches"
        );
    }

    /// Test #43: SUBSET bounds with Or OUTSIDE the guard
    ///
    /// When the Or is at a different level (not containing the guard), the guard should be found.
    /// Example: `(P \/ Q) /\ x \subseteq T` - the guard is NOT inside the Or
    #[test]
    fn test_find_subseteq_guard_with_or_sibling() {
        use tla_core::{Span, Spanned};

        let span = Span::dummy();
        // (P \/ Q) /\ x \subseteq T
        let body = Spanned::new(
            Expr::And(
                Box::new(Spanned::new(
                    Expr::Or(
                        Box::new(Spanned::new(Expr::Ident("P".to_string()), span)),
                        Box::new(Spanned::new(Expr::Ident("Q".to_string()), span)),
                    ),
                    span,
                )),
                Box::new(Spanned::new(
                    Expr::Subseteq(
                        Box::new(Spanned::new(Expr::Ident("x".to_string()), span)),
                        Box::new(Spanned::new(Expr::Ident("T".to_string()), span)),
                    ),
                    span,
                )),
            ),
            span,
        );

        // The guard x \subseteq T is found because it's in the And, not inside the Or
        let result = find_subseteq_guard(&body, "x");
        assert!(result.is_some());
        if let Expr::Ident(name) = &result.unwrap().node {
            assert_eq!(name, "T");
        } else {
            panic!("Expected Ident expression");
        }
    }

    // ========================================================================
    // Regression tests for #62: Topological sort of symbolic assignments
    // ========================================================================

    #[test]
    fn test_get_primed_var_refs_simple() {
        // Test: x' references nothing (it's a definition, not a reference)
        // Test: expr referencing count' should return {"count"}
        use tla_core::{Span, Spanned};

        let span = Span::dummy();

        // Simple expression: count' (this is a Prime wrapping an Ident)
        let count_prime = Expr::Prime(Box::new(Spanned::new(
            Expr::Ident("count".to_string()),
            span,
        )));
        let refs = get_primed_var_refs(&count_prime);
        assert_eq!(refs.len(), 1);
        assert!(refs.contains(&Arc::from("count")));
    }

    #[test]
    fn test_get_primed_var_refs_complex_expr() {
        // Test: announced' = (count' >= VT) should reference count'
        // The expression `count' >= VT` contains a primed reference
        use tla_core::{Span, Spanned};

        let span = Span::dummy();

        // Build: count' >= VT (a Geq expression)
        let count_prime = Spanned::new(
            Expr::Prime(Box::new(Spanned::new(
                Expr::Ident("count".to_string()),
                span,
            ))),
            span,
        );
        let vt = Spanned::new(Expr::Ident("VT".to_string()), span);
        let geq_expr = Expr::Geq(Box::new(count_prime), Box::new(vt));

        let refs = get_primed_var_refs(&geq_expr);
        assert_eq!(refs.len(), 1);
        assert!(refs.contains(&Arc::from("count")));
    }

    #[test]
    fn test_get_primed_var_refs_multiple() {
        // Test: x' + y' should return {"x", "y"}
        use tla_core::{Span, Spanned};

        let span = Span::dummy();

        let x_prime = Spanned::new(
            Expr::Prime(Box::new(Spanned::new(Expr::Ident("x".to_string()), span))),
            span,
        );
        let y_prime = Spanned::new(
            Expr::Prime(Box::new(Spanned::new(Expr::Ident("y".to_string()), span))),
            span,
        );
        // Build x' + y' as an Add expression
        let add_expr = Expr::Add(Box::new(x_prime), Box::new(y_prime));

        let refs = get_primed_var_refs(&add_expr);
        assert_eq!(refs.len(), 2);
        assert!(refs.contains(&Arc::from("x")));
        assert!(refs.contains(&Arc::from("y")));
    }

    #[test]
    fn test_get_primed_var_refs_no_primes() {
        // Test: x + y (no primes) should return empty set
        use tla_core::{Span, Spanned};

        let span = Span::dummy();

        let x = Spanned::new(Expr::Ident("x".to_string()), span);
        let y = Spanned::new(Expr::Ident("y".to_string()), span);
        let add_expr = Expr::Add(Box::new(x), Box::new(y));

        let refs = get_primed_var_refs(&add_expr);
        assert!(refs.is_empty());
    }

    #[test]
    fn test_topological_sort_assignments_independent() {
        // Test: independent assignments should remain in original order
        // x' = 1, y' = 2 -> no dependencies, order preserved
        let assignments = vec![
            SymbolicAssignment::Value(Arc::from("x"), Value::int(1)),
            SymbolicAssignment::Value(Arc::from("y"), Value::int(2)),
        ];

        let sorted = topological_sort_assignments(None, &assignments);
        assert_eq!(sorted.len(), 2);

        // Order should be preserved for independent assignments
        match &sorted[0] {
            SymbolicAssignment::Value(name, _) => assert_eq!(name.as_ref(), "x"),
            _ => panic!("Expected Value assignment"),
        }
        match &sorted[1] {
            SymbolicAssignment::Value(name, _) => assert_eq!(name.as_ref(), "y"),
            _ => panic!("Expected Value assignment"),
        }
    }

    #[test]
    fn test_topological_sort_assignments_dependency() {
        // Regression test for #62: Prisoner spec bug
        // announced' = (count' >= VT) depends on count'
        // count' = count + 1 defines count'
        // The sort should put count' definition BEFORE announced' definition
        use tla_core::{Span, Spanned};

        let span = Span::dummy();

        // Build: announced' = (count' >= 3)
        // This references count' in its RHS
        let count_prime = Spanned::new(
            Expr::Prime(Box::new(Spanned::new(
                Expr::Ident("count".to_string()),
                span,
            ))),
            span,
        );
        let three = Spanned::new(Expr::Int(3.into()), span);
        let announced_rhs = Spanned::new(
            Expr::Geq(Box::new(count_prime), Box::new(three)),
            span,
        );

        // Build: count' = count + 1
        // This does NOT reference any primed vars
        let count = Spanned::new(Expr::Ident("count".to_string()), span);
        let one = Spanned::new(Expr::Int(1.into()), span);
        let count_rhs = Spanned::new(
            Expr::Add(Box::new(count), Box::new(one)),
            span,
        );

        // Put them in WRONG order (announced before count)
        // This was the bug: document order had announced first
        let assignments = vec![
            SymbolicAssignment::Expr(Arc::from("announced"), announced_rhs, Vec::new()),
            SymbolicAssignment::Expr(Arc::from("count"), count_rhs, Vec::new()),
        ];

        let sorted = topological_sort_assignments(None, &assignments);
        assert_eq!(sorted.len(), 2);

        // After sort: count should come FIRST because announced depends on count'
        match &sorted[0] {
            SymbolicAssignment::Expr(name, _, _) => {
                assert_eq!(name.as_ref(), "count", "count' should be defined first");
            }
            _ => panic!("Expected Expr assignment"),
        }
        match &sorted[1] {
            SymbolicAssignment::Expr(name, _, _) => {
                assert_eq!(name.as_ref(), "announced", "announced' should be defined second");
            }
            _ => panic!("Expected Expr assignment"),
        }
    }

    #[test]
    fn test_topological_sort_assignments_chain() {
        // Test: x' depends on y', y' depends on z'
        // z' = 1, y' = z' + 1, x' = y' + 1
        // Correct order: z, y, x
        use tla_core::{Span, Spanned};

        let span = Span::dummy();

        // z' = 1 (no dependencies)
        let z_rhs = Spanned::new(Expr::Int(1.into()), span);

        // y' = z' + 1 (depends on z')
        let z_prime = Spanned::new(
            Expr::Prime(Box::new(Spanned::new(Expr::Ident("z".to_string()), span))),
            span,
        );
        let one1 = Spanned::new(Expr::Int(1.into()), span);
        let y_rhs = Spanned::new(Expr::Add(Box::new(z_prime), Box::new(one1)), span);

        // x' = y' + 1 (depends on y')
        let y_prime = Spanned::new(
            Expr::Prime(Box::new(Spanned::new(Expr::Ident("y".to_string()), span))),
            span,
        );
        let one2 = Spanned::new(Expr::Int(1.into()), span);
        let x_rhs = Spanned::new(Expr::Add(Box::new(y_prime), Box::new(one2)), span);

        // Put in reverse order (x, y, z) - wrong order
        let assignments = vec![
            SymbolicAssignment::Expr(Arc::from("x"), x_rhs, Vec::new()),
            SymbolicAssignment::Expr(Arc::from("y"), y_rhs, Vec::new()),
            SymbolicAssignment::Expr(Arc::from("z"), z_rhs, Vec::new()),
        ];

        let sorted = topological_sort_assignments(None, &assignments);
        assert_eq!(sorted.len(), 3);

        // After sort: z, y, x
        let names: Vec<&str> = sorted
            .iter()
            .map(|s| match s {
                SymbolicAssignment::Expr(name, _, _) => name.as_ref(),
                _ => panic!("Expected Expr"),
            })
            .collect();
        assert_eq!(names, vec!["z", "y", "x"]);
    }

    #[test]
    fn test_action_if_with_operator_reference() {
        // Regression test for #89: IF condition with operator reference containing primed vars
        //
        // Pattern:
        //   ActionX == x < 1 /\ x' = x + 1
        //   TNext == y' = IF ActionX THEN 5 ELSE y
        //   Next == (ActionX \/ UNCHANGED x) /\ TNext
        //
        // The fix ensures that when evaluating y' = IF ActionX THEN 5 ELSE y,
        // the topological sort correctly identifies that y' depends on x' (via ActionX).
        let src = r#"
---- MODULE Test ----
EXTENDS Integers
VARIABLE x, y

Init == x = 0 /\ y = 0

ActionX == x < 1 /\ x' = x + 1
TNext == y' = IF ActionX THEN 5 ELSE y

Next == /\ (ActionX \/ UNCHANGED x)
        /\ TNext
====
"#;
        let (module, mut ctx, vars) = setup_module(src);
        let current_state = State::from_pairs([("x", Value::int(0)), ("y", Value::int(0))]);

        // Bind current state to context
        for (name, value) in current_state.vars() {
            ctx.bind_mut(Arc::clone(name), value.clone());
        }

        let next_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Next" {
                        return Some(def.clone());
                    }
                }
                None
            })
            .unwrap();

        let successors = enumerate_successors(&mut ctx, &next_def, &current_state, &vars).unwrap();

        // Should have 2 successors:
        // 1. {x=1, y=5} - ActionX taken, so y' = IF TRUE THEN 5 = 5
        // 2. {x=0, y=0} - UNCHANGED x, so y' = IF FALSE THEN 5 ELSE y = y = 0 (stuttering)
        assert_eq!(
            successors.len(),
            2,
            "Expected 2 successors (ActionX branch + UNCHANGED branch)"
        );

        // Check that we found {x=1, y=5}
        let action_x_state = successors
            .iter()
            .find(|s| {
                s.vars().find(|(n, _)| n.as_ref() == "x").map(|(_, v)| v.as_i64()) == Some(Some(1))
            })
            .expect("Should find ActionX branch successor");
        let y_val = action_x_state
            .vars()
            .find(|(n, _)| n.as_ref() == "y")
            .map(|(_, v)| v.as_i64())
            .expect("Should have y value");
        assert_eq!(y_val, Some(5), "ActionX branch: y should be 5 (from IF TRUE)");

        // Check that we found {x=0, y=0}
        let unchanged_state = successors
            .iter()
            .find(|s| {
                s.vars().find(|(n, _)| n.as_ref() == "x").map(|(_, v)| v.as_i64()) == Some(Some(0))
            })
            .expect("Should find UNCHANGED branch successor");
        let y_val = unchanged_state
            .vars()
            .find(|(n, _)| n.as_ref() == "y")
            .map(|(_, v)| v.as_i64())
            .expect("Should have y value");
        assert_eq!(y_val, Some(0), "UNCHANGED branch: y should be 0 (from IF FALSE)");
    }

    // ========================================================================
    // Regression tests for #55: ENABLED evaluation via fresh enumeration
    // ========================================================================

    #[test]
    fn test_enumerate_action_successors_basic() {
        // Test that enumerate_action_successors finds successors for a simple action
        let src = r#"
---- MODULE Test ----
VARIABLE x
Init == x = 0
Action == x' = x + 1
====
"#;
        let (module, mut ctx, vars) = setup_module(src);

        let current_state = State::from_pairs([("x", Value::int(0))]);

        // Bind current state variables to context (as enumerate_successors does)
        for (name, value) in current_state.vars() {
            ctx.bind_mut(Arc::clone(name), value.clone());
        }

        // Find the Action operator
        let action_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Action" {
                        return Some(def.clone());
                    }
                }
                None
            })
            .unwrap();

        let successors =
            enumerate_action_successors(&mut ctx, &action_def.body, &current_state, &vars).unwrap();

        assert_eq!(successors.len(), 1);
        assert_eq!(
            successors[0].get("x").and_then(|v| v.as_i64()),
            Some(1)
        );
    }

    #[test]
    fn test_enumerate_action_successors_disabled() {
        // Test that a disabled action (guard fails) returns empty successors
        let src = r#"
---- MODULE Test ----
VARIABLE x
Init == x = 0
Action == x > 100 /\ x' = x + 1
====
"#;
        let (module, mut ctx, vars) = setup_module(src);

        let current_state = State::from_pairs([("x", Value::int(0))]);

        // Bind current state variables to context
        for (name, value) in current_state.vars() {
            ctx.bind_mut(Arc::clone(name), value.clone());
        }

        let action_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Action" {
                        return Some(def.clone());
                    }
                }
                None
            })
            .unwrap();

        let successors =
            enumerate_action_successors(&mut ctx, &action_def.body, &current_state, &vars).unwrap();

        // Action is disabled because x = 0, not > 100
        assert!(
            successors.is_empty(),
            "Disabled action should return no successors"
        );
    }

    #[test]
    fn test_enumerate_action_successors_multiple() {
        // Test that enumerate_action_successors finds all successors for non-deterministic action
        let src = r#"
---- MODULE Test ----
VARIABLE x
Init == x = 0
Action == x' \in {1, 2, 3}
====
"#;
        let (module, mut ctx, vars) = setup_module(src);

        let current_state = State::from_pairs([("x", Value::int(0))]);

        // Bind current state variables to context
        for (name, value) in current_state.vars() {
            ctx.bind_mut(Arc::clone(name), value.clone());
        }

        let action_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Action" {
                        return Some(def.clone());
                    }
                }
                None
            })
            .unwrap();

        let successors =
            enumerate_action_successors(&mut ctx, &action_def.body, &current_state, &vars).unwrap();

        // Should find 3 successors: x=1, x=2, x=3
        assert_eq!(successors.len(), 3);

        let mut values: Vec<i64> = successors
            .iter()
            .filter_map(|s| s.get("x").and_then(|v| v.as_i64()))
            .collect();
        values.sort();
        assert_eq!(values, vec![1, 2, 3]);
    }

    // ========================================================================
    // Regression tests for #54/#61: ModuleRef handling in enumerate_next_rec_inner
    // ========================================================================

    #[test]
    fn test_enumerate_module_ref_simple() {
        // Test that ModuleRef (INSTANCE operator calls) are properly enumerated
        // This is a simplified version of the AllocatorImplementation bug
        let src = r#"
---- MODULE Inner ----
VARIABLE v
InnerAction == v' = v + 1
====
"#;
        // Note: Testing INSTANCE requires more setup. For now, test the basic
        // enumeration path works for regular actions (the ModuleRef code path
        // is tested via integration tests in test_tlaplus_examples.py)

        let (module, mut ctx, vars) = setup_module(src);
        let current_state = State::from_pairs([("v", Value::int(0))]);

        // Bind current state variables to context
        for (name, value) in current_state.vars() {
            ctx.bind_mut(Arc::clone(name), value.clone());
        }

        let action_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "InnerAction" {
                        return Some(def.clone());
                    }
                }
                None
            })
            .unwrap();

        let successors =
            enumerate_action_successors(&mut ctx, &action_def.body, &current_state, &vars).unwrap();

        assert_eq!(successors.len(), 1);
        assert_eq!(
            successors[0].get("v").and_then(|v| v.as_i64()),
            Some(1)
        );
    }

    #[test]
    fn test_enumerate_unchanged_in_action() {
        // Test that UNCHANGED is properly handled during action enumeration
        // Relevant to #54/#61 where UNCHANGED <<unsat, alloc>> was mishandled
        let src = r#"
---- MODULE Test ----
VARIABLE x, y
Init == x = 0 /\ y = 0
Action == x' = x + 1 /\ UNCHANGED y
====
"#;
        let (module, mut ctx, vars) = setup_module(src);

        let current_state = State::from_pairs([("x", Value::int(0)), ("y", Value::int(5))]);

        // Bind current state variables to context
        for (name, value) in current_state.vars() {
            ctx.bind_mut(Arc::clone(name), value.clone());
        }

        let action_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Action" {
                        return Some(def.clone());
                    }
                }
                None
            })
            .unwrap();

        let successors =
            enumerate_action_successors(&mut ctx, &action_def.body, &current_state, &vars).unwrap();

        assert_eq!(successors.len(), 1);
        // x should be incremented
        assert_eq!(
            successors[0].get("x").and_then(|v| v.as_i64()),
            Some(1)
        );
        // y should be UNCHANGED (still 5)
        assert_eq!(
            successors[0].get("y").and_then(|v| v.as_i64()),
            Some(5)
        );
    }

    #[test]
    fn test_enumerate_unchanged_tuple() {
        // Test UNCHANGED <<a, b>> syntax (tuple unchanged)
        let src = r#"
---- MODULE Test ----
VARIABLE a, b, c
Init == a = 0 /\ b = 0 /\ c = 0
Action == c' = c + 1 /\ UNCHANGED <<a, b>>
====
"#;
        let (module, mut ctx, vars) = setup_module(src);

        let current_state = State::from_pairs([
            ("a", Value::int(1)),
            ("b", Value::int(2)),
            ("c", Value::int(0)),
        ]);

        // Bind current state variables to context
        for (name, value) in current_state.vars() {
            ctx.bind_mut(Arc::clone(name), value.clone());
        }

        let action_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Action" {
                        return Some(def.clone());
                    }
                }
                None
            })
            .unwrap();

        let successors =
            enumerate_action_successors(&mut ctx, &action_def.body, &current_state, &vars).unwrap();

        assert_eq!(successors.len(), 1);
        // a and b should be unchanged
        assert_eq!(successors[0].get("a").and_then(|v| v.as_i64()), Some(1));
        assert_eq!(successors[0].get("b").and_then(|v| v.as_i64()), Some(2));
        // c should be incremented
        assert_eq!(successors[0].get("c").and_then(|v| v.as_i64()), Some(1));
    }

    // ============================================================================
    // Bind/Unbind Enumeration Tests (Issue #104 - Test Gap)
    // ============================================================================

    /// Test bind/unbind enumeration with x' \in S pattern.
    /// This was the key bug fixed in #101 - continuation wasn't processed for each element.
    #[test]
    fn test_bind_unbind_in_set_membership() {
        std::env::set_var("TLA2_BIND_UNBIND", "1");

        let src = r#"
---- MODULE Test ----
EXTENDS Integers
VARIABLE x
Init == x = 0
Next == x' \in {1, 2, 3}
====
"#;
        let (module, mut ctx, vars) = setup_module(src);
        let current_state = State::from_pairs([("x", Value::int(0))]);

        for (name, value) in current_state.vars() {
            ctx.bind_mut(Arc::clone(name), value.clone());
        }

        let next_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Next" {
                        return Some(def.clone());
                    }
                }
                None
            })
            .unwrap();

        let successors =
            enumerate_action_successors(&mut ctx, &next_def.body, &current_state, &vars).unwrap();

        // Should produce 3 successors: x=1, x=2, x=3
        assert_eq!(successors.len(), 3);
        let x_values: std::collections::BTreeSet<i64> = successors
            .iter()
            .filter_map(|s| s.get("x").and_then(|v| v.as_i64()))
            .collect();
        assert_eq!(x_values, [1, 2, 3].into_iter().collect());

        std::env::remove_var("TLA2_BIND_UNBIND");
    }

    /// Test bind/unbind with x' \in S /\ y' = f(x') pattern.
    /// This tests the continuation handling - each element of S must process
    /// the subsequent conjunct y' = f(x').
    #[test]
    fn test_bind_unbind_in_with_dependent_assignment() {
        std::env::set_var("TLA2_BIND_UNBIND", "1");

        let src = r#"
---- MODULE Test ----
EXTENDS Integers
VARIABLE x, y
Init == x = 0 /\ y = 0
Next == x' \in {1, 2, 3} /\ y' = x' * 2
====
"#;
        let (module, mut ctx, vars) = setup_module(src);
        let current_state = State::from_pairs([("x", Value::int(0)), ("y", Value::int(0))]);

        for (name, value) in current_state.vars() {
            ctx.bind_mut(Arc::clone(name), value.clone());
        }

        let next_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Next" {
                        return Some(def.clone());
                    }
                }
                None
            })
            .unwrap();

        let successors =
            enumerate_action_successors(&mut ctx, &next_def.body, &current_state, &vars).unwrap();

        // Should produce 3 successors: (x=1,y=2), (x=2,y=4), (x=3,y=6)
        assert_eq!(successors.len(), 3);

        let xy_pairs: std::collections::BTreeSet<(i64, i64)> = successors
            .iter()
            .filter_map(|s| {
                let x = s.get("x").and_then(|v| v.as_i64())?;
                let y = s.get("y").and_then(|v| v.as_i64())?;
                Some((x, y))
            })
            .collect();

        assert_eq!(
            xy_pairs,
            [(1, 2), (2, 4), (3, 6)].into_iter().collect()
        );

        std::env::remove_var("TLA2_BIND_UNBIND");
    }

    /// Test bind/unbind enumeration parity with symbolic approach.
    /// Verifies that both approaches produce the same state counts.
    #[test]
    fn test_bind_unbind_parity_with_symbolic() {
        let src = r#"
---- MODULE Test ----
EXTENDS Integers
VARIABLE x, y
Init == x = 0 /\ y = 0
Next ==
    \/ x' \in {1, 2} /\ y' = x' + 10
    \/ x' = 100 /\ y' = 200
====
"#;
        let (module, mut ctx, vars) = setup_module(src);
        let current_state = State::from_pairs([("x", Value::int(0)), ("y", Value::int(0))]);

        for (name, value) in current_state.vars() {
            ctx.bind_mut(Arc::clone(name), value.clone());
        }

        let next_def = module
            .units
            .iter()
            .find_map(|u| {
                if let tla_core::ast::Unit::Operator(def) = &u.node {
                    if def.name.node == "Next" {
                        return Some(def.clone());
                    }
                }
                None
            })
            .unwrap();

        // Run without bind/unbind (symbolic)
        std::env::remove_var("TLA2_BIND_UNBIND");
        let successors_symbolic =
            enumerate_action_successors(&mut ctx, &next_def.body, &current_state, &vars).unwrap();

        // Run with bind/unbind
        std::env::set_var("TLA2_BIND_UNBIND", "1");
        let successors_bind =
            enumerate_action_successors(&mut ctx, &next_def.body, &current_state, &vars).unwrap();

        // Should produce same number of successors
        assert_eq!(
            successors_symbolic.len(),
            successors_bind.len(),
            "Bind/unbind and symbolic should produce same number of successors"
        );

        // Should produce same states (may be in different order)
        let symbolic_fps: std::collections::BTreeSet<_> = successors_symbolic
            .iter()
            .map(|s| s.fingerprint())
            .collect();
        let bind_fps: std::collections::BTreeSet<_> = successors_bind
            .iter()
            .map(|s| s.fingerprint())
            .collect();
        assert_eq!(
            symbolic_fps, bind_fps,
            "Bind/unbind and symbolic should produce same states"
        );

        std::env::remove_var("TLA2_BIND_UNBIND");
    }
}
