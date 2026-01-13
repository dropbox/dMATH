//! Expression evaluator for TLA+
//!
//! This module implements evaluation of TLA+ expressions to runtime values.
//! The evaluator handles:
//! - Literal values (booleans, integers, strings)
//! - Logical operators (and, or, not, implies, equiv)
//! - Arithmetic operators (+, -, *, /, %, ^, ..)
//! - Set operations (union, intersection, subset, membership)
//! - Function and record operations
//! - Quantifiers (forall, exists, choose)
//! - Control flow (if-then-else, case, let)
//!
//! # Environment
//!
//! The evaluator uses an environment (Env) to track variable bindings.
//! Environments are immutable and use structural sharing for efficiency.

use crate::enumerate::eval_enabled;
use crate::error::{EvalError, EvalResult};
use crate::state::value_fingerprint;
use crate::value::{
    big_union, boolean_set, cartesian_product, intern_string, range_set, ClosureValue,
    ComponentDomain, FuncSetValue, FuncValue, IntIntervalFunc, IntervalValue, KSubsetValue,
    LazyDomain, LazyFuncValue, RecordBuilder, RecordSetValue, RecordValue, SeqSetValue, SetBuilder,
    SetCapValue, SetCupValue, SetDiffValue, SetPredValue, SortedSet, SubsetValue, TupleSetValue,
    UnionValue, Value,
};
use crate::var_index::{VarIndex, VarRegistry};
use im::{HashMap, OrdSet};
use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, ToPrimitive, Zero};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tla_core::ast::{
    BoundPattern, BoundVar, ExceptPathElement, ExceptSpec, Expr, Module, ModuleTarget, OperatorDef,
    Substitution,
};
use tla_core::{Span, Spanned};

static SHARED_CTX_NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Evaluation environment: variable bindings
pub type Env = HashMap<Arc<str>, Value>;

/// Operator definition storage
pub type OpEnv = HashMap<String, OperatorDef>;

/// Instance declaration info for INSTANCE WITH evaluation
#[derive(Debug, Clone)]
pub struct InstanceInfo {
    /// The module being instanced
    pub module_name: String,
    /// Substitutions from the WITH clause
    pub substitutions: Vec<Substitution>,
}

/// Reference to an operator (either user-defined or built-in)
#[derive(Debug, Clone)]
enum OperatorRef {
    /// User-defined operator (looked up in ops)
    UserDefined(String),
    /// Built-in operator (+, -, *, etc.)
    BuiltIn(String),
}

/// Apply a binary operator (user-defined or built-in) to two values
fn apply_binary_op(
    ctx: &EvalCtx,
    op_ref: &OperatorRef,
    left: Value,
    right: Value,
    span: Option<Span>,
) -> EvalResult<Value> {
    match op_ref {
        OperatorRef::UserDefined(op_name) => {
            // Look up user-defined operator
            let op_def = ctx.get_op(op_name).ok_or_else(|| EvalError::UndefinedOp {
                name: op_name.clone(),
                span,
            })?;
            if op_def.params.len() != 2 {
                return Err(EvalError::ArityMismatch {
                    op: op_name.clone(),
                    expected: 2,
                    got: op_def.params.len(),
                    span,
                });
            }
            let new_ctx = ctx
                .bind(op_def.params[0].name.node.as_str(), left)
                .bind(op_def.params[1].name.node.as_str(), right);
            eval(&new_ctx, &op_def.body)
        }
        OperatorRef::BuiltIn(op) => {
            // Apply built-in operator directly
            apply_builtin_binary_op(op, left, right, span)
        }
    }
}

/// Apply a built-in binary operator
fn apply_builtin_binary_op(
    op: &str,
    left: Value,
    right: Value,
    span: Option<Span>,
) -> EvalResult<Value> {
    match op {
        // Basic arithmetic - use int_arith_op helper
        "+" => int_arith_op(left, right, i64::checked_add, |a, b| a + b, span),
        "-" => int_arith_op(left, right, i64::checked_sub, |a, b| a - b, span),
        "*" => int_arith_op(left, right, i64::checked_mul, |a, b| a * b, span),

        // Division operations - use int_div_op helper (includes zero check)
        "/" => int_div_op(left, right, i64::checked_div, |a, b| a / b, span),
        "\\div" => int_div_op(
            left,
            right,
            |a, b| Some(a.div_euclid(b)),
            |a, b| a.div_floor(&b),
            span,
        ),
        "%" => int_div_op(
            left,
            right,
            |a, b| Some(a.rem_euclid(b)),
            |a, b| {
                // Euclidean modulo (always non-negative for positive divisor)
                let r = a % &b;
                (r + &b) % &b
            },
            span,
        ),

        // Power - use int_pow_op helper
        "^" => int_pow_op(left, right, span),

        "\\cup" => {
            // Fast path: single-element union (e.g., sent \cup {<<x, y>>})
            // Check if element already present before cloning
            if let Value::Set(ref b_set) = right {
                if b_set.len() == 1 {
                    if let Value::Set(ref a_set) = left {
                        let elem = b_set.iter().next().unwrap();
                        // Short-circuit: if element already in set, return original unchanged
                        if a_set.contains(elem) {
                            return Ok(left.clone());
                        }
                        // Insert returns a new SortedSet
                        let result = a_set.insert(elem.clone());
                        return Ok(Value::Set(result));
                    }
                }
            }
            // Symmetric fast path: {x} \cup larger_set
            if let Value::Set(ref a_set) = left {
                if a_set.len() == 1 {
                    if let Value::Set(ref b_set) = right {
                        let elem = a_set.iter().next().unwrap();
                        // Short-circuit: if element already in set, return original unchanged
                        if b_set.contains(elem) {
                            return Ok(right.clone());
                        }
                        // Insert returns a new SortedSet
                        let result = b_set.insert(elem.clone());
                        return Ok(Value::Set(result));
                    }
                }
            }
            // Handle both Set and Interval using to_sorted_set
            let a = left
                .to_sorted_set()
                .ok_or_else(|| EvalError::type_error("Set", &left, span))?;
            let b = right
                .to_sorted_set()
                .ok_or_else(|| EvalError::type_error("Set", &right, span))?;
            Ok(Value::Set(a.union(&b)))
        }
        "\\cap" => {
            // Fast path: single-element intersection (e.g., {x} \cap S)
            // Result is either {x} if x in S, or {} otherwise
            if let Value::Set(ref a_set) = left {
                if a_set.len() == 1 {
                    if let Value::Set(ref b_set) = right {
                        // Check if the single element is in the other set
                        let elem = a_set.iter().next().unwrap();
                        if b_set.contains(elem) {
                            return Ok(left.clone());
                        } else {
                            return Ok(Value::Set(SortedSet::new()));
                        }
                    }
                }
            }
            // Symmetric fast path: S \cap {x}
            if let Value::Set(ref b_set) = right {
                if b_set.len() == 1 {
                    if let Value::Set(ref a_set) = left {
                        let elem = b_set.iter().next().unwrap();
                        if a_set.contains(elem) {
                            return Ok(right.clone());
                        } else {
                            return Ok(Value::Set(SortedSet::new()));
                        }
                    }
                }
            }
            // Handle both Set and Interval using to_sorted_set
            let a = left
                .to_sorted_set()
                .ok_or_else(|| EvalError::type_error("Set", &left, span))?;
            let b = right
                .to_sorted_set()
                .ok_or_else(|| EvalError::type_error("Set", &right, span))?;
            Ok(Value::Set(a.intersection(&b)))
        }
        "\\" => {
            // Fast path: single-element difference (e.g., Corr \ {self})
            // Check if element present before cloning
            if let Value::Set(ref b_set) = right {
                if b_set.len() == 1 {
                    if let Value::Set(ref a_set) = left {
                        let elem = b_set.iter().next().unwrap();
                        // Short-circuit: if element not in set, return original unchanged
                        if !a_set.contains(elem) {
                            return Ok(left.clone());
                        }
                        // Remove returns a new SortedSet
                        let result = a_set.remove(elem);
                        return Ok(Value::Set(result));
                    }
                }
            }
            // Handle both Set and Interval using to_sorted_set
            let a = left
                .to_sorted_set()
                .ok_or_else(|| EvalError::type_error("Set", &left, span))?;
            let b = right
                .to_sorted_set()
                .ok_or_else(|| EvalError::type_error("Set", &right, span))?;
            Ok(Value::Set(a.difference(&b)))
        }
        "\\o" => {
            // Sequence concatenation
            fn func_seq_len(f: &FuncValue) -> Option<usize> {
                let mut expected: i64 = 1;
                for key in f.domain_iter() {
                    if key.as_i64()? != expected {
                        return None;
                    }
                    expected += 1;
                }
                Some((expected - 1) as usize)
            }

            fn seq_like_len(v: &Value) -> Option<usize> {
                if let Some(elems) = v.as_seq_or_tuple_elements() {
                    return Some(elems.len());
                }
                match v {
                    Value::IntFunc(f) if f.min == 1 => Some(f.len()),
                    Value::Func(f) => func_seq_len(f),
                    _ => None,
                }
            }

            let a_len =
                seq_like_len(&left).ok_or_else(|| EvalError::type_error("Seq", &left, span))?;
            let b_len =
                seq_like_len(&right).ok_or_else(|| EvalError::type_error("Seq", &right, span))?;

            let mut result: Vec<Value> = Vec::with_capacity(a_len + b_len);

            if let Some(elems) = left.as_seq_or_tuple_elements() {
                result.extend(elems.iter().cloned());
            } else {
                match &left {
                    Value::IntFunc(f) if f.min == 1 => result.extend(f.values.iter().cloned()),
                    Value::Func(f) => {
                        let mut expected: i64 = 1;
                        for (k, v) in f.mapping_iter() {
                            let Some(idx) = k.as_i64() else {
                                return Err(EvalError::type_error("Seq", &left, span));
                            };
                            if idx != expected {
                                return Err(EvalError::type_error("Seq", &left, span));
                            }
                            expected += 1;
                            result.push(v.clone());
                        }
                    }
                    _ => return Err(EvalError::type_error("Seq", &left, span)),
                }
            }

            if let Some(elems) = right.as_seq_or_tuple_elements() {
                result.extend(elems.iter().cloned());
            } else {
                match &right {
                    Value::IntFunc(f) if f.min == 1 => result.extend(f.values.iter().cloned()),
                    Value::Func(f) => {
                        let mut expected: i64 = 1;
                        for (k, v) in f.mapping_iter() {
                            let Some(idx) = k.as_i64() else {
                                return Err(EvalError::type_error("Seq", &right, span));
                            };
                            if idx != expected {
                                return Err(EvalError::type_error("Seq", &right, span));
                            }
                            expected += 1;
                            result.push(v.clone());
                        }
                    }
                    _ => return Err(EvalError::type_error("Seq", &right, span)),
                }
            }
            Ok(Value::Seq(result.into()))
        }
        _ => Err(EvalError::Internal {
            message: format!("Unknown built-in binary operator: {}", op),
            span,
        }),
    }
}

// ============================================================================
// Integer arithmetic helper functions
// ============================================================================
// These helpers consolidate the SmallInt fast path + BigInt fallback pattern
// used throughout arithmetic evaluation.

/// Apply binary arithmetic operation with SmallInt fast path
///
/// Used for +, -, * which don't need division-by-zero checks.
fn int_arith_op(
    left: Value,
    right: Value,
    small_op: impl Fn(i64, i64) -> Option<i64>,
    big_op: impl Fn(BigInt, BigInt) -> BigInt,
    span: Option<Span>,
) -> EvalResult<Value> {
    // SmallInt fast path
    if let (Value::SmallInt(a), Value::SmallInt(b)) = (&left, &right) {
        if let Some(result) = small_op(*a, *b) {
            return Ok(Value::SmallInt(result));
        }
        // Overflow: fall through to BigInt
    }
    // BigInt path
    let a = left
        .to_bigint()
        .ok_or_else(|| EvalError::type_error("Int", &left, span))?;
    let b = right
        .to_bigint()
        .ok_or_else(|| EvalError::type_error("Int", &right, span))?;
    Ok(Value::big_int(big_op(a, b)))
}

/// Apply division operation with SmallInt fast path and zero check
///
/// Used for /, \div, % which need division-by-zero checks.
/// The `small_op` returns `Option<i64>` to handle potential overflow.
fn int_div_op(
    left: Value,
    right: Value,
    small_op: impl Fn(i64, i64) -> Option<i64>,
    big_op: impl Fn(BigInt, BigInt) -> BigInt,
    span: Option<Span>,
) -> EvalResult<Value> {
    // SmallInt fast path
    if let (Value::SmallInt(a), Value::SmallInt(b)) = (&left, &right) {
        if *b == 0 {
            return Err(EvalError::DivisionByZero { span });
        }
        if let Some(result) = small_op(*a, *b) {
            return Ok(Value::SmallInt(result));
        }
        // Overflow (e.g., MIN / -1): fall through to BigInt
    }
    // BigInt path
    let a = left
        .to_bigint()
        .ok_or_else(|| EvalError::type_error("Int", &left, span))?;
    let b = right
        .to_bigint()
        .ok_or_else(|| EvalError::type_error("Int", &right, span))?;
    if b.is_zero() {
        return Err(EvalError::DivisionByZero { span });
    }
    Ok(Value::big_int(big_op(a, b)))
}

/// Apply power operation with SmallInt fast path
///
/// Special handling: exponent must be non-negative and fit in u32.
fn int_pow_op(left: Value, right: Value, span: Option<Span>) -> EvalResult<Value> {
    // SmallInt fast path for small exponents
    if let (Value::SmallInt(base), Value::SmallInt(exp)) = (&left, &right) {
        if *exp >= 0 && *exp <= 62 {
            if let Some(result) = base.checked_pow(*exp as u32) {
                return Ok(Value::SmallInt(result));
            }
        }
        // Overflow or large exponent: fall through to BigInt
    }
    // BigInt path
    let base = left
        .to_bigint()
        .ok_or_else(|| EvalError::type_error("Int", &left, span))?;
    let exp = right
        .to_bigint()
        .ok_or_else(|| EvalError::type_error("Int", &right, span))?;
    let exp_u32 = exp.to_u32().ok_or_else(|| EvalError::Internal {
        message: "Exponent too large or negative".into(),
        span,
    })?;
    Ok(Value::big_int(base.pow(exp_u32)))
}

/// TLC configuration values exposed via TLCGet("config")
/// These are static configuration values that don't change during model checking.
#[derive(Debug, Clone)]
pub struct TlcConfig {
    /// Exploration mode: "bfs" (model checking), "generate" (random behaviors), or "simulate"
    pub mode: Arc<str>,
    /// Max depth (-1 for unlimited)
    pub depth: i64,
    /// Whether deadlock checking is enabled
    pub deadlock: bool,
}

impl Default for TlcConfig {
    fn default() -> Self {
        TlcConfig {
            mode: Arc::from("bfs"),
            depth: -1,
            deadlock: true,
        }
    }
}

/// Shared immutable context - wrapped in Arc for cheap cloning
/// Contains data that never changes during evaluation (operators, instances, etc.)
#[derive(Debug, Clone)]
pub struct SharedCtx {
    /// Stable identifier for this shared context (used to scope thread-local caches).
    pub id: u64,
    /// Operator definitions
    pub ops: OpEnv,
    /// Named instances: instance_name -> InstanceInfo
    /// For `InChan == INSTANCE Channel WITH ...`, stores "InChan" -> {module: "Channel", subs: ...}
    pub instances: HashMap<String, InstanceInfo>,
    /// Operators from instanced modules: module_name -> op_name -> OperatorDef
    /// Stores operators from INSTANCE'd modules (not yet substituted)
    pub instance_ops: HashMap<String, OpEnv>,
    /// Operator replacements: old_name -> new_name (for config `CONSTANT Op <- Replacement`)
    pub op_replacements: HashMap<String, String>,
    /// Variable name registry for O(1) index lookup
    /// Populated once at module load time
    pub var_registry: VarRegistry,
    /// TLC configuration for TLCGet("config")
    pub tlc_config: TlcConfig,
    /// Constants overridden by config file.
    /// When evaluating an identifier, if it's in this set, check env bindings
    /// BEFORE operator definitions. This allows config to override operator
    /// definitions like `Done == CHOOSE v : v \notin Reg` with model values.
    pub config_constants: std::collections::HashSet<String>,
}

impl SharedCtx {
    /// Create a new shared context
    pub fn new() -> Self {
        SharedCtx {
            id: SHARED_CTX_NEXT_ID.fetch_add(1, Ordering::Relaxed),
            ops: HashMap::new(),
            instances: HashMap::new(),
            instance_ops: HashMap::new(),
            op_replacements: HashMap::new(),
            var_registry: VarRegistry::new(),
            tlc_config: TlcConfig::default(),
            config_constants: std::collections::HashSet::new(),
        }
    }

    /// Create a shared context with a pre-populated variable registry
    pub fn with_var_registry(var_registry: VarRegistry) -> Self {
        SharedCtx {
            id: SHARED_CTX_NEXT_ID.fetch_add(1, Ordering::Relaxed),
            ops: HashMap::new(),
            instances: HashMap::new(),
            instance_ops: HashMap::new(),
            op_replacements: HashMap::new(),
            var_registry,
            tlc_config: TlcConfig::default(),
            config_constants: std::collections::HashSet::new(),
        }
    }
}

impl Default for SharedCtx {
    fn default() -> Self {
        Self::new()
    }
}

/// Evaluation context
///
/// The context is split into two parts for efficient cloning:
/// - `shared`: Arc-wrapped immutable data (operators, instances) - O(1) clone
/// - `env`, `next_state`, `local_ops`: Per-evaluation mutable data
///
/// For performance during state enumeration, temporary bindings can be pushed
/// to `local_stack` using `push_binding()` and restored using `pop_to_mark()`.
/// This avoids HashMap allocation overhead for millions of state transitions.
/// Maximum recursion depth for function evaluation.
/// This prevents stack overflow when evaluating deeply recursive TLA+ functions
/// like `f[i \in 0..N] == ... f[i-1] ...`. TLC uses a similar limit (default 100).
/// We use a higher limit (500) but also track general eval depth.
pub const MAX_RECURSION_DEPTH: u32 = 500;

// Thread-local counter for tracking eval recursion depth.
// This catches stack overflow from any source of deep recursion in eval,
// not just recursive TLA+ functions.
std::thread_local! {
    static EVAL_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

// Thread-local cache for INSTANCE substitution evaluation within a single top-level `eval` call.
//
// INSTANCE substitutions like `inbox <- Node2Nat(EWD998ChanInbox)` can be referenced many times
// within a single action predicate evaluation (e.g., `inbox[0]`, `Len(inbox[0])`, ...). Without
// caching, we end up re-evaluating the substitution RHS repeatedly which can be extremely costly.
//
// Safety: This cache is cleared at the beginning and end of each *top-level* `eval` invocation
// (depth==1), so cached values never outlive the environment they were computed in.
std::thread_local! {
    static SUBST_CACHE: std::cell::RefCell<
        std::collections::HashMap<(bool, String), Value>
    > = std::cell::RefCell::new(std::collections::HashMap::new());
}

// Thread-local cache for user-defined operator results within a guard evaluation scope.
//
// Specs like bosco have guards that call the same operator multiple times with the same arguments:
//   rcvd01(self) >= N - T /\ rcvd0(self) >= moreNplus3Tdiv2 /\ rcvd0(self) < moreNplus3Tdiv2
//
// Each call to `rcvd0(self)` evaluates `Cardinality({m \in rcvd'[self] : m[2] = "ECHO0"})` which
// iterates over all elements in rcvd'[self]. Caching avoids redundant evaluations.
//
// Baseline alignment:
// TLC caches evaluation results via LazyValue and validates cache hits with
// `TLCState.isSubset(s0, cachedS0)` / `TLCState.isSubset(s1, cachedS1)`, rather than requiring
// an exact match on the full state.
//
// In Rust we implement a *sound* subset-style cache by tracking the concrete dependencies
// (state vars, next-state vars, and captured locals) actually read during evaluation, and
// reusing a cached value only when all recorded dependencies still match the current context.

#[derive(Clone, Hash, PartialEq, Eq)]
struct OpResultCacheKey {
    shared_id: u64,
    local_ops_id: usize,
    instance_subs_id: usize,
    op_name: String,
    args: Arc<[Value]>,
}

#[derive(Clone, Default)]
struct OpEvalDeps {
    // Captured locals from the *caller* scope (below base_stack_len).
    // These matter for LET-defined operators that close over bound variables.
    local: Vec<(Arc<str>, Value)>,
    // Reads of unprimed state variables.
    state: Vec<(VarIndex, Value)>,
    // Reads of primed (next-state) variables, plus unprimed reads while evaluating in next-state mode.
    next: Vec<(VarIndex, Value)>,
    inconsistent: bool,
}

impl OpEvalDeps {
    fn record_local(&mut self, name: &str, value: &Value) {
        let name: Arc<str> = Arc::from(name);
        if let Some((_, existing)) = self.local.iter_mut().find(|(n, _)| n.as_ref() == name.as_ref())
        {
            if existing != value {
                self.inconsistent = true;
            }
            return;
        }
        self.local.push((name, value.clone()));
    }

    fn record_state(&mut self, idx: VarIndex, value: &Value) {
        if let Some((_, existing)) = self.state.iter_mut().find(|(i, _)| *i == idx) {
            if existing != value {
                self.inconsistent = true;
            }
            return;
        }
        self.state.push((idx, value.clone()));
    }

    fn record_next(&mut self, idx: VarIndex, value: &Value) {
        if let Some((_, existing)) = self.next.iter_mut().find(|(i, _)| *i == idx) {
            if existing != value {
                self.inconsistent = true;
            }
            return;
        }
        self.next.push((idx, value.clone()));
    }

    fn merge_from(&mut self, other: &OpEvalDeps) {
        if other.inconsistent {
            self.inconsistent = true;
        }
        for (name, value) in &other.local {
            self.record_local(name.as_ref(), value);
        }
        for (idx, value) in &other.state {
            self.record_state(*idx, value);
        }
        for (idx, value) in &other.next {
            self.record_next(*idx, value);
        }
    }
}

struct OpDepFrame {
    deps: OpEvalDeps,
}

std::thread_local! {
    static OP_DEP_STACK: std::cell::RefCell<Vec<OpDepFrame>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StateLookupMode {
    Current,
    Next,
}

std::thread_local! {
    static STATE_LOOKUP_MODE: std::cell::Cell<StateLookupMode> = const { std::cell::Cell::new(StateLookupMode::Current) };
}

fn with_state_lookup_mode<T>(mode: StateLookupMode, f: impl FnOnce() -> T) -> T {
    STATE_LOOKUP_MODE.with(|m| {
        let prev = m.get();
        m.set(mode);
        let out = f();
        m.set(prev);
        out
    })
}

fn current_state_lookup_mode() -> StateLookupMode {
    STATE_LOOKUP_MODE.with(|m| m.get())
}

fn record_local_read(_resolved_stack_index: usize, name: &str, value: &Value) {
    // Issue #70 fix: Always record local reads as dependencies, regardless of stack position.
    //
    // Previously, we only recorded dependencies for variables at indices < base_stack_len,
    // intending to exclude the current operator's parameters (which are in the cache key).
    // But this caused a bug when variables came from outer-scope operator calls (like
    // UpOther(n) calling valSent(m)), where `n` was added by an intermediate operator
    // and fell at/after the current operator's base_stack_len.
    //
    // Recording all local reads is safe because:
    // 1. The current operator's params are already in the cache key as args
    // 2. Extra dependency tracking is harmless (just slightly redundant validation)
    // 3. The cache validation correctly handles all recorded dependencies
    OP_DEP_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        let Some(top) = stack.last_mut() else {
            // Not inside operator dep tracking - this is normal for most lookups
            return;
        };
        top.deps.record_local(name, value);
    });
}

fn record_state_read(idx: VarIndex, value: &Value) {
    OP_DEP_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        let Some(top) = stack.last_mut() else {
            // Not inside operator dep tracking - this is normal for most lookups
            return;
        };
        match current_state_lookup_mode() {
            StateLookupMode::Current => top.deps.record_state(idx, value),
            StateLookupMode::Next => top.deps.record_next(idx, value),
        }
    });
}

fn record_next_read(idx: VarIndex, value: &Value) {
    OP_DEP_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        let Some(top) = stack.last_mut() else { return };
        top.deps.record_next(idx, value);
    });
}

fn propagate_cached_deps(deps: &OpEvalDeps) {
    OP_DEP_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        let Some(top) = stack.last_mut() else { return };
        top.deps.merge_from(deps);
    });
}

fn eval_with_dep_tracking(ctx: &EvalCtx, expr: &Spanned<Expr>) -> EvalResult<(Value, OpEvalDeps)> {
    OP_DEP_STACK.with(|stack| {
        stack.borrow_mut().push(OpDepFrame {
            deps: OpEvalDeps::default(),
        });
    });

    let result = eval(ctx, expr);

    let frame = OP_DEP_STACK.with(|stack| {
        stack
            .borrow_mut()
            .pop()
            .expect("OP_DEP_STACK push/pop must be balanced")
    });

    match result {
        Ok(v) => {
            propagate_cached_deps(&frame.deps);
            Ok((v, frame.deps))
        }
        Err(e) => Err(e),
    }
}

/// Cached operator result with TLC-style subset validation.
#[derive(Clone)]
struct CachedOpResult {
    value: Value,
    deps: OpEvalDeps,
}

fn op_cache_entry_valid(ctx: &EvalCtx, entry: &CachedOpResult) -> bool {
    if entry.deps.inconsistent {
        return false;
    }

    // Validate captured locals by name resolution against the *current* local_stack.
    for (name, expected) in &entry.deps.local {
        let found = ctx
            .local_stack
            .iter()
            .rev()
            .find(|(n, _)| n.as_ref() == name.as_ref())
            .map(|(_, v)| v);
        match found {
            Some(v) if v == expected => {}
            _ => return false,
        }
    }

    // Validate state deps against the current state array.
    // Issue #73: Only require state_env if there are state dependencies.
    // This allows caching of pure operators (like IsStronglyConnected in YoYoAllGraphs Init)
    // that don't read state variables, even during Init enumeration where state_env is None.
    if !entry.deps.state.is_empty() {
        let Some(state_env) = ctx.state_env else {
            return false;
        };
        for (idx, expected) in &entry.deps.state {
            let actual = unsafe { state_env.get_unchecked(idx.as_usize()) };
            if actual != expected {
                return false;
            }
        }
    }

    // Validate next-state deps (if any) against whatever next-state context is available.
    if !entry.deps.next.is_empty() {
        if let Some(next_env) = ctx.next_state_env {
            for (idx, expected) in &entry.deps.next {
                let actual = unsafe { next_env.get_unchecked(idx.as_usize()) };
                if actual != expected {
                    return false;
                }
            }
        } else if let Some(next_state) = &ctx.next_state {
            for (idx, expected) in &entry.deps.next {
                let name = ctx.var_registry().name(*idx);
                let Some(actual) = next_state.get(name) else {
                    return false;
                };
                if actual != expected {
                    return false;
                }
            }
        } else {
            return false;
        }
    }

    true
}

std::thread_local! {
    static OP_RESULT_CACHE: std::cell::RefCell<
        std::collections::HashMap<OpResultCacheKey, CachedOpResult>
    > = std::cell::RefCell::new(std::collections::HashMap::new());
}

// Issue #73 fix: Separate multi-entry cache for zero-arg operators.
//
// Key: (shared_id, op_name) - simpler than OpResultCacheKey.
// Does NOT include local_ops_id because Arc pointers change on every enumeration
// step even when the logical content is identical (see enumerate.rs:4281-4286).
//
// Value: Vec of cached results - allows multiple entries per key with different deps.
// This is critical because the same operator may be called with different local contexts
// (e.g., different values of bound variables from enclosing scopes) or different state.
// On lookup, we scan the Vec to find an entry with deps matching the current context.
//
// Validation uses op_cache_entry_valid() which checks deps.local and deps.state.
std::thread_local! {
    static ZERO_ARG_OP_CACHE: std::cell::RefCell<
        std::collections::HashMap<(u64, String), Vec<CachedOpResult>>
    > = std::cell::RefCell::new(std::collections::HashMap::new());
}

/// Maximum entries per key in ZERO_ARG_OP_CACHE to prevent unbounded memory growth.
/// When exceeded, oldest entries are evicted.
const ZERO_ARG_CACHE_MAX_ENTRIES_PER_KEY: usize = 16;

/// Test-only: Get the current size of OP_RESULT_CACHE.
#[cfg(test)]
pub(crate) fn op_result_cache_len() -> usize {
    OP_RESULT_CACHE.with(|cache| cache.borrow().len())
}

/// Test-only: Clear the OP_RESULT_CACHE for test isolation.
#[cfg(test)]
pub(crate) fn op_result_cache_clear() {
    OP_RESULT_CACHE.with(|cache| cache.borrow_mut().clear());
}

/// Test-only: Get the total number of entries in ZERO_ARG_OP_CACHE (summed across all keys).
#[cfg(test)]
pub(crate) fn zero_arg_op_cache_len() -> usize {
    ZERO_ARG_OP_CACHE.with(|cache| {
        cache.borrow().values().map(|v| v.len()).sum()
    })
}

/// Test-only: Clear the ZERO_ARG_OP_CACHE for test isolation.
#[cfg(test)]
pub(crate) fn zero_arg_op_cache_clear() {
    ZERO_ARG_OP_CACHE.with(|cache| cache.borrow_mut().clear());
}

/// Maximum eval depth before returning an error. Each eval call uses significant
/// stack space, so we limit this to prevent stack overflow. With a 64MB stack,
/// we can safely handle depths around 10000 (conservative to leave headroom for
/// stack operations between depth checks).
const MAX_EVAL_DEPTH: u32 = 10000;

pub struct EvalCtx {
    /// Shared immutable context (Arc-wrapped for cheap cloning)
    pub shared: Arc<SharedCtx>,
    /// Variable environment (base bindings)
    pub env: Env,
    /// Next-state values for resolving primed variables (x' -> value)
    /// Arc-wrapped for cheap cloning during bind()
    pub next_state: Option<Arc<Env>>,
    /// Local operator definitions (from LET expressions)
    /// These shadow the shared ops
    /// Arc-wrapped for cheap cloning during bind()
    pub local_ops: Option<Arc<OpEnv>>,
    /// Stack of temporary bindings for fast push/pop during enumeration
    /// Checked before `env` during lookups. O(1) push, O(1) pop to mark.
    local_stack: Vec<(Arc<str>, Value)>,
    /// Array-based state variable storage for O(1) access.
    ///
    /// This points at the current state's `[Value]` array (usually from `ArrayState`)
    /// and is only valid while the caller keeps that array alive. This avoids
    /// per-state cloning/allocation in the BFS hot path.
    state_env: Option<StateEnvRef>,
    /// Array-based next-state variable storage for O(1) primed variable access.
    ///
    /// Similar to `state_env`, but for next-state values (primed variables like x').
    /// Used by `prime_guards_hold_in_next_array` to avoid HashMap construction overhead.
    next_state_env: Option<StateEnvRef>,
    /// Current recursion depth for function evaluation.
    /// Prevents stack overflow when evaluating deeply recursive TLA+ functions.
    recursion_depth: u32,
    /// Active INSTANCE substitutions for the current evaluation scope.
    ///
    /// When evaluating an operator from an instanced module (e.g., `C!Spec`), we store the
    /// substitutions applied to that instance so nested instances inside that module can
    /// inherit them (TLC composes substitutions through nested INSTANCE declarations).
    instance_substitutions: Option<Arc<Vec<Substitution>>>,
}

/// Borrowed state variable environment (current state's values array).
///
/// Safety: `ptr` must remain valid for the duration of any evaluation that uses it.
#[derive(Clone, Copy)]
pub(crate) struct StateEnvRef {
    ptr: *const Value,
    len: usize,
}

impl StateEnvRef {
    #[inline]
    fn from_slice(values: &[Value]) -> Self {
        StateEnvRef {
            ptr: values.as_ptr(),
            len: values.len(),
        }
    }

    #[inline(always)]
    unsafe fn get_unchecked<'a>(self, idx: usize) -> &'a Value {
        debug_assert!(idx < self.len);
        &*self.ptr.add(idx)
    }
}

impl EvalCtx {
    /// Create a new evaluation context
    pub fn new() -> Self {
        EvalCtx {
            shared: Arc::new(SharedCtx::new()),
            env: HashMap::new(),
            next_state: None,
            local_ops: None,
            local_stack: Vec::new(),
            state_env: None,
            next_state_env: None,
            recursion_depth: 0,
            instance_substitutions: None,
        }
    }

    /// Create context with variable bindings
    pub fn with_env(env: Env) -> Self {
        EvalCtx {
            shared: Arc::new(SharedCtx::new()),
            env,
            next_state: None,
            local_ops: None,
            local_stack: Vec::new(),
            state_env: None,
            next_state_env: None,
            recursion_depth: 0,
            instance_substitutions: None,
        }
    }

    /// Create context with next-state bindings for primed variable resolution
    pub fn with_next_state(&self, next_state: Env) -> Self {
        EvalCtx {
            shared: Arc::clone(&self.shared),
            env: self.env.clone(),
            next_state: Some(Arc::new(next_state)),
            local_ops: self.local_ops.clone(),
            local_stack: self.local_stack.clone(), // Preserve local bindings
            state_env: self.state_env,
            next_state_env: None, // Clear array-based next state when using HashMap
            recursion_depth: self.recursion_depth,
            instance_substitutions: self.instance_substitutions.clone(),
        }
    }

    /// Extend environment with a new binding
    pub fn bind(&self, name: impl Into<Arc<str>>, value: Value) -> Self {
        let name_arc = name.into();
        // If there's already a local_stack binding for this name, we need to
        // add the new binding to local_stack as well (not just env), because
        // lookup() checks local_stack first. Otherwise the old binding shadows
        // the new one - causing bugs like TCommit variable shadowing issue.
        //
        // IMPORTANT: Also check if the name is in local_ops - if so, we MUST add
        // to local_stack to shadow the local operator. Otherwise, when we look up
        // the identifier, local_ops is checked before env, and we'd find the
        // operator definition instead of the bound value. This caused infinite
        // recursion when a CHOOSE-bound variable had the same name as a LET definition.
        let mut new_stack = self.local_stack.clone();
        let needs_stack_shadow = new_stack.iter().any(|(n, _)| n == &name_arc)
            || self
                .local_ops
                .as_ref()
                .is_some_and(|ops| ops.contains_key(name_arc.as_ref()));
        if needs_stack_shadow {
            new_stack.push((Arc::clone(&name_arc), value.clone()));
        }
        EvalCtx {
            shared: Arc::clone(&self.shared),
            env: self.env.update(name_arc, value),
            next_state: self.next_state.clone(),
            local_ops: self.local_ops.clone(),
            local_stack: new_stack,
            state_env: self.state_env,
            next_state_env: self.next_state_env,
            recursion_depth: self.recursion_depth,
            instance_substitutions: self.instance_substitutions.clone(),
        }
    }

    /// Extend environment with a new local binding.
    ///
    /// Unlike `bind`, this always pushes onto `local_stack` so compiled expressions that
    /// reference locals by stack depth (e.g. `CompiledExpr::LocalVar`) can resolve the
    /// correct value.
    pub fn bind_local(&self, name: impl Into<Arc<str>>, value: Value) -> Self {
        let name_arc = name.into();
        let mut new_stack = self.local_stack.clone();
        new_stack.push((Arc::clone(&name_arc), value.clone()));
        EvalCtx {
            shared: Arc::clone(&self.shared),
            env: self.env.update(name_arc, value),
            next_state: self.next_state.clone(),
            local_ops: self.local_ops.clone(),
            local_stack: new_stack,
            state_env: self.state_env,
            next_state_env: self.next_state_env,
            recursion_depth: self.recursion_depth,
            instance_substitutions: self.instance_substitutions.clone(),
        }
    }

    /// Extend environment with multiple bindings
    /// Bindings are added to local_stack to properly shadow any existing bindings
    /// with the same name (e.g., when operator parameters shadow EXISTS-bound variables)
    pub fn bind_all(&self, bindings: impl IntoIterator<Item = (Arc<str>, Value)>) -> Self {
        let mut env = self.env.clone();
        let mut local_stack = self.local_stack.clone();
        for (name, value) in bindings {
            // Add to both env (for backward compat) and local_stack (for proper shadowing)
            env.insert(Arc::clone(&name), value.clone());
            local_stack.push((name, value));
        }
        EvalCtx {
            shared: Arc::clone(&self.shared),
            env,
            next_state: self.next_state.clone(),
            local_ops: self.local_ops.clone(),
            local_stack,
            state_env: self.state_env,
            next_state_env: self.next_state_env,
            recursion_depth: self.recursion_depth,
            instance_substitutions: self.instance_substitutions.clone(),
        }
    }

    /// Create a context with local operator definitions (for LET expressions)
    pub fn with_local_ops(&self, local_ops: OpEnv) -> Self {
        EvalCtx {
            shared: Arc::clone(&self.shared),
            env: self.env.clone(),
            next_state: self.next_state.clone(),
            local_ops: Some(Arc::new(local_ops)),
            local_stack: self.local_stack.clone(), // Preserve local bindings
            state_env: self.state_env,
            next_state_env: self.next_state_env,
            recursion_depth: self.recursion_depth,
            instance_substitutions: self.instance_substitutions.clone(),
        }
    }

    /// Create a context with INSTANCE substitutions active for this scope.
    pub fn with_instance_substitutions(&self, subs: Vec<Substitution>) -> Self {
        EvalCtx {
            shared: Arc::clone(&self.shared),
            env: self.env.clone(),
            next_state: self.next_state.clone(),
            local_ops: self.local_ops.clone(),
            local_stack: self.local_stack.clone(),
            state_env: self.state_env,
            next_state_env: self.next_state_env,
            recursion_depth: self.recursion_depth,
            instance_substitutions: Some(Arc::new(subs)),
        }
    }

    /// Get active INSTANCE substitutions for this scope.
    pub fn instance_substitutions(&self) -> Option<&[Substitution]> {
        self.instance_substitutions.as_ref().map(|v| v.as_slice())
    }

    /// Create a context without INSTANCE substitutions.
    /// Used when evaluating substitution expressions (the RHS of WITH clauses)
    /// which are written in the outer module's context.
    pub fn without_instance_substitutions(&self) -> Self {
        EvalCtx {
            shared: Arc::clone(&self.shared),
            env: self.env.clone(),
            next_state: self.next_state.clone(),
            local_ops: self.local_ops.clone(),
            local_stack: self.local_stack.clone(),
            state_env: self.state_env,
            next_state_env: self.next_state_env,
            recursion_depth: self.recursion_depth,
            instance_substitutions: None,
        }
    }

    /// Create a context without local operator definitions.
    ///
    /// This is primarily used when evaluating operators from an outer module while inside an
    /// INSTANCE scope. The instanced module's operator namespace must not shadow outer-module
    /// operator bodies (e.g., a local `Node` operator shadowing an outer constant `Node`).
    pub fn without_local_ops(&self) -> Self {
        EvalCtx {
            shared: Arc::clone(&self.shared),
            env: self.env.clone(),
            next_state: self.next_state.clone(),
            local_ops: None,
            local_stack: self.local_stack.clone(),
            state_env: self.state_env,
            next_state_env: self.next_state_env,
            recursion_depth: self.recursion_depth,
            instance_substitutions: self.instance_substitutions.clone(),
        }
    }

    /// Create a context with explicit environment bindings, clearing state_env.
    ///
    /// This is used for evaluating expressions in a specific state when
    /// the state_env pointer might be stale or pointing to a different state.
    /// The env HashMap is used for variable lookups instead of state_env.
    pub fn with_explicit_env(&self, env: Env) -> Self {
        EvalCtx {
            shared: Arc::clone(&self.shared),
            env,
            next_state: None,
            local_ops: self.local_ops.clone(),
            local_stack: self.local_stack.clone(),
            state_env: None, // Clear - use env HashMap instead
            next_state_env: None,
            recursion_depth: self.recursion_depth,
            instance_substitutions: self.instance_substitutions.clone(),
        }
    }

    /// Add an operator replacement (for config `CONSTANT Op <- Replacement`)
    pub fn add_op_replacement(&mut self, from: String, to: String) {
        Arc::make_mut(&mut self.shared)
            .op_replacements
            .insert(from, to);
    }

    /// Register a config-overridden constant.
    /// This marks the constant so that lookups check env bindings BEFORE operator definitions.
    pub fn add_config_constant(&mut self, name: String) {
        Arc::make_mut(&mut self.shared)
            .config_constants
            .insert(name);
    }

    /// Check if a name is a config-overridden constant.
    #[inline]
    pub fn is_config_constant(&self, name: &str) -> bool {
        self.shared.config_constants.contains(name)
    }

    /// Register a variable in the variable registry
    /// Returns the assigned VarIndex
    pub fn register_var(&mut self, name: impl Into<Arc<str>>) -> crate::var_index::VarIndex {
        Arc::make_mut(&mut self.shared).var_registry.register(name)
    }

    /// Register multiple variables in the variable registry
    /// Call this once at module load time with all state variables
    pub fn register_vars(&mut self, names: impl IntoIterator<Item = impl Into<Arc<str>>>) {
        let registry = &mut Arc::make_mut(&mut self.shared).var_registry;
        for name in names {
            registry.register(name);
        }
    }

    /// Get a reference to the variable registry
    #[inline]
    pub fn var_registry(&self) -> &VarRegistry {
        &self.shared.var_registry
    }

    /// Check if state_env is set (array-based fast path is active)
    #[inline]
    pub fn has_state_env(&self) -> bool {
        self.state_env.is_some()
    }

    /// Set the TLC configuration (for TLCGet("config"))
    pub fn set_tlc_config(&mut self, config: TlcConfig) {
        Arc::make_mut(&mut self.shared).tlc_config = config;
    }

    /// Get the index of a variable (if registered)
    #[inline]
    pub fn var_index(&self, name: &str) -> Option<crate::var_index::VarIndex> {
        self.shared.var_registry.get(name)
    }

    /// Resolve operator name through replacements (for config `CONSTANT Op <- Replacement`)
    ///
    /// This is critical for compiled_guard.rs to properly handle operator replacements
    /// when extracting next-state assignments from actions.
    pub fn resolve_op_name<'a>(&'a self, name: &'a str) -> &'a str {
        self.shared
            .op_replacements
            .get(name)
            .map(|s| s.as_str())
            .unwrap_or(name)
    }

    /// Add an operator definition
    pub fn define_op(&mut self, name: String, def: OperatorDef) {
        Arc::make_mut(&mut self.shared).ops.insert(name, def);
    }

    /// Look up a value in the environment
    /// Priority order:
    /// 1. local_stack (most recent bindings, e.g., quantified variables)
    /// 2. state_env array (O(1) access for state variables when set)
    /// 3. env HashMap (fallback)
    #[inline]
    pub fn lookup(&self, name: &str) -> Option<&Value> {
        // Check local stack first (reverse order for most recent)
        // This handles quantified variables that may shadow state variables
        // Skip this check if local_stack is empty (common in state access)
        if !self.local_stack.is_empty() {
            for (i, (n, v)) in self.local_stack.iter().enumerate().rev() {
                if n.as_ref() == name {
                    // Issue #70 fix: Always record local reads for dependency tracking
                    record_local_read(i, name, v);
                    if current_state_lookup_mode() == StateLookupMode::Next {
                        if let Some(idx) = self.shared.var_registry.get(name) {
                            record_next_read(idx, v);
                        }
                    }
                    return Some(v);
                }
            }
        }
        // If state_env is set, try O(1) array access for state variables
        if let Some(state_env) = self.state_env {
            if let Some(idx) = self.shared.var_registry.get(name) {
                // Safety: state_env is only set while the backing `[Value]` slice is alive.
                let v = unsafe { state_env.get_unchecked(idx.as_usize()) };
                record_state_read(idx, v);
                return Some(v);
            }
        }
        // Fall back to env HashMap
        let v = self.env.get(name);
        // Record dependency for state variables (fixes Issue #70: cache invalidation
        // when state_env is not set but vars are bound via bind_mut to env)
        if let (Some(v), Some(idx)) = (v, self.shared.var_registry.get(name)) {
            if current_state_lookup_mode() == StateLookupMode::Next {
                record_next_read(idx, v);
            } else {
                record_state_read(idx, v);
            }
        }
        v
    }

    /// Returns true if `name` currently has a local (stack) binding.
    ///
    /// This is primarily used to avoid misclassifying patterns like `x' = x` as
    /// UNCHANGED when `x` is a bound/quantified variable shadowing the state var `x`.
    #[inline]
    pub fn has_local_binding(&self, name: &str) -> bool {
        if self.local_stack.is_empty() {
            return false;
        }
        self.local_stack
            .iter()
            .rev()
            .any(|(n, _)| n.as_ref() == name)
    }

    /// Returns true if no local bindings (EXISTS bounds, operator params) are on the stack.
    #[inline]
    pub fn local_stack_is_empty(&self) -> bool {
        self.local_stack.is_empty()
    }

    /// Bind state variables from an ArrayState for O(1) access during evaluation.
    ///
    /// This sets `state_env` to point at the provided values slice, enabling fast
    /// state variable lookups via `VarIndex` instead of HashMap-based lookups.
    ///
    /// Returns the previous `state_env` so callers can restore it (supports nesting).
    #[inline]
    pub(crate) fn bind_state_array(&mut self, values: &[Value]) -> Option<StateEnvRef> {
        self.state_env.replace(StateEnvRef::from_slice(values))
    }

    /// Restore a previous `state_env` returned by `bind_state_array()`.
    #[inline]
    pub(crate) fn restore_state_env(&mut self, prev: Option<StateEnvRef>) {
        self.state_env = prev;
    }

    /// Bind next-state variables from an ArrayState for O(1) primed variable access.
    ///
    /// This sets `next_state_env` to point at the provided values slice, enabling fast
    /// primed variable lookups via `VarIndex` instead of HashMap-based lookups.
    ///
    /// Returns the previous `next_state_env` so callers can restore it (supports nesting).
    #[inline]
    pub(crate) fn bind_next_state_array(&mut self, values: &[Value]) -> Option<StateEnvRef> {
        self.next_state_env.replace(StateEnvRef::from_slice(values))
    }

    /// Restore a previous `next_state_env` returned by `bind_next_state_array()`.
    #[inline]
    pub(crate) fn restore_next_state_env(&mut self, prev: Option<StateEnvRef>) {
        self.next_state_env = prev;
    }

    /// Check if next_state_env is currently set (for fast path decisions).
    ///
    /// Used by Fallback evaluation to skip HashMap construction when array-based
    /// next-state lookups are already available.
    #[inline]
    pub(crate) fn has_next_state_env(&self) -> bool {
        self.next_state_env.is_some()
    }

    /// Push a binding to the local stack - O(1) operation
    /// Use with mark_stack() and pop_to_mark() for scoped bindings
    #[inline]
    pub fn push_binding(&mut self, name: Arc<str>, value: Value) {
        self.local_stack.push((name, value));
    }

    /// Get current stack position for later restoration - O(1)
    #[inline]
    pub fn mark_stack(&self) -> usize {
        self.local_stack.len()
    }

    /// Pop all bindings back to a marked position - O(k) where k is bindings removed
    #[inline]
    pub fn pop_to_mark(&mut self, mark: usize) {
        self.local_stack.truncate(mark);
    }

    /// Debug: return a string representation of current stack bindings
    pub fn stack_bindings_debug(&self) -> String {
        if self.local_stack.is_empty() {
            return "[]".to_string();
        }
        let bindings: Vec<String> = self
            .local_stack
            .iter()
            .map(|(name, val)| format!("{}={}", name, val))
            .collect();
        format!("[{}]", bindings.join(", "))
    }

    /// Get local variable by depth - O(1) access
    ///
    /// depth=0 is the most recent binding, depth=1 is one before that, etc.
    /// Returns None if depth exceeds stack size.
    #[inline]
    pub fn get_local_by_depth(&self, depth: u8) -> Option<&Value> {
        let len = self.local_stack.len();
        if (depth as usize) < len {
            Some(&self.local_stack[len - 1 - depth as usize].1)
        } else {
            None
        }
    }

    /// Get a reference to the current local bindings (EXISTS bounds, operator params).
    /// Used to capture bindings alongside expressions for deferred evaluation.
    #[inline]
    pub fn get_local_bindings(&self) -> &[(Arc<str>, Value)] {
        &self.local_stack
    }

    /// Create a new context with the given captured bindings restored.
    /// Used to evaluate expressions captured during enumeration.
    pub fn with_captured_bindings(&self, bindings: &[(Arc<str>, Value)]) -> Self {
        EvalCtx {
            shared: Arc::clone(&self.shared),
            env: self.env.clone(),
            next_state: self.next_state.clone(),
            local_ops: self.local_ops.clone(),
            local_stack: bindings.to_vec(),
            state_env: self.state_env,
            next_state_env: self.next_state_env,
            recursion_depth: self.recursion_depth,
            instance_substitutions: self.instance_substitutions.clone(),
        }
    }

    /// Look up an operator by name
    /// First checks local_ops (from LET), then shared ops
    pub fn get_op(&self, name: &str) -> Option<&OperatorDef> {
        // Check local ops first (for LET expressions)
        if let Some(ref local) = self.local_ops {
            if let Some(def) = local.get(name) {
                return Some(def);
            }
        }
        // Fall back to shared ops
        self.shared.ops.get(name)
    }

    /// Register a named INSTANCE declaration
    ///
    /// For `InChan == INSTANCE Channel WITH Data <- Message, chan <- in`:
    /// - instance_name = "InChan"
    /// - module_name = "Channel"
    /// - substitutions = [(Data, Message), (chan, in)]
    pub fn register_instance(&mut self, instance_name: String, info: InstanceInfo) {
        Arc::make_mut(&mut self.shared)
            .instances
            .insert(instance_name, info);
    }

    /// Load operators from an instanced module
    ///
    /// Stores the operators separately so they can be looked up by instance reference
    pub fn load_instance_module(&mut self, module_name: String, module: &Module) {
        let mut ops = HashMap::new();
        for unit in &module.units {
            match &unit.node {
                tla_core::ast::Unit::Operator(def) => {
                    ops.insert(def.name.node.clone(), def.clone());
                }
                tla_core::ast::Unit::Theorem(thm) => {
                    // Named theorems are accessible as zero-argument operators
                    if let Some(name) = &thm.name {
                        let def = OperatorDef {
                            name: name.clone(),
                            params: vec![],
                            body: thm.body.clone(),
                            local: false,
                        };
                        ops.insert(name.node.clone(), def);
                    }
                }
                _ => {}
            }
        }
        Arc::make_mut(&mut self.shared)
            .instance_ops
            .insert(module_name, ops);
    }

    /// Load operator definitions from a module
    ///
    /// This also detects named instances (operators with InstanceExpr body) and
    /// registers them in the instances map. To actually use those instances,
    /// you must also call `load_instance_module` for each instanced module.
    pub fn load_module(&mut self, module: &Module) {
        let shared = Arc::make_mut(&mut self.shared);
        for unit in &module.units {
            match &unit.node {
                tla_core::ast::Unit::Operator(def) => {
                    // Check if this is a named instance: InChan == INSTANCE Channel WITH ...
                    if let Expr::InstanceExpr(module_name, substitutions) = &def.body.node {
                        // Register the instance
                        let instance_name = def.name.node.clone();
                        shared.instances.insert(
                            instance_name,
                            InstanceInfo {
                                module_name: module_name.clone(),
                                substitutions: substitutions.clone(),
                            },
                        );
                    } else {
                        // Regular operator
                        shared.ops.insert(def.name.node.clone(), def.clone());
                    }
                }
                tla_core::ast::Unit::Theorem(thm) => {
                    // Named theorems are accessible as zero-argument operators
                    // e.g., THEOREM QuorumNonEmpty == \A Q \in Quorum : Q # {}
                    // can be referenced as QuorumNonEmpty in ASSUME statements
                    if let Some(name) = &thm.name {
                        let def = OperatorDef {
                            name: name.clone(),
                            params: vec![],
                            body: thm.body.clone(),
                            local: false,
                        };
                        shared.ops.insert(name.node.clone(), def);
                    }
                }
                tla_core::ast::Unit::Assume(assume) => {
                    // Named assumes are accessible as zero-argument operators
                    // e.g., ASSUME PaxosAssume == /\ IsFiniteSet(Replicas) /\ ...
                    // can be referenced as PaxosAssume elsewhere
                    if let Some(name) = &assume.name {
                        let def = OperatorDef {
                            name: name.clone(),
                            params: vec![],
                            body: assume.expr.clone(),
                            local: false,
                        };
                        shared.ops.insert(name.node.clone(), def);
                    }
                }
                _ => {}
            }
        }
    }

    /// Load operator definitions from multiple modules (e.g., extended modules)
    ///
    /// Modules should be provided in dependency order (extended modules first, main module last).
    /// Later modules can override earlier definitions.
    pub fn load_modules(&mut self, modules: &[&Module]) {
        for module in modules {
            self.load_module(module);
        }
    }

    /// Get operator from an instanced module
    ///
    /// Looks up the operator in instance_ops only. For this to work, you must
    /// call `load_instance_module` for each module used in named instances.
    pub fn get_instance_op(&self, module_name: &str, op_name: &str) -> Option<&OperatorDef> {
        self.shared
            .instance_ops
            .get(module_name)
            .and_then(|ops| ops.get(op_name))
    }

    /// Get instance info by name
    pub fn get_instance(&self, name: &str) -> Option<&InstanceInfo> {
        self.shared.instances.get(name)
    }

    /// Check if an operator is defined
    pub fn has_op(&self, name: &str) -> bool {
        if let Some(ref local) = self.local_ops {
            if local.contains_key(name) {
                return true;
            }
        }
        self.shared.ops.contains_key(name)
    }

    /// Evaluate a named operator (with no arguments)
    pub fn eval_op(&self, name: &str) -> EvalResult<Value> {
        let def = self.get_op(name).ok_or_else(|| EvalError::UndefinedOp {
            name: name.to_string(),
            span: None,
        })?;

        // For a zero-argument operator, just evaluate its body
        if !def.params.is_empty() {
            return Err(EvalError::ArityMismatch {
                op: name.to_string(),
                expected: def.params.len(),
                got: 0,
                span: None,
            });
        }

        eval(self, &def.body)
    }

    /// Bind a variable mutably in the current context
    pub fn bind_mut(&mut self, name: impl Into<Arc<str>>, value: Value) {
        self.env.insert(name.into(), value);
    }

    /// Push a scope (for nested evaluation) - returns saved env for later restore
    pub fn save_scope(&self) -> Env {
        self.env.clone()
    }

    /// Restore a saved scope
    pub fn restore_scope(&mut self, saved: Env) {
        self.env = saved;
    }

    // Convenience accessors for shared context fields (read-only)

    /// Get reference to shared ops
    #[inline]
    pub fn ops(&self) -> &OpEnv {
        &self.shared.ops
    }

    /// Get reference to shared instances
    #[inline]
    pub fn instances(&self) -> &HashMap<String, InstanceInfo> {
        &self.shared.instances
    }

    /// Get reference to shared instance_ops
    #[inline]
    pub fn instance_ops(&self) -> &HashMap<String, OpEnv> {
        &self.shared.instance_ops
    }

    /// Get reference to shared op_replacements
    #[inline]
    pub fn op_replacements(&self) -> &HashMap<String, String> {
        &self.shared.op_replacements
    }
}

impl Default for EvalCtx {
    fn default() -> Self {
        Self::new()
    }
}

fn func_in_func_set(
    ctx: &EvalCtx,
    func: &FuncValue,
    domain_expr: &Spanned<Expr>,
    range_expr: &Spanned<Expr>,
) -> EvalResult<bool> {
    let dv = eval(ctx, domain_expr)?;

    // Handle both Set and Interval domains
    let domain = dv
        .to_ord_set()
        .ok_or_else(|| EvalError::type_error("Set", &dv, Some(domain_expr.span)))?;

    if func.domain_as_ord_set() != domain {
        return Ok(false);
    }
    if func.domain_len() != domain.len() {
        return Ok(false);
    }

    // If the range doesn't require lazy handling, evaluate it once and then do fast lookups.
    // Otherwise, defer to eval_membership_lazy so we don't eagerly enumerate huge sets like SUBSET S.
    let range_val = if is_lazy_membership_expr(&range_expr.node) {
        None
    } else {
        Some(eval(ctx, range_expr)?)
    };

    for d in domain.iter() {
        let v = match func.mapping_get(d) {
            Some(v) => v,
            None => return Ok(false),
        };

        let in_range = match &range_val {
            Some(Value::Set(range)) => range.contains(v),
            Some(Value::Interval(iv)) => iv.contains(v),
            Some(Value::ModelValue(name)) => match name.as_ref() {
                "Nat" => match v {
                    Value::SmallInt(n) => *n >= 0,
                    Value::Int(n) => *n >= BigInt::zero(),
                    _ => false,
                },
                "Int" => matches!(v, Value::SmallInt(_) | Value::Int(_)),
                "Real" => matches!(v, Value::SmallInt(_) | Value::Int(_)), // Int ⊆ Real, TLC doesn't support actual reals
                _ => {
                    return Err(EvalError::type_error(
                        "Set",
                        range_val.as_ref().unwrap(),
                        Some(range_expr.span),
                    ));
                }
            },
            // Handle SetPred: check membership via predicate evaluation
            Some(Value::SetPred(spv)) => {
                check_set_pred_membership(ctx, v, spv, Some(range_expr.span))?
            }
            // Handle lazy values that support set_contains
            Some(other) if other.set_contains(v).is_some() => other.set_contains(v).unwrap(),
            Some(other) => {
                return Err(EvalError::type_error("Set", other, Some(range_expr.span)));
            }
            None => eval_membership_lazy(ctx, v.clone(), range_expr)?,
        };

        if !in_range {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Check if an expression requires lazy membership checking (FuncSet, Powerset, Seq, RecordSet)
fn is_lazy_membership_expr(expr: &Expr) -> bool {
    match expr {
        Expr::FuncSet(_, _) | Expr::Powerset(_) | Expr::RecordSet(_) => true,
        // Union: can check membership lazily as (x \in A) \/ (x \in B)
        Expr::Union(_, _) => true,
        Expr::Apply(op, args) => {
            // Check for Seq(S) pattern
            if let Expr::Ident(name) = &op.node {
                name == "Seq" && args.len() == 1
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Check if an expression (potentially through Ident resolution) requires lazy membership checking
fn needs_lazy_membership(ctx: &EvalCtx, expr: &Spanned<Expr>) -> bool {
    if is_lazy_membership_expr(&expr.node) {
        return true;
    }
    // Also check if expr is an Ident that resolves to a lazy membership expression
    if let Expr::Ident(name) = &expr.node {
        let resolved_name = ctx.resolve_op_name(name);
        if let Some(def) = ctx.get_op(resolved_name) {
            if def.params.is_empty() && is_lazy_membership_expr(&def.body.node) {
                return true;
            }
        }
    }
    false
}

/// Check if a set is a valid sequence domain (i.e., {1, 2, ..., n} for some n >= 0)
fn is_sequence_domain(domain: &OrdSet<Value>) -> bool {
    if domain.is_empty() {
        return true; // Empty sequence has empty domain
    }

    // Check that all elements are consecutive positive integers starting from 1
    let mut expected: i64 = 1;
    for val in domain.iter() {
        match val {
            Value::SmallInt(n) if *n == expected => {
                expected += 1;
            }
            Value::Int(n) if *n == BigInt::from(expected) => {
                expected += 1;
            }
            _ => return false,
        }
    }
    true
}

/// Check if a value is a member of a set expression, with lazy handling for SUBSET, [S -> T], Seq(S), and RecordSet
/// This avoids eager enumeration of large/infinite sets
fn eval_membership_lazy(ctx: &EvalCtx, value: Value, set_expr: &Spanned<Expr>) -> EvalResult<bool> {
    // If the expression is an identifier, resolve it and recursively check if the underlying
    // expression is a lazy membership expression (RecordSet with infinite fields, etc.)
    if let Expr::Ident(name) = &set_expr.node {
        let resolved_name = ctx.resolve_op_name(name);
        if let Some(def) = ctx.get_op(resolved_name) {
            if def.params.is_empty() && is_lazy_membership_expr(&def.body.node) {
                return eval_membership_lazy(ctx, value, &def.body);
            }
        }
    }

    // Handle SUBSET lazily: v \in SUBSET S <==> v is a set AND v \subseteq S
    if let Expr::Powerset(inner) = &set_expr.node {
        // Handle both Set and Interval values
        if let Some(iter) = value.iter_set() {
            // Check if every element of the set is \in inner (recursively)
            for elem in iter {
                if !eval_membership_lazy(ctx, elem, inner)? {
                    return Ok(false);
                }
            }
            return Ok(true);
        }
        // Value is not a set-like type
        return Ok(false);
    }

    // Handle Union lazily: v \in (A \cup B) <==> (v \in A) \/ (v \in B)
    // This is critical for efficient type checking in specs like MultiPaxos where
    // Messages = PrepareMsgs \cup PrepareReplyMsgs \cup AcceptMsgs \cup ...
    // Without lazy union, we'd compute the full Cartesian product of all message types.
    if let Expr::Union(left, right) = &set_expr.node {
        // Short-circuit: if v \in left, we're done
        if eval_membership_lazy(ctx, value.clone(), left)? {
            return Ok(true);
        }
        // Otherwise check right side
        return eval_membership_lazy(ctx, value, right);
    }

    // Handle [S -> T] lazily: v \in [S -> T] <==> v is a function with domain S and range in T
    if let Expr::FuncSet(domain_expr, range_expr) = &set_expr.node {
        match &value {
            Value::Func(f) => return func_in_func_set(ctx, f, domain_expr, range_expr),
            // IntFunc is an array-backed function with integer interval domain
            Value::IntFunc(f) => {
                // Check domain: function set domain must equal min..max
                let expected_domain: OrdSet<Value> = (f.min..=f.max).map(Value::SmallInt).collect();
                let domain_val = eval(ctx, domain_expr)?;
                let actual_domain = domain_val.to_ord_set().ok_or_else(|| {
                    EvalError::type_error("Set", &domain_val, Some(domain_expr.span))
                })?;
                if actual_domain != expected_domain {
                    return Ok(false);
                }
                // Check range: all values must be in range set
                for val in f.values.iter() {
                    if !eval_membership_lazy(ctx, val.clone(), range_expr)? {
                        return Ok(false);
                    }
                }
                return Ok(true);
            }
            // Tuples/Seqs are functions with domain 1..n
            Value::Tuple(elems) => {
                // Check domain: expected is 1..n
                let expected_domain: OrdSet<Value> = if elems.is_empty() {
                    OrdSet::new()
                } else {
                    (1..=elems.len())
                        .map(|i| Value::SmallInt(i as i64))
                        .collect()
                };
                let domain_val = eval(ctx, domain_expr)?;
                let actual_domain = domain_val.to_ord_set().ok_or_else(|| {
                    EvalError::type_error("Set", &domain_val, Some(domain_expr.span))
                })?;
                if actual_domain != expected_domain {
                    return Ok(false);
                }
                // Check range: all elements must be in range set
                for elem in elems.iter() {
                    if !eval_membership_lazy(ctx, elem.clone(), range_expr)? {
                        return Ok(false);
                    }
                }
                return Ok(true);
            }
            Value::Seq(seq) => {
                // Check domain: expected is 1..n
                let expected_domain: OrdSet<Value> = if seq.is_empty() {
                    OrdSet::new()
                } else {
                    (1..=seq.len()).map(|i| Value::SmallInt(i as i64)).collect()
                };
                let domain_val = eval(ctx, domain_expr)?;
                let actual_domain = domain_val.to_ord_set().ok_or_else(|| {
                    EvalError::type_error("Set", &domain_val, Some(domain_expr.span))
                })?;
                if actual_domain != expected_domain {
                    return Ok(false);
                }
                // Check range: all elements must be in range set
                for elem in seq.iter() {
                    if !eval_membership_lazy(ctx, elem.clone(), range_expr)? {
                        return Ok(false);
                    }
                }
                return Ok(true);
            }
            _ => return Ok(false),
        }
    }

    // Handle Seq(S) lazily: v \in Seq(S) <==> v is a sequence AND all elements are in S
    // Seq(S) is represented as Apply(Ident("Seq"), [S])
    if let Expr::Apply(op, args) = &set_expr.node {
        if let Expr::Ident(name) = &op.node {
            if name == "Seq" && args.len() == 1 {
                let elem_set_expr = &args[0];
                // Check if value is a sequence/tuple and all elements are in S
                if let Some(elems) = value.as_seq_or_tuple_elements() {
                    for elem in elems.iter() {
                        if !eval_membership_lazy(ctx, elem.clone(), elem_set_expr)? {
                            return Ok(false);
                        }
                    }
                    return Ok(true);
                }
                return match &value {
                    // TLA+ treats functions 1..n -> T as sequences
                    Value::Func(f) => {
                        // Check if domain is 1..n for some n
                        if !is_sequence_domain(&f.domain_as_ord_set()) {
                            return Ok(false);
                        }
                        // Check all values are in S
                        for v in f.mapping_values() {
                            if !eval_membership_lazy(ctx, v.clone(), elem_set_expr)? {
                                return Ok(false);
                            }
                        }
                        Ok(true)
                    }
                    // IntFunc with domain 1..n is also a sequence
                    Value::IntFunc(f) => {
                        // Check if domain is 1..n for some n
                        if f.min != 1 {
                            return Ok(false);
                        }
                        // Check all values are in S
                        for v in f.values.iter() {
                            if !eval_membership_lazy(ctx, v.clone(), elem_set_expr)? {
                                return Ok(false);
                            }
                        }
                        Ok(true)
                    }
                    _ => Ok(false),
                };
            }
        }
    }

    // Handle RecordSet lazily: v \in [f1: S1, f2: S2, ...] <==> v is a record with exactly those fields AND v.f1 \in S1 AND v.f2 \in S2 AND ...
    if let Expr::RecordSet(fields) = &set_expr.node {
        match &value {
            Value::Record(rec) => {
                // Check that record has exactly the same fields
                if rec.len() != fields.len() {
                    return Ok(false);
                }
                // Check each field
                for (field_name, field_set_expr) in fields {
                    match rec.get(field_name.node.as_str()) {
                        Some(field_val) => {
                            if !eval_membership_lazy(ctx, field_val.clone(), field_set_expr)? {
                                return Ok(false);
                            }
                        }
                        None => return Ok(false), // Record doesn't have this field
                    }
                }
                return Ok(true);
            }
            _ => return Ok(false),
        }
    }

    // For other expressions, evaluate eagerly and check membership
    let set_val = eval(ctx, set_expr)?;

    // Handle ModelValue for infinite sets (Nat, Int, Real)
    if let Value::ModelValue(name) = &set_val {
        return match name.as_ref() {
            "Nat" => match &value {
                Value::SmallInt(n) => Ok(*n >= 0),
                Value::Int(n) => Ok(*n >= BigInt::zero()),
                _ => Ok(false),
            },
            "Int" => Ok(matches!(&value, Value::SmallInt(_) | Value::Int(_))),
            "Real" => Ok(matches!(&value, Value::SmallInt(_) | Value::Int(_))), // Int ⊆ Real
            _ => Err(EvalError::type_error("Set", &set_val, Some(set_expr.span))),
        };
    }

    // Handle both Set and Interval using set_contains
    let contains = set_val
        .set_contains(&value)
        .ok_or_else(|| EvalError::type_error("Set", &set_val, Some(set_expr.span)))?;
    Ok(contains)
}

/// Check membership in a SetPred value: v \in {x \in S : P(x)}
///
/// This is true iff: v \in S AND P(v) is TRUE
///
/// The predicate P is evaluated with the bound variable bound to v,
/// using the SetPred's captured environment merged with the current context.
fn check_set_pred_membership(
    ctx: &EvalCtx,
    value: &Value,
    spv: &SetPredValue,
    span: Option<Span>,
) -> EvalResult<bool> {
    // First check: is value in the source set?
    let in_source = if let Some(contains) = spv.source.set_contains(value) {
        contains
    } else if let Value::SetPred(inner_spv) = spv.source.as_ref() {
        // Nested SetPred - recursive check
        check_set_pred_membership(ctx, value, inner_spv, span)?
    } else {
        // Can't check source membership - this shouldn't happen for valid SetPred
        return Err(EvalError::Internal {
            message: "SetPred source doesn't support membership check".into(),
            span,
        });
    };

    if !in_source {
        return Ok(false);
    }

    // Second check: does the predicate hold for this value?
    // Create evaluation context with SetPred's captured env and current ctx's shared context
    let mut env = spv.env.clone();
    // Also include current context's bindings (but SetPred's env takes precedence for captures)
    for (k, v) in ctx.env.iter() {
        if !env.contains_key(k.as_ref()) {
            env.insert(Arc::clone(k), v.clone());
        }
    }

    let pred_ctx = EvalCtx {
        shared: ctx.shared.clone(),
        env,
        next_state: ctx.next_state.clone(),
        local_ops: ctx.local_ops.clone(),
        local_stack: Vec::new(),
        state_env: ctx.state_env,
        next_state_env: ctx.next_state_env,
        recursion_depth: ctx.recursion_depth,
        instance_substitutions: ctx.instance_substitutions.clone(),
    };

    // Bind the variable to the value being tested
    let bound_ctx = bind_bound_var(&pred_ctx, &spv.bound, value, span)?;

    // Evaluate the predicate
    let pred_result = eval(&bound_ctx, &spv.pred)?;
    let pred_bool = pred_result
        .as_bool()
        .ok_or_else(|| EvalError::type_error("BOOLEAN", &pred_result, Some(spv.pred.span)))?;

    Ok(pred_bool)
}

/// Check if an expression mentions a specific operator name (identifier).
/// Used to detect self-referential/recursive function definitions.
fn expr_mentions_op(expr: &Spanned<Expr>, target: &str) -> bool {
    match &expr.node {
        Expr::Bool(_) | Expr::Int(_) | Expr::String(_) | Expr::OpRef(_) => false,
        Expr::Ident(name) => name == target,
        Expr::Apply(op, args) => {
            expr_mentions_op(op, target) || args.iter().any(|a| expr_mentions_op(a, target))
        }
        Expr::ModuleRef(_, _, args) => args.iter().any(|a| expr_mentions_op(a, target)),
        Expr::InstanceExpr(_, subs) => subs.iter().any(|s| expr_mentions_op(&s.to, target)),
        Expr::Lambda(_params, body) => expr_mentions_op(body, target),

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
        | Expr::FuncSet(a, b)
        | Expr::LeadsTo(a, b)
        | Expr::WeakFair(a, b)
        | Expr::StrongFair(a, b) => expr_mentions_op(a, target) || expr_mentions_op(b, target),

        Expr::Not(a)
        | Expr::Neg(a)
        | Expr::Powerset(a)
        | Expr::BigUnion(a)
        | Expr::Domain(a)
        | Expr::Prime(a)
        | Expr::Always(a)
        | Expr::Eventually(a)
        | Expr::Enabled(a)
        | Expr::Unchanged(a)
        | Expr::RecordAccess(a, _) => expr_mentions_op(a, target),

        Expr::Forall(bounds, body) | Expr::Exists(bounds, body) => {
            bounds.iter().any(|b| {
                b.domain
                    .as_ref()
                    .is_some_and(|d| expr_mentions_op(d, target))
            }) || expr_mentions_op(body, target)
        }
        Expr::Choose(bound, body) => {
            bound
                .domain
                .as_ref()
                .is_some_and(|d| expr_mentions_op(d, target))
                || expr_mentions_op(body, target)
        }

        Expr::SetEnum(elems) | Expr::Tuple(elems) | Expr::Times(elems) => {
            elems.iter().any(|e| expr_mentions_op(e, target))
        }
        Expr::SetBuilder(expr, bounds) => {
            expr_mentions_op(expr, target)
                || bounds.iter().any(|b| {
                    b.domain
                        .as_ref()
                        .is_some_and(|d| expr_mentions_op(d, target))
                })
        }
        Expr::SetFilter(bound, pred) => {
            bound
                .domain
                .as_ref()
                .is_some_and(|d| expr_mentions_op(d, target))
                || expr_mentions_op(pred, target)
        }

        Expr::FuncDef(bounds, body) => {
            bounds.iter().any(|b| {
                b.domain
                    .as_ref()
                    .is_some_and(|d| expr_mentions_op(d, target))
            }) || expr_mentions_op(body, target)
        }
        Expr::FuncApply(f, arg) => expr_mentions_op(f, target) || expr_mentions_op(arg, target),
        Expr::Except(f, specs) => {
            expr_mentions_op(f, target)
                || specs.iter().any(|s| {
                    s.path.iter().any(|p| match p {
                        tla_core::ast::ExceptPathElement::Index(idx) => {
                            expr_mentions_op(idx, target)
                        }
                        tla_core::ast::ExceptPathElement::Field(_) => false,
                    }) || expr_mentions_op(&s.value, target)
                })
        }

        Expr::Record(fields) => fields
            .iter()
            .any(|(_name, expr)| expr_mentions_op(expr, target)),
        Expr::RecordSet(fields) => fields
            .iter()
            .any(|(_name, expr)| expr_mentions_op(expr, target)),

        Expr::If(c, t, e) => {
            expr_mentions_op(c, target)
                || expr_mentions_op(t, target)
                || expr_mentions_op(e, target)
        }
        Expr::Case(arms, other) => {
            arms.iter()
                .any(|a| expr_mentions_op(&a.guard, target) || expr_mentions_op(&a.body, target))
                || other.as_ref().is_some_and(|d| expr_mentions_op(d, target))
        }

        Expr::Let(defs, body) => {
            defs.iter().any(|d| expr_mentions_op(&d.body, target)) || expr_mentions_op(body, target)
        }
    }
}

/// Red zone size: when stack has less than this remaining, grow.
///
/// This must be large enough to handle match arms that allocate sizable stack frames.
/// If the red zone is too small, a single large frame allocation can jump across the
/// guard page before `stacker::maybe_grow` gets a chance to switch stacks.
const STACK_RED_ZONE: usize = 1024 * 1024; // 1MB red zone
/// Stack growth size: how much to grow when we hit the red zone
const STACK_GROW_SIZE: usize = 16 * 1024 * 1024; // 16MB growth - plenty of room

/// Check if an error indicates a disabled action in TLC semantics.
///
/// TLC treats certain runtime errors (NotInDomain, IndexOutOfBounds, etc.) as
/// meaning the action is disabled rather than propagating the error. This is
/// used in OR evaluation to allow one branch to fail while the other succeeds.
fn is_action_disabling_error(err: &EvalError) -> bool {
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

/// Evaluate an expression in the given context
pub fn eval(ctx: &EvalCtx, expr: &Spanned<Expr>) -> EvalResult<Value> {
    // Use stacker::maybe_grow to grow stack on demand.
    // The eval_expr function has a huge match statement with large stack frames
    // per arm, so even modest recursion depth can overflow the stack.
    stacker::maybe_grow(STACK_RED_ZONE, STACK_GROW_SIZE, || {
        eval_inner(ctx, expr)
    })
}

/// Inner eval implementation (wrapped by stacker in eval())
fn eval_inner(ctx: &EvalCtx, expr: &Spanned<Expr>) -> EvalResult<Value> {
    // Track eval recursion depth to prevent stack overflow
    let depth = EVAL_DEPTH.with(|d| {
        let current = d.get();
        d.set(current + 1);
        current + 1
    });

    // Clear per-eval caches at the start of a top-level evaluation.
    if depth == 1 {
        SUBST_CACHE.with(|c| c.borrow_mut().clear());
        // Periodically clear operator caches to prevent unbounded growth.
        OP_RESULT_CACHE.with(|cache| {
            let mut c = cache.borrow_mut();
            if c.len() > 10000 {
                c.clear();
            }
        });
        ZERO_ARG_OP_CACHE.with(|cache| {
            let mut c = cache.borrow_mut();
            if c.len() > 10000 {
                c.clear();
            }
        });
    }

    // Check depth limit before recursing - use a lower check threshold
    // to catch deep recursion before we hit actual stack overflow
    if depth > MAX_EVAL_DEPTH {
        EVAL_DEPTH.with(|d| d.set(d.get() - 1));
        return Err(EvalError::Internal {
            message: format!(
                "Maximum evaluation depth ({}) exceeded. This may indicate infinite \
                 recursion or an overly complex expression.",
                MAX_EVAL_DEPTH
            ),
            span: Some(expr.span),
        });
    }

    let result = eval_expr(ctx, &expr.node, Some(expr.span));

    // Restore depth on exit
    EVAL_DEPTH.with(|d| {
        let current = d.get();
        d.set(current - 1);
        if current == 1 {
            // Clear SUBST_CACHE after the top-level call to avoid retaining large Values.
            // OP_RESULT_CACHE uses generation-based invalidation, so no clearing needed here.
            SUBST_CACHE.with(|c| c.borrow_mut().clear());
        }
    });

    result
}

/// Evaluate an expression with optional span for error reporting
fn eval_expr(ctx: &EvalCtx, expr: &Expr, span: Option<Span>) -> EvalResult<Value> {
    match expr {
        // === Literals ===
        Expr::Bool(b) => Ok(Value::Bool(*b)),
        Expr::Int(n) => Ok(Value::big_int(n.clone())),
        Expr::String(s) => Ok(Value::String(intern_string(s.as_str()))),

        // === Variables and zero-arg operators ===
        Expr::Ident(name) => {
            // 1) Check bound variables (local_stack) first.
            //
            // This intentionally does NOT consult state/env bindings yet: INSTANCE substitutions
            // must be able to override same-named outer variables.
            if !ctx.local_stack.is_empty() {
                for (i, (n, v)) in ctx.local_stack.iter().enumerate().rev() {
                    if n.as_ref() == name {
                        // Issue #70 fix: Always record local reads for dependency tracking,
                        // regardless of state lookup mode. The mode only affects how STATE
                        // variables are treated (current vs next), not local bindings.
                        record_local_read(i, name, v);
                        // Additionally, if this is a state variable in Next mode, record as next_read
                        if current_state_lookup_mode() == StateLookupMode::Next {
                            if let Some(idx) = ctx.var_registry().get(name) {
                                record_next_read(idx, v);
                            }
                        }
                        return Ok(v.clone());
                    }
                }
            }

            // Apply operator replacement if configured (e.g., NumActors <- n).
            let resolved_name = ctx.resolve_op_name(name);

            // Replacement target may be bound (e.g., EXISTS n \in ... : NumActors = n).
            if resolved_name != name && !ctx.local_stack.is_empty() {
                for (i, (n, v)) in ctx.local_stack.iter().enumerate().rev() {
                    if n.as_ref() == resolved_name {
                        // Issue #70 fix: Always record local reads for dependency tracking
                        record_local_read(i, resolved_name, v);
                        if current_state_lookup_mode() == StateLookupMode::Next {
                            if let Some(idx) = ctx.var_registry().get(resolved_name) {
                                record_next_read(idx, v);
                            }
                        }
                        return Ok(v.clone());
                    }
                }
            }

            // 2) Apply active INSTANCE substitutions (TLC composes these through nested instances).
            //
            // This must happen BEFORE checking state/env bindings so that an instanced module's
            // variables/constants can be mapped to expressions (e.g., active <- Node2Nat(active)).
            if let Some(subs) = ctx.instance_substitutions() {
                if let Some(sub) = subs.iter().find(|s| s.from.node == resolved_name) {
                    let cache_key = (ctx.next_state.is_some(), resolved_name.to_string());
                    if let Some(cached) = SUBST_CACHE.with(|c| c.borrow().get(&cache_key).cloned())
                    {
                        return Ok(cached);
                    }

                    // Substitution RHS is written in the OUTER module's context, so do not
                    // apply the current INSTANCE substitutions while evaluating it.
                    let outer_ctx = ctx.without_instance_substitutions();
                    let value = eval(&outer_ctx, &sub.to)?;
                    SUBST_CACHE.with(|c| {
                        c.borrow_mut().insert(cache_key, value.clone());
                    });
                    return Ok(value);
                }
            }

            // 3) Check local (in-scope) zero-argument operators next.
            //
            // IMPORTANT: When evaluating an operator from the *outer* module while we're inside an
            // INSTANCE scope, instance substitutions must NOT apply to that outer operator body.
            // Only operators from the instanced module (local_ops) should see substitutions.
            if let Some(def) = ctx
                .local_ops
                .as_ref()
                .and_then(|local| local.get(resolved_name))
            {
                if def.params.is_empty() {
                    // Handle module-level recursive function definitions:
                    // nat2node[i \in S] == ... nat2node[i-1] ...
                    // These are lowered to: nat2node == [i \in S |-> ... nat2node[i-1] ...]
                    // Without special handling, evaluating the FuncDef eagerly tries to build
                    // the full mapping, which re-enters nat2node and causes 0 states or stack overflow.
                    if let Expr::FuncDef(bounds, func_body) = &def.body.node {
                        if expr_mentions_op(func_body, resolved_name) {
                            // Issue #100 fix: Before creating a new LazyFunc, check if one already
                            // exists in env. This happens when:
                            //   LET S == {0, 1}
                            //       Max[T \in SUBSET S] == ... Max[T \ {n}] ...
                            //   IN Max[S]
                            // The outer LET adds Max to both local_ops (as an operator def) and
                            // env (as a LazyFunc). The inner recursive call should use the LazyFunc
                            // from env to share memoization, not create a new one from local_ops.
                            if let Some(existing) = ctx.env.get(resolved_name) {
                                if matches!(existing, Value::LazyFunc(_)) {
                                    return Ok(existing.clone());
                                }
                            }
                            // Create a LazyFunc for recursive function definitions
                            let domain_val = if bounds.len() == 1 {
                                let domain_expr =
                                    bounds[0].domain.as_ref().ok_or_else(|| EvalError::Internal {
                                        message: "Function definition requires bounded variable"
                                            .into(),
                                        span,
                                    })?;
                                eval(ctx, domain_expr)?
                            } else {
                                let mut components = Vec::with_capacity(bounds.len());
                                for b in bounds {
                                    let domain_expr = b.domain.as_ref().ok_or_else(|| {
                                        EvalError::Internal {
                                            message:
                                                "Function definition requires bounded variable"
                                                    .into(),
                                            span,
                                        }
                                    })?;
                                    components.push(eval(ctx, domain_expr)?);
                                }
                                Value::TupleSet(TupleSetValue::new(components))
                            };

                            if !domain_val.is_set() {
                                return Err(EvalError::type_error(
                                    "Set",
                                    &domain_val,
                                    Some(def.body.span),
                                ));
                            }

                            let op_name = Arc::from(resolved_name);
                            let lazy = if bounds.len() == 1 {
                                LazyFuncValue::new(
                                    Some(Arc::clone(&op_name)),
                                    LazyDomain::General(Box::new(domain_val)),
                                    bounds[0].clone(),
                                    *func_body.clone(),
                                    ctx.env.clone(),
                                )
                            } else {
                                LazyFuncValue::new_multi(
                                    Some(Arc::clone(&op_name)),
                                    LazyDomain::General(Box::new(domain_val)),
                                    bounds.clone(),
                                    *func_body.clone(),
                                    ctx.env.clone(),
                                )
                            };

                            return Ok(Value::LazyFunc(Box::new(lazy)));
                        }
                    }

                    // Issue #73 fix: Memoization for zero-arg local_ops using ZERO_ARG_OP_CACHE.
                    // LET operators in local_ops are evaluated lazily for correct dependency
                    // tracking (#70), but without memoization the same body can be evaluated
                    // many times during enumeration (e.g., YoYoAllGraphs UpOther action).
                    //
                    // KEY FIX: Use simple (shared_id, op_name) key instead of including local_ops_id.
                    // Arc pointers change on every enumeration step even when content is identical
                    // (see enumerate.rs:4281-4286 where local_ops contents are cloned into new Arcs).
                    // The op_name uniquely identifies the definition within a given shared context,
                    // and dep validation handles state-dependent results correctly.
                    let cache_key = (ctx.shared.id, resolved_name.to_string());

                    // Check cache for hit - scan Vec entries to find one with matching deps
                    if let Some(result) = ZERO_ARG_OP_CACHE.with(|cache| {
                        let cache = cache.borrow();
                        let entries = cache.get(&cache_key)?;
                        // Scan entries to find one with matching deps
                        for entry in entries {
                            if op_cache_entry_valid(ctx, entry) {
                                propagate_cached_deps(&entry.deps);
                                return Some(entry.value.clone());
                            }
                        }
                        None
                    }) {
                        return Ok(result);
                    }

                    // Cache miss - evaluate with dependency tracking
                    let (val, deps) = eval_with_dep_tracking(ctx, &def.body)?;

                    let result = match val {
                        Value::LazyFunc(mut f) => {
                            if f.name.is_none() {
                                f.name = Some(Arc::from(resolved_name));
                            }
                            Value::LazyFunc(f)
                        }
                        other => other,
                    };

                    // Store in cache for future reuse
                    if !deps.inconsistent {
                        ZERO_ARG_OP_CACHE.with(|cache| {
                            let mut cache = cache.borrow_mut();
                            let entries = cache.entry(cache_key).or_insert_with(Vec::new);
                            // Evict oldest entries if over limit
                            while entries.len() >= ZERO_ARG_CACHE_MAX_ENTRIES_PER_KEY {
                                entries.remove(0);
                            }
                            entries.push(CachedOpResult {
                                value: result.clone(),
                                deps,
                            });
                        });
                    }

                    return Ok(result);
                }
            }

            // 4) If not a local operator, fall back to state/environment bindings.
            if let Some(value) = ctx.lookup(name) {
                return Ok(value.clone());
            }
            if resolved_name != name {
                if let Some(value) = ctx.lookup(resolved_name) {
                    return Ok(value.clone());
                }
            }

            // 5) Finally, check shared (outer-module) zero-argument operators.
            if let Some(def) = ctx.shared.ops.get(resolved_name) {
                if def.params.is_empty() {
                    let outer_ctx = ctx
                        .without_instance_substitutions()
                        .without_local_ops();

                    if let Expr::FuncDef(bounds, func_body) = &def.body.node {
                        if expr_mentions_op(func_body, resolved_name) {
                            let domain_val = if bounds.len() == 1 {
                                let domain_expr =
                                    bounds[0].domain.as_ref().ok_or_else(|| EvalError::Internal {
                                        message: "Function definition requires bounded variable"
                                            .into(),
                                        span,
                                    })?;
                                eval(&outer_ctx, domain_expr)?
                            } else {
                                let mut components = Vec::with_capacity(bounds.len());
                                for b in bounds {
                                    let domain_expr = b.domain.as_ref().ok_or_else(|| {
                                        EvalError::Internal {
                                            message:
                                                "Function definition requires bounded variable"
                                                    .into(),
                                            span,
                                        }
                                    })?;
                                    components.push(eval(&outer_ctx, domain_expr)?);
                                }
                                Value::TupleSet(TupleSetValue::new(components))
                            };

                            if !domain_val.is_set() {
                                return Err(EvalError::type_error(
                                    "Set",
                                    &domain_val,
                                    Some(def.body.span),
                                ));
                            }

                            let op_name = Arc::from(resolved_name);
                            let lazy = if bounds.len() == 1 {
                                LazyFuncValue::new(
                                    Some(Arc::clone(&op_name)),
                                    LazyDomain::General(Box::new(domain_val)),
                                    bounds[0].clone(),
                                    *func_body.clone(),
                                    outer_ctx.env.clone(),
                                )
                            } else {
                                LazyFuncValue::new_multi(
                                    Some(Arc::clone(&op_name)),
                                    LazyDomain::General(Box::new(domain_val)),
                                    bounds.clone(),
                                    *func_body.clone(),
                                    outer_ctx.env.clone(),
                                )
                            };

                            return Ok(Value::LazyFunc(Box::new(lazy)));
                        }
                    }

                    // Issue #73 fix: Memoization for zero-arg shared.ops using ZERO_ARG_OP_CACHE.
                    // Same pattern as local_ops caching above - simple (shared_id, op_name) key.
                    let cache_key = (outer_ctx.shared.id, resolved_name.to_string());

                    // Check cache for hit - scan Vec entries to find one with matching deps
                    if let Some(result) = ZERO_ARG_OP_CACHE.with(|cache| {
                        let cache = cache.borrow();
                        let entries = cache.get(&cache_key)?;
                        for entry in entries {
                            if op_cache_entry_valid(&outer_ctx, entry) {
                                propagate_cached_deps(&entry.deps);
                                return Some(entry.value.clone());
                            }
                        }
                        None
                    }) {
                        return Ok(result);
                    }

                    // Cache miss - evaluate with dependency tracking
                    let (val, deps) = eval_with_dep_tracking(&outer_ctx, &def.body)?;

                    let result = match val {
                        Value::LazyFunc(mut f) => {
                            if f.name.is_none() {
                                f.name = Some(Arc::from(resolved_name));
                            }
                            Value::LazyFunc(f)
                        }
                        other => other,
                    };

                    // Store in cache for future reuse
                    if !deps.inconsistent {
                        ZERO_ARG_OP_CACHE.with(|cache| {
                            let mut cache = cache.borrow_mut();
                            let entries = cache.entry(cache_key).or_insert_with(Vec::new);
                            while entries.len() >= ZERO_ARG_CACHE_MAX_ENTRIES_PER_KEY {
                                entries.remove(0);
                            }
                            entries.push(CachedOpResult {
                                value: result.clone(),
                                deps,
                            });
                        });
                    }

                    return Ok(result);
                }
            }
            // Check for zero-argument builtins (BOOLEAN, etc.)
            if let Some(result) = eval_builtin(ctx, resolved_name, &[], span)? {
                return Ok(result);
            }
            // Not found
            Err(EvalError::UndefinedVar {
                name: resolved_name.to_string(),
                span,
            })
        }

        // === Operator application ===
        Expr::Apply(op_expr, args) => eval_apply(ctx, op_expr, args, span),

        // === Operator Reference ===
        // OpRef represents a bare operator like + used as a value (e.g., FoldFunctionOnSet(+, 0, f, S))
        // These are not evaluated directly - they're handled specially by higher-order operators
        Expr::OpRef(op) => Err(EvalError::Internal {
            message: format!(
                "Operator reference '{}' cannot be evaluated directly; it must be used with a higher-order operator like FoldFunctionOnSet",
                op
            ),
            span,
        }),

        // === Lambda ===
        Expr::Lambda(_params, _body) => {
            // Lambdas are stored as closures - for now we don't support them as values
            // They should be applied immediately in operator calls
            Err(EvalError::Internal {
                message: "Lambda expressions cannot be evaluated as values".into(),
                span,
            })
        }

        // === Module Reference ===
        Expr::ModuleRef(target, op_name, args) => {
            eval_module_ref_target(ctx, target, op_name, args, span)
        }

        // === Instance Expression ===
        // InstanceExpr is used to define named instances (InChan == INSTANCE Channel WITH ...)
        // It should not be evaluated as a value - the instance should be registered at load time
        Expr::InstanceExpr(module, _subs) => Err(EvalError::Internal {
            message: format!(
                "INSTANCE {} expression should be registered at load time, not evaluated",
                module
            ),
            span,
        }),

        // === Logic ===
        Expr::And(a, b) => {
            let av = eval(ctx, a)?;
            let ab = av
                .as_bool()
                .ok_or_else(|| EvalError::type_error("BOOLEAN", &av, Some(a.span)))?;
            if !ab {
                return Ok(Value::Bool(false)); // Short-circuit
            }
            let bv = eval(ctx, b)?;
            let bb = bv
                .as_bool()
                .ok_or_else(|| EvalError::type_error("BOOLEAN", &bv, Some(b.span)))?;
            Ok(Value::Bool(bb))
        }

        Expr::Or(a, b) => {
            // TLC semantics: In action contexts, certain runtime errors in one branch
            // should be treated as FALSE for that branch, allowing the other branch to be tried.
            // This handles cases like `CounterAction(p) \/ StandardAction(p)` where
            // StandardAction might fail with NotInDomain for certain values of p.
            let av = match eval(ctx, a) {
                Ok(v) => v,
                Err(e) if is_action_disabling_error(&e) => Value::Bool(false),
                Err(e) => return Err(e),
            };
            let ab = av
                .as_bool()
                .ok_or_else(|| EvalError::type_error("BOOLEAN", &av, Some(a.span)))?;
            if ab {
                return Ok(Value::Bool(true)); // Short-circuit
            }
            let bv = match eval(ctx, b) {
                Ok(v) => v,
                Err(e) if is_action_disabling_error(&e) => Value::Bool(false),
                Err(e) => return Err(e),
            };
            let bb = bv
                .as_bool()
                .ok_or_else(|| EvalError::type_error("BOOLEAN", &bv, Some(b.span)))?;
            Ok(Value::Bool(bb))
        }

        Expr::Not(a) => {
            let av = eval(ctx, a)?;
            let ab = av
                .as_bool()
                .ok_or_else(|| EvalError::type_error("BOOLEAN", &av, Some(a.span)))?;
            Ok(Value::Bool(!ab))
        }

        Expr::Implies(a, b) => {
            let av = eval(ctx, a)?;
            let ab = av
                .as_bool()
                .ok_or_else(|| EvalError::type_error("BOOLEAN", &av, Some(a.span)))?;
            if !ab {
                return Ok(Value::Bool(true)); // FALSE => X is TRUE
            }
            let bv = eval(ctx, b)?;
            let bb = bv
                .as_bool()
                .ok_or_else(|| EvalError::type_error("BOOLEAN", &bv, Some(b.span)))?;
            Ok(Value::Bool(bb))
        }

        Expr::Equiv(a, b) => {
            let av = eval(ctx, a)?;
            let ab = av
                .as_bool()
                .ok_or_else(|| EvalError::type_error("BOOLEAN", &av, Some(a.span)))?;
            let bv = eval(ctx, b)?;
            let bb = bv
                .as_bool()
                .ok_or_else(|| EvalError::type_error("BOOLEAN", &bv, Some(b.span)))?;
            Ok(Value::Bool(ab == bb))
        }

        // === Quantifiers ===
        Expr::Forall(bounds, body) => eval_forall(ctx, bounds, body, span),
        Expr::Exists(bounds, body) => eval_exists(ctx, bounds, body, span),
        Expr::Choose(bound, body) => eval_choose(ctx, bound, body, span),

        // === Sets ===
        Expr::SetEnum(elems) => {
            let mut set = SetBuilder::new();
            for elem in elems {
                set.insert(eval(ctx, elem)?);
            }
            Ok(set.build_value())
        }

        Expr::SetBuilder(expr, bounds) => eval_set_builder(ctx, expr, bounds, span),
        Expr::SetFilter(bound, pred) => eval_set_filter(ctx, bound, pred, span),

        Expr::In(a, b) => {
            let av = eval(ctx, a)?;

            // Use lazy membership check for FuncSet, Powerset, Seq, and RecordSet
            // This avoids enumerating large/infinite sets
            if needs_lazy_membership(ctx, b) {
                return Ok(Value::Bool(eval_membership_lazy(ctx, av, b)?));
            }

            let bv = eval(ctx, b)?;

            // Handle membership in infinite sets (Nat, Int, Real)
            if let Value::ModelValue(name) = &bv {
                return match name.as_ref() {
                    "Nat" => match &av {
                        Value::SmallInt(n) => Ok(Value::Bool(*n >= 0)),
                        Value::Int(n) => Ok(Value::Bool(*n >= BigInt::zero())),
                        _ => Ok(Value::Bool(false)),
                    },
                    "Int" => Ok(Value::Bool(matches!(&av, Value::SmallInt(_) | Value::Int(_)))),
                    "Real" => Ok(Value::Bool(matches!(&av, Value::SmallInt(_) | Value::Int(_)))), // Int ⊆ Real
                    _ => Err(EvalError::type_error("Set", &bv, Some(b.span))),
                };
            }

            // Handle SetPred: check source membership, then evaluate predicate
            if let Value::SetPred(spv) = &bv {
                return Ok(Value::Bool(check_set_pred_membership(ctx, &av, spv, span)?));
            }

            // Handle both Set and Interval using set_contains
            let contains = bv
                .set_contains(&av)
                .ok_or_else(|| EvalError::type_error("Set", &bv, Some(b.span)))?;
            Ok(Value::Bool(contains))
        }

        Expr::NotIn(a, b) => {
            let av = eval(ctx, a)?;

            // Use lazy membership check for FuncSet, Powerset, Seq, and RecordSet
            // This avoids enumerating large/infinite sets
            if needs_lazy_membership(ctx, b) {
                return Ok(Value::Bool(!eval_membership_lazy(ctx, av, b)?));
            }

            let bv = eval(ctx, b)?;

            // Handle membership in infinite sets (Nat, Int, Real)
            if let Value::ModelValue(name) = &bv {
                return match name.as_ref() {
                    "Nat" => match &av {
                        Value::SmallInt(n) => Ok(Value::Bool(*n < 0)),
                        Value::Int(n) => Ok(Value::Bool(*n < BigInt::zero())),
                        _ => Ok(Value::Bool(true)),
                    },
                    "Int" => Ok(Value::Bool(!matches!(&av, Value::SmallInt(_) | Value::Int(_)))),
                    "Real" => Ok(Value::Bool(!matches!(&av, Value::SmallInt(_) | Value::Int(_)))), // Int ⊆ Real
                    _ => Err(EvalError::type_error("Set", &bv, Some(b.span))),
                };
            }

            // Handle SetPred: check source membership, then evaluate predicate
            if let Value::SetPred(spv) = &bv {
                return Ok(Value::Bool(!check_set_pred_membership(ctx, &av, spv, span)?));
            }

            // Handle both Set and Interval using set_contains
            let contains = bv
                .set_contains(&av)
                .ok_or_else(|| EvalError::type_error("Set", &bv, Some(b.span)))?;
            Ok(Value::Bool(!contains))
        }

        Expr::Subseteq(a, b) => {
            let av = eval(ctx, a)?;
            let bv = eval(ctx, b)?;

            // Fast path: when both are Value::Set(OrdSet), check subset relationship
            if let (Value::Set(sa), Value::Set(sb)) = (&av, &bv) {
                // Quick rejection: if A is larger than B, A cannot be a subset of B
                if sa.len() > sb.len() {
                    return Ok(Value::Bool(false));
                }
                // Empty set is subset of any set
                if sa.is_empty() {
                    return Ok(Value::Bool(true));
                }
                // For small sets, use contains() loop which avoids iterator overhead
                // The im crate's is_subset() creates iterators that traverse B-tree nodes
                // For sets with <= 16 elements, linear contains() checks are faster
                if sa.len() <= 16 {
                    for elem in sa.iter() {
                        if !sb.contains(elem) {
                            return Ok(Value::Bool(false));
                        }
                    }
                    return Ok(Value::Bool(true));
                }
                // For larger sets, use the bulk is_subset operation
                return Ok(Value::Bool(sa.is_subset(sb)));
            }

            // Fast path: when av is an Interval and bv is Set(OrdSet)
            // Check interval bounds against set membership
            if let (Value::Interval(iv), Value::Set(sb)) = (&av, &bv) {
                // For small intervals, check each element
                if let Some(len) = iv.len().to_usize() {
                    if len <= 32 {
                        for v in iv.iter_values() {
                            if !sb.contains(&v) {
                                return Ok(Value::Bool(false));
                            }
                        }
                        return Ok(Value::Bool(true));
                    }
                }
                // Fall through to general case for large intervals
            }

            // TLC requires the left side to be enumerable; the right side only needs membership.
            let iter = av.iter_set().ok_or_else(|| EvalError::Internal {
                message: "SubsetEq requires enumerable left-hand set".into(),
                span,
            })?;

            for elem in iter {
                let in_b = match &bv {
                    Value::ModelValue(name) => match name.as_ref() {
                        "Nat" => match &elem {
                            Value::SmallInt(n) => *n >= 0,
                            Value::Int(n) => *n >= BigInt::zero(),
                            _ => false,
                        },
                        "Int" => matches!(&elem, Value::SmallInt(_) | Value::Int(_)),
                        "Real" => matches!(&elem, Value::SmallInt(_) | Value::Int(_)), // Int ⊆ Real
                        _ => {
                            return Err(EvalError::type_error("Set", &bv, Some(b.span)));
                        }
                    },
                    // Handle SetPred: check membership via predicate evaluation
                    Value::SetPred(spv) => check_set_pred_membership(ctx, &elem, spv, span)?,
                    _ => bv
                        .set_contains(&elem)
                        .ok_or_else(|| EvalError::type_error("Set", &bv, Some(b.span)))?,
                };
                if !in_b {
                    return Ok(Value::Bool(false));
                }
            }
            Ok(Value::Bool(true))
        }

        Expr::Union(a, b) => {
            let av = eval(ctx, a)?;
            let bv = eval(ctx, b)?;
            // Check that both operands are sets
            if !av.is_set() {
                return Err(EvalError::type_error("Set", &av, Some(a.span)));
            }
            if !bv.is_set() {
                return Err(EvalError::type_error("Set", &bv, Some(b.span)));
            }
            // Try eager evaluation if both operands are enumerable
            match (av.to_sorted_set(), bv.to_sorted_set()) {
                (Some(sa), Some(sb)) => Ok(Value::Set(sa.union(&sb))),
                // Otherwise, return lazy SetCup
                _ => Ok(Value::SetCup(SetCupValue::new(av, bv))),
            }
        }

        Expr::Intersect(a, b) => {
            let av = eval(ctx, a)?;
            let bv = eval(ctx, b)?;
            // Check that both operands are sets
            if !av.is_set() {
                return Err(EvalError::type_error("Set", &av, Some(a.span)));
            }
            if !bv.is_set() {
                return Err(EvalError::type_error("Set", &bv, Some(b.span)));
            }

            // Special handling for SetPred - compute intersection eagerly if possible
            // SetPred requires evaluation context for membership checks, so we handle it here
            match (&av, &bv) {
                // Case: enumerable set ∩ SetPred - iterate left, filter by SetPred membership
                (_, Value::SetPred(spv)) if av.iter_set().is_some() => {
                    let mut result = SetBuilder::new();
                    for elem in av.iter_set().unwrap() {
                        if check_set_pred_membership(ctx, &elem, spv, span)? {
                            result.insert(elem);
                        }
                    }
                    return Ok(result.build_value());
                }
                // Case: SetPred ∩ enumerable set - iterate right, filter by SetPred membership
                (Value::SetPred(spv), _) if bv.iter_set().is_some() => {
                    let mut result = SetBuilder::new();
                    for elem in bv.iter_set().unwrap() {
                        if check_set_pred_membership(ctx, &elem, spv, span)? {
                            result.insert(elem);
                        }
                    }
                    return Ok(result.build_value());
                }
                _ => {}
            }

            // Try eager evaluation if both operands are enumerable
            match (av.to_sorted_set(), bv.to_sorted_set()) {
                (Some(sa), Some(sb)) => Ok(Value::Set(sa.intersection(&sb))),
                // Otherwise, return lazy SetCap (can still enumerate if at least one is enumerable)
                _ => Ok(Value::SetCap(SetCapValue::new(av, bv))),
            }
        }

        Expr::SetMinus(a, b) => {
            let av = eval(ctx, a)?;
            let bv = eval(ctx, b)?;
            // Check that both operands are sets
            if !av.is_set() {
                return Err(EvalError::type_error("Set", &av, Some(a.span)));
            }
            if !bv.is_set() {
                return Err(EvalError::type_error("Set", &bv, Some(b.span)));
            }

            // Special handling for SetPred on RHS - check membership via predicate evaluation
            if let Value::SetPred(spv) = &bv {
                if let Some(iter) = av.iter_set() {
                    let mut result = SetBuilder::new();
                    for elem in iter {
                        // Keep elements NOT in the SetPred
                        if !check_set_pred_membership(ctx, &elem, spv, span)? {
                            result.insert(elem);
                        }
                    }
                    return Ok(result.build_value());
                }
            }

            // Try eager evaluation if LHS is enumerable
            match av.to_sorted_set() {
                Some(sa) => {
                    // Can compute eagerly - filter by non-membership in RHS
                    let result: SortedSet = sa
                        .iter()
                        .filter(|v| !bv.set_contains(v).unwrap_or(false))
                        .cloned()
                        .collect();
                    Ok(Value::Set(result))
                }
                // LHS not enumerable, return lazy SetDiff
                None => Ok(Value::SetDiff(SetDiffValue::new(av, bv))),
            }
        }

        Expr::Powerset(a) => {
            let av = eval(ctx, a)?;
            // Check that it's a set-like value
            if !av.is_set() {
                return Err(EvalError::type_error("Set", &av, Some(a.span)));
            }
            // Return lazy Subset value instead of eagerly computing powerset
            Ok(Value::Subset(SubsetValue::new(av)))
        }

        Expr::BigUnion(a) => {
            let av = eval(ctx, a)?;
            // Accept any set-like value
            if !av.is_set() {
                return Err(EvalError::type_error("Set", &av, Some(a.span)));
            }
            // For small, fully enumerable sets, compute eagerly for efficiency
            // For large or non-enumerable sets, return lazy UnionValue
            if let Some(sa) = av.to_ord_set() {
                // Check if all inner sets are enumerable and total size is reasonable
                let mut can_eager = true;
                let mut total_size = 0usize;
                for elem in sa.iter() {
                    if let Some(inner_set) = elem.to_ord_set() {
                        total_size += inner_set.len();
                        if total_size > 10000 {
                            can_eager = false;
                            break;
                        }
                    } else {
                        can_eager = false;
                        break;
                    }
                }
                if can_eager {
                    // Eager evaluation for small sets
                    return big_union(&sa).ok_or(EvalError::TypeError {
                        expected: "Set of Sets",
                        got: "Set containing non-Set",
                        span,
                    });
                }
            }
            // Lazy evaluation - return UnionValue
            Ok(Value::BigUnion(UnionValue::new(av)))
        }

        // === Functions ===
        Expr::FuncDef(bounds, body) => eval_func_def(ctx, bounds, body, span),
        Expr::FuncApply(func_expr, arg_expr) => eval_func_apply(ctx, func_expr, arg_expr, span),
        Expr::Domain(func_expr) => {
            let fv = eval(ctx, func_expr)?;
            match &fv {
                Value::Func(f) => Ok(Value::Set(f.domain_as_sorted_set())),
                Value::IntFunc(f) => {
                    // DOMAIN of IntFunc is the integer interval min..max
                    Ok(range_set(&BigInt::from(f.min), &BigInt::from(f.max)))
                }
                Value::Seq(_) | Value::Tuple(_) => {
                    // DOMAIN of sequence/tuple is 1..Len(s)
                    let s = fv.as_seq_or_tuple_elements().unwrap();
                    if s.is_empty() {
                        Ok(Value::empty_set())
                    } else {
                        Ok(range_set(&BigInt::one(), &BigInt::from(s.len())))
                    }
                }
                Value::Record(r) => {
                    // DOMAIN of record is set of field names
                    let names: SortedSet = r.keys().map(|k| Value::String(k.clone())).collect();
                    Ok(Value::Set(names))
                }
                Value::LazyFunc(f) => {
                    // DOMAIN of LazyFunc is the set representation of its domain type
                    match &f.domain {
                        LazyDomain::Nat => Ok(Value::ModelValue(Arc::from("Nat"))),
                        LazyDomain::Int => Ok(Value::ModelValue(Arc::from("Int"))),
                        LazyDomain::Real => Ok(Value::ModelValue(Arc::from("Real"))),
                        LazyDomain::String => Ok(Value::StringSet),
                        LazyDomain::Product(components) => {
                            // For multi-argument functions, domain is a cartesian product
                            let sets: Vec<Value> = components
                                .iter()
                                .map(|c| match c {
                                    ComponentDomain::Nat => Value::ModelValue(Arc::from("Nat")),
                                    ComponentDomain::Int => Value::ModelValue(Arc::from("Int")),
                                    ComponentDomain::Real => Value::ModelValue(Arc::from("Real")),
                                    ComponentDomain::String => Value::StringSet,
                                    ComponentDomain::Finite(s) => Value::Set(SortedSet::from_ord_set(s)),
                                })
                                .collect();
                            Ok(Value::TupleSet(TupleSetValue::new(sets)))
                        }
                        LazyDomain::General(v) => {
                            // For general domains, return the stored domain value directly
                            Ok(v.as_ref().clone())
                        }
                    }
                }
                _ => Err(EvalError::type_error(
                    "Function/Seq/Tuple/Record",
                    &fv,
                    Some(func_expr.span),
                )),
            }
        }
        Expr::Except(func_expr, specs) => eval_except(ctx, func_expr, specs, span),
        Expr::FuncSet(domain, range) => {
            let dv = eval(ctx, domain)?;
            let rv = eval(ctx, range)?;
            // Check that both are set-like values
            if !dv.is_set() {
                return Err(EvalError::type_error("Set", &dv, Some(domain.span)));
            }
            if !rv.is_set() {
                return Err(EvalError::type_error("Set", &rv, Some(range.span)));
            }
            // Return lazy FuncSet value instead of eagerly computing all functions
            Ok(Value::FuncSet(FuncSetValue::new(dv, rv)))
        }

        // === Records ===
        Expr::Record(fields) => {
            let mut builder = RecordBuilder::with_capacity(fields.len());
            for (name, val_expr) in fields {
                builder.insert(intern_string(name.node.as_str()), eval(ctx, val_expr)?);
            }
            Ok(Value::Record(builder.build()))
        }

        Expr::RecordAccess(rec_expr, field) => {
            let rv = eval(ctx, rec_expr)?;
            let rec = rv
                .as_record()
                .ok_or_else(|| EvalError::type_error("Record", &rv, Some(rec_expr.span)))?;
            let field_name: Arc<str> = intern_string(field.node.as_str());
            rec.get(&field_name)
                .cloned()
                .ok_or_else(|| EvalError::NoSuchField {
                    field: field.node.clone(),
                    span,
                })
        }

        Expr::RecordSet(fields) => {
            let mut field_sets = Vec::new();
            for (name, set_expr) in fields {
                let sv = eval(ctx, set_expr)?;
                if !sv.is_set() {
                    return Err(EvalError::type_error("Set", &sv, Some(set_expr.span)));
                }
                field_sets.push((intern_string(name.node.as_str()), sv));
            }
            Ok(Value::RecordSet(RecordSetValue::new(field_sets)))
        }

        // === Tuples and Sequences ===
        Expr::Tuple(elems) => {
            let mut vals = Vec::new();
            for elem in elems {
                vals.push(eval(ctx, elem)?);
            }
            Ok(Value::Tuple(vals.into()))
        }

        Expr::Times(factors) => {
            // Evaluate each factor to a set-like value and create a lazy TupleSetValue
            let mut components = Vec::with_capacity(factors.len());
            for factor in factors {
                let fv = eval(ctx, factor)?;
                if !fv.is_set() {
                    return Err(EvalError::type_error("Set", &fv, Some(factor.span)));
                }
                components.push(fv);
            }
            Ok(Value::TupleSet(TupleSetValue::new(components)))
        }

        // === Primed variables (next-state lookup) ===
        Expr::Prime(inner) => {
            // Fast path: array-based next-state lookup (O(1) via VarIndex)
            if let Some(next_env) = ctx.next_state_env {
                if let Expr::Ident(name) = &inner.node {
                    // Check for INSTANCE substitution before short-circuit
                    let has_instance_sub = ctx
                        .instance_substitutions()
                        .is_some_and(|subs| subs.iter().any(|s| s.from.node == *name));

                    if !has_instance_sub {
                        // Look up by VarIndex for O(1) access
                        if let Some(idx) = ctx.var_registry().get(name) {
                            // Safety: next_env must remain valid for the evaluation duration
                            let value = unsafe { next_env.get_unchecked(idx.as_usize()) };
                            record_next_read(idx, value);
                            return Ok(value.clone());
                        }
                    }
                }
            }

            // Fall back to HashMap-based next_state
            let Some(next_state) = &ctx.next_state else {
                // No array-based next_state_env and no HashMap next_state
                if ctx.next_state_env.is_none() {
                    return Err(EvalError::Internal {
                        message: "Primed variable cannot be evaluated (no next-state context)".into(),
                        span,
                    });
                }
                // next_state_env is set but we couldn't resolve - fall through to complex eval
                // This handles complex primed expressions like (f(x))' that aren't simple variables
                let mut next_ctx = ctx.clone();
                next_ctx.next_state_env = None;
                // Swap state_env with next_state_env for the inner evaluation
                next_ctx.state_env = ctx.next_state_env;
                return eval(&next_ctx, inner);
            };

            // If this is a state variable, resolve it from the next-state environment.
            if let Expr::Ident(name) = &inner.node {
                // If this identifier has an active INSTANCE substitution, we must evaluate the
                // substituted expression in the next-state context (so don't short-circuit with a
                // direct next_state lookup).
                let has_instance_sub = ctx
                    .instance_substitutions()
                    .is_some_and(|subs| subs.iter().any(|s| s.from.node == *name));

                if !has_instance_sub {
                    if let Some(value) = next_state.get(name.as_str()) {
                        if std::env::var("TLA2_DEBUG_PRIME").is_ok() {
                            eprintln!("Prime: direct lookup {} -> {}", name, value);
                        }
                        if let Some(idx) = ctx.var_registry().get(name) {
                            record_next_read(idx, value);
                        }
                        return Ok(value.clone());
                    }
                }
            }

            if std::env::var("TLA2_DEBUG_PRIME").is_ok() {
                eprintln!("Prime: complex expression, inner={:?}", std::mem::discriminant(&inner.node));
                eprintln!("  next_state vars: {:?}", next_state.keys().collect::<Vec<_>>());
            }
            // Evaluate `inner` with state variables interpreted in the *next* state.
            //
            // When `next_state_env` is set (array-based fast path from prime_guards_hold_in_next_array),
            // swap it into state_env so that state variable lookups go to next-state values.
            // This avoids HashMap iteration and enables O(1) lookups.
            //
            // When `state_env` is set (ArrayState fast path), state variable lookups prefer
            // `state_env` over `env`, so binding next-state values into `env` is insufficient.
            //
            // We handle this in multiple modes:
            // - `next_state_env` is set: swap it into `state_env` for O(1) lookups
            // - Full `next_state` HashMap: clear `state_env` and bind all vars into `env`
            // - Partial `next_state`: shadow via `local_stack` (falls back to current state)
            if ctx.next_state_env.is_some() {
                // Fast path: swap next_state_env -> state_env
                let mut next_ctx = ctx.clone();
                next_ctx.next_state = None;
                next_ctx.next_state_env = None;
                next_ctx.state_env = ctx.next_state_env;
                return with_state_lookup_mode(StateLookupMode::Next, || eval(&next_ctx, inner));
            }
            let mut next_ctx = ctx.clone();
            next_ctx.next_state = None;
            next_ctx.next_state_env = None;
            if next_ctx.state_env.is_some() {
                let is_full_next_state = next_state.len() == ctx.var_registry().len();
                if is_full_next_state {
                    next_ctx.state_env = None;
                    for (name, value) in next_state.iter() {
                        next_ctx.env.insert(Arc::clone(name), value.clone());
                    }
                } else {
                    for (name, value) in next_state.iter() {
                        // Avoid overriding a locally-bound variable that shadows a state var.
                        if !next_ctx.has_local_binding(name.as_ref()) {
                            next_ctx.local_stack.push((Arc::clone(name), value.clone()));
                        }
                    }
                }
            } else {
                for (name, value) in next_state.iter() {
                    next_ctx.env.insert(Arc::clone(name), value.clone());
                }
            }
            let result = with_state_lookup_mode(StateLookupMode::Next, || eval(&next_ctx, inner));
            if std::env::var("TLA2_DEBUG_PRIME").is_ok() {
                eprintln!("Prime: result={:?}", result);
            }
            result
        }

        // === Action-level helper ===
        Expr::Unchanged(inner) => {
            let Some(next_state) = &ctx.next_state else {
                return Err(EvalError::Internal {
                    message: "UNCHANGED cannot be evaluated (no next-state context)".into(),
                    span,
                });
            };

            // Evaluate `inner` in the current-state environment.
            let cur_v = eval(ctx, inner)?;

            // Evaluate `inner` in the next-state environment by rebinding all
            // next-state variables to their primed values as unprimed names.
            let mut next_ctx = ctx.clone();
            if next_ctx.state_env.is_some() {
                let is_full_next_state = next_state.len() == ctx.var_registry().len();
                if is_full_next_state {
                    next_ctx.state_env = None;
                    for (name, value) in next_state.iter() {
                        next_ctx.env.insert(Arc::clone(name), value.clone());
                    }
                } else {
                    for (name, value) in next_state.iter() {
                        if !next_ctx.has_local_binding(name.as_ref()) {
                            next_ctx.local_stack.push((Arc::clone(name), value.clone()));
                        }
                    }
                }
            } else {
                for (name, value) in next_state.iter() {
                    next_ctx.env.insert(Arc::clone(name), value.clone());
                }
            }
            let next_v = with_state_lookup_mode(StateLookupMode::Next, || eval(&next_ctx, inner))?;

            Ok(Value::Bool(cur_v == next_v))
        }

        // === ENABLED operator ===
        // ENABLED(A) is true iff action A has at least one successor from the current state.
        // This is a state-level operator that can appear in action guards.
        Expr::Enabled(action) => {
            // Get variable names from the context's var registry
            let vars: Vec<Arc<str>> = ctx.shared.var_registry.names().to_vec();

            // ENABLED is a state-level operator. If we're currently evaluating in an
            // action context (i.e., `ctx.next_state` is set), we must ignore that
            // particular next-state binding and instead existentially quantify over
            // possible successors of `action` from the current state.
            let mut eval_ctx = ctx.clone();
            eval_ctx.next_state = None;
            let result = eval_enabled(&mut eval_ctx, action, &vars)?;
            Ok(Value::Bool(result))
        }

        // === Temporal (not evaluated - model checking only) ===
        Expr::Always(_)
        | Expr::Eventually(_)
        | Expr::LeadsTo(_, _)
        | Expr::WeakFair(_, _)
        | Expr::StrongFair(_, _)
        => Err(EvalError::Internal {
            message: "Temporal operators cannot be directly evaluated".into(),
            span,
        }),

        // === Control ===
        Expr::If(cond, then_branch, else_branch) => {
            let cv = eval(ctx, cond)?;
            let cb = cv
                .as_bool()
                .ok_or_else(|| EvalError::type_error("BOOLEAN", &cv, Some(cond.span)))?;
            if cb {
                eval(ctx, then_branch)
            } else {
                eval(ctx, else_branch)
            }
        }

        Expr::Case(arms, other) => {
            for arm in arms {
                let gv = eval(ctx, &arm.guard)?;
                let gb = gv
                    .as_bool()
                    .ok_or_else(|| EvalError::type_error("BOOLEAN", &gv, Some(arm.guard.span)))?;
                if gb {
                    return eval(ctx, &arm.body);
                }
            }
            if let Some(default) = other {
                eval(ctx, default)
            } else {
                Err(EvalError::Internal {
                    message: "CASE: no arm matched and no OTHER clause".into(),
                    span,
                })
            }
        }

        Expr::Let(defs, body) => {
            // LET binds local operator definitions (including zero-arg operators).
            // Definitions can be recursive, so we must register them before evaluating the body.
            // Use local_ops to avoid cloning the entire shared context.
            let mut local_ops: OpEnv = match &ctx.local_ops {
                Some(ops) => (**ops).clone(),
                None => OpEnv::default(),
            };
            for def in defs {
                local_ops.insert(def.name.node.clone(), def.clone());
            }
            let mut new_ctx = EvalCtx {
                shared: Arc::clone(&ctx.shared),
                env: ctx.env.clone(),
                next_state: ctx.next_state.clone(),
                local_ops: Some(Arc::new(local_ops)),
                // Preserve local_stack bindings from enclosing scopes (e.g., EXISTS-bound vars,
                // operator parameters). Clearing this caused "Undefined variable" errors when
                // LET expressions referenced variables bound by enclosing quantifiers/operators.
                local_stack: ctx.local_stack.clone(),
                state_env: ctx.state_env,
                next_state_env: ctx.next_state_env,
                recursion_depth: ctx.recursion_depth,
                instance_substitutions: ctx.instance_substitutions.clone(),
            };

            // Recursive local functions like:
            //
            //   LET f[x \\in S] == IF ... THEN ... ELSE ... f[...] ... IN ...
            //
            // are parsed as zero-arg operator defs whose body is a `FuncDef`. Evaluating that
            // `FuncDef` eagerly tries to build the full mapping, which immediately re-enters
            // `f` and can stack overflow (e.g. SchedulingAllocator's PermSeqs/perms).
            //
            // Detect this pattern and represent the function as a memoized LazyFunc instead.
            fn expr_mentions_ident(expr: &Expr, target: &str) -> bool {
                match expr {
                    Expr::Bool(_) | Expr::Int(_) | Expr::String(_) | Expr::OpRef(_) => false,
                    Expr::Ident(name) => name == target,
                    Expr::Apply(op, args) => {
                        expr_mentions_ident(&op.node, target)
                            || args
                                .iter()
                                .any(|a| expr_mentions_ident(&a.node, target))
                    }
                    Expr::ModuleRef(_, _, args) => {
                        args.iter().any(|a| expr_mentions_ident(&a.node, target))
                    }
                    Expr::InstanceExpr(_, subs) => subs
                        .iter()
                        .any(|s| expr_mentions_ident(&s.to.node, target)),
                    Expr::Lambda(_params, body) => expr_mentions_ident(&body.node, target),

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
                    | Expr::FuncSet(a, b)
                    | Expr::LeadsTo(a, b)
                    | Expr::WeakFair(a, b)
                    | Expr::StrongFair(a, b) => {
                        expr_mentions_ident(&a.node, target) || expr_mentions_ident(&b.node, target)
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
                    | Expr::Unchanged(a)
                    | Expr::RecordAccess(a, _) => expr_mentions_ident(&a.node, target),

                    Expr::Forall(bounds, body) | Expr::Exists(bounds, body) => bounds
                        .iter()
                        .any(|b| {
                            b.domain
                                .as_ref()
                                .is_some_and(|d| expr_mentions_ident(&d.node, target))
                        })
                        || expr_mentions_ident(&body.node, target),
                    Expr::Choose(bound, body) => bound
                        .domain
                        .as_ref()
                        .is_some_and(|d| expr_mentions_ident(&d.node, target))
                        || expr_mentions_ident(&body.node, target),

                    Expr::SetEnum(elems) | Expr::Tuple(elems) | Expr::Times(elems) => elems
                        .iter()
                        .any(|e| expr_mentions_ident(&e.node, target)),
                    Expr::SetBuilder(expr, bounds) => expr_mentions_ident(&expr.node, target)
                        || bounds.iter().any(|b| {
                            b.domain
                                .as_ref()
                                .is_some_and(|d| expr_mentions_ident(&d.node, target))
                        }),
                    Expr::SetFilter(bound, pred) => bound
                        .domain
                        .as_ref()
                        .is_some_and(|d| expr_mentions_ident(&d.node, target))
                        || expr_mentions_ident(&pred.node, target),

                    Expr::FuncDef(bounds, body) => bounds.iter().any(|b| {
                        b.domain
                            .as_ref()
                            .is_some_and(|d| expr_mentions_ident(&d.node, target))
                    }) || expr_mentions_ident(&body.node, target),
                    Expr::FuncApply(f, arg) => {
                        expr_mentions_ident(&f.node, target) || expr_mentions_ident(&arg.node, target)
                    }
                    Expr::Except(f, specs) => {
                        expr_mentions_ident(&f.node, target)
                            || specs.iter().any(|s| {
                                s.path.iter().any(|p| match p {
                                    tla_core::ast::ExceptPathElement::Index(idx) => {
                                        expr_mentions_ident(&idx.node, target)
                                    }
                                    tla_core::ast::ExceptPathElement::Field(_) => false,
                                }) || expr_mentions_ident(&s.value.node, target)
                            })
                    }

                    Expr::Record(fields) => fields
                        .iter()
                        .any(|(_name, expr)| expr_mentions_ident(&expr.node, target)),
                    Expr::RecordSet(fields) => fields
                        .iter()
                        .any(|(_name, expr)| expr_mentions_ident(&expr.node, target)),

                    Expr::If(c, t, e) => {
                        expr_mentions_ident(&c.node, target)
                            || expr_mentions_ident(&t.node, target)
                            || expr_mentions_ident(&e.node, target)
                    }
                    Expr::Case(arms, other) => arms.iter().any(|a| {
                        expr_mentions_ident(&a.guard.node, target)
                            || expr_mentions_ident(&a.body.node, target)
                    }) || other
                        .as_ref()
                        .is_some_and(|d| expr_mentions_ident(&d.node, target)),

                    Expr::Let(defs, body) => defs
                        .iter()
                        .any(|d| expr_mentions_ident(&d.body.node, target))
                        || expr_mentions_ident(&body.node, target),
                }
            }

            // Cache zero-argument definitions as values for this LET scope.
            // This preserves call-by-value behavior (and avoids recomputation) while
            // still allowing recursion through the operator environment.
            for def in defs {
                if def.params.is_empty() {
                    // Named instances inside LET (e.g. `LET G == INSTANCE Graphs IN ...`) are
                    // represented as zero-arg operator defs whose body is an `InstanceExpr`.
                    //
                    // `InstanceExpr` is not a value and must not be evaluated eagerly here.
                    // Module references (`G!Op`) resolve the instance info from the operator
                    // definition body via `eval_module_ref`.
                    if matches!(&def.body.node, Expr::InstanceExpr(_, _)) {
                        continue;
                    }

                    if let Expr::FuncDef(bounds, func_body) = &def.body.node {
                        // Handle directly-recursive local function definitions by using LazyFunc.
                        let def_name = def.name.node.as_str();
                        if expr_mentions_ident(&func_body.node, def_name) {
                            let domain_val = if bounds.len() == 1 {
                                let domain_expr =
                                    bounds[0].domain.as_ref().ok_or_else(|| EvalError::Internal {
                                        message: "Function definition requires bounded variable"
                                            .into(),
                                        span,
                                    })?;
                                eval(&new_ctx, domain_expr)?
                            } else {
                                let mut components = Vec::with_capacity(bounds.len());
                                for b in bounds {
                                    let domain_expr = b.domain.as_ref().ok_or_else(|| {
                                        EvalError::Internal {
                                            message:
                                                "Function definition requires bounded variable"
                                                    .into(),
                                            span,
                                        }
                                    })?;
                                    components.push(eval(&new_ctx, domain_expr)?);
                                }
                                Value::TupleSet(TupleSetValue::new(components))
                            };

                            if !domain_val.is_set() {
                                return Err(EvalError::type_error(
                                    "Set",
                                    &domain_val,
                                    Some(def.body.span),
                                ));
                            }

                            let name = Arc::from(def_name);
                            let lazy = if bounds.len() == 1 {
                                LazyFuncValue::new(
                                    Some(Arc::clone(&name)),
                                    LazyDomain::General(Box::new(domain_val)),
                                    bounds[0].clone(),
                                    *func_body.clone(),
                                    new_ctx.env.clone(),
                                )
                            } else {
                                LazyFuncValue::new_multi(
                                    Some(Arc::clone(&name)),
                                    LazyDomain::General(Box::new(domain_val)),
                                    bounds.clone(),
                                    *func_body.clone(),
                                    new_ctx.env.clone(),
                                )
                            };

                            new_ctx.env.insert(name, Value::LazyFunc(Box::new(lazy)));
                            continue;
                        }
                    }
                    // Non-recursive zero-arg definitions are evaluated lazily when accessed.
                    // They are registered in local_ops and evaluated via Expr::Ident handling.
                    // Eager evaluation here would cause errors when definitions are never used
                    // (e.g., CHOOSE with no matching element in unused LET bindings).
                }
            }

            eval(&new_ctx, body)
        }

        // === Comparison ===
        Expr::Eq(a, b) => {
            let av = eval(ctx, a)?;
            let bv = eval(ctx, b)?;
            // Handle SetPred specially - need to materialize lazy sets for comparison
            Ok(Value::Bool(values_equal(ctx, &av, &bv)?))
        }

        Expr::Neq(a, b) => {
            let av = eval(ctx, a)?;
            let bv = eval(ctx, b)?;
            // Handle SetPred specially - need to materialize lazy sets for comparison
            Ok(Value::Bool(!values_equal(ctx, &av, &bv)?))
        }

        Expr::Lt(a, b) => eval_comparison(ctx, a, b, |x, y| x < y),
        Expr::Leq(a, b) => eval_comparison(ctx, a, b, |x, y| x <= y),
        Expr::Gt(a, b) => eval_comparison(ctx, a, b, |x, y| x > y),
        Expr::Geq(a, b) => eval_comparison(ctx, a, b, |x, y| x >= y),

        // === Arithmetic ===
        // Basic arithmetic - use eval_arith helper with SmallInt fast path
        Expr::Add(a, b) => eval_arith(ctx, a, b, i64::checked_add, |x, y| x + y),
        Expr::Sub(a, b) => eval_arith(ctx, a, b, i64::checked_sub, |x, y| x - y),
        Expr::Mul(a, b) => eval_arith(ctx, a, b, i64::checked_mul, |x, y| x * y),

        // Division operations - use eval_div helper (includes zero check)
        Expr::Div(a, b) => eval_div(ctx, a, b, i64::checked_div, |x, y| x / y, span),
        Expr::IntDiv(a, b) => eval_div(
            ctx,
            a,
            b,
            |x, y| Some(x.div_euclid(y)),
            |x, y| x.div_floor(&y),
            span,
        ),
        Expr::Mod(a, b) => eval_div(
            ctx,
            a,
            b,
            |x, y| Some(x.rem_euclid(y)),
            |x, y| {
                // Euclidean modulo (always non-negative for positive divisor)
                let r = x % &y;
                (r + &y) % &y
            },
            span,
        ),

        // Power - use eval_pow helper
        Expr::Pow(a, b) => eval_pow(ctx, a, b, span),
        Expr::Neg(a) => {
            let av = eval(ctx, a)?;
            // SmallInt fast path
            if let Value::SmallInt(n) = av {
                if let Some(result) = n.checked_neg() {
                    return Ok(Value::SmallInt(result));
                }
            }
            let an = av
                .to_bigint()
                .ok_or_else(|| EvalError::type_error("Int", &av, Some(a.span)))?;
            Ok(Value::big_int(-an))
        }
        Expr::Range(a, b) => {
            let av = eval(ctx, a)?;
            let bv = eval(ctx, b)?;
            let an = av
                .to_bigint()
                .ok_or_else(|| EvalError::type_error("Int", &av, Some(a.span)))?;
            let bn = bv
                .to_bigint()
                .ok_or_else(|| EvalError::type_error("Int", &bv, Some(b.span)))?;
            Ok(range_set(&an, &bn))
        }
    }
}

impl Clone for EvalCtx {
    fn clone(&self) -> Self {
        EvalCtx {
            shared: Arc::clone(&self.shared),
            env: self.env.clone(),
            next_state: self.next_state.clone(),
            local_ops: self.local_ops.clone(),
            // Clone local_stack to preserve bindings from enclosing scopes
            // (e.g., quantifier bindings that need to be visible in cloned contexts)
            local_stack: self.local_stack.clone(),
            state_env: self.state_env,
            next_state_env: self.next_state_env,
            recursion_depth: self.recursion_depth,
            instance_substitutions: self.instance_substitutions.clone(),
        }
    }
}

// === Helper functions ===

// === DyadicRationals helpers ===

/// Create a dyadic rational record [num |-> n, den |-> d]
fn make_dyadic_rational(num: i64, den: i64) -> Value {
    let fields = vec![
        ("num".to_string(), Value::int(num)),
        ("den".to_string(), Value::int(den)),
    ];
    Value::Record(fields.into())
}

/// Extract numerator and denominator from a dyadic rational record
fn extract_dyadic(v: &Value, span: Option<Span>) -> EvalResult<(i64, i64)> {
    if let Some(rec) = v.as_record() {
        let num = rec
            .get("num")
            .and_then(|v| v.as_int())
            .and_then(|n| n.to_i64())
            .ok_or_else(|| EvalError::Internal {
                message: "DyadicRational requires 'num' field as integer".into(),
                span,
            })?;
        let den = rec
            .get("den")
            .and_then(|v| v.as_int())
            .and_then(|n| n.to_i64())
            .ok_or_else(|| EvalError::Internal {
                message: "DyadicRational requires 'den' field as integer".into(),
                span,
            })?;
        Ok((num, den))
    } else {
        Err(EvalError::Internal {
            message: format!("Expected DyadicRational record, got {:?}", v),
            span,
        })
    }
}

/// GCD using Euclidean algorithm
fn gcd(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Reduce a fraction to lowest terms
fn reduce_fraction(num: i64, den: i64) -> (i64, i64) {
    if num == 0 {
        return (0, 1);
    }
    let g = gcd(num, den);
    (num / g, den / g)
}

/// Compare two values for equality, handling SetPred specially
/// SetPred values store lazy set comprehensions that need context to materialize
fn values_equal(ctx: &EvalCtx, a: &Value, b: &Value) -> EvalResult<bool> {
    use crate::value::SetPredValue;

    // Fast path: if neither is SetPred, use standard equality
    let a_is_setpred = matches!(a, Value::SetPred(_));
    let b_is_setpred = matches!(b, Value::SetPred(_));

    if !a_is_setpred && !b_is_setpred {
        return Ok(a == b);
    }

    // At least one is SetPred - need to materialize and compare extensionally
    // Both are set-like types, so compare by elements

    // Helper to materialize a SetPred value
    fn materialize_setpred(ctx: &EvalCtx, spv: &SetPredValue) -> EvalResult<OrdSet<Value>> {
        // Check if source is enumerable
        let iter = spv.source.iter_set().ok_or_else(|| EvalError::Internal {
            message: "Cannot enumerate infinite set for equality comparison".into(),
            span: None,
        })?;

        let mut result = OrdSet::new();
        for elem in iter {
            // Create context with bound variable
            let bound_name: Arc<str> = Arc::from(spv.bound.name.node.as_str());
            let elem_ctx = ctx.bind(bound_name, elem.clone());
            // Also restore captured environment
            let full_ctx = spv
                .env
                .iter()
                .fold(elem_ctx, |acc, (k, v)| acc.bind(k.clone(), v.clone()));
            // Evaluate predicate
            match eval(&full_ctx, &spv.pred) {
                Ok(pv) => {
                    if pv.as_bool().unwrap_or(false) {
                        result.insert(elem);
                    }
                }
                Err(EvalError::NotInDomain { .. }) | Err(EvalError::IndexOutOfBounds { .. }) => {
                    // Treat undefined as false (TLC behavior)
                }
                Err(e) => return Err(e),
            }
        }
        Ok(result)
    }

    // Get elements from both sides
    let a_elements: OrdSet<Value> = match a {
        Value::SetPred(spv) => materialize_setpred(ctx, spv)?,
        _ => a.to_ord_set().ok_or_else(|| EvalError::Internal {
            message: "Expected set value in equality comparison".into(),
            span: None,
        })?,
    };

    let b_elements: OrdSet<Value> = match b {
        Value::SetPred(spv) => materialize_setpred(ctx, spv)?,
        _ => b.to_ord_set().ok_or_else(|| EvalError::Internal {
            message: "Expected set value in equality comparison".into(),
            span: None,
        })?,
    };

    // Compare extensionally
    Ok(a_elements == b_elements)
}

/// Evaluate binary arithmetic expression with SmallInt fast path
fn eval_arith(
    ctx: &EvalCtx,
    a: &Spanned<Expr>,
    b: &Spanned<Expr>,
    small_op: impl Fn(i64, i64) -> Option<i64>,
    big_op: impl Fn(BigInt, BigInt) -> BigInt,
) -> EvalResult<Value> {
    let av = eval(ctx, a)?;
    let bv = eval(ctx, b)?;
    int_arith_op(av, bv, small_op, big_op, Some(a.span))
}

/// Evaluate division expression with SmallInt fast path and zero check
fn eval_div(
    ctx: &EvalCtx,
    a: &Spanned<Expr>,
    b: &Spanned<Expr>,
    small_op: impl Fn(i64, i64) -> Option<i64>,
    big_op: impl Fn(BigInt, BigInt) -> BigInt,
    span: Option<Span>,
) -> EvalResult<Value> {
    let av = eval(ctx, a)?;
    let bv = eval(ctx, b)?;
    int_div_op(av, bv, small_op, big_op, span)
}

/// Evaluate power expression with SmallInt fast path
fn eval_pow(
    ctx: &EvalCtx,
    a: &Spanned<Expr>,
    b: &Spanned<Expr>,
    span: Option<Span>,
) -> EvalResult<Value> {
    let av = eval(ctx, a)?;
    let bv = eval(ctx, b)?;
    int_pow_op(av, bv, span)
}

fn eval_comparison(
    ctx: &EvalCtx,
    a: &Spanned<Expr>,
    b: &Spanned<Expr>,
    cmp: impl Fn(&BigInt, &BigInt) -> bool,
) -> EvalResult<Value> {
    let av = eval(ctx, a)?;
    let bv = eval(ctx, b)?;
    // SmallInt fast path for comparisons
    if let (Value::SmallInt(an), Value::SmallInt(bn)) = (&av, &bv) {
        let a_big = BigInt::from(*an);
        let b_big = BigInt::from(*bn);
        return Ok(Value::Bool(cmp(&a_big, &b_big)));
    }
    let an = av
        .to_bigint()
        .ok_or_else(|| EvalError::type_error("Int", &av, Some(a.span)))?;
    let bn = bv
        .to_bigint()
        .ok_or_else(|| EvalError::type_error("Int", &bv, Some(b.span)))?;
    Ok(Value::Bool(cmp(&an, &bn)))
}

fn eval_apply(
    ctx: &EvalCtx,
    op_expr: &Spanned<Expr>,
    args: &[Spanned<Expr>],
    span: Option<Span>,
) -> EvalResult<Value> {
    // Check if it's an identifier (operator name or closure variable)
    if let Expr::Ident(name) = &op_expr.node {
        // First check if this is a closure bound in the environment
        // Use lookup() to check local_stack first for O(1) enumeration bindings
        if let Some(Value::Closure(closure)) = ctx.lookup(name) {
            return apply_closure(ctx, closure, args, span);
        }
        // If it's a non-closure value in env, fall through to check ops

        // Apply operator replacement if configured (e.g., Seq <- BoundedSeq)
        let resolved_name = ctx.resolve_op_name(name);

        // Check for user-defined operators (allows shadowing stdlib)
        if let Some(def) = ctx.get_op(resolved_name) {
            if def.params.len() != args.len() {
                return Err(EvalError::ArityMismatch {
                    op: resolved_name.to_string(),
                    expected: def.params.len(),
                    got: args.len(),
                    span,
                });
            }

            // Check if any parameter appears primed in the body
            // If so, use expression substitution (call-by-name) instead of value binding
            // This is required for TLA+ semantics like Action1(x,y) == x' = [x EXCEPT ![1] = y']
            let needs_substitution = def
                .params
                .iter()
                .any(|param| expr_has_primed_param(&def.body.node, &param.name.node));

            if needs_substitution {
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
                return eval(ctx, &substituted_body);
            }

            // No primed parameters - use call-by-value (faster)
            let mut bindings = Vec::new();
            for (param, arg) in def.params.iter().zip(args.iter()) {
                let value = if param.arity > 0 {
                    // Higher-order parameter - expects an operator argument
                    create_closure_from_arg(ctx, arg, &param.name.node, param.arity, span)?
                } else {
                    // Regular parameter - evaluate normally
                    eval(ctx, arg)?
                };
                bindings.push((Arc::from(param.name.node.as_str()), value));
            }

            // Operator result caching (issue #16).
            // Uses dependency-based validation rather than full-state fingerprinting.
            //
            // KEY: If an operator never reads primed variables, it will have no recorded
            // next-state deps, allowing cache hits across different next-state candidates.

            let args: Arc<[Value]> = bindings.iter().map(|(_, v)| v.clone()).collect::<Vec<_>>().into();
            let cache_key = OpResultCacheKey {
                shared_id: ctx.shared.id,
                local_ops_id: ctx
                    .local_ops
                    .as_ref()
                    .map(|a| Arc::as_ptr(a) as usize)
                    .unwrap_or(0),
                instance_subs_id: ctx
                    .instance_substitutions
                    .as_ref()
                    .map(|a| Arc::as_ptr(a) as usize)
                    .unwrap_or(0),
                op_name: resolved_name.to_string(),
                args: Arc::clone(&args),
            };

            // IMPORTANT: Only enable caching when `state_env` is set (array-based).
            // When callers bind state variables only via the HashMap `env`, we cannot validate
            // state-dependent operator results cheaply/safely in the general case.
            let caching_enabled = ctx.state_env.is_some();

            if caching_enabled {
                if let Some(result) = OP_RESULT_CACHE.with(|cache| {
                    let cache = cache.borrow();
                    let entry = cache.get(&cache_key)?;
                    if op_cache_entry_valid(ctx, entry) {
                        propagate_cached_deps(&entry.deps);
                        Some(entry.value.clone())
                    } else {
                        None
                    }
                }) {
                    return Ok(result);
                }
            }

            // Cache miss - evaluate and store deps for future reuse.
            let new_ctx = ctx.bind_all(bindings);
            let (result, deps) = eval_with_dep_tracking(&new_ctx, &def.body)?;

            if caching_enabled && !deps.inconsistent {
                OP_RESULT_CACHE.with(|cache| {
                    cache.borrow_mut().insert(
                        cache_key,
                        CachedOpResult {
                            value: result.clone(),
                            deps,
                        },
                    );
                });
            }

            return Ok(result);
        }

        // Check for built-in operators from stdlib (after user-defined)
        // Use resolved name for builtins too
        if let Some(result) = eval_builtin(ctx, resolved_name, args, span)? {
            return Ok(result);
        }

        // Undefined operator
        return Err(EvalError::UndefinedOp {
            name: resolved_name.to_string(),
            span,
        });
    }

    // If we get here with Apply, evaluate the operator expression
    // It might be a closure or other callable value
    let fv = eval(ctx, op_expr)?;
    if let Value::Closure(closure) = &fv {
        return apply_closure(ctx, closure, args, span);
    }

    Err(EvalError::Internal {
        message: format!("Cannot apply non-operator value: {:?}", fv),
        span,
    })
}

/// Collect all conjuncts from a conjunction expression.
/// Used for conjunct selection syntax like `Def!1`, `Def!2`, etc.
fn collect_conjuncts(expr: &Spanned<Expr>) -> Vec<Spanned<Expr>> {
    fn collect_inner(expr: &Spanned<Expr>, out: &mut Vec<Spanned<Expr>>) {
        match &expr.node {
            Expr::And(left, right) => {
                collect_inner(left, out);
                collect_inner(right, out);
            }
            _ => {
                out.push(expr.clone());
            }
        }
    }
    let mut result = Vec::new();
    collect_inner(expr, &mut result);
    result
}

/// Evaluate a module reference expression with any ModuleTarget type
///
/// Handles:
/// - Named: M!Op
/// - Parameterized: IS(x, y)!Op
/// - Chained: A!B!C!D
fn eval_module_ref_target(
    ctx: &EvalCtx,
    target: &ModuleTarget,
    op_name: &str,
    args: &[Spanned<Expr>],
    span: Option<Span>,
) -> EvalResult<Value> {
    match target {
        ModuleTarget::Named(name) => eval_module_ref(ctx, name, op_name, args, span),
        ModuleTarget::Parameterized(name, _params) => {
            // Parameterized instances like IS(x, y)!Op
            // For now, just use the instance name - parameterized instantiation
            // is handled during loading when the instance is registered
            eval_module_ref(ctx, name, op_name, args, span)
        }
        ModuleTarget::Chained(base_expr) => {
            // Chained module reference like A!B!C!D
            // First, resolve the base (A!B!C) to get the intermediate instance context
            eval_chained_module_ref(ctx, base_expr, op_name, args, span)
        }
    }
}

/// Evaluate a chained module reference (A!B!C!D)
///
/// This recursively resolves the chain by:
/// 1. Evaluating the base module reference to get the intermediate instance info
/// 2. Looking up the final operator within that instance's context
fn eval_chained_module_ref(
    ctx: &EvalCtx,
    base_expr: &Spanned<Expr>,
    final_op_name: &str,
    final_args: &[Spanned<Expr>],
    span: Option<Span>,
) -> EvalResult<Value> {
    // The base_expr should be a ModuleRef
    let Expr::ModuleRef(base_target, intermediate_op, _intermediate_args) = &base_expr.node else {
        return Err(EvalError::Internal {
            message: format!(
                "Chained module reference base must be ModuleRef, got: {:?}",
                base_expr.node
            ),
            span,
        });
    };

    // Recursively resolve the chain to get the innermost instance context
    // We need to get the InstanceInfo from evaluating the chain
    let instance_info = resolve_module_ref_chain(ctx, base_target, intermediate_op, span)?;

    // Now look up the final operator in that instance's module
    let op_def = ctx
        .get_instance_op(&instance_info.module_name, final_op_name)
        .ok_or_else(|| EvalError::UndefinedOp {
            name: format!("...!{}", final_op_name),
            span,
        })?
        .clone();

    // Check arity
    if op_def.params.len() != final_args.len() {
        return Err(EvalError::ArityMismatch {
            op: format!("...!{}", final_op_name),
            expected: op_def.params.len(),
            got: final_args.len(),
            span,
        });
    }

    // Get instance-local operators for nested references
    let instance_local_ops: OpEnv = ctx
        .shared
        .instance_ops
        .get(&instance_info.module_name)
        .cloned()
        .unwrap_or_default();

    // Build argument bindings
    let mut bindings = Vec::new();
    for (param, arg) in op_def.params.iter().zip(final_args.iter()) {
        let value = eval(ctx, arg)?;
        bindings.push((Arc::from(param.name.node.as_str()), value));
    }

    // Evaluate with instance context
    let new_ctx = ctx
        .with_local_ops(instance_local_ops)
        .bind_all(bindings)
        .with_instance_substitutions(instance_info.substitutions);

    eval(&new_ctx, &op_def.body)
}

/// Resolve a module reference chain to get the final InstanceInfo
///
/// For A!B!C, this resolves:
/// 1. A -> get InstanceInfo for A
/// 2. B in A's module -> get InstanceInfo for B (with A's substitutions composed)
fn resolve_module_ref_chain(
    ctx: &EvalCtx,
    target: &ModuleTarget,
    op_name: &str,
    span: Option<Span>,
) -> EvalResult<InstanceInfo> {
    match target {
        ModuleTarget::Named(name) => {
            // Base case: simple instance reference
            let instance_info = get_instance_info(ctx, name, op_name, span)?;
            Ok(instance_info)
        }
        ModuleTarget::Parameterized(name, _params) => {
            // Parameterized instance - treat like named for now
            let instance_info = get_instance_info(ctx, name, op_name, span)?;
            Ok(instance_info)
        }
        ModuleTarget::Chained(base_expr) => {
            // Recursive case: first resolve the base chain
            let Expr::ModuleRef(base_target, base_op, _) = &base_expr.node else {
                return Err(EvalError::Internal {
                    message: "Chained base must be ModuleRef".to_string(),
                    span,
                });
            };

            // Resolve the base chain first
            let base_info = resolve_module_ref_chain(ctx, base_target, base_op, span)?;

            // Now look up op_name in base_info's module
            // It should be an operator that evaluates to an InstanceExpr
            let op_def = ctx
                .get_instance_op(&base_info.module_name, op_name)
                .ok_or_else(|| EvalError::UndefinedOp {
                    name: format!("...!{}", op_name),
                    span,
                })?
                .clone();

            // The operator body should be an InstanceExpr
            // Apply base substitutions first
            let substituted = apply_substitutions(&op_def.body, &base_info.substitutions);

            match &substituted.node {
                Expr::InstanceExpr(module_name, subs) => {
                    // Compose the substitutions
                    let composed_subs = compose_substitutions(subs, Some(&base_info.substitutions));
                    Ok(InstanceInfo {
                        module_name: module_name.clone(),
                        substitutions: composed_subs,
                    })
                }
                _ => Err(EvalError::UndefinedOp {
                    name: format!(
                        "{}!{} is not an instance definition",
                        base_info.module_name, op_name
                    ),
                    span,
                }),
            }
        }
    }
}

/// Helper to get InstanceInfo from instance name and operator name
fn get_instance_info(
    ctx: &EvalCtx,
    instance_name: &str,
    op_name: &str,
    span: Option<Span>,
) -> EvalResult<InstanceInfo> {
    // Check if the "instance" is actually an operator that defines an instance
    if let Some(def) = ctx.get_op(instance_name) {
        if let Expr::InstanceExpr(module_name, substitutions) = &def.body.node {
            // If we're already evaluating inside an instance, compose this instance's
            // substitutions through the outer instance's substitutions.
            let effective_instance_subs =
                compose_substitutions(substitutions, ctx.instance_substitutions());

            // Now look up op_name in this module - it should be an InstanceExpr too
            let op_def = ctx
                .get_instance_op(module_name, op_name)
                .ok_or_else(|| EvalError::UndefinedOp {
                    name: format!("{}!{}", instance_name, op_name),
                    span,
                })?
                .clone();

            // Apply the instance's substitutions to the operator body
            let substituted = apply_substitutions(&op_def.body, &effective_instance_subs);

            match &substituted.node {
                Expr::InstanceExpr(inner_module, inner_subs) => {
                    // Compose substitutions through nested instances.
                    let composed =
                        compose_substitutions(inner_subs, Some(&effective_instance_subs));
                    return Ok(InstanceInfo {
                        module_name: inner_module.clone(),
                        substitutions: composed,
                    });
                }
                _ => {
                    return Err(EvalError::UndefinedOp {
                        name: format!(
                            "{}!{} is not an instance definition",
                            instance_name, op_name
                        ),
                        span,
                    });
                }
            }
        }
    }

    // Try looking up as a registered instance
    if let Some(info) = ctx.get_instance(instance_name) {
        let effective_instance_subs =
            compose_substitutions(&info.substitutions, ctx.instance_substitutions());

        let op_def = ctx
            .get_instance_op(&info.module_name, op_name)
            .ok_or_else(|| EvalError::UndefinedOp {
                name: format!("{}!{}", instance_name, op_name),
                span,
            })?
            .clone();

        // Apply the instance's substitutions to the operator body
        let substituted = apply_substitutions(&op_def.body, &effective_instance_subs);

        match &substituted.node {
            Expr::InstanceExpr(inner_module, inner_subs) => {
                let composed = compose_substitutions(inner_subs, Some(&effective_instance_subs));
                return Ok(InstanceInfo {
                    module_name: inner_module.clone(),
                    substitutions: composed,
                });
            }
            _ => {
                return Err(EvalError::UndefinedOp {
                    name: format!(
                        "{}!{} is not an instance definition",
                        instance_name, op_name
                    ),
                    span,
                });
            }
        }
    }

    Err(EvalError::UndefinedOp {
        name: format!("{}!{}", instance_name, op_name),
        span,
    })
}

/// Compose instance substitutions
fn compose_substitutions(
    inner_subs: &[Substitution],
    outer_subs: Option<&[Substitution]>,
) -> Vec<Substitution> {
    let Some(outer_subs) = outer_subs else {
        return inner_subs.to_vec();
    };
    if outer_subs.is_empty() {
        return inner_subs.to_vec();
    }

    // TLC composes substitutions through nested instances by:
    // 1) translating inner substitution RHS into the outer context, and
    // 2) inheriting any outer substitutions not overridden by the inner instance.
    let mut combined = Vec::with_capacity(inner_subs.len() + outer_subs.len());
    let mut overridden: std::collections::HashSet<&str> = std::collections::HashSet::new();

    for sub in inner_subs {
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

/// Evaluate a module reference expression (M!Op or M!Op(args))
///
/// This handles INSTANCE WITH substitutions:
/// - Looks up the instance by name to get module name and substitutions
/// - Finds the operator in the instanced module
/// - Applies substitutions to the operator body
/// - Evaluates with any provided arguments
fn eval_module_ref(
    ctx: &EvalCtx,
    instance_name: &str,
    op_name: &str,
    args: &[Spanned<Expr>],
    span: Option<Span>,
) -> EvalResult<Value> {
    // Check for conjunct selection: Def!n where Def is an operator with a conjunction body
    // and n is a numeric index (1-based)
    if let Ok(conjunct_idx) = op_name.parse::<usize>() {
        if conjunct_idx > 0 {
            // This might be conjunct selection, not module reference
            if let Some(def) = ctx.get_op(instance_name) {
                // Check if the operator body is a conjunction
                if let Expr::And(_, _) = &def.body.node {
                    // Collect all conjuncts from the definition
                    let conjuncts = collect_conjuncts(&def.body);
                    let idx = conjunct_idx - 1; // Convert to 0-based
                    if idx < conjuncts.len() {
                        // Evaluate the selected conjunct
                        return eval(ctx, &conjuncts[idx]);
                    } else {
                        return Err(EvalError::UndefinedOp {
                            name: format!(
                                "{}!{} (conjunct index {} out of range, definition has {} conjuncts)",
                                instance_name, op_name, conjunct_idx, conjuncts.len()
                            ),
                            span,
                        });
                    }
                }
            }
        }
    }

    // Resolve the instance.
    //
    // TLC allows nested named instances: when evaluating `Outer!Op` we must be able to resolve
    // instance references inside that module (e.g., `cleanInstance!Spec`) even if those
    // instances are not declared in the main module.
    //
    // We support this by:
    // 1) looking for a globally-registered instance (from loading the main/extended modules), and
    // 2) falling back to a locally-visible operator definition whose body is an `InstanceExpr`
    //    (available via `instance_local_ops` when evaluating an instanced module).
    let instance_info: InstanceInfo = if let Some(info) = ctx.get_instance(instance_name) {
        info.clone()
    } else if let Some(def) = ctx.get_op(instance_name) {
        match &def.body.node {
            Expr::InstanceExpr(module_name, substitutions) => InstanceInfo {
                module_name: module_name.clone(),
                substitutions: substitutions.clone(),
            },
            _ => {
                return Err(EvalError::UndefinedOp {
                    name: format!("{}!{}", instance_name, op_name),
                    span,
                });
            }
        }
    } else {
        return Err(EvalError::UndefinedOp {
            name: format!("{}!{}", instance_name, op_name),
            span,
        });
    };

    // Get the operator from the instanced module
    // If not found, check for implicit substitution: when INSTANCE M is used without WITH,
    // VARIABLEs and CONSTANTs in M with matching names in the current scope are substituted.
    let op_def = match ctx.get_instance_op(&instance_info.module_name, op_name) {
        Some(def) => def.clone(),
        None => {
            // Check for explicit substitution first
            for sub in &instance_info.substitutions {
                if sub.from.node == op_name {
                    // Evaluate the substituted expression
                    return eval(ctx, &sub.to);
                }
            }

            // Check for implicit substitution: the current context might have an operator
            // with the same name that implicitly substitutes for a VARIABLE/CONSTANT
            // in the instanced module
            if let Some(outer_def) = ctx.get_op(op_name) {
                if outer_def.params.len() == args.len() {
                    // Use the outer module's operator as implicit substitution
                    if args.is_empty() {
                        return eval(ctx, &outer_def.body);
                    } else {
                        // With arguments, bind them and evaluate
                        let mut bindings = Vec::new();
                        for (param, arg) in outer_def.params.iter().zip(args.iter()) {
                            let value = eval(ctx, arg)?;
                            bindings.push((Arc::from(param.name.node.as_str()), value));
                        }
                        let new_ctx = ctx.bind_all(bindings);
                        return eval(&new_ctx, &outer_def.body);
                    }
                }
            }

            return Err(EvalError::UndefinedOp {
                name: format!("{}!{}", instance_name, op_name),
                span,
            });
        }
    };

    // Check arity
    if op_def.params.len() != args.len() {
        return Err(EvalError::ArityMismatch {
            op: format!("{}!{}", instance_name, op_name),
            expected: op_def.params.len(),
            got: args.len(),
            span,
        });
    }

    // Compose INSTANCE substitutions through nested instances.
    //
    // - `instance_info.substitutions` maps names in `instance_info.module_name` into expressions
    //   in the *current* module context.
    // - `ctx.instance_substitutions()` maps names in the current module into expressions in the
    //   *outer* module context (if we're already evaluating inside an instance).
    //
    // TLC composes these so that nested module references see the outer substitutions as well.
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
        let mut overridden: std::collections::HashSet<&str> = std::collections::HashSet::new();

        // First, take the inner substitutions, translating their RHS into the outer context.
        for sub in instance_subs {
            overridden.insert(sub.from.node.as_str());
            combined.push(Substitution {
                from: sub.from.clone(),
                to: apply_substitutions(&sub.to, outer_subs),
            });
        }

        // Then, inherit any outer substitutions not overridden by the inner instance.
        for sub in outer_subs {
            if overridden.contains(sub.from.node.as_str()) {
                continue;
            }
            combined.push(sub.clone());
        }

        combined
    }

    let effective_substitutions =
        compose_instance_substitutions(&instance_info.substitutions, ctx.instance_substitutions());

    // Evaluate instance module operators in a scope where unqualified operator names
    // resolve to the instanced module's definitions (not the current module's).
    //
    // This is required for specs like EWD998PCal where the property references
    // `EWD998!Next`, and `Next` calls `SendMsg`, `RecvMsg`, etc., which must resolve
    // within module `EWD998` even when the current module also defines operators
    // with the same names.
    let instance_local_ops: OpEnv = ctx
        .shared
        .instance_ops
        .get(&instance_info.module_name)
        .cloned()
        .unwrap_or_default();

    // Bind parameters to arguments
    let mut bindings = Vec::new();
    for (param, arg) in op_def.params.iter().zip(args.iter()) {
        let value = eval(ctx, arg)?;
        bindings.push((Arc::from(param.name.node.as_str()), value));
    }

    // Evaluate the operator body with parameter bindings and instance-local operator scope.
    //
    // INSTANCE substitutions are kept active in the evaluation context; the evaluator applies
    // them at identifier resolution time (with per-eval caching to avoid repeated recomputation).
    let new_ctx = ctx
        .with_local_ops(instance_local_ops)
        .bind_all(bindings)
        .with_instance_substitutions(effective_substitutions);
    eval(&new_ctx, &op_def.body)
}

/// Check if an expression contains ANY Prime expressions (primed variables).
/// Used to determine whether operator result caching needs next_state context.
pub fn expr_has_any_prime(expr: &Expr) -> bool {
    match expr {
        Expr::Prime(_) => true,
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
        | Expr::FuncSet(a, b) => expr_has_any_prime(&a.node) || expr_has_any_prime(&b.node),
        Expr::Not(a)
        | Expr::Neg(a)
        | Expr::Powerset(a)
        | Expr::BigUnion(a)
        | Expr::Domain(a)
        | Expr::Always(a)
        | Expr::Eventually(a)
        | Expr::Enabled(a) => expr_has_any_prime(&a.node),
        // UNCHANGED(x) means x' = x, which implicitly contains a primed variable
        Expr::Unchanged(_) => true,
        Expr::FuncApply(f, arg) => expr_has_any_prime(&f.node) || expr_has_any_prime(&arg.node),
        Expr::If(c, t, e) => {
            expr_has_any_prime(&c.node) || expr_has_any_prime(&t.node) || expr_has_any_prime(&e.node)
        }
        Expr::Apply(op, args) => {
            expr_has_any_prime(&op.node) || args.iter().any(|a| expr_has_any_prime(&a.node))
        }
        Expr::Tuple(elems) | Expr::SetEnum(elems) | Expr::Times(elems) => {
            elems.iter().any(|e| expr_has_any_prime(&e.node))
        }
        Expr::Record(fields) | Expr::RecordSet(fields) => {
            fields.iter().any(|(_, e)| expr_has_any_prime(&e.node))
        }
        Expr::RecordAccess(r, _) => expr_has_any_prime(&r.node),
        Expr::Except(base, specs) => {
            expr_has_any_prime(&base.node)
                || specs.iter().any(|s| {
                    s.path.iter().any(|p| match p {
                        ExceptPathElement::Index(idx) => expr_has_any_prime(&idx.node),
                        ExceptPathElement::Field(_) => false,
                    }) || expr_has_any_prime(&s.value.node)
                })
        }
        Expr::Forall(bounds, body) | Expr::Exists(bounds, body) => {
            bounds.iter().any(|b| {
                b.domain
                    .as_ref()
                    .map(|d| expr_has_any_prime(&d.node))
                    .unwrap_or(false)
            }) || expr_has_any_prime(&body.node)
        }
        Expr::SetBuilder(e, bounds) => {
            bounds.iter().any(|b| {
                b.domain
                    .as_ref()
                    .map(|d| expr_has_any_prime(&d.node))
                    .unwrap_or(false)
            }) || expr_has_any_prime(&e.node)
        }
        Expr::Choose(bound, pred) => {
            bound
                .domain
                .as_ref()
                .map(|d| expr_has_any_prime(&d.node))
                .unwrap_or(false)
                || expr_has_any_prime(&pred.node)
        }
        Expr::Let(defs, body) => {
            defs.iter().any(|d| expr_has_any_prime(&d.body.node)) || expr_has_any_prime(&body.node)
        }
        Expr::Case(arms, other) => {
            arms.iter()
                .any(|arm| expr_has_any_prime(&arm.guard.node) || expr_has_any_prime(&arm.body.node))
                || other
                    .as_ref()
                    .map(|o| expr_has_any_prime(&o.node))
                    .unwrap_or(false)
        }
        Expr::Lambda(_, body) => expr_has_any_prime(&body.node),
        Expr::FuncDef(bounds, body) => {
            bounds.iter().any(|b| {
                b.domain
                    .as_ref()
                    .map(|d| expr_has_any_prime(&d.node))
                    .unwrap_or(false)
            }) || expr_has_any_prime(&body.node)
        }
        Expr::SetFilter(bound, pred) => {
            bound
                .domain
                .as_ref()
                .map(|d| expr_has_any_prime(&d.node))
                .unwrap_or(false)
                || expr_has_any_prime(&pred.node)
        }
        Expr::InstanceExpr(_, _) => false,
        Expr::Ident(_) | Expr::Int(_) | Expr::String(_) | Expr::Bool(_) | Expr::OpRef(_) => false,
        Expr::ModuleRef(_, _, args) => args.iter().any(|a| expr_has_any_prime(&a.node)),
    }
}

/// Check if an expression contains a primed reference to a specific parameter name.
/// This is used to detect when call-by-name semantics are needed for operator evaluation.
///
/// For example, in `Action1(c,d) == c' = [c EXCEPT ![1] = d']`:
/// - `expr_has_primed_param(body, "d")` returns true because `d'` appears
/// - `expr_has_primed_param(body, "c")` returns false (c appears but not primed as c')
pub fn expr_has_primed_param(expr: &Expr, param_name: &str) -> bool {
    match expr {
        // Check if this is Prime(Ident(param_name))
        Expr::Prime(inner) => {
            if let Expr::Ident(name) = &inner.node {
                if name == param_name {
                    return true;
                }
            }
            // Also check recursively inside the primed expression
            expr_has_primed_param(&inner.node, param_name)
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
        | Expr::FuncSet(a, b)
        | Expr::WeakFair(a, b)
        | Expr::StrongFair(a, b)
        | Expr::LeadsTo(a, b) => {
            expr_has_primed_param(&a.node, param_name) || expr_has_primed_param(&b.node, param_name)
        }
        Expr::Not(a)
        | Expr::Neg(a)
        | Expr::Powerset(a)
        | Expr::BigUnion(a)
        | Expr::Domain(a)
        | Expr::Always(a)
        | Expr::Eventually(a)
        | Expr::Enabled(a)
        | Expr::Unchanged(a) => expr_has_primed_param(&a.node, param_name),
        Expr::FuncApply(f, arg) => {
            expr_has_primed_param(&f.node, param_name)
                || expr_has_primed_param(&arg.node, param_name)
        }
        Expr::If(c, t, e) => {
            expr_has_primed_param(&c.node, param_name)
                || expr_has_primed_param(&t.node, param_name)
                || expr_has_primed_param(&e.node, param_name)
        }
        Expr::Apply(op, args) => {
            expr_has_primed_param(&op.node, param_name)
                || args
                    .iter()
                    .any(|a| expr_has_primed_param(&a.node, param_name))
        }
        Expr::Tuple(elems) | Expr::SetEnum(elems) | Expr::Times(elems) => elems
            .iter()
            .any(|e| expr_has_primed_param(&e.node, param_name)),
        Expr::Record(fields) | Expr::RecordSet(fields) => fields
            .iter()
            .any(|(_, e)| expr_has_primed_param(&e.node, param_name)),
        Expr::RecordAccess(r, _) => expr_has_primed_param(&r.node, param_name),
        Expr::Except(base, specs) => {
            expr_has_primed_param(&base.node, param_name)
                || specs.iter().any(|s| {
                    s.path.iter().any(|p| match p {
                        ExceptPathElement::Index(idx) => {
                            expr_has_primed_param(&idx.node, param_name)
                        }
                        ExceptPathElement::Field(_) => false,
                    }) || expr_has_primed_param(&s.value.node, param_name)
                })
        }
        Expr::Forall(bounds, body) | Expr::Exists(bounds, body) => {
            // Don't recurse into body if param_name is shadowed by a bound var
            let is_shadowed = bounds.iter().any(|b| b.name.node == param_name);
            if is_shadowed {
                // Only check domains
                bounds.iter().any(|b| {
                    b.domain
                        .as_ref()
                        .map(|d| expr_has_primed_param(&d.node, param_name))
                        .unwrap_or(false)
                })
            } else {
                bounds.iter().any(|b| {
                    b.domain
                        .as_ref()
                        .map(|d| expr_has_primed_param(&d.node, param_name))
                        .unwrap_or(false)
                }) || expr_has_primed_param(&body.node, param_name)
            }
        }
        Expr::SetBuilder(e, bounds) => {
            let is_shadowed = bounds.iter().any(|b| b.name.node == param_name);
            if is_shadowed {
                bounds.iter().any(|b| {
                    b.domain
                        .as_ref()
                        .map(|d| expr_has_primed_param(&d.node, param_name))
                        .unwrap_or(false)
                })
            } else {
                bounds.iter().any(|b| {
                    b.domain
                        .as_ref()
                        .map(|d| expr_has_primed_param(&d.node, param_name))
                        .unwrap_or(false)
                }) || expr_has_primed_param(&e.node, param_name)
            }
        }
        Expr::Choose(bound, pred) => {
            let is_shadowed = bound.name.node == param_name;
            if is_shadowed {
                bound
                    .domain
                    .as_ref()
                    .map(|d| expr_has_primed_param(&d.node, param_name))
                    .unwrap_or(false)
            } else {
                bound
                    .domain
                    .as_ref()
                    .map(|d| expr_has_primed_param(&d.node, param_name))
                    .unwrap_or(false)
                    || expr_has_primed_param(&pred.node, param_name)
            }
        }
        Expr::Let(defs, body) => {
            // Check if param is shadowed by any LET definition
            let is_shadowed = defs.iter().any(|d| d.name.node == param_name);
            if is_shadowed {
                defs.iter()
                    .any(|d| expr_has_primed_param(&d.body.node, param_name))
            } else {
                defs.iter()
                    .any(|d| expr_has_primed_param(&d.body.node, param_name))
                    || expr_has_primed_param(&body.node, param_name)
            }
        }
        Expr::Case(arms, other) => {
            arms.iter().any(|arm| {
                expr_has_primed_param(&arm.guard.node, param_name)
                    || expr_has_primed_param(&arm.body.node, param_name)
            }) || other
                .as_ref()
                .map(|o| expr_has_primed_param(&o.node, param_name))
                .unwrap_or(false)
        }
        Expr::Lambda(_, body) => expr_has_primed_param(&body.node, param_name),
        Expr::FuncDef(bounds, body) => {
            let is_shadowed = bounds.iter().any(|b| b.name.node == param_name);
            if is_shadowed {
                bounds.iter().any(|b| {
                    b.domain
                        .as_ref()
                        .map(|d| expr_has_primed_param(&d.node, param_name))
                        .unwrap_or(false)
                })
            } else {
                bounds.iter().any(|b| {
                    b.domain
                        .as_ref()
                        .map(|d| expr_has_primed_param(&d.node, param_name))
                        .unwrap_or(false)
                }) || expr_has_primed_param(&body.node, param_name)
            }
        }
        Expr::SetFilter(bound, pred) => {
            let is_shadowed = bound.name.node == param_name;
            if is_shadowed {
                bound
                    .domain
                    .as_ref()
                    .map(|d| expr_has_primed_param(&d.node, param_name))
                    .unwrap_or(false)
            } else {
                bound
                    .domain
                    .as_ref()
                    .map(|d| expr_has_primed_param(&d.node, param_name))
                    .unwrap_or(false)
                    || expr_has_primed_param(&pred.node, param_name)
            }
        }
        // InstanceExpr - just has substitutions which don't contain primed params in normal usage
        Expr::InstanceExpr(_, _) => false,
        // Leaf nodes - don't contain primed params
        Expr::Ident(_) | Expr::Int(_) | Expr::String(_) | Expr::Bool(_) | Expr::OpRef(_) => false,
        // ModuleRef can contain primed params in its args
        Expr::ModuleRef(_, _, args) => args
            .iter()
            .any(|a| expr_has_primed_param(&a.node, param_name)),
    }
}

/// Apply substitutions to an expression
///
/// For INSTANCE Channel WITH Data <- Message, chan <- in:
/// - Replace all occurrences of "Data" with the expression "Message"
/// - Replace all occurrences of "chan" with the expression "in"
pub fn apply_substitutions(expr: &Spanned<Expr>, subs: &[Substitution]) -> Spanned<Expr> {
    // Build a map of substitutions
    let sub_map: std::collections::HashMap<&str, &Spanned<Expr>> =
        subs.iter().map(|s| (s.from.node.as_str(), &s.to)).collect();

    substitute_in_expr(expr, &sub_map)
}

/// Get all bound names from a BoundVar (includes main name and pattern names)
fn get_bound_names(bv: &BoundVar) -> Vec<&str> {
    let mut names = vec![bv.name.node.as_str()];
    if let Some(pattern) = &bv.pattern {
        match pattern {
            BoundPattern::Var(v) => names.push(v.node.as_str()),
            BoundPattern::Tuple(vs) => names.extend(vs.iter().map(|v| v.node.as_str())),
        }
    }
    names
}

/// Recursively substitute identifiers in an expression
fn substitute_in_expr(
    expr: &Spanned<Expr>,
    subs: &std::collections::HashMap<&str, &Spanned<Expr>>,
) -> Spanned<Expr> {
    let new_node = match &expr.node {
        Expr::Ident(name) => {
            // If this identifier should be substituted, return the substitution
            if let Some(replacement) = subs.get(name.as_str()) {
                return (*replacement).clone();
            }
            Expr::Ident(name.clone())
        }
        // Recursively substitute in compound expressions
        Expr::Apply(op, args) => {
            let new_op = Box::new(substitute_in_expr(op, subs));
            let new_args: Vec<_> = args.iter().map(|a| substitute_in_expr(a, subs)).collect();
            Expr::Apply(new_op, new_args)
        }
        Expr::And(a, b) => Expr::And(
            Box::new(substitute_in_expr(a, subs)),
            Box::new(substitute_in_expr(b, subs)),
        ),
        Expr::Or(a, b) => Expr::Or(
            Box::new(substitute_in_expr(a, subs)),
            Box::new(substitute_in_expr(b, subs)),
        ),
        Expr::Not(a) => Expr::Not(Box::new(substitute_in_expr(a, subs))),
        Expr::Implies(a, b) => Expr::Implies(
            Box::new(substitute_in_expr(a, subs)),
            Box::new(substitute_in_expr(b, subs)),
        ),
        Expr::Equiv(a, b) => Expr::Equiv(
            Box::new(substitute_in_expr(a, subs)),
            Box::new(substitute_in_expr(b, subs)),
        ),
        Expr::Eq(a, b) => Expr::Eq(
            Box::new(substitute_in_expr(a, subs)),
            Box::new(substitute_in_expr(b, subs)),
        ),
        Expr::Neq(a, b) => Expr::Neq(
            Box::new(substitute_in_expr(a, subs)),
            Box::new(substitute_in_expr(b, subs)),
        ),
        Expr::In(a, b) => Expr::In(
            Box::new(substitute_in_expr(a, subs)),
            Box::new(substitute_in_expr(b, subs)),
        ),
        Expr::NotIn(a, b) => Expr::NotIn(
            Box::new(substitute_in_expr(a, subs)),
            Box::new(substitute_in_expr(b, subs)),
        ),
        Expr::Prime(a) => Expr::Prime(Box::new(substitute_in_expr(a, subs))),
        Expr::RecordAccess(r, field) => {
            Expr::RecordAccess(Box::new(substitute_in_expr(r, subs)), field.clone())
        }
        Expr::FuncApply(f, arg) => Expr::FuncApply(
            Box::new(substitute_in_expr(f, subs)),
            Box::new(substitute_in_expr(arg, subs)),
        ),
        Expr::If(cond, then_e, else_e) => Expr::If(
            Box::new(substitute_in_expr(cond, subs)),
            Box::new(substitute_in_expr(then_e, subs)),
            Box::new(substitute_in_expr(else_e, subs)),
        ),
        Expr::SetEnum(elems) => {
            let new_elems: Vec<_> = elems.iter().map(|e| substitute_in_expr(e, subs)).collect();
            Expr::SetEnum(new_elems)
        }
        Expr::Tuple(elems) => {
            let new_elems: Vec<_> = elems.iter().map(|e| substitute_in_expr(e, subs)).collect();
            Expr::Tuple(new_elems)
        }
        Expr::Record(fields) => {
            let new_fields: Vec<_> = fields
                .iter()
                .map(|(name, e)| (name.clone(), substitute_in_expr(e, subs)))
                .collect();
            Expr::Record(new_fields)
        }
        Expr::Except(base, specs) => {
            let new_base = Box::new(substitute_in_expr(base, subs));
            let new_specs: Vec<_> = specs
                .iter()
                .map(|spec| ExceptSpec {
                    path: spec
                        .path
                        .iter()
                        .map(|elem| match elem {
                            ExceptPathElement::Index(idx_expr) => {
                                ExceptPathElement::Index(substitute_in_expr(idx_expr, subs))
                            }
                            ExceptPathElement::Field(f) => ExceptPathElement::Field(f.clone()),
                        })
                        .collect(),
                    value: substitute_in_expr(&spec.value, subs),
                })
                .collect();
            Expr::Except(new_base, new_specs)
        }
        Expr::Forall(bounds, body) => {
            // Don't substitute bound variables in the body
            let bound_names: std::collections::HashSet<_> =
                bounds.iter().flat_map(|b| get_bound_names(b)).collect();
            let filtered_subs: std::collections::HashMap<_, _> = subs
                .iter()
                .filter(|(k, _)| !bound_names.contains(*k))
                .map(|(&k, &v)| (k, v))
                .collect();
            let new_bounds: Vec<_> = bounds
                .iter()
                .map(|b| BoundVar {
                    name: b.name.clone(),
                    pattern: b.pattern.clone(),
                    domain: b
                        .domain
                        .as_ref()
                        .map(|d| Box::new(substitute_in_expr(d, subs))),
                })
                .collect();
            Expr::Forall(
                new_bounds,
                Box::new(substitute_in_expr(body, &filtered_subs)),
            )
        }
        Expr::Exists(bounds, body) => {
            let bound_names: std::collections::HashSet<_> =
                bounds.iter().flat_map(|b| get_bound_names(b)).collect();
            let filtered_subs: std::collections::HashMap<_, _> = subs
                .iter()
                .filter(|(k, _)| !bound_names.contains(*k))
                .map(|(&k, &v)| (k, v))
                .collect();
            let new_bounds: Vec<_> = bounds
                .iter()
                .map(|b| BoundVar {
                    name: b.name.clone(),
                    pattern: b.pattern.clone(),
                    domain: b
                        .domain
                        .as_ref()
                        .map(|d| Box::new(substitute_in_expr(d, subs))),
                })
                .collect();
            Expr::Exists(
                new_bounds,
                Box::new(substitute_in_expr(body, &filtered_subs)),
            )
        }
        Expr::Choose(bound, body) => {
            let bound_names: std::collections::HashSet<_> =
                get_bound_names(bound).into_iter().collect();
            let filtered_subs: std::collections::HashMap<_, _> = subs
                .iter()
                .filter(|(k, _)| !bound_names.contains(*k))
                .map(|(&k, &v)| (k, v))
                .collect();
            let new_bound = BoundVar {
                name: bound.name.clone(),
                pattern: bound.pattern.clone(),
                domain: bound
                    .domain
                    .as_ref()
                    .map(|d| Box::new(substitute_in_expr(d, subs))),
            };
            Expr::Choose(
                new_bound,
                Box::new(substitute_in_expr(body, &filtered_subs)),
            )
        }
        Expr::Let(defs, body) => {
            // Don't substitute bound operators in the body
            let bound_names: std::collections::HashSet<_> =
                defs.iter().map(|d| d.name.node.as_str()).collect();
            let filtered_subs: std::collections::HashMap<_, _> = subs
                .iter()
                .filter(|(k, _)| !bound_names.contains(*k))
                .map(|(&k, &v)| (k, v))
                .collect();
            let new_defs: Vec<_> = defs
                .iter()
                .map(|d| OperatorDef {
                    name: d.name.clone(),
                    params: d.params.clone(),
                    body: substitute_in_expr(&d.body, subs),
                    local: d.local,
                })
                .collect();
            Expr::Let(new_defs, Box::new(substitute_in_expr(body, &filtered_subs)))
        }
        Expr::FuncDef(bounds, body) => {
            let bound_names: std::collections::HashSet<_> =
                bounds.iter().flat_map(|b| get_bound_names(b)).collect();
            let filtered_subs: std::collections::HashMap<_, _> = subs
                .iter()
                .filter(|(k, _)| !bound_names.contains(*k))
                .map(|(&k, &v)| (k, v))
                .collect();
            let new_bounds: Vec<_> = bounds
                .iter()
                .map(|b| BoundVar {
                    name: b.name.clone(),
                    pattern: b.pattern.clone(),
                    domain: b
                        .domain
                        .as_ref()
                        .map(|d| Box::new(substitute_in_expr(d, subs))),
                })
                .collect();
            Expr::FuncDef(
                new_bounds,
                Box::new(substitute_in_expr(body, &filtered_subs)),
            )
        }
        Expr::SetBuilder(body, bounds) => {
            let bound_names: std::collections::HashSet<_> =
                bounds.iter().flat_map(|b| get_bound_names(b)).collect();
            let filtered_subs: std::collections::HashMap<_, _> = subs
                .iter()
                .filter(|(k, _)| !bound_names.contains(*k))
                .map(|(&k, &v)| (k, v))
                .collect();
            let new_bounds: Vec<_> = bounds
                .iter()
                .map(|b| BoundVar {
                    name: b.name.clone(),
                    pattern: b.pattern.clone(),
                    domain: b
                        .domain
                        .as_ref()
                        .map(|d| Box::new(substitute_in_expr(d, subs))),
                })
                .collect();
            Expr::SetBuilder(
                Box::new(substitute_in_expr(body, &filtered_subs)),
                new_bounds,
            )
        }
        Expr::SetFilter(bound, pred) => {
            let bound_names: std::collections::HashSet<_> =
                get_bound_names(bound).into_iter().collect();
            let filtered_subs: std::collections::HashMap<_, _> = subs
                .iter()
                .filter(|(k, _)| !bound_names.contains(*k))
                .map(|(&k, &v)| (k, v))
                .collect();
            let new_bound = BoundVar {
                name: bound.name.clone(),
                pattern: bound.pattern.clone(),
                domain: bound
                    .domain
                    .as_ref()
                    .map(|d| Box::new(substitute_in_expr(d, subs))),
            };
            Expr::SetFilter(
                new_bound,
                Box::new(substitute_in_expr(pred, &filtered_subs)),
            )
        }
        // Handle remaining binary operators
        Expr::Lt(a, b) => Expr::Lt(
            Box::new(substitute_in_expr(a, subs)),
            Box::new(substitute_in_expr(b, subs)),
        ),
        Expr::Leq(a, b) => Expr::Leq(
            Box::new(substitute_in_expr(a, subs)),
            Box::new(substitute_in_expr(b, subs)),
        ),
        Expr::Gt(a, b) => Expr::Gt(
            Box::new(substitute_in_expr(a, subs)),
            Box::new(substitute_in_expr(b, subs)),
        ),
        Expr::Geq(a, b) => Expr::Geq(
            Box::new(substitute_in_expr(a, subs)),
            Box::new(substitute_in_expr(b, subs)),
        ),
        Expr::Add(a, b) => Expr::Add(
            Box::new(substitute_in_expr(a, subs)),
            Box::new(substitute_in_expr(b, subs)),
        ),
        Expr::Sub(a, b) => Expr::Sub(
            Box::new(substitute_in_expr(a, subs)),
            Box::new(substitute_in_expr(b, subs)),
        ),
        Expr::Mul(a, b) => Expr::Mul(
            Box::new(substitute_in_expr(a, subs)),
            Box::new(substitute_in_expr(b, subs)),
        ),
        Expr::Div(a, b) => Expr::Div(
            Box::new(substitute_in_expr(a, subs)),
            Box::new(substitute_in_expr(b, subs)),
        ),
        Expr::IntDiv(a, b) => Expr::IntDiv(
            Box::new(substitute_in_expr(a, subs)),
            Box::new(substitute_in_expr(b, subs)),
        ),
        Expr::Mod(a, b) => Expr::Mod(
            Box::new(substitute_in_expr(a, subs)),
            Box::new(substitute_in_expr(b, subs)),
        ),
        Expr::Pow(a, b) => Expr::Pow(
            Box::new(substitute_in_expr(a, subs)),
            Box::new(substitute_in_expr(b, subs)),
        ),
        Expr::Range(a, b) => Expr::Range(
            Box::new(substitute_in_expr(a, subs)),
            Box::new(substitute_in_expr(b, subs)),
        ),
        Expr::Union(a, b) => Expr::Union(
            Box::new(substitute_in_expr(a, subs)),
            Box::new(substitute_in_expr(b, subs)),
        ),
        Expr::Intersect(a, b) => Expr::Intersect(
            Box::new(substitute_in_expr(a, subs)),
            Box::new(substitute_in_expr(b, subs)),
        ),
        Expr::SetMinus(a, b) => Expr::SetMinus(
            Box::new(substitute_in_expr(a, subs)),
            Box::new(substitute_in_expr(b, subs)),
        ),
        Expr::Subseteq(a, b) => Expr::Subseteq(
            Box::new(substitute_in_expr(a, subs)),
            Box::new(substitute_in_expr(b, subs)),
        ),
        Expr::Times(elems) => {
            let new_elems: Vec<_> = elems.iter().map(|e| substitute_in_expr(e, subs)).collect();
            Expr::Times(new_elems)
        }
        Expr::FuncSet(a, b) => Expr::FuncSet(
            Box::new(substitute_in_expr(a, subs)),
            Box::new(substitute_in_expr(b, subs)),
        ),
        Expr::RecordSet(fields) => {
            let new_fields: Vec<_> = fields
                .iter()
                .map(|(name, e)| (name.clone(), substitute_in_expr(e, subs)))
                .collect();
            Expr::RecordSet(new_fields)
        }
        Expr::Domain(a) => Expr::Domain(Box::new(substitute_in_expr(a, subs))),
        Expr::Powerset(a) => Expr::Powerset(Box::new(substitute_in_expr(a, subs))),
        Expr::BigUnion(a) => Expr::BigUnion(Box::new(substitute_in_expr(a, subs))),
        Expr::Neg(a) => Expr::Neg(Box::new(substitute_in_expr(a, subs))),
        Expr::Case(arms, other) => {
            let new_arms: Vec<_> = arms
                .iter()
                .map(|arm| tla_core::ast::CaseArm {
                    guard: substitute_in_expr(&arm.guard, subs),
                    body: substitute_in_expr(&arm.body, subs),
                })
                .collect();
            let new_other = other
                .as_ref()
                .map(|o| Box::new(substitute_in_expr(o, subs)));
            Expr::Case(new_arms, new_other)
        }
        Expr::Lambda(params, body) => {
            // Don't substitute bound parameters
            let bound_names: std::collections::HashSet<_> =
                params.iter().map(|p| p.node.as_str()).collect();
            let filtered_subs: std::collections::HashMap<_, _> = subs
                .iter()
                .filter(|(k, _)| !bound_names.contains(*k))
                .map(|(&k, &v)| (k, v))
                .collect();
            Expr::Lambda(
                params.clone(),
                Box::new(substitute_in_expr(body, &filtered_subs)),
            )
        }
        Expr::ModuleRef(m, op, args) => {
            let new_args: Vec<_> = args.iter().map(|a| substitute_in_expr(a, subs)).collect();
            Expr::ModuleRef(m.clone(), op.clone(), new_args)
        }
        // InstanceExpr shouldn't appear in expression bodies, but handle it for completeness
        Expr::InstanceExpr(module, inst_subs) => {
            let new_subs: Vec<_> = inst_subs
                .iter()
                .map(|sub| tla_core::ast::Substitution {
                    from: sub.from.clone(),
                    to: substitute_in_expr(&sub.to, subs),
                })
                .collect();
            Expr::InstanceExpr(module.clone(), new_subs)
        }
        // Temporal and action operators
        Expr::Always(a) => Expr::Always(Box::new(substitute_in_expr(a, subs))),
        Expr::Eventually(a) => Expr::Eventually(Box::new(substitute_in_expr(a, subs))),
        Expr::LeadsTo(a, b) => Expr::LeadsTo(
            Box::new(substitute_in_expr(a, subs)),
            Box::new(substitute_in_expr(b, subs)),
        ),
        Expr::WeakFair(vars, a) => Expr::WeakFair(
            Box::new(substitute_in_expr(vars, subs)),
            Box::new(substitute_in_expr(a, subs)),
        ),
        Expr::StrongFair(vars, a) => Expr::StrongFair(
            Box::new(substitute_in_expr(vars, subs)),
            Box::new(substitute_in_expr(a, subs)),
        ),
        Expr::Enabled(a) => Expr::Enabled(Box::new(substitute_in_expr(a, subs))),
        Expr::Unchanged(a) => Expr::Unchanged(Box::new(substitute_in_expr(a, subs))),
        // Literals don't need substitution
        Expr::Bool(b) => Expr::Bool(*b),
        Expr::Int(n) => Expr::Int(n.clone()),
        Expr::String(s) => Expr::String(s.clone()),
        // Operator references don't need substitution
        Expr::OpRef(op) => Expr::OpRef(op.clone()),
    };

    Spanned::new(new_node, expr.span)
}

/// Create a closure from an argument expression for a higher-order parameter
fn create_closure_from_arg(
    ctx: &EvalCtx,
    arg: &Spanned<Expr>,
    param_name: &str,
    expected_arity: usize,
    _span: Option<Span>,
) -> EvalResult<Value> {
    match &arg.node {
        Expr::Lambda(lambda_params, body) => {
            let params: Vec<String> = lambda_params.iter().map(|p| p.node.clone()).collect();
            if params.len() != expected_arity {
                return Err(EvalError::ArityMismatch {
                    op: format!("<lambda:{}>", param_name),
                    expected: expected_arity,
                    got: params.len(),
                    span: Some(arg.span),
                });
            }
            Ok(Value::Closure(ClosureValue::new(
                params,
                (**body).clone(),
                ctx.env.clone(),
            )))
        }
        Expr::OpRef(op) => {
            // Built-in operator passed as a higher-order argument (e.g., ReduceSet(\\cap, ...)).
            // Represent it as a closure whose body is the OpRef. `apply_closure(_with_values)`
            // has a fast path for OpRef bodies.
            if expected_arity != 2 {
                return Err(EvalError::Internal {
                    message: format!(
                        "Expected {}-ary operator for higher-order parameter '{}', got built-in '{}'",
                        expected_arity, param_name, op
                    ),
                    span: Some(arg.span),
                });
            }
            Ok(Value::Closure(ClosureValue::new(
                vec!["x".to_string(), "y".to_string()],
                Spanned {
                    node: Expr::OpRef(op.clone()),
                    span: arg.span,
                },
                ctx.env.clone(),
            )))
        }
        Expr::Ident(name) => {
            // Could be an operator name being passed as argument
            // Check if it's already a closure in env (use lookup for local_stack support)
            if let Some(Value::Closure(c)) = ctx.lookup(name) {
                if c.params.len() != expected_arity {
                    return Err(EvalError::ArityMismatch {
                        op: name.clone(),
                        expected: expected_arity,
                        got: c.params.len(),
                        span: Some(arg.span),
                    });
                }
                return Ok(Value::Closure(c.clone()));
            }
            // Check if it's a user-defined operator
            if let Some(def) = ctx.get_op(name) {
                if def.params.len() != expected_arity {
                    return Err(EvalError::ArityMismatch {
                        op: name.clone(),
                        expected: expected_arity,
                        got: def.params.len(),
                        span: Some(arg.span),
                    });
                }
                // Create a closure that wraps the operator
                // The operator's parameters become the closure's parameters
                let params: Vec<String> = def.params.iter().map(|p| p.name.node.clone()).collect();
                Ok(Value::Closure(ClosureValue::new(
                    params,
                    def.body.clone(),
                    ctx.env.clone(),
                )))
            } else {
                // Check if it might be a built-in stdlib operator (Add, Half, etc.)
                // Create a closure that calls the built-in via Apply
                if expected_arity == 2 {
                    // Create parameters for the closure
                    let params = vec!["__x".to_string(), "__y".to_string()];
                    // Create the body: Apply(name, [__x, __y])
                    let body = Spanned {
                        node: Expr::Apply(
                            Box::new(Spanned {
                                node: Expr::Ident(name.clone()),
                                span: arg.span,
                            }),
                            vec![
                                Spanned {
                                    node: Expr::Ident("__x".to_string()),
                                    span: arg.span,
                                },
                                Spanned {
                                    node: Expr::Ident("__y".to_string()),
                                    span: arg.span,
                                },
                            ],
                        ),
                        span: arg.span,
                    };
                    Ok(Value::Closure(ClosureValue::new(
                        params,
                        body,
                        ctx.env.clone(),
                    )))
                } else if expected_arity == 1 {
                    let params = vec!["__x".to_string()];
                    let body = Spanned {
                        node: Expr::Apply(
                            Box::new(Spanned {
                                node: Expr::Ident(name.clone()),
                                span: arg.span,
                            }),
                            vec![Spanned {
                                node: Expr::Ident("__x".to_string()),
                                span: arg.span,
                            }],
                        ),
                        span: arg.span,
                    };
                    Ok(Value::Closure(ClosureValue::new(
                        params,
                        body,
                        ctx.env.clone(),
                    )))
                } else {
                    Err(EvalError::Internal {
                        message: format!(
                            "Expected operator for higher-order parameter '{}', got undefined '{}'",
                            param_name, name
                        ),
                        span: Some(arg.span),
                    })
                }
            }
        }
        _ => Err(EvalError::Internal {
            message: format!(
                "Expected lambda or operator for higher-order parameter '{}'",
                param_name
            ),
            span: Some(arg.span),
        }),
    }
}

/// Apply a closure to arguments
fn apply_closure(
    ctx: &EvalCtx,
    closure: &ClosureValue,
    args: &[Spanned<Expr>],
    span: Option<Span>,
) -> EvalResult<Value> {
    if closure.params.len() != args.len() {
        return Err(EvalError::ArityMismatch {
            op: format!("<closure#{}>", closure.id),
            expected: closure.params.len(),
            got: args.len(),
            span,
        });
    }

    // Fast-path: closure wraps a built-in operator reference (OpRef).
    if closure.params.len() == 2 {
        if let Expr::OpRef(op) = &closure.body.node {
            let left = eval(ctx, &args[0])?;
            let right = eval(ctx, &args[1])?;
            return apply_builtin_binary_op(op, left, right, span);
        }
    }

    // Evaluate arguments and bind to closure parameters
    let mut bindings = Vec::new();
    for (param, arg) in closure.params.iter().zip(args.iter()) {
        bindings.push((Arc::from(param.as_str()), eval(ctx, arg)?));
    }

    // Create context with closure's captured environment + new bindings
    // The closure environment takes precedence, then we add new bindings
    let closure_ctx = EvalCtx {
        shared: Arc::clone(&ctx.shared),
        env: closure.env.clone(),
        next_state: ctx.next_state.clone(),
        local_ops: ctx.local_ops.clone(),
        local_stack: Vec::new(),
        state_env: ctx.state_env,
        next_state_env: ctx.next_state_env,
        recursion_depth: ctx.recursion_depth,
        instance_substitutions: ctx.instance_substitutions.clone(),
    };
    let ctx_with_bindings = closure_ctx.bind_all(bindings);
    eval(&ctx_with_bindings, &closure.body)
}

/// Apply a closure to already-evaluated arguments.
fn apply_closure_with_values(
    ctx: &EvalCtx,
    closure: &ClosureValue,
    args: &[Value],
    span: Option<Span>,
) -> EvalResult<Value> {
    if closure.params.len() != args.len() {
        return Err(EvalError::ArityMismatch {
            op: format!("<closure#{}>", closure.id),
            expected: closure.params.len(),
            got: args.len(),
            span,
        });
    }

    // Fast-path: closure wraps a built-in operator reference (OpRef).
    if closure.params.len() == 2 {
        if let Expr::OpRef(op) = &closure.body.node {
            return apply_builtin_binary_op(op, args[0].clone(), args[1].clone(), span);
        }
    }

    // Bind provided values to closure parameters.
    let bindings = closure
        .params
        .iter()
        .zip(args.iter())
        .map(|(param, value)| (Arc::from(param.as_str()), value.clone()))
        .collect::<Vec<_>>();

    // Create context with closure's captured environment + new bindings
    // The closure environment takes precedence, then we add new bindings
    let closure_ctx = EvalCtx {
        shared: Arc::clone(&ctx.shared),
        env: closure.env.clone(),
        next_state: ctx.next_state.clone(),
        local_ops: ctx.local_ops.clone(),
        local_stack: Vec::new(),
        state_env: ctx.state_env,
        next_state_env: ctx.next_state_env,
        recursion_depth: ctx.recursion_depth,
        instance_substitutions: ctx.instance_substitutions.clone(),
    };
    let ctx_with_bindings = closure_ctx.bind_all(bindings);
    eval(&ctx_with_bindings, &closure.body)
}

/// Evaluate built-in operators from the standard library
fn eval_builtin(
    ctx: &EvalCtx,
    name: &str,
    args: &[Spanned<Expr>],
    span: Option<Span>,
) -> EvalResult<Option<Value>> {
    match name {
        // === Built-in operator symbols (callable via Apply / operator replacement) ===
        "+" | "-" | "*" | "/" | "%" | "^" | "\\div" | "\\cup" | "\\cap" | "\\" => {
            check_arity(name, args, 2, span)?;
            let left = eval(ctx, &args[0])?;
            let right = eval(ctx, &args[1])?;
            Ok(Some(apply_builtin_binary_op(name, left, right, span)?))
        }

        // === Naturals/Integers/Reals ===
        "Nat" => Ok(Some(Value::model_value("Nat"))), // Infinite set marker
        "Int" => Ok(Some(Value::model_value("Int"))), // Infinite set marker
        "Real" => Ok(Some(Value::model_value("Real"))), // Infinite set marker (Int ⊆ Real)
        "Infinity" => {
            // TLC: Infinity is defined but errors on evaluation
            // We return a ModelValue that will error if used in arithmetic
            Ok(Some(Value::model_value("Infinity")))
        }
        "BOOLEAN" => Ok(Some(boolean_set())),

        // === Sequences ===
        "Len" => {
            check_arity(name, args, 1, span)?;
            let sv = eval(ctx, &args[0])?;
            match &sv {
                Value::Seq(_) | Value::Tuple(_) => {
                    let s = sv.as_seq_or_tuple_elements().unwrap();
                    Ok(Some(Value::int(s.len() as i64)))
                }
                Value::String(s) => Ok(Some(Value::int(s.len() as i64))),
                Value::IntFunc(f) => {
                    if f.min == 1 {
                        Ok(Some(Value::int(f.len() as i64)))
                    } else {
                        Err(EvalError::type_error("Seq", &sv, Some(args[0].span)))
                    }
                }
                Value::Func(f) => {
                    // In TLA+, sequences are functions with domain 1..n.
                    // Only accept functions whose domain is exactly {1, 2, ..., n} (or empty).
                    let mut expected: i64 = 1;
                    for key in f.domain_iter() {
                        let Some(k) = key.as_i64() else {
                            return Err(EvalError::type_error("Seq", &sv, Some(args[0].span)));
                        };
                        if k != expected {
                            return Err(EvalError::type_error("Seq", &sv, Some(args[0].span)));
                        }
                        expected += 1;
                    }
                    Ok(Some(Value::int(expected - 1)))
                }
                _ => Err(EvalError::type_error("Seq", &sv, Some(args[0].span))),
            }
        }

        "Head" => {
            check_arity(name, args, 1, span)?;
            let sv = eval(ctx, &args[0])?;
            let seq = sv
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv, Some(args[0].span)))?;
            Ok(Some(seq.first().cloned().ok_or(
                EvalError::IndexOutOfBounds {
                    index: 1,
                    len: 0,
                    span,
                },
            )?))
        }

        "Tail" => {
            check_arity(name, args, 1, span)?;
            let sv = eval(ctx, &args[0])?;
            // Fast path: use O(log n) tail for SeqValue
            if let Some(seq_value) = sv.as_seq_value() {
                if seq_value.is_empty() {
                    return Err(EvalError::IndexOutOfBounds {
                        index: 1,
                        len: 0,
                        span,
                    });
                }
                return Ok(Some(Value::Seq(seq_value.tail())));
            }
            // Fallback for Tuple
            let seq = sv
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv, Some(args[0].span)))?;
            if seq.is_empty() {
                return Err(EvalError::IndexOutOfBounds {
                    index: 1,
                    len: 0,
                    span,
                });
            }
            Ok(Some(Value::Seq(seq[1..].to_vec().into())))
        }

        "Append" => {
            check_arity(name, args, 2, span)?;
            let sv = eval(ctx, &args[0])?;
            let elem = eval(ctx, &args[1])?;
            // Fast path: use O(log n) append for SeqValue
            if let Some(seq_value) = sv.as_seq_value() {
                return Ok(Some(Value::Seq(seq_value.append(elem))));
            }
            // Fallback for Tuple
            let seq = sv
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv, Some(args[0].span)))?;
            let mut new_seq = Vec::with_capacity(seq.len() + 1);
            new_seq.extend(seq.iter().cloned());
            new_seq.push(elem);
            Ok(Some(Value::Seq(new_seq.into())))
        }

        "SubSeq" => {
            check_arity(name, args, 3, span)?;
            let sv = eval(ctx, &args[0])?;
            let mv = eval(ctx, &args[1])?;
            let nv = eval(ctx, &args[2])?;
            let m = mv
                .as_i64()
                .ok_or_else(|| EvalError::type_error("Int", &mv, Some(args[1].span)))?;
            let n = nv
                .as_i64()
                .ok_or_else(|| EvalError::type_error("Int", &nv, Some(args[2].span)))?;

            // TLA+ SubSeq(s, m, n) is 1-indexed
            if m < 1 || n < m - 1 {
                return Ok(Some(Value::Seq(Vec::new().into())));
            }
            let start = (m - 1) as usize;
            match &sv {
                Value::Seq(seq_value) => {
                    // Fast path: use O(log n) subseq for SeqValue
                    let end = (n as usize).min(seq_value.len());
                    if start >= seq_value.len() {
                        return Ok(Some(Value::Seq(Vec::new().into())));
                    }
                    Ok(Some(Value::Seq(seq_value.subseq(start, end))))
                }
                Value::Tuple(_) => {
                    let seq = sv.as_seq_or_tuple_elements().unwrap();
                    let end = (n as usize).min(seq.len());
                    if start >= seq.len() {
                        return Ok(Some(Value::Seq(Vec::new().into())));
                    }
                    Ok(Some(Value::Seq(seq[start..end].to_vec().into())))
                }
                Value::IntFunc(f) if f.min == 1 => {
                    let end = (n as usize).min(f.len());
                    if start >= f.len() {
                        return Ok(Some(Value::Seq(Vec::new().into())));
                    }
                    Ok(Some(Value::Seq(f.values[start..end].to_vec().into())))
                }
                Value::Func(f) => {
                    // In TLA+, sequences are functions with domain 1..n. Accept such functions here.
                    let mut expected: i64 = 1;
                    for key in f.domain_iter() {
                        if key.as_i64().unwrap_or(0) != expected {
                            return Err(EvalError::type_error("Seq", &sv, Some(args[0].span)));
                        }
                        expected += 1;
                    }
                    let len = (expected - 1) as usize;
                    let end = (n as usize).min(len);
                    if start >= len {
                        return Ok(Some(Value::Seq(Vec::new().into())));
                    }
                    let mut out = Vec::with_capacity(end.saturating_sub(start));
                    for i in (start + 1)..=end {
                        let key = Value::SmallInt(i as i64);
                        let Some(v) = f.apply(&key) else {
                            return Err(EvalError::Internal {
                                message: format!(
                                    "SubSeq: function domain includes {} but mapping has no value",
                                    i
                                ),
                                span,
                            });
                        };
                        out.push(v.clone());
                    }
                    Ok(Some(Value::Seq(out.into())))
                }
                _ => Err(EvalError::type_error("Seq", &sv, Some(args[0].span))),
            }
        }

        "Seq" => {
            check_arity(name, args, 1, span)?;
            // Seq(S) is the set of all finite sequences over S - infinite in general
            let base = eval(ctx, &args[0])?;
            Ok(Some(Value::SeqSet(SeqSetValue::new(base))))
        }

        // Sequence concatenation
        "\\o" | "\\circ" => {
            check_arity(name, args, 2, span)?;
            let sv1 = eval(ctx, &args[0])?;
            let sv2 = eval(ctx, &args[1])?;
            fn func_seq_len(f: &FuncValue) -> Option<usize> {
                let mut expected: i64 = 1;
                for key in f.domain_iter() {
                    if key.as_i64()? != expected {
                        return None;
                    }
                    expected += 1;
                }
                Some((expected - 1) as usize)
            }

            fn seq_like_len(v: &Value) -> Option<usize> {
                if let Some(elems) = v.as_seq_or_tuple_elements() {
                    return Some(elems.len());
                }
                match v {
                    Value::IntFunc(f) if f.min == 1 => Some(f.len()),
                    Value::Func(f) => func_seq_len(f),
                    _ => None,
                }
            }

            let len1 = seq_like_len(&sv1)
                .ok_or_else(|| EvalError::type_error("Seq", &sv1, Some(args[0].span)))?;
            let len2 = seq_like_len(&sv2)
                .ok_or_else(|| EvalError::type_error("Seq", &sv2, Some(args[1].span)))?;

            let mut result: Vec<Value> = Vec::with_capacity(len1 + len2);

            match &sv1 {
                Value::Seq(_) | Value::Tuple(_) => {
                    let s = sv1.as_seq_or_tuple_elements().unwrap();
                    result.extend(s.iter().cloned());
                }
                Value::IntFunc(f) if f.min == 1 => result.extend(f.values.iter().cloned()),
                Value::Func(f) => {
                    let mut expected: i64 = 1;
                    for (k, v) in f.mapping_iter() {
                        let Some(idx) = k.as_i64() else {
                            return Err(EvalError::type_error("Seq", &sv1, Some(args[0].span)));
                        };
                        if idx != expected {
                            return Err(EvalError::type_error("Seq", &sv1, Some(args[0].span)));
                        }
                        expected += 1;
                        result.push(v.clone());
                    }
                }
                _ => return Err(EvalError::type_error("Seq", &sv1, Some(args[0].span))),
            }

            match &sv2 {
                Value::Seq(_) | Value::Tuple(_) => {
                    let s = sv2.as_seq_or_tuple_elements().unwrap();
                    result.extend(s.iter().cloned());
                }
                Value::IntFunc(f) if f.min == 1 => result.extend(f.values.iter().cloned()),
                Value::Func(f) => {
                    let mut expected: i64 = 1;
                    for (k, v) in f.mapping_iter() {
                        let Some(idx) = k.as_i64() else {
                            return Err(EvalError::type_error("Seq", &sv2, Some(args[1].span)));
                        };
                        if idx != expected {
                            return Err(EvalError::type_error("Seq", &sv2, Some(args[1].span)));
                        }
                        expected += 1;
                        result.push(v.clone());
                    }
                }
                _ => return Err(EvalError::type_error("Seq", &sv2, Some(args[1].span))),
            }
            Ok(Some(Value::Seq(result.into())))
        }

        "SelectSeq" => {
            // SelectSeq(s, Test) - filter sequence by a test operator
            // Test can be either:
            // - An operator name (Ident)
            // - An inline lambda expression (Lambda)
            check_arity(name, args, 2, span)?;
            let sv = eval(ctx, &args[0])?;
            let seq = sv
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv, Some(args[0].span)))?;

            // Handle both named operator and inline lambda
            let mut result = Vec::new();
            match &args[1].node {
                Expr::Ident(test_name) => {
                    // Named operator: SelectSeq(s, OpName)
                    let test_def = ctx
                        .get_op(test_name)
                        .ok_or_else(|| EvalError::UndefinedOp {
                            name: test_name.clone(),
                            span,
                        })?;

                    if test_def.params.len() != 1 {
                        return Err(EvalError::ArityMismatch {
                            op: test_name.clone(),
                            expected: 1,
                            got: test_def.params.len(),
                            span,
                        });
                    }

                    for elem in seq.iter() {
                        let test_ctx = ctx.bind(test_def.params[0].name.node.clone(), elem.clone());
                        let test_result = eval(&test_ctx, &test_def.body)?;
                        let passed = test_result
                            .as_bool()
                            .ok_or_else(|| EvalError::type_error("BOOLEAN", &test_result, span))?;
                        if passed {
                            result.push(elem.clone());
                        }
                    }
                }
                Expr::Lambda(params, body) => {
                    // Inline lambda: SelectSeq(s, LAMBDA x: P(x))
                    if params.len() != 1 {
                        return Err(EvalError::ArityMismatch {
                            op: "SelectSeq lambda".to_string(),
                            expected: 1,
                            got: params.len(),
                            span,
                        });
                    }

                    let param_name = &params[0].node;
                    for elem in seq.iter() {
                        let test_ctx = ctx.bind(param_name.clone(), elem.clone());
                        let test_result = eval(&test_ctx, body)?;
                        let passed = test_result
                            .as_bool()
                            .ok_or_else(|| EvalError::type_error("BOOLEAN", &test_result, span))?;
                        if passed {
                            result.push(elem.clone());
                        }
                    }
                }
                _ => {
                    return Err(EvalError::Internal {
                        message: "SelectSeq requires an operator name or lambda as second argument"
                            .into(),
                        span,
                    })
                }
            }
            Ok(Some(Value::Seq(result.into())))
        }

        // === FiniteSets ===
        "Cardinality" => {
            check_arity(name, args, 1, span)?;

            // Optimization: For set filter expressions {x \in S : P(x)}, count directly
            // without building the intermediate set. This is especially important for
            // patterns like Cardinality({m \in rcvd'[self] : m[2] = "ECHO0"}) in bosco.
            if let Expr::SetFilter(bound, pred) = &args[0].node {
                if let Some(domain_expr) = &bound.domain {
                    let domain_val = eval(ctx, domain_expr)?;
                    let count_result =
                        count_set_filter_elements(ctx, &domain_val, bound, pred, span);
                    if let Some(count) = count_result? {
                        return Ok(Some(Value::int(count as i64)));
                    }
                }
            }

            // Fall back to standard evaluation for non-filter sets
            let sv = eval(ctx, &args[0])?;
            match sv.set_len() {
                Some(n) => Ok(Some(Value::big_int(n))),
                None if sv.is_set() => Err(EvalError::Internal {
                    message: "Cardinality not supported for this set value".into(),
                    span,
                }),
                None => Err(EvalError::type_error("Set", &sv, Some(args[0].span))),
            }
        }

        "IsFiniteSet" => {
            check_arity(name, args, 1, span)?;
            let sv = eval(ctx, &args[0])?;
            // All sets we can represent are finite
            Ok(Some(Value::Bool(sv.is_set())))
        }

        // === TLC ===
        "Print" => {
            // Print(val, result) prints val and returns result
            check_arity(name, args, 2, span)?;
            let val = eval(ctx, &args[0])?;
            let result = eval(ctx, &args[1])?;
            eprintln!("TLC Print: {}", val);
            Ok(Some(result))
        }

        "PrintT" => {
            // PrintT(val) prints val and returns TRUE
            check_arity(name, args, 1, span)?;
            let val = eval(ctx, &args[0])?;
            eprintln!("TLC Print: {}", val);
            Ok(Some(Value::Bool(true)))
        }

        "Assert" => {
            check_arity(name, args, 2, span)?;
            let val = eval(ctx, &args[0])?;
            let cond = val
                .as_bool()
                .ok_or_else(|| EvalError::type_error("BOOLEAN", &val, Some(args[0].span)))?;
            if !cond {
                let msg = eval(ctx, &args[1])?;
                return Err(EvalError::Internal {
                    message: format!("Assertion failed: {}", msg),
                    span,
                });
            }
            Ok(Some(Value::Bool(true)))
        }

        "ToString" => {
            check_arity(name, args, 1, span)?;
            let val = eval(ctx, &args[0])?;
            // Use interning to enable pointer-based equality for repeated strings
            Ok(Some(Value::String(intern_string(&format!("{}", val)))))
        }

        // d :> e - create single-element function [d |-> e]
        ":>" => {
            check_arity(name, args, 2, span)?;
            let domain_elem = eval(ctx, &args[0])?;
            let range_elem = eval(ctx, &args[1])?;
            // Create single-element function directly, avoiding im::OrdSet/OrdMap overhead
            Ok(Some(Value::Func(FuncValue::from_sorted_entries(vec![(
                domain_elem,
                range_elem,
            )]))))
        }

        // f @@ g - merge two functions (f takes priority for overlapping domains)
        "@@" => {
            check_arity(name, args, 2, span)?;
            let fv = eval(ctx, &args[0])?;
            let gv = eval(ctx, &args[1])?;
            match (fv, gv) {
                (Value::Record(f), Value::Record(g)) => {
                    // Preserve record values for record field access syntax (r.field).
                    // f @@ g means f overrides g for overlapping keys (TLC semantics)
                    let mut merged = g;
                    for (k, v) in f.iter() {
                        merged = merged.update(Arc::clone(k), v.clone());
                    }
                    Ok(Some(Value::Record(merged)))
                }
                (fv, gv) => {
                    let f = fv.to_func_coerced().ok_or_else(|| {
                        EvalError::type_error("Function", &fv, Some(args[0].span))
                    })?;
                    let g = gv.to_func_coerced().ok_or_else(|| {
                        EvalError::type_error("Function", &gv, Some(args[1].span))
                    })?;

                    // Merge: union of domains, f takes priority for overlapping keys
                    // Build combined entries: all of f, plus any from g not in f's domain
                    let mut entries: Vec<(Value, Value)> = f.entries().to_vec();

                    // Add mappings from g that are not in f's domain
                    for (k, v) in g.mapping_iter() {
                        if !f.domain_contains(k) {
                            entries.push((k.clone(), v.clone()));
                        }
                    }

                    // Sort entries by key to maintain FuncValue invariant
                    entries.sort_by(|a, b| a.0.cmp(&b.0));

                    Ok(Some(Value::Func(FuncValue::from_sorted_entries(entries))))
                }
            }
        }

        "SortSeq" => {
            // SortSeq(s, Op) - sort sequence using a comparator operator
            // Op(a, b) should return TRUE if a should come before b
            check_arity(name, args, 2, span)?;
            let sv = eval(ctx, &args[0])?;
            let seq = sv
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv, Some(args[0].span)))?;

            // Get the comparator operator name from the second argument
            let cmp_name = match &args[1].node {
                Expr::Ident(name) => name.clone(),
                _ => {
                    return Err(EvalError::Internal {
                        message: "SortSeq requires an operator name as second argument".into(),
                        span,
                    })
                }
            };

            // Get the operator definition
            let cmp_def = ctx
                .get_op(&cmp_name)
                .ok_or_else(|| EvalError::UndefinedOp {
                    name: cmp_name.clone(),
                    span,
                })?;

            if cmp_def.params.len() != 2 {
                return Err(EvalError::ArityMismatch {
                    op: cmp_name,
                    expected: 2,
                    got: cmp_def.params.len(),
                    span,
                });
            }

            // Clone the sequence and sort it
            let mut result: Vec<Value> = seq.to_vec();

            // Use a simple insertion sort to avoid closure issues
            // (Rust's sort_by requires FnMut but we need to evaluate in ctx)
            for i in 1..result.len() {
                let mut j = i;
                while j > 0 {
                    // Compare result[j-1] with result[j]
                    let cmp_ctx = ctx
                        .bind(cmp_def.params[0].name.node.clone(), result[j - 1].clone())
                        .bind(cmp_def.params[1].name.node.clone(), result[j].clone());
                    let cmp_result = eval(&cmp_ctx, &cmp_def.body)?;
                    let a_before_b = cmp_result
                        .as_bool()
                        .ok_or_else(|| EvalError::type_error("BOOLEAN", &cmp_result, span))?;
                    if a_before_b {
                        break; // result[j-1] should come before result[j], no swap needed
                    }
                    result.swap(j - 1, j);
                    j -= 1;
                }
            }

            Ok(Some(Value::Seq(result.into())))
        }

        "Permutations" => {
            // Permutations(S) - set of all permutation functions on set S
            // TLC semantics: returns a set of bijections [S -> S]
            // For {a, b}: returns {[a |-> a, b |-> b], [a |-> b, b |-> a]}
            check_arity(name, args, 1, span)?;
            let sv = eval(ctx, &args[0])?;
            let set = sv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv, Some(args[0].span)))?;

            let elements: Vec<Value> = set.iter().cloned().collect();
            let n = elements.len();

            // Guard against combinatorial explosion (n! grows fast)
            if n > 10 {
                return Err(EvalError::Internal {
                    message: format!(
                        "Permutations of {} elements would be too large ({}! permutations)",
                        n, n
                    ),
                    span,
                });
            }

            // Generate all permutation functions
            let mut perms = OrdSet::new();
            generate_permutation_functions(&elements, &[], &mut perms);
            Ok(Some(Value::Set(SortedSet::from_ord_set(&perms))))
        }

        // JavaTime - returns current wall clock time in seconds since epoch
        "JavaTime" => {
            check_arity(name, args, 0, span)?;
            use std::time::{SystemTime, UNIX_EPOCH};
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
            // Zero MSB to prevent negative values (same as TLC)
            Ok(Some(Value::int(now & 0x7FFFFFFF)))
        }

        // TLCGet(x) - get TLC register or config
        // TLCGet("config") returns configuration record with mode, depth, deadlock, etc.
        // TLCGet(i) for integer i returns TLC register (stub - returns 0)
        "TLCGet" => {
            check_arity(name, args, 1, span)?;
            let idx = eval(ctx, &args[0])?;

            // Handle string arguments like "config", "level", "stats"
            if idx.as_string() == Some("config") {
                // Return config record with mode, depth, deadlock
                let config = &ctx.shared.tlc_config;
                return Ok(Some(Value::record([
                    ("mode", Value::String(config.mode.clone())),
                    ("depth", Value::int(config.depth)),
                    ("deadlock", Value::Bool(config.deadlock)),
                ])));
            }

            // TLC registers are for debugging; return 0 for model checking
            Ok(Some(Value::int(0)))
        }

        // TLCSet(i, v) - set TLC register (stub - returns TRUE)
        "TLCSet" => {
            check_arity(name, args, 2, span)?;
            let _idx = eval(ctx, &args[0])?;
            let _val = eval(ctx, &args[1])?;
            // TLC registers are for debugging; noop for model checking
            Ok(Some(Value::Bool(true)))
        }

        // RandomElement(S) - return a random element from set S
        "RandomElement" => {
            check_arity(name, args, 1, span)?;
            let sv = eval(ctx, &args[0])?;
            let mut iter = sv
                .iter_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv, Some(args[0].span)))?;
            let Some(elem) = iter.next() else {
                return Err(EvalError::Internal {
                    message: "RandomElement requires non-empty set".into(),
                    span,
                });
            };
            // For model checking, return the first element (deterministic).
            // In simulation mode, this would return a random element.
            Ok(Some(elem))
        }

        // TLCEval(v) - force evaluation of lazy value (noop - TLA2 is already eager)
        "TLCEval" => {
            check_arity(name, args, 1, span)?;
            Ok(Some(eval(ctx, &args[0])?))
        }

        // Any / ANY - the set of all values (infinite, non-enumerable)
        // TLC provides this via TLC!Any and the AnySet module.
        "Any" | "ANY" => {
            check_arity(name, args, 0, span)?;
            Ok(Some(Value::AnySet))
        }

        // === TLCExt operators ===

        // AssertError(msg, expr) - like Assert but with custom error message
        "AssertError" => {
            check_arity(name, args, 2, span)?;
            let msg_val = eval(ctx, &args[0])?;
            let msg = msg_val
                .as_string()
                .ok_or_else(|| EvalError::type_error("String", &msg_val, Some(args[0].span)))?;
            let cond = eval(ctx, &args[1])?;
            let is_true = cond.as_bool().unwrap_or(false);
            if !is_true {
                eprintln!("AssertError: {}", msg);
            }
            Ok(Some(Value::Bool(is_true)))
        }

        // AssertEq(a, b) - like = but prints values if not equal
        "AssertEq" => {
            check_arity(name, args, 2, span)?;
            let a = eval(ctx, &args[0])?;
            let b = eval(ctx, &args[1])?;
            if a != b {
                eprintln!("AssertEq failed:");
                eprintln!("  Left:  {}", a);
                eprintln!("  Right: {}", b);
            }
            Ok(Some(Value::Bool(a == b)))
        }

        // TLCDefer(expr) - defer evaluation (stub: just evaluate)
        "TLCDefer" => {
            check_arity(name, args, 1, span)?;
            // In TLC, this defers evaluation to when the successor state is chosen.
            // For model checking, we just evaluate immediately.
            Ok(Some(eval(ctx, &args[0])?))
        }

        // PickSuccessor(expr) - interactive successor selection (stub: TRUE)
        "PickSuccessor" => {
            check_arity(name, args, 1, span)?;
            // Evaluate condition for any side effects, but TLA2 doesn't do interactive
            // selection, so always return TRUE (non-interactive mode behavior)
            let _cond = eval(ctx, &args[0])?;
            Ok(Some(Value::Bool(true)))
        }

        // TLCNoOp(val) - returns val unchanged (debugging hook)
        "TLCNoOp" => {
            check_arity(name, args, 1, span)?;
            Ok(Some(eval(ctx, &args[0])?))
        }

        // TLCModelValue(str) - create a model value from string
        "TLCModelValue" => {
            check_arity(name, args, 1, span)?;
            let sv = eval(ctx, &args[0])?;
            let s = sv
                .as_string()
                .ok_or_else(|| EvalError::type_error("String", &sv, Some(args[0].span)))?;
            // Use interning to enable pointer-based equality
            Ok(Some(Value::model_value(s)))
        }

        // TLCCache(expr, closure) - caching (stub: just evaluate expr)
        "TLCCache" => {
            check_arity(name, args, 2, span)?;
            // In TLC, this caches based on closure.
            // For model checking, we just evaluate the expression.
            Ok(Some(eval(ctx, &args[0])?))
        }

        // TLCGetOrDefault(key, defaultVal) - like TLCGet but returns default if not set
        "TLCGetOrDefault" => {
            check_arity(name, args, 2, span)?;
            let _key = eval(ctx, &args[0])?;
            let default_val = eval(ctx, &args[1])?;
            // TLA2 stub: TLC registers are not persisted, so always return default
            // In real TLC, this checks if key has been set via TLCSet and returns that
            // or the default value otherwise.
            Ok(Some(default_val))
        }

        // TLCGetAndSet(key, Op, val, defaultVal) - atomic get-and-set
        "TLCGetAndSet" => {
            check_arity(name, args, 4, span)?;
            let _key = eval(ctx, &args[0])?;
            // Op is a binary operator name
            let _op = &args[1];
            let _val = eval(ctx, &args[2])?;
            let default_val = eval(ctx, &args[3])?;
            // TLA2 stub: Since we don't persist TLC registers, this just returns the default.
            // Semantics: oldVal = TLCGetOrDefault(key, defaultVal), then TLCSet(key, Op(oldVal, val))
            // Returns oldVal (the value before the set)
            Ok(Some(default_val))
        }

        // TLCFP(val) - returns the fingerprint of a value as an integer
        "TLCFP" => {
            check_arity(name, args, 1, span)?;
            let val = eval(ctx, &args[0])?;
            // Compute fingerprint and return lower 32 bits as integer
            // TLC returns lower 32 bits because TLA+ integers can't represent 64-bit values easily
            let fp = value_fingerprint(&val);
            // Return as signed 32-bit integer (TLC behavior)
            let fp32 = (fp & 0xFFFFFFFF) as i32;
            Ok(Some(Value::int(fp32 as i64)))
        }

        // TLCEvalDefinition(defName) - evaluate a definition by name
        // Useful for dynamically accessing definitions
        "TLCEvalDefinition" => {
            check_arity(name, args, 1, span)?;
            let name_val = eval(ctx, &args[0])?;
            let def_name = name_val
                .as_string()
                .ok_or_else(|| EvalError::type_error("String", &name_val, Some(args[0].span)))?;

            // Look up the definition
            let op_def = ctx.get_op(def_name).ok_or_else(|| EvalError::UndefinedOp {
                name: def_name.to_string(),
                span,
            })?;

            // Must be zero-arity
            if !op_def.params.is_empty() {
                return Err(EvalError::ArityMismatch {
                    op: def_name.to_string(),
                    expected: 0,
                    got: op_def.params.len(),
                    span,
                });
            }

            // Evaluate the body
            Ok(Some(eval(ctx, &op_def.body)?))
        }

        // Trace - returns the current trace as a sequence of records
        // TLA2 stub: Returns empty sequence since trace not available during evaluation
        "Trace" => {
            check_arity(name, args, 0, span)?;
            // In TLC, this returns the full trace from initial state to current state.
            // TLA2 doesn't have access to trace during evaluation - would need architectural
            // changes to pass trace through EvalCtx. Return empty sequence as stub.
            Ok(Some(Value::Tuple(Vec::new().into())))
        }

        // CounterExample - returns counterexample graph in POSTCONDITION scope
        // TLA2 stub: Returns empty record since not in postcondition context
        "CounterExample" => {
            check_arity(name, args, 0, span)?;
            // In TLC, this returns [state: States, action: Actions] counterexample graph
            // when evaluated in POSTCONDITION scope. TLA2 doesn't support postconditions yet.
            // Return empty structure as stub.
            Ok(Some(Value::Record(RecordValue::new())))
        }

        // ToTrace(ce) - convert CounterExample to trace sequence
        // TLA2 stub: Returns empty sequence since CounterExample is empty
        "ToTrace" => {
            check_arity(name, args, 1, span)?;
            let _ce = eval(ctx, &args[0])?;
            // In TLC, this converts a CounterExample to a sequence of states.
            // Since our CounterExample is a stub, return empty sequence.
            Ok(Some(Value::Tuple(Vec::new().into())))
        }

        // === FiniteSetsExt operators ===

        // Quantify(S, P) - count elements of S satisfying predicate P
        "Quantify" => {
            check_arity(name, args, 2, span)?;
            let sv = eval(ctx, &args[0])?;
            let set = sv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv, Some(args[0].span)))?;

            // Get the predicate operator name from the second argument
            let pred_name = match &args[1].node {
                Expr::Ident(name) => name.clone(),
                _ => {
                    return Err(EvalError::Internal {
                        message: "Quantify requires an operator name as second argument".into(),
                        span,
                    })
                }
            };

            // Get the operator definition
            let pred_def = ctx
                .get_op(&pred_name)
                .ok_or_else(|| EvalError::UndefinedOp {
                    name: pred_name.clone(),
                    span,
                })?;

            if pred_def.params.len() != 1 {
                return Err(EvalError::ArityMismatch {
                    op: pred_name,
                    expected: 1,
                    got: pred_def.params.len(),
                    span,
                });
            }

            let mut count = 0i64;
            for elem in set.iter() {
                let new_ctx = ctx.bind(pred_def.params[0].name.node.as_str(), elem.clone());
                let result = eval(&new_ctx, &pred_def.body)?;
                if result.as_bool().unwrap_or(false) {
                    count += 1;
                }
            }
            Ok(Some(Value::int(count)))
        }

        // Ksubsets(S, k) - all k-element subsets of S
        // Returns a lazy KSubsetValue for efficient membership checking
        "Ksubsets" => {
            check_arity(name, args, 2, span)?;
            let sv = eval(ctx, &args[0])?;
            // Accept any set-like value (Set, Interval, etc.)
            if !sv.is_set() {
                return Err(EvalError::type_error("Set", &sv, Some(args[0].span)));
            }
            let kv = eval(ctx, &args[1])?;
            let k = kv
                .as_int()
                .ok_or_else(|| EvalError::type_error("Int", &kv, Some(args[1].span)))?
                .to_i64()
                .unwrap_or(0) as usize;

            // Return lazy KSubsetValue - enumeration happens on-demand
            Ok(Some(Value::KSubset(KSubsetValue::new(sv, k))))
        }

        // SymDiff(S, T) - symmetric difference: (S \ T) \cup (T \ S)
        "SymDiff" => {
            check_arity(name, args, 2, span)?;
            let sv1 = eval(ctx, &args[0])?;
            let set1 = sv1
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv1, Some(args[0].span)))?;
            let sv2 = eval(ctx, &args[1])?;
            let set2 = sv2
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv2, Some(args[1].span)))?;

            let mut result = Vec::new();
            // Elements in set1 but not in set2
            for elem in set1.iter() {
                if !set2.contains(elem) {
                    result.push(elem.clone());
                }
            }
            // Elements in set2 but not in set1
            for elem in set2.iter() {
                if !set1.contains(elem) {
                    result.push(elem.clone());
                }
            }
            Ok(Some(Value::set(result)))
        }

        // Flatten(SS) - union of a set of sets
        "Flatten" => {
            check_arity(name, args, 1, span)?;
            let sv = eval(ctx, &args[0])?;
            let outer_set = sv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv, Some(args[0].span)))?;

            let mut result = Vec::new();
            for inner in outer_set.iter() {
                let inner_set = inner
                    .as_set()
                    .ok_or_else(|| EvalError::type_error("Set", inner, Some(args[0].span)))?;
                for elem in inner_set.iter() {
                    if !result.contains(elem) {
                        result.push(elem.clone());
                    }
                }
            }
            Ok(Some(Value::set(result)))
        }

        // Choose(S) - return an arbitrary element of S (first in sorted order)
        "Choose" => {
            check_arity(name, args, 1, span)?;
            let sv = eval(ctx, &args[0])?;
            let set = sv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv, Some(args[0].span)))?;

            set.iter()
                .next()
                .cloned()
                .ok_or_else(|| EvalError::Internal {
                    message: "Choose requires non-empty set".into(),
                    span,
                })
                .map(Some)
        }

        // Sum(S) - sum of all elements in a set of integers
        "Sum" => {
            check_arity(name, args, 1, span)?;
            let sv = eval(ctx, &args[0])?;
            let set = sv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv, Some(args[0].span)))?;

            let mut total = BigInt::zero();
            for elem in set.iter() {
                if let Some(n) = elem.as_int() {
                    total += n;
                } else {
                    return Err(EvalError::type_error("Int", elem, Some(args[0].span)));
                }
            }
            Ok(Some(Value::big_int(total)))
        }

        // Product(S) - product of all elements in a set of integers
        "Product" => {
            check_arity(name, args, 1, span)?;
            let sv = eval(ctx, &args[0])?;
            let set = sv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv, Some(args[0].span)))?;

            let mut total = BigInt::one();
            for elem in set.iter() {
                if let Some(n) = elem.as_int() {
                    total *= n;
                } else {
                    return Err(EvalError::type_error("Int", elem, Some(args[0].span)));
                }
            }
            Ok(Some(Value::big_int(total)))
        }

        // ReduceSet(op, S, base) - like FoldSet but different argument order
        "ReduceSet" => {
            check_arity(name, args, 3, span)?;

            // Get the operator name from the first argument
            let op_name = match &args[0].node {
                Expr::Ident(name) => name.clone(),
                _ => {
                    return Err(EvalError::Internal {
                        message: "ReduceSet requires operator name as first argument".into(),
                        span,
                    })
                }
            };

            let sv = eval(ctx, &args[1])?;
            let set = sv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv, Some(args[1].span)))?;
            let base = eval(ctx, &args[2])?;

            // Get the operator definition
            let op_def = ctx.get_op(&op_name).ok_or_else(|| EvalError::UndefinedOp {
                name: op_name.clone(),
                span,
            })?;

            if op_def.params.len() != 2 {
                return Err(EvalError::ArityMismatch {
                    op: op_name,
                    expected: 2,
                    got: op_def.params.len(),
                    span,
                });
            }

            // Fold over the set elements
            let mut result = base;
            for elem in set.iter() {
                let new_ctx = ctx
                    .bind(op_def.params[0].name.node.as_str(), result)
                    .bind(op_def.params[1].name.node.as_str(), elem.clone());
                result = eval(&new_ctx, &op_def.body)?;
            }

            Ok(Some(result))
        }

        // Mean(S) - average of a set of integers (integer division)
        "Mean" => {
            check_arity(name, args, 1, span)?;
            let sv = eval(ctx, &args[0])?;
            let set = sv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv, Some(args[0].span)))?;

            if set.is_empty() {
                return Err(EvalError::Internal {
                    message: "Mean requires non-empty set".into(),
                    span,
                });
            }

            let mut total = BigInt::zero();
            let mut count = BigInt::zero();
            for elem in set.iter() {
                if let Some(n) = elem.as_int() {
                    total += n;
                    count += 1;
                } else {
                    return Err(EvalError::type_error("Int", elem, Some(args[0].span)));
                }
            }
            // Integer division (floor)
            use num_integer::Integer;
            Ok(Some(Value::big_int(total.div_floor(&count))))
        }

        // MapThenSumSet(Op, S) - map a unary operator over a set, then sum the results
        // MapThenSumSet(Op, S) == LET R == {Op(x) : x \in S} IN Sum(R)
        "MapThenSumSet" => {
            check_arity(name, args, 2, span)?;

            // Get the operator name from the first argument
            let op_name = match &args[0].node {
                Expr::Ident(name) => name.clone(),
                _ => {
                    return Err(EvalError::Internal {
                        message: "MapThenSumSet requires operator name as first argument".into(),
                        span,
                    })
                }
            };

            let sv = eval(ctx, &args[1])?;
            let set = sv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv, Some(args[1].span)))?;

            // Get the operator definition
            let op_def = ctx.get_op(&op_name).ok_or_else(|| EvalError::UndefinedOp {
                name: op_name.clone(),
                span,
            })?;

            if op_def.params.len() != 1 {
                return Err(EvalError::ArityMismatch {
                    op: op_name,
                    expected: 1,
                    got: op_def.params.len(),
                    span,
                });
            }

            // Map and sum in one pass
            let mut total = BigInt::zero();
            for elem in set.iter() {
                let new_ctx = ctx.bind(op_def.params[0].name.node.as_str(), elem.clone());
                let mapped = eval(&new_ctx, &op_def.body)?;
                // Use to_bigint() to handle both SmallInt and Int variants
                if let Some(n) = mapped.to_bigint() {
                    total += n;
                } else {
                    return Err(EvalError::type_error("Int", &mapped, Some(args[0].span)));
                }
            }
            Ok(Some(Value::big_int(total)))
        }

        // Choices(SS) - the set of all choice functions for a set of sets
        // Choices(SS) == { f \in [SS -> UNION SS] : \A S \in SS : f[S] \in S }
        // For each set S in SS, f picks one element from S.
        "Choices" => {
            check_arity(name, args, 1, span)?;
            let ssv = eval(ctx, &args[0])?;
            let set_of_sets = ssv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &ssv, Some(args[0].span)))?;

            // Collect the sets as vectors of elements
            let mut sets_vec: Vec<(Value, Vec<Value>)> = Vec::new();
            for s in set_of_sets.iter() {
                let inner = s
                    .as_set()
                    .ok_or_else(|| EvalError::type_error("Set", s, Some(args[0].span)))?;
                if inner.is_empty() {
                    // If any set is empty, there are no choice functions
                    return Ok(Some(Value::set(vec![])));
                }
                sets_vec.push((s.clone(), inner.iter().cloned().collect()));
            }

            if sets_vec.is_empty() {
                // Empty set of sets -> one choice function: the empty function
                return Ok(Some(Value::set(vec![Value::Func(
                    FuncValue::from_sorted_entries(vec![]),
                )])));
            }

            // Sort by key so we can build sorted entries directly
            sets_vec.sort_by(|(a, _), (b, _)| a.cmp(b));

            // Generate all combinations using cartesian product
            let mut result_functions: Vec<Value> = Vec::new();
            let mut indices: Vec<usize> = vec![0; sets_vec.len()];

            loop {
                // Build a function from the current indices - entries are already sorted
                let entries: Vec<(Value, Value)> = sets_vec
                    .iter()
                    .enumerate()
                    .map(|(i, (set_key, elements))| (set_key.clone(), elements[indices[i]].clone()))
                    .collect();
                result_functions.push(Value::Func(FuncValue::from_sorted_entries(entries)));

                // Advance indices (like incrementing a multi-digit counter)
                let mut pos = sets_vec.len();
                loop {
                    if pos == 0 {
                        // Done - all combinations exhausted
                        return Ok(Some(Value::set(result_functions)));
                    }
                    pos -= 1;
                    indices[pos] += 1;
                    if indices[pos] < sets_vec[pos].1.len() {
                        break;
                    }
                    indices[pos] = 0;
                }
            }
        }

        // ChooseUnique(S, P) - the unique element of S satisfying predicate P
        // Requires exactly one element satisfies P, otherwise error
        "ChooseUnique" => {
            check_arity(name, args, 2, span)?;
            let sv = eval(ctx, &args[0])?;
            let set = sv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv, Some(args[0].span)))?;

            // Get the predicate (should be a lambda with one parameter)
            let (param, pred_body) = match &args[1].node {
                Expr::Lambda(params, body) if params.len() == 1 => {
                    (params[0].node.clone(), body.as_ref())
                }
                _ => {
                    return Err(EvalError::Internal {
                        message: "ChooseUnique requires a lambda predicate as second argument"
                            .into(),
                        span,
                    })
                }
            };

            let mut found: Option<Value> = None;
            for elem in set.iter() {
                let pred_ctx = ctx.bind(param.as_str(), elem.clone());
                let pred_result = eval(&pred_ctx, pred_body)?;
                if pred_result.as_bool() == Some(true) {
                    if found.is_some() {
                        return Err(EvalError::Internal {
                            message: "ChooseUnique: more than one element satisfies predicate"
                                .into(),
                            span,
                        });
                    }
                    found = Some(elem.clone());
                }
            }

            found
                .ok_or_else(|| EvalError::Internal {
                    message: "ChooseUnique: no element satisfies predicate".into(),
                    span,
                })
                .map(Some)
        }

        // === Other common stdlib ===
        "Min" => {
            check_arity(name, args, 1, span)?;
            let sv = eval(ctx, &args[0])?;
            let set = sv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv, Some(args[0].span)))?;
            set.iter()
                .filter_map(|v| v.as_int())
                .min()
                .map(|n| Value::big_int(n.clone()))
                .ok_or_else(|| EvalError::Internal {
                    message: "Min requires non-empty set of integers".into(),
                    span,
                })
                .map(Some)
        }

        "Max" => {
            check_arity(name, args, 1, span)?;
            let sv = eval(ctx, &args[0])?;
            let set = sv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv, Some(args[0].span)))?;
            set.iter()
                .filter_map(|v| v.as_int())
                .max()
                .map(|n| Value::big_int(n.clone()))
                .ok_or_else(|| EvalError::Internal {
                    message: "Max requires non-empty set of integers".into(),
                    span,
                })
                .map(Some)
        }

        // === Strings module ===

        // STRING - the set of all strings (infinite)
        "STRING" => {
            check_arity(name, args, 0, span)?;
            Ok(Some(Value::StringSet))
        }

        "Abs" => {
            // Abs(n) - absolute value of an integer
            check_arity(name, args, 1, span)?;
            let nv = eval(ctx, &args[0])?;
            // SmallInt fast path
            if let Value::SmallInt(n) = nv {
                return Ok(Some(Value::SmallInt(n.abs())));
            }
            let n = nv
                .to_bigint()
                .ok_or_else(|| EvalError::type_error("Int", &nv, Some(args[0].span)))?;
            use num_traits::Signed;
            Ok(Some(Value::big_int(n.abs())))
        }

        "Sign" => {
            // Sign(n) - returns -1, 0, or 1 based on sign of n
            check_arity(name, args, 1, span)?;
            let nv = eval(ctx, &args[0])?;
            // SmallInt fast path
            if let Value::SmallInt(n) = nv {
                return Ok(Some(Value::SmallInt(n.signum())));
            }
            let n = nv
                .to_bigint()
                .ok_or_else(|| EvalError::type_error("Int", &nv, Some(args[0].span)))?;
            use num_traits::Signed;
            let signum = n.signum();
            Ok(Some(Value::big_int(signum)))
        }

        "Range" => {
            // Range(f) - the set of all values in the function's mapping (co-domain image)
            check_arity(name, args, 1, span)?;
            let fv = eval(ctx, &args[0])?;
            let values: OrdSet<Value> = match &fv {
                Value::Func(func) => func.mapping_values().cloned().collect(),
                Value::IntFunc(func) => func.values.iter().cloned().collect(),
                Value::Seq(_) | Value::Tuple(_) => fv
                    .as_seq_or_tuple_elements()
                    .unwrap()
                    .iter()
                    .cloned()
                    .collect(),
                _ => {
                    return Err(EvalError::type_error(
                        "Function/Seq",
                        &fv,
                        Some(args[0].span),
                    ))
                }
            };
            Ok(Some(Value::Set(SortedSet::from_ord_set(&values))))
        }

        // === Functions module ===
        "Restrict" => {
            // Restrict(f, S) - restrict domain of function f to elements in set S
            // Restrict(f, S) == [x \in S |-> f[x]]
            check_arity(name, args, 2, span)?;
            let fv = eval(ctx, &args[0])?;
            let func = fv
                .to_func_coerced()
                .ok_or_else(|| EvalError::type_error("Function", &fv, Some(args[0].span)))?;
            let sv = eval(ctx, &args[1])?;
            let set = sv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv, Some(args[1].span)))?;

            // Filter entries to those whose keys are in the set
            let entries: Vec<(Value, Value)> = func
                .mapping_iter()
                .filter(|(k, _)| set.contains(k))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();

            Ok(Some(Value::Func(FuncValue::from_sorted_entries(entries))))
        }

        "IsInjective" => {
            // IsInjective(f) - TRUE if f is injective (one-to-one)
            // IsInjective(f) == \A a, b \in DOMAIN f: f[a] = f[b] => a = b
            check_arity(name, args, 1, span)?;
            let fv = eval(ctx, &args[0])?;
            // Allow: Value contains OnceLock for fingerprint caching, but Hash/Eq are stable
            #[allow(clippy::mutable_key_type)]
            let mut seen: std::collections::HashSet<Value> = std::collections::HashSet::new();

            match &fv {
                Value::Func(func) => {
                    for val in func.mapping_values() {
                        if !seen.insert(val.clone()) {
                            return Ok(Some(Value::Bool(false)));
                        }
                    }
                }
                Value::IntFunc(func) => {
                    for val in func.values.iter() {
                        if !seen.insert(val.clone()) {
                            return Ok(Some(Value::Bool(false)));
                        }
                    }
                }
                Value::Seq(_) | Value::Tuple(_) => {
                    let seq = fv.as_seq_or_tuple_elements().unwrap();
                    for val in seq.iter() {
                        if !seen.insert(val.clone()) {
                            return Ok(Some(Value::Bool(false)));
                        }
                    }
                }
                Value::Record(rec) => {
                    for (_k, val) in rec.iter() {
                        if !seen.insert(val.clone()) {
                            return Ok(Some(Value::Bool(false)));
                        }
                    }
                }
                _ => {
                    return Err(EvalError::type_error("Function", &fv, Some(args[0].span)));
                }
            }

            Ok(Some(Value::Bool(true)))
        }

        "IsSurjective" => {
            // IsSurjective(f, S, T) - TRUE if f restricted to S maps onto T
            // IsSurjective(f, S, T) == \A t \in T: \E s \in S: f[s] = t
            check_arity(name, args, 3, span)?;
            let fv = eval(ctx, &args[0])?;
            let func = fv
                .to_func_coerced()
                .ok_or_else(|| EvalError::type_error("Function", &fv, Some(args[0].span)))?;
            let sv = eval(ctx, &args[1])?;
            let source = sv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv, Some(args[1].span)))?;
            let tv = eval(ctx, &args[2])?;
            let target = tv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &tv, Some(args[2].span)))?;

            // Get range of f restricted to S
            let range: OrdSet<Value> = func
                .mapping_iter()
                .filter(|(k, _)| source.contains(k))
                .map(|(_, v)| v.clone())
                .collect();

            // Check if every element of T is in the range
            let is_surjective = target.iter().all(|t| range.contains(t));
            Ok(Some(Value::Bool(is_surjective)))
        }

        "IsBijection" => {
            // IsBijection(f, S, T) - TRUE if f is a bijection from S to T
            // IsBijection(f, S, T) == IsInjective(Restrict(f, S)) /\ IsSurjective(f, S, T)
            check_arity(name, args, 3, span)?;
            let fv = eval(ctx, &args[0])?;
            let func = fv
                .to_func_coerced()
                .ok_or_else(|| EvalError::type_error("Function", &fv, Some(args[0].span)))?;
            let sv = eval(ctx, &args[1])?;
            let source = sv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv, Some(args[1].span)))?;
            let tv = eval(ctx, &args[2])?;
            let target = tv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &tv, Some(args[2].span)))?;

            // Get restricted function values
            let restricted: Vec<&Value> = func
                .mapping_iter()
                .filter(|(k, _)| source.contains(k))
                .map(|(_, v)| v)
                .collect();

            // Check injective: all values are unique
            let unique_count = restricted
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len();
            let is_injective = restricted.len() == unique_count;

            // Get range of restricted function
            let range: OrdSet<Value> = restricted.into_iter().cloned().collect();

            // Check surjective: every element of T is in the range
            let is_surjective = target.iter().all(|t| range.contains(t));

            Ok(Some(Value::Bool(is_injective && is_surjective)))
        }

        "Inverse" => {
            // Inverse(f, S, T) - inverse of function f from S to T
            // Only valid if f is a bijection from S to T
            // Inverse(f, S, T) == [t \in T |-> CHOOSE s \in S: f[s] = t]
            check_arity(name, args, 3, span)?;
            let fv = eval(ctx, &args[0])?;
            let func = fv
                .to_func_coerced()
                .ok_or_else(|| EvalError::type_error("Function", &fv, Some(args[0].span)))?;
            let sv = eval(ctx, &args[1])?;
            let source = sv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv, Some(args[1].span)))?;
            let tv = eval(ctx, &args[2])?;
            let target = tv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &tv, Some(args[2].span)))?;

            // Build inverse mapping
            let mut inverse_entries: Vec<(Value, Value)> = Vec::new();

            for t in target.iter() {
                // Find s \in S such that f[s] = t
                let s = func
                    .mapping_iter()
                    .find(|(k, v)| source.contains(k) && *v == t)
                    .map(|(k, _)| k.clone());

                if let Some(s) = s {
                    inverse_entries.push((t.clone(), s));
                }
            }

            // Sort entries by key to maintain FuncValue invariant
            inverse_entries.sort_by(|a, b| a.0.cmp(&b.0));
            Ok(Some(Value::Func(FuncValue::from_sorted_entries(
                inverse_entries,
            ))))
        }

        "Injection" => {
            // Injection(S, T) - the set of all injective functions from S to T
            // Injection(S, T) == { f \in [S -> T] : IsInjective(f) }
            // An injection maps distinct domain elements to distinct range elements
            check_arity(name, args, 2, span)?;
            let sv = eval(ctx, &args[0])?;
            let source = sv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv, Some(args[0].span)))?;
            let tv = eval(ctx, &args[1])?;
            let target = tv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &tv, Some(args[1].span)))?;

            // For injection to exist, |S| <= |T|
            let source_elems: Vec<Value> = source.iter().cloned().collect();
            let target_elems: Vec<Value> = target.iter().cloned().collect();

            if source_elems.len() > target_elems.len() {
                return Ok(Some(Value::Set(SortedSet::new())));
            }

            let mut injections = OrdSet::new();

            // Generate all injective mappings using permutations
            fn generate_injections(
                source: &[Value],
                target: &[Value],
                src_idx: usize,
                used: &mut Vec<bool>,
                current_mapping: &mut im::OrdMap<Value, Value>,
                current_domain: &mut OrdSet<Value>,
                injections: &mut OrdSet<Value>,
            ) {
                if src_idx == source.len() {
                    injections.insert(Value::Func(FuncValue::new(
                        current_domain.clone(),
                        current_mapping.clone(),
                    )));
                    return;
                }

                for (t_idx, t) in target.iter().enumerate() {
                    if !used[t_idx] {
                        used[t_idx] = true;
                        current_domain.insert(source[src_idx].clone());
                        current_mapping.insert(source[src_idx].clone(), t.clone());

                        generate_injections(
                            source,
                            target,
                            src_idx + 1,
                            used,
                            current_mapping,
                            current_domain,
                            injections,
                        );

                        current_mapping.remove(&source[src_idx]);
                        current_domain.remove(&source[src_idx]);
                        used[t_idx] = false;
                    }
                }
            }

            let mut used = vec![false; target_elems.len()];
            let mut current_mapping = im::OrdMap::new();
            let mut current_domain = OrdSet::new();
            generate_injections(
                &source_elems,
                &target_elems,
                0,
                &mut used,
                &mut current_mapping,
                &mut current_domain,
                &mut injections,
            );

            Ok(Some(Value::Set(SortedSet::from_ord_set(&injections))))
        }

        "Surjection" => {
            // Surjection(S, T) - the set of all surjective functions from S to T
            // Surjection(S, T) == { f \in [S -> T] : IsSurjective(f, S, T) }
            // A surjection covers every element in T at least once
            check_arity(name, args, 2, span)?;
            let sv = eval(ctx, &args[0])?;
            let source = sv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv, Some(args[0].span)))?;
            let tv = eval(ctx, &args[1])?;
            let target = tv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &tv, Some(args[1].span)))?;

            // For surjection to exist, |S| >= |T|
            let source_elems: Vec<Value> = source.iter().cloned().collect();
            let target_elems: Vec<Value> = target.iter().cloned().collect();

            if source_elems.len() < target_elems.len() {
                return Ok(Some(Value::Set(SortedSet::new())));
            }

            // Generate all functions [S -> T] and filter to surjections
            // This is expensive but correct
            let mut surjections = OrdSet::new();

            fn generate_all_funcs(
                source: &[Value],
                target: &[Value],
                src_idx: usize,
                current_mapping: &mut im::OrdMap<Value, Value>,
                current_domain: &mut OrdSet<Value>,
                surjections: &mut OrdSet<Value>,
            ) {
                if src_idx == source.len() {
                    // Check if surjective
                    let range: OrdSet<Value> = current_mapping.values().cloned().collect();
                    if target.iter().all(|t| range.contains(t)) {
                        surjections.insert(Value::Func(FuncValue::new(
                            current_domain.clone(),
                            current_mapping.clone(),
                        )));
                    }
                    return;
                }

                for t in target.iter() {
                    current_domain.insert(source[src_idx].clone());
                    current_mapping.insert(source[src_idx].clone(), t.clone());

                    generate_all_funcs(
                        source,
                        target,
                        src_idx + 1,
                        current_mapping,
                        current_domain,
                        surjections,
                    );

                    current_mapping.remove(&source[src_idx]);
                    current_domain.remove(&source[src_idx]);
                }
            }

            let mut current_mapping = im::OrdMap::new();
            let mut current_domain = OrdSet::new();
            generate_all_funcs(
                &source_elems,
                &target_elems,
                0,
                &mut current_mapping,
                &mut current_domain,
                &mut surjections,
            );

            Ok(Some(Value::Set(SortedSet::from_ord_set(&surjections))))
        }

        "Bijection" => {
            // Bijection(S, T) - the set of all bijective functions from S to T
            // Bijection(S, T) == Injection(S, T) \cap Surjection(S, T)
            // A bijection is both injective and surjective; requires |S| = |T|
            check_arity(name, args, 2, span)?;
            let sv = eval(ctx, &args[0])?;
            let source = sv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv, Some(args[0].span)))?;
            let tv = eval(ctx, &args[1])?;
            let target = tv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &tv, Some(args[1].span)))?;

            let source_elems: Vec<Value> = source.iter().cloned().collect();
            let target_elems: Vec<Value> = target.iter().cloned().collect();

            // Bijection requires same cardinality
            if source_elems.len() != target_elems.len() {
                return Ok(Some(Value::Set(SortedSet::new())));
            }

            // Generate all permutations (bijections are permutations of target)
            let mut bijections = OrdSet::new();

            fn generate_bijections(
                source: &[Value],
                target: &[Value],
                src_idx: usize,
                used: &mut Vec<bool>,
                current_mapping: &mut im::OrdMap<Value, Value>,
                current_domain: &mut OrdSet<Value>,
                bijections: &mut OrdSet<Value>,
            ) {
                if src_idx == source.len() {
                    bijections.insert(Value::Func(FuncValue::new(
                        current_domain.clone(),
                        current_mapping.clone(),
                    )));
                    return;
                }

                for (t_idx, t) in target.iter().enumerate() {
                    if !used[t_idx] {
                        used[t_idx] = true;
                        current_domain.insert(source[src_idx].clone());
                        current_mapping.insert(source[src_idx].clone(), t.clone());

                        generate_bijections(
                            source,
                            target,
                            src_idx + 1,
                            used,
                            current_mapping,
                            current_domain,
                            bijections,
                        );

                        current_mapping.remove(&source[src_idx]);
                        current_domain.remove(&source[src_idx]);
                        used[t_idx] = false;
                    }
                }
            }

            let mut used = vec![false; target_elems.len()];
            let mut current_mapping = im::OrdMap::new();
            let mut current_domain = OrdSet::new();
            generate_bijections(
                &source_elems,
                &target_elems,
                0,
                &mut used,
                &mut current_mapping,
                &mut current_domain,
                &mut bijections,
            );

            Ok(Some(Value::Set(SortedSet::from_ord_set(&bijections))))
        }

        "ExistsInjection" => {
            // ExistsInjection(S, T) - TRUE iff there exists an injection from S to T
            // ExistsInjection(S, T) == Cardinality(S) <= Cardinality(T)
            check_arity(name, args, 2, span)?;
            let sv = eval(ctx, &args[0])?;
            let source = sv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv, Some(args[0].span)))?;
            let tv = eval(ctx, &args[1])?;
            let target = tv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &tv, Some(args[1].span)))?;

            Ok(Some(Value::Bool(source.len() <= target.len())))
        }

        "ExistsSurjection" => {
            // ExistsSurjection(S, T) - TRUE iff there exists a surjection from S to T
            // ExistsSurjection(S, T) == Cardinality(S) >= Cardinality(T)
            check_arity(name, args, 2, span)?;
            let sv = eval(ctx, &args[0])?;
            let source = sv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv, Some(args[0].span)))?;
            let tv = eval(ctx, &args[1])?;
            let target = tv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &tv, Some(args[1].span)))?;

            Ok(Some(Value::Bool(source.len() >= target.len())))
        }

        "ExistsBijection" => {
            // ExistsBijection(S, T) - TRUE iff there exists a bijection from S to T
            // ExistsBijection(S, T) == Cardinality(S) = Cardinality(T)
            check_arity(name, args, 2, span)?;
            let sv = eval(ctx, &args[0])?;
            let source = sv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv, Some(args[0].span)))?;
            let tv = eval(ctx, &args[1])?;
            let target = tv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &tv, Some(args[1].span)))?;

            Ok(Some(Value::Bool(source.len() == target.len())))
        }

        "Reverse" => {
            // Reverse(s) - reverse the elements of a sequence
            check_arity(name, args, 1, span)?;
            let sv = eval(ctx, &args[0])?;
            let seq = sv
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv, Some(args[0].span)))?;
            let reversed: Vec<Value> = seq.iter().rev().cloned().collect();
            Ok(Some(Value::Seq(reversed.into())))
        }

        "Front" => {
            // Front(s) - all but the last element (opposite of Tail)
            check_arity(name, args, 1, span)?;
            let sv = eval(ctx, &args[0])?;
            // Fast path: use O(log n) front for SeqValue
            if let Some(seq_value) = sv.as_seq_value() {
                if seq_value.is_empty() {
                    return Err(EvalError::IndexOutOfBounds {
                        index: 1,
                        len: 0,
                        span,
                    });
                }
                return Ok(Some(Value::Seq(seq_value.front())));
            }
            // Fallback for Tuple
            let seq = sv
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv, Some(args[0].span)))?;
            if seq.is_empty() {
                return Err(EvalError::IndexOutOfBounds {
                    index: 1,
                    len: 0,
                    span,
                });
            }
            Ok(Some(Value::Seq(seq[..seq.len() - 1].to_vec().into())))
        }

        "Last" => {
            // Last(s) - the last element of a sequence
            check_arity(name, args, 1, span)?;
            let sv = eval(ctx, &args[0])?;
            let seq = sv
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv, Some(args[0].span)))?;
            Ok(Some(seq.last().cloned().ok_or(
                EvalError::IndexOutOfBounds {
                    index: 1,
                    len: 0,
                    span,
                },
            )?))
        }

        "SetToSeq" => {
            // SetToSeq(S) - convert a set to an arbitrary sequence
            // The order is deterministic (based on Value ordering) but arbitrary
            check_arity(name, args, 1, span)?;
            let sv = eval(ctx, &args[0])?;
            let set = sv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv, Some(args[0].span)))?;
            let seq: Vec<Value> = set.iter().cloned().collect();
            Ok(Some(Value::Seq(seq.into())))
        }

        "SetToSortSeq" => {
            // SetToSortSeq(S, Op) - convert set to sequence sorted by comparator Op
            // Op(a, b) should return TRUE iff a < b
            check_arity(name, args, 2, span)?;
            let sv = eval(ctx, &args[0])?;
            let set = sv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv, Some(args[0].span)))?;

            // Get the comparator operator name from the second argument
            let cmp_name = match &args[1].node {
                Expr::Ident(name) => name.clone(),
                _ => {
                    return Err(EvalError::Internal {
                        message: "SetToSortSeq requires an operator name as second argument".into(),
                        span,
                    })
                }
            };

            // Get the operator definition
            let cmp_def = ctx
                .get_op(&cmp_name)
                .ok_or_else(|| EvalError::UndefinedOp {
                    name: cmp_name.clone(),
                    span,
                })?;

            if cmp_def.params.len() != 2 {
                return Err(EvalError::ArityMismatch {
                    op: cmp_name,
                    expected: 2,
                    got: cmp_def.params.len(),
                    span,
                });
            }

            // Collect and sort elements using the comparator
            let mut elements: Vec<Value> = set.iter().cloned().collect();
            // Use a simple insertion sort with the custom comparator
            for i in 1..elements.len() {
                let mut j = i;
                while j > 0 {
                    let new_ctx = ctx
                        .bind(cmp_def.params[0].name.node.as_str(), elements[j].clone())
                        .bind(
                            cmp_def.params[1].name.node.as_str(),
                            elements[j - 1].clone(),
                        );
                    let less = eval(&new_ctx, &cmp_def.body)?.as_bool().unwrap_or(false);
                    if less {
                        elements.swap(j, j - 1);
                        j -= 1;
                    } else {
                        break;
                    }
                }
            }
            Ok(Some(Value::Seq(elements.into())))
        }

        "ToSet" => {
            // ToSet(s) - convert sequence to set (range of sequence values)
            // ToSet(s) == { s[i] : i \in DOMAIN s }
            check_arity(name, args, 1, span)?;
            let sv = eval(ctx, &args[0])?;
            let seq = sv
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv, Some(args[0].span)))?;
            let set: OrdSet<Value> = seq.iter().cloned().collect();
            Ok(Some(Value::Set(SortedSet::from_ord_set(&set))))
        }

        "Cons" => {
            // Cons(e, s) - prepend element e to sequence s
            // Cons(e, s) == <<e>> \o s
            check_arity(name, args, 2, span)?;
            let elem = eval(ctx, &args[0])?;
            let sv = eval(ctx, &args[1])?;
            let seq = sv
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv, Some(args[1].span)))?;
            let mut new_seq = vec![elem];
            new_seq.extend(seq.iter().cloned());
            Ok(Some(Value::Seq(new_seq.into())))
        }

        "Contains" => {
            // Contains(s, e) - TRUE if sequence s contains element e
            // Contains(s, e) == \E i \in DOMAIN s : s[i] = e
            check_arity(name, args, 2, span)?;
            let sv = eval(ctx, &args[0])?;
            let seq = sv
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv, Some(args[0].span)))?;
            let elem = eval(ctx, &args[1])?;
            Ok(Some(Value::Bool(seq.contains(&elem))))
        }

        "IsPrefix" => {
            // IsPrefix(s, t) - TRUE if s is a prefix of t
            // IsPrefix(s, t) == SubSeq(t, 1, Len(s)) = s
            check_arity(name, args, 2, span)?;
            let sv1 = eval(ctx, &args[0])?;
            let seq1 = sv1
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv1, Some(args[0].span)))?;
            let sv2 = eval(ctx, &args[1])?;
            let seq2 = sv2
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv2, Some(args[1].span)))?;
            let is_prefix = seq2.len() >= seq1.len() && seq1[..] == seq2[..seq1.len()];
            Ok(Some(Value::Bool(is_prefix)))
        }

        "IsSuffix" => {
            // IsSuffix(s, t) - TRUE if s is a suffix of t
            // IsSuffix(s, t) == SubSeq(t, Len(t) - Len(s) + 1, Len(t)) = s
            check_arity(name, args, 2, span)?;
            let sv1 = eval(ctx, &args[0])?;
            let seq1 = sv1
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv1, Some(args[0].span)))?;
            let sv2 = eval(ctx, &args[1])?;
            let seq2 = sv2
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv2, Some(args[1].span)))?;
            let is_suffix = seq2.len() >= seq1.len() && seq1[..] == seq2[seq2.len() - seq1.len()..];
            Ok(Some(Value::Bool(is_suffix)))
        }

        "IsStrictPrefix" => {
            // IsStrictPrefix(s, t) - TRUE if s is a strict prefix of t (s != t)
            // IsStrictPrefix(s, t) == IsPrefix(s, t) /\ s /= t
            check_arity(name, args, 2, span)?;
            let sv1 = eval(ctx, &args[0])?;
            let seq1 = sv1
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv1, Some(args[0].span)))?;
            let sv2 = eval(ctx, &args[1])?;
            let seq2 = sv2
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv2, Some(args[1].span)))?;
            let is_strict_prefix = seq1.len() < seq2.len() && seq1[..] == seq2[..seq1.len()];
            Ok(Some(Value::Bool(is_strict_prefix)))
        }

        "IsStrictSuffix" => {
            // IsStrictSuffix(s, t) - TRUE if s is a strict suffix of t (s != t)
            // IsStrictSuffix(s, t) == IsSuffix(s, t) /\ s /= t
            check_arity(name, args, 2, span)?;
            let sv1 = eval(ctx, &args[0])?;
            let seq1 = sv1
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv1, Some(args[0].span)))?;
            let sv2 = eval(ctx, &args[1])?;
            let seq2 = sv2
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv2, Some(args[1].span)))?;
            let is_strict_suffix =
                seq1.len() < seq2.len() && seq1[..] == seq2[seq2.len() - seq1.len()..];
            Ok(Some(Value::Bool(is_strict_suffix)))
        }

        "Snoc" => {
            // Snoc(s, e) - append element e to sequence s (opposite of Cons)
            // Snoc(s, e) == Append(s, e)
            check_arity(name, args, 2, span)?;
            let sv = eval(ctx, &args[0])?;
            let elem = eval(ctx, &args[1])?;
            // Fast path: use O(log n) append for SeqValue
            if let Some(seq_value) = sv.as_seq_value() {
                return Ok(Some(Value::Seq(seq_value.append(elem))));
            }
            // Fallback for Tuple
            let seq = sv
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv, Some(args[0].span)))?;
            let mut new_seq: Vec<Value> = seq.to_vec();
            new_seq.push(elem);
            Ok(Some(Value::Seq(new_seq.into())))
        }

        "Prefixes" => {
            // Prefixes(s) - the set of all prefixes of sequence s (including s and <<>>)
            // Prefixes(s) == { SubSeq(s, 1, n) : n \in 0..Len(s) }
            check_arity(name, args, 1, span)?;
            let sv = eval(ctx, &args[0])?;
            let seq = sv
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv, Some(args[0].span)))?;
            let mut prefixes = OrdSet::new();
            for i in 0..=seq.len() {
                prefixes.insert(Value::Seq(seq[..i].to_vec().into()));
            }
            Ok(Some(Value::Set(SortedSet::from_ord_set(&prefixes))))
        }

        "Suffixes" => {
            // Suffixes(s) - the set of all suffixes of sequence s (including s and <<>>)
            // Suffixes(s) == { SubSeq(s, n, Len(s)) : n \in 1..Len(s)+1 }
            check_arity(name, args, 1, span)?;
            let sv = eval(ctx, &args[0])?;
            let seq = sv
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv, Some(args[0].span)))?;
            let mut suffixes = OrdSet::new();
            for i in 0..=seq.len() {
                suffixes.insert(Value::Seq(seq[i..].to_vec().into()));
            }
            Ok(Some(Value::Set(SortedSet::from_ord_set(&suffixes))))
        }

        "SelectInSeq" => {
            // SelectInSeq(s, Test) - find the first index i such that Test(s[i]) is TRUE
            // Returns 0 if no such index exists (TLA+ uses 0 as "not found")
            check_arity(name, args, 2, span)?;
            let sv = eval(ctx, &args[0])?;
            let seq = sv
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv, Some(args[0].span)))?;

            let test_name = match &args[1].node {
                Expr::Ident(name) => name.clone(),
                _ => {
                    return Err(EvalError::Internal {
                        message: "SelectInSeq requires operator name as second argument".into(),
                        span,
                    })
                }
            };

            let test_def = ctx
                .get_op(&test_name)
                .ok_or_else(|| EvalError::UndefinedOp {
                    name: test_name.clone(),
                    span,
                })?;

            if test_def.params.len() != 1 {
                return Err(EvalError::ArityMismatch {
                    op: test_name,
                    expected: 1,
                    got: test_def.params.len(),
                    span,
                });
            }

            for (i, elem) in seq.iter().enumerate() {
                let new_ctx = ctx.bind(test_def.params[0].name.node.as_str(), elem.clone());
                if let Some(true) = eval(&new_ctx, &test_def.body)?.as_bool() {
                    return Ok(Some(Value::SmallInt(i as i64 + 1)));
                }
            }
            Ok(Some(Value::SmallInt(0)))
        }

        "SelectLastInSeq" => {
            // SelectLastInSeq(s, Test) - find the last index i such that Test(s[i]) is TRUE
            // Returns 0 if no such index exists
            check_arity(name, args, 2, span)?;
            let sv = eval(ctx, &args[0])?;
            let seq = sv
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv, Some(args[0].span)))?;

            let test_name = match &args[1].node {
                Expr::Ident(name) => name.clone(),
                _ => {
                    return Err(EvalError::Internal {
                        message: "SelectLastInSeq requires operator name as second argument".into(),
                        span,
                    })
                }
            };

            let test_def = ctx
                .get_op(&test_name)
                .ok_or_else(|| EvalError::UndefinedOp {
                    name: test_name.clone(),
                    span,
                })?;

            if test_def.params.len() != 1 {
                return Err(EvalError::ArityMismatch {
                    op: test_name,
                    expected: 1,
                    got: test_def.params.len(),
                    span,
                });
            }

            let mut last_idx = 0i64;
            for (i, elem) in seq.iter().enumerate() {
                let new_ctx = ctx.bind(test_def.params[0].name.node.as_str(), elem.clone());
                if let Some(true) = eval(&new_ctx, &test_def.body)?.as_bool() {
                    last_idx = (i as i64) + 1;
                }
            }
            Ok(Some(Value::SmallInt(last_idx)))
        }

        "BoundedSeq" | "SeqOf" => {
            // BoundedSeq(S, n) == UNION { [1..m -> S] : m \in 0..n }
            // The set of all sequences of elements from S with length at most n
            check_arity(name, args, 2, span)?;
            let sv = eval(ctx, &args[0])?;
            let set = sv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv, Some(args[0].span)))?;
            let nv = eval(ctx, &args[1])?;
            let n = nv
                .as_int()
                .ok_or_else(|| EvalError::type_error("Int", &nv, Some(args[1].span)))?
                .to_i64()
                .unwrap_or(0);

            if n < 0 {
                return Ok(Some(Value::Set(SortedSet::new())));
            }

            // Generate all sequences of length 0 to n
            let elements: Vec<Value> = set.iter().cloned().collect();
            let mut all_seqs = OrdSet::new();

            // Length 0: empty sequence
            all_seqs.insert(Value::Seq(Vec::new().into()));

            // For each length from 1 to n, generate all sequences
            fn generate_seqs(
                elements: &[Value],
                max_len: usize,
                current: Vec<Value>,
                all: &mut OrdSet<Value>,
            ) {
                if current.len() <= max_len {
                    all.insert(Value::Seq(current.clone().into()));
                    if current.len() < max_len {
                        for e in elements {
                            let mut next = current.clone();
                            next.push(e.clone());
                            generate_seqs(elements, max_len, next, all);
                        }
                    }
                }
            }

            generate_seqs(&elements, n as usize, vec![], &mut all_seqs);
            Ok(Some(Value::Set(SortedSet::from_ord_set(&all_seqs))))
        }

        "TupleOf" => {
            // TupleOf(S, n) == [1..n -> S]
            // The set of all n-tuples with elements from S
            check_arity(name, args, 2, span)?;
            let sv = eval(ctx, &args[0])?;
            let set = sv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv, Some(args[0].span)))?;
            let nv = eval(ctx, &args[1])?;
            let n = nv
                .as_int()
                .ok_or_else(|| EvalError::type_error("Int", &nv, Some(args[1].span)))?
                .to_i64()
                .unwrap_or(0);

            if n < 0 {
                return Ok(Some(Value::Set(SortedSet::new())));
            }
            if n == 0 {
                // Only the empty tuple
                let mut result = OrdSet::new();
                result.insert(Value::Seq(Vec::new().into()));
                return Ok(Some(Value::Set(SortedSet::from_ord_set(&result))));
            }

            // Generate all n-tuples
            let elements: Vec<Value> = set.iter().cloned().collect();
            let mut all_tuples = OrdSet::new();

            fn generate_tuples(
                elements: &[Value],
                n: usize,
                current: Vec<Value>,
                all: &mut OrdSet<Value>,
            ) {
                if current.len() == n {
                    all.insert(Value::Seq(current.into()));
                } else {
                    for e in elements {
                        let mut next = current.clone();
                        next.push(e.clone());
                        generate_tuples(elements, n, next, all);
                    }
                }
            }

            generate_tuples(&elements, n as usize, vec![], &mut all_tuples);
            Ok(Some(Value::Set(SortedSet::from_ord_set(&all_tuples))))
        }

        "Indices" => {
            // Indices(s) - the set {1, ..., Len(s)}
            // Indices(s) == 1..Len(s)
            check_arity(name, args, 1, span)?;
            let sv = eval(ctx, &args[0])?;
            let seq = sv
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv, Some(args[0].span)))?;
            Ok(Some(Value::Interval(IntervalValue::new(
                BigInt::one(),
                BigInt::from(seq.len() as i64),
            ))))
        }

        "InsertAt" => {
            // InsertAt(s, i, e) - insert e at position i (1-indexed)
            // Elements at i and beyond shift right
            check_arity(name, args, 3, span)?;
            let sv = eval(ctx, &args[0])?;
            let seq = sv
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv, Some(args[0].span)))?;
            let iv = eval(ctx, &args[1])?;
            let i = iv
                .as_int()
                .ok_or_else(|| EvalError::type_error("Int", &iv, Some(args[1].span)))?
                .to_i64()
                .unwrap_or(0);
            let elem = eval(ctx, &args[2])?;

            // TLA+ is 1-indexed; i must be in 1..Len(s)+1
            if i < 1 || i > (seq.len() as i64 + 1) {
                return Err(EvalError::IndexOutOfBounds {
                    index: i,
                    len: seq.len(),
                    span,
                });
            }
            let idx = (i - 1) as usize;
            let mut new_seq: Vec<Value> = seq.to_vec();
            new_seq.insert(idx, elem);
            Ok(Some(Value::Seq(new_seq.into())))
        }

        "RemoveAt" => {
            // RemoveAt(s, i) - remove element at position i (1-indexed)
            check_arity(name, args, 2, span)?;
            let sv = eval(ctx, &args[0])?;
            let seq = sv
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv, Some(args[0].span)))?;
            let iv = eval(ctx, &args[1])?;
            let i = iv
                .as_int()
                .ok_or_else(|| EvalError::type_error("Int", &iv, Some(args[1].span)))?
                .to_i64()
                .unwrap_or(0);

            // TLA+ is 1-indexed; i must be in 1..Len(s)
            if i < 1 || i > seq.len() as i64 {
                return Err(EvalError::IndexOutOfBounds {
                    index: i,
                    len: seq.len(),
                    span,
                });
            }
            let idx = (i - 1) as usize;
            let mut new_seq: Vec<Value> = seq.to_vec();
            new_seq.remove(idx);
            Ok(Some(Value::Seq(new_seq.into())))
        }

        "ReplaceAt" => {
            // ReplaceAt(s, i, e) - replace element at position i with e (1-indexed)
            check_arity(name, args, 3, span)?;
            let sv = eval(ctx, &args[0])?;
            let seq = sv
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv, Some(args[0].span)))?;
            let iv = eval(ctx, &args[1])?;
            let i = iv
                .as_int()
                .ok_or_else(|| EvalError::type_error("Int", &iv, Some(args[1].span)))?
                .to_i64()
                .unwrap_or(0);
            let elem = eval(ctx, &args[2])?;

            // TLA+ is 1-indexed; i must be in 1..Len(s)
            if i < 1 || i > seq.len() as i64 {
                return Err(EvalError::IndexOutOfBounds {
                    index: i,
                    len: seq.len(),
                    span,
                });
            }
            let idx = (i - 1) as usize;
            let mut new_seq: Vec<Value> = seq.to_vec();
            new_seq[idx] = elem;
            Ok(Some(Value::Seq(new_seq.into())))
        }

        "Remove" => {
            // Remove(s, e) - remove first occurrence of e from sequence s
            check_arity(name, args, 2, span)?;
            let sv = eval(ctx, &args[0])?;
            let seq = sv
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv, Some(args[0].span)))?;
            let elem = eval(ctx, &args[1])?;

            let mut new_seq: Vec<Value> = seq.to_vec();
            if let Some(pos) = new_seq.iter().position(|x| x == &elem) {
                new_seq.remove(pos);
            }
            Ok(Some(Value::Seq(new_seq.into())))
        }

        "ReplaceAll" => {
            // ReplaceAll(s, old, new) - replace all occurrences of old with new in sequence s
            check_arity(name, args, 3, span)?;
            let sv = eval(ctx, &args[0])?;
            let seq = sv
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv, Some(args[0].span)))?;
            let old_elem = eval(ctx, &args[1])?;
            let new_elem = eval(ctx, &args[2])?;

            let new_seq: Vec<Value> = seq
                .iter()
                .map(|x| {
                    if x == &old_elem {
                        new_elem.clone()
                    } else {
                        x.clone()
                    }
                })
                .collect();
            Ok(Some(Value::Seq(new_seq.into())))
        }

        "Interleave" => {
            // Interleave(s, t) - interleave two sequences
            // Interleave(<<a,b,c>>, <<1,2,3>>) == <<a, 1, b, 2, c, 3>>
            // If one sequence is longer, its remaining elements are appended
            check_arity(name, args, 2, span)?;
            let sv1 = eval(ctx, &args[0])?;
            let seq1 = sv1
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv1, Some(args[0].span)))?;
            let sv2 = eval(ctx, &args[1])?;
            let seq2 = sv2
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv2, Some(args[1].span)))?;

            let max_len = seq1.len().max(seq2.len());
            let mut result: Vec<Value> = Vec::with_capacity(seq1.len() + seq2.len());
            for i in 0..max_len {
                if i < seq1.len() {
                    result.push(seq1[i].clone());
                }
                if i < seq2.len() {
                    result.push(seq2[i].clone());
                }
            }
            Ok(Some(Value::Seq(result.into())))
        }

        "SubSeqs" => {
            // SubSeqs(s) - set of all contiguous subsequences of s (including empty sequence)
            // SubSeqs(<<a,b>>) == { <<>>, <<a>>, <<b>>, <<a,b>> }
            check_arity(name, args, 1, span)?;
            let sv = eval(ctx, &args[0])?;
            let seq = sv
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv, Some(args[0].span)))?;

            let mut subseqs: OrdSet<Value> = OrdSet::new();
            // Include empty sequence
            subseqs.insert(Value::Seq(Vec::new().into()));

            // Generate all contiguous subsequences
            let len = seq.len();
            for start in 0..len {
                for end in (start + 1)..=len {
                    let subseq: Vec<Value> = seq[start..end].to_vec();
                    subseqs.insert(Value::Seq(subseq.into()));
                }
            }
            Ok(Some(Value::Set(SortedSet::from_ord_set(&subseqs))))
        }

        "SetToSeqs" => {
            // SetToSeqs(S) - set of all permutations (orderings) of set S
            // SetToSeqs({1,2}) == { <<1,2>>, <<2,1>> }
            check_arity(name, args, 1, span)?;
            let sv = eval(ctx, &args[0])?;
            let set = sv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv, Some(args[0].span)))?;

            // Generate all permutations of the set elements
            let elements: Vec<Value> = set.iter().cloned().collect();
            let mut result: OrdSet<Value> = OrdSet::new();

            fn generate_permutations(
                elements: &[Value],
                current: &mut Vec<Value>,
                used: &mut Vec<bool>,
                result: &mut OrdSet<Value>,
            ) {
                if current.len() == elements.len() {
                    result.insert(Value::Seq(current.clone().into()));
                    return;
                }
                for i in 0..elements.len() {
                    if !used[i] {
                        used[i] = true;
                        current.push(elements[i].clone());
                        generate_permutations(elements, current, used, result);
                        current.pop();
                        used[i] = false;
                    }
                }
            }

            let mut current = Vec::new();
            let mut used = vec![false; elements.len()];
            generate_permutations(&elements, &mut current, &mut used, &mut result);
            Ok(Some(Value::Set(SortedSet::from_ord_set(&result))))
        }

        "AllSubSeqs" => {
            // AllSubSeqs(s) - set of ALL subsequences (not necessarily contiguous)
            // AllSubSeqs(<<a,b>>) == { <<>>, <<a>>, <<b>>, <<a,b>> }
            // This includes all 2^n subsequences where elements maintain relative order
            check_arity(name, args, 1, span)?;
            let sv = eval(ctx, &args[0])?;
            let seq = sv
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv, Some(args[0].span)))?;

            let mut result: OrdSet<Value> = OrdSet::new();
            let n = seq.len();

            // Generate all 2^n subsequences using bitmask
            for mask in 0..(1u64 << n) {
                let mut subseq: Vec<Value> = Vec::new();
                for (i, item) in seq.iter().enumerate().take(n) {
                    if mask & (1 << i) != 0 {
                        subseq.push(item.clone());
                    }
                }
                result.insert(Value::Seq(subseq.into()));
            }
            Ok(Some(Value::Set(SortedSet::from_ord_set(&result))))
        }

        "FoldLeftDomain" => {
            // FoldLeftDomain(Op, base, s) - fold left with index available
            // Op(acc, elem, idx) where idx is 1-indexed
            check_arity(name, args, 3, span)?;

            let op_name = match &args[0].node {
                Expr::Ident(name) => name.clone(),
                _ => {
                    return Err(EvalError::Internal {
                        message: "FoldLeftDomain requires operator name as first argument".into(),
                        span,
                    })
                }
            };

            let base = eval(ctx, &args[1])?;
            let sv = eval(ctx, &args[2])?;
            let seq = sv
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv, Some(args[2].span)))?;

            let op_def = ctx.get_op(&op_name).ok_or_else(|| EvalError::UndefinedOp {
                name: op_name.clone(),
                span,
            })?;

            if op_def.params.len() != 3 {
                return Err(EvalError::ArityMismatch {
                    op: op_name,
                    expected: 3,
                    got: op_def.params.len(),
                    span,
                });
            }

            let mut result = base;
            for (i, elem) in seq.iter().enumerate() {
                let idx = Value::SmallInt(i as i64 + 1); // 1-indexed
                let new_ctx = ctx
                    .bind(op_def.params[0].name.node.as_str(), result)
                    .bind(op_def.params[1].name.node.as_str(), elem.clone())
                    .bind(op_def.params[2].name.node.as_str(), idx);
                result = eval(&new_ctx, &op_def.body)?;
            }
            Ok(Some(result))
        }

        "FoldRightDomain" => {
            // FoldRightDomain(Op, s, base) - fold right with index available
            // Op(elem, acc, idx) where idx is 1-indexed
            check_arity(name, args, 3, span)?;

            let op_name = match &args[0].node {
                Expr::Ident(name) => name.clone(),
                _ => {
                    return Err(EvalError::Internal {
                        message: "FoldRightDomain requires operator name as first argument".into(),
                        span,
                    })
                }
            };

            let sv = eval(ctx, &args[1])?;
            let seq = sv
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv, Some(args[1].span)))?;
            let base = eval(ctx, &args[2])?;

            let op_def = ctx.get_op(&op_name).ok_or_else(|| EvalError::UndefinedOp {
                name: op_name.clone(),
                span,
            })?;

            if op_def.params.len() != 3 {
                return Err(EvalError::ArityMismatch {
                    op: op_name,
                    expected: 3,
                    got: op_def.params.len(),
                    span,
                });
            }

            // Fold from right to left, indices are still from original sequence
            let mut result = base;
            for (i, elem) in seq.iter().enumerate().rev() {
                let idx = Value::SmallInt(i as i64 + 1); // 1-indexed
                let new_ctx = ctx
                    .bind(op_def.params[0].name.node.as_str(), elem.clone())
                    .bind(op_def.params[1].name.node.as_str(), result)
                    .bind(op_def.params[2].name.node.as_str(), idx);
                result = eval(&new_ctx, &op_def.body)?;
            }
            Ok(Some(result))
        }

        "LongestCommonPrefix" => {
            // LongestCommonPrefix(seqs) - longest common prefix of a set of sequences
            // LongestCommonPrefix({<<1,2,3>>, <<1,2,4>>}) == <<1,2>>
            check_arity(name, args, 1, span)?;
            let sv = eval(ctx, &args[0])?;
            let seqs_set = sv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv, Some(args[0].span)))?;

            if seqs_set.is_empty() {
                // Empty set -> empty sequence
                return Ok(Some(Value::Seq(Vec::new().into())));
            }

            // Get all sequences from the set
            let seqs: Vec<std::borrow::Cow<'_, [Value]>> = seqs_set
                .iter()
                .map(|v| {
                    v.as_seq()
                        .ok_or_else(|| EvalError::type_error("Seq", v, Some(args[0].span)))
                })
                .collect::<Result<Vec<_>, _>>()?;

            if seqs.is_empty() {
                return Ok(Some(Value::Seq(Vec::new().into())));
            }

            // Find the minimum length
            let min_len = seqs.iter().map(|s| s.len()).min().unwrap_or(0);

            // Find the longest common prefix
            let mut lcp_len = 0;
            for i in 0..min_len {
                let first_elem = &seqs[0][i];
                if seqs.iter().all(|s| &s[i] == first_elem) {
                    lcp_len = i + 1;
                } else {
                    break;
                }
            }

            let lcp: Vec<Value> = seqs[0][..lcp_len].to_vec();
            Ok(Some(Value::Seq(lcp.into())))
        }

        "CommonPrefixes" => {
            // CommonPrefixes(seqs) - set of all common prefixes of a set of sequences
            // CommonPrefixes({<<1,2,3>>, <<1,2,4>>}) == { <<>>, <<1>>, <<1,2>> }
            check_arity(name, args, 1, span)?;
            let sv = eval(ctx, &args[0])?;
            let seqs_set = sv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv, Some(args[0].span)))?;

            if seqs_set.is_empty() {
                // Empty set -> set containing only empty sequence
                let mut result = OrdSet::new();
                result.insert(Value::Seq(Vec::new().into()));
                return Ok(Some(Value::Set(SortedSet::from_ord_set(&result))));
            }

            // Get all sequences from the set
            let seqs: Vec<std::borrow::Cow<'_, [Value]>> = seqs_set
                .iter()
                .map(|v| {
                    v.as_seq()
                        .ok_or_else(|| EvalError::type_error("Seq", v, Some(args[0].span)))
                })
                .collect::<Result<Vec<_>, _>>()?;

            // Find minimum length for potential common prefixes
            let min_len = seqs.iter().map(|s| s.len()).min().unwrap_or(0);

            // Find all common prefixes up to LCP
            let mut result: OrdSet<Value> = OrdSet::new();
            result.insert(Value::Seq(Vec::new().into())); // Empty sequence is always a common prefix

            for len in 1..=min_len {
                let prefix = &seqs[0][..len];
                if seqs
                    .iter()
                    .all(|s| s.len() >= len && s[..len] == prefix[..])
                {
                    result.insert(Value::Seq(prefix.to_vec().into()));
                } else {
                    break; // Once we find a non-common prefix, no longer ones can be common
                }
            }

            Ok(Some(Value::Set(SortedSet::from_ord_set(&result))))
        }

        "FlattenSeq" => {
            // FlattenSeq(ss) - flatten a sequence of sequences into a single sequence
            // FlattenSeq(<<s1, s2, ...>>) == s1 \o s2 \o ...
            check_arity(name, args, 1, span)?;
            let sv = eval(ctx, &args[0])?;
            let outer_seq = sv
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv, Some(args[0].span)))?;

            let mut result: Vec<Value> = Vec::new();
            for inner in outer_seq.iter() {
                let inner_seq = inner
                    .as_seq()
                    .ok_or_else(|| EvalError::type_error("Seq", inner, Some(args[0].span)))?;
                result.extend(inner_seq.iter().cloned());
            }
            Ok(Some(Value::Seq(result.into())))
        }

        "Zip" => {
            // Zip(s, t) - zip two sequences into a sequence of pairs
            // Zip(<<a,b>>, <<1,2>>) == << <<a,1>>, <<b,2>> >>
            // Result length is min(Len(s), Len(t))
            check_arity(name, args, 2, span)?;
            let sv1 = eval(ctx, &args[0])?;
            let seq1 = sv1
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv1, Some(args[0].span)))?;
            let sv2 = eval(ctx, &args[1])?;
            let seq2 = sv2
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv2, Some(args[1].span)))?;

            let result: Vec<Value> = seq1
                .iter()
                .zip(seq2.iter())
                .map(|(a, b)| Value::Tuple(vec![a.clone(), b.clone()].into()))
                .collect();
            Ok(Some(Value::Seq(result.into())))
        }

        "FoldLeft" => {
            // FoldLeft(Op, base, s) - fold left over sequence (alias for FoldSeq)
            check_arity(name, args, 3, span)?;

            let op_name = match &args[0].node {
                Expr::Ident(name) => name.clone(),
                _ => {
                    return Err(EvalError::Internal {
                        message: "FoldLeft requires operator name as first argument".into(),
                        span,
                    })
                }
            };

            let base = eval(ctx, &args[1])?;
            let sv = eval(ctx, &args[2])?;
            let seq = sv
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv, Some(args[2].span)))?;

            let op_def = ctx.get_op(&op_name).ok_or_else(|| EvalError::UndefinedOp {
                name: op_name.clone(),
                span,
            })?;

            if op_def.params.len() != 2 {
                return Err(EvalError::ArityMismatch {
                    op: op_name,
                    expected: 2,
                    got: op_def.params.len(),
                    span,
                });
            }

            let mut result = base;
            for elem in seq.iter() {
                let new_ctx = ctx
                    .bind(op_def.params[0].name.node.as_str(), result)
                    .bind(op_def.params[1].name.node.as_str(), elem.clone());
                result = eval(&new_ctx, &op_def.body)?;
            }

            Ok(Some(result))
        }

        "FoldRight" => {
            // FoldRight(Op, s, base) - fold right over sequence
            // Note: argument order is Op, s, base (unlike FoldLeft which is Op, base, s)
            check_arity(name, args, 3, span)?;

            let op_name = match &args[0].node {
                Expr::Ident(name) => name.clone(),
                _ => {
                    return Err(EvalError::Internal {
                        message: "FoldRight requires operator name as first argument".into(),
                        span,
                    })
                }
            };

            let sv = eval(ctx, &args[1])?;
            let seq = sv
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv, Some(args[1].span)))?;
            let base = eval(ctx, &args[2])?;

            let op_def = ctx.get_op(&op_name).ok_or_else(|| EvalError::UndefinedOp {
                name: op_name.clone(),
                span,
            })?;

            if op_def.params.len() != 2 {
                return Err(EvalError::ArityMismatch {
                    op: op_name,
                    expected: 2,
                    got: op_def.params.len(),
                    span,
                });
            }

            // Fold from right to left
            let mut result = base;
            for elem in seq.iter().rev() {
                let new_ctx = ctx
                    .bind(op_def.params[0].name.node.as_str(), elem.clone())
                    .bind(op_def.params[1].name.node.as_str(), result);
                result = eval(&new_ctx, &op_def.body)?;
            }

            Ok(Some(result))
        }

        "FoldSet" => {
            // FoldSet(Op, base, S) - fold a binary operator over a set
            // Op is the name of a binary operator
            check_arity(name, args, 3, span)?;

            // Get the operator name from the first argument
            let op_name = match &args[0].node {
                Expr::Ident(name) => name.clone(),
                _ => {
                    return Err(EvalError::Internal {
                        message: "FoldSet requires operator name as first argument".into(),
                        span,
                    })
                }
            };

            let base = eval(ctx, &args[1])?;
            let sv = eval(ctx, &args[2])?;
            let set = sv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv, Some(args[2].span)))?;

            // Get the operator definition
            let op_def = ctx.get_op(&op_name).ok_or_else(|| EvalError::UndefinedOp {
                name: op_name.clone(),
                span,
            })?;

            if op_def.params.len() != 2 {
                return Err(EvalError::ArityMismatch {
                    op: op_name,
                    expected: 2,
                    got: op_def.params.len(),
                    span,
                });
            }

            // Fold over the set elements
            let mut result = base;
            for elem in set.iter() {
                let new_ctx = ctx
                    .bind(op_def.params[0].name.node.as_str(), result)
                    .bind(op_def.params[1].name.node.as_str(), elem.clone());
                result = eval(&new_ctx, &op_def.body)?;
            }

            Ok(Some(result))
        }

        "FoldFunction" => {
            // FoldFunction(Op, base, f) - fold a binary operator over a function's range
            // Op is a binary operator argument (name, operator parameter, lambda, or built-in OpRef).
            check_arity(name, args, 3, span)?;

            let base = eval(ctx, &args[1])?;
            let fv = eval(ctx, &args[2])?;

            // Resolve Op to a closure of arity 2 (supports operator parameters and lambdas).
            let op_value = create_closure_from_arg(ctx, &args[0], "FoldFunction", 2, span)?;
            let Value::Closure(op_closure) = op_value else {
                return Err(EvalError::Internal {
                    message: "FoldFunction expected an operator argument".into(),
                    span,
                });
            };

            // Fold over the function's values (in domain order).
            let mut result = base;
            match &fv {
                Value::Func(func) => {
                    for (_key, value) in func.mapping_iter() {
                        let acc = result;
                        result = apply_closure_with_values(
                            ctx,
                            &op_closure,
                            &[value.clone(), acc],
                            span,
                        )?;
                    }
                }
                Value::IntFunc(func) => {
                    for value in func.values.iter() {
                        let acc = result;
                        result = apply_closure_with_values(
                            ctx,
                            &op_closure,
                            &[value.clone(), acc],
                            span,
                        )?;
                    }
                }
                Value::Seq(_) | Value::Tuple(_) => {
                    let seq = fv.as_seq_or_tuple_elements().unwrap();
                    for value in seq.iter() {
                        let acc = result;
                        result = apply_closure_with_values(
                            ctx,
                            &op_closure,
                            &[value.clone(), acc],
                            span,
                        )?;
                    }
                }
                Value::Record(rec) => {
                    for (_k, value) in rec.iter() {
                        let acc = result;
                        result = apply_closure_with_values(
                            ctx,
                            &op_closure,
                            &[value.clone(), acc],
                            span,
                        )?;
                    }
                }
                _ => {
                    return Err(EvalError::type_error(
                        "Function/Seq/Tuple/Record",
                        &fv,
                        Some(args[2].span),
                    ))
                }
            }

            Ok(Some(result))
        }

        "FoldFunctionOnSet" => {
            // FoldFunctionOnSet(Op, base, f, S) - fold a binary operator over a function's range
            // restricted to keys in set S
            // Op is the name of a binary operator or a built-in operator reference
            check_arity(name, args, 4, span)?;

            // Get the operator: either a user-defined operator name or a built-in operator reference
            let op_ref = match &args[0].node {
                Expr::Ident(name) => OperatorRef::UserDefined(name.clone()),
                Expr::OpRef(op) => OperatorRef::BuiltIn(op.clone()),
                _ => {
                    return Err(EvalError::Internal {
                        message: "FoldFunctionOnSet requires operator name as first argument"
                            .into(),
                        span,
                    })
                }
            };

            let base = eval(ctx, &args[1])?;
            let fv = eval(ctx, &args[2])?;
            let sv = eval(ctx, &args[3])?;
            let subset = sv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv, Some(args[3].span)))?;

            // Fold over the function's values for keys in S
            let mut result = base;
            match &fv {
                Value::Func(func) => {
                    for key in subset.iter() {
                        if let Some(value) = func.mapping_get(key) {
                            result = apply_binary_op(ctx, &op_ref, value.clone(), result, span)?;
                        }
                    }
                }
                Value::IntFunc(func) => {
                    for key in subset.iter() {
                        if let Some(value) = func.apply(key) {
                            result = apply_binary_op(ctx, &op_ref, value.clone(), result, span)?;
                        }
                    }
                }
                Value::Seq(_) | Value::Tuple(_) => {
                    // Sequences are functions from 1..n
                    let seq = fv.as_seq_or_tuple_elements().unwrap();
                    for key in subset.iter() {
                        if let Some(i) = key.as_i64() {
                            if i >= 1 && (i as usize) <= seq.len() {
                                let value = &seq[(i - 1) as usize];
                                result =
                                    apply_binary_op(ctx, &op_ref, value.clone(), result, span)?;
                            }
                        }
                    }
                }
                _ => return Err(EvalError::type_error("Func/Seq", &fv, Some(args[2].span))),
            }

            Ok(Some(result))
        }

        // RestrictDomain(f, P) - restrict function to domain elements satisfying predicate P
        // RestrictDomain(f, P) == [x \in DOMAIN f |-> f[x] : P(x)]
        "RestrictDomain" => {
            check_arity(name, args, 2, span)?;
            let fv = eval(ctx, &args[0])?;
            let func = fv
                .to_func_coerced()
                .ok_or_else(|| EvalError::type_error("Func", &fv, Some(args[0].span)))?;

            // Get the predicate (should be a lambda with one parameter)
            let (param, pred_body) = match &args[1].node {
                Expr::Lambda(params, body) if params.len() == 1 => {
                    (params[0].node.clone(), body.as_ref())
                }
                _ => {
                    return Err(EvalError::Internal {
                        message: "RestrictDomain requires a lambda predicate as second argument"
                            .into(),
                        span,
                    })
                }
            };

            let mut new_entries: Vec<(Value, Value)> = Vec::new();
            for (key, val) in func.mapping_iter() {
                let pred_ctx = ctx.bind(param.as_str(), key.clone());
                let pred_result = eval(&pred_ctx, pred_body)?;
                if pred_result.as_bool() == Some(true) {
                    new_entries.push((key.clone(), val.clone()));
                }
            }

            Ok(Some(Value::Func(FuncValue::from_sorted_entries(
                new_entries,
            ))))
        }

        // RestrictValues(f, P) - restrict function to domain elements whose range values satisfy P
        // RestrictValues(f, P) == [x \in DOMAIN f |-> f[x] : P(f[x])]
        "RestrictValues" => {
            check_arity(name, args, 2, span)?;
            let fv = eval(ctx, &args[0])?;
            let func = fv
                .to_func_coerced()
                .ok_or_else(|| EvalError::type_error("Func", &fv, Some(args[0].span)))?;

            // Get the predicate (should be a lambda with one parameter)
            let (param, pred_body) = match &args[1].node {
                Expr::Lambda(params, body) if params.len() == 1 => {
                    (params[0].node.clone(), body.as_ref())
                }
                _ => {
                    return Err(EvalError::Internal {
                        message: "RestrictValues requires a lambda predicate as second argument"
                            .into(),
                        span,
                    })
                }
            };

            let mut new_entries: Vec<(Value, Value)> = Vec::new();
            for (key, val) in func.mapping_iter() {
                let pred_ctx = ctx.bind(param.as_str(), val.clone());
                let pred_result = eval(&pred_ctx, pred_body)?;
                if pred_result.as_bool() == Some(true) {
                    new_entries.push((key.clone(), val.clone()));
                }
            }

            Ok(Some(Value::Func(FuncValue::from_sorted_entries(
                new_entries,
            ))))
        }

        // IsRestriction(f, g) - TRUE if f is a restriction of g
        // IsRestriction(f, g) == DOMAIN f \subseteq DOMAIN g /\ \A x \in DOMAIN f: f[x] = g[x]
        "IsRestriction" => {
            check_arity(name, args, 2, span)?;
            let fv = eval(ctx, &args[0])?;
            let func_f = fv
                .to_func_coerced()
                .ok_or_else(|| EvalError::type_error("Func", &fv, Some(args[0].span)))?;
            let gv = eval(ctx, &args[1])?;
            let func_g = gv
                .to_func_coerced()
                .ok_or_else(|| EvalError::type_error("Func", &gv, Some(args[1].span)))?;

            // Check: DOMAIN f \subseteq DOMAIN g
            for key in func_f.domain_iter() {
                if !func_g.domain_contains(key) {
                    return Ok(Some(Value::Bool(false)));
                }
            }

            // Check: \A x \in DOMAIN f: f[x] = g[x]
            for key in func_f.domain_iter() {
                let f_val = func_f.mapping_get(key);
                let g_val = func_g.mapping_get(key);
                if f_val != g_val {
                    return Ok(Some(Value::Bool(false)));
                }
            }

            Ok(Some(Value::Bool(true)))
        }

        // Pointwise(Op, f, g) - pointwise combination of two functions
        // Pointwise(Op, f, g) == [x \in DOMAIN f \cap DOMAIN g |-> Op(f[x], g[x])]
        "Pointwise" => {
            check_arity(name, args, 3, span)?;

            // Get the operator name from the first argument
            let op_name = match &args[0].node {
                Expr::Ident(name) => name.clone(),
                _ => {
                    return Err(EvalError::Internal {
                        message: "Pointwise requires operator name as first argument".into(),
                        span,
                    })
                }
            };

            let fv = eval(ctx, &args[1])?;
            let func_f = fv
                .to_func_coerced()
                .ok_or_else(|| EvalError::type_error("Func", &fv, Some(args[1].span)))?;
            let gv = eval(ctx, &args[2])?;
            let func_g = gv
                .to_func_coerced()
                .ok_or_else(|| EvalError::type_error("Func", &gv, Some(args[2].span)))?;

            // Get the operator definition
            let op_def = ctx.get_op(&op_name).ok_or_else(|| EvalError::UndefinedOp {
                name: op_name.clone(),
                span,
            })?;

            if op_def.params.len() != 2 {
                return Err(EvalError::ArityMismatch {
                    op: op_name,
                    expected: 2,
                    got: op_def.params.len(),
                    span,
                });
            }

            // Compute intersection of domains
            let mut new_entries: Vec<(Value, Value)> = Vec::new();

            for key in func_f.domain_iter() {
                if func_g.domain_contains(key) {
                    if let (Some(f_val), Some(g_val)) =
                        (func_f.mapping_get(key), func_g.mapping_get(key))
                    {
                        let new_ctx = ctx
                            .bind(op_def.params[0].name.node.as_str(), f_val.clone())
                            .bind(op_def.params[1].name.node.as_str(), g_val.clone());
                        let result_val = eval(&new_ctx, &op_def.body)?;
                        new_entries.push((key.clone(), result_val));
                    }
                }
            }

            Ok(Some(Value::Func(FuncValue::from_sorted_entries(
                new_entries,
            ))))
        }

        // AntiFunction(f) - reverses key-value pairs (only valid for injective functions)
        // AntiFunction(f) == [y \in Range(f) |-> CHOOSE x \in DOMAIN f : f[x] = y]
        "AntiFunction" => {
            check_arity(name, args, 1, span)?;
            let fv = eval(ctx, &args[0])?;
            let func = fv
                .to_func_coerced()
                .ok_or_else(|| EvalError::type_error("Func", &fv, Some(args[0].span)))?;

            let mut seen_values: OrdSet<Value> = OrdSet::new();
            let mut new_entries: Vec<(Value, Value)> = Vec::new();

            for (key, val) in func.mapping_iter() {
                if seen_values.contains(val) {
                    return Err(EvalError::Internal {
                        message: "AntiFunction requires an injective function".into(),
                        span,
                    });
                }
                seen_values.insert(val.clone());
                new_entries.push((val.clone(), key.clone()));
            }

            // Sort entries by new key (the original values)
            new_entries.sort_by(|a, b| a.0.cmp(&b.0));
            Ok(Some(Value::Func(FuncValue::from_sorted_entries(
                new_entries,
            ))))
        }

        "FoldSeq" => {
            // FoldSeq(Op, base, s) - fold a binary operator over a sequence (left to right)
            // Op is the name of a binary operator
            check_arity(name, args, 3, span)?;

            // Get the operator name from the first argument
            let op_name = match &args[0].node {
                Expr::Ident(name) => name.clone(),
                _ => {
                    return Err(EvalError::Internal {
                        message: "FoldSeq requires operator name as first argument".into(),
                        span,
                    })
                }
            };

            let base = eval(ctx, &args[1])?;
            let sv = eval(ctx, &args[2])?;
            let seq = sv
                .as_seq()
                .ok_or_else(|| EvalError::type_error("Seq", &sv, Some(args[2].span)))?;

            // Get the operator definition
            let op_def = ctx.get_op(&op_name).ok_or_else(|| EvalError::UndefinedOp {
                name: op_name.clone(),
                span,
            })?;

            if op_def.params.len() != 2 {
                return Err(EvalError::ArityMismatch {
                    op: op_name,
                    expected: 2,
                    got: op_def.params.len(),
                    span,
                });
            }

            // Fold over the sequence elements (left to right)
            let mut result = base;
            for elem in seq.iter() {
                let new_ctx = ctx
                    .bind(op_def.params[0].name.node.as_str(), result)
                    .bind(op_def.params[1].name.node.as_str(), elem.clone());
                result = eval(&new_ctx, &op_def.body)?;
            }

            Ok(Some(result))
        }

        // === Bags module ===
        "EmptyBag" => {
            // EmptyBag - the empty bag (empty function)
            check_arity(name, args, 0, span)?;
            Ok(Some(Value::Func(FuncValue::from_sorted_entries(vec![]))))
        }

        "IsABag" => {
            // IsABag(B) - check if B is a valid bag (function with positive integer values)
            // In TLA+, a bag is a function from elements to positive integers (counts)
            // Tuples/Sequences are also functions (from 1..n to values), so they can be bags too
            check_arity(name, args, 1, span)?;
            let bv = eval(ctx, &args[0])?;

            // Handle Value::Func
            if let Some(func) = bv.as_func() {
                // Check all values are positive integers
                for v in func.mapping_values() {
                    match v.as_int() {
                        Some(n) if n > BigInt::zero() => continue,
                        _ => return Ok(Some(Value::Bool(false))),
                    }
                }
                return Ok(Some(Value::Bool(true)));
            }

            // Handle Value::Tuple and Value::Seq
            // These are functions from 1..n to values
            if let Some(seq) = bv.as_seq() {
                // Check all values are positive integers
                for v in seq.iter() {
                    match v.as_int() {
                        Some(n) if n > BigInt::zero() => continue,
                        _ => return Ok(Some(Value::Bool(false))),
                    }
                }
                return Ok(Some(Value::Bool(true)));
            }

            // Handle Value::IntFunc (integer-keyed function)
            if let Value::IntFunc(int_func) = &bv {
                for v in int_func.values.iter() {
                    match v.as_int() {
                        Some(n) if n > BigInt::zero() => continue,
                        _ => return Ok(Some(Value::Bool(false))),
                    }
                }
                return Ok(Some(Value::Bool(true)));
            }

            // Handle Value::Record (records are functions from string keys to values)
            if let Some(rec) = bv.as_record() {
                for v in rec.values() {
                    match v.as_int() {
                        Some(n) if n > BigInt::zero() => continue,
                        _ => return Ok(Some(Value::Bool(false))),
                    }
                }
                return Ok(Some(Value::Bool(true)));
            }

            // Not a function-like value
            Ok(Some(Value::Bool(false)))
        }

        "SetToBag" => {
            // SetToBag(S) - convert set to bag (each element has count 1)
            check_arity(name, args, 1, span)?;
            let sv = eval(ctx, &args[0])?;
            let set_iter = sv
                .iter_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv, Some(args[0].span)))?;
            let mut entries: Vec<(Value, Value)> = Vec::new();
            for elem in set_iter {
                entries.push((elem.clone(), Value::SmallInt(1)));
            }
            // Sort entries - iteration order may not be sorted
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            Ok(Some(Value::Func(FuncValue::from_sorted_entries(entries))))
        }

        "BagToSet" => {
            // BagToSet(B) - get the underlying set (domain of bag)
            check_arity(name, args, 1, span)?;
            let bv = eval(ctx, &args[0])?;
            let func = bv
                .to_func_coerced()
                .ok_or_else(|| EvalError::type_error("Bag/Function", &bv, Some(args[0].span)))?;
            Ok(Some(Value::Set(func.domain_as_sorted_set())))
        }

        "BagIn" => {
            // BagIn(e, B) - is e in bag B with count > 0
            check_arity(name, args, 2, span)?;
            let ev = eval(ctx, &args[0])?;
            let bv = eval(ctx, &args[1])?;
            let func = bv
                .to_func_coerced()
                .ok_or_else(|| EvalError::type_error("Bag/Function", &bv, Some(args[1].span)))?;
            match func.mapping_get(&ev) {
                Some(count) => {
                    let n = count
                        .as_int()
                        .ok_or_else(|| EvalError::type_error("Int", count, span))?;
                    Ok(Some(Value::Bool(n > BigInt::zero())))
                }
                None => Ok(Some(Value::Bool(false))),
            }
        }

        "CopiesIn" => {
            // CopiesIn(e, B) - count of e in bag B (0 if not present)
            check_arity(name, args, 2, span)?;
            let ev = eval(ctx, &args[0])?;
            let bv = eval(ctx, &args[1])?;
            let func = bv
                .to_func_coerced()
                .ok_or_else(|| EvalError::type_error("Bag/Function", &bv, Some(args[1].span)))?;
            match func.mapping_get(&ev) {
                Some(count) => Ok(Some(count.clone())),
                None => Ok(Some(Value::SmallInt(0))),
            }
        }

        "BagCardinality" => {
            // BagCardinality(B) - total count of all elements
            check_arity(name, args, 1, span)?;
            let bv = eval(ctx, &args[0])?;
            let func = bv
                .to_func_coerced()
                .ok_or_else(|| EvalError::type_error("Bag/Function", &bv, Some(args[0].span)))?;
            let mut total = BigInt::zero();
            for v in func.mapping_values() {
                let n = v
                    .as_int()
                    .ok_or_else(|| EvalError::type_error("Int", v, span))?;
                if n <= BigInt::zero() {
                    return Err(EvalError::Internal {
                        message: "BagCardinality expects a bag (positive integer counts)".into(),
                        span,
                    });
                }
                total += n;
            }
            Ok(Some(Value::big_int(total)))
        }

        "BagCup" | "\\oplus" => {
            // BagCup(B1, B2) or B1 (+) B2 - bag union (add counts)
            check_arity(name, args, 2, span)?;
            let bv1 = eval(ctx, &args[0])?;
            let bv2 = eval(ctx, &args[1])?;
            let func1 = bv1
                .to_func_coerced()
                .ok_or_else(|| EvalError::type_error("Bag/Function", &bv1, Some(args[0].span)))?;
            let func2 = bv2
                .to_func_coerced()
                .ok_or_else(|| EvalError::type_error("Bag/Function", &bv2, Some(args[1].span)))?;

            // Build new mapping: combine counts
            let mut entries: Vec<(Value, Value)> = func1.entries().to_vec();
            // Value's interior mutability (fingerprint cache) doesn't affect Hash/Eq
            #[allow(clippy::mutable_key_type)]
            let mut entries_map: std::collections::HashMap<Value, usize> = entries
                .iter()
                .enumerate()
                .map(|(i, (k, _))| (k.clone(), i))
                .collect();

            for (key, val2) in func2.mapping_iter() {
                let n2 = val2
                    .as_int()
                    .ok_or_else(|| EvalError::type_error("Int", val2, span))?;
                if let Some(&idx) = entries_map.get(key) {
                    let n1 = entries[idx]
                        .1
                        .as_int()
                        .ok_or_else(|| EvalError::type_error("Int", &entries[idx].1, span))?;
                    entries[idx].1 = Value::big_int(n1 + n2);
                } else {
                    let idx = entries.len();
                    entries.push((key.clone(), Value::big_int(n2.clone())));
                    entries_map.insert(key.clone(), idx);
                }
            }
            // Sort by key
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            Ok(Some(Value::Func(FuncValue::from_sorted_entries(entries))))
        }

        "BagDiff" | "\\ominus" => {
            // BagDiff(B1, B2) or B1 (-) B2 - bag difference (subtract counts)
            check_arity(name, args, 2, span)?;
            let bv1 = eval(ctx, &args[0])?;
            let bv2 = eval(ctx, &args[1])?;
            let func1 = bv1
                .to_func_coerced()
                .ok_or_else(|| EvalError::type_error("Bag/Function", &bv1, Some(args[0].span)))?;
            let func2 = bv2
                .to_func_coerced()
                .ok_or_else(|| EvalError::type_error("Bag/Function", &bv2, Some(args[1].span)))?;

            let mut entries: Vec<(Value, Value)> = Vec::new();

            for (key, val1) in func1.mapping_iter() {
                let n1 = val1
                    .as_int()
                    .ok_or_else(|| EvalError::type_error("Int", val1, span))?;
                let diff = match func2.mapping_get(key) {
                    Some(val2) => {
                        let n2 = val2
                            .as_int()
                            .ok_or_else(|| EvalError::type_error("Int", val2, span))?;
                        n1 - n2
                    }
                    None => n1.clone(),
                };
                if diff > BigInt::zero() {
                    entries.push((key.clone(), Value::big_int(diff)));
                }
            }
            Ok(Some(Value::Func(FuncValue::from_sorted_entries(entries))))
        }

        "BagUnion" => {
            // BagUnion(S) - bag union of all elements in set S of bags
            check_arity(name, args, 1, span)?;
            let sv = eval(ctx, &args[0])?;
            let set_iter = sv
                .iter_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv, Some(args[0].span)))?;

            // Value's interior mutability (fingerprint cache) doesn't affect Hash/Eq
            #[allow(clippy::mutable_key_type)]
            let mut counts: std::collections::HashMap<Value, BigInt> =
                std::collections::HashMap::new();

            for bag in set_iter {
                let func = bag.to_func_coerced().ok_or_else(|| {
                    EvalError::type_error("Bag/Function", &bag, Some(args[0].span))
                })?;
                for (key, val) in func.mapping_iter() {
                    let n = val
                        .as_int()
                        .ok_or_else(|| EvalError::type_error("Int", val, span))?;
                    if n <= BigInt::zero() {
                        return Err(EvalError::Internal {
                            message: "BagUnion expects a set of bags (positive integer counts)"
                                .into(),
                            span,
                        });
                    }
                    *counts.entry(key.clone()).or_insert_with(BigInt::zero) += n;
                }
            }

            let mut entries: Vec<(Value, Value)> = counts
                .into_iter()
                .map(|(k, v)| (k, Value::big_int(v)))
                .collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            Ok(Some(Value::Func(FuncValue::from_sorted_entries(entries))))
        }

        "BagOfAll" => {
            // BagOfAll(F(_), B) - map bag elements through unary operator, preserving counts
            check_arity(name, args, 2, span)?;

            fn eval_unary_bag_map(
                ctx: &EvalCtx,
                op: &Spanned<Expr>,
                arg_value: Value,
                span: Option<Span>,
            ) -> EvalResult<Value> {
                match &op.node {
                    Expr::Lambda(params, body) => {
                        if params.len() != 1 {
                            return Err(EvalError::ArityMismatch {
                                op: "<lambda>".into(),
                                expected: 1,
                                got: params.len(),
                                span,
                            });
                        }
                        let param_name = params[0].node.clone();
                        let new_ctx = ctx.bind(param_name, arg_value);
                        eval(&new_ctx, body)
                    }
                    Expr::Ident(_) => {
                        // Apply by name/closure via Expr::Apply with a temporary bound variable.
                        let tmp = "__tla2_bagofall_arg".to_string();
                        let call = Spanned::new(
                            Expr::Apply(
                                Box::new(op.clone()),
                                vec![Spanned::new(Expr::Ident(tmp.clone()), op.span)],
                            ),
                            op.span,
                        );
                        let call_ctx = ctx.bind(tmp, arg_value);
                        eval(&call_ctx, &call)
                    }
                    Expr::ModuleRef(instance, op_name, existing_args) => {
                        if !existing_args.is_empty() {
                            return Err(EvalError::Internal {
                                message:
                                    "BagOfAll expects an operator reference for its first argument"
                                        .into(),
                                span: Some(op.span),
                            });
                        }
                        let tmp = "__tla2_bagofall_arg".to_string();
                        let call = Spanned::new(
                            Expr::ModuleRef(
                                instance.clone(),
                                op_name.clone(),
                                vec![Spanned::new(Expr::Ident(tmp.clone()), op.span)],
                            ),
                            op.span,
                        );
                        let call_ctx = ctx.bind(tmp, arg_value);
                        eval(&call_ctx, &call)
                    }
                    _ => Err(EvalError::Internal {
                        message: "BagOfAll expects an operator (name/module ref) or LAMBDA".into(),
                        span: Some(op.span),
                    }),
                }
            }

            let bv = eval(ctx, &args[1])?;
            let func = bv
                .to_func_coerced()
                .ok_or_else(|| EvalError::type_error("Bag/Function", &bv, Some(args[1].span)))?;

            // Value's interior mutability (fingerprint cache) doesn't affect Hash/Eq
            #[allow(clippy::mutable_key_type)]
            let mut counts: std::collections::HashMap<Value, BigInt> =
                std::collections::HashMap::new();

            for (key, val) in func.mapping_iter() {
                let n = val
                    .as_int()
                    .ok_or_else(|| EvalError::type_error("Int", val, span))?;
                if n <= BigInt::zero() {
                    return Err(EvalError::Internal {
                        message: "BagOfAll expects a bag (positive integer counts)".into(),
                        span,
                    });
                }

                let mapped = eval_unary_bag_map(ctx, &args[0], key.clone(), span)?;

                *counts.entry(mapped).or_insert_with(BigInt::zero) += n;
            }

            let mut entries: Vec<(Value, Value)> = counts
                .into_iter()
                .map(|(k, v)| (k, Value::big_int(v)))
                .collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            Ok(Some(Value::Func(FuncValue::from_sorted_entries(entries))))
        }

        "SubBag" => {
            // SubBag(B) - set of all subbags of bag B
            check_arity(name, args, 1, span)?;
            let bv = eval(ctx, &args[0])?;
            let func = bv
                .to_func_coerced()
                .ok_or_else(|| EvalError::type_error("Bag/Function", &bv, Some(args[0].span)))?;

            // Enumerate all choices of counts (0..=B[e]) for each element e.
            // This is exponential in |DOMAIN B| and can be enormous; TLC also
            // enumerates for finite bags.
            let mut elems: Vec<(Value, u64)> = Vec::new();
            for (key, val) in func.mapping_iter() {
                let n = val
                    .as_int()
                    .ok_or_else(|| EvalError::type_error("Int", val, span))?;
                if n <= BigInt::zero() {
                    return Err(EvalError::Internal {
                        message: "SubBag expects a bag (positive integer counts)".into(),
                        span,
                    });
                }
                let max = n.to_u64().ok_or_else(|| EvalError::Internal {
                    message: "SubBag count too large to enumerate".into(),
                    span,
                })?;
                elems.push((key.clone(), max));
            }

            fn enumerate_subbags(
                elems: &[(Value, u64)],
                idx: usize,
                entries: Vec<(Value, Value)>,
                out: &mut OrdSet<Value>,
            ) {
                if idx == elems.len() {
                    out.insert(Value::Func(FuncValue::from_sorted_entries(entries)));
                    return;
                }

                let (elem, max) = &elems[idx];

                // Count 0: element not present
                enumerate_subbags(elems, idx + 1, entries.clone(), out);

                // Counts 1..=max: element present with chosen multiplicity
                for c in 1..=*max {
                    let mut next_entries = entries.clone();
                    next_entries.push((elem.clone(), Value::SmallInt(c as i64)));
                    enumerate_subbags(elems, idx + 1, next_entries, out);
                }
            }

            let mut out = OrdSet::new();
            enumerate_subbags(&elems, 0, Vec::new(), &mut out);
            Ok(Some(Value::Set(SortedSet::from_ord_set(&out))))
        }

        "SqSubseteq" | "\\sqsubseteq" => {
            // SqSubseteq(B1, B2) or B1 \sqsubseteq B2 - bag subset
            // All counts in B1 must be <= corresponding counts in B2
            check_arity(name, args, 2, span)?;
            let bv1 = eval(ctx, &args[0])?;
            let bv2 = eval(ctx, &args[1])?;
            let func1 = bv1
                .to_func_coerced()
                .ok_or_else(|| EvalError::type_error("Bag/Function", &bv1, Some(args[0].span)))?;
            let func2 = bv2
                .to_func_coerced()
                .ok_or_else(|| EvalError::type_error("Bag/Function", &bv2, Some(args[1].span)))?;

            for (key, val1) in func1.mapping_iter() {
                let n1 = val1
                    .as_int()
                    .ok_or_else(|| EvalError::type_error("Int", val1, span))?;
                let n2 = match func2.mapping_get(key) {
                    Some(val2) => val2
                        .as_int()
                        .ok_or_else(|| EvalError::type_error("Int", val2, span))?,
                    None => BigInt::zero(),
                };
                if n1 > n2 {
                    return Ok(Some(Value::Bool(false)));
                }
            }
            Ok(Some(Value::Bool(true)))
        }

        // === BagsExt module ===
        "BagAdd" => {
            // BagAdd(B, e) - add 1 to count of e in bag B
            check_arity(name, args, 2, span)?;
            let bv = eval(ctx, &args[0])?;
            let ev = eval(ctx, &args[1])?;
            let func = bv
                .as_func()
                .ok_or_else(|| EvalError::type_error("Bag/Function", &bv, Some(args[0].span)))?;

            // Copy entries and update/add the element count
            let mut entries: Vec<(Value, Value)> = func.entries().to_vec();
            let mut found = false;
            for entry in entries.iter_mut() {
                if entry.0 == ev {
                    let n = entry
                        .1
                        .as_int()
                        .ok_or_else(|| EvalError::type_error("Int", &entry.1, span))?;
                    entry.1 = Value::big_int(n + BigInt::one());
                    found = true;
                    break;
                }
            }
            if !found {
                entries.push((ev, Value::SmallInt(1)));
                entries.sort_by(|a, b| a.0.cmp(&b.0));
            }
            Ok(Some(Value::Func(FuncValue::from_sorted_entries(entries))))
        }

        "BagRemove" => {
            // BagRemove(B, e) - remove 1 from count of e in bag B
            check_arity(name, args, 2, span)?;
            let bv = eval(ctx, &args[0])?;
            let ev = eval(ctx, &args[1])?;
            let func = bv
                .as_func()
                .ok_or_else(|| EvalError::type_error("Bag/Function", &bv, Some(args[0].span)))?;

            // Copy entries and update/remove the element count
            let mut entries: Vec<(Value, Value)> = Vec::new();
            for (key, val) in func.mapping_iter() {
                if *key == ev {
                    let n = val
                        .as_int()
                        .ok_or_else(|| EvalError::type_error("Int", val, span))?;
                    let new_count = n - BigInt::one();
                    if new_count > BigInt::zero() {
                        entries.push((key.clone(), Value::big_int(new_count)));
                    }
                    // else: drop this entry (count becomes 0 or negative)
                } else {
                    entries.push((key.clone(), val.clone()));
                }
            }
            Ok(Some(Value::Func(FuncValue::from_sorted_entries(entries))))
        }

        "BagRemoveAll" => {
            // BagRemoveAll(B, e) - completely remove e from bag B
            check_arity(name, args, 2, span)?;
            let bv = eval(ctx, &args[0])?;
            let ev = eval(ctx, &args[1])?;
            let func = bv
                .as_func()
                .ok_or_else(|| EvalError::type_error("Bag/Function", &bv, Some(args[0].span)))?;

            // Copy entries excluding the element
            let entries: Vec<(Value, Value)> = func
                .mapping_iter()
                .filter(|(k, _)| *k != &ev)
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            Ok(Some(Value::Func(FuncValue::from_sorted_entries(entries))))
        }

        "FoldBag" => {
            // FoldBag(op, base, B) - fold a binary operator over all elements in a bag
            // Each element e with count n appears n times in the fold
            check_arity(name, args, 3, span)?;

            // Get the operator name from the first argument
            let op_name = match &args[0].node {
                Expr::Ident(name) => name.clone(),
                _ => {
                    return Err(EvalError::Internal {
                        message: "FoldBag requires operator name as first argument".into(),
                        span,
                    })
                }
            };

            let base = eval(ctx, &args[1])?;
            let bv = eval(ctx, &args[2])?;
            let func = bv
                .as_func()
                .ok_or_else(|| EvalError::type_error("Bag/Function", &bv, Some(args[2].span)))?;

            // Get the operator definition
            let op_def = ctx.get_op(&op_name).ok_or_else(|| EvalError::UndefinedOp {
                name: op_name.clone(),
                span,
            })?;

            if op_def.params.len() != 2 {
                return Err(EvalError::ArityMismatch {
                    op: op_name,
                    expected: 2,
                    got: op_def.params.len(),
                    span,
                });
            }

            // Fold over the bag elements (each element appears count times)
            let mut result = base;
            for (elem, count_val) in func.mapping_iter() {
                let count = count_val
                    .as_int()
                    .ok_or_else(|| EvalError::type_error("Int", count_val, span))?
                    .to_i64()
                    .unwrap_or(0);
                // Apply the operator count times for this element
                for _ in 0..count {
                    let new_ctx = ctx
                        .bind(op_def.params[0].name.node.as_str(), result)
                        .bind(op_def.params[1].name.node.as_str(), elem.clone());
                    result = eval(&new_ctx, &op_def.body)?;
                }
            }

            Ok(Some(result))
        }

        "SumBag" => {
            // SumBag(B) - sum of element * count for each element in bag
            // SumBag([1 |-> 2, 3 |-> 1]) = 1*2 + 3*1 = 5
            check_arity(name, args, 1, span)?;
            let bv = eval(ctx, &args[0])?;
            let func = bv
                .as_func()
                .ok_or_else(|| EvalError::type_error("Bag/Function", &bv, Some(args[0].span)))?;

            let mut sum = BigInt::zero();
            for (elem, count_val) in func.mapping_iter() {
                let elem_int = elem
                    .as_int()
                    .ok_or_else(|| EvalError::type_error("Int", elem, span))?;
                let count = count_val
                    .as_int()
                    .ok_or_else(|| EvalError::type_error("Int", count_val, span))?;
                sum += elem_int * count;
            }
            Ok(Some(Value::big_int(sum)))
        }

        "ProductBag" => {
            // ProductBag(B) - product of element^count for each element in bag
            // ProductBag([2 |-> 3, 3 |-> 2]) = 2^3 * 3^2 = 8 * 9 = 72
            check_arity(name, args, 1, span)?;
            let bv = eval(ctx, &args[0])?;
            let func = bv
                .as_func()
                .ok_or_else(|| EvalError::type_error("Bag/Function", &bv, Some(args[0].span)))?;

            let mut product = BigInt::one();
            for (elem, count_val) in func.mapping_iter() {
                let elem_int = elem
                    .as_int()
                    .ok_or_else(|| EvalError::type_error("Int", elem, span))?;
                let count = count_val
                    .as_int()
                    .ok_or_else(|| EvalError::type_error("Int", count_val, span))?
                    .to_i64()
                    .unwrap_or(0);
                // Multiply product by elem^count
                for _ in 0..count {
                    product *= &elem_int;
                }
            }
            Ok(Some(Value::big_int(product)))
        }

        // === TransitiveClosure module ===
        "TransitiveClosure" | "Warshall" => {
            // TransitiveClosure(R) - compute the transitive closure of relation R
            // R is a set of pairs <<a, b>> representing edges a -> b
            // Uses Warshall's algorithm (Floyd-Warshall for reachability)
            check_arity(name, args, 1, span)?;
            let rv = eval(ctx, &args[0])?;
            let rel = rv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &rv, Some(args[0].span)))?;

            // Collect all vertices and build edge map
            let mut vertices: Vec<Value> = Vec::new();
            // Allow: Value has interior mutability for lazy evaluation but Eq/Hash don't depend on it
            #[allow(clippy::mutable_key_type)]
            let mut vertex_index: std::collections::HashMap<Value, usize> =
                std::collections::HashMap::new();

            // First pass: collect all vertices from the relation
            for pair in rel.iter() {
                let tuple = pair
                    .as_seq_or_tuple_elements()
                    .ok_or_else(|| EvalError::type_error("Tuple", pair, span))?;
                if tuple.len() != 2 {
                    return Err(EvalError::Internal {
                        message: format!(
                            "TransitiveClosure requires pairs, got tuple of length {}",
                            tuple.len()
                        ),
                        span,
                    });
                }
                for elem in tuple.iter() {
                    if !vertex_index.contains_key(elem) {
                        vertex_index.insert(elem.clone(), vertices.len());
                        vertices.push(elem.clone());
                    }
                }
            }

            let n = vertices.len();
            if n == 0 {
                return Ok(Some(Value::Set(SortedSet::new())));
            }

            // Build adjacency matrix
            let mut matrix = vec![vec![false; n]; n];
            for pair in rel.iter() {
                let tuple = pair.as_seq_or_tuple_elements().unwrap();
                let i = vertex_index[&tuple[0]];
                let j = vertex_index[&tuple[1]];
                matrix[i][j] = true;
            }

            // Warshall's algorithm for transitive closure
            // Note: Index-based loops are clearest for matrix algorithms with cross-references
            #[allow(clippy::needless_range_loop)]
            for k in 0..n {
                for i in 0..n {
                    if matrix[i][k] {
                        for j in 0..n {
                            if matrix[k][j] {
                                matrix[i][j] = true;
                            }
                        }
                    }
                }
            }

            // Build result set from matrix
            let mut result = OrdSet::new();
            for i in 0..n {
                for j in 0..n {
                    if matrix[i][j] {
                        result.insert(Value::Tuple(
                            vec![vertices[i].clone(), vertices[j].clone()].into(),
                        ));
                    }
                }
            }
            Ok(Some(Value::Set(SortedSet::from_ord_set(&result))))
        }

        "ReflexiveTransitiveClosure" => {
            // ReflexiveTransitiveClosure(R, S) - compute R+ (reflexive transitive closure)
            // This is TransitiveClosure(R) union {<<x, x>> : x \in S}
            check_arity(name, args, 2, span)?;
            let rv = eval(ctx, &args[0])?;
            let rel = rv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &rv, Some(args[0].span)))?;
            let sv = eval(ctx, &args[1])?;
            let domain = sv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv, Some(args[1].span)))?;

            // Collect all vertices from both relation and domain set
            let mut vertices: Vec<Value> = Vec::new();
            // Allow: Value has interior mutability for lazy evaluation but Eq/Hash don't depend on it
            #[allow(clippy::mutable_key_type)]
            let mut vertex_index: std::collections::HashMap<Value, usize> =
                std::collections::HashMap::new();

            // Add vertices from domain set first
            for elem in domain.iter() {
                if !vertex_index.contains_key(elem) {
                    vertex_index.insert(elem.clone(), vertices.len());
                    vertices.push(elem.clone());
                }
            }

            // Add vertices from relation
            for pair in rel.iter() {
                let tuple = pair
                    .as_seq_or_tuple_elements()
                    .ok_or_else(|| EvalError::type_error("Tuple", pair, span))?;
                if tuple.len() != 2 {
                    return Err(EvalError::Internal {
                        message: format!(
                            "ReflexiveTransitiveClosure requires pairs, got tuple of length {}",
                            tuple.len()
                        ),
                        span,
                    });
                }
                for elem in tuple.iter() {
                    if !vertex_index.contains_key(elem) {
                        vertex_index.insert(elem.clone(), vertices.len());
                        vertices.push(elem.clone());
                    }
                }
            }

            let n = vertices.len();
            if n == 0 {
                return Ok(Some(Value::Set(SortedSet::new())));
            }

            // Build adjacency matrix with reflexive edges
            let mut matrix = vec![vec![false; n]; n];

            // Add reflexive edges for all vertices in domain
            for elem in domain.iter() {
                if let Some(&i) = vertex_index.get(elem) {
                    matrix[i][i] = true;
                }
            }

            // Add edges from relation
            for pair in rel.iter() {
                let tuple = pair.as_seq_or_tuple_elements().unwrap();
                let i = vertex_index[&tuple[0]];
                let j = vertex_index[&tuple[1]];
                matrix[i][j] = true;
            }

            // Warshall's algorithm for transitive closure
            // Note: Index-based loops are clearest for matrix algorithms with cross-references
            #[allow(clippy::needless_range_loop)]
            for k in 0..n {
                for i in 0..n {
                    if matrix[i][k] {
                        for j in 0..n {
                            if matrix[k][j] {
                                matrix[i][j] = true;
                            }
                        }
                    }
                }
            }

            // Build result set from matrix
            let mut result = OrdSet::new();
            for i in 0..n {
                for j in 0..n {
                    if matrix[i][j] {
                        result.insert(Value::Tuple(
                            vec![vertices[i].clone(), vertices[j].clone()].into(),
                        ));
                    }
                }
            }
            Ok(Some(Value::Set(SortedSet::from_ord_set(&result))))
        }

        "ConnectedNodes" => {
            // ConnectedNodes(R) - the set of all nodes connected by relation R
            // Returns the union of all first and second elements of pairs in R
            check_arity(name, args, 1, span)?;
            let rv = eval(ctx, &args[0])?;
            let rel = rv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &rv, Some(args[0].span)))?;

            let mut nodes = OrdSet::new();
            for pair in rel.iter() {
                let tuple = pair
                    .as_seq_or_tuple_elements()
                    .ok_or_else(|| EvalError::type_error("Tuple", pair, span))?;
                if tuple.len() != 2 {
                    return Err(EvalError::Internal {
                        message: format!(
                            "ConnectedNodes requires pairs, got tuple of length {}",
                            tuple.len()
                        ),
                        span,
                    });
                }
                nodes.insert(tuple[0].clone());
                nodes.insert(tuple[1].clone());
            }
            Ok(Some(Value::Set(SortedSet::from_ord_set(&nodes))))
        }

        // === Randomization module ===

        // RandomSubset(k, S) - return a random k-element subset of S
        // For model checking, we use a deterministic selection based on set elements
        "RandomSubset" => {
            check_arity(name, args, 2, span)?;
            let kv = eval(ctx, &args[0])?;
            let k = kv
                .as_int()
                .ok_or_else(|| EvalError::type_error("Int", &kv, Some(args[0].span)))?
                .to_usize()
                .ok_or_else(|| EvalError::Internal {
                    message: "RandomSubset requires non-negative integer".into(),
                    span,
                })?;
            let sv = eval(ctx, &args[1])?;
            let set = sv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv, Some(args[1].span)))?;

            let n = set.len();
            if k > n {
                return Err(EvalError::Internal {
                    message: format!("RandomSubset: k={} exceeds set cardinality={}", k, n),
                    span,
                });
            }

            // For model checking, use deterministic selection (first k elements)
            // This ensures reproducibility of counterexamples
            let elements: Vec<_> = set.iter().take(k).cloned().collect();
            Ok(Some(Value::set(elements)))
        }

        // RandomSetOfSubsets(k, n, S) - return a set of k random subsets of S
        // Each subset has elements selected with probability n/|S|
        // For model checking, we generate deterministic subsets
        "RandomSetOfSubsets" => {
            check_arity(name, args, 3, span)?;
            let kv = eval(ctx, &args[0])?;
            let k = kv
                .as_int()
                .ok_or_else(|| EvalError::type_error("Int", &kv, Some(args[0].span)))?
                .to_usize()
                .ok_or_else(|| EvalError::Internal {
                    message: "RandomSetOfSubsets requires non-negative k".into(),
                    span,
                })?;
            let nv = eval(ctx, &args[1])?;
            let n = nv
                .as_int()
                .ok_or_else(|| EvalError::type_error("Int", &nv, Some(args[1].span)))?
                .to_usize()
                .ok_or_else(|| EvalError::Internal {
                    message: "RandomSetOfSubsets requires non-negative n".into(),
                    span,
                })?;
            let sv = eval(ctx, &args[2])?;
            let set = sv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv, Some(args[2].span)))?;

            let card = set.len();
            if card == 0 {
                // Empty set: return set containing only the empty set
                return Ok(Some(Value::set(vec![Value::set(vec![])])));
            }

            if n > card {
                return Err(EvalError::Internal {
                    message: format!(
                        "RandomSetOfSubsets: n={} exceeds set cardinality={}",
                        n, card
                    ),
                    span,
                });
            }

            // For model checking, generate k deterministic subsets of varying sizes
            // The i-th subset contains elements at positions i % card, (i+1) % card, ... up to n elements
            let elements: Vec<_> = set.iter().cloned().collect();
            let mut subsets = OrdSet::new();

            for i in 0..k {
                let mut subset_elems = Vec::new();
                // Create subset by taking n elements starting at offset i
                for j in 0..n {
                    let idx = (i + j) % card;
                    subset_elems.push(elements[idx].clone());
                }
                // Deduplicate (in case n > card)
                let subset_set: OrdSet<_> = subset_elems.into_iter().collect();
                subsets.insert(Value::Set(SortedSet::from_ord_set(&subset_set)));
            }

            Ok(Some(Value::Set(SortedSet::from_ord_set(&subsets))))
        }

        // RandomSubsetSet(k, prob_str, S) - like RandomSetOfSubsets but with explicit probability string
        // prob_str is a string representation of probability (e.g., "0.5")
        "RandomSubsetSet" => {
            check_arity(name, args, 3, span)?;
            let kv = eval(ctx, &args[0])?;
            let k = kv
                .as_int()
                .ok_or_else(|| EvalError::type_error("Int", &kv, Some(args[0].span)))?
                .to_usize()
                .ok_or_else(|| EvalError::Internal {
                    message: "RandomSubsetSet requires non-negative k".into(),
                    span,
                })?;
            let prob_sv = eval(ctx, &args[1])?;
            let prob_str = prob_sv
                .as_string()
                .ok_or_else(|| EvalError::type_error("String", &prob_sv, Some(args[1].span)))?;
            let prob: f64 = prob_str.parse().map_err(|_| EvalError::Internal {
                message: format!("RandomSubsetSet: cannot parse probability '{}'", prob_str),
                span,
            })?;
            if !(0.0..=1.0).contains(&prob) {
                return Err(EvalError::Internal {
                    message: format!(
                        "RandomSubsetSet: probability {} must be in range [0, 1]",
                        prob
                    ),
                    span,
                });
            }
            let sv = eval(ctx, &args[2])?;
            let set = sv
                .as_set()
                .ok_or_else(|| EvalError::type_error("Set", &sv, Some(args[2].span)))?;

            let card = set.len();
            if card == 0 {
                return Ok(Some(Value::set(vec![Value::set(vec![])])));
            }

            // Derive n from probability: n = floor(prob * card)
            let n = (prob * card as f64).floor() as usize;

            let elements: Vec<_> = set.iter().cloned().collect();
            let mut subsets = OrdSet::new();

            for i in 0..k {
                let mut subset_elems = Vec::new();
                for j in 0..n {
                    let idx = (i + j) % card;
                    subset_elems.push(elements[idx].clone());
                }
                let subset_set: OrdSet<_> = subset_elems.into_iter().collect();
                subsets.insert(Value::Set(SortedSet::from_ord_set(&subset_set)));
            }

            Ok(Some(Value::Set(SortedSet::from_ord_set(&subsets))))
        }

        // === Json module ===

        // ToJson(value) - convert value to JSON string representation
        "ToJson" => {
            check_arity(name, args, 1, span)?;
            let val = eval(ctx, &args[0])?;
            let json_str = value_to_json(&val);
            Ok(Some(Value::String(json_str.into())))
        }

        // ToJsonArray(value) - convert sequence/tuple to JSON array string
        "ToJsonArray" => {
            check_arity(name, args, 1, span)?;
            let val = eval(ctx, &args[0])?;
            let json_str = value_to_json_array(&val)?;
            Ok(Some(Value::String(json_str.into())))
        }

        // ToJsonObject(value) - convert record/function to JSON object string
        "ToJsonObject" => {
            check_arity(name, args, 1, span)?;
            let val = eval(ctx, &args[0])?;
            let json_str = value_to_json_object(&val)?;
            Ok(Some(Value::String(json_str.into())))
        }

        // JsonSerialize(filename, value) - serialize value to JSON file
        // Returns TRUE on success (for model checking, we just return TRUE)
        "JsonSerialize" => {
            check_arity(name, args, 2, span)?;
            let filename_val = eval(ctx, &args[0])?;
            let _filename = filename_val.as_string().ok_or_else(|| {
                EvalError::type_error("String", &filename_val, Some(args[0].span))
            })?;
            let val = eval(ctx, &args[1])?;
            let _json_str = value_to_json(&val);
            // In model checking mode, we don't actually write to files
            // This would require side effects which aren't appropriate
            // Return TRUE to indicate success
            Ok(Some(Value::Bool(true)))
        }

        // JsonDeserialize(filename) - deserialize JSON from file
        // Returns a TLA+ value (for model checking, returns empty record)
        "JsonDeserialize" => {
            check_arity(name, args, 1, span)?;
            let filename_val = eval(ctx, &args[0])?;
            let _filename = filename_val.as_string().ok_or_else(|| {
                EvalError::type_error("String", &filename_val, Some(args[0].span))
            })?;
            // In model checking mode, we can't actually read files
            // Return empty record as placeholder
            Ok(Some(Value::Record(RecordValue::new())))
        }

        // ndJsonSerialize(filename, value) - serialize to newline-delimited JSON
        "ndJsonSerialize" => {
            check_arity(name, args, 2, span)?;
            let filename_val = eval(ctx, &args[0])?;
            let _filename = filename_val.as_string().ok_or_else(|| {
                EvalError::type_error("String", &filename_val, Some(args[0].span))
            })?;
            let _val = eval(ctx, &args[1])?;
            // Return TRUE to indicate success
            Ok(Some(Value::Bool(true)))
        }

        // ndJsonDeserialize(filename) - deserialize newline-delimited JSON from file
        "ndJsonDeserialize" => {
            check_arity(name, args, 1, span)?;
            let filename_val = eval(ctx, &args[0])?;
            let _filename = filename_val.as_string().ok_or_else(|| {
                EvalError::type_error("String", &filename_val, Some(args[0].span))
            })?;
            // Return empty sequence as placeholder
            Ok(Some(Value::Seq(Vec::new().into())))
        }

        // === DyadicRationals module ===
        // Dyadic rationals are fractions with denominator = 2^n
        // Represented as records [num |-> n, den |-> d]

        // Zero - the zero dyadic rational [num |-> 0, den |-> 1]
        "Zero" if args.is_empty() => Ok(Some(make_dyadic_rational(0, 1))),

        // One - the one dyadic rational [num |-> 1, den |-> 1]
        "One" if args.is_empty() => Ok(Some(make_dyadic_rational(1, 1))),

        // IsDyadicRational(r) - check if r.den is a power of 2
        "IsDyadicRational" => {
            check_arity(name, args, 1, span)?;
            let rv = eval(ctx, &args[0])?;
            // Dyadic rationals are records with num and den fields
            let is_dyadic = if let Some(rec) = rv.as_record() {
                if let (Some(den_val), Some(_num_val)) = (rec.get("den"), rec.get("num")) {
                    if let Some(den) = den_val.as_int() {
                        let den_i64 = den.to_i64().unwrap_or(0);
                        // Check if den is a power of 2 (including 1 = 2^0)
                        den_i64 > 0 && (den_i64 & (den_i64 - 1)) == 0
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            };
            Ok(Some(Value::Bool(is_dyadic)))
        }

        // Add(p, q) - add two dyadic rationals
        "Add" => {
            check_arity(name, args, 2, span)?;
            let pv = eval(ctx, &args[0])?;
            let qv = eval(ctx, &args[1])?;

            // Extract num/den from both records
            let (p_num, p_den) = extract_dyadic(&pv, span)?;
            let (q_num, q_den) = extract_dyadic(&qv, span)?;

            // Short circuit: if p is zero, return q
            if p_num == 0 {
                return Ok(Some(qv));
            }

            // For dyadic rationals, LCM of denominators is just the max
            let lcm = std::cmp::max(p_den, q_den);

            // Scale both fractions to have the same denominator
            let p_scaled = p_num * (lcm / p_den);
            let q_scaled = q_num * (lcm / q_den);

            // Add numerators
            let sum_num = p_scaled + q_scaled;

            // Reduce the fraction
            let (reduced_num, reduced_den) = reduce_fraction(sum_num, lcm);

            Ok(Some(make_dyadic_rational(reduced_num, reduced_den)))
        }

        // Half(p) - divide by 2 (double the denominator)
        "Half" => {
            check_arity(name, args, 1, span)?;
            let pv = eval(ctx, &args[0])?;
            let (num, den) = extract_dyadic(&pv, span)?;

            // Double the denominator and reduce
            let (reduced_num, reduced_den) = reduce_fraction(num, den * 2);

            Ok(Some(make_dyadic_rational(reduced_num, reduced_den)))
        }

        // PrettyPrint(p) - string representation of dyadic rational
        "PrettyPrint" => {
            check_arity(name, args, 1, span)?;
            let pv = eval(ctx, &args[0])?;
            let (num, den) = extract_dyadic(&pv, span)?;

            let s = if num == 0 {
                "0".to_string()
            } else if den == 1 {
                num.to_string()
            } else {
                format!("{}/{}", num, den)
            };

            Ok(Some(Value::String(intern_string(&s))))
        }

        // === TLAPS (TLA+ Proof System) Operators ===
        // All TLAPS operators return TRUE - they are proof backend pragmas
        // that TLC ignores during model checking.

        // Zero-arity TLAPS operators (SMT solvers, provers, tactics)
        "SMT"
        | "CVC3"
        | "Yices"
        | "veriT"
        | "Z3"
        | "Spass"
        | "SimpleArithmetic"
        | "Zenon"
        | "SlowZenon"
        | "SlowerZenon"
        | "VerySlowZenon"
        | "SlowestZenon"
        | "Isa"
        | "Auto"
        | "Force"
        | "Blast"
        | "SimplifyAndSolve"
        | "Simplification"
        | "AutoBlast"
        | "LS4"
        | "PTL"
        | "PropositionalTemporalLogic"
        | "AllProvers"
        | "AllSMT"
        | "AllIsa"
        | "SetExtensionality"
        | "NoSetContainsEverything"
        | "IsaWithSetExtensionality" => {
            if args.is_empty() {
                return Ok(Some(Value::Bool(true)));
            }
            // If called with args, fall through to Ok(None) for error handling
            Ok(None)
        }

        // Parameterized TLAPS operators - take 1 arg (timeout or tactic), return TRUE
        "SMTT" | "CVC3T" | "YicesT" | "veriTT" | "Z3T" | "SpassT" | "ZenonT" | "IsaT" | "IsaM"
        | "AllProversT" | "AllSMTT" | "AllIsaT" => {
            if args.len() == 1 {
                // Evaluate arg for side effects, but ignore the value
                let _ = eval(ctx, &args[0])?;
                return Ok(Some(Value::Bool(true)));
            }
            // Wrong arity - fall through for error
            Ok(None)
        }

        // IsaMT takes 2 args (tactic, timeout)
        "IsaMT" => {
            if args.len() == 2 {
                let _ = eval(ctx, &args[0])?;
                let _ = eval(ctx, &args[1])?;
                return Ok(Some(Value::Bool(true)));
            }
            Ok(None)
        }

        // === Graphs Module ===
        // These operators work with graph records [node |-> Set, edge |-> Set of pairs]

        // IsDirectedGraph(G) - check if G is a valid directed graph record
        // G must be a record with 'node' and 'edge' fields where edge ⊆ node × node
        "IsDirectedGraph" => {
            check_arity(name, args, 1, span)?;
            let gv = eval(ctx, &args[0])?;

            // Check if it's a record with 'node' and 'edge' fields
            let func = match gv.to_func_coerced() {
                Some(f) => f,
                None => return Ok(Some(Value::Bool(false))),
            };

            // Get node and edge fields
            let node_key = Value::String(intern_string("node"));
            let edge_key = Value::String(intern_string("edge"));

            let nodes = match func.apply(&node_key) {
                Some(n) => n.clone(),
                None => return Ok(Some(Value::Bool(false))),
            };

            let edges = match func.apply(&edge_key) {
                Some(e) => e.clone(),
                None => return Ok(Some(Value::Bool(false))),
            };

            // Verify record has exactly {node, edge} keys
            if func.domain_len() != 2 {
                return Ok(Some(Value::Bool(false)));
            }

            // Check nodes is a set
            let node_set = match nodes.as_set() {
                Some(s) => s,
                None => return Ok(Some(Value::Bool(false))),
            };

            // Check edges is a set
            let edge_set = match edges.as_set() {
                Some(s) => s,
                None => return Ok(Some(Value::Bool(false))),
            };

            // Check edge ⊆ node × node (each edge is a pair of nodes)
            for edge in edge_set.iter() {
                // Each edge must be a tuple <<n1, n2>>
                if let Value::Tuple(pair) = edge {
                    if pair.len() != 2 {
                        return Ok(Some(Value::Bool(false)));
                    }
                    // Both elements must be in nodes
                    if !node_set.contains(&pair[0]) || !node_set.contains(&pair[1]) {
                        return Ok(Some(Value::Bool(false)));
                    }
                } else {
                    return Ok(Some(Value::Bool(false)));
                }
            }

            Ok(Some(Value::Bool(true)))
        }

        // IsUndirectedGraph(G) - check if G is a valid undirected graph
        // UndirectedGraphs module uses set-based edges {a,b}
        // G must be [node |-> Set, edge |-> Set of two-element sets]
        "IsUndirectedGraph" => {
            check_arity(name, args, 1, span)?;
            let gv = eval(ctx, &args[0])?;

            let func = match gv.to_func_coerced() {
                Some(f) => f,
                None => return Ok(Some(Value::Bool(false))),
            };

            let node_key = Value::String(intern_string("node"));
            let edge_key = Value::String(intern_string("edge"));

            let nodes = match func.apply(&node_key) {
                Some(n) => n.clone(),
                None => return Ok(Some(Value::Bool(false))),
            };

            let edges = match func.apply(&edge_key) {
                Some(e) => e.clone(),
                None => return Ok(Some(Value::Bool(false))),
            };

            if func.domain_len() != 2 {
                return Ok(Some(Value::Bool(false)));
            }

            // Check both are set-like values
            if !nodes.is_set() || !edges.is_set() {
                return Ok(Some(Value::Bool(false)));
            }

            // Iterate over edges - use iter_set() which handles intervals, etc.
            let edge_iter = match edges.iter_set() {
                Some(iter) => iter,
                None => return Ok(Some(Value::Bool(false))),
            };

            // Each edge must be a 2-element set {a,b} where a,b are nodes
            for edge in edge_iter {
                // Edge should be a set (in UndirectedGraphs)
                if let Some(edge_inner) = edge.to_sorted_set() {
                    // Must be exactly 2 elements
                    if edge_inner.len() != 2 {
                        return Ok(Some(Value::Bool(false)));
                    }
                    // Both elements must be in nodes
                    for elem in edge_inner.iter() {
                        match nodes.set_contains(elem) {
                            Some(true) => {}
                            _ => return Ok(Some(Value::Bool(false))),
                        }
                    }
                } else {
                    return Ok(Some(Value::Bool(false)));
                }
            }

            Ok(Some(Value::Bool(true)))
        }

        // IsStronglyConnected(G) - check if graph has exactly one connected component
        // For undirected graphs: all nodes are reachable from each other
        "IsStronglyConnected" => {
            check_arity(name, args, 1, span)?;
            let gv = eval(ctx, &args[0])?;

            let func = match gv.to_func_coerced() {
                Some(f) => f,
                None => return Ok(Some(Value::Bool(false))),
            };

            let node_key = Value::String(intern_string("node"));
            let edge_key = Value::String(intern_string("edge"));

            let nodes = match func.apply(&node_key) {
                Some(n) => n.clone(),
                None => return Ok(Some(Value::Bool(false))),
            };

            let edges = match func.apply(&edge_key) {
                Some(e) => e.clone(),
                None => return Ok(Some(Value::Bool(false))),
            };

            // Convert nodes to SortedSet (handles intervals, etc.)
            let node_set = match nodes.to_sorted_set() {
                Some(s) => s,
                None => return Ok(Some(Value::Bool(false))),
            };

            // Get edge iterator (handles intervals, etc.)
            let edge_iter = match edges.iter_set() {
                Some(iter) => iter,
                None => return Ok(Some(Value::Bool(false))),
            };

            // Empty graph is trivially connected
            if node_set.is_empty() {
                return Ok(Some(Value::Bool(true)));
            }

            // Build adjacency list
            let nodes_vec: Vec<Value> = node_set.iter().cloned().collect();
            let mut visited = vec![false; nodes_vec.len()];
            // Allow: Value contains OnceLock for fingerprint caching, but Hash/Eq are stable
            #[allow(clippy::mutable_key_type)]
            let mut node_to_idx: std::collections::HashMap<&Value, usize> =
                std::collections::HashMap::new();
            for (i, n) in nodes_vec.iter().enumerate() {
                node_to_idx.insert(n, i);
            }

            // Build adjacency from edges (handles both set {a,b} and tuple <<a,b>>)
            let mut adj: Vec<Vec<usize>> = vec![vec![]; nodes_vec.len()];
            for edge in edge_iter {
                if let Some(edge_inner) = edge.to_sorted_set() {
                    // Set-based edge {a,b}
                    let elems: Vec<_> = edge_inner.iter().collect();
                    if elems.len() == 2 {
                        if let (Some(&i), Some(&j)) =
                            (node_to_idx.get(elems[0]), node_to_idx.get(elems[1]))
                        {
                            adj[i].push(j);
                            adj[j].push(i);
                        }
                    }
                } else if let Value::Tuple(pair) = &edge {
                    // Tuple-based edge <<a,b>>
                    if pair.len() == 2 {
                        if let (Some(&i), Some(&j)) =
                            (node_to_idx.get(&pair[0]), node_to_idx.get(&pair[1]))
                        {
                            adj[i].push(j);
                            adj[j].push(i);
                        }
                    }
                }
            }

            // BFS from first node
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(0);
            visited[0] = true;

            while let Some(curr) = queue.pop_front() {
                for &neighbor in &adj[curr] {
                    if !visited[neighbor] {
                        visited[neighbor] = true;
                        queue.push_back(neighbor);
                    }
                }
            }

            // Check all nodes visited
            Ok(Some(Value::Bool(visited.iter().all(|&v| v))))
        }

        // === Bitwise Module (CommunityModules) ===

        // ^^ - bitwise XOR (infix)
        "^^" => {
            check_arity(name, args, 2, span)?;
            let a = eval(ctx, &args[0])?;
            let av = a
                .as_int()
                .ok_or_else(|| EvalError::type_error("Int", &a, Some(args[0].span)))?;
            let b = eval(ctx, &args[1])?;
            let bv = b
                .as_int()
                .ok_or_else(|| EvalError::type_error("Int", &b, Some(args[1].span)))?;
            // Use signed BigInt XOR which performs bitwise XOR
            // For non-negative integers, this gives the standard result
            Ok(Some(Value::big_int(av ^ bv)))
        }

        // & - bitwise AND (infix)
        "&" => {
            check_arity(name, args, 2, span)?;
            let a = eval(ctx, &args[0])?;
            let av = a
                .as_int()
                .ok_or_else(|| EvalError::type_error("Int", &a, Some(args[0].span)))?;
            let b = eval(ctx, &args[1])?;
            let bv = b
                .as_int()
                .ok_or_else(|| EvalError::type_error("Int", &b, Some(args[1].span)))?;
            Ok(Some(Value::big_int(av & bv)))
        }

        // | - bitwise OR (infix)
        "|" => {
            check_arity(name, args, 2, span)?;
            let a = eval(ctx, &args[0])?;
            let av = a
                .as_int()
                .ok_or_else(|| EvalError::type_error("Int", &a, Some(args[0].span)))?;
            let b = eval(ctx, &args[1])?;
            let bv = b
                .as_int()
                .ok_or_else(|| EvalError::type_error("Int", &b, Some(args[1].span)))?;
            Ok(Some(Value::big_int(av | bv)))
        }

        // Not(a) - bitwise NOT
        // Note: BigInt NOT is -(a+1) which gives two's complement behavior.
        // For TLA+ Bitwise module compatibility, we compute !a for non-negative a
        // as a pattern that inverts all set bits up to the highest bit position.
        // However, the standard TLA+ Bitwise module Not operation returns -(a+1).
        "Not" => {
            check_arity(name, args, 1, span)?;
            let a = eval(ctx, &args[0])?;
            let av = a
                .as_int()
                .ok_or_else(|| EvalError::type_error("Int", &a, Some(args[0].span)))?;
            // BigInt bitwise NOT: !a = -(a + 1) (two's complement)
            Ok(Some(Value::big_int(!av)))
        }

        // shiftR(n, pos) - logical right shift
        "shiftR" => {
            check_arity(name, args, 2, span)?;
            let n = eval(ctx, &args[0])?;
            let nv = n
                .as_int()
                .ok_or_else(|| EvalError::type_error("Int", &n, Some(args[0].span)))?;
            let pos = eval(ctx, &args[1])?;
            let posv = pos
                .as_int()
                .ok_or_else(|| EvalError::type_error("Int", &pos, Some(args[1].span)))?;
            // Convert pos to u64 for shift (TLA+ should only use reasonable shift amounts)
            use num_traits::ToPrimitive;
            let shift = posv.to_u64().ok_or_else(|| EvalError::Internal {
                message: format!("shiftR position {} is too large", posv),
                span,
            })?;
            Ok(Some(Value::big_int(nv >> shift)))
        }

        // Not a built-in
        _ => Ok(None),
    }
}

fn check_arity(
    name: &str,
    args: &[Spanned<Expr>],
    expected: usize,
    span: Option<Span>,
) -> EvalResult<()> {
    if args.len() != expected {
        Err(EvalError::ArityMismatch {
            op: name.into(),
            expected,
            got: args.len(),
            span,
        })
    } else {
        Ok(())
    }
}

/// Generate all permutation functions on a set (used by TLC!Permutations for symmetry)
/// Each permutation is a bijective function [S -> S]
fn generate_permutation_functions(
    domain: &[Value],       // Original domain elements (sorted order)
    range_prefix: &[usize], // Current permutation indices being built
    result: &mut OrdSet<Value>,
) {
    let n = domain.len();

    if range_prefix.len() == n {
        // Complete permutation: build the function from sorted entries
        // domain is sorted, so entries will be in sorted order
        let entries: Vec<(Value, Value)> = range_prefix
            .iter()
            .enumerate()
            .map(|(i, &j)| (domain[i].clone(), domain[j].clone()))
            .collect();
        result.insert(Value::Func(FuncValue::from_sorted_entries(entries)));
        return;
    }

    // Try each unused index for the next position
    for j in 0..n {
        if !range_prefix.contains(&j) {
            let mut new_prefix = range_prefix.to_vec();
            new_prefix.push(j);
            generate_permutation_functions(domain, &new_prefix, result);
        }
    }
}

/// Bind a bound variable to an element, handling tuple pattern destructuring
fn bind_bound_var(
    ctx: &EvalCtx,
    bound: &BoundVar,
    elem: &Value,
    span: Option<Span>,
) -> EvalResult<EvalCtx> {
    match &bound.pattern {
        Some(BoundPattern::Tuple(vars)) => {
            // Destructure tuple and bind each variable
            let tuple = elem
                .as_tuple()
                .ok_or_else(|| EvalError::type_error("Tuple", elem, span))?;
            if tuple.len() != vars.len() {
                return Err(EvalError::Internal {
                    message: format!(
                        "Tuple pattern has {} variables but element has {} components",
                        vars.len(),
                        tuple.len()
                    ),
                    span,
                });
            }
            let mut new_ctx = ctx.clone();
            for (var, val) in vars.iter().zip(tuple.iter()) {
                new_ctx = new_ctx.bind(var.node.clone(), val.clone());
            }
            Ok(new_ctx)
        }
        Some(BoundPattern::Var(var)) => Ok(ctx.bind(var.node.clone(), elem.clone())),
        None => Ok(ctx.bind(bound.name.node.clone(), elem.clone())),
    }
}

/// Bind a bound variable to an element using bind_local (always adds to local_stack).
/// This is used for LazyFunc parameters to ensure dependency tracking works correctly.
/// Issue #100: Without using bind_local, parameters bound only in env are not tracked
/// as dependencies by eval_with_dep_tracking, causing ZERO_ARG_OP_CACHE to return
/// stale values when the bound variable changes across recursive calls.
fn bind_local_bound_var(
    ctx: &EvalCtx,
    bound: &BoundVar,
    elem: &Value,
    span: Option<Span>,
) -> EvalResult<EvalCtx> {
    match &bound.pattern {
        Some(BoundPattern::Tuple(vars)) => {
            // Destructure tuple and bind each variable
            let tuple = elem
                .as_tuple()
                .ok_or_else(|| EvalError::type_error("Tuple", elem, span))?;
            if tuple.len() != vars.len() {
                return Err(EvalError::Internal {
                    message: format!(
                        "Tuple pattern has {} variables but element has {} components",
                        vars.len(),
                        tuple.len()
                    ),
                    span,
                });
            }
            let mut new_ctx = ctx.clone();
            for (var, val) in vars.iter().zip(tuple.iter()) {
                new_ctx = new_ctx.bind_local(var.node.clone(), val.clone());
            }
            Ok(new_ctx)
        }
        Some(BoundPattern::Var(var)) => Ok(ctx.bind_local(var.node.clone(), elem.clone())),
        None => Ok(ctx.bind_local(bound.name.node.clone(), elem.clone())),
    }
}

/// Push a bound variable binding onto a mutable context's stack.
/// Use with `ctx.mark_stack()` before and `ctx.pop_to_mark(mark)` after the body evaluation.
/// This avoids allocating a new EvalCtx per element in hot loops.
fn push_bound_var_mut(
    ctx: &mut EvalCtx,
    bound: &BoundVar,
    elem: &Value,
    span: Option<Span>,
) -> EvalResult<()> {
    match &bound.pattern {
        Some(BoundPattern::Tuple(vars)) => {
            // Destructure tuple and bind each variable
            let tuple = elem
                .as_tuple()
                .ok_or_else(|| EvalError::type_error("Tuple", elem, span))?;
            if tuple.len() != vars.len() {
                return Err(EvalError::Internal {
                    message: format!(
                        "Tuple pattern has {} variables but element has {} components",
                        vars.len(),
                        tuple.len()
                    ),
                    span,
                });
            }
            for (var, val) in vars.iter().zip(tuple.iter()) {
                let name: Arc<str> = Arc::from(var.node.as_str());
                ctx.push_binding(Arc::clone(&name), val.clone());
                ctx.bind_mut(name, val.clone());
            }
            Ok(())
        }
        Some(BoundPattern::Var(var)) => {
            let name: Arc<str> = Arc::from(var.node.as_str());
            ctx.push_binding(Arc::clone(&name), elem.clone());
            ctx.bind_mut(name, elem.clone());
            Ok(())
        }
        None => {
            let name: Arc<str> = Arc::from(bound.name.node.as_str());
            ctx.push_binding(Arc::clone(&name), elem.clone());
            ctx.bind_mut(name, elem.clone());
            Ok(())
        }
    }
}

/// Evaluate \A bounds : body
fn eval_forall(
    ctx: &EvalCtx,
    bounds: &[BoundVar],
    body: &Spanned<Expr>,
    span: Option<Span>,
) -> EvalResult<Value> {
    if bounds.is_empty() {
        return eval(ctx, body);
    }

    let first = &bounds[0];
    let domain = first.domain.as_ref().ok_or_else(|| EvalError::Internal {
        message: "Forall requires bounded quantification".into(),
        span,
    })?;

    let dv = eval(ctx, domain)?;
    let iter = dv
        .iter_set()
        .ok_or_else(|| EvalError::type_error("Set", &dv, Some(domain.span)))?;

    for elem in iter {
        let new_ctx = bind_bound_var(ctx, first, &elem, span)?;
        let result = eval_forall(&new_ctx, &bounds[1..], body, span)?;
        let b = result
            .as_bool()
            .ok_or_else(|| EvalError::type_error("BOOLEAN", &result, Some(body.span)))?;
        if !b {
            return Ok(Value::Bool(false));
        }
    }

    Ok(Value::Bool(true))
}

/// Evaluate \E bounds : body
fn eval_exists(
    ctx: &EvalCtx,
    bounds: &[BoundVar],
    body: &Spanned<Expr>,
    span: Option<Span>,
) -> EvalResult<Value> {
    if bounds.is_empty() {
        return eval(ctx, body);
    }

    let first = &bounds[0];
    let domain = first.domain.as_ref().ok_or_else(|| EvalError::Internal {
        message: "Exists requires bounded quantification".into(),
        span,
    })?;

    let dv = eval(ctx, domain)?;
    let iter = dv
        .iter_set()
        .ok_or_else(|| EvalError::type_error("Set", &dv, Some(domain.span)))?;

    for elem in iter {
        let new_ctx = bind_bound_var(ctx, first, &elem, span)?;
        let result = eval_exists(&new_ctx, &bounds[1..], body, span)?;
        let b = result
            .as_bool()
            .ok_or_else(|| EvalError::type_error("BOOLEAN", &result, Some(body.span)))?;
        if b {
            return Ok(Value::Bool(true));
        }
    }

    Ok(Value::Bool(false))
}

/// Evaluate CHOOSE x \in S : P
fn eval_choose(
    ctx: &EvalCtx,
    bound: &BoundVar,
    body: &Spanned<Expr>,
    span: Option<Span>,
) -> EvalResult<Value> {
    let domain = bound.domain.as_ref().ok_or_else(|| EvalError::Internal {
        message: "CHOOSE requires bounded quantification".into(),
        span,
    })?;

    let dv = eval(ctx, domain)?;
    let iter = dv
        .iter_set()
        .ok_or_else(|| EvalError::type_error("Set", &dv, Some(domain.span)))?;

    for elem in iter {
        let new_ctx = bind_bound_var(ctx, bound, &elem, span)?;
        let result = eval(&new_ctx, body)?;
        let b = result
            .as_bool()
            .ok_or_else(|| EvalError::type_error("BOOLEAN", &result, Some(body.span)))?;
        if b {
            return Ok(elem);
        }
    }

    Err(EvalError::ChooseFailed { span })
}

/// Evaluate {expr : x \in S, y \in T, ...}
fn eval_set_builder(
    ctx: &EvalCtx,
    expr: &Spanned<Expr>,
    bounds: &[BoundVar],
    span: Option<Span>,
) -> EvalResult<Value> {
    let mut result = OrdSet::new();
    eval_set_builder_rec(ctx, expr, bounds, &mut result, span)?;
    Ok(Value::Set(SortedSet::from_ord_set(&result)))
}

fn eval_set_builder_rec(
    ctx: &EvalCtx,
    expr: &Spanned<Expr>,
    bounds: &[BoundVar],
    result: &mut OrdSet<Value>,
    span: Option<Span>,
) -> EvalResult<()> {
    if bounds.is_empty() {
        result.insert(eval(ctx, expr)?);
        return Ok(());
    }

    let first = &bounds[0];
    let domain = first.domain.as_ref().ok_or_else(|| EvalError::Internal {
        message: "Set builder requires bounded variables".into(),
        span,
    })?;

    let dv = eval(ctx, domain)?;
    let iter = dv
        .iter_set()
        .ok_or_else(|| EvalError::type_error("Set", &dv, Some(domain.span)))?;

    for elem in iter {
        let new_ctx = bind_bound_var(ctx, first, &elem, span)?;
        eval_set_builder_rec(&new_ctx, expr, &bounds[1..], result, span)?;
    }

    Ok(())
}

/// Evaluate {x \in S : P}
fn eval_set_filter(
    ctx: &EvalCtx,
    bound: &BoundVar,
    pred: &Spanned<Expr>,
    span: Option<Span>,
) -> EvalResult<Value> {
    let domain = bound.domain.as_ref().ok_or_else(|| EvalError::Internal {
        message: "Set filter requires bounded variable".into(),
        span,
    })?;

    let dv = eval(ctx, domain)?;

    // If the domain is non-enumerable (STRING, AnySet), return a lazy SetPredValue
    // This allows membership checking without full enumeration
    if dv.iter_set().is_none() {
        // Verify it's actually a set-like type (not just something that can't be iterated)
        if !dv.is_set() {
            return Err(EvalError::type_error("Set", &dv, Some(domain.span)));
        }
        // Return a lazy SetPredValue that can check membership but not enumerate
        return Ok(Value::SetPred(SetPredValue::new(
            dv,
            bound.clone(),
            pred.clone(),
            ctx.env.clone(),
        )));
    }

    // Enumerable domain: evaluate eagerly
    let iter = dv.iter_set().unwrap();

    // Optimization: reuse a single EvalCtx with push/pop instead of cloning per element
    let mut local_ctx = ctx.clone();
    let mark = local_ctx.mark_stack();
    let mut result = SetBuilder::new();
    for elem in iter {
        push_bound_var_mut(&mut local_ctx, bound, &elem, span)?;
        // Evaluate predicate, treating certain errors as false
        // This matches TLC behavior where undefined predicates exclude the element
        let include = match eval(&local_ctx, pred) {
            Ok(pv) => pv
                .as_bool()
                .ok_or_else(|| EvalError::type_error("BOOLEAN", &pv, Some(pred.span)))?,
            Err(EvalError::NotInDomain { .. }) => false,
            Err(EvalError::IndexOutOfBounds { .. }) => false,
            Err(e) => return Err(e),
        };
        if include {
            result.insert(elem);
        }
        local_ctx.pop_to_mark(mark);
    }

    Ok(result.build_value())
}

/// Count elements in a set filter without building the set.
/// Returns None if the domain is not enumerable.
fn count_set_filter_elements(
    ctx: &EvalCtx,
    domain_val: &Value,
    bound: &BoundVar,
    pred: &Spanned<Expr>,
    span: Option<Span>,
) -> EvalResult<Option<usize>> {
    let iter = match domain_val.iter_set() {
        Some(it) => it,
        None => return Ok(None), // Not enumerable
    };

    let mut count: usize = 0;
    let mut local_ctx = ctx.clone();
    let mark = local_ctx.mark_stack();
    for elem in iter {
        push_bound_var_mut(&mut local_ctx, bound, &elem, span)?;
        // Evaluate predicate, treating certain errors as false
        let include = match eval(&local_ctx, pred) {
            Ok(pv) => pv
                .as_bool()
                .ok_or_else(|| EvalError::type_error("BOOLEAN", &pv, Some(pred.span)))?,
            Err(EvalError::NotInDomain { .. }) => false,
            Err(EvalError::IndexOutOfBounds { .. }) => false,
            Err(e) => return Err(e),
        };
        if include {
            count += 1;
        }
        local_ctx.pop_to_mark(mark);
    }
    Ok(Some(count))
}

/// Evaluate [x \in S |-> expr]
fn eval_func_def(
    ctx: &EvalCtx,
    bounds: &[BoundVar],
    body: &Spanned<Expr>,
    span: Option<Span>,
) -> EvalResult<Value> {
    // Single-variable function
    if bounds.len() == 1 {
        let bound = &bounds[0];
        let domain_expr = bound.domain.as_ref().ok_or_else(|| EvalError::Internal {
            message: "Function definition requires bounded variable".into(),
            span,
        })?;

        let dv = eval(ctx, domain_expr)?;

        // Handle special case for Nat/Int/Real lazy functions
        if let Value::ModelValue(name) = &dv {
            return match name.as_ref() {
                "Nat" => Ok(Value::LazyFunc(Box::new(LazyFuncValue::new(
                    None,
                    LazyDomain::Nat,
                    bound.clone(),
                    body.clone(),
                    ctx.env.clone(),
                )))),
                "Int" => Ok(Value::LazyFunc(Box::new(LazyFuncValue::new(
                    None,
                    LazyDomain::Int,
                    bound.clone(),
                    body.clone(),
                    ctx.env.clone(),
                )))),
                "Real" => Ok(Value::LazyFunc(Box::new(LazyFuncValue::new(
                    None,
                    LazyDomain::Real,
                    bound.clone(),
                    body.clone(),
                    ctx.env.clone(),
                )))),
                _ => Err(EvalError::type_error("Set", &dv, Some(domain_expr.span))),
            };
        }

        // Handle TupleSet (Cartesian product) with potentially infinite components
        // e.g., [p \in Nat \X Int |-> ...] should create a lazy function
        if let Value::TupleSet(ts) = &dv {
            // Check if any component is infinite
            let mut component_domains = Vec::new();
            let mut has_infinite = false;

            for comp in ts.components.iter() {
                let comp_domain = match comp.as_ref() {
                    Value::ModelValue(name) => match name.as_ref() {
                        "Nat" => {
                            has_infinite = true;
                            ComponentDomain::Nat
                        }
                        "Int" => {
                            has_infinite = true;
                            ComponentDomain::Int
                        }
                        "Real" => {
                            has_infinite = true;
                            ComponentDomain::Real
                        }
                        _ => {
                            return Err(EvalError::type_error(
                                "Set",
                                comp.as_ref(),
                                Some(domain_expr.span),
                            ));
                        }
                    },
                    Value::StringSet => {
                        has_infinite = true;
                        ComponentDomain::String
                    }
                    other => {
                        // Finite domain - get the OrdSet
                        let d = other.to_ord_set().ok_or_else(|| {
                            EvalError::type_error("Set", other, Some(domain_expr.span))
                        })?;
                        ComponentDomain::Finite(d)
                    }
                };
                component_domains.push(comp_domain);
            }

            // If any component is infinite, create a lazy function over the product domain
            if has_infinite {
                return Ok(Value::LazyFunc(Box::new(LazyFuncValue::new(
                    None,
                    LazyDomain::Product(component_domains),
                    bound.clone(),
                    body.clone(),
                    ctx.env.clone(),
                ))));
            }
            // Fall through to finite set handling below
        }

        // Optimization: Use IntIntervalFunc for integer interval domains
        // This is much faster for EXCEPT operations (array clone vs B-tree ops)
        // IMPORTANT: When domain is 1..n, create a Seq instead of IntFunc
        // because in TLA+ sequences are functions from 1..n and sequence operations
        // (Head, Tail, Append, etc.) need to work on them.
        if let Value::Interval(intv) = &dv {
            // Check if the interval fits in reasonable i64 bounds
            if let (Some(min), Some(max)) = (intv.low.to_i64(), intv.high.to_i64()) {
                let size = (max - min + 1) as usize;
                // Only use array representation for reasonably sized intervals
                if size <= 1_000_000 {
                    // Optimization: reuse a single EvalCtx with push/pop instead of cloning per element
                    let mut local_ctx = ctx.clone();
                    let mark = local_ctx.mark_stack();
                    let mut values = Vec::with_capacity(size);
                    for i in min..=max {
                        let elem = Value::SmallInt(i);
                        push_bound_var_mut(&mut local_ctx, bound, &elem, span)?;
                        let val = eval(&local_ctx, body)?;
                        values.push(val);
                        local_ctx.pop_to_mark(mark);
                    }
                    // If domain is 1..n, this is a sequence - create Seq for compatibility
                    // with sequence operations (Head, Tail, Append, Len, etc.)
                    if min == 1 {
                        return Ok(Value::Seq(values.into()));
                    }
                    return Ok(Value::IntFunc(IntIntervalFunc::new(min, max, values)));
                }
            }
        }

        // Handle all set-like types (Set, Interval, Subset, FuncSet, RecordSet, TupleSet)
        if dv.is_set() {
            // Try to enumerate the set - if it's infinite/non-enumerable, create a lazy function
            // Check enumerability by attempting to get an iterator (returns None if infinite)
            if let Some(iter) = dv.clone().iter_set() {
                // Finite set - evaluate eagerly
                // Collect domain elements and their mapped values in sorted order
                let mut local_ctx = ctx.clone();
                let mark = local_ctx.mark_stack();
                let mut entries: Vec<(Value, Value)> = Vec::new();
                let mut is_seq_domain = true;
                let mut expected_seq_idx: i64 = 1;
                for elem in iter {
                    // Track if domain is {1, 2, ..., n}
                    if is_seq_domain {
                        if elem.as_i64() == Some(expected_seq_idx) {
                            expected_seq_idx += 1;
                        } else {
                            is_seq_domain = false;
                        }
                    }
                    // Use push_bound_var_mut to handle tuple patterns
                    push_bound_var_mut(&mut local_ctx, bound, &elem, span)?;
                    let val = eval(&local_ctx, body)?;
                    entries.push((elem, val));
                    local_ctx.pop_to_mark(mark);
                }
                // IMPORTANT: If domain is exactly {1, 2, ..., n}, create a Seq instead of Func
                // because in TLA+ sequences are functions from 1..n and sequence operations
                // (Head, Tail, Append, etc.) need to work on them.
                if entries.is_empty() {
                    // Empty function is the empty sequence
                    return Ok(Value::Seq(Vec::new().into()));
                }
                if is_seq_domain {
                    // Domain is 1..n; entries were collected in key-sorted order
                    let seq_values: Vec<Value> = entries.into_iter().map(|(_, v)| v).collect();
                    return Ok(Value::Seq(seq_values.into()));
                }
                return Ok(Value::Func(FuncValue::from_sorted_entries(entries)));
            } else {
                // Non-enumerable set (e.g., SUBSET Int, [S -> T] over infinite sets)
                // Create a lazy function with the general domain
                return Ok(Value::LazyFunc(Box::new(LazyFuncValue::new(
                    None,
                    LazyDomain::General(Box::new(dv)),
                    bound.clone(),
                    body.clone(),
                    ctx.env.clone(),
                ))));
            }
        }

        return Err(EvalError::type_error("Set", &dv, Some(domain_expr.span)));
    }

    // Multi-variable function: [x \in S, y \in T |-> e]
    // Domain is S \X T, mapping uses tuples as keys
    // First, evaluate all domains and determine whether they are enumerable.
    let mut domain_values: Vec<Value> = Vec::with_capacity(bounds.len());
    let mut finite_domains: Vec<Option<OrdSet<Value>>> = Vec::with_capacity(bounds.len());
    let mut all_enumerable = true;

    for bound in bounds {
        let domain_expr = bound.domain.as_ref().ok_or_else(|| EvalError::Internal {
            message: "Function definition requires bounded variable".into(),
            span,
        })?;
        let dv = eval(ctx, domain_expr)?;

        if !dv.is_set() {
            return Err(EvalError::type_error("Set", &dv, Some(domain_expr.span)));
        }

        let ord = dv.to_ord_set();
        if ord.is_none() {
            all_enumerable = false;
        }

        domain_values.push(dv);
        finite_domains.push(ord);
    }

    // If any domain is non-enumerable (e.g., Int, Nat, SUBSET Int, Int \X Int),
    // create a lazy function.
    if !all_enumerable {
        // Prefer the efficient Product domain when each component can be represented
        // as a simple ComponentDomain (Nat/Int/Real/String or finite set). Otherwise,
        // fall back to a general tuple-set domain for correct membership checking.
        let mut component_domains: Vec<ComponentDomain> = Vec::with_capacity(bounds.len());
        let mut can_use_product = true;

        for (dv, ord) in domain_values.iter().zip(finite_domains.iter()) {
            let comp = match dv {
                Value::ModelValue(name) => match name.as_ref() {
                    "Nat" => ComponentDomain::Nat,
                    "Int" => ComponentDomain::Int,
                    "Real" => ComponentDomain::Real,
                    _ => {
                        can_use_product = false;
                        break;
                    }
                },
                Value::StringSet => ComponentDomain::String,
                _ => match ord {
                    Some(s) => ComponentDomain::Finite(s.clone()),
                    None => {
                        can_use_product = false;
                        break;
                    }
                },
            };
            component_domains.push(comp);
        }

        let domain = if can_use_product {
            LazyDomain::Product(component_domains)
        } else {
            LazyDomain::General(Box::new(Value::TupleSet(TupleSetValue::new(
                domain_values.clone(),
            ))))
        };

        return Ok(Value::LazyFunc(Box::new(LazyFuncValue::new_multi(
            None,
            domain,
            bounds.to_vec(),
            (*body).clone(),
            ctx.env.clone(),
        ))));
    }

    // All domains are finite + enumerable - eagerly compute the function.
    let finite_domains: Vec<OrdSet<Value>> = finite_domains
        .into_iter()
        .map(|d| d.expect("all_enumerable implies all domains are enumerable"))
        .collect();

    let domain_refs: Vec<_> = finite_domains.iter().collect();
    let product = cartesian_product(&domain_refs);
    let product_set = product.as_set().unwrap();

    // Optimization: reuse a single EvalCtx with push/pop instead of cloning per tuple
    // Build entries directly in sorted order (iter() returns sorted elements)
    let mut local_ctx = ctx.clone();
    let mark = local_ctx.mark_stack();
    let mut entries: Vec<(Value, Value)> = Vec::with_capacity(product_set.len());
    for tuple_val in product_set.iter() {
        let tuple = tuple_val.as_tuple().unwrap();
        for (i, bound) in bounds.iter().enumerate() {
            // Use push_bound_var_mut for each component to handle patterns
            push_bound_var_mut(&mut local_ctx, bound, &tuple[i], span)?;
        }
        let val = eval(&local_ctx, body)?;
        entries.push((tuple_val.clone(), val));
        local_ctx.pop_to_mark(mark);
    }

    Ok(Value::Func(FuncValue::from_sorted_entries(entries)))
}

/// Evaluate f[x]
fn eval_func_apply(
    ctx: &EvalCtx,
    func_expr: &Spanned<Expr>,
    arg_expr: &Spanned<Expr>,
    span: Option<Span>,
) -> EvalResult<Value> {
    let fv = eval(ctx, func_expr)?;
    let arg = eval(ctx, arg_expr)?;

    match &fv {
        Value::Func(f) => f
            .apply(&arg)
            .cloned()
            .ok_or_else(|| EvalError::NotInDomain {
                arg: format!("{}", arg),
                span,
            }),
        Value::IntFunc(f) => f
            .apply(&arg)
            .cloned()
            .ok_or_else(|| EvalError::NotInDomain {
                arg: format!("{}", arg),
                span,
            }),
        Value::LazyFunc(f) => {
            // Domain checking using the in_domain method
            if !f.in_domain(&arg) {
                return Err(EvalError::NotInDomain {
                    arg: format!("{}", arg),
                    span,
                });
            }

            // Memoized evaluation (avoid holding the lock while evaluating recursively)
            if let Some(v) = {
                let memo = f.memo.lock().map_err(|_| EvalError::Internal {
                    message: "Lazy function memoization mutex poisoned".into(),
                    span,
                })?;
                memo.get(&arg).cloned()
            } {
                return Ok(v);
            }

            // Check recursion depth to prevent stack overflow on deeply recursive functions
            let new_depth = ctx.recursion_depth + 1;
            if new_depth > MAX_RECURSION_DEPTH {
                return Err(EvalError::Internal {
                    message: format!(
                        "Maximum recursion depth ({}) exceeded in function evaluation. \
                         This may indicate infinite recursion or an overly deep recursive definition.",
                        MAX_RECURSION_DEPTH
                    ),
                    span,
                });
            }

            let mut env = f.env.clone();
            if let Some(name) = &f.name {
                env.insert(Arc::clone(name), Value::LazyFunc(f.clone()));
            }

            let base_ctx = EvalCtx {
                shared: Arc::clone(&ctx.shared),
                env,
                next_state: ctx.next_state.clone(),
                local_ops: ctx.local_ops.clone(),
                local_stack: Vec::new(),
                state_env: ctx.state_env,
                next_state_env: ctx.next_state_env,
                recursion_depth: new_depth,
                instance_substitutions: ctx.instance_substitutions.clone(),
            };

            // Bind variables based on number of bounds
            // Issue #100 fix: Use bind_local_bound_var to ensure LazyFunc parameters
            // are added to local_stack for dependency tracking. Without this, the
            // ZERO_ARG_OP_CACHE can return stale values when the bound variable changes.
            let bound_ctx = if f.bounds.len() == 1 {
                // Single-arg function: bind the single variable to the argument
                bind_local_bound_var(&base_ctx, &f.bounds[0], &arg, span)?
            } else {
                // Multi-arg function: arg should be a tuple, bind each component
                let components = arg
                    .as_tuple()
                    .ok_or_else(|| EvalError::type_error("Tuple", &arg, span))?;
                if components.len() != f.bounds.len() {
                    return Err(EvalError::type_error("Tuple", &arg, span));
                }
                let mut ctx = base_ctx;
                for (i, bound) in f.bounds.iter().enumerate() {
                    ctx = bind_local_bound_var(&ctx, bound, &components[i], span)?;
                }
                ctx
            };

            let value = eval(&bound_ctx, f.body.as_ref())?;

            {
                let mut memo = f.memo.lock().map_err(|_| EvalError::Internal {
                    message: "Lazy function memoization mutex poisoned".into(),
                    span,
                })?;
                memo.insert(arg.clone(), value.clone());
                #[cfg(feature = "memory-stats")]
                crate::value::memory_stats::inc_memo_entry();
            }

            Ok(value)
        }
        Value::Seq(s) => {
            // Sequence indexing is 1-based
            let idx = arg
                .as_i64()
                .ok_or_else(|| EvalError::type_error("Int", &arg, Some(arg_expr.span)))?;
            if idx < 1 || idx as usize > s.len() {
                return Err(EvalError::IndexOutOfBounds {
                    index: idx,
                    len: s.len(),
                    span,
                });
            }
            Ok(s[(idx - 1) as usize].clone())
        }
        Value::Tuple(t) => {
            // Tuple indexing is 1-based
            let idx = arg
                .as_i64()
                .ok_or_else(|| EvalError::type_error("Int", &arg, Some(arg_expr.span)))?;
            if idx < 1 || idx as usize > t.len() {
                return Err(EvalError::IndexOutOfBounds {
                    index: idx,
                    len: t.len(),
                    span,
                });
            }
            Ok(t[(idx - 1) as usize].clone())
        }
        Value::Record(r) => {
            let field = arg
                .as_string()
                .ok_or_else(|| EvalError::type_error("STRING", &arg, Some(arg_expr.span)))?;
            let field_arc: Arc<str> = intern_string(field);
            r.get(&field_arc)
                .cloned()
                .ok_or_else(|| EvalError::NoSuchField {
                    field: field.into(),
                    span,
                })
        }
        _ => Err(EvalError::type_error(
            "Function/Seq/Record",
            &fv,
            Some(func_expr.span),
        )),
    }
}

/// Evaluate [f EXCEPT ![x] = y, ![a][b] = c, ...]
fn eval_except(
    ctx: &EvalCtx,
    func_expr: &Spanned<Expr>,
    specs: &[ExceptSpec],
    span: Option<Span>,
) -> EvalResult<Value> {
    let fv = eval(ctx, func_expr)?;
    let mut result = fv;

    for spec in specs {
        result = apply_except_spec(ctx, result, &spec.path, &spec.value, span)?;
    }

    Ok(result)
}

/// Apply a single EXCEPT spec to a value, supporting nested paths
/// For `![a].b = v`: first index into the function/seq at `a`, then update field `b`
fn apply_except_spec(
    ctx: &EvalCtx,
    value: Value,
    path: &[ExceptPathElement],
    new_value_expr: &Spanned<Expr>,
    span: Option<Span>,
) -> EvalResult<Value> {
    if path.is_empty() {
        // Base case: we've navigated to the target, evaluate and return new value
        // @ refers to the old value at this position
        let new_ctx = ctx.bind("@", value);
        return eval(&new_ctx, new_value_expr);
    }

    // Recursive case: navigate one level and recurse
    let (first, rest) = (&path[0], &path[1..]);

    match (value, first) {
        // Function with index: f[idx]
        (Value::Func(f), ExceptPathElement::Index(idx_expr)) => {
            let idx = eval(ctx, idx_expr)?;
            // TLC treats EXCEPT updates outside DOMAIN as a no-op (with a warning).
            // We match this behavior by ignoring the spec if idx is not in DOMAIN.
            if !f.domain_contains(&idx) {
                return Ok(Value::Func(f));
            }
            let old_val = f.apply(&idx).cloned().ok_or_else(|| EvalError::Internal {
                message: format!(
                    "Function domain contains {:?} but mapping has no value",
                    idx
                ),
                span,
            })?;
            let new_val = apply_except_spec(ctx, old_val, rest, new_value_expr, span)?;
            Ok(Value::Func(f.except(idx, new_val)))
        }
        // IntFunc with index: [i \in 1..N |-> ...] EXCEPT ![k] = v
        // Optimized path for array-backed functions
        (Value::IntFunc(f), ExceptPathElement::Index(idx_expr)) => {
            let idx = eval(ctx, idx_expr)?;
            // Check if idx is in the domain (integer within min..max)
            let old_val = match f.apply(&idx) {
                Some(v) => v.clone(),
                None => return Ok(Value::IntFunc(f)), // Out of domain - no-op
            };
            let new_val = apply_except_spec(ctx, old_val, rest, new_value_expr, span)?;
            Ok(Value::IntFunc(f.except(&idx, new_val)))
        }
        // LazyFunc with index: [i \in Nat |-> ...] EXCEPT ![k] = v
        (Value::LazyFunc(f), ExceptPathElement::Index(idx_expr)) => {
            let idx = eval(ctx, idx_expr)?;
            // Check if idx is in the domain
            if !f.in_domain(&idx) {
                return Ok(Value::LazyFunc(f));
            }
            // Get the old value at idx (evaluate lazily if not in memo)
            let old_val = {
                let memo = f.memo.lock().map_err(|_| EvalError::Internal {
                    message: "LazyFunc memo mutex poisoned".into(),
                    span,
                })?;
                if let Some(v) = memo.get(&idx) {
                    v.clone()
                } else {
                    drop(memo); // Release lock before recursive eval

                    // Check recursion depth to prevent stack overflow
                    let new_depth = ctx.recursion_depth + 1;
                    if new_depth > MAX_RECURSION_DEPTH {
                        return Err(EvalError::Internal {
                            message: format!(
                                "Maximum recursion depth ({}) exceeded in EXCEPT evaluation.",
                                MAX_RECURSION_DEPTH
                            ),
                            span,
                        });
                    }

                    let mut env = f.env.clone();
                    if let Some(name) = &f.name {
                        env.insert(Arc::clone(name), Value::LazyFunc(f.clone()));
                    }
                    let base_ctx = EvalCtx {
                        shared: Arc::clone(&ctx.shared),
                        env,
                        next_state: ctx.next_state.clone(),
                        local_ops: ctx.local_ops.clone(),
                        local_stack: Vec::new(),
                        state_env: ctx.state_env,
                        next_state_env: ctx.next_state_env,
                        recursion_depth: new_depth,
                        instance_substitutions: ctx.instance_substitutions.clone(),
                    };
                    // Bind variables based on number of bounds
                    let bound_ctx = if f.bounds.len() == 1 {
                        bind_bound_var(&base_ctx, &f.bounds[0], &idx, span)?
                    } else {
                        // Multi-arg function: idx should be a tuple
                        let components = idx
                            .as_tuple()
                            .ok_or_else(|| EvalError::type_error("Tuple", &idx, span))?;
                        let mut ctx_acc = base_ctx;
                        for (i, bound) in f.bounds.iter().enumerate() {
                            ctx_acc = bind_bound_var(&ctx_acc, bound, &components[i], span)?;
                        }
                        ctx_acc
                    };
                    eval(&bound_ctx, f.body.as_ref())?
                }
            };
            let new_val = apply_except_spec(ctx, old_val, rest, new_value_expr, span)?;
            // Create a new LazyFunc with the exception added to memo
            let new_func = f.with_exception(idx, new_val);
            Ok(Value::LazyFunc(Box::new(new_func)))
        }
        // Record with field: r.field
        (Value::Record(r), ExceptPathElement::Field(field)) => {
            let field_arc: Arc<str> = intern_string(field.node.as_str());
            let Some(old_val) = r.get(&field_arc).cloned() else {
                // Match TLC: EXCEPT update to a missing record field is ignored (with a warning).
                return Ok(Value::Record(r));
            };
            let new_val = apply_except_spec(ctx, old_val, rest, new_value_expr, span)?;
            let new_rec = r.insert(field_arc, new_val);
            Ok(Value::Record(new_rec))
        }
        // Record with index (string key): r[idx]
        (Value::Record(r), ExceptPathElement::Index(idx_expr)) => {
            let idx = eval(ctx, idx_expr)?;
            let field = idx
                .as_string()
                .ok_or_else(|| EvalError::type_error("STRING", &idx, Some(idx_expr.span)))?;
            let field_arc: Arc<str> = intern_string(field);
            let Some(old_val) = r.get(&field_arc).cloned() else {
                // Match TLC: EXCEPT update to a missing record field is ignored (with a warning).
                return Ok(Value::Record(r));
            };
            let new_val = apply_except_spec(ctx, old_val, rest, new_value_expr, span)?;
            let new_rec = r.insert(field_arc, new_val);
            Ok(Value::Record(new_rec))
        }
        // Sequence with index: s[i]
        (Value::Seq(s), ExceptPathElement::Index(idx_expr)) => {
            let idx = eval(ctx, idx_expr)?;
            let i = idx
                .as_i64()
                .ok_or_else(|| EvalError::type_error("Int", &idx, Some(idx_expr.span)))?;
            // TLC: EXCEPT update outside domain is a no-op (with warning)
            // Sequence domain is 1..Len(s), so out-of-bounds is outside domain
            if i < 1 || i as usize > s.len() {
                return Ok(Value::Seq(s));
            }
            let old_val = s[(i - 1) as usize].clone();
            let new_val = apply_except_spec(ctx, old_val.clone(), rest, new_value_expr, span)?;
            // Optimization: if value unchanged, avoid cloning the entire sequence
            if old_val == new_val {
                return Ok(Value::Seq(s));
            }
            let mut new_s: Vec<Value> = s.iter().cloned().collect();
            new_s[(i - 1) as usize] = new_val;
            Ok(Value::Seq(new_s.into()))
        }
        // Tuple with index: t[i]
        (Value::Tuple(t), ExceptPathElement::Index(idx_expr)) => {
            let idx = eval(ctx, idx_expr)?;
            let i = idx
                .as_i64()
                .ok_or_else(|| EvalError::type_error("Int", &idx, Some(idx_expr.span)))?;
            if i < 1 || i as usize > t.len() {
                return Err(EvalError::IndexOutOfBounds {
                    index: i,
                    len: t.len(),
                    span,
                });
            }
            let old_val = t[(i - 1) as usize].clone();
            let new_val = apply_except_spec(ctx, old_val.clone(), rest, new_value_expr, span)?;
            // Optimization: if value unchanged, avoid cloning the entire tuple
            if old_val == new_val {
                return Ok(Value::Tuple(t));
            }
            let mut new_t: Vec<Value> = t.iter().cloned().collect();
            new_t[(i - 1) as usize] = new_val;
            Ok(Value::Tuple(new_t.into()))
        }
        // Type mismatches
        (Value::Func(_), ExceptPathElement::Field(_)) => Err(EvalError::Internal {
            message: "Field access on function not supported".into(),
            span,
        }),
        (Value::Seq(_), ExceptPathElement::Field(_)) => Err(EvalError::Internal {
            message: "Field access on sequence not supported".into(),
            span,
        }),
        (Value::Tuple(_), ExceptPathElement::Field(_)) => Err(EvalError::Internal {
            message: "Field access on tuple not supported".into(),
            span,
        }),
        (v, _) => Err(EvalError::type_error("Function/Record/Seq", &v, span)),
    }
}

// === JSON conversion helpers ===

/// Convert a TLA+ value to its JSON string representation
fn value_to_json(val: &Value) -> String {
    match val {
        Value::Bool(b) => if *b { "true" } else { "false" }.to_string(),
        Value::SmallInt(n) => n.to_string(),
        Value::Int(n) => n.to_string(),
        Value::String(s) => format!("\"{}\"", escape_json_string(s)),
        Value::ModelValue(s) => format!("\"{}\"", escape_json_string(s)),
        Value::Seq(_) | Value::Tuple(_) => {
            let elems = val.as_seq_or_tuple_elements().unwrap();
            let items: Vec<String> = elems.iter().map(value_to_json).collect();
            format!("[{}]", items.join(","))
        }
        Value::Set(set) => {
            let items: Vec<String> = set.iter().map(value_to_json).collect();
            format!("[{}]", items.join(","))
        }
        Value::Record(fields) => {
            let items: Vec<String> = fields
                .iter()
                .map(|(k, v)| format!("\"{}\":{}", escape_json_string(k), value_to_json(v)))
                .collect();
            format!("{{{}}}", items.join(","))
        }
        Value::Func(func) => {
            // Functions with integer domains: convert to array
            // Functions with string domains: convert to object
            let domain: Vec<_> = func.domain_iter().cloned().collect();
            let all_ints = domain.iter().all(|k| k.is_int());
            let all_strings = domain.iter().all(|k| k.as_string().is_some());

            if all_strings && !domain.is_empty() {
                // Convert to JSON object
                let items: Vec<String> = func
                    .mapping_iter()
                    .map(|(k, v)| {
                        let key_str = k.as_string().unwrap();
                        format!("\"{}\":{}", escape_json_string(key_str), value_to_json(v))
                    })
                    .collect();
                format!("{{{}}}", items.join(","))
            } else if all_ints && !domain.is_empty() {
                // Convert to JSON array (sorted by key) - entries are already sorted
                let items: Vec<String> =
                    func.mapping_iter().map(|(_, v)| value_to_json(v)).collect();
                format!("[{}]", items.join(","))
            } else {
                // Mixed domain: convert to array of [key, value] pairs
                let items: Vec<String> = func
                    .mapping_iter()
                    .map(|(k, v)| format!("[{},{}]", value_to_json(k), value_to_json(v)))
                    .collect();
                format!("[{}]", items.join(","))
            }
        }
        Value::IntFunc(func) => {
            // IntFunc is always integer-indexed: convert to JSON array
            let items: Vec<String> = func.values.iter().map(value_to_json).collect();
            format!("[{}]", items.join(","))
        }
        Value::Interval(interval) => {
            if let (Some(lo), Some(hi)) = (interval.low.to_i64(), interval.high.to_i64()) {
                let items: Vec<String> = (lo..=hi).map(|i| i.to_string()).collect();
                format!("[{}]", items.join(","))
            } else {
                // Large intervals: represent as object with bounds
                format!("{{\"from\":{},\"to\":{}}}", interval.low, interval.high)
            }
        }
        Value::Subset(sv) => {
            // SUBSET S is too large; represent as special object
            format!("{{\"subset\":{}}}", value_to_json(&sv.base))
        }
        Value::FuncSet(fv) => {
            format!(
                "{{\"funcset\":{{\"domain\":{},\"codomain\":{}}}}}",
                value_to_json(&fv.domain),
                value_to_json(&fv.codomain)
            )
        }
        Value::RecordSet(rv) => {
            let items: Vec<String> = rv
                .fields
                .iter()
                .map(|(k, v)| format!("\"{}\":{}", escape_json_string(k), value_to_json(v)))
                .collect();
            format!("{{\"recordset\":{{{}}}}}", items.join(","))
        }
        Value::TupleSet(tv) => {
            let items: Vec<String> = tv.components.iter().map(|v| value_to_json(v)).collect();
            format!("{{\"tupleset\":[{}]}}", items.join(","))
        }
        Value::SetCup(scv) => {
            // Lazy union - if enumerable, convert to array; otherwise represent as structure
            if let Some(set) = scv.to_ord_set() {
                let items: Vec<String> = set.iter().map(value_to_json).collect();
                format!("[{}]", items.join(","))
            } else {
                format!(
                    "{{\"setcup\":[{},{}]}}",
                    value_to_json(&scv.set1),
                    value_to_json(&scv.set2)
                )
            }
        }
        Value::SetCap(scv) => {
            // Lazy intersection - if enumerable, convert to array; otherwise represent as structure
            if let Some(set) = scv.to_ord_set() {
                let items: Vec<String> = set.iter().map(value_to_json).collect();
                format!("[{}]", items.join(","))
            } else {
                format!(
                    "{{\"setcap\":[{},{}]}}",
                    value_to_json(&scv.set1),
                    value_to_json(&scv.set2)
                )
            }
        }
        Value::SetDiff(sdv) => {
            // Lazy difference - if enumerable, convert to array; otherwise represent as structure
            if let Some(set) = sdv.to_ord_set() {
                let items: Vec<String> = set.iter().map(value_to_json).collect();
                format!("[{}]", items.join(","))
            } else {
                format!(
                    "{{\"setdiff\":[{},{}]}}",
                    value_to_json(&sdv.set1),
                    value_to_json(&sdv.set2)
                )
            }
        }
        Value::SetPred(spv) => {
            // SetPred can't be enumerated without evaluation context
            format!(
                "{{\"setpred\":{{\"source\":{},\"id\":{}}}}}",
                value_to_json(&spv.source),
                spv.id
            )
        }
        Value::KSubset(ksv) => {
            // KSubset - if enumerable, convert to array; otherwise represent as structure
            if let Some(set) = ksv.to_ord_set() {
                let items: Vec<String> = set.iter().map(value_to_json).collect();
                format!("[{}]", items.join(","))
            } else {
                format!(
                    "{{\"ksubset\":{{\"base\":{},\"k\":{}}}}}",
                    value_to_json(&ksv.base),
                    ksv.k
                )
            }
        }
        Value::BigUnion(uv) => {
            // BigUnion - if enumerable, convert to array; otherwise represent as structure
            if let Some(set) = uv.to_ord_set() {
                let items: Vec<String> = set.iter().map(value_to_json).collect();
                format!("[{}]", items.join(","))
            } else {
                format!("{{\"union\":{}}}", value_to_json(&uv.set))
            }
        }
        Value::LazyFunc(_) => "\"<lazy-function>\"".to_string(),
        Value::Closure(_) => "\"<closure>\"".to_string(),
        Value::StringSet => "\"STRING\"".to_string(),
        Value::AnySet => "\"ANY\"".to_string(),
        Value::SeqSet(_) => "\"Seq(...)\"".to_string(),
    }
}

/// Convert a TLA+ sequence/tuple value to JSON array string
fn value_to_json_array(val: &Value) -> EvalResult<String> {
    match val {
        Value::Seq(_) | Value::Tuple(_) => {
            let elems = val.as_seq_or_tuple_elements().unwrap();
            let items: Vec<String> = elems.iter().map(value_to_json).collect();
            Ok(format!("[{}]", items.join(",")))
        }
        Value::Set(set) => {
            let items: Vec<String> = set.iter().map(value_to_json).collect();
            Ok(format!("[{}]", items.join(",")))
        }
        _ => Err(EvalError::Internal {
            message: format!("ToJsonArray requires Seq/Tuple/Set, got {:?}", val),
            span: None,
        }),
    }
}

/// Convert a TLA+ record/function value to JSON object string
fn value_to_json_object(val: &Value) -> EvalResult<String> {
    match val {
        Value::Record(fields) => {
            let items: Vec<String> = fields
                .iter()
                .map(|(k, v)| format!("\"{}\":{}", escape_json_string(k), value_to_json(v)))
                .collect();
            Ok(format!("{{{}}}", items.join(",")))
        }
        Value::Func(func) => {
            // Convert function to object (keys must be strings or ints)
            let items: Vec<String> = func
                .mapping_iter()
                .map(|(k, v)| {
                    let key_str = match k {
                        Value::String(s) => s.to_string(),
                        Value::Int(n) => n.to_string(),
                        _ => format!("{}", k),
                    };
                    format!("\"{}\":{}", escape_json_string(&key_str), value_to_json(v))
                })
                .collect();
            Ok(format!("{{{}}}", items.join(",")))
        }
        _ => Err(EvalError::Internal {
            message: format!("ToJsonObject requires Record/Func, got {:?}", val),
            span: None,
        }),
    }
}

/// Escape special characters in a JSON string
fn escape_json_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c if c.is_control() => {
                result.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => result.push(c),
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use tla_core::{lower, parse_to_syntax_tree, FileId};

    fn eval_str(src: &str) -> EvalResult<Value> {
        // Wrap expression in a module for parsing
        // Use raw string to preserve backslashes
        let module_src = format!("---- MODULE Test ----\n\nOp == {}\n\n====", src);
        let tree = parse_to_syntax_tree(&module_src);
        let lower_result = lower(FileId(0), &tree);
        // Always print errors for debugging
        if !lower_result.errors.is_empty() {
            eprintln!("Lower errors: {:?}", lower_result.errors);
        }
        let module = match lower_result.module {
            Some(m) => m,
            None => {
                eprintln!("Source:\n{}", module_src);
                eprintln!("Lower errors: {:?}", lower_result.errors);
                panic!("Failed to lower module");
            }
        };

        // Find the Op definition and evaluate its body
        for unit in &module.units {
            if let tla_core::ast::Unit::Operator(def) = &unit.node {
                if def.name.node == "Op" {
                    let ctx = EvalCtx::new();
                    return eval(&ctx, &def.body);
                }
            }
        }
        eprintln!("Source:\n{}", module_src);
        eprintln!("Units found: {}", module.units.len());
        for (i, unit) in module.units.iter().enumerate() {
            eprintln!("Unit {}: {:?}", i, &unit.node);
        }
        panic!("Op not found");
    }

    #[test]
    fn test_eval_literals() {
        assert_eq!(eval_str("TRUE").unwrap(), Value::Bool(true));
        assert_eq!(eval_str("FALSE").unwrap(), Value::Bool(false));
        assert_eq!(eval_str("42").unwrap(), Value::int(42));
        assert_eq!(eval_str("-5").unwrap(), Value::int(-5));
        assert_eq!(eval_str("\"hello\"").unwrap(), Value::string("hello"));
    }

    #[test]
    fn test_eval_logic() {
        // Use raw strings to avoid escaping issues
        assert_eq!(eval_str(r#"TRUE /\ FALSE"#).unwrap(), Value::Bool(false));
        assert_eq!(eval_str(r#"TRUE \/ FALSE"#).unwrap(), Value::Bool(true));
        assert_eq!(eval_str("~TRUE").unwrap(), Value::Bool(false));
        assert_eq!(eval_str("FALSE => TRUE").unwrap(), Value::Bool(true));
        assert_eq!(eval_str("TRUE <=> TRUE").unwrap(), Value::Bool(true));
    }

    #[test]
    fn test_eval_arithmetic() {
        assert_eq!(eval_str("2 + 3").unwrap(), Value::int(5));
        assert_eq!(eval_str("10 - 4").unwrap(), Value::int(6));
        assert_eq!(eval_str("3 * 7").unwrap(), Value::int(21));
        assert_eq!(eval_str(r#"10 \div 3"#).unwrap(), Value::int(3));
        assert_eq!(eval_str("10 % 3").unwrap(), Value::int(1));
        // TLA+ uses Euclidean modulo: -1 % 5 = 4 (not -1 like Rust)
        assert_eq!(eval_str("(-1) % 5").unwrap(), Value::int(4));
        assert_eq!(eval_str("(-6) % 5").unwrap(), Value::int(4));
        assert_eq!(eval_str("(-5) % 5").unwrap(), Value::int(0));
        assert_eq!(eval_str("2^10").unwrap(), Value::int(1024));
    }

    #[test]
    fn test_random_element_accepts_lazy_sets() {
        let elem = eval_str(r#"RandomElement(SUBSET {1,2})"#).unwrap();
        let powerset = eval_str(r#"SUBSET {1,2}"#).unwrap();
        assert_eq!(powerset.set_contains(&elem), Some(true));
    }

    #[test]
    fn test_eval_let_recursive_function_def_uses_lazy_func() {
        // Regression test: recursive local function definitions like PermSeqs(perms) in
        // SchedulingAllocator previously stack overflowed due to eager FuncDef evaluation.
        let v = eval_str(
            r#"LET perms[ss \in SUBSET {1,2}] ==
                   IF ss = {} THEN {<<>>}
                   ELSE LET ps == [x \in ss |->
                                     { Append(sq, x) : sq \in perms[ss \ {x}] }]
                        IN UNION { ps[x] : x \in ss }
               IN perms[{1,2}]"#,
        )
        .unwrap();

        let set = v.as_set().expect("Expected set result");
        assert_eq!(set.len(), 2);
        assert!(set.contains(&Value::Seq(vec![Value::int(1), Value::int(2)].into())));
        assert!(set.contains(&Value::Seq(vec![Value::int(2), Value::int(1)].into())));
    }

    #[test]
    fn test_eval_recursive_max_over_subset() {
        // Issue #100: Recursive Maximum function over SUBSET domain
        // This is the pattern from PaxosCommit.tla

        // Single element case
        assert_eq!(
            eval_str(
                r#"LET S == {0}
                       Max[T \in SUBSET S] ==
                           IF T = {} THEN -1
                           ELSE LET n == CHOOSE n \in T : TRUE
                                    rmax == Max[T \ {n}]
                                IN IF n >= rmax THEN n ELSE rmax
                   IN Max[S]"#
            )
            .unwrap(),
            Value::int(0)
        );

        // Two element case - inline recursive call
        assert_eq!(
            eval_str(
                r#"LET S == {0, 1}
                       Max[T \in SUBSET S] ==
                           IF T = {} THEN -1
                           ELSE LET n == CHOOSE n \in T : TRUE
                                IN IF n >= Max[T \ {n}] THEN n ELSE Max[T \ {n}]
                   IN Max[S]"#
            )
            .unwrap(),
            Value::int(1)
        );

        // Two element case - nested LET with rmax binding
        assert_eq!(
            eval_str(
                r#"LET S == {0, 1}
                       Max[T \in SUBSET S] ==
                           IF T = {} THEN -1
                           ELSE LET n == CHOOSE n \in T : TRUE
                                    rmax == Max[T \ {n}]
                                IN IF n >= rmax THEN n ELSE rmax
                   IN Max[S]"#
            )
            .unwrap(),
            Value::int(1)
        );

        // Three element case to verify deeper recursion
        assert_eq!(
            eval_str(
                r#"LET S == {1, 2, 3}
                       Max[T \in SUBSET S] ==
                           IF T = {} THEN -1
                           ELSE LET n == CHOOSE n \in T : TRUE
                                    rmax == Max[T \ {n}]
                                IN IF n >= rmax THEN n ELSE rmax
                   IN Max[S]"#
            )
            .unwrap(),
            Value::int(3)
        );
    }

    #[test]
    fn test_eval_apply_builtin_operator_via_replacement() {
        let module_src = "---- MODULE Test ----\n\nOp == Plus(1, 2)\n\n====";
        let tree = parse_to_syntax_tree(module_src);
        let lower_result = lower(FileId(0), &tree);
        assert!(
            lower_result.errors.is_empty(),
            "Lower errors: {:?}",
            lower_result.errors
        );
        let module = lower_result.module.expect("Expected module");

        let mut ctx = EvalCtx::new();
        ctx.load_module(&module);
        ctx.add_op_replacement("Plus".to_string(), "+".to_string());

        assert_eq!(ctx.eval_op("Op").unwrap(), Value::int(3));
    }

    #[test]
    fn test_eval_comparison() {
        assert_eq!(eval_str("1 < 2").unwrap(), Value::Bool(true));
        assert_eq!(eval_str("2 <= 2").unwrap(), Value::Bool(true));
        assert_eq!(eval_str("3 > 2").unwrap(), Value::Bool(true));
        assert_eq!(eval_str("2 >= 3").unwrap(), Value::Bool(false));
        assert_eq!(eval_str("1 = 1").unwrap(), Value::Bool(true));
        assert_eq!(eval_str("1 /= 2").unwrap(), Value::Bool(true));
    }

    #[test]
    fn test_eval_sets() {
        assert_eq!(
            eval_str("{1, 2, 3}").unwrap(),
            Value::set([Value::int(1), Value::int(2), Value::int(3)])
        );
        assert_eq!(eval_str(r#"1 \in {1, 2}"#).unwrap(), Value::Bool(true));
        assert_eq!(eval_str(r#"3 \notin {1, 2}"#).unwrap(), Value::Bool(true));
        assert_eq!(
            eval_str(r#"{1} \cup {2}"#).unwrap(),
            Value::set([Value::int(1), Value::int(2)])
        );
        assert_eq!(
            eval_str(r#"{1, 2} \cap {2, 3}"#).unwrap(),
            Value::set([Value::int(2)])
        );
        assert_eq!(
            eval_str(r#"{1, 2} \ {2}"#).unwrap(),
            Value::set([Value::int(1)])
        );
    }

    #[test]
    fn test_eval_prime_of_derived_operator_uses_next_state_env() {
        // Prime should work for non-variable expressions like `Foo'` where Foo is a zero-arg
        // operator (e.g., a derived view over variables).
        let src = r#"
---- MODULE Test ----
Foo == x
Op == Foo'
===="#;

        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        assert!(
            lower_result.errors.is_empty(),
            "lower errors: {:?}",
            lower_result.errors
        );
        let module = lower_result.module.expect("lower produced no module");

        let mut ctx = EvalCtx::new();
        ctx.load_module(&module);
        ctx.bind_mut("x", Value::int(1));

        let mut next = Env::new();
        next.insert(Arc::from("x"), Value::int(2));
        ctx.next_state = Some(Arc::new(next));

        let op_def = ctx.get_op("Op").expect("Op not found").clone();
        let v = eval(&ctx, &op_def.body).unwrap();
        assert_eq!(v, Value::int(2));
    }

    #[test]
    fn test_eval_module_ref_resolves_unqualified_ops_within_instance_module() {
        // Regression: when evaluating an instance reference `Inst!Next(1)`, operator calls inside
        // `Next` must resolve within the instanced module, not against the current module.
        //
        // This matters when both modules define the same operator name with different arities.
        let mod_m = r#"
---- MODULE M ----
EXTENDS Integers
SendMsg(x) == x + 1
Next(i) == SendMsg(i)
===="#;
        let mod_main = r#"
---- MODULE Main ----
EXTENDS Integers
SendMsg(x, y) == x + y
Inst == INSTANCE M
Op == Inst!Next(1)
===="#;

        let tree_m = parse_to_syntax_tree(mod_m);
        let lower_m = lower(FileId(0), &tree_m);
        assert!(
            lower_m.errors.is_empty(),
            "lower M errors: {:?}",
            lower_m.errors
        );
        let module_m = lower_m.module.expect("lower produced no module M");

        let tree_main = parse_to_syntax_tree(mod_main);
        let lower_main = lower(FileId(0), &tree_main);
        assert!(
            lower_main.errors.is_empty(),
            "lower Main errors: {:?}",
            lower_main.errors
        );
        let module_main = lower_main.module.expect("lower produced no module Main");

        let mut ctx = EvalCtx::new();
        ctx.load_module(&module_main);
        ctx.load_instance_module("M".to_string(), &module_m);

        let op_def = ctx.get_op("Op").expect("Op not found").clone();
        let v = eval(&ctx, &op_def.body).unwrap();
        assert_eq!(v, Value::int(2));
    }

    #[test]
    fn test_eval_func_set_membership_with_nat_range() {
        assert_eq!(
            eval_str(r#"([x \in {1, 2} |-> x] \in [{1, 2} -> Nat])"#).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            eval_str(r#"([x \in {1} |-> -1] \in [{1} -> Nat])"#).unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn test_eval_range() {
        assert_eq!(
            eval_str("1..3").unwrap(),
            Value::set([Value::int(1), Value::int(2), Value::int(3)])
        );
        // Empty range
        let empty = eval_str("5..3").unwrap();
        assert!(empty.as_set().unwrap().is_empty());
    }

    #[test]
    fn test_eval_tuples() {
        assert_eq!(
            eval_str("<<1, 2, 3>>").unwrap(),
            Value::tuple([Value::int(1), Value::int(2), Value::int(3)])
        );
    }

    /// Regression test for #62: tuple subscript evaluation
    ///
    /// Tests that tuples can be indexed using 1-based subscripting.
    /// This is the standard TLA+ pattern for accessing tuple elements.
    #[test]
    fn test_eval_tuple_subscript() {
        // Basic 1-based indexing
        assert_eq!(eval_str("<<10, 20, 30>>[1]").unwrap(), Value::int(10));
        assert_eq!(eval_str("<<10, 20, 30>>[2]").unwrap(), Value::int(20));
        assert_eq!(eval_str("<<10, 20, 30>>[3]").unwrap(), Value::int(30));

        // Nested tuple access
        assert_eq!(
            eval_str("<<1, <<2, 3>>, 4>>[2]").unwrap(),
            Value::tuple([Value::int(2), Value::int(3)])
        );
        assert_eq!(eval_str("<<1, <<2, 3>>, 4>>[2][1]").unwrap(), Value::int(2));
        assert_eq!(eval_str("<<1, <<2, 3>>, 4>>[2][2]").unwrap(), Value::int(3));

        // Index out of bounds should error
        assert!(eval_str("<<1, 2, 3>>[0]").is_err());
        assert!(eval_str("<<1, 2, 3>>[4]").is_err());
        assert!(eval_str("<<>>[1]").is_err());

        // Empty tuple has no valid indices
        assert!(eval_str("<<>>[0]").is_err());
    }

    #[test]
    fn test_eval_records() {
        let r = eval_str("[x |-> 1, y |-> 2]").unwrap();
        let rec = r.as_record().unwrap();
        assert_eq!(rec.get(&Arc::from("x")), Some(&Value::int(1)));
        assert_eq!(rec.get(&Arc::from("y")), Some(&Value::int(2)));
    }

    #[test]
    fn test_eval_record_set_with_interval_field() {
        let v = eval_str("[x: 1..3]").unwrap();
        assert!(matches!(v, Value::RecordSet(_)));
        assert_eq!(
            v,
            Value::set([
                Value::record([("x", Value::int(1))]),
                Value::record([("x", Value::int(2))]),
                Value::record([("x", Value::int(3))]),
            ])
        );
    }

    #[test]
    fn test_choose_over_record_set_is_deterministic() {
        let v = eval_str(r#"CHOOSE r \in [x: 1..3] : TRUE"#).unwrap();
        let rec = v.as_record().unwrap();
        assert_eq!(rec.get("x"), Some(&Value::int(1)));
    }

    #[test]
    fn test_cardinality_of_record_set() {
        assert_eq!(
            eval_str("Cardinality([a: 1..3, b: {1, 2}])").unwrap(),
            Value::int(6)
        );
    }

    #[test]
    fn test_eval_if_then_else() {
        assert_eq!(eval_str("IF TRUE THEN 1 ELSE 2").unwrap(), Value::int(1));
        assert_eq!(eval_str("IF FALSE THEN 1 ELSE 2").unwrap(), Value::int(2));
    }

    #[test]
    fn test_eval_case() {
        assert_eq!(
            eval_str("CASE TRUE -> 1 [] FALSE -> 2").unwrap(),
            Value::int(1)
        );
        assert_eq!(
            eval_str("CASE FALSE -> 1 [] TRUE -> 2").unwrap(),
            Value::int(2)
        );
    }

    #[test]
    fn test_eval_let() {
        assert_eq!(eval_str("LET x == 5 IN x + 1").unwrap(), Value::int(6));
        assert_eq!(
            eval_str("LET a == 2 b == 3 IN a * b").unwrap(),
            Value::int(6)
        );
    }

    #[test]
    fn test_eval_recursive_let_function_over_nat() {
        assert_eq!(
            eval_str(r#"LET f[n \in Nat] == IF n = 0 THEN 0 ELSE f[n-1] + 1 IN f[3]"#).unwrap(),
            Value::int(3)
        );
    }

    #[test]
    fn test_eval_transitive_closure_style_let_recursion() {
        // This evaluation is deeply recursive and can overflow the default test thread stack.
        let expr = r#"
LET S == {1, 2, 3}
    R == [x, y \in S |-> (x = 1 /\ y = 2) \/ (x = 2 /\ y = 3)]
    N == Cardinality(S)
    trcl[n \in Nat] ==
        [x, y \in S |->
            IF n = 0
            THEN R[x, y]
            ELSE \/ trcl[n-1][x, y]
                 \/ \E z \in S : trcl[n-1][x, z] /\ trcl[n-1][z, y]]
IN  /\ trcl[N][1, 3]
    /\ ~trcl[N][3, 1]
    /\ ~trcl[N][1, 1]
"#;

        let handle = std::thread::Builder::new()
            .name("test_eval_transitive_closure_style_let_recursion".to_string())
            .stack_size(16 * 1024 * 1024)
            .spawn(move || eval_str(expr))
            .expect("spawn test thread");

        let result = handle.join().expect("join test thread");
        assert_eq!(result.unwrap(), Value::Bool(true));
    }

    #[test]
    fn test_eval_forall() {
        assert_eq!(
            eval_str(r#"\A x \in {1, 2, 3} : x > 0"#).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            eval_str(r#"\A x \in {1, 2, 3} : x > 2"#).unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn test_eval_exists() {
        assert_eq!(
            eval_str(r#"\E x \in {1, 2, 3} : x > 2"#).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            eval_str(r#"\E x \in {1, 2, 3} : x > 5"#).unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn test_eval_choose() {
        // CHOOSE returns the first element satisfying the predicate
        let v = eval_str(r#"CHOOSE x \in {1, 2, 3} : x > 1"#).unwrap();
        assert!(v.as_i64().unwrap() > 1);
    }

    #[test]
    fn test_eval_set_builder() {
        assert_eq!(
            eval_str(r#"{x * 2 : x \in {1, 2, 3}}"#).unwrap(),
            Value::set([Value::int(2), Value::int(4), Value::int(6)])
        );
    }

    #[test]
    fn test_eval_set_builder_multi_var() {
        // Test {<<x, y>> : x, y \in {1, 2}}
        // First, let's check the CST/AST
        let src = r#"
---- MODULE Test ----
S == {<<x, y>> : x, y \in {1, 2}}
====
"#;
        let tree = tla_core::parse_to_syntax_tree(src);
        eprintln!("=== CST for multi-var set builder ===");
        eprintln!("{:#?}", tree);

        let lower_result = tla_core::lower(tla_core::FileId(0), &tree);
        eprintln!("Lower errors: {:?}", lower_result.errors);
        let module = lower_result.module.unwrap();
        for unit in &module.units {
            if let tla_core::ast::Unit::Operator(def) = &unit.node {
                eprintln!("Op '{}' body: {:?}", def.name.node, def.body.node);
            }
        }

        // Setup eval context
        let mut ctx = EvalCtx::new();
        ctx.load_module(&module);

        let result = ctx.eval_op("S");
        eprintln!("Multi-var set builder result: {:?}", result);
        let val = result.unwrap();
        let set = val.as_set().unwrap();
        assert_eq!(set.len(), 4); // 2*2 = 4 tuples
    }

    #[test]
    fn test_eval_set_filter() {
        assert_eq!(
            eval_str(r#"{x \in {1, 2, 3, 4} : x > 2}"#).unwrap(),
            Value::set([Value::int(3), Value::int(4)])
        );
    }

    #[test]
    fn test_eval_func_def() {
        // Domain {1, 2} is exactly 1..2, so this becomes a Seq in TLA+ semantics
        let f = eval_str(r#"[x \in {1, 2} |-> x * 10]"#).unwrap();
        // Since domain is {1, 2} = 1..2, this is now a sequence
        let seq = f
            .as_seq()
            .expect("Function with domain 1..n should be a Seq");
        assert_eq!(seq.len(), 2);
        assert_eq!(seq[0], Value::int(10)); // seq[1] in TLA+ = seq[0] in Rust
        assert_eq!(seq[1], Value::int(20)); // seq[2] in TLA+ = seq[1] in Rust

        // Test with non-sequence domain (e.g., {0, 1} - doesn't start at 1)
        let f = eval_str(r#"[x \in {0, 1} |-> x * 10]"#).unwrap();
        let func = f
            .as_func()
            .expect("Function with domain not starting at 1 should be Func");
        assert_eq!(func.apply(&Value::int(0)), Some(&Value::int(0)));
        assert_eq!(func.apply(&Value::int(1)), Some(&Value::int(10)));
    }

    #[test]
    fn test_eval_except_at_reference() {
        // EXCEPT RHS can reference `@` (old value). This is essential for patterns like:
        //   [f EXCEPT ![i] = @ \\cup {x}]
        // Since domain 1..1 = {1}, which is 1..n, the result is a Seq
        let f = eval_str(r#"[ [i \in 1..1 |-> {2}] EXCEPT ![1] = @ \cup {1} ]"#).unwrap();
        let set = match &f {
            Value::Seq(s) => s[0].as_set().expect("Element should be a set"),
            Value::IntFunc(func) => func.apply(&Value::int(1)).unwrap().as_set().unwrap(),
            Value::Func(func) => func.apply(&Value::int(1)).unwrap().as_set().unwrap(),
            _ => panic!("Expected sequence or function value, got {:?}", f),
        };
        assert_eq!(set.len(), 2);
        assert!(set.contains(&Value::int(1)));
        assert!(set.contains(&Value::int(2)));

        // And for set difference:
        let f = eval_str(r#"[ [i \in 1..1 |-> {1, 2}] EXCEPT ![1] = @ \ {1} ]"#).unwrap();
        let set = match &f {
            Value::Seq(s) => s[0].as_set().expect("Element should be a set"),
            Value::IntFunc(func) => func.apply(&Value::int(1)).unwrap().as_set().unwrap(),
            Value::Func(func) => func.apply(&Value::int(1)).unwrap().as_set().unwrap(),
            _ => panic!("Expected sequence or function value, got {:?}", f),
        };
        assert_eq!(set.len(), 1);
        assert!(set.contains(&Value::int(2)));
    }

    #[test]
    fn test_eval_except_ignores_out_of_domain_function_update() {
        // Domain {1} = 1..1 is now a sequence with the new semantics
        // EXCEPT ![2] = 3 on a seq of length 1 should be a no-op (index out of bounds)
        let f = eval_str(r#"[ [i \in {1} |-> 0] EXCEPT ![2] = 3 ]"#).unwrap();
        let seq = f.as_seq().expect("Function with domain 1..n should be Seq");
        assert_eq!(seq.len(), 1);
        assert_eq!(seq[0], Value::int(0)); // Unchanged

        // Accessing index 2 on a seq of length 1 should error
        let err = eval_str(r#"([ [i \in {1} |-> 0] EXCEPT ![2] = 3 ])[2]"#).unwrap_err();
        assert!(matches!(err, EvalError::IndexOutOfBounds { .. }));

        // Test with non-1..n domain (e.g., {0}) which remains a Func
        let f = eval_str(r#"[ [i \in {0} |-> 0] EXCEPT ![1] = 3 ]"#).unwrap();
        let func = f
            .as_func()
            .expect("Function with domain not starting at 1 should be Func");
        assert!(func.domain_contains(&Value::int(0)));
        assert!(!func.domain_contains(&Value::int(1)));
        assert_eq!(func.apply(&Value::int(0)), Some(&Value::int(0)));
        assert_eq!(func.apply(&Value::int(1)), None);
    }

    #[test]
    fn test_eval_except_ignores_missing_record_field_update() {
        let r = eval_str(r#"[ [a |-> 1] EXCEPT !.b = 2 ]"#).unwrap();
        let rec = r.as_record().unwrap();
        assert_eq!(rec.len(), 1);
        assert_eq!(rec.get(&Arc::from("a")), Some(&Value::int(1)));
        assert_eq!(rec.get(&Arc::from("b")), None);

        let err = eval_str(r#"([ [a |-> 1] EXCEPT !.b = 2 ])["b"]"#).unwrap_err();
        assert!(matches!(err, EvalError::NoSuchField { .. }));
    }

    #[test]
    fn test_eval_except_updates_existing_record_field() {
        let r = eval_str(r#"[ [a |-> 1] EXCEPT !.a = 2 ]"#).unwrap();
        let rec = r.as_record().unwrap();
        assert_eq!(rec.len(), 1);
        assert_eq!(rec.get(&Arc::from("a")), Some(&Value::int(2)));
    }

    #[test]
    fn test_eval_func_operator_tuple_pattern_destructures() {
        let src = r#"
---- MODULE Test ----
sc[<<x, y>> \in {<<1, 2>>}] == x + y
Test == sc[<<1, 2>>]
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        assert!(
            lower_result.errors.is_empty(),
            "Errors: {:?}",
            lower_result.errors
        );

        let module = lower_result.module.unwrap();
        let mut ctx = EvalCtx::new();
        ctx.load_module(&module);

        let val = ctx.eval_op("Test").unwrap();
        assert_eq!(val, Value::int(3));
    }

    #[test]
    fn test_eval_func_operator_tuple_pattern_sc_game_of_life_shape() {
        let src = r#"
---- MODULE Test ----
EXTENDS Integers
CONSTANT N
VARIABLE grid

Pos == {<<x, y>> : x, y \in 1..N}
Grid == [p \in Pos |-> TRUE]

sc[<<x, y>> \in (0 .. N + 1) \X (0 .. N + 1)] ==
    CASE \/ x = 0 \/ y = 0
         \/ x > N \/ y > N
         \/ ~grid[<<x, y>>] -> 0
    [] OTHER -> 1

Test == sc[<<1, 1>>]
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        assert!(
            lower_result.errors.is_empty(),
            "Errors: {:?}",
            lower_result.errors
        );

        let module = lower_result.module.unwrap();
        let mut ctx = EvalCtx::new();
        ctx.load_module(&module);

        ctx.bind_mut("N", Value::int(2));
        let grid = ctx.eval_op("Grid").unwrap();
        ctx.bind_mut("grid", grid);

        let val = ctx.eval_op("Test").unwrap();
        assert_eq!(val, Value::int(1));
    }

    #[test]
    fn test_eval_game_of_life_score_no_undefined_x() {
        // This evaluation is deeply recursive and can overflow the default test thread stack.
        let src = r#"
---- MODULE Test ----
EXTENDS Integers
CONSTANT N
VARIABLE grid

RECURSIVE Sum(_, _)
Sum(f, S) == IF S = {} THEN 0
                       ELSE LET x == CHOOSE x \in S : TRUE
                            IN  f[x] + Sum(f, S \ {x})

Pos == {<<x, y>> : x, y \in 1..N}
Grid == [p \in Pos |-> TRUE]

sc[<<x, y>> \in (0 .. N + 1) \X (0 .. N + 1)] ==
    CASE \/ x = 0 \/ y = 0
         \/ x > N \/ y > N
         \/ ~grid[<<x, y>>] -> 0
    [] OTHER -> 1

score(p) == LET nbrs == {x \in {-1, 0, 1} \X
                               {-1, 0, 1} : x /= <<0, 0>>}
                points == {<<p[1] + x, p[2] + y>> : <<x, y>> \in nbrs}
            IN Sum(sc, points)

Test == score(<<1, 1>>)
====
"#;

        let handle = std::thread::Builder::new()
            .name("test_eval_game_of_life_score_no_undefined_x".to_string())
            .stack_size(16 * 1024 * 1024)
            .spawn(move || {
                let tree = parse_to_syntax_tree(src);
                let lower_result = lower(FileId(0), &tree);
                assert!(
                    lower_result.errors.is_empty(),
                    "Errors: {:?}",
                    lower_result.errors
                );

                let module = lower_result.module.unwrap();
                let mut ctx = EvalCtx::new();
                ctx.load_module(&module);

                ctx.bind_mut("N", Value::int(2));
                let grid = ctx.eval_op("Grid").unwrap();
                ctx.bind_mut("grid", grid);

                ctx.eval_op("Test")
            })
            .expect("spawn test thread");

        let val = handle
            .join()
            .expect("join test thread")
            .expect("Test should evaluate");
        assert!(val.as_i64().is_some(), "Expected Int, got {:?}", val);
    }

    #[test]
    fn test_eval_recursive_sum_with_choose_and_let() {
        let src = r#"
---- MODULE Test ----
RECURSIVE Sum(_, _)
Sum(f, S) == IF S = {} THEN 0
                       ELSE LET x == CHOOSE x \in S : TRUE
                            IN  f[x] + Sum(f, S \ {x})

Test == LET f == [x \in {1, 2} |-> 1]
            IN Sum(f, {1, 2})
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        assert!(
            lower_result.errors.is_empty(),
            "Errors: {:?}",
            lower_result.errors
        );

        let module = lower_result.module.unwrap();
        let mut ctx = EvalCtx::new();
        ctx.load_module(&module);

        let val = ctx.eval_op("Test").unwrap();
        assert_eq!(val, Value::int(2));
    }

    #[test]
    fn test_eval_cardinality() {
        assert_eq!(eval_str("Cardinality({1, 2, 3})").unwrap(), Value::int(3));
        assert_eq!(eval_str("Cardinality({})").unwrap(), Value::int(0));
    }

    #[test]
    fn test_eval_len() {
        assert_eq!(eval_str("Len(<<1, 2, 3>>)").unwrap(), Value::int(3));
        assert_eq!(eval_str("Len(<<>>)").unwrap(), Value::int(0));
    }

    #[test]
    fn test_eval_head_tail() {
        assert_eq!(eval_str("Head(<<1, 2, 3>>)").unwrap(), Value::int(1));
        assert_eq!(
            eval_str("Tail(<<1, 2, 3>>)").unwrap(),
            Value::seq([Value::int(2), Value::int(3)])
        );
    }

    #[test]
    fn test_eval_head_tail_on_func_constructed_seq() {
        // Functions with domain 1..n are semantically sequences in TLA+
        // Head/Tail must work on them (issue: AlternatingBit Lose function)
        assert_eq!(
            eval_str(r#"Head([x \in 1..3 |-> x * 2])"#).unwrap(),
            Value::int(2)
        );
        assert_eq!(
            eval_str(r#"Tail([x \in 1..3 |-> x * 2])"#).unwrap(),
            Value::seq([Value::int(4), Value::int(6)])
        );
        // Len must also work
        assert_eq!(
            eval_str(r#"Len([x \in 1..3 |-> x * 2])"#).unwrap(),
            Value::int(3)
        );
        // Test the pattern from AlternatingBit Lose function
        // [j \in 1..(Len(q)-1) |-> ...] should work with Tail
        assert_eq!(
            eval_str(r#"Tail([j \in 1..2 |-> j])"#).unwrap(),
            Value::seq([Value::int(2)])
        );
    }

    #[test]
    fn test_eval_append() {
        assert_eq!(
            eval_str("Append(<<1, 2>>, 3)").unwrap(),
            Value::seq([Value::int(1), Value::int(2), Value::int(3)])
        );
    }

    #[test]
    fn test_eval_subseq() {
        assert_eq!(
            eval_str("SubSeq(<<1, 2, 3, 4>>, 2, 3)").unwrap(),
            Value::seq([Value::int(2), Value::int(3)])
        );
    }

    #[test]
    fn test_eval_subseq_accepts_seq_like_functions() {
        // SubSeq must accept any function value with domain 1..n, not just <<...>> literals.
        assert_eq!(
            eval_str(r#"SubSeq([i \in 1..4 |-> i], 2, 3)"#).unwrap(),
            Value::seq([Value::int(2), Value::int(3)])
        );
        assert_eq!(
            eval_str(r#"SubSeq([i \in {1, 2, 3, 4} |-> i], 2, 3)"#).unwrap(),
            Value::seq([Value::int(2), Value::int(3)])
        );
    }

    #[test]
    fn test_eval_func_override_accepts_records_and_sequences() {
        // Records: preserve record value so r.field works.
        let v = eval_str(r#"([dst |-> "a"] @@ [data |-> 15])"#).unwrap();
        assert!(matches!(v, Value::Record(_)));
        let rec = v.as_record().unwrap();
        assert_eq!(rec.get(&Arc::from("data")), Some(&Value::int(15)));
        assert_eq!(rec.get(&Arc::from("dst")), Some(&Value::string("a")));
        assert_eq!(
            eval_str(r#"([dst |-> "a"] @@ [data |-> 15]).data"#).unwrap(),
            Value::int(15)
        );

        // Sequences: @@ should accept tuples/seqs and singleton functions.
        assert_eq!(
            eval_str(r#"(1 :> "a" @@ <<>>)"#).unwrap(),
            Value::seq([Value::string("a")])
        );
        assert_eq!(
            eval_str(r#"(2 :> "x" @@ <<"a", "b", "c">>)"#).unwrap(),
            Value::seq([Value::string("a"), Value::string("x"), Value::string("c")])
        );
    }

    /// Helper to evaluate with multiple operator definitions
    fn eval_with_ops(defs: &str, expr: &str) -> EvalResult<Value> {
        let module_src = format!(
            "---- MODULE Test ----\n\n{}\n\nOp == {}\n\n====",
            defs, expr
        );
        let tree = parse_to_syntax_tree(&module_src);
        let lower_result = lower(FileId(0), &tree);
        if !lower_result.errors.is_empty() {
            eprintln!("Lower errors: {:?}", lower_result.errors);
        }
        let module = match lower_result.module {
            Some(m) => m,
            None => {
                eprintln!("Source:\n{}", module_src);
                eprintln!("Lower errors: {:?}", lower_result.errors);
                panic!("Failed to lower module");
            }
        };

        // Load all operator definitions
        let mut ctx = EvalCtx::new();
        ctx.load_module(&module);

        // Evaluate the Op
        ctx.eval_op("Op")
    }

    #[test]
    fn test_eval_selectseq() {
        // SelectSeq(s, Test) - filter sequence by test operator
        assert_eq!(
            eval_with_ops(
                "IsEven(x) == x % 2 = 0",
                "SelectSeq(<<1, 2, 3, 4, 5, 6>>, IsEven)"
            )
            .unwrap(),
            Value::seq([Value::int(2), Value::int(4), Value::int(6)])
        );

        // Empty result
        assert_eq!(
            eval_with_ops(
                "IsNegative(x) == x < 0",
                "SelectSeq(<<1, 2, 3>>, IsNegative)"
            )
            .unwrap(),
            Value::seq([])
        );

        // All pass
        assert_eq!(
            eval_with_ops(
                "IsPositive(x) == x > 0",
                "SelectSeq(<<1, 2, 3>>, IsPositive)"
            )
            .unwrap(),
            Value::seq([Value::int(1), Value::int(2), Value::int(3)])
        );
    }

    #[test]
    fn test_eval_sortseq() {
        // SortSeq(s, Op) - sort sequence using comparator
        // Ascending order
        assert_eq!(
            eval_with_ops(
                "LessThan(a, b) == a < b",
                "SortSeq(<<3, 1, 4, 1, 5>>, LessThan)"
            )
            .unwrap(),
            Value::seq([
                Value::int(1),
                Value::int(1),
                Value::int(3),
                Value::int(4),
                Value::int(5)
            ])
        );

        // Descending order
        assert_eq!(
            eval_with_ops(
                "GreaterThan(a, b) == a > b",
                "SortSeq(<<3, 1, 4, 1, 5>>, GreaterThan)"
            )
            .unwrap(),
            Value::seq([
                Value::int(5),
                Value::int(4),
                Value::int(3),
                Value::int(1),
                Value::int(1)
            ])
        );

        // Empty sequence
        assert_eq!(
            eval_with_ops("LessThan(a, b) == a < b", "SortSeq(<<>>, LessThan)").unwrap(),
            Value::seq([])
        );

        // Single element
        assert_eq!(
            eval_with_ops("LessThan(a, b) == a < b", "SortSeq(<<42>>, LessThan)").unwrap(),
            Value::seq([Value::int(42)])
        );
    }

    #[test]
    fn test_eval_permutations() {
        // Permutations of empty set: just the empty function
        let empty_perms = eval_str("Permutations({})").unwrap();
        let empty_set = empty_perms.as_set().unwrap();
        assert_eq!(empty_set.len(), 1);
        // Check it contains the empty function
        for v in empty_set.iter() {
            assert!(v.as_func().is_some());
            let f = v.as_func().unwrap();
            assert!(f.domain_is_empty());
        }

        // Permutations of single element: identity function only
        let single_perms = eval_str("Permutations({1})").unwrap();
        let single_set = single_perms.as_set().unwrap();
        assert_eq!(single_set.len(), 1);
        // Check the function maps 1 |-> 1
        for v in single_set.iter() {
            let f = v.as_func().unwrap();
            assert_eq!(f.mapping_get(&Value::int(1)), Some(&Value::int(1)));
        }

        // Permutations of two elements: {1, 2} -> 2! = 2 permutation functions
        // [1 |-> 1, 2 |-> 2] (identity) and [1 |-> 2, 2 |-> 1] (swap)
        let two_perms = eval_str("Permutations({1, 2})").unwrap();
        let two_set = two_perms.as_set().unwrap();
        assert_eq!(two_set.len(), 2);
        // Check all are functions
        for v in two_set.iter() {
            assert!(v.as_func().is_some());
        }

        // Permutations of three elements: 3! = 6 permutation functions
        let three_perms = eval_str("Permutations({1, 2, 3})").unwrap();
        let three_set = three_perms.as_set().unwrap();
        assert_eq!(three_set.len(), 6);
        // All should be bijections [S -> S]
        for v in three_set.iter() {
            let f = v.as_func().unwrap();
            assert_eq!(f.domain_len(), 3);
        }
    }

    #[test]
    fn test_eval_reverse() {
        // Reverse a sequence
        assert_eq!(
            eval_str("Reverse(<<1, 2, 3>>)").unwrap(),
            Value::seq([Value::int(3), Value::int(2), Value::int(1)])
        );

        // Reverse empty sequence
        assert_eq!(eval_str("Reverse(<<>>)").unwrap(), Value::seq([]));

        // Reverse single element
        assert_eq!(
            eval_str("Reverse(<<42>>)").unwrap(),
            Value::seq([Value::int(42)])
        );

        // Reverse with strings
        assert_eq!(
            eval_str(r#"Reverse(<<"a", "b", "c">>)"#).unwrap(),
            Value::seq([Value::string("c"), Value::string("b"), Value::string("a")])
        );
    }

    #[test]
    fn test_eval_front() {
        // Front - all but last element
        assert_eq!(
            eval_str("Front(<<1, 2, 3>>)").unwrap(),
            Value::seq([Value::int(1), Value::int(2)])
        );

        // Front of single element
        assert_eq!(eval_str("Front(<<42>>)").unwrap(), Value::seq([]));

        // Front of empty sequence should error
        assert!(eval_str("Front(<<>>)").is_err());
    }

    #[test]
    fn test_eval_last() {
        // Last element of sequence
        assert_eq!(eval_str("Last(<<1, 2, 3>>)").unwrap(), Value::int(3));

        // Last of single element
        assert_eq!(eval_str("Last(<<42>>)").unwrap(), Value::int(42));

        // Last of empty sequence should error
        assert!(eval_str("Last(<<>>)").is_err());
    }

    #[test]
    fn test_eval_settoseq() {
        // SetToSeq converts set to sequence (order is deterministic but arbitrary)
        let result = eval_str("SetToSeq({1, 2, 3})").unwrap();
        let seq = result.as_seq().unwrap();
        assert_eq!(seq.len(), 3);
        // Check all elements are present
        assert!(seq.contains(&Value::int(1)));
        assert!(seq.contains(&Value::int(2)));
        assert!(seq.contains(&Value::int(3)));

        // Empty set
        assert_eq!(eval_str("SetToSeq({})").unwrap(), Value::seq([]));
    }

    #[test]
    fn test_eval_foldset() {
        // FoldSet(Op, base, S) - sum of set elements
        assert_eq!(
            eval_with_ops("Add(a, b) == a + b", "FoldSet(Add, 0, {1, 2, 3})").unwrap(),
            Value::int(6)
        );

        // Product of set elements
        assert_eq!(
            eval_with_ops("Mul(a, b) == a * b", "FoldSet(Mul, 1, {2, 3, 4})").unwrap(),
            Value::int(24)
        );

        // Empty set returns base
        assert_eq!(
            eval_with_ops("Add(a, b) == a + b", "FoldSet(Add, 100, {})").unwrap(),
            Value::int(100)
        );
    }

    #[test]
    fn test_eval_foldseq() {
        // FoldSeq(Op, base, s) - sum of sequence elements
        assert_eq!(
            eval_with_ops("Add(a, b) == a + b", "FoldSeq(Add, 0, <<1, 2, 3>>)").unwrap(),
            Value::int(6)
        );

        // Build string by concatenation (demonstrates order matters)
        // Note: FoldSeq goes left to right, so we build "cba" from <<a,b,c>> with prepend
        // Actually let's test with subtraction to show order
        assert_eq!(
            eval_with_ops("Sub(a, b) == a - b", "FoldSeq(Sub, 10, <<1, 2, 3>>)").unwrap(),
            Value::int(4) // ((10 - 1) - 2) - 3 = 4
        );

        // Empty sequence returns base
        assert_eq!(
            eval_with_ops("Add(a, b) == a + b", "FoldSeq(Add, 42, <<>>)").unwrap(),
            Value::int(42)
        );
    }

    #[test]
    fn test_eval_foldfunction() {
        // FoldFunction(Op, base, f) - sum of function range values
        assert_eq!(
            eval_with_ops(
                "Add(a, b) == a + b\nf == [x \\in {1,2,3} |-> x * 2]",
                "FoldFunction(Add, 0, f)"
            )
            .unwrap(),
            Value::int(12) // 2 + 4 + 6 = 12
        );

        // Single element function
        assert_eq!(
            eval_with_ops(
                "Add(a, b) == a + b\ng == [x \\in {1} |-> x * 10]",
                "FoldFunction(Add, 5, g)"
            )
            .unwrap(),
            Value::int(15) // 5 + 10 = 15
        );
    }

    #[test]
    fn test_eval_foldfunctiononset() {
        // FoldFunctionOnSet(Op, base, f, S) - fold over function values for keys in S
        assert_eq!(
            eval_with_ops(
                "Add(a, b) == a + b\nf == [x \\in {1,2,3,4} |-> x * 2]",
                "FoldFunctionOnSet(Add, 0, f, {1,2})"
            )
            .unwrap(),
            Value::int(6) // 2 + 4 = 6 (only keys 1 and 2)
        );

        // Full domain
        assert_eq!(
            eval_with_ops(
                "Add(a, b) == a + b\nf == [x \\in {1,2,3} |-> x * 2]",
                "FoldFunctionOnSet(Add, 0, f, {1,2,3})"
            )
            .unwrap(),
            Value::int(12) // 2 + 4 + 6 = 12
        );

        // Empty subset returns base
        assert_eq!(
            eval_with_ops(
                "Add(a, b) == a + b\nf == [x \\in {1,2,3} |-> x * 2]",
                "FoldFunctionOnSet(Add, 100, f, {})"
            )
            .unwrap(),
            Value::int(100)
        );
    }

    #[test]
    fn test_eval_abs() {
        // Abs(n) - absolute value
        assert_eq!(eval_str("Abs(5)").unwrap(), Value::int(5));
        assert_eq!(eval_str("Abs(-5)").unwrap(), Value::int(5));
        assert_eq!(eval_str("Abs(0)").unwrap(), Value::int(0));
        assert_eq!(eval_str("Abs(-100)").unwrap(), Value::int(100));
    }

    #[test]
    fn test_eval_sign() {
        // Sign(n) - returns -1, 0, or 1
        assert_eq!(eval_str("Sign(42)").unwrap(), Value::int(1));
        assert_eq!(eval_str("Sign(-42)").unwrap(), Value::int(-1));
        assert_eq!(eval_str("Sign(0)").unwrap(), Value::int(0));
        assert_eq!(eval_str("Sign(1)").unwrap(), Value::int(1));
        assert_eq!(eval_str("Sign(-1)").unwrap(), Value::int(-1));
    }

    #[test]
    fn test_eval_range_operator() {
        // Range(f) - set of all values in function (image of codomain)
        assert_eq!(
            eval_str(r#"Range([x \in {1,2,3} |-> x * 2])"#).unwrap(),
            Value::set(vec![Value::int(2), Value::int(4), Value::int(6)])
        );

        // Duplicates are collapsed since it's a set
        // Note: [x \in S |-> c] where c is constant is parsed differently,
        // using a record-like constant function syntax. Use x+0 to ensure variable reference.
        assert_eq!(
            eval_str(r#"Range([x \in {1,2,3} |-> 1 + 0])"#).unwrap(),
            Value::set(vec![Value::int(1)])
        );

        // Empty function
        assert_eq!(
            eval_str(r#"Range([x \in {} |-> x + 0])"#).unwrap(),
            Value::empty_set()
        );
    }

    #[test]
    fn test_func_in_func_set() {
        // Function set membership test: [x \in S |-> e] \in [S -> T]

        // Create function: [x \in {1,2} |-> "b0"]
        let func = eval_str(r#"[x \in {1,2} |-> "b0"]"#).unwrap();
        println!("func = {:?}", func);

        // Create function set: [{1,2} -> {"b0", "b1"}]
        let func_set = eval_str(r#"[{1,2} -> {"b0", "b1"}]"#).unwrap();
        println!("func_set = {:?}", func_set);

        // The function should be in the function set
        let result = eval_str(r#"[x \in {1,2} |-> "b0"] \in [{1,2} -> {"b0", "b1"}]"#).unwrap();
        println!("result = {:?}", result);

        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn test_lazy_powerset_membership() {
        // Lazy powerset membership: x \in SUBSET S <==> x \subseteq S
        // This tests that we don't enumerate the full powerset

        // {1} is a subset of {1, 2, 3}
        assert_eq!(
            eval_str(r#"{1} \in SUBSET {1, 2, 3}"#).unwrap(),
            Value::Bool(true)
        );

        // {1, 2} is a subset of {1, 2, 3}
        assert_eq!(
            eval_str(r#"{1, 2} \in SUBSET {1, 2, 3}"#).unwrap(),
            Value::Bool(true)
        );

        // {} is a subset of any set
        assert_eq!(
            eval_str(r#"{} \in SUBSET {1, 2, 3}"#).unwrap(),
            Value::Bool(true)
        );

        // {4} is NOT a subset of {1, 2, 3}
        assert_eq!(
            eval_str(r#"{4} \in SUBSET {1, 2, 3}"#).unwrap(),
            Value::Bool(false)
        );

        // Non-set values are not in SUBSET
        assert_eq!(
            eval_str(r#"1 \in SUBSET {1, 2, 3}"#).unwrap(),
            Value::Bool(false)
        );

        // Test \notin SUBSET
        assert_eq!(
            eval_str(r#"{4} \notin SUBSET {1, 2, 3}"#).unwrap(),
            Value::Bool(true)
        );

        assert_eq!(
            eval_str(r#"{1} \notin SUBSET {1, 2, 3}"#).unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn test_lazy_nested_powerset_membership() {
        // Test nested powersets like SlidingPuzzles: board \in SUBSET (SUBSET Pos)
        // This checks that lazy evaluation works recursively

        // {{1}, {2}} is a set of subsets of {1, 2, 3}
        // So {{1}, {2}} \in SUBSET SUBSET {1, 2, 3} should be TRUE
        assert_eq!(
            eval_str(r#"{{1}, {2}} \in SUBSET SUBSET {1, 2, 3}"#).unwrap(),
            Value::Bool(true)
        );

        // {{1}, {4}} should be FALSE because {4} is not a subset of {1, 2, 3}
        assert_eq!(
            eval_str(r#"{{1}, {4}} \in SUBSET SUBSET {1, 2, 3}"#).unwrap(),
            Value::Bool(false)
        );

        // Empty set of sets is valid
        assert_eq!(
            eval_str(r#"{} \in SUBSET SUBSET {1, 2, 3}"#).unwrap(),
            Value::Bool(true)
        );

        // {1} (not a set of sets) should be FALSE
        assert_eq!(
            eval_str(r#"{{1, 2}, 3} \in SUBSET SUBSET {1, 2, 3}"#).unwrap(),
            Value::Bool(false)
        );

        // Test \notin for nested powerset
        assert_eq!(
            eval_str(r#"{{1}, {4}} \notin SUBSET SUBSET {1, 2, 3}"#).unwrap(),
            Value::Bool(true)
        );
    }

    #[test]
    fn test_lazy_seq_membership() {
        // Lazy Seq membership: x \in Seq(S) <==> x is a sequence AND all elements are in S
        // This tests that we don't enumerate the infinite set Seq(S)

        // Empty sequence is in Seq(S) for any S
        assert_eq!(
            eval_str(r#"<<>> \in Seq({1, 2, 3})"#).unwrap(),
            Value::Bool(true)
        );

        // <<1, 2>> is in Seq({1, 2, 3})
        assert_eq!(
            eval_str(r#"<<1, 2>> \in Seq({1, 2, 3})"#).unwrap(),
            Value::Bool(true)
        );

        // <<1, 4>> is NOT in Seq({1, 2, 3}) because 4 is not in {1, 2, 3}
        assert_eq!(
            eval_str(r#"<<1, 4>> \in Seq({1, 2, 3})"#).unwrap(),
            Value::Bool(false)
        );

        // Non-sequence values are not in Seq(S)
        assert_eq!(
            eval_str(r#"1 \in Seq({1, 2, 3})"#).unwrap(),
            Value::Bool(false)
        );

        // A set is not a sequence
        assert_eq!(
            eval_str(r#"{1, 2} \in Seq({1, 2, 3})"#).unwrap(),
            Value::Bool(false)
        );

        // Test \notin Seq
        assert_eq!(
            eval_str(r#"<<1, 4>> \notin Seq({1, 2, 3})"#).unwrap(),
            Value::Bool(true)
        );

        assert_eq!(
            eval_str(r#"<<1, 2>> \notin Seq({1, 2, 3})"#).unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn test_lazy_nested_seq_membership() {
        // Test nested sequences: Seq(S \times T) pattern from ReadersWriters
        // waiting \in Seq({"read", "write"} \times Actors)

        // Empty sequence is valid
        assert_eq!(
            eval_str(r#"<<>> \in Seq({"a", "b"} \times {1, 2})"#).unwrap(),
            Value::Bool(true)
        );

        // <<<<"a", 1>>, <<"b", 2>>>> is a valid sequence of pairs
        assert_eq!(
            eval_str(r#"<< <<"a", 1>>, <<"b", 2>> >> \in Seq({"a", "b"} \times {1, 2})"#).unwrap(),
            Value::Bool(true)
        );

        // <<<<"c", 1>>>> is NOT valid because "c" is not in {"a", "b"}
        assert_eq!(
            eval_str(r#"<< <<"c", 1>> >> \in Seq({"a", "b"} \times {1, 2})"#).unwrap(),
            Value::Bool(false)
        );

        // <<<<"a", 3>>>> is NOT valid because 3 is not in {1, 2}
        assert_eq!(
            eval_str(r#"<< <<"a", 3>> >> \in Seq({"a", "b"} \times {1, 2})"#).unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn test_lazy_recordset_membership_with_int() {
        // Test RecordSet membership with infinite sets (Int/Nat)
        // This is the pattern used in EWD998: Token == [pos : Node, q : Int, color : Color]
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        let src = r#"
---- MODULE Test ----
EXTENDS Integers
Node == 0..3
Color == {"white", "black"}
Token == [pos : Node, q : Int, color : Color]

ValidToken == [color |-> "black", pos |-> 1, q |-> 0]
InvalidPos == [color |-> "white", pos |-> 5, q |-> 0]
InvalidColor == [color |-> "green", pos |-> 1, q |-> 0]
MissingField == [color |-> "black", pos |-> 1]

TestValid == ValidToken \in Token
TestInvalidPos == InvalidPos \in Token
TestInvalidColor == InvalidColor \in Token
TestMissingField == MissingField \in Token
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        assert!(
            lower_result.errors.is_empty(),
            "Lower errors: {:?}",
            lower_result.errors
        );
        let module = lower_result.module.unwrap();

        let mut ctx = EvalCtx::new();
        ctx.load_module(&module);

        // Valid record should be in Token
        let result = ctx.eval_op("TestValid").expect("TestValid");
        assert_eq!(result, Value::Bool(true), "Valid token should be in Token");

        // Invalid pos (5 not in 0..3) should NOT be in Token
        let result = ctx.eval_op("TestInvalidPos").expect("TestInvalidPos");
        assert_eq!(
            result,
            Value::Bool(false),
            "Invalid pos should not be in Token"
        );

        // Invalid color should NOT be in Token
        let result = ctx.eval_op("TestInvalidColor").expect("TestInvalidColor");
        assert_eq!(
            result,
            Value::Bool(false),
            "Invalid color should not be in Token"
        );

        // Missing field should NOT be in Token
        let result = ctx.eval_op("TestMissingField").expect("TestMissingField");
        assert_eq!(
            result,
            Value::Bool(false),
            "Missing field should not be in Token"
        );
    }

    #[test]
    fn test_func_in_func_set_with_context() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Test with context: operator definitions like Barrier's TypeOK
        let src = r#"
---- MODULE Test ----
CONSTANT N
ProcSet == 1..N
pc == [p \in ProcSet |-> "b0"]
TypeOK == pc \in [ProcSet -> {"b0", "b1"}]
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        // Debug: print the TypeOK operator definition AST
        for unit in &module.units {
            if let tla_core::ast::Unit::Operator(op_def) = &unit.node {
                if op_def.name.node == "TypeOK" {
                    println!("TypeOK body AST: {:?}", op_def.body.node);
                }
            }
        }

        let mut ctx = EvalCtx::new();
        ctx.load_module(&module);
        ctx.bind_mut("N", Value::int(3));

        // First evaluate pc
        let pc_val = ctx.eval_op("pc").expect("pc should evaluate");
        println!("pc = {:?}", pc_val);

        // Then evaluate ProcSet
        let procset = ctx.eval_op("ProcSet").expect("ProcSet should evaluate");
        println!("ProcSet = {:?}", procset);

        // Then evaluate TypeOK
        let type_ok = ctx.eval_op("TypeOK").expect("TypeOK should evaluate");
        println!("TypeOK = {:?}", type_ok);

        assert_eq!(type_ok, Value::Bool(true));
    }
}

#[test]
fn test_eval_tuple_pattern_binding() {
    // Test tuple pattern destructuring: {x + y : <<x, y>> \in S}
    let src = r#"
---- MODULE Test ----
S == {<<1, 2>>, <<3, 4>>}
F == {x + y : <<x, y>> \in S}
====
"#;
    let tree = tla_core::parse_to_syntax_tree(src);
    eprintln!("=== CST for tuple pattern ===");
    // Print CST structure
    fn print_tree(node: &tla_core::SyntaxNode, indent: usize) {
        let text = node.text().to_string();
        let text_len = text.len();
        let kind = node.kind();
        if text_len < 80 {
            eprintln!("{:indent$}{:?}: {:?}", "", kind, text, indent = indent);
        } else {
            eprintln!(
                "{:indent$}{:?}: <{} chars>",
                "",
                kind,
                text_len,
                indent = indent
            );
        }
        for child in node.children() {
            print_tree(&child, indent + 2);
        }
    }
    print_tree(&tree, 0);

    let lower_result = tla_core::lower(tla_core::FileId(0), &tree);
    eprintln!("Lower errors: {:?}", lower_result.errors);
    let module = lower_result.module.unwrap();
    for unit in &module.units {
        if let tla_core::ast::Unit::Operator(def) = &unit.node {
            eprintln!("Op '{}' body: {:?}", def.name.node, def.body.node);
        }
    }

    // Setup eval context
    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    let result = ctx.eval_op("F");
    eprintln!("Tuple pattern binding result: {:?}", result);
    let val = result.unwrap();
    let set = val.as_set().unwrap();
    // {1+2, 3+4} = {3, 7}
    assert_eq!(set.len(), 2);
    assert!(set.contains(&Value::int(3)));
    assert!(set.contains(&Value::int(7)));
}

#[test]
fn test_record_set_membership() {
    // Test: is [smoking |-> FALSE] in [smoking: BOOLEAN]?
    // And: is f \in [S -> [smoking: BOOLEAN]] where f = [x \in S |-> [smoking |-> FALSE]]?
    let src = r#"
---- MODULE Test ----
S == {1, 2}
BoolSet == BOOLEAN
RecSet == [smoking: BOOLEAN]
F == [x \in S |-> [smoking |-> FALSE]]
FuncSet == [S -> [smoking: BOOLEAN]]
RecordInRecSet == [smoking |-> FALSE] \in RecSet
FuncInFuncSet == F \in FuncSet
====
"#;
    let tree = tla_core::parse_to_syntax_tree(src);
    let lower_result = tla_core::lower(tla_core::FileId(0), &tree);
    eprintln!("Lower errors: {:?}", lower_result.errors);
    let module = lower_result.module.unwrap();

    // Print AST for RecSet
    for unit in &module.units {
        if let tla_core::ast::Unit::Operator(def) = &unit.node {
            eprintln!("Op '{}' body: {:?}", def.name.node, def.body.node);
        }
    }

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    // First check BOOLEAN evaluates correctly
    let bool_set = ctx.eval_op("BoolSet").expect("BoolSet");
    eprintln!("BoolSet (BOOLEAN) = {:?}", bool_set);

    // Evaluate each piece for debugging
    let rec_set = ctx.eval_op("RecSet").expect("RecSet");
    eprintln!("RecSet = {:?}", rec_set);

    let f = ctx.eval_op("F").expect("F");
    eprintln!("F = {:?}", f);

    let func_set = ctx.eval_op("FuncSet").expect("FuncSet");
    eprintln!("FuncSet = {:?}", func_set);

    let rec_in_rec_set = ctx.eval_op("RecordInRecSet").expect("RecordInRecSet");
    eprintln!("RecordInRecSet = {:?}", rec_in_rec_set);
    assert_eq!(
        rec_in_rec_set,
        Value::Bool(true),
        "Record should be in record set"
    );

    let func_in_func_set = ctx.eval_op("FuncInFuncSet").expect("FuncInFuncSet");
    eprintln!("FuncInFuncSet = {:?}", func_in_func_set);
    assert_eq!(
        func_in_func_set,
        Value::Bool(true),
        "Function should be in function set"
    );
}

#[test]
fn test_record_dot_access_in_filter() {
    // Test: {r \in S : F[r].smoking = TRUE} where F[r] = [smoking |-> FALSE] for all r
    let src = r#"
---- MODULE Test ----
EXTENDS Integers, FiniteSets
S == {1, 2}
F == [r \in S |-> [smoking |-> FALSE]]
FApply == F[1]
FApplySmoking == F[1].smoking
FilterSet == {r \in S : F[r].smoking = TRUE}
FilterSetFalse == {r \in S : F[r].smoking = FALSE}
FilterCount == Cardinality(FilterSet)
AtMostOne == Cardinality({r \in S : F[r].smoking}) <= 1
====
"#;
    let tree = tla_core::parse_to_syntax_tree(src);
    let lower_result = tla_core::lower(tla_core::FileId(0), &tree);
    assert!(lower_result.errors.is_empty());
    let module = lower_result.module.unwrap();

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    // Verify F[1].smoking evaluates correctly
    let f_apply_smoking = ctx.eval_op("FApplySmoking").expect("FApplySmoking");
    assert_eq!(
        f_apply_smoking,
        Value::Bool(false),
        "F[1].smoking should be FALSE"
    );

    // Verify {r \in S : F[r].smoking = TRUE} is empty
    let filter_set = ctx.eval_op("FilterSet").expect("FilterSet");
    assert_eq!(
        filter_set,
        Value::empty_set(),
        "No elements should satisfy smoking = TRUE"
    );

    // Verify Cardinality is 0
    let filter_count = ctx.eval_op("FilterCount").expect("FilterCount");
    assert_eq!(
        filter_count,
        Value::int(0),
        "Cardinality of empty set should be 0"
    );

    // Verify AtMostOne is TRUE
    let at_most_one = ctx.eval_op("AtMostOne").expect("AtMostOne");
    assert_eq!(at_most_one, Value::Bool(true), "0 <= 1 should be TRUE");
}

#[test]
fn test_toset_with_selectseq() {
    // Test the pattern used in ReadersWriters:
    // ToSet(s) == { s[i] : i \in DOMAIN s }
    // WaitingToRead == { p[2] : p \in ToSet(SelectSeq(waiting, is_read)) }
    use tla_core::{lower, parse_to_syntax_tree, FileId};

    let src = r#"
---- MODULE Test ----
EXTENDS Sequences
VARIABLE waiting

ToSet(s) == { s[i] : i \in DOMAIN s }
is_read(p) == p[1] = "read"
WaitingToRead == { p[2] : p \in ToSet(SelectSeq(waiting, is_read)) }

\* Test operator that returns waiting actors
TestWaiting == WaitingToRead
====
"#;
    let tree = parse_to_syntax_tree(src);
    let lower_result = lower(FileId(0), &tree);
    if !lower_result.errors.is_empty() {
        eprintln!("Lower errors: {:?}", lower_result.errors);
    }
    let module = lower_result.module.expect("module should parse");

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    // Test with empty waiting queue
    ctx.bind_mut("waiting".to_string(), Value::seq([]));
    let result = ctx.eval_op("TestWaiting");
    assert!(
        result.is_ok(),
        "WaitingToRead should evaluate with empty queue: {:?}",
        result
    );
    assert_eq!(result.unwrap(), Value::empty_set());

    // Test with one read request
    ctx.bind_mut(
        "waiting".to_string(),
        Value::seq([Value::tuple([Value::string("read"), Value::int(1)])]),
    );
    let result = ctx.eval_op("TestWaiting");
    assert!(
        result.is_ok(),
        "WaitingToRead should evaluate with one read: {:?}",
        result
    );
    let waiting_set = result.unwrap().as_set().unwrap().clone();
    assert!(
        waiting_set.contains(&Value::int(1)),
        "Actor 1 should be in WaitingToRead"
    );

    // Test with mixed read/write requests
    ctx.bind_mut(
        "waiting".to_string(),
        Value::seq([
            Value::tuple([Value::string("read"), Value::int(1)]),
            Value::tuple([Value::string("write"), Value::int(2)]),
            Value::tuple([Value::string("read"), Value::int(3)]),
        ]),
    );
    let result = ctx.eval_op("TestWaiting");
    assert!(
        result.is_ok(),
        "WaitingToRead should evaluate with mixed requests: {:?}",
        result
    );
    let waiting_set = result.unwrap().as_set().unwrap().clone();
    assert_eq!(waiting_set.len(), 2, "Should have 2 read requests");
    assert!(waiting_set.contains(&Value::int(1)));
    assert!(waiting_set.contains(&Value::int(3)));
    // Actor 2 is a write request, not a read
    assert!(!waiting_set.contains(&Value::int(2)));
}

#[test]
fn test_cartesian_product_membership_nat_int() {
    // Test <<2, -3>> \in Nat \X Int - should be TRUE since 2 \in Nat and -3 \in Int
    let src = r#"
---- MODULE Test ----
EXTENDS Naturals, Integers
CrossProduct == Nat \X Int
TupleVal == <<2, -3>>
Result == TupleVal \in CrossProduct
====
"#;
    let tree = tla_core::parse_to_syntax_tree(src);
    let lower_result = tla_core::lower(tla_core::FileId(0), &tree);
    eprintln!("Lower errors: {:?}", lower_result.errors);
    let module = lower_result.module.expect("Module should lower");

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    // Debug: check what CrossProduct evaluates to
    let cross_result = ctx.eval_op("CrossProduct");
    eprintln!("CrossProduct = {:?}", cross_result);

    // Debug: check what TupleVal evaluates to
    let tuple_result = ctx.eval_op("TupleVal");
    eprintln!("TupleVal = {:?}", tuple_result);

    // Check the membership result
    let result = ctx.eval_op("Result");
    eprintln!("Result = {:?}", result);
    assert!(result.is_ok(), "Should not error: {:?}", result);
    assert_eq!(
        result.unwrap(),
        Value::Bool(true),
        "<<2, -3>> should be in Nat \\X Int"
    );
}

#[test]
fn test_let_preserves_local_stack_bindings() {
    // Regression test: LET expressions must preserve local_stack bindings from enclosing scopes.
    // This was a bug where LET created a new context with empty local_stack, causing
    // "Undefined variable" errors when the LET body referenced variables bound by
    // enclosing operators or quantifiers (e.g., EXISTS-bound vars, operator parameters).
    //
    // Pattern that triggered the bug (from Chameneos spec):
    //   Op(cid) == ... LET v == f[cid][1] IN [f EXCEPT ![cid] = <<v, ...>>]
    //   Next == \E c \in IDs : Op(c)
    let src = r#"
---- MODULE Test ----
EXTENDS Integers
Op(x) == LET y == x + 1 IN y * 2
Result == Op(5)
====
"#;
    let tree = tla_core::parse_to_syntax_tree(src);
    let lower_result = tla_core::lower(tla_core::FileId(0), &tree);
    let module = lower_result.module.expect("Module should lower");

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    // Op(5) should evaluate: y = 5 + 1 = 6, result = 6 * 2 = 12
    let result = ctx.eval_op("Result");
    assert!(result.is_ok(), "Should not error: {:?}", result);
    assert_eq!(result.unwrap(), Value::int(12));
}

#[test]
fn test_bitwise_xor() {
    // Test the ^^ (XOR) operator from the Bitwise module
    // XOR truth table: 0 ^^ 0 = 0, 0 ^^ 1 = 1, 1 ^^ 0 = 1, 1 ^^ 1 = 0
    // Example: 5 ^^ 3 = 101 ^^ 011 = 110 = 6
    let src = r#"
---- MODULE Test ----
EXTENDS Integers, Bitwise
Xor1 == 5 ^^ 3      \* 5 XOR 3 = 6
Xor2 == 0 ^^ 0      \* 0
Xor3 == 12 ^^ 12    \* Self-XOR = 0
Xor4 == 10 ^^ 5     \* 1010 ^^ 0101 = 1111 = 15
====
"#;
    let tree = tla_core::parse_to_syntax_tree(src);
    let lower_result = tla_core::lower(tla_core::FileId(0), &tree);
    let module = lower_result.module.expect("Module should lower");

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    assert_eq!(ctx.eval_op("Xor1").unwrap(), Value::int(6));
    assert_eq!(ctx.eval_op("Xor2").unwrap(), Value::int(0));
    assert_eq!(ctx.eval_op("Xor3").unwrap(), Value::int(0));
    assert_eq!(ctx.eval_op("Xor4").unwrap(), Value::int(15));
}

#[test]
fn test_bitwise_and() {
    // Test the & (AND) operator from the Bitwise module
    let src = r#"
---- MODULE Test ----
EXTENDS Integers, Bitwise
And1 == 5 & 3      \* 101 & 011 = 001 = 1
And2 == 12 & 10    \* 1100 & 1010 = 1000 = 8
And3 == 7 & 0      \* 0
And4 == 255 & 255  \* 255
====
"#;
    let tree = tla_core::parse_to_syntax_tree(src);
    let lower_result = tla_core::lower(tla_core::FileId(0), &tree);
    let module = lower_result.module.expect("Module should lower");

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    assert_eq!(ctx.eval_op("And1").unwrap(), Value::int(1));
    assert_eq!(ctx.eval_op("And2").unwrap(), Value::int(8));
    assert_eq!(ctx.eval_op("And3").unwrap(), Value::int(0));
    assert_eq!(ctx.eval_op("And4").unwrap(), Value::int(255));
}

#[test]
fn test_bitwise_or() {
    // Test the | (OR) operator from the Bitwise module
    let src = r#"
---- MODULE Test ----
EXTENDS Integers, Bitwise
Or1 == 5 | 3      \* 101 | 011 = 111 = 7
Or2 == 12 | 10    \* 1100 | 1010 = 1110 = 14
Or3 == 7 | 0      \* 7
Or4 == 0 | 0      \* 0
====
"#;
    let tree = tla_core::parse_to_syntax_tree(src);
    let lower_result = tla_core::lower(tla_core::FileId(0), &tree);
    let module = lower_result.module.expect("Module should lower");

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    assert_eq!(ctx.eval_op("Or1").unwrap(), Value::int(7));
    assert_eq!(ctx.eval_op("Or2").unwrap(), Value::int(14));
    assert_eq!(ctx.eval_op("Or3").unwrap(), Value::int(7));
    assert_eq!(ctx.eval_op("Or4").unwrap(), Value::int(0));
}

#[test]
fn test_bitwise_shiftr() {
    // Test shiftR(n, pos) - logical right shift
    let src = r#"
---- MODULE Test ----
EXTENDS Integers, Bitwise
Shift1 == shiftR(8, 1)   \* 1000 >> 1 = 0100 = 4
Shift2 == shiftR(8, 2)   \* 1000 >> 2 = 0010 = 2
Shift3 == shiftR(15, 2)  \* 1111 >> 2 = 0011 = 3
Shift4 == shiftR(1, 1)   \* 1 >> 1 = 0
====
"#;
    let tree = tla_core::parse_to_syntax_tree(src);
    let lower_result = tla_core::lower(tla_core::FileId(0), &tree);
    let module = lower_result.module.expect("Module should lower");

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    assert_eq!(ctx.eval_op("Shift1").unwrap(), Value::int(4));
    assert_eq!(ctx.eval_op("Shift2").unwrap(), Value::int(2));
    assert_eq!(ctx.eval_op("Shift3").unwrap(), Value::int(3));
    assert_eq!(ctx.eval_op("Shift4").unwrap(), Value::int(0));
}

// ============================================================================
// OP_RESULT_CACHE Tests (Issue #16)
//
// These tests verify the operator result caching mechanism works correctly.
// The cache key is (op_name, args_fingerprint) and validity depends on
// state_fp and next_state_fp matching.
//
// Note: Caching only activates for operators WITH parameters (non-zero arity).
// ============================================================================

#[test]
fn test_op_cache_hit_same_args_same_state() {
    // Test: Calling the same operator with same args returns cached result.
    // Evidence: Cache length should be 1 after two calls.
    op_result_cache_clear();

    let src = r#"
---- MODULE Test ----
VARIABLE x
Add(a, b) == a + b
Test == Add(1, 2) + Add(1, 2)
====
"#;
    let tree = tla_core::parse_to_syntax_tree(src);
    let lower_result = tla_core::lower(tla_core::FileId(0), &tree);
    let module = lower_result.module.expect("Module should lower");

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    // Set up state environment
    let state = vec![Value::int(10)]; // x = 10
    ctx.register_var("x");
    ctx.bind_state_array(&state);

    let result = ctx.eval_op("Test").unwrap();
    assert_eq!(result, Value::int(6)); // 3 + 3

    // Add(1, 2) was called twice but should only have one cache entry
    // (same op, same args, same state context)
    assert_eq!(op_result_cache_len(), 1, "Cache should have exactly 1 entry for repeated identical calls");
}

#[test]
fn test_op_cache_miss_different_args() {
    // Test: Different arguments create different cache entries.
    op_result_cache_clear();

    let src = r#"
---- MODULE Test ----
VARIABLE x
Add(a, b) == a + b
Test == Add(1, 2) + Add(2, 3)
====
"#;
    let tree = tla_core::parse_to_syntax_tree(src);
    let lower_result = tla_core::lower(tla_core::FileId(0), &tree);
    let module = lower_result.module.expect("Module should lower");

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    let state = vec![Value::int(10)];
    ctx.register_var("x");
    ctx.bind_state_array(&state);

    let result = ctx.eval_op("Test").unwrap();
    assert_eq!(result, Value::int(8)); // 3 + 5

    // Two different calls: Add(1,2) and Add(2,3) should create 2 cache entries
    assert_eq!(op_result_cache_len(), 2, "Different args should create different cache entries");
}

#[test]
fn test_op_cache_miss_state_change() {
    // Test: When state_env changes, cached results for state-dependent operators
    // should be invalidated (cache miss, return correct result).
    //
    // Note: Caching only happens for operators WITH parameters.
    op_result_cache_clear();

    let src = r#"
---- MODULE Test ----
VARIABLE x
GetX(dummy) == x + dummy
Test1 == GetX(0)
Test2 == GetX(0)
====
"#;
    let tree = tla_core::parse_to_syntax_tree(src);
    let lower_result = tla_core::lower(tla_core::FileId(0), &tree);
    let module = lower_result.module.expect("Module should lower");

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);
    ctx.register_var("x");

    // First evaluation with x = 10
    let state1 = vec![Value::int(10)];
    ctx.bind_state_array(&state1);
    let result1 = ctx.eval_op("Test1").unwrap();
    assert_eq!(result1, Value::int(10)); // x + 0 = 10

    let cache_len_after_first = op_result_cache_len();
    assert!(cache_len_after_first >= 1, "Cache should have entry after first call");

    // Second evaluation with different state x = 20
    let state2 = vec![Value::int(20)];
    ctx.bind_state_array(&state2);
    let result2 = ctx.eval_op("Test2").unwrap();
    assert_eq!(result2, Value::int(20)); // x + 0 = 20

    // The result being correct (20 not cached 10) proves cache invalidation works.
}

#[test]
fn test_op_cache_primed_operator_next_state_miss() {
    // Test: Operators using primed variables (x') must invalidate when
    // next_state_env changes.
    op_result_cache_clear();

    let src = r#"
---- MODULE Test ----
VARIABLE x
GetNextX(dummy) == x' + dummy
Test1 == GetNextX(0)
Test2 == GetNextX(0)
====
"#;
    let tree = tla_core::parse_to_syntax_tree(src);
    let lower_result = tla_core::lower(tla_core::FileId(0), &tree);
    let module = lower_result.module.expect("Module should lower");

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);
    ctx.register_var("x");

    // Set up current state
    let state = vec![Value::int(10)];
    ctx.bind_state_array(&state);

    // First evaluation with next_state x' = 100
    let next_state1 = vec![Value::int(100)];
    ctx.bind_next_state_array(&next_state1);
    let result1 = ctx.eval_op("Test1").unwrap();
    assert_eq!(result1, Value::int(100)); // x' + 0 = 100

    // Second evaluation with different next_state x' = 200
    let next_state2 = vec![Value::int(200)];
    ctx.bind_next_state_array(&next_state2);
    let result2 = ctx.eval_op("Test2").unwrap();
    assert_eq!(result2, Value::int(200), "Primed operator must return new next_state value");

    // The key correctness check: result2 should be 200, not cached 100.
}

#[test]
fn test_op_cache_non_primed_ignores_next_state() {
    // Test: Operators WITHOUT primed variables should NOT invalidate when
    // next_state_env changes. This is the key optimization for bosco.
    op_result_cache_clear();

    let src = r#"
---- MODULE Test ----
VARIABLE x
GetX(dummy) == x + dummy
Test1 == GetX(0)
Test2 == GetX(0)
====
"#;
    let tree = tla_core::parse_to_syntax_tree(src);
    let lower_result = tla_core::lower(tla_core::FileId(0), &tree);
    let module = lower_result.module.expect("Module should lower");

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);
    ctx.register_var("x");

    // Set up current state (stays constant)
    let state = vec![Value::int(42)];
    ctx.bind_state_array(&state);

    // First evaluation with next_state x' = 100
    let next_state1 = vec![Value::int(100)];
    ctx.bind_next_state_array(&next_state1);
    let result1 = ctx.eval_op("Test1").unwrap();
    assert_eq!(result1, Value::int(42)); // x + 0 = 42

    let cache_len_after_first = op_result_cache_len();

    // Second evaluation with DIFFERENT next_state x' = 999
    // GetX doesn't use x', so this should be a cache HIT
    let next_state2 = vec![Value::int(999)];
    ctx.bind_next_state_array(&next_state2);
    let result2 = ctx.eval_op("Test2").unwrap();
    assert_eq!(result2, Value::int(42)); // Still 42

    let cache_len_after_second = op_result_cache_len();

    // Cache length should NOT increase - same entry reused
    assert_eq!(
        cache_len_after_first, cache_len_after_second,
        "Non-primed operator should cache-hit when only next_state changes"
    );
}

#[test]
fn test_op_cache_size_cap_clears_at_10000() {
    // Test: Cache clears when size exceeds 10000 entries.
    // This prevents unbounded memory growth.
    op_result_cache_clear();

    let state = vec![Value::int(0)];

    // Add 10001 different cache entries by calling Identity(i) with different i
    // The cache clearing happens at the START of top-level eval when len > 10000
    for i in 0..=10001 {
        let expr_src = format!(
            "---- MODULE Test{} ----\nVARIABLE x\nIdentity(n) == n\nOp == Identity({})\n====",
            i, i
        );
        let tree = tla_core::parse_to_syntax_tree(&expr_src);
        let lower_result = tla_core::lower(tla_core::FileId(0), &tree);
        let module = lower_result.module.expect("Module should lower");
        let mut eval_ctx = EvalCtx::new();
        eval_ctx.load_module(&module);
        eval_ctx.register_var("x");
        eval_ctx.bind_state_array(&state);
        let _ = eval_ctx.eval_op("Op");
    }

    // After exceeding 10000, the cache should have been cleared at some point
    let final_len = op_result_cache_len();
    assert!(
        final_len < 10001,
        "Cache should have been cleared when exceeding 10000 entries, but has {} entries",
        final_len
    );
}

#[test]
fn test_op_cache_correctness_with_complex_args() {
    // Test: Cache correctly handles complex argument values (sets).
    op_result_cache_clear();

    let src = r#"
---- MODULE Test ----
VARIABLE x
SetSize(s) == Cardinality(s)
Test1 == SetSize({1, 2, 3})
Test2 == SetSize({1, 2, 3})
Test3 == SetSize({4, 5})
====
"#;
    let tree = tla_core::parse_to_syntax_tree(src);
    let lower_result = tla_core::lower(tla_core::FileId(0), &tree);
    let module = lower_result.module.expect("Module should lower");

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    let state = vec![Value::int(0)];
    ctx.register_var("x");
    ctx.bind_state_array(&state);

    let result1 = ctx.eval_op("Test1").unwrap();
    let result2 = ctx.eval_op("Test2").unwrap();
    let result3 = ctx.eval_op("Test3").unwrap();

    assert_eq!(result1, Value::int(3));
    assert_eq!(result2, Value::int(3));
    assert_eq!(result3, Value::int(2));

    // Test1 and Test2 should share cache (same set)
    // Test3 should have separate entry (different set)
    assert_eq!(
        op_result_cache_len(),
        2,
        "Same set args should share cache entry, different sets should not"
    );
}

#[test]
fn test_op_cache_expr_has_any_prime_detection() {
    // Test: Verify expr_has_any_prime correctly identifies primed expressions.
    // This is crucial for cache correctness.

    // Test a simple primed variable
    let src1 = r#"
---- MODULE Test ----
VARIABLE x
HasPrime == x'
====
"#;
    let tree1 = tla_core::parse_to_syntax_tree(src1);
    let lower_result1 = tla_core::lower(tla_core::FileId(0), &tree1);
    let module1 = lower_result1.module.expect("Module should lower");

    // Find the HasPrime operator and check its body
    let def1 = module1
        .units
        .iter()
        .find_map(|u| {
            if let tla_core::ast::Unit::Operator(d) = &u.node {
                if d.name.node == "HasPrime" {
                    return Some(d);
                }
            }
            None
        })
        .expect("HasPrime operator should exist");
    assert!(
        expr_has_any_prime(&def1.body.node),
        "x' should be detected as having primes"
    );

    // Test an expression without primes
    let src2 = r#"
---- MODULE Test ----
VARIABLE x
NoPrime == x + 1
====
"#;
    let tree2 = tla_core::parse_to_syntax_tree(src2);
    let lower_result2 = tla_core::lower(tla_core::FileId(0), &tree2);
    let module2 = lower_result2.module.expect("Module should lower");

    let def2 = module2
        .units
        .iter()
        .find_map(|u| {
            if let tla_core::ast::Unit::Operator(d) = &u.node {
                if d.name.node == "NoPrime" {
                    return Some(d);
                }
            }
            None
        })
        .expect("NoPrime operator should exist");
    assert!(
        !expr_has_any_prime(&def2.body.node),
        "x + 1 should NOT be detected as having primes"
    );
}

#[test]
fn test_op_cache_nested_operator_calls() {
    // Test: Cache works correctly with nested operator calls.
    op_result_cache_clear();

    let src = r#"
---- MODULE Test ----
VARIABLE x
Inner(n) == n * 2
Outer(n) == Inner(n) + Inner(n)
Test == Outer(5)
====
"#;
    let tree = tla_core::parse_to_syntax_tree(src);
    let lower_result = tla_core::lower(tla_core::FileId(0), &tree);
    let module = lower_result.module.expect("Module should lower");

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    let state = vec![Value::int(0)];
    ctx.register_var("x");
    ctx.bind_state_array(&state);

    let result = ctx.eval_op("Test").unwrap();
    assert_eq!(result, Value::int(20)); // (5*2) + (5*2) = 10 + 10 = 20

    // Inner(5) called twice inside Outer - should hit cache
    // Outer(5) called once
    // Expected cache entries: Inner(5), Outer(5) = 2
    assert_eq!(
        op_result_cache_len(),
        2,
        "Nested calls should cache both inner and outer operators"
    );
}
