//! Compiled guards for fast evaluation without full AST traversal.
//!
//! This module provides a specialized representation for guard expressions
//! that can be evaluated efficiently in the BFS hot path. Instead of
//! pattern-matching on the full AST and doing string lookups, compiled
//! guards use integer indices and specialized evaluation paths.
//!
//! # Motivation
//!
//! In the bcastFolklore benchmark, guards like `pc[self] = "V1"` are evaluated
//! millions of times. The full `eval()` function does:
//! 1. Pattern matching on Expr enum (50+ variants)
//! 2. String lookups for variable names
//! 3. Recursive calls through Spanned wrappers
//!
//! Compiled guards reduce this to:
//! 1. Direct array access by integer index
//! 2. No string comparison
//! 3. Minimal branching
//!
//! # Supported Patterns
//!
//! Guards:
//! - `var = const` / `var # const` - Variable equals/not-equals constant
//! - `f[k] = v` / `f[k] # v` - Function application equals/not-equals constant
//! - `x < y`, `x <= y`, `x > y`, `x >= y` - Integer comparisons
//! - `a \in S`, `a \notin S` - Set membership/non-membership
//! - `A \subseteq B` - Subset relation
//! - `a /\ b`, `a \/ b`, `~a` - Boolean combinations
//! - `a => b`, `a <=> b` - Implication and equivalence
//!
//! Expressions:
//! - Arithmetic: `+`, `-`, `*`, `\div`, `%` - Integer operations
//! - Sets: `\cup`, `\cap`, `\` - Union, intersection, difference
//! - EXCEPT: `[f EXCEPT ![k] = v]` - Function update
//! - Tuples: `<<a, b, ...>>` and set enumerations `{a, b, ...}`
//!
//! Complex expressions fall back to the standard `eval()` function.

use crate::enumerate::{expr_contains_any_prime, LocalScope};
use crate::error::{EvalError, EvalResult};
use crate::eval::{apply_substitutions, eval, Env, EvalCtx, OpEnv};
use crate::state::{compute_diff_fingerprint, ArrayState, DiffChanges, DiffSuccessor};
use crate::value::{
    intern_string, FuncValue, IntIntervalFunc, RecordBuilder, SetBuilder, SortedSet,
};
use crate::var_index::{VarIndex, VarRegistry};
use crate::Value;
use num_integer::Integer;
use num_traits::{ToPrimitive, Zero};
use std::sync::Arc;
use tla_core::ast::{Expr, Substitution};
use tla_core::Spanned;

use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::collections::HashSet;

// Compilation depth limit to prevent infinite recursion from recursive operators
const MAX_COMPILE_DEPTH: u32 = 100;
thread_local! {
    static COMPILE_DEPTH: Cell<u32> = const { Cell::new(0) };
}

// Thread-local set of variables with satisfied SUBSET bounds.
//
// When SUBSET bounds optimization generates valid subsets, the bound constraints
// (x ⊆ upper and lower ⊆ x) are satisfied by construction. We track the variable
// names here so guard evaluation can skip redundant is_subset checks.
//
// Entry format: (var_name, is_upper_bound) - true for x⊆upper, false for lower⊆x
thread_local! {
    static SATISFIED_UPPER_BOUNDS: RefCell<HashSet<Arc<str>>> = RefCell::new(HashSet::new());
    static SATISFIED_LOWER_BOUNDS: RefCell<HashSet<Arc<str>>> = RefCell::new(HashSet::new());
}

/// Register that a variable has satisfied SUBSET bounds.
/// Call this before enumerating subsets that satisfy the given bounds.
///
/// # Arguments
/// * `var_name` - The EXISTS-bound variable name
/// * `has_upper` - If true, x ⊆ upper_bound is satisfied by construction
/// * `has_lower` - If true, lower_bound ⊆ x is satisfied by construction
pub fn register_subset_bounds(var_name: &str, has_upper: bool, has_lower: bool) {
    register_subset_bounds_interned(&intern_string(var_name), has_upper, has_lower);
}

/// Register SUBSET bounds using an already-interned variable name.
pub fn register_subset_bounds_interned(var_name: &Arc<str>, has_upper: bool, has_lower: bool) {
    if has_upper {
        SATISFIED_UPPER_BOUNDS.with(|bounds| {
            bounds.borrow_mut().insert(Arc::clone(var_name));
        });
    }
    if has_lower {
        SATISFIED_LOWER_BOUNDS.with(|bounds| {
            bounds.borrow_mut().insert(Arc::clone(var_name));
        });
    }
}

/// Unregister SUBSET bounds for a variable.
/// Call this after finishing enumeration.
pub fn unregister_subset_bounds(var_name: &str) {
    unregister_subset_bounds_interned(&intern_string(var_name));
}

/// Unregister SUBSET bounds using an already-interned variable name.
pub fn unregister_subset_bounds_interned(var_name: &Arc<str>) {
    SATISFIED_UPPER_BOUNDS.with(|bounds| {
        bounds.borrow_mut().remove(var_name.as_ref());
    });
    SATISFIED_LOWER_BOUNDS.with(|bounds| {
        bounds.borrow_mut().remove(var_name.as_ref());
    });
}

/// Apply nested EXCEPT: [f EXCEPT ![k1][k2]...![kn] = new_val]
///
/// For a path [k1, k2, ..., kn], the semantics are:
/// - Navigate to f[k1][k2]...[k(n-1)]
/// - Apply [that EXCEPT ![kn] = new_val]
/// - Work backwards rebuilding each level
fn apply_nested_except(base: Value, keys: &[Value], new_val: Value) -> EvalResult<Value> {
    // Must have at least 2 keys for nested EXCEPT
    debug_assert!(keys.len() >= 2, "ExceptNested requires at least 2 keys");

    // Navigate down to collect intermediate values
    let mut path_values: Vec<Value> = Vec::with_capacity(keys.len());
    path_values.push(base);

    for key in keys.iter().take(keys.len() - 1) {
        let current = path_values.last().unwrap();
        let next_val = get_at_key(current, key)?;
        path_values.push(next_val);
    }

    // Start with the innermost update
    let last_key = &keys[keys.len() - 1];
    let innermost = path_values.pop().unwrap();
    let mut result = apply_single_except(innermost, last_key, new_val)?;

    // Work backwards, applying except at each level
    for i in (0..keys.len() - 1).rev() {
        let outer = path_values.pop().unwrap();
        result = apply_single_except(outer, &keys[i], result)?;
    }

    Ok(result)
}

/// Get value at a key from a function/sequence/record
fn get_at_key(container: &Value, key: &Value) -> EvalResult<Value> {
    match container {
        Value::Func(f) => f.apply(key).cloned().ok_or_else(|| EvalError::NotInDomain {
            arg: format!("{key:?}"),
            span: None,
        }),
        Value::IntFunc(f) => f.apply(key).cloned().ok_or_else(|| EvalError::NotInDomain {
            arg: format!("{key:?}"),
            span: None,
        }),
        _ if container.as_seq_or_tuple_elements().is_some() => {
            let s = container.as_seq_or_tuple_elements().unwrap();
            if let Some(idx) = key.as_i64() {
                let index = (idx - 1) as usize; // TLA+ is 1-indexed
                s.get(index).cloned().ok_or(EvalError::IndexOutOfBounds {
                    index: idx,
                    len: s.len(),
                    span: None,
                })
            } else {
                Err(EvalError::TypeError {
                    expected: "Int",
                    got: key.type_name(),
                    span: None,
                })
            }
        }
        Value::Record(r) => {
            if let Value::String(s) = key {
                r.get(s.as_ref())
                    .cloned()
                    .ok_or_else(|| EvalError::NoSuchField {
                        field: s.to_string(),
                        span: None,
                    })
            } else {
                Err(EvalError::TypeError {
                    expected: "String",
                    got: key.type_name(),
                    span: None,
                })
            }
        }
        _ => Err(EvalError::TypeError {
            expected: "function, sequence, or record",
            got: container.type_name(),
            span: None,
        }),
    }
}

/// Apply single-level EXCEPT: [container EXCEPT ![key] = new_val]
fn apply_single_except(container: Value, key: &Value, new_val: Value) -> EvalResult<Value> {
    match container {
        Value::Func(f) => Ok(Value::Func(f.except(key.clone(), new_val))),
        Value::IntFunc(f) => Ok(Value::IntFunc(f.except(key, new_val))),
        Value::Seq(s) => {
            if let Some(idx) = key.as_i64() {
                let index = (idx - 1) as usize;
                if index >= s.len() {
                    return Err(EvalError::IndexOutOfBounds {
                        index: idx,
                        len: s.len(),
                        span: None,
                    });
                }
                // Short-circuit: if value unchanged, return original without cloning
                if s[index] == new_val {
                    return Ok(Value::Seq(s));
                }
                let mut new_vec: Vec<Value> = s.iter().cloned().collect();
                new_vec[index] = new_val;
                Ok(Value::Seq(new_vec.into()))
            } else {
                Err(EvalError::TypeError {
                    expected: "Int",
                    got: key.type_name(),
                    span: None,
                })
            }
        }
        Value::Tuple(s) => {
            if let Some(idx) = key.as_i64() {
                let index = (idx - 1) as usize;
                if index >= s.len() {
                    return Err(EvalError::IndexOutOfBounds {
                        index: idx,
                        len: s.len(),
                        span: None,
                    });
                }
                // Short-circuit: if value unchanged, return original without cloning
                if s[index] == new_val {
                    return Ok(Value::Tuple(s));
                }
                let mut new_vec: Vec<Value> = s.iter().cloned().collect();
                new_vec[index] = new_val;
                Ok(Value::Tuple(new_vec.into()))
            } else {
                Err(EvalError::TypeError {
                    expected: "Int",
                    got: key.type_name(),
                    span: None,
                })
            }
        }
        Value::Record(r) => {
            if let Value::String(s) = key {
                Ok(Value::Record(r.update(s.clone(), new_val)))
            } else {
                Err(EvalError::TypeError {
                    expected: "String",
                    got: key.type_name(),
                    span: None,
                })
            }
        }
        _ => Err(EvalError::TypeError {
            expected: "function, sequence, or record",
            got: container.type_name(),
            span: None,
        }),
    }
}

/// Check if a subset relation is already satisfied by SUBSET bounds.
/// Returns true if the guard can be skipped (i.e., guaranteed true).
fn is_subset_satisfied(left: &CompiledExpr, right: &CompiledExpr) -> bool {
    // Check if left is a bound variable with upper bound (x ⊆ upper)
    if let CompiledExpr::LocalVar { name, .. } = left {
        if SATISFIED_UPPER_BOUNDS.with(|b| b.borrow().contains(name.as_ref())) {
            return true;
        }
    }

    // Check if right is a bound variable with lower bound (lower ⊆ x)
    if let CompiledExpr::LocalVar { name, .. } = right {
        if SATISFIED_LOWER_BOUNDS.with(|b| b.borrow().contains(name.as_ref())) {
            return true;
        }
    }

    false
}

/// Fast subset check with early rejection.
///
/// Uses size comparison for quick rejection, then element-wise contains() for small sets.
/// For sets with <= 16 elements, iterating with contains() is faster than is_subset()
/// because it avoids iterator infrastructure overhead.
#[inline]
fn fast_is_subset(a: &SortedSet, b: &SortedSet) -> bool {
    // Quick rejection: if A is larger than B, A cannot be a subset of B
    if a.len() > b.len() {
        return false;
    }

    // Empty set is a subset of any set
    if a.is_empty() {
        return true;
    }

    // For small sets, use contains() loop which avoids iterator overhead
    if a.len() <= 16 {
        for elem in a.iter() {
            if !b.contains(elem) {
                return false;
            }
        }
        return true;
    }

    // Fall back to standard is_subset for larger sets
    a.is_subset(b)
}

/// Comparison operator for integer comparisons
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Lt,  // <
    Le,  // <=
    Gt,  // >
    Ge,  // >=
    Eq,  // =
    Neq, // #
}

/// Binary operation for compiled expressions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    /// Set union: a \cup b
    SetUnion,
    /// Set difference: a \ b
    SetDiff,
    /// Set intersection: a \cap b
    SetIntersect,
    /// Integer addition: a + b
    IntAdd,
    /// Integer subtraction: a - b
    IntSub,
    /// Integer multiplication: a * b
    IntMul,
    /// Integer division: a \div b
    IntDiv,
    /// Integer modulo: a % b
    IntMod,
}

/// Which state a variable reference should read from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateRef {
    Current,
    Next,
}

/// Compiled expression for use within guards and assignments.
///
/// Represents a sub-expression that produces a value for comparison or assignment.
#[derive(Debug, Clone)]
pub enum CompiledExpr {
    /// Constant value (pre-evaluated during compilation)
    Const(Value),

    /// State variable by index (O(1) lookup from ArrayState)
    StateVar { state: StateRef, idx: VarIndex },

    /// Local binding from EXISTS quantifier (O(1) lookup from EvalCtx local_stack)
    ///
    /// The `depth` field indicates how many bindings from the end of local_stack
    /// this variable is located. depth=0 means most recent binding.
    /// At eval time: local_stack[local_stack.len() - 1 - depth]
    LocalVar { name: Arc<str>, depth: u8 },

    /// Function application: `state_var[key]` where key is a compiled expression
    FuncApp {
        state: StateRef,
        func_var: VarIndex,
        key: Box<CompiledExpr>,
    },

    /// Specialized: `state_var[local_var]` where state_var is known to be IntFunc at compile time.
    /// Fuses LocalVar lookup + IntFunc array indexing to avoid recursive evaluation and cloning.
    /// This is a common pattern in PlusCal-generated specs: `pc[self]`, `c[i]`, etc.
    FuncAppIntFuncLocalVar {
        state: StateRef,
        func_var: VarIndex,
        key_depth: u8,
    },

    /// Binary operation: left op right
    BinOp {
        left: Box<CompiledExpr>,
        op: BinOp,
        right: Box<CompiledExpr>,
    },

    /// EXCEPT: [f EXCEPT ![key] = value]
    Except {
        func: Box<CompiledExpr>,
        key: Box<CompiledExpr>,
        value: Box<CompiledExpr>,
    },

    /// Nested EXCEPT: `[f EXCEPT ![key1][key2]...![keyN] = value]`
    /// For path `![k1][k2]` with value v, semantics are:
    /// `[f EXCEPT ![k1] = [f[k1] EXCEPT ![k2] = v]]`
    ExceptNested {
        func: Box<CompiledExpr>,
        keys: Vec<CompiledExpr>,
        value: Box<CompiledExpr>,
    },

    /// Set enumeration: {elem1, elem2, ...}
    SetEnum(Vec<CompiledExpr>),

    /// Tuple: <<elem1, elem2, ...>>
    Tuple(Vec<CompiledExpr>),

    /// Dynamic function application: `func[key]` where func is a compiled expression
    /// (not a direct state variable). Used when func is a local variable or computed value.
    DynFuncApp {
        func: Box<CompiledExpr>,
        key: Box<CompiledExpr>,
    },

    /// Boolean negation: ~expr
    Not(Box<CompiledExpr>),

    /// Conditional expression: IF cond THEN then_expr ELSE else_expr
    IfThenElse {
        condition: Box<CompiledExpr>,
        then_branch: Box<CompiledExpr>,
        else_branch: Box<CompiledExpr>,
    },

    /// Function definition: [var_name \in domain |-> body]
    /// Constructs a function by evaluating body for each element in domain.
    /// This is used for expressions like `[r \in Proc |-> IF r = p THEN ... ELSE ...]`
    FuncDef {
        var_name: Arc<str>,
        domain: Box<CompiledExpr>,
        body: Box<CompiledExpr>,
    },

    /// Sequence Head: Head(seq) - returns first element
    SeqHead(Box<CompiledExpr>),

    /// Sequence Tail: Tail(seq) - returns sequence without first element
    SeqTail(Box<CompiledExpr>),

    /// Sequence Append: Append(seq, elem) - returns sequence with elem appended
    SeqAppend {
        seq: Box<CompiledExpr>,
        elem: Box<CompiledExpr>,
    },

    /// Sequence Len: Len(seq) - returns length of sequence
    SeqLen(Box<CompiledExpr>),

    /// Set Cardinality: Cardinality(set) - returns number of elements
    SetCardinality(Box<CompiledExpr>),

    /// Sequence Range: Range(seq) - returns set of values in sequence
    SeqRange(Box<CompiledExpr>),

    /// Record constructor: [field1 |-> val1, field2 |-> val2, ...]
    Record(Vec<(Arc<str>, CompiledExpr)>),

    /// Boolean equality: left = right (returns Bool)
    BoolEq {
        left: Box<CompiledExpr>,
        right: Box<CompiledExpr>,
    },

    /// Record field access: expr.field
    RecordAccess {
        record: Box<CompiledExpr>,
        field: Arc<str>,
    },

    /// Fall back to full AST evaluation.
    /// The `reads_next` flag indicates whether the expression contains primed variables
    /// that need to be looked up from the next-state context during evaluation.
    Fallback {
        expr: Spanned<Expr>,
        reads_next: bool,
    },
}

/// Compiled guard for fast evaluation in BFS hot path.
///
/// Guards are boolean expressions that determine whether a transition
/// is enabled. Compiled guards use integer indices instead of string
/// lookups, achieving O(1) variable access.
#[derive(Debug, Clone)]
pub enum CompiledGuard {
    /// Always true (trivial guard)
    True,

    /// Always false (dead guard - action never enabled)
    False,

    /// Expression equals constant: expr = const
    EqConst { expr: CompiledExpr, expected: Value },

    /// Expression not-equals constant: expr # const
    NeqConst { expr: CompiledExpr, expected: Value },

    /// Specialized guard: `state_var[local_var] = const`
    /// Combines IntFunc lookup with LocalVar key and constant comparison.
    /// This is a common pattern in PlusCal-generated specs: `pc[self] = "label"`
    /// Fuses 3 operations into 1: LocalVar lookup + IntFunc apply + compare.
    IntFuncLocalVarEqConst {
        state: StateRef,
        func_var: VarIndex,
        key_depth: u8,
        expected: Value,
    },

    /// Specialized guard: `state_var[local_var] # const` (not-equals variant)
    IntFuncLocalVarNeqConst {
        state: StateRef,
        func_var: VarIndex,
        key_depth: u8,
        expected: Value,
    },

    /// Integer comparison: left op right
    IntCmp {
        left: CompiledExpr,
        op: CmpOp,
        right: CompiledExpr,
    },

    /// Set membership: elem \in set
    In {
        elem: CompiledExpr,
        set: CompiledExpr,
    },

    /// Subset: a \subseteq b
    Subseteq {
        left: CompiledExpr,
        right: CompiledExpr,
    },

    /// Conjunction: a /\ b (short-circuit evaluation)
    And(Vec<CompiledGuard>),

    /// Disjunction: a \/ b (short-circuit evaluation)
    Or(Vec<CompiledGuard>),

    /// Negation: ~a
    Not(Box<CompiledGuard>),

    /// Universal quantifier: \A x \in S : P(x)
    /// Short-circuits to false on first false body evaluation.
    ForAll {
        var_name: Arc<str>,
        domain: CompiledExpr,
        body: Box<CompiledGuard>,
    },

    /// Existential quantifier: \E x \in S : P(x)
    /// Short-circuits to true on first true body evaluation.
    Exists {
        var_name: Arc<str>,
        domain: CompiledExpr,
        body: Box<CompiledGuard>,
    },

    /// Implication: a => b (equivalent to ~a \/ b)
    Implies {
        antecedent: Box<CompiledGuard>,
        consequent: Box<CompiledGuard>,
    },

    /// Fall back to full AST evaluation (complex expression).
    /// The `reads_next` flag indicates whether the expression contains primed variables.
    Fallback {
        expr: Spanned<Expr>,
        reads_next: bool,
    },
}

impl CompiledExpr {
    /// Return true if this expression reads from the *next* state (contains primed state-vars).
    ///
    /// This is used to decide if assignment evaluation requires a progressive next-state scratch
    /// (so later assignments can observe primed values computed earlier in the same conjunction).
    pub fn reads_next_state(&self) -> bool {
        match self {
            CompiledExpr::Const(_) | CompiledExpr::LocalVar { .. } => false,

            CompiledExpr::StateVar { state, .. } => *state == StateRef::Next,

            CompiledExpr::FuncApp { state, key, .. } => {
                *state == StateRef::Next || key.reads_next_state()
            }
            CompiledExpr::FuncAppIntFuncLocalVar { state, .. } => *state == StateRef::Next,

            CompiledExpr::BinOp { left, right, .. } => {
                left.reads_next_state() || right.reads_next_state()
            }
            CompiledExpr::Except { func, key, value } => {
                func.reads_next_state() || key.reads_next_state() || value.reads_next_state()
            }
            CompiledExpr::ExceptNested { func, keys, value } => {
                func.reads_next_state()
                    || keys.iter().any(|k| k.reads_next_state())
                    || value.reads_next_state()
            }
            CompiledExpr::SetEnum(elems) | CompiledExpr::Tuple(elems) => {
                elems.iter().any(|e| e.reads_next_state())
            }
            CompiledExpr::DynFuncApp { func, key } => {
                func.reads_next_state() || key.reads_next_state()
            }
            CompiledExpr::Not(inner)
            | CompiledExpr::SeqHead(inner)
            | CompiledExpr::SeqTail(inner)
            | CompiledExpr::SeqLen(inner)
            | CompiledExpr::SetCardinality(inner)
            | CompiledExpr::SeqRange(inner) => inner.reads_next_state(),
            CompiledExpr::IfThenElse {
                condition,
                then_branch,
                else_branch,
            } => {
                condition.reads_next_state()
                    || then_branch.reads_next_state()
                    || else_branch.reads_next_state()
            }
            CompiledExpr::FuncDef { domain, body, .. } => {
                domain.reads_next_state() || body.reads_next_state()
            }
            CompiledExpr::SeqAppend { seq, elem } => {
                seq.reads_next_state() || elem.reads_next_state()
            }
            CompiledExpr::Record(fields) => fields.iter().any(|(_k, v)| v.reads_next_state()),
            CompiledExpr::BoolEq { left, right } => {
                left.reads_next_state() || right.reads_next_state()
            }
            CompiledExpr::RecordAccess { record, .. } => record.reads_next_state(),
            CompiledExpr::Fallback { reads_next, .. } => *reads_next,
        }
    }

    /// Evaluate a compiled expression against an ArrayState.
    ///
    /// Returns the computed Value. For Fallback expressions, uses the full eval().
    /// Hot path: called millions of times during model checking.
    #[inline(always)]
    pub fn eval_with_array(&self, ctx: &EvalCtx, array_state: &ArrayState) -> EvalResult<Value> {
        self.eval_with_arrays(ctx, array_state, array_state)
    }

    /// Evaluate a compiled expression against current and next ArrayStates.
    ///
    /// This is used for validating prime guards, where some sub-expressions must read from the
    /// next-state (primed) values.
    /// Hot path: called millions of times during model checking.
    #[inline(always)]
    pub fn eval_with_arrays(
        &self,
        ctx: &EvalCtx,
        current_array: &ArrayState,
        next_array: &ArrayState,
    ) -> EvalResult<Value> {
        match self {
            CompiledExpr::Const(v) => Ok(v.clone()),

            CompiledExpr::StateVar { state, idx } => Ok(match state {
                StateRef::Current => current_array.values()[idx.0 as usize].clone(),
                StateRef::Next => next_array.values()[idx.0 as usize].clone(),
            }),

            CompiledExpr::LocalVar { name, depth } => {
                // O(1) access via depth index
                ctx.get_local_by_depth(*depth)
                    .cloned()
                    .ok_or_else(|| EvalError::UndefinedVar {
                        name: name.to_string(),
                        span: None,
                    })
            }

            CompiledExpr::FuncApp {
                state,
                func_var,
                key,
            } => {
                let values = match state {
                    StateRef::Current => current_array.values(),
                    StateRef::Next => next_array.values(),
                };
                let func_val = &values[func_var.0 as usize];
                let key_val = key.eval_with_arrays(ctx, current_array, next_array)?;

                // Fast path for function application
                match func_val {
                    Value::Func(f) => {
                        f.apply(&key_val)
                            .cloned()
                            .ok_or_else(|| EvalError::NotInDomain {
                                arg: format!("{:?}", key_val),
                                span: None,
                            })
                    }
                    Value::IntFunc(f) => {
                        f.apply(&key_val)
                            .cloned()
                            .ok_or_else(|| EvalError::NotInDomain {
                                arg: format!("{:?}", key_val),
                                span: None,
                            })
                    }
                    Value::LazyFunc(_) => {
                        // LazyFunc requires full eval context - fall back
                        Err(EvalError::Internal {
                            message: "LazyFunc in compiled expr - should use fallback".into(),
                            span: None,
                        })
                    }
                    Value::Record(r) => {
                        if let Value::String(s) = &key_val {
                            r.get(s.as_ref())
                                .cloned()
                                .ok_or_else(|| EvalError::NoSuchField {
                                    field: s.to_string(),
                                    span: None,
                                })
                        } else {
                            Err(EvalError::TypeError {
                                expected: "String",
                                got: key_val.type_name(),
                                span: None,
                            })
                        }
                    }
                    Value::Seq(s) => {
                        // Sequence indexing is 1-based
                        let idx = key_val.as_i64().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: key_val.type_name(),
                            span: None,
                        })?;
                        if idx < 1 || idx as usize > s.len() {
                            return Err(EvalError::IndexOutOfBounds {
                                index: idx,
                                len: s.len(),
                                span: None,
                            });
                        }
                        Ok(s[(idx - 1) as usize].clone())
                    }
                    Value::Tuple(t) => {
                        // Tuple indexing is 1-based
                        let idx = key_val.as_i64().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: key_val.type_name(),
                            span: None,
                        })?;
                        if idx < 1 || idx as usize > t.len() {
                            return Err(EvalError::IndexOutOfBounds {
                                index: idx,
                                len: t.len(),
                                span: None,
                            });
                        }
                        Ok(t[(idx - 1) as usize].clone())
                    }
                    _ => Err(EvalError::TypeError {
                        expected: "function/sequence",
                        got: func_val.type_name(),
                        span: None,
                    }),
                }
            }

            // Specialized: state_var[local_var] for IntFunc - fuses LocalVar lookup + IntFunc indexing
            CompiledExpr::FuncAppIntFuncLocalVar {
                state,
                func_var,
                key_depth,
            } => {
                // Get local binding value (the key for function application)
                let key_val =
                    ctx.get_local_by_depth(*key_depth)
                        .ok_or_else(|| EvalError::Internal {
                            message: "FuncAppIntFuncLocalVar: local var not found".into(),
                            span: None,
                        })?;

                // Get state array and function value
                let values = match state {
                    StateRef::Current => current_array.values(),
                    StateRef::Next => next_array.values(),
                };
                let func_val = &values[func_var.0 as usize];

                // Fast path for IntFunc with integer key
                if let Value::IntFunc(f) = func_val {
                    let key_i64 = match key_val {
                        Value::SmallInt(n) => *n,
                        Value::Int(n) => n.to_i64().ok_or_else(|| EvalError::Internal {
                            message: "FuncAppIntFuncLocalVar: key too large".into(),
                            span: None,
                        })?,
                        _ => {
                            return Err(EvalError::TypeError {
                                expected: "Int",
                                got: key_val.type_name(),
                                span: None,
                            })
                        }
                    };
                    if key_i64 >= f.min && key_i64 <= f.max {
                        let result = &f.values[(key_i64 - f.min) as usize];
                        return Ok(result.clone());
                    }
                    return Err(EvalError::NotInDomain {
                        arg: format!("{}", key_i64),
                        span: None,
                    });
                }

                // Fast path for Seq with integer key (1-based indexing)
                // This is common because [p \in 1..N |-> ...] creates a Seq
                if let Value::Seq(s) = func_val {
                    let idx = match key_val {
                        Value::SmallInt(n) => *n,
                        Value::Int(n) => n.to_i64().ok_or_else(|| EvalError::Internal {
                            message: "FuncAppSeqLocalVar: key too large".into(),
                            span: None,
                        })?,
                        _ => {
                            return Err(EvalError::TypeError {
                                expected: "Int",
                                got: key_val.type_name(),
                                span: None,
                            })
                        }
                    };
                    if idx < 1 || idx as usize > s.len() {
                        return Err(EvalError::IndexOutOfBounds {
                            index: idx,
                            len: s.len(),
                            span: None,
                        });
                    }
                    return Ok(s[(idx - 1) as usize].clone());
                }

                // Fallback for regular Func and other function-like types
                match func_val {
                    Value::Func(f) => {
                        f.apply(key_val)
                            .cloned()
                            .ok_or_else(|| EvalError::NotInDomain {
                                arg: format!("{:?}", key_val),
                                span: None,
                            })
                    }
                    Value::Record(r) => {
                        if let Value::String(s) = key_val {
                            r.get(s.as_ref())
                                .cloned()
                                .ok_or_else(|| EvalError::NoSuchField {
                                    field: s.to_string(),
                                    span: None,
                                })
                        } else {
                            Err(EvalError::TypeError {
                                expected: "String",
                                got: key_val.type_name(),
                                span: None,
                            })
                        }
                    }
                    Value::Seq(s) => {
                        // Sequence indexing is 1-based
                        let idx = key_val.as_i64().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: key_val.type_name(),
                            span: None,
                        })?;
                        if idx < 1 || idx as usize > s.len() {
                            return Err(EvalError::IndexOutOfBounds {
                                index: idx,
                                len: s.len(),
                                span: None,
                            });
                        }
                        Ok(s[(idx - 1) as usize].clone())
                    }
                    Value::Tuple(t) => {
                        // Tuple indexing is 1-based
                        let idx = key_val.as_i64().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: key_val.type_name(),
                            span: None,
                        })?;
                        if idx < 1 || idx as usize > t.len() {
                            return Err(EvalError::IndexOutOfBounds {
                                index: idx,
                                len: t.len(),
                                span: None,
                            });
                        }
                        Ok(t[(idx - 1) as usize].clone())
                    }
                    _ => Err(EvalError::TypeError {
                        expected: "function/sequence",
                        got: func_val.type_name(),
                        span: None,
                    }),
                }
            }

            CompiledExpr::BinOp { left, op, right } => {
                let lv = left.eval_with_arrays(ctx, current_array, next_array)?;
                let rv = right.eval_with_arrays(ctx, current_array, next_array)?;

                match op {
                    BinOp::SetUnion => {
                        let ls = lv.to_sorted_set().ok_or_else(|| EvalError::TypeError {
                            expected: "set",
                            got: lv.type_name(),
                            span: None,
                        })?;
                        let rs = rv.to_sorted_set().ok_or_else(|| EvalError::TypeError {
                            expected: "set",
                            got: rv.type_name(),
                            span: None,
                        })?;
                        Ok(Value::Set(ls.union(&rs)))
                    }
                    BinOp::SetDiff => {
                        let ls = lv.to_sorted_set().ok_or_else(|| EvalError::TypeError {
                            expected: "set",
                            got: lv.type_name(),
                            span: None,
                        })?;
                        let rs = rv.to_sorted_set().ok_or_else(|| EvalError::TypeError {
                            expected: "set",
                            got: rv.type_name(),
                            span: None,
                        })?;
                        Ok(Value::Set(ls.difference(&rs)))
                    }
                    BinOp::IntAdd => {
                        // SmallInt fast path
                        if let (Value::SmallInt(l), Value::SmallInt(r)) = (&lv, &rv) {
                            if let Some(sum) = l.checked_add(*r) {
                                return Ok(Value::SmallInt(sum));
                            }
                        }
                        // BigInt fallback
                        let l = lv.to_bigint().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: lv.type_name(),
                            span: None,
                        })?;
                        let r = rv.to_bigint().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: rv.type_name(),
                            span: None,
                        })?;
                        Ok(Value::big_int(l + r))
                    }
                    BinOp::IntSub => {
                        // SmallInt fast path
                        if let (Value::SmallInt(l), Value::SmallInt(r)) = (&lv, &rv) {
                            if let Some(diff) = l.checked_sub(*r) {
                                return Ok(Value::SmallInt(diff));
                            }
                        }
                        // BigInt fallback
                        let l = lv.to_bigint().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: lv.type_name(),
                            span: None,
                        })?;
                        let r = rv.to_bigint().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: rv.type_name(),
                            span: None,
                        })?;
                        Ok(Value::big_int(l - r))
                    }
                    BinOp::SetIntersect => {
                        let ls = lv.to_sorted_set().ok_or_else(|| EvalError::TypeError {
                            expected: "set",
                            got: lv.type_name(),
                            span: None,
                        })?;
                        let rs = rv.to_sorted_set().ok_or_else(|| EvalError::TypeError {
                            expected: "set",
                            got: rv.type_name(),
                            span: None,
                        })?;
                        Ok(Value::Set(ls.intersection(&rs)))
                    }
                    BinOp::IntMul => {
                        // SmallInt fast path
                        if let (Value::SmallInt(l), Value::SmallInt(r)) = (&lv, &rv) {
                            if let Some(prod) = l.checked_mul(*r) {
                                return Ok(Value::SmallInt(prod));
                            }
                        }
                        // BigInt fallback
                        let l = lv.to_bigint().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: lv.type_name(),
                            span: None,
                        })?;
                        let r = rv.to_bigint().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: rv.type_name(),
                            span: None,
                        })?;
                        Ok(Value::big_int(l * r))
                    }
                    BinOp::IntDiv => {
                        // SmallInt fast path
                        if let (Value::SmallInt(l), Value::SmallInt(r)) = (&lv, &rv) {
                            if *r != 0 {
                                // TLA+ uses truncating division toward negative infinity
                                let quot = l.div_euclid(*r);
                                return Ok(Value::SmallInt(quot));
                            }
                        }
                        // BigInt fallback
                        let l = lv.to_bigint().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: lv.type_name(),
                            span: None,
                        })?;
                        let r = rv.to_bigint().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: rv.type_name(),
                            span: None,
                        })?;
                        if r.is_zero() {
                            return Err(EvalError::DivisionByZero { span: None });
                        }
                        Ok(Value::big_int(l.div_floor(&r)))
                    }
                    BinOp::IntMod => {
                        // SmallInt fast path
                        if let (Value::SmallInt(l), Value::SmallInt(r)) = (&lv, &rv) {
                            if *r != 0 {
                                // TLA+ modulo: always returns non-negative when divisor is positive
                                let rem = l.rem_euclid(*r);
                                return Ok(Value::SmallInt(rem));
                            }
                        }
                        // BigInt fallback
                        let l = lv.to_bigint().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: lv.type_name(),
                            span: None,
                        })?;
                        let r = rv.to_bigint().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: rv.type_name(),
                            span: None,
                        })?;
                        if r.is_zero() {
                            return Err(EvalError::DivisionByZero { span: None });
                        }
                        Ok(Value::big_int(l.mod_floor(&r)))
                    }
                }
            }

            CompiledExpr::Except { func, key, value } => {
                // Use eval_cow for key since IntFunc.except() takes key by reference,
                // avoiding unnecessary clone for the common IntFunc case
                let func_val = func.eval_with_arrays(ctx, current_array, next_array)?;
                let key_cow = key.eval_cow(ctx, current_array, next_array)?;
                let new_val = value.eval_with_arrays(ctx, current_array, next_array)?;

                match func_val {
                    Value::Func(f) => {
                        // Func.except needs owned key
                        Ok(Value::Func(f.except(key_cow.into_owned(), new_val)))
                    }
                    Value::IntFunc(f) => {
                        // IntFunc.except takes key by reference - no clone needed!
                        Ok(Value::IntFunc(f.except(&key_cow, new_val)))
                    }
                    _ if func_val.as_seq_or_tuple_elements().is_some() => {
                        // Seq/Tuple EXCEPT: [seq EXCEPT ![i] = v] updates index i (1-indexed)
                        let s = func_val.as_seq_or_tuple_elements().unwrap();
                        if let Some(idx) = key_cow.as_i64() {
                            let index = (idx - 1) as usize; // TLA+ is 1-indexed
                            if index >= s.len() {
                                return Err(EvalError::IndexOutOfBounds {
                                    index: idx,
                                    len: s.len(),
                                    span: None,
                                });
                            }
                            // Short-circuit: if value unchanged, return original without cloning
                            if s[index] == new_val {
                                return Ok(func_val);
                            }
                            let mut new_vec: Vec<Value> = s.to_vec();
                            new_vec[index] = new_val;
                            // Return same type as input
                            if matches!(func_val, Value::Seq(_)) {
                                Ok(Value::Seq(new_vec.into()))
                            } else {
                                Ok(Value::Tuple(new_vec.into()))
                            }
                        } else {
                            Err(EvalError::TypeError {
                                expected: "Int",
                                got: key_cow.type_name(),
                                span: None,
                            })
                        }
                    }
                    Value::Record(r) => {
                        if let Value::String(s) = &*key_cow {
                            let new_rec = r.update(s.clone(), new_val);
                            Ok(Value::Record(new_rec))
                        } else {
                            Err(EvalError::TypeError {
                                expected: "String",
                                got: key_cow.type_name(),
                                span: None,
                            })
                        }
                    }
                    _ => Err(EvalError::TypeError {
                        expected: "function, sequence, or record",
                        got: func_val.type_name(),
                        span: None,
                    }),
                }
            }

            CompiledExpr::ExceptNested { func, keys, value } => {
                // For [f EXCEPT ![k1][k2] = v], semantics are:
                // [f EXCEPT ![k1] = [f[k1] EXCEPT ![k2] = v]]
                // We navigate to f[k1][k2]...[k(n-1)], update with final key, then rebuild
                let func_val = func.eval_with_arrays(ctx, current_array, next_array)?;
                let new_val = value.eval_with_arrays(ctx, current_array, next_array)?;

                // Evaluate all keys
                let key_vals: Vec<Value> = keys
                    .iter()
                    .map(|k| k.eval_with_arrays(ctx, current_array, next_array))
                    .collect::<Result<_, _>>()?;

                // Apply nested except
                apply_nested_except(func_val, &key_vals, new_val)
            }

            CompiledExpr::SetEnum(elems) => {
                let mut set = SetBuilder::new();
                for elem in elems {
                    set.insert(elem.eval_with_arrays(ctx, current_array, next_array)?);
                }
                Ok(set.build_value())
            }

            CompiledExpr::Tuple(elems) => {
                let mut vals = Vec::with_capacity(elems.len());
                for elem in elems {
                    vals.push(elem.eval_with_arrays(ctx, current_array, next_array)?);
                }
                Ok(Value::Tuple(vals.into()))
            }

            CompiledExpr::Record(fields) => {
                let mut builder = RecordBuilder::with_capacity(fields.len());
                for (name, expr) in fields {
                    let val = expr.eval_with_arrays(ctx, current_array, next_array)?;
                    builder.insert(Arc::clone(name), val);
                }
                Ok(Value::Record(builder.build()))
            }

            CompiledExpr::BoolEq { left, right } => {
                let left_val = left.eval_with_arrays(ctx, current_array, next_array)?;
                let right_val = right.eval_with_arrays(ctx, current_array, next_array)?;
                Ok(Value::Bool(left_val == right_val))
            }

            // Dynamic function application: func[key] where func is a compiled expression
            CompiledExpr::DynFuncApp { func, key } => {
                let func_val = func.eval_with_arrays(ctx, current_array, next_array)?;
                let key_val = key.eval_with_arrays(ctx, current_array, next_array)?;

                match func_val {
                    Value::Func(f) => {
                        f.apply(&key_val)
                            .cloned()
                            .ok_or_else(|| EvalError::NotInDomain {
                                arg: format!("{key_val:?}"),
                                span: None,
                            })
                    }
                    Value::IntFunc(f) => {
                        f.apply(&key_val)
                            .cloned()
                            .ok_or_else(|| EvalError::NotInDomain {
                                arg: format!("{key_val:?}"),
                                span: None,
                            })
                    }
                    _ if func_val.as_seq_or_tuple_elements().is_some() => {
                        let t = func_val.as_seq_or_tuple_elements().unwrap();
                        // Tuple/Seq indexing: <<a, b, c>>[2] = b (1-indexed in TLA+)
                        if let Some(idx) = key_val.as_i64() {
                            let index = (idx - 1) as usize; // TLA+ tuples/seqs are 1-indexed
                            t.get(index).cloned().ok_or(EvalError::IndexOutOfBounds {
                                index: idx,
                                len: t.len(),
                                span: None,
                            })
                        } else {
                            Err(EvalError::TypeError {
                                expected: "Int",
                                got: key_val.type_name(),
                                span: None,
                            })
                        }
                    }
                    Value::Record(r) => {
                        if let Value::String(s) = &key_val {
                            r.get(s.as_ref())
                                .cloned()
                                .ok_or_else(|| EvalError::NoSuchField {
                                    field: s.to_string(),
                                    span: None,
                                })
                        } else {
                            Err(EvalError::TypeError {
                                expected: "String",
                                got: key_val.type_name(),
                                span: None,
                            })
                        }
                    }
                    _ => Err(EvalError::TypeError {
                        expected: "function, tuple, or record",
                        got: func_val.type_name(),
                        span: None,
                    }),
                }
            }

            CompiledExpr::Not(inner) => {
                let val = inner.eval_with_arrays(ctx, current_array, next_array)?;
                match val {
                    Value::Bool(b) => Ok(Value::Bool(!b)),
                    _ => Err(EvalError::TypeError {
                        expected: "Bool",
                        got: val.type_name(),
                        span: None,
                    }),
                }
            }

            CompiledExpr::IfThenElse {
                condition,
                then_branch,
                else_branch,
            } => {
                let cond_val = condition.eval_with_arrays(ctx, current_array, next_array)?;
                match cond_val {
                    Value::Bool(true) => {
                        then_branch.eval_with_arrays(ctx, current_array, next_array)
                    }
                    Value::Bool(false) => {
                        else_branch.eval_with_arrays(ctx, current_array, next_array)
                    }
                    _ => Err(EvalError::TypeError {
                        expected: "Bool",
                        got: cond_val.type_name(),
                        span: None,
                    }),
                }
            }

            CompiledExpr::FuncDef {
                var_name,
                domain,
                body,
            } => {
                // Evaluate the domain to get the set of keys
                let domain_val = domain.eval_with_arrays(ctx, current_array, next_array)?;

                // Handle IntInterval domain (common case: 1..N)
                if let Value::Interval(intv) = &domain_val {
                    if let (Some(min), Some(max)) = (intv.low.to_i64(), intv.high.to_i64()) {
                        let size = (max - min + 1) as usize;
                        if size <= 1_000_000 {
                            let mut local_ctx = ctx.clone();
                            let mark = local_ctx.mark_stack();
                            let mut values = Vec::with_capacity(size);
                            for i in min..=max {
                                let elem = Value::SmallInt(i);
                                local_ctx.push_binding(Arc::clone(var_name), elem.clone());
                                local_ctx.bind_mut(Arc::clone(var_name), elem);
                                let val =
                                    body.eval_with_arrays(&local_ctx, current_array, next_array)?;
                                values.push(val);
                                local_ctx.pop_to_mark(mark);
                            }
                            // If domain is 1..n, this is a sequence
                            if min == 1 {
                                return Ok(Value::Seq(values.into()));
                            }
                            return Ok(Value::IntFunc(IntIntervalFunc::new(min, max, values)));
                        }
                    }
                }

                // General set domain
                let domain_set =
                    domain_val
                        .to_sorted_set()
                        .ok_or_else(|| EvalError::TypeError {
                            expected: "set",
                            got: domain_val.type_name(),
                            span: None,
                        })?;

                let mut local_ctx = ctx.clone();
                let mark = local_ctx.mark_stack();

                // Check for sequence domain {1, 2, ..., n} (empty set => empty sequence)
                let mut is_seq_domain = true;
                let mut expected: i64 = 1;

                let mut entries: Vec<(Value, Value)> = Vec::with_capacity(domain_set.len());
                for elem in domain_set.iter() {
                    if is_seq_domain {
                        if elem.as_i64() == Some(expected) {
                            expected += 1;
                        } else {
                            is_seq_domain = false;
                        }
                    }

                    let elem_owned = elem.clone();
                    local_ctx.push_binding(Arc::clone(var_name), elem_owned.clone());
                    local_ctx.bind_mut(Arc::clone(var_name), elem_owned.clone());
                    let val = body.eval_with_arrays(&local_ctx, current_array, next_array)?;
                    entries.push((elem_owned, val));
                    local_ctx.pop_to_mark(mark);
                }

                if is_seq_domain {
                    // Domain is 1..n; entries were collected in key-sorted order.
                    let seq_values: Vec<Value> = entries.into_iter().map(|(_, v)| v).collect();
                    return Ok(Value::Seq(seq_values.into()));
                }

                Ok(Value::Func(FuncValue::from_sorted_entries(entries)))
            }

            CompiledExpr::SeqHead(seq_expr) => {
                let seq_val = seq_expr.eval_with_arrays(ctx, current_array, next_array)?;
                if let Some(s) = seq_val.as_seq_or_tuple_elements() {
                    if s.is_empty() {
                        return Err(EvalError::IndexOutOfBounds {
                            index: 1,
                            len: 0,
                            span: None,
                        });
                    }
                    Ok(s[0].clone())
                } else {
                    Err(EvalError::TypeError {
                        expected: "sequence",
                        got: seq_val.type_name(),
                        span: None,
                    })
                }
            }

            CompiledExpr::SeqTail(seq_expr) => {
                let seq_val = seq_expr.eval_with_arrays(ctx, current_array, next_array)?;
                if let Some(s) = seq_val.as_seq_or_tuple_elements() {
                    if s.is_empty() {
                        return Err(EvalError::IndexOutOfBounds {
                            index: 1,
                            len: 0,
                            span: None,
                        });
                    }
                    Ok(Value::Seq(s[1..].to_vec().into()))
                } else {
                    Err(EvalError::TypeError {
                        expected: "sequence",
                        got: seq_val.type_name(),
                        span: None,
                    })
                }
            }

            CompiledExpr::SeqAppend { seq, elem } => {
                let seq_val = seq.eval_with_arrays(ctx, current_array, next_array)?;
                let elem_val = elem.eval_with_arrays(ctx, current_array, next_array)?;
                match seq_val {
                    Value::Seq(s) => {
                        let mut new_seq = s.to_vec();
                        new_seq.push(elem_val);
                        Ok(Value::Seq(new_seq.into()))
                    }
                    Value::Tuple(s) => {
                        let mut new_seq = s.to_vec();
                        new_seq.push(elem_val);
                        Ok(Value::Seq(new_seq.into()))
                    }
                    _ => Err(EvalError::TypeError {
                        expected: "sequence",
                        got: seq_val.type_name(),
                        span: None,
                    }),
                }
            }

            CompiledExpr::SeqLen(seq_expr) => {
                let seq_val = seq_expr.eval_with_arrays(ctx, current_array, next_array)?;
                // Check for Seq/Tuple first
                if let Some(s) = seq_val.as_seq_or_tuple_elements() {
                    return Ok(Value::SmallInt(s.len() as i64));
                }
                match &seq_val {
                    Value::String(s) => Ok(Value::SmallInt(s.len() as i64)),
                    Value::IntFunc(f) => {
                        if f.min == 1 {
                            Ok(Value::SmallInt(f.len() as i64))
                        } else {
                            Err(EvalError::TypeError {
                                expected: "sequence or string",
                                got: seq_val.type_name(),
                                span: None,
                            })
                        }
                    }
                    Value::Func(f) => {
                        // In TLA+, sequences are functions with domain 1..n.
                        // Only accept functions whose domain is exactly {1, 2, ..., n} (or empty).
                        let mut expected: i64 = 1;
                        for key in f.domain_iter() {
                            let Some(k) = key.as_i64() else {
                                return Err(EvalError::TypeError {
                                    expected: "sequence or string",
                                    got: seq_val.type_name(),
                                    span: None,
                                });
                            };
                            if k != expected {
                                return Err(EvalError::TypeError {
                                    expected: "sequence or string",
                                    got: seq_val.type_name(),
                                    span: None,
                                });
                            }
                            expected += 1;
                        }
                        Ok(Value::SmallInt(expected - 1))
                    }
                    _ => Err(EvalError::TypeError {
                        expected: "sequence or string",
                        got: seq_val.type_name(),
                        span: None,
                    }),
                }
            }

            CompiledExpr::SetCardinality(set_expr) => {
                let set_val = set_expr.eval_with_arrays(ctx, current_array, next_array)?;
                // Use set_len() method which handles all set types
                match set_val.set_len() {
                    Some(c) => Ok(Value::big_int(c)),
                    None => Err(EvalError::TypeError {
                        expected: "set",
                        got: set_val.type_name(),
                        span: None,
                    }),
                }
            }

            CompiledExpr::SeqRange(seq_expr) => {
                let seq_val = seq_expr.eval_with_arrays(ctx, current_array, next_array)?;
                if let Some(s) = seq_val.as_seq_or_tuple_elements() {
                    let set = SortedSet::from_iter(s.iter().cloned());
                    Ok(Value::Set(set))
                } else {
                    Err(EvalError::TypeError {
                        expected: "sequence",
                        got: seq_val.type_name(),
                        span: None,
                    })
                }
            }

            CompiledExpr::RecordAccess { record, field } => {
                let record_val = record.eval_with_arrays(ctx, current_array, next_array)?;
                match &record_val {
                    Value::Record(fields) => {
                        fields
                            .get(field.as_ref())
                            .cloned()
                            .ok_or_else(|| EvalError::NoSuchField {
                                field: field.to_string(),
                                span: None,
                            })
                    }
                    _ => Err(EvalError::TypeError {
                        expected: "record",
                        got: record_val.type_name(),
                        span: None,
                    }),
                }
            }

            CompiledExpr::Fallback { expr, reads_next } => {
                // If this Fallback expression reads primed variables and we have a progressive
                // next-state context (next_array != current_array), build a next_state env
                // so that primed variable lookups can succeed.
                //
                // This is the narrow fix for issue #14: expressions like `IF HCnxt THEN ...`
                // where HCnxt contains `hr'` need access to the progressive next-state values.
                //
                // OPTIMIZATION (Re: #56): If next_state_env is already set (by caller via
                // bind_next_state_array), use it directly to avoid rebuilding HashMap per
                // Fallback evaluation. This is critical for nested Fallbacks in prime guards
                // like `Cardinality({p \in P : \E m \in rcvd'[self] : ...})` where many
                // Fallback evaluations would otherwise each build O(n) HashMaps.
                if *reads_next && !std::ptr::eq(current_array, next_array) {
                    // Fast path: if next_state_env is already set up, eval() can use it
                    // for O(1) primed variable lookups without rebuilding a HashMap.
                    if ctx.has_next_state_env() {
                        return eval(ctx, expr);
                    }
                    // Slow path: build HashMap for next-state lookups
                    let registry = ctx.var_registry();
                    let mut next_state = Env::new();
                    for (idx, name) in registry.iter() {
                        next_state
                            .insert(Arc::clone(name), next_array.values()[idx.as_usize()].clone());
                    }
                    let eval_ctx = ctx.with_next_state(next_state);
                    eval(&eval_ctx, expr)
                } else {
                    eval(ctx, expr)
                }
            }
        }
    }

    /// Evaluate a compiled expression, returning a `Cow<Value>` to avoid cloning
    /// when the result is a direct reference to existing data.
    ///
    /// This is a performance optimization: for leaf expressions (Const, StateVar,
    /// LocalVar, FuncApp), we return `Cow::Borrowed` instead of cloning the value.
    /// For computed expressions (BinOp, SetEnum, etc.), we return `Cow::Owned`.
    ///
    /// **Use this method for guards and other read-only contexts.**
    /// For assignment values where ownership is required, use `eval_with_arrays`.
    #[inline]
    pub fn eval_cow<'a>(
        &'a self,
        ctx: &'a EvalCtx,
        current_array: &'a ArrayState,
        next_array: &'a ArrayState,
    ) -> EvalResult<Cow<'a, Value>> {
        match self {
            // Leaf expressions: return borrowed references (no clone!)
            CompiledExpr::Const(v) => Ok(Cow::Borrowed(v)),

            CompiledExpr::StateVar { state, idx } => {
                let values = match state {
                    StateRef::Current => current_array.values(),
                    StateRef::Next => next_array.values(),
                };
                Ok(Cow::Borrowed(&values[idx.0 as usize]))
            }

            CompiledExpr::LocalVar { name, depth } => ctx
                .get_local_by_depth(*depth)
                .map(Cow::Borrowed)
                .ok_or_else(|| EvalError::UndefinedVar {
                    name: name.to_string(),
                    span: None,
                }),

            CompiledExpr::FuncApp {
                state,
                func_var,
                key,
            } => {
                let values = match state {
                    StateRef::Current => current_array.values(),
                    StateRef::Next => next_array.values(),
                };
                let func_val = &values[func_var.0 as usize];
                let key_val = key.eval_cow(ctx, current_array, next_array)?;

                // Fast path for function application - return borrowed reference
                match func_val {
                    Value::Func(f) => {
                        f.apply(&key_val)
                            .map(Cow::Borrowed)
                            .ok_or_else(|| EvalError::NotInDomain {
                                arg: format!("{key_val:?}"),
                                span: None,
                            })
                    }
                    Value::IntFunc(f) => {
                        f.apply(&key_val)
                            .map(Cow::Borrowed)
                            .ok_or_else(|| EvalError::NotInDomain {
                                arg: format!("{key_val:?}"),
                                span: None,
                            })
                    }
                    Value::LazyFunc(_) => Err(EvalError::Internal {
                        message: "LazyFunc in compiled expr - should use fallback".into(),
                        span: None,
                    }),
                    Value::Record(r) => {
                        if let Value::String(s) = &*key_val {
                            r.get(s.as_ref()).map(Cow::Borrowed).ok_or_else(|| {
                                EvalError::NoSuchField {
                                    field: s.to_string(),
                                    span: None,
                                }
                            })
                        } else {
                            Err(EvalError::TypeError {
                                expected: "String",
                                got: key_val.type_name(),
                                span: None,
                            })
                        }
                    }
                    Value::Seq(s) => {
                        // Sequence indexing is 1-based
                        let idx = key_val.as_i64().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: key_val.type_name(),
                            span: None,
                        })?;
                        if idx < 1 || idx as usize > s.len() {
                            return Err(EvalError::IndexOutOfBounds {
                                index: idx,
                                len: s.len(),
                                span: None,
                            });
                        }
                        Ok(Cow::Borrowed(&s[(idx - 1) as usize]))
                    }
                    Value::Tuple(t) => {
                        // Tuple indexing is 1-based
                        let idx = key_val.as_i64().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: key_val.type_name(),
                            span: None,
                        })?;
                        if idx < 1 || idx as usize > t.len() {
                            return Err(EvalError::IndexOutOfBounds {
                                index: idx,
                                len: t.len(),
                                span: None,
                            });
                        }
                        Ok(Cow::Borrowed(&t[(idx - 1) as usize]))
                    }
                    _ => Err(EvalError::TypeError {
                        expected: "function/sequence",
                        got: func_val.type_name(),
                        span: None,
                    }),
                }
            }

            // Specialized: state_var[local_var] for IntFunc - returns borrowed reference
            CompiledExpr::FuncAppIntFuncLocalVar {
                state,
                func_var,
                key_depth,
            } => {
                let key_val =
                    ctx.get_local_by_depth(*key_depth)
                        .ok_or_else(|| EvalError::Internal {
                            message: "FuncAppIntFuncLocalVar: local var not found".into(),
                            span: None,
                        })?;

                let values = match state {
                    StateRef::Current => current_array.values(),
                    StateRef::Next => next_array.values(),
                };
                let func_val = &values[func_var.0 as usize];

                // Fast path for IntFunc with integer key
                if let Value::IntFunc(f) = func_val {
                    let key_i64 = match key_val {
                        Value::SmallInt(n) => *n,
                        Value::Int(n) => n.to_i64().ok_or_else(|| EvalError::Internal {
                            message: "FuncAppIntFuncLocalVar: key too large".into(),
                            span: None,
                        })?,
                        _ => {
                            return Err(EvalError::TypeError {
                                expected: "Int",
                                got: key_val.type_name(),
                                span: None,
                            })
                        }
                    };
                    if key_i64 >= f.min && key_i64 <= f.max {
                        let result = &f.values[(key_i64 - f.min) as usize];
                        return Ok(Cow::Borrowed(result));
                    }
                    return Err(EvalError::NotInDomain {
                        arg: format!("{}", key_i64),
                        span: None,
                    });
                }

                // Fast path for Seq with integer key (1-based indexing)
                // This is common because [p \in 1..N |-> ...] creates a Seq
                if let Value::Seq(s) = func_val {
                    let idx = match key_val {
                        Value::SmallInt(n) => *n,
                        Value::Int(n) => n.to_i64().ok_or_else(|| EvalError::Internal {
                            message: "FuncAppSeqLocalVar: key too large".into(),
                            span: None,
                        })?,
                        _ => {
                            return Err(EvalError::TypeError {
                                expected: "Int",
                                got: key_val.type_name(),
                                span: None,
                            })
                        }
                    };
                    if idx < 1 || idx as usize > s.len() {
                        return Err(EvalError::IndexOutOfBounds {
                            index: idx,
                            len: s.len(),
                            span: None,
                        });
                    }
                    return Ok(Cow::Borrowed(&s[(idx - 1) as usize]));
                }

                // Fallback for regular Func - must return Cow::Borrowed for function values
                match func_val {
                    Value::Func(f) => {
                        f.apply(key_val)
                            .map(Cow::Borrowed)
                            .ok_or_else(|| EvalError::NotInDomain {
                                arg: format!("{:?}", key_val),
                                span: None,
                            })
                    }
                    Value::Record(r) => {
                        if let Value::String(s) = key_val {
                            r.get(s.as_ref()).map(Cow::Borrowed).ok_or_else(|| {
                                EvalError::NoSuchField {
                                    field: s.to_string(),
                                    span: None,
                                }
                            })
                        } else {
                            Err(EvalError::TypeError {
                                expected: "String",
                                got: key_val.type_name(),
                                span: None,
                            })
                        }
                    }
                    Value::Seq(s) => {
                        // Sequence indexing is 1-based
                        let idx = key_val.as_i64().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: key_val.type_name(),
                            span: None,
                        })?;
                        if idx < 1 || idx as usize > s.len() {
                            return Err(EvalError::IndexOutOfBounds {
                                index: idx,
                                len: s.len(),
                                span: None,
                            });
                        }
                        Ok(Cow::Borrowed(&s[(idx - 1) as usize]))
                    }
                    Value::Tuple(t) => {
                        // Tuple indexing is 1-based
                        let idx = key_val.as_i64().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: key_val.type_name(),
                            span: None,
                        })?;
                        if idx < 1 || idx as usize > t.len() {
                            return Err(EvalError::IndexOutOfBounds {
                                index: idx,
                                len: t.len(),
                                span: None,
                            });
                        }
                        Ok(Cow::Borrowed(&t[(idx - 1) as usize]))
                    }
                    _ => Err(EvalError::TypeError {
                        expected: "function/sequence",
                        got: func_val.type_name(),
                        span: None,
                    }),
                }
            }

            // Computed expressions: must return owned values
            CompiledExpr::BinOp { left, op, right } => {
                let lv = left.eval_cow(ctx, current_array, next_array)?;
                let rv = right.eval_cow(ctx, current_array, next_array)?;

                let result = match op {
                    BinOp::SetUnion => {
                        let ls = lv.to_sorted_set().ok_or_else(|| EvalError::TypeError {
                            expected: "set",
                            got: lv.type_name(),
                            span: None,
                        })?;
                        let rs = rv.to_sorted_set().ok_or_else(|| EvalError::TypeError {
                            expected: "set",
                            got: rv.type_name(),
                            span: None,
                        })?;
                        Value::Set(ls.union(&rs))
                    }
                    BinOp::SetDiff => {
                        let ls = lv.to_sorted_set().ok_or_else(|| EvalError::TypeError {
                            expected: "set",
                            got: lv.type_name(),
                            span: None,
                        })?;
                        let rs = rv.to_sorted_set().ok_or_else(|| EvalError::TypeError {
                            expected: "set",
                            got: rv.type_name(),
                            span: None,
                        })?;
                        Value::Set(ls.difference(&rs))
                    }
                    BinOp::IntAdd => {
                        // SmallInt fast path
                        if let (Value::SmallInt(l), Value::SmallInt(r)) = (&*lv, &*rv) {
                            if let Some(sum) = l.checked_add(*r) {
                                return Ok(Cow::Owned(Value::SmallInt(sum)));
                            }
                        }
                        // BigInt fallback
                        let l = lv.to_bigint().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: lv.type_name(),
                            span: None,
                        })?;
                        let r = rv.to_bigint().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: rv.type_name(),
                            span: None,
                        })?;
                        Value::big_int(l + r)
                    }
                    BinOp::IntSub => {
                        if let (Value::SmallInt(l), Value::SmallInt(r)) = (&*lv, &*rv) {
                            if let Some(diff) = l.checked_sub(*r) {
                                return Ok(Cow::Owned(Value::SmallInt(diff)));
                            }
                        }
                        let l = lv.to_bigint().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: lv.type_name(),
                            span: None,
                        })?;
                        let r = rv.to_bigint().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: rv.type_name(),
                            span: None,
                        })?;
                        Value::big_int(l - r)
                    }
                    BinOp::IntMul => {
                        if let (Value::SmallInt(l), Value::SmallInt(r)) = (&*lv, &*rv) {
                            if let Some(prod) = l.checked_mul(*r) {
                                return Ok(Cow::Owned(Value::SmallInt(prod)));
                            }
                        }
                        let l = lv.to_bigint().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: lv.type_name(),
                            span: None,
                        })?;
                        let r = rv.to_bigint().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: rv.type_name(),
                            span: None,
                        })?;
                        Value::big_int(l * r)
                    }
                    BinOp::IntDiv => {
                        if let (Value::SmallInt(l), Value::SmallInt(r)) = (&*lv, &*rv) {
                            if *r == 0 {
                                return Err(EvalError::DivisionByZero { span: None });
                            }
                            return Ok(Cow::Owned(Value::SmallInt(l.div_floor(r))));
                        }
                        let l = lv.to_bigint().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: lv.type_name(),
                            span: None,
                        })?;
                        let r = rv.to_bigint().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: rv.type_name(),
                            span: None,
                        })?;
                        if r.is_zero() {
                            return Err(EvalError::DivisionByZero { span: None });
                        }
                        Value::big_int(l.div_floor(&r))
                    }
                    BinOp::IntMod => {
                        if let (Value::SmallInt(l), Value::SmallInt(r)) = (&*lv, &*rv) {
                            if *r == 0 {
                                return Err(EvalError::DivisionByZero { span: None });
                            }
                            return Ok(Cow::Owned(Value::SmallInt(l.mod_floor(r))));
                        }
                        let l = lv.to_bigint().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: lv.type_name(),
                            span: None,
                        })?;
                        let r = rv.to_bigint().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: rv.type_name(),
                            span: None,
                        })?;
                        if r.is_zero() {
                            return Err(EvalError::DivisionByZero { span: None });
                        }
                        Value::big_int(l.mod_floor(&r))
                    }
                    BinOp::SetIntersect => {
                        let ls = lv.to_sorted_set().ok_or_else(|| EvalError::TypeError {
                            expected: "set",
                            got: lv.type_name(),
                            span: None,
                        })?;
                        let rs = rv.to_sorted_set().ok_or_else(|| EvalError::TypeError {
                            expected: "set",
                            got: rv.type_name(),
                            span: None,
                        })?;
                        Value::Set(ls.intersection(&rs))
                    }
                };
                Ok(Cow::Owned(result))
            }

            CompiledExpr::Except { func, key, value } => {
                // EXCEPT needs owned values for modification, but use eval_cow for key
                // since IntFunc.except() takes key by reference, avoiding unnecessary clone
                let func_val = func.eval_with_arrays(ctx, current_array, next_array)?;
                let key_cow = key.eval_cow(ctx, current_array, next_array)?;
                let new_val = value.eval_with_arrays(ctx, current_array, next_array)?;

                let result = match func_val {
                    Value::Func(f) => {
                        Value::Func(f.except(key_cow.into_owned(), new_val))
                    }
                    Value::IntFunc(f) => {
                        Value::IntFunc(f.except(&key_cow, new_val))
                    }
                    _ if func_val.as_seq_or_tuple_elements().is_some() => {
                        // Seq/Tuple EXCEPT: [seq EXCEPT ![i] = v] updates index i (1-indexed)
                        let s = func_val.as_seq_or_tuple_elements().unwrap();
                        if let Some(idx) = key_cow.as_i64() {
                            let index = (idx - 1) as usize; // TLA+ is 1-indexed
                            if index >= s.len() {
                                return Err(EvalError::IndexOutOfBounds {
                                    index: idx,
                                    len: s.len(),
                                    span: None,
                                });
                            }
                            // Short-circuit: if value unchanged, return original without cloning
                            if s[index] == new_val {
                                return Ok(Cow::Owned(func_val));
                            }
                            let mut new_vec: Vec<Value> = s.to_vec();
                            new_vec[index] = new_val;
                            // Return same type as input
                            if matches!(func_val, Value::Seq(_)) {
                                Value::Seq(new_vec.into())
                            } else {
                                Value::Tuple(new_vec.into())
                            }
                        } else {
                            return Err(EvalError::TypeError {
                                expected: "Int",
                                got: key_cow.type_name(),
                                span: None,
                            });
                        }
                    }
                    Value::Record(r) => {
                        if let Value::String(s) = &*key_cow {
                            Value::Record(r.update(s.clone(), new_val))
                        } else {
                            return Err(EvalError::TypeError {
                                expected: "String",
                                got: key_cow.type_name(),
                                span: None,
                            });
                        }
                    }
                    _ => {
                        return Err(EvalError::TypeError {
                            expected: "function, sequence, or record",
                            got: func_val.type_name(),
                            span: None,
                        })
                    }
                };
                Ok(Cow::Owned(result))
            }

            CompiledExpr::ExceptNested { func, keys, value } => {
                let func_val = func.eval_with_arrays(ctx, current_array, next_array)?;
                let new_val = value.eval_with_arrays(ctx, current_array, next_array)?;

                let key_vals: Vec<Value> = keys
                    .iter()
                    .map(|k| k.eval_with_arrays(ctx, current_array, next_array))
                    .collect::<Result<_, _>>()?;

                Ok(Cow::Owned(apply_nested_except(
                    func_val, &key_vals, new_val,
                )?))
            }

            CompiledExpr::SetEnum(elems) => {
                let mut set = SetBuilder::new();
                for elem in elems {
                    set.insert(elem.eval_cow(ctx, current_array, next_array)?.into_owned());
                }
                Ok(Cow::Owned(set.build_value()))
            }

            CompiledExpr::Tuple(elems) => {
                let mut vals = Vec::with_capacity(elems.len());
                for elem in elems {
                    vals.push(elem.eval_cow(ctx, current_array, next_array)?.into_owned());
                }
                Ok(Cow::Owned(Value::Tuple(vals.into())))
            }

            CompiledExpr::Record(fields) => {
                let mut builder = RecordBuilder::with_capacity(fields.len());
                for (name, expr) in fields {
                    let val = expr.eval_cow(ctx, current_array, next_array)?.into_owned();
                    builder.insert(Arc::clone(name), val);
                }
                Ok(Cow::Owned(Value::Record(builder.build())))
            }

            CompiledExpr::BoolEq { left, right } => {
                let left_val = left.eval_cow(ctx, current_array, next_array)?;
                let right_val = right.eval_cow(ctx, current_array, next_array)?;
                Ok(Cow::Owned(Value::Bool(*left_val == *right_val)))
            }

            // Dynamic function application: func[key] where func is a compiled expression
            CompiledExpr::DynFuncApp { func, key } => {
                let func_val = func.eval_cow(ctx, current_array, next_array)?;
                let key_val = key.eval_cow(ctx, current_array, next_array)?;

                let result = match &*func_val {
                    Value::Func(f) => {
                        f.apply(&key_val)
                            .cloned()
                            .ok_or_else(|| EvalError::NotInDomain {
                                arg: format!("{key_val:?}"),
                                span: None,
                            })?
                    }
                    Value::IntFunc(f) => {
                        f.apply(&key_val)
                            .cloned()
                            .ok_or_else(|| EvalError::NotInDomain {
                                arg: format!("{key_val:?}"),
                                span: None,
                            })?
                    }
                    _ if func_val.as_seq_or_tuple_elements().is_some() => {
                        let t = func_val.as_seq_or_tuple_elements().unwrap();
                        if let Some(idx) = key_val.as_i64() {
                            let index = (idx - 1) as usize;
                            t.get(index).cloned().ok_or(EvalError::IndexOutOfBounds {
                                index: idx,
                                len: t.len(),
                                span: None,
                            })?
                        } else {
                            return Err(EvalError::TypeError {
                                expected: "Int",
                                got: key_val.type_name(),
                                span: None,
                            });
                        }
                    }
                    Value::Record(r) => {
                        if let Value::String(s) = &*key_val {
                            r.get(s.as_ref())
                                .cloned()
                                .ok_or_else(|| EvalError::NoSuchField {
                                    field: s.to_string(),
                                    span: None,
                                })?
                        } else {
                            return Err(EvalError::TypeError {
                                expected: "String",
                                got: key_val.type_name(),
                                span: None,
                            });
                        }
                    }
                    _ => {
                        return Err(EvalError::TypeError {
                            expected: "function, tuple, sequence, or record",
                            got: func_val.type_name(),
                            span: None,
                        });
                    }
                };
                Ok(Cow::Owned(result))
            }

            CompiledExpr::Not(inner) => {
                let val = inner.eval_cow(ctx, current_array, next_array)?;
                match &*val {
                    Value::Bool(b) => Ok(Cow::Owned(Value::Bool(!b))),
                    _ => Err(EvalError::TypeError {
                        expected: "Bool",
                        got: val.type_name(),
                        span: None,
                    }),
                }
            }

            CompiledExpr::IfThenElse {
                condition,
                then_branch,
                else_branch,
            } => {
                let cond_val = condition.eval_cow(ctx, current_array, next_array)?;
                match &*cond_val {
                    Value::Bool(true) => then_branch.eval_cow(ctx, current_array, next_array),
                    Value::Bool(false) => else_branch.eval_cow(ctx, current_array, next_array),
                    _ => Err(EvalError::TypeError {
                        expected: "Bool",
                        got: cond_val.type_name(),
                        span: None,
                    }),
                }
            }

            CompiledExpr::FuncDef {
                var_name,
                domain,
                body,
            } => {
                // Evaluate the domain to get the set of keys
                let domain_val = domain.eval_cow(ctx, current_array, next_array)?;

                // Handle IntInterval domain (common case: 1..N)
                if let Value::Interval(intv) = &*domain_val {
                    if let (Some(min), Some(max)) = (intv.low.to_i64(), intv.high.to_i64()) {
                        let size = (max - min + 1) as usize;
                        if size <= 1_000_000 {
                            let mut local_ctx = ctx.clone();
                            let mark = local_ctx.mark_stack();
                            let mut values = Vec::with_capacity(size);
                            for i in min..=max {
                                let elem = Value::SmallInt(i);
                                local_ctx.push_binding(Arc::clone(var_name), elem.clone());
                                local_ctx.bind_mut(Arc::clone(var_name), elem);
                                let val = body
                                    .eval_cow(&local_ctx, current_array, next_array)?
                                    .into_owned();
                                values.push(val);
                                local_ctx.pop_to_mark(mark);
                            }
                            // If domain is 1..n, this is a sequence
                            if min == 1 {
                                return Ok(Cow::Owned(Value::Seq(values.into())));
                            }
                            return Ok(Cow::Owned(Value::IntFunc(IntIntervalFunc::new(
                                min, max, values,
                            ))));
                        }
                    }
                }

                // General set domain
                let domain_set =
                    domain_val
                        .to_sorted_set()
                        .ok_or_else(|| EvalError::TypeError {
                            expected: "set",
                            got: domain_val.type_name(),
                            span: None,
                        })?;

                let mut local_ctx = ctx.clone();
                let mark = local_ctx.mark_stack();

                // Check for sequence domain {1, 2, ..., n} (empty set => empty sequence)
                let mut is_seq_domain = true;
                let mut expected: i64 = 1;

                let mut entries: Vec<(Value, Value)> = Vec::with_capacity(domain_set.len());
                for elem in domain_set.iter() {
                    if is_seq_domain {
                        if elem.as_i64() == Some(expected) {
                            expected += 1;
                        } else {
                            is_seq_domain = false;
                        }
                    }

                    let elem_owned = elem.clone();
                    local_ctx.push_binding(Arc::clone(var_name), elem_owned.clone());
                    local_ctx.bind_mut(Arc::clone(var_name), elem_owned.clone());
                    let val = body
                        .eval_cow(&local_ctx, current_array, next_array)?
                        .into_owned();
                    entries.push((elem_owned, val));
                    local_ctx.pop_to_mark(mark);
                }

                if is_seq_domain {
                    // Domain is 1..n; entries were collected in key-sorted order.
                    let seq_values: Vec<Value> = entries.into_iter().map(|(_, v)| v).collect();
                    return Ok(Cow::Owned(Value::Seq(seq_values.into())));
                }

                Ok(Cow::Owned(Value::Func(FuncValue::from_sorted_entries(
                    entries,
                ))))
            }

            CompiledExpr::SeqHead(seq_expr) => {
                let seq_val = seq_expr.eval_cow(ctx, current_array, next_array)?;
                if let Some(s) = seq_val.as_seq_or_tuple_elements() {
                    if s.is_empty() {
                        return Err(EvalError::IndexOutOfBounds {
                            index: 1,
                            len: 0,
                            span: None,
                        });
                    }
                    Ok(Cow::Owned(s[0].clone()))
                } else {
                    Err(EvalError::TypeError {
                        expected: "sequence",
                        got: seq_val.type_name(),
                        span: None,
                    })
                }
            }

            CompiledExpr::SeqTail(seq_expr) => {
                let seq_val = seq_expr.eval_cow(ctx, current_array, next_array)?;
                if let Some(s) = seq_val.as_seq_or_tuple_elements() {
                    if s.is_empty() {
                        return Err(EvalError::IndexOutOfBounds {
                            index: 1,
                            len: 0,
                            span: None,
                        });
                    }
                    Ok(Cow::Owned(Value::Seq(s[1..].to_vec().into())))
                } else {
                    Err(EvalError::TypeError {
                        expected: "sequence",
                        got: seq_val.type_name(),
                        span: None,
                    })
                }
            }

            CompiledExpr::SeqAppend { seq, elem } => {
                let seq_val = seq.eval_cow(ctx, current_array, next_array)?;
                let elem_val = elem.eval_cow(ctx, current_array, next_array)?.into_owned();
                match &*seq_val {
                    Value::Seq(s) => {
                        let mut new_seq = s.to_vec();
                        new_seq.push(elem_val);
                        Ok(Cow::Owned(Value::Seq(new_seq.into())))
                    }
                    Value::Tuple(s) => {
                        let mut new_seq = s.to_vec();
                        new_seq.push(elem_val);
                        Ok(Cow::Owned(Value::Seq(new_seq.into())))
                    }
                    _ => Err(EvalError::TypeError {
                        expected: "sequence",
                        got: seq_val.type_name(),
                        span: None,
                    }),
                }
            }

            CompiledExpr::SeqLen(seq_expr) => {
                let seq_val = seq_expr.eval_cow(ctx, current_array, next_array)?;
                // Check for Seq/Tuple first
                if let Some(s) = seq_val.as_seq_or_tuple_elements() {
                    return Ok(Cow::Owned(Value::SmallInt(s.len() as i64)));
                }
                match &*seq_val {
                    Value::String(s) => Ok(Cow::Owned(Value::SmallInt(s.len() as i64))),
                    Value::IntFunc(f) => {
                        if f.min == 1 {
                            Ok(Cow::Owned(Value::SmallInt(f.len() as i64)))
                        } else {
                            Err(EvalError::TypeError {
                                expected: "sequence or string",
                                got: seq_val.type_name(),
                                span: None,
                            })
                        }
                    }
                    Value::Func(f) => {
                        // In TLA+, sequences are functions with domain 1..n.
                        // Only accept functions whose domain is exactly {1, 2, ..., n} (or empty).
                        let mut expected: i64 = 1;
                        for key in f.domain_iter() {
                            let Some(k) = key.as_i64() else {
                                return Err(EvalError::TypeError {
                                    expected: "sequence or string",
                                    got: seq_val.type_name(),
                                    span: None,
                                });
                            };
                            if k != expected {
                                return Err(EvalError::TypeError {
                                    expected: "sequence or string",
                                    got: seq_val.type_name(),
                                    span: None,
                                });
                            }
                            expected += 1;
                        }
                        Ok(Cow::Owned(Value::SmallInt(expected - 1)))
                    }
                    _ => Err(EvalError::TypeError {
                        expected: "sequence or string",
                        got: seq_val.type_name(),
                        span: None,
                    }),
                }
            }

            CompiledExpr::SetCardinality(set_expr) => {
                let set_val = set_expr.eval_cow(ctx, current_array, next_array)?;
                match set_val.set_len() {
                    Some(c) => Ok(Cow::Owned(Value::big_int(c))),
                    None => Err(EvalError::TypeError {
                        expected: "set",
                        got: set_val.type_name(),
                        span: None,
                    }),
                }
            }

            CompiledExpr::SeqRange(seq_expr) => {
                let seq_val = seq_expr.eval_cow(ctx, current_array, next_array)?;
                if let Some(s) = seq_val.as_seq_or_tuple_elements() {
                    let set = SortedSet::from_iter(s.iter().cloned());
                    Ok(Cow::Owned(Value::Set(set)))
                } else {
                    Err(EvalError::TypeError {
                        expected: "sequence",
                        got: seq_val.type_name(),
                        span: None,
                    })
                }
            }

            CompiledExpr::RecordAccess { record, field } => {
                let record_val = record.eval_cow(ctx, current_array, next_array)?;
                match &*record_val {
                    Value::Record(fields) => fields
                        .get(field.as_ref())
                        .cloned()
                        .map(Cow::Owned)
                        .ok_or_else(|| EvalError::NoSuchField {
                            field: field.to_string(),
                            span: None,
                        }),
                    _ => Err(EvalError::TypeError {
                        expected: "record",
                        got: record_val.type_name(),
                        span: None,
                    }),
                }
            }

            CompiledExpr::Fallback { expr, reads_next } => {
                // Same logic as eval_with_arrays: provide next_state context for primed lookups
                // OPTIMIZATION (Re: #56): Skip HashMap rebuild if next_state_env is already set.
                if *reads_next && !std::ptr::eq(current_array, next_array) {
                    // Fast path: use pre-bound next_state_env for O(1) lookups
                    if ctx.has_next_state_env() {
                        return eval(ctx, expr).map(Cow::Owned);
                    }
                    // Slow path: build HashMap
                    let registry = ctx.var_registry();
                    let mut next_state = Env::new();
                    for (idx, name) in registry.iter() {
                        next_state
                            .insert(Arc::clone(name), next_array.values()[idx.as_usize()].clone());
                    }
                    let eval_ctx = ctx.with_next_state(next_state);
                    eval(&eval_ctx, expr).map(Cow::Owned)
                } else {
                    eval(ctx, expr).map(Cow::Owned)
                }
            }
        }
    }

    /// Evaluate a compiled expression against a raw value slice.
    ///
    /// This is used during initial state enumeration filtering where we have values
    /// as a `&[Value]` slice (from `enumerate_constraints_to_bulk`) rather than an
    /// `ArrayState`. This avoids creating temporary ArrayState wrappers.
    ///
    /// Note: `StateRef::Next` is not supported and will return an error.
    #[inline]
    pub fn eval_with_values<'a>(
        &'a self,
        ctx: &'a EvalCtx,
        values: &'a [Value],
    ) -> EvalResult<Cow<'a, Value>> {
        match self {
            CompiledExpr::Const(v) => Ok(Cow::Borrowed(v)),

            CompiledExpr::StateVar { state, idx } => match state {
                StateRef::Current => Ok(Cow::Borrowed(&values[idx.0 as usize])),
                StateRef::Next => Err(EvalError::Internal {
                    message: "Next state not available during enumeration filter".into(),
                    span: None,
                }),
            },

            CompiledExpr::LocalVar { name, depth } => ctx
                .get_local_by_depth(*depth)
                .map(Cow::Borrowed)
                .ok_or_else(|| EvalError::UndefinedVar {
                    name: name.to_string(),
                    span: None,
                }),

            CompiledExpr::FuncApp {
                state,
                func_var,
                key,
            } => {
                if *state == StateRef::Next {
                    return Err(EvalError::Internal {
                        message: "Next state not available during enumeration filter".into(),
                        span: None,
                    });
                }
                let func_val = &values[func_var.0 as usize];
                let key_val = key.eval_with_values(ctx, values)?;

                match func_val {
                    Value::Func(f) => {
                        f.apply(&key_val)
                            .map(Cow::Borrowed)
                            .ok_or_else(|| EvalError::NotInDomain {
                                arg: format!("{key_val:?}"),
                                span: None,
                            })
                    }
                    Value::IntFunc(f) => {
                        f.apply(&key_val)
                            .map(Cow::Borrowed)
                            .ok_or_else(|| EvalError::NotInDomain {
                                arg: format!("{key_val:?}"),
                                span: None,
                            })
                    }
                    Value::Record(r) => {
                        if let Value::String(s) = &*key_val {
                            r.get(s.as_ref()).map(Cow::Borrowed).ok_or_else(|| {
                                EvalError::NoSuchField {
                                    field: s.to_string(),
                                    span: None,
                                }
                            })
                        } else {
                            Err(EvalError::TypeError {
                                expected: "String",
                                got: key_val.type_name(),
                                span: None,
                            })
                        }
                    }
                    Value::Seq(s) => {
                        // Sequence indexing is 1-based
                        let idx = key_val.as_i64().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: key_val.type_name(),
                            span: None,
                        })?;
                        if idx < 1 || idx as usize > s.len() {
                            return Err(EvalError::IndexOutOfBounds {
                                index: idx,
                                len: s.len(),
                                span: None,
                            });
                        }
                        Ok(Cow::Owned(s[(idx - 1) as usize].clone()))
                    }
                    Value::Tuple(t) => {
                        // Tuple indexing is 1-based
                        let idx = key_val.as_i64().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: key_val.type_name(),
                            span: None,
                        })?;
                        if idx < 1 || idx as usize > t.len() {
                            return Err(EvalError::IndexOutOfBounds {
                                index: idx,
                                len: t.len(),
                                span: None,
                            });
                        }
                        Ok(Cow::Owned(t[(idx - 1) as usize].clone()))
                    }
                    _ => Err(EvalError::TypeError {
                        expected: "function, sequence, tuple, or record",
                        got: func_val.type_name(),
                        span: None,
                    }),
                }
            }

            // Specialized: state_var[local_var] for IntFunc
            CompiledExpr::FuncAppIntFuncLocalVar {
                state,
                func_var,
                key_depth,
            } => {
                if *state == StateRef::Next {
                    return Err(EvalError::Internal {
                        message: "Next state not available during enumeration filter".into(),
                        span: None,
                    });
                }

                let key_val =
                    ctx.get_local_by_depth(*key_depth)
                        .ok_or_else(|| EvalError::Internal {
                            message: "FuncAppIntFuncLocalVar: local var not found".into(),
                            span: None,
                        })?;

                let func_val = &values[func_var.0 as usize];

                // Fast path for IntFunc with integer key
                if let Value::IntFunc(f) = func_val {
                    let key_i64 = match key_val {
                        Value::SmallInt(n) => *n,
                        Value::Int(n) => n.to_i64().ok_or_else(|| EvalError::Internal {
                            message: "FuncAppIntFuncLocalVar: key too large".into(),
                            span: None,
                        })?,
                        _ => {
                            return Err(EvalError::TypeError {
                                expected: "Int",
                                got: key_val.type_name(),
                                span: None,
                            })
                        }
                    };
                    if key_i64 >= f.min && key_i64 <= f.max {
                        let result = &f.values[(key_i64 - f.min) as usize];
                        return Ok(Cow::Borrowed(result));
                    }
                    return Err(EvalError::NotInDomain {
                        arg: format!("{}", key_i64),
                        span: None,
                    });
                }

                // Fallback for regular Func
                match func_val {
                    Value::Func(f) => {
                        f.apply(key_val)
                            .map(Cow::Borrowed)
                            .ok_or_else(|| EvalError::NotInDomain {
                                arg: format!("{:?}", key_val),
                                span: None,
                            })
                    }
                    Value::Record(r) => {
                        if let Value::String(s) = key_val {
                            r.get(s.as_ref()).map(Cow::Borrowed).ok_or_else(|| {
                                EvalError::NoSuchField {
                                    field: s.to_string(),
                                    span: None,
                                }
                            })
                        } else {
                            Err(EvalError::TypeError {
                                expected: "String",
                                got: key_val.type_name(),
                                span: None,
                            })
                        }
                    }
                    Value::Seq(s) => {
                        // Sequence indexing is 1-based
                        let idx = key_val.as_i64().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: key_val.type_name(),
                            span: None,
                        })?;
                        if idx < 1 || idx as usize > s.len() {
                            return Err(EvalError::IndexOutOfBounds {
                                index: idx,
                                len: s.len(),
                                span: None,
                            });
                        }
                        Ok(Cow::Owned(s[(idx - 1) as usize].clone()))
                    }
                    Value::Tuple(t) => {
                        // Tuple indexing is 1-based
                        let idx = key_val.as_i64().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: key_val.type_name(),
                            span: None,
                        })?;
                        if idx < 1 || idx as usize > t.len() {
                            return Err(EvalError::IndexOutOfBounds {
                                index: idx,
                                len: t.len(),
                                span: None,
                            });
                        }
                        Ok(Cow::Owned(t[(idx - 1) as usize].clone()))
                    }
                    _ => Err(EvalError::TypeError {
                        expected: "function, sequence, tuple, or record",
                        got: func_val.type_name(),
                        span: None,
                    }),
                }
            }

            CompiledExpr::BinOp { left, op, right } => {
                let lv = left.eval_with_values(ctx, values)?;
                let rv = right.eval_with_values(ctx, values)?;

                let result = match op {
                    BinOp::SetUnion => {
                        let ls = lv.to_sorted_set().ok_or_else(|| EvalError::TypeError {
                            expected: "set",
                            got: lv.type_name(),
                            span: None,
                        })?;
                        let rs = rv.to_sorted_set().ok_or_else(|| EvalError::TypeError {
                            expected: "set",
                            got: rv.type_name(),
                            span: None,
                        })?;
                        Value::Set(ls.union(&rs))
                    }
                    BinOp::SetDiff => {
                        let ls = lv.to_sorted_set().ok_or_else(|| EvalError::TypeError {
                            expected: "set",
                            got: lv.type_name(),
                            span: None,
                        })?;
                        let rs = rv.to_sorted_set().ok_or_else(|| EvalError::TypeError {
                            expected: "set",
                            got: rv.type_name(),
                            span: None,
                        })?;
                        Value::Set(ls.difference(&rs))
                    }
                    BinOp::IntAdd => {
                        if let (Value::SmallInt(l), Value::SmallInt(r)) = (&*lv, &*rv) {
                            if let Some(sum) = l.checked_add(*r) {
                                return Ok(Cow::Owned(Value::SmallInt(sum)));
                            }
                        }
                        let l = lv.to_bigint().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: lv.type_name(),
                            span: None,
                        })?;
                        let r = rv.to_bigint().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: rv.type_name(),
                            span: None,
                        })?;
                        Value::big_int(l + r)
                    }
                    BinOp::IntSub => {
                        if let (Value::SmallInt(l), Value::SmallInt(r)) = (&*lv, &*rv) {
                            if let Some(diff) = l.checked_sub(*r) {
                                return Ok(Cow::Owned(Value::SmallInt(diff)));
                            }
                        }
                        let l = lv.to_bigint().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: lv.type_name(),
                            span: None,
                        })?;
                        let r = rv.to_bigint().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: rv.type_name(),
                            span: None,
                        })?;
                        Value::big_int(l - r)
                    }
                    BinOp::IntMul => {
                        if let (Value::SmallInt(l), Value::SmallInt(r)) = (&*lv, &*rv) {
                            if let Some(prod) = l.checked_mul(*r) {
                                return Ok(Cow::Owned(Value::SmallInt(prod)));
                            }
                        }
                        let l = lv.to_bigint().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: lv.type_name(),
                            span: None,
                        })?;
                        let r = rv.to_bigint().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: rv.type_name(),
                            span: None,
                        })?;
                        Value::big_int(l * r)
                    }
                    BinOp::IntDiv => {
                        if let (Value::SmallInt(l), Value::SmallInt(r)) = (&*lv, &*rv) {
                            if *r == 0 {
                                return Err(EvalError::DivisionByZero { span: None });
                            }
                            return Ok(Cow::Owned(Value::SmallInt(l.div_floor(r))));
                        }
                        let l = lv.to_bigint().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: lv.type_name(),
                            span: None,
                        })?;
                        let r = rv.to_bigint().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: rv.type_name(),
                            span: None,
                        })?;
                        if r.is_zero() {
                            return Err(EvalError::DivisionByZero { span: None });
                        }
                        Value::big_int(l.div_floor(&r))
                    }
                    BinOp::IntMod => {
                        if let (Value::SmallInt(l), Value::SmallInt(r)) = (&*lv, &*rv) {
                            if *r == 0 {
                                return Err(EvalError::DivisionByZero { span: None });
                            }
                            return Ok(Cow::Owned(Value::SmallInt(l.mod_floor(r))));
                        }
                        let l = lv.to_bigint().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: lv.type_name(),
                            span: None,
                        })?;
                        let r = rv.to_bigint().ok_or_else(|| EvalError::TypeError {
                            expected: "Int",
                            got: rv.type_name(),
                            span: None,
                        })?;
                        if r.is_zero() {
                            return Err(EvalError::DivisionByZero { span: None });
                        }
                        Value::big_int(l.mod_floor(&r))
                    }
                    BinOp::SetIntersect => {
                        let ls = lv.to_sorted_set().ok_or_else(|| EvalError::TypeError {
                            expected: "set",
                            got: lv.type_name(),
                            span: None,
                        })?;
                        let rs = rv.to_sorted_set().ok_or_else(|| EvalError::TypeError {
                            expected: "set",
                            got: rv.type_name(),
                            span: None,
                        })?;
                        Value::Set(ls.intersection(&rs))
                    }
                };
                Ok(Cow::Owned(result))
            }

            CompiledExpr::Except { func, key, value } => {
                let func_val = func.eval_with_values(ctx, values)?.into_owned();
                let key_cow = key.eval_with_values(ctx, values)?;
                let new_val = value.eval_with_values(ctx, values)?.into_owned();

                let result = match func_val {
                    Value::Func(f) => {
                        Value::Func(f.except(key_cow.into_owned(), new_val))
                    }
                    Value::IntFunc(f) => {
                        Value::IntFunc(f.except(&key_cow, new_val))
                    }
                    _ if func_val.as_seq_or_tuple_elements().is_some() => {
                        // Seq/Tuple EXCEPT: [seq EXCEPT ![i] = v] updates index i (1-indexed)
                        let s = func_val.as_seq_or_tuple_elements().unwrap();
                        if let Some(idx) = key_cow.as_i64() {
                            let index = (idx - 1) as usize; // TLA+ is 1-indexed
                            if index >= s.len() {
                                return Err(EvalError::IndexOutOfBounds {
                                    index: idx,
                                    len: s.len(),
                                    span: None,
                                });
                            }
                            // Short-circuit: if value unchanged, return original without cloning
                            if s[index] == new_val {
                                return Ok(Cow::Owned(func_val));
                            }
                            let mut new_vec: Vec<Value> = s.to_vec();
                            new_vec[index] = new_val;
                            // Return same type as input
                            if matches!(func_val, Value::Seq(_)) {
                                Value::Seq(new_vec.into())
                            } else {
                                Value::Tuple(new_vec.into())
                            }
                        } else {
                            return Err(EvalError::TypeError {
                                expected: "Int",
                                got: key_cow.type_name(),
                                span: None,
                            });
                        }
                    }
                    Value::Record(r) => {
                        if let Value::String(s) = &*key_cow {
                            Value::Record(r.update(s.clone(), new_val))
                        } else {
                            return Err(EvalError::TypeError {
                                expected: "String",
                                got: key_cow.type_name(),
                                span: None,
                            });
                        }
                    }
                    _ => {
                        return Err(EvalError::TypeError {
                            expected: "function, sequence, or record",
                            got: func_val.type_name(),
                            span: None,
                        })
                    }
                };
                Ok(Cow::Owned(result))
            }

            CompiledExpr::ExceptNested { func, keys, value } => {
                let func_val = func.eval_with_values(ctx, values)?.into_owned();
                let new_val = value.eval_with_values(ctx, values)?.into_owned();

                let key_vals: Vec<Value> = keys
                    .iter()
                    .map(|k| k.eval_with_values(ctx, values).map(|v| v.into_owned()))
                    .collect::<Result<_, _>>()?;

                Ok(Cow::Owned(apply_nested_except(
                    func_val, &key_vals, new_val,
                )?))
            }

            CompiledExpr::SetEnum(elems) => {
                let mut set = SetBuilder::new();
                for elem in elems {
                    set.insert(elem.eval_with_values(ctx, values)?.into_owned());
                }
                Ok(Cow::Owned(set.build_value()))
            }

            CompiledExpr::Tuple(elems) => {
                let mut vals = Vec::with_capacity(elems.len());
                for elem in elems {
                    vals.push(elem.eval_with_values(ctx, values)?.into_owned());
                }
                Ok(Cow::Owned(Value::Tuple(vals.into())))
            }

            CompiledExpr::Record(fields) => {
                let mut builder = RecordBuilder::with_capacity(fields.len());
                for (name, expr) in fields {
                    let val = expr.eval_with_values(ctx, values)?.into_owned();
                    builder.insert(Arc::clone(name), val);
                }
                Ok(Cow::Owned(Value::Record(builder.build())))
            }

            CompiledExpr::BoolEq { left, right } => {
                let left_val = left.eval_with_values(ctx, values)?;
                let right_val = right.eval_with_values(ctx, values)?;
                Ok(Cow::Owned(Value::Bool(*left_val == *right_val)))
            }

            // Dynamic function application: func[key] where func is a compiled expression
            CompiledExpr::DynFuncApp { func, key } => {
                let func_val = func.eval_with_values(ctx, values)?;
                let key_val = key.eval_with_values(ctx, values)?;

                let result = match &*func_val {
                    Value::Func(f) => {
                        f.apply(&key_val)
                            .cloned()
                            .ok_or_else(|| EvalError::NotInDomain {
                                arg: format!("{key_val:?}"),
                                span: None,
                            })?
                    }
                    Value::IntFunc(f) => {
                        f.apply(&key_val)
                            .cloned()
                            .ok_or_else(|| EvalError::NotInDomain {
                                arg: format!("{key_val:?}"),
                                span: None,
                            })?
                    }
                    _ if func_val.as_seq_or_tuple_elements().is_some() => {
                        let t = func_val.as_seq_or_tuple_elements().unwrap();
                        if let Some(idx) = key_val.as_i64() {
                            let index = (idx - 1) as usize;
                            t.get(index).cloned().ok_or(EvalError::IndexOutOfBounds {
                                index: idx,
                                len: t.len(),
                                span: None,
                            })?
                        } else {
                            return Err(EvalError::TypeError {
                                expected: "Int",
                                got: key_val.type_name(),
                                span: None,
                            });
                        }
                    }
                    Value::Record(r) => {
                        if let Value::String(s) = &*key_val {
                            r.get(s.as_ref())
                                .cloned()
                                .ok_or_else(|| EvalError::NoSuchField {
                                    field: s.to_string(),
                                    span: None,
                                })?
                        } else {
                            return Err(EvalError::TypeError {
                                expected: "String",
                                got: key_val.type_name(),
                                span: None,
                            });
                        }
                    }
                    _ => {
                        return Err(EvalError::TypeError {
                            expected: "function, tuple, sequence, or record",
                            got: func_val.type_name(),
                            span: None,
                        });
                    }
                };
                Ok(Cow::Owned(result))
            }

            CompiledExpr::Not(inner) => {
                let val = inner.eval_with_values(ctx, values)?;
                match &*val {
                    Value::Bool(b) => Ok(Cow::Owned(Value::Bool(!b))),
                    _ => Err(EvalError::TypeError {
                        expected: "Bool",
                        got: val.type_name(),
                        span: None,
                    }),
                }
            }

            CompiledExpr::IfThenElse {
                condition,
                then_branch,
                else_branch,
            } => {
                let cond_val = condition.eval_with_values(ctx, values)?;
                match &*cond_val {
                    Value::Bool(true) => then_branch.eval_with_values(ctx, values),
                    Value::Bool(false) => else_branch.eval_with_values(ctx, values),
                    _ => Err(EvalError::TypeError {
                        expected: "Bool",
                        got: cond_val.type_name(),
                        span: None,
                    }),
                }
            }

            CompiledExpr::FuncDef {
                var_name,
                domain,
                body,
            } => {
                // Evaluate the domain to get the set of keys
                let domain_val = domain.eval_with_values(ctx, values)?;

                // Handle IntInterval domain (common case: 1..N)
                if let Value::Interval(intv) = &*domain_val {
                    if let (Some(min), Some(max)) = (intv.low.to_i64(), intv.high.to_i64()) {
                        let size = (max - min + 1) as usize;
                        if size <= 1_000_000 {
                            let mut local_ctx = ctx.clone();
                            let mark = local_ctx.mark_stack();
                            let mut func_values = Vec::with_capacity(size);
                            for i in min..=max {
                                let elem = Value::SmallInt(i);
                                local_ctx.push_binding(Arc::clone(var_name), elem.clone());
                                local_ctx.bind_mut(Arc::clone(var_name), elem);
                                let val = body.eval_with_values(&local_ctx, values)?.into_owned();
                                func_values.push(val);
                                local_ctx.pop_to_mark(mark);
                            }
                            // If domain is 1..n, this is a sequence
                            if min == 1 {
                                return Ok(Cow::Owned(Value::Seq(func_values.into())));
                            }
                            return Ok(Cow::Owned(Value::IntFunc(IntIntervalFunc::new(
                                min,
                                max,
                                func_values,
                            ))));
                        }
                    }
                }

                // General set domain
                let domain_set =
                    domain_val
                        .to_sorted_set()
                        .ok_or_else(|| EvalError::TypeError {
                            expected: "set",
                            got: domain_val.type_name(),
                            span: None,
                        })?;

                let mut local_ctx = ctx.clone();
                let mark = local_ctx.mark_stack();

                // Check for sequence domain {1, 2, ..., n} (empty set => empty sequence)
                let mut is_seq_domain = true;
                let mut expected: i64 = 1;

                let mut entries: Vec<(Value, Value)> = Vec::with_capacity(domain_set.len());
                for elem in domain_set.iter() {
                    if is_seq_domain {
                        if elem.as_i64() == Some(expected) {
                            expected += 1;
                        } else {
                            is_seq_domain = false;
                        }
                    }

                    let elem_owned = elem.clone();
                    local_ctx.push_binding(Arc::clone(var_name), elem_owned.clone());
                    local_ctx.bind_mut(Arc::clone(var_name), elem_owned.clone());
                    let val = body.eval_with_values(&local_ctx, values)?.into_owned();
                    entries.push((elem_owned, val));
                    local_ctx.pop_to_mark(mark);
                }

                if is_seq_domain {
                    // Domain is 1..n; entries were collected in key-sorted order.
                    let seq_values: Vec<Value> = entries.into_iter().map(|(_, v)| v).collect();
                    return Ok(Cow::Owned(Value::Seq(seq_values.into())));
                }

                Ok(Cow::Owned(Value::Func(FuncValue::from_sorted_entries(
                    entries,
                ))))
            }

            CompiledExpr::SeqHead(seq_expr) => {
                let seq_val = seq_expr.eval_with_values(ctx, values)?;
                if let Some(s) = seq_val.as_seq_or_tuple_elements() {
                    if s.is_empty() {
                        return Err(EvalError::IndexOutOfBounds {
                            index: 1,
                            len: 0,
                            span: None,
                        });
                    }
                    Ok(Cow::Owned(s[0].clone()))
                } else {
                    Err(EvalError::TypeError {
                        expected: "sequence",
                        got: seq_val.type_name(),
                        span: None,
                    })
                }
            }

            CompiledExpr::SeqTail(seq_expr) => {
                let seq_val = seq_expr.eval_with_values(ctx, values)?;
                if let Some(s) = seq_val.as_seq_or_tuple_elements() {
                    if s.is_empty() {
                        return Err(EvalError::IndexOutOfBounds {
                            index: 1,
                            len: 0,
                            span: None,
                        });
                    }
                    Ok(Cow::Owned(Value::Seq(s[1..].to_vec().into())))
                } else {
                    Err(EvalError::TypeError {
                        expected: "sequence",
                        got: seq_val.type_name(),
                        span: None,
                    })
                }
            }

            CompiledExpr::SeqAppend { seq, elem } => {
                let seq_val = seq.eval_with_values(ctx, values)?;
                let elem_val = elem.eval_with_values(ctx, values)?.into_owned();
                match &*seq_val {
                    Value::Seq(s) => {
                        let mut new_seq = s.to_vec();
                        new_seq.push(elem_val);
                        Ok(Cow::Owned(Value::Seq(new_seq.into())))
                    }
                    Value::Tuple(s) => {
                        let mut new_seq = s.to_vec();
                        new_seq.push(elem_val);
                        Ok(Cow::Owned(Value::Seq(new_seq.into())))
                    }
                    _ => Err(EvalError::TypeError {
                        expected: "sequence",
                        got: seq_val.type_name(),
                        span: None,
                    }),
                }
            }

            CompiledExpr::SeqLen(seq_expr) => {
                let seq_val = seq_expr.eval_with_values(ctx, values)?;
                // Check for Seq/Tuple first
                if let Some(s) = seq_val.as_seq_or_tuple_elements() {
                    return Ok(Cow::Owned(Value::SmallInt(s.len() as i64)));
                }
                match &*seq_val {
                    Value::String(s) => Ok(Cow::Owned(Value::SmallInt(s.len() as i64))),
                    Value::IntFunc(f) => {
                        if f.min == 1 {
                            Ok(Cow::Owned(Value::SmallInt(f.len() as i64)))
                        } else {
                            Err(EvalError::TypeError {
                                expected: "sequence or string",
                                got: seq_val.type_name(),
                                span: None,
                            })
                        }
                    }
                    Value::Func(f) => {
                        // In TLA+, sequences are functions with domain 1..n.
                        // Only accept functions whose domain is exactly {1, 2, ..., n} (or empty).
                        let mut expected: i64 = 1;
                        for key in f.domain_iter() {
                            let Some(k) = key.as_i64() else {
                                return Err(EvalError::TypeError {
                                    expected: "sequence or string",
                                    got: seq_val.type_name(),
                                    span: None,
                                });
                            };
                            if k != expected {
                                return Err(EvalError::TypeError {
                                    expected: "sequence or string",
                                    got: seq_val.type_name(),
                                    span: None,
                                });
                            }
                            expected += 1;
                        }
                        Ok(Cow::Owned(Value::SmallInt(expected - 1)))
                    }
                    _ => Err(EvalError::TypeError {
                        expected: "sequence or string",
                        got: seq_val.type_name(),
                        span: None,
                    }),
                }
            }

            CompiledExpr::SetCardinality(set_expr) => {
                let set_val = set_expr.eval_with_values(ctx, values)?;
                match set_val.set_len() {
                    Some(c) => Ok(Cow::Owned(Value::big_int(c))),
                    None => Err(EvalError::TypeError {
                        expected: "set",
                        got: set_val.type_name(),
                        span: None,
                    }),
                }
            }

            CompiledExpr::SeqRange(seq_expr) => {
                let seq_val = seq_expr.eval_with_values(ctx, values)?;
                if let Some(s) = seq_val.as_seq_or_tuple_elements() {
                    let set = SortedSet::from_iter(s.iter().cloned());
                    Ok(Cow::Owned(Value::Set(set)))
                } else {
                    Err(EvalError::TypeError {
                        expected: "sequence",
                        got: seq_val.type_name(),
                        span: None,
                    })
                }
            }

            CompiledExpr::RecordAccess { record, field } => {
                let record_val = record.eval_with_values(ctx, values)?;
                match &*record_val {
                    Value::Record(fields) => fields
                        .get(field.as_ref())
                        .cloned()
                        .map(Cow::Owned)
                        .ok_or_else(|| EvalError::NoSuchField {
                            field: field.to_string(),
                            span: None,
                        }),
                    _ => Err(EvalError::TypeError {
                        expected: "record",
                        got: record_val.type_name(),
                        span: None,
                    }),
                }
            }

            CompiledExpr::Fallback { expr, .. } => eval(ctx, expr).map(Cow::Owned),
        }
    }
}

impl CompiledGuard {
    /// Evaluate a compiled guard against an ArrayState.
    ///
    /// Returns true if the guard is satisfied, false otherwise.
    /// This is the fast path for guard evaluation in the BFS loop.
    /// Hot path: called millions of times during model checking.
    #[inline(always)]
    pub fn eval_with_array(&self, ctx: &mut EvalCtx, array_state: &ArrayState) -> EvalResult<bool> {
        self.eval_with_arrays(ctx, array_state, array_state)
    }

    /// Evaluate a compiled guard with both current and next ArrayStates available.
    ///
    /// Used for validating prime guards that reference next-state values.
    /// Hot path: called millions of times during model checking.
    #[inline(always)]
    pub fn eval_with_arrays(
        &self,
        ctx: &mut EvalCtx,
        current_array: &ArrayState,
        next_array: &ArrayState,
    ) -> EvalResult<bool> {
        match self {
            CompiledGuard::True => Ok(true),
            CompiledGuard::False => Ok(false),

            CompiledGuard::EqConst { expr, expected } => {
                // Use eval_cow to avoid cloning for simple lookups
                let val = expr.eval_cow(ctx, current_array, next_array)?;
                Ok(*val == *expected)
            }

            CompiledGuard::NeqConst { expr, expected } => {
                // Use eval_cow to avoid cloning for simple lookups
                let val = expr.eval_cow(ctx, current_array, next_array)?;
                Ok(*val != *expected)
            }

            // Specialized fused guard: state_var[local_var] = const
            // Inlines LocalVar lookup + IntFunc apply + comparison in one path
            CompiledGuard::IntFuncLocalVarEqConst {
                state,
                func_var,
                key_depth,
                expected,
            } => {
                // Get local binding value (the key for function application)
                let key_val =
                    ctx.get_local_by_depth(*key_depth)
                        .ok_or_else(|| EvalError::Internal {
                            message: "IntFuncLocalVarEqConst: local var not found".into(),
                            span: None,
                        })?;

                // Get state array and function value
                let values = match state {
                    StateRef::Current => current_array.values(),
                    StateRef::Next => next_array.values(),
                };
                let func_val = &values[func_var.0 as usize];

                // Try integer key fast path for IntFunc
                let key_i64_opt = match key_val {
                    Value::SmallInt(n) => Some(*n),
                    Value::Int(n) => n.to_i64(),
                    _ => None,
                };

                if let Some(key_i64) = key_i64_opt {
                    // Fast path: IntFunc with direct array index
                    if let Value::IntFunc(f) = func_val {
                        if key_i64 >= f.min && key_i64 <= f.max {
                            let result = &f.values[(key_i64 - f.min) as usize];
                            return Ok(result == expected);
                        }
                        return Ok(false); // Key out of bounds
                    }
                }

                // Fallback for other function types (including non-integer keys)
                let result = match func_val {
                    Value::Func(f) => f.apply(key_val),
                    _ => {
                        if debug_guard() {
                            eprintln!(
                                "DEBUG IntFuncLocalVarEqConst: func_val is not Func: {:?}",
                                func_val.type_name()
                            );
                        }
                        return Ok(false); // Not a function
                    }
                };
                if debug_guard() {
                    eprintln!(
                        "DEBUG IntFuncLocalVarEqConst: key={:?}, result={:?}, expected={:?}",
                        key_val, result, expected
                    );
                }
                Ok(result.map(|v| v == expected).unwrap_or(false))
            }

            // Specialized fused guard: state_var[local_var] # const
            CompiledGuard::IntFuncLocalVarNeqConst {
                state,
                func_var,
                key_depth,
                expected,
            } => {
                let key_val =
                    ctx.get_local_by_depth(*key_depth)
                        .ok_or_else(|| EvalError::Internal {
                            message: "IntFuncLocalVarNeqConst: local var not found".into(),
                            span: None,
                        })?;

                let values = match state {
                    StateRef::Current => current_array.values(),
                    StateRef::Next => next_array.values(),
                };
                let func_val = &values[func_var.0 as usize];

                // Try integer key fast path for IntFunc
                let key_i64_opt = match key_val {
                    Value::SmallInt(n) => Some(*n),
                    Value::Int(n) => n.to_i64(),
                    _ => None,
                };

                if let Some(key_i64) = key_i64_opt {
                    if let Value::IntFunc(f) = func_val {
                        if key_i64 >= f.min && key_i64 <= f.max {
                            let result = &f.values[(key_i64 - f.min) as usize];
                            return Ok(result != expected);
                        }
                        return Ok(true); // Key out of bounds, values differ
                    }
                }

                // Fallback for other function types (including non-integer keys)
                let result = match func_val {
                    Value::Func(f) => f.apply(key_val),
                    _ => return Ok(true),
                };
                Ok(result.map(|v| v != expected).unwrap_or(true))
            }

            CompiledGuard::IntCmp { left, op, right } => {
                // Use eval_cow to avoid cloning for simple lookups
                let lv = left.eval_cow(ctx, current_array, next_array)?;
                let rv = right.eval_cow(ctx, current_array, next_array)?;

                // SmallInt fast path
                if let (Value::SmallInt(l), Value::SmallInt(r)) = (&*lv, &*rv) {
                    return Ok(match op {
                        CmpOp::Lt => l < r,
                        CmpOp::Le => l <= r,
                        CmpOp::Gt => l > r,
                        CmpOp::Ge => l >= r,
                        CmpOp::Eq => l == r,
                        CmpOp::Neq => l != r,
                    });
                }

                // Try BigInt conversion for all values
                let l_int_opt = lv.to_bigint();
                let r_int_opt = rv.to_bigint();

                // If both are integers, use integer comparison
                if let (Some(l_int), Some(r_int)) = (&l_int_opt, &r_int_opt) {
                    return Ok(match op {
                        CmpOp::Lt => l_int < r_int,
                        CmpOp::Le => l_int <= r_int,
                        CmpOp::Gt => l_int > r_int,
                        CmpOp::Ge => l_int >= r_int,
                        CmpOp::Eq => l_int == r_int,
                        CmpOp::Neq => l_int != r_int,
                    });
                }

                // For Eq/Neq, handle non-integer types using Value equality
                // This supports sets, strings, records, etc.
                match op {
                    CmpOp::Eq => Ok(*lv == *rv),
                    CmpOp::Neq => Ok(*lv != *rv),
                    // For ordering comparisons (<, <=, >, >=), require integers
                    _ => Err(EvalError::TypeError {
                        expected: "Int",
                        got: if l_int_opt.is_none() {
                            lv.type_name()
                        } else {
                            rv.type_name()
                        },
                        span: None,
                    }),
                }
            }

            CompiledGuard::In { elem, set } => {
                // Use eval_cow to avoid cloning for simple lookups
                let elem_val = elem.eval_cow(ctx, current_array, next_array)?;
                let set_val = set.eval_cow(ctx, current_array, next_array)?;

                // Use set_contains() directly - it's lazy for Subset, FuncSet, etc.
                // Avoids expensive to_ord_set() enumeration for lazy set types
                if let Some(result) = set_val.set_contains(&elem_val) {
                    Ok(result)
                } else {
                    Err(EvalError::TypeError {
                        expected: "set",
                        got: set_val.type_name(),
                        span: None,
                    })
                }
            }

            CompiledGuard::Subseteq { left, right } => {
                // Fast path: check if this subset relation is already satisfied by SUBSET bounds
                if is_subset_satisfied(left, right) {
                    return Ok(true);
                }

                // Use eval_cow to avoid cloning for simple lookups
                let left_val = left.eval_cow(ctx, current_array, next_array)?;
                let right_val = right.eval_cow(ctx, current_array, next_array)?;

                // Left side must be enumerable
                let left_set = left_val
                    .to_sorted_set()
                    .ok_or_else(|| EvalError::TypeError {
                        expected: "set",
                        got: left_val.type_name(),
                        span: None,
                    })?;

                // Handle ModelValue for infinite sets (Nat, Int, Real)
                // These cannot be converted to SortedSet but support set_contains()
                if let Value::ModelValue(_) = right_val.as_ref() {
                    // Check each element of left set for membership in infinite set
                    for elem in left_set.iter() {
                        if !right_val.set_contains(elem).unwrap_or(false) {
                            return Ok(false);
                        }
                    }
                    return Ok(true);
                }

                let right_set = right_val
                    .to_sorted_set()
                    .ok_or_else(|| EvalError::TypeError {
                        expected: "set",
                        got: right_val.type_name(),
                        span: None,
                    })?;

                Ok(fast_is_subset(&left_set, &right_set))
            }

            CompiledGuard::And(guards) => {
                for guard in guards {
                    if !guard.eval_with_arrays(ctx, current_array, next_array)? {
                        return Ok(false); // Short-circuit
                    }
                }
                Ok(true)
            }

            CompiledGuard::Or(guards) => {
                for guard in guards {
                    if guard.eval_with_arrays(ctx, current_array, next_array)? {
                        return Ok(true); // Short-circuit
                    }
                }
                Ok(false)
            }

            CompiledGuard::Not(inner) => {
                Ok(!inner.eval_with_arrays(ctx, current_array, next_array)?)
            }

            CompiledGuard::ForAll {
                var_name,
                domain,
                body,
            } => {
                let domain_val = domain.eval_cow(ctx, current_array, next_array)?;
                let set = domain_val
                    .to_sorted_set()
                    .ok_or_else(|| EvalError::TypeError {
                        expected: "set",
                        got: domain_val.type_name(),
                        span: None,
                    })?;
                // Short-circuit: return false on first false
                let mark = ctx.mark_stack();
                for elem in set.iter() {
                    ctx.push_binding(Arc::clone(var_name), elem.clone());
                    let result = body.eval_with_arrays(ctx, current_array, next_array);
                    ctx.pop_to_mark(mark);
                    if !result? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }

            CompiledGuard::Exists {
                var_name,
                domain,
                body,
            } => {
                let domain_val = domain.eval_cow(ctx, current_array, next_array)?;
                let set = domain_val
                    .to_sorted_set()
                    .ok_or_else(|| EvalError::TypeError {
                        expected: "set",
                        got: domain_val.type_name(),
                        span: None,
                    })?;
                // Short-circuit: return true on first true
                let mark = ctx.mark_stack();
                for elem in set.iter() {
                    ctx.push_binding(Arc::clone(var_name), elem.clone());
                    let result = body.eval_with_arrays(ctx, current_array, next_array);
                    ctx.pop_to_mark(mark);
                    if result? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }

            CompiledGuard::Implies {
                antecedent,
                consequent,
            } => {
                // a => b is equivalent to ~a \/ b
                if !antecedent.eval_with_arrays(ctx, current_array, next_array)? {
                    return Ok(true); // Antecedent false, implication true
                }
                consequent.eval_with_arrays(ctx, current_array, next_array)
            }

            CompiledGuard::Fallback { expr, reads_next } => {
                // Provide next_state context for primed lookups if needed
                // OPTIMIZATION (Re: #56): Skip HashMap rebuild if next_state_env is already set.
                let val = if *reads_next && !std::ptr::eq(current_array, next_array) {
                    // Fast path: use pre-bound next_state_env for O(1) lookups
                    if ctx.has_next_state_env() {
                        eval(ctx, expr)?
                    } else {
                        // Slow path: build HashMap
                        let registry = ctx.var_registry();
                        let mut next_state = Env::new();
                        for (idx, name) in registry.iter() {
                            next_state
                                .insert(Arc::clone(name), next_array.values()[idx.as_usize()].clone());
                        }
                        let eval_ctx = ctx.with_next_state(next_state);
                        eval(&eval_ctx, expr)?
                    }
                } else {
                    eval(ctx, expr)?
                };
                val.as_bool().ok_or_else(|| EvalError::TypeError {
                    expected: "BOOLEAN",
                    got: val.type_name(),
                    span: Some(expr.span),
                })
            }
        }
    }

    /// Evaluate a compiled guard against a raw value slice.
    ///
    /// This is used during initial state enumeration filtering where we have values
    /// as a `&[Value]` slice (from `enumerate_constraints_to_bulk`) rather than an
    /// `ArrayState`. This avoids creating temporary ArrayState wrappers.
    ///
    /// Note: Next-state references are not supported and will return an error.
    /// The context is mutable to allow quantifier variable binding during evaluation.
    #[inline]
    pub fn eval_with_values(&self, ctx: &mut EvalCtx, values: &[Value]) -> EvalResult<bool> {
        match self {
            CompiledGuard::True => Ok(true),
            CompiledGuard::False => Ok(false),

            CompiledGuard::EqConst { expr, expected } => {
                let val = expr.eval_with_values(ctx, values)?;
                Ok(*val == *expected)
            }

            CompiledGuard::NeqConst { expr, expected } => {
                let val = expr.eval_with_values(ctx, values)?;
                Ok(*val != *expected)
            }

            // Specialized fused guard for value slices
            CompiledGuard::IntFuncLocalVarEqConst {
                state,
                func_var,
                key_depth,
                expected,
            } => {
                // Next-state not supported in eval_with_values
                if *state == StateRef::Next {
                    return Err(EvalError::Internal {
                        message: "Next-state reference in eval_with_values".into(),
                        span: None,
                    });
                }

                let key_val =
                    ctx.get_local_by_depth(*key_depth)
                        .ok_or_else(|| EvalError::Internal {
                            message: "IntFuncLocalVarEqConst: local var not found".into(),
                            span: None,
                        })?;

                let func_val = &values[func_var.0 as usize];

                // Try integer key fast path for IntFunc
                let key_i64_opt = match key_val {
                    Value::SmallInt(n) => Some(*n),
                    Value::Int(n) => n.to_i64(),
                    _ => None,
                };

                if let Some(key_i64) = key_i64_opt {
                    if let Value::IntFunc(f) = func_val {
                        if key_i64 >= f.min && key_i64 <= f.max {
                            let result = &f.values[(key_i64 - f.min) as usize];
                            return Ok(result == expected);
                        }
                        return Ok(false);
                    }
                }

                // Fallback for other function types (including non-integer keys)
                let result = match func_val {
                    Value::Func(f) => f.apply(key_val),
                    _ => return Ok(false),
                };
                Ok(result.map(|v| v == expected).unwrap_or(false))
            }

            CompiledGuard::IntFuncLocalVarNeqConst {
                state,
                func_var,
                key_depth,
                expected,
            } => {
                if *state == StateRef::Next {
                    return Err(EvalError::Internal {
                        message: "Next-state reference in eval_with_values".into(),
                        span: None,
                    });
                }

                let key_val =
                    ctx.get_local_by_depth(*key_depth)
                        .ok_or_else(|| EvalError::Internal {
                            message: "IntFuncLocalVarNeqConst: local var not found".into(),
                            span: None,
                        })?;

                let func_val = &values[func_var.0 as usize];

                // Try integer key fast path for IntFunc
                let key_i64_opt = match key_val {
                    Value::SmallInt(n) => Some(*n),
                    Value::Int(n) => n.to_i64(),
                    _ => None,
                };

                if let Some(key_i64) = key_i64_opt {
                    if let Value::IntFunc(f) = func_val {
                        if key_i64 >= f.min && key_i64 <= f.max {
                            let result = &f.values[(key_i64 - f.min) as usize];
                            return Ok(result != expected);
                        }
                        return Ok(true);
                    }
                }

                // Fallback for other function types (including non-integer keys)
                let result = match func_val {
                    Value::Func(f) => f.apply(key_val),
                    _ => return Ok(true),
                };
                Ok(result.map(|v| v != expected).unwrap_or(true))
            }

            CompiledGuard::IntCmp { left, op, right } => {
                let lv = left.eval_with_values(ctx, values)?;
                let rv = right.eval_with_values(ctx, values)?;

                // SmallInt fast path
                if let (Value::SmallInt(l), Value::SmallInt(r)) = (&*lv, &*rv) {
                    return Ok(match op {
                        CmpOp::Lt => l < r,
                        CmpOp::Le => l <= r,
                        CmpOp::Gt => l > r,
                        CmpOp::Ge => l >= r,
                        CmpOp::Eq => l == r,
                        CmpOp::Neq => l != r,
                    });
                }

                // Try BigInt conversion for all values
                let l_int_opt = lv.to_bigint();
                let r_int_opt = rv.to_bigint();

                // If both are integers, use integer comparison
                if let (Some(l_int), Some(r_int)) = (&l_int_opt, &r_int_opt) {
                    return Ok(match op {
                        CmpOp::Lt => l_int < r_int,
                        CmpOp::Le => l_int <= r_int,
                        CmpOp::Gt => l_int > r_int,
                        CmpOp::Ge => l_int >= r_int,
                        CmpOp::Eq => l_int == r_int,
                        CmpOp::Neq => l_int != r_int,
                    });
                }

                // For Eq/Neq, handle non-integer types using Value equality
                // This supports sets, strings, records, etc.
                match op {
                    CmpOp::Eq => Ok(*lv == *rv),
                    CmpOp::Neq => Ok(*lv != *rv),
                    // For ordering comparisons (<, <=, >, >=), require integers
                    _ => Err(EvalError::TypeError {
                        expected: "Int",
                        got: if l_int_opt.is_none() {
                            lv.type_name()
                        } else {
                            rv.type_name()
                        },
                        span: None,
                    }),
                }
            }

            CompiledGuard::In { elem, set } => {
                let elem_val = elem.eval_with_values(ctx, values)?;
                let set_val = set.eval_with_values(ctx, values)?;

                if let Some(result) = set_val.set_contains(&elem_val) {
                    Ok(result)
                } else {
                    Err(EvalError::TypeError {
                        expected: "set",
                        got: set_val.type_name(),
                        span: None,
                    })
                }
            }

            CompiledGuard::Subseteq { left, right } => {
                // Fast path: check if this subset relation is already satisfied by SUBSET bounds
                if is_subset_satisfied(left, right) {
                    return Ok(true);
                }

                let left_val = left.eval_with_values(ctx, values)?;
                let right_val = right.eval_with_values(ctx, values)?;

                // Left side must be enumerable
                let left_set = left_val
                    .to_sorted_set()
                    .ok_or_else(|| EvalError::TypeError {
                        expected: "set",
                        got: left_val.type_name(),
                        span: None,
                    })?;

                // Handle ModelValue for infinite sets (Nat, Int, Real)
                // These cannot be converted to SortedSet but support set_contains()
                if let Value::ModelValue(_) = right_val.as_ref() {
                    // Check each element of left set for membership in infinite set
                    for elem in left_set.iter() {
                        if !right_val.set_contains(elem).unwrap_or(false) {
                            return Ok(false);
                        }
                    }
                    return Ok(true);
                }

                let right_set = right_val
                    .to_sorted_set()
                    .ok_or_else(|| EvalError::TypeError {
                        expected: "set",
                        got: right_val.type_name(),
                        span: None,
                    })?;

                Ok(fast_is_subset(&left_set, &right_set))
            }

            CompiledGuard::And(guards) => {
                for guard in guards {
                    if !guard.eval_with_values(ctx, values)? {
                        return Ok(false); // Short-circuit
                    }
                }
                Ok(true)
            }

            CompiledGuard::Or(guards) => {
                for guard in guards {
                    if guard.eval_with_values(ctx, values)? {
                        return Ok(true); // Short-circuit
                    }
                }
                Ok(false)
            }

            CompiledGuard::Not(inner) => Ok(!inner.eval_with_values(ctx, values)?),

            CompiledGuard::ForAll {
                var_name,
                domain,
                body,
            } => {
                let domain_val = domain.eval_with_values(ctx, values)?;
                let set = domain_val
                    .to_sorted_set()
                    .ok_or_else(|| EvalError::TypeError {
                        expected: "set",
                        got: domain_val.type_name(),
                        span: None,
                    })?;
                // Short-circuit: return false on first false
                let mark = ctx.mark_stack();
                for elem in set.iter() {
                    ctx.push_binding(Arc::clone(var_name), elem.clone());
                    let result = body.eval_with_values(ctx, values);
                    ctx.pop_to_mark(mark);
                    if !result? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }

            CompiledGuard::Exists {
                var_name,
                domain,
                body,
            } => {
                let domain_val = domain.eval_with_values(ctx, values)?;
                let set = domain_val
                    .to_sorted_set()
                    .ok_or_else(|| EvalError::TypeError {
                        expected: "set",
                        got: domain_val.type_name(),
                        span: None,
                    })?;
                // Short-circuit: return true on first true
                let mark = ctx.mark_stack();
                for elem in set.iter() {
                    ctx.push_binding(Arc::clone(var_name), elem.clone());
                    let result = body.eval_with_values(ctx, values);
                    ctx.pop_to_mark(mark);
                    if result? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }

            CompiledGuard::Implies {
                antecedent,
                consequent,
            } => {
                // a => b is equivalent to ~a \/ b
                if !antecedent.eval_with_values(ctx, values)? {
                    return Ok(true); // Antecedent false, implication true
                }
                consequent.eval_with_values(ctx, values)
            }

            CompiledGuard::Fallback { expr, .. } => {
                let val = eval(ctx, expr)?;
                val.as_bool().ok_or_else(|| EvalError::TypeError {
                    expected: "BOOLEAN",
                    got: val.type_name(),
                    span: Some(expr.span),
                })
            }
        }
    }
}

// ============================================================================
// GUARD COMPILER - Compile AST guards to CompiledGuard
// ============================================================================

/// Check if a set expression requires lazy membership checking.
/// This includes Powerset, FuncSet, RecordSet, and Seq - sets that would be
/// expensive or infinite to enumerate eagerly.
fn needs_lazy_membership_set(ctx: &EvalCtx, expr: &Spanned<Expr>) -> bool {
    // Check the expression directly
    if is_lazy_membership_expr(&expr.node) {
        return true;
    }
    // Also check if expr is an Ident that resolves to a lazy membership expression
    if let Expr::Ident(name) = &expr.node {
        // Resolve operator name through replacements (e.g., Op <- Replacement)
        let resolved_name = ctx.resolve_op_name(name);
        if let Some(def) = ctx.get_op(resolved_name) {
            if def.params.is_empty() && is_lazy_membership_expr(&def.body.node) {
                return true;
            }
        }
    }
    false
}

/// Check if an expression type requires lazy membership (without resolution)
fn is_lazy_membership_expr(expr: &Expr) -> bool {
    match expr {
        Expr::FuncSet(_, _) | Expr::Powerset(_) | Expr::RecordSet(_) => true,
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

/// Compile a guard expression to a CompiledGuard.
///
/// This analyzes the expression structure and produces an optimized
/// representation for fast evaluation. Complex expressions fall back
/// to the standard eval() function.
///
/// The `local_scope` parameter tracks EXISTS-bound variables for O(1) lookup.
///
/// Note: Quantifiers (Forall, Exists) and Implies compile to optimized variants when possible.
/// Multi-variable quantifiers and pattern destructuring still fall back to full `eval()`.
fn debug_guard() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("TLA2_DEBUG_GUARD").is_ok())
}

pub fn compile_guard(
    ctx: &EvalCtx,
    expr: &Spanned<Expr>,
    registry: &VarRegistry,
    local_scope: &LocalScope,
) -> CompiledGuard {
    if debug_guard() {
        eprintln!("DEBUG compile_guard: expr={:?}", expr.node);
    }
    let result = compile_guard_expr(ctx, &expr.node, expr, registry, local_scope);
    if debug_guard() {
        eprintln!("DEBUG compile_guard: result={:?}", result);
    }
    result
}

/// Compile a guard expression for use with eval_with_values (mutable context).
///
/// This uses the filter-focused compiler which targets `eval_with_values` (raw `&[Value]` slices)
/// rather than ArrayState-backed evaluation.
///
/// Use this for enumeration filter predicates where the guard will be evaluated
/// with eval_with_values against raw value slices.
pub fn compile_guard_for_filter(
    ctx: &EvalCtx,
    expr: &Spanned<Expr>,
    registry: &VarRegistry,
    local_scope: &LocalScope,
) -> CompiledGuard {
    compile_guard_expr_for_filter(ctx, &expr.node, expr, registry, local_scope)
}

/// Compile a value expression for efficient evaluation against an ArrayState.
///
/// This is used for domain bound expressions (like `sent`, `rcvd[self]`) in EXISTS
/// quantifiers where the domain is SUBSET(S) with state-dependent bounds.
///
/// The compiled expression can be evaluated using `CompiledExpr::eval_with_array()`.
pub fn compile_domain_expr(
    ctx: &EvalCtx,
    expr: &Spanned<Expr>,
    registry: &VarRegistry,
    local_scope: &LocalScope,
) -> CompiledExpr {
    compile_value_expr(ctx, expr, registry, local_scope)
}

/// Compile a prime guard (an expression that references next-state values) to a CompiledGuard.
///
/// Returns `None` if compilation would require falling back to full AST evaluation.
#[cfg(test)]
pub(crate) fn compile_prime_guard(
    ctx: &EvalCtx,
    expr: &Spanned<Expr>,
    registry: &VarRegistry,
    local_scope: &LocalScope,
) -> Option<CompiledGuard> {
    let compiled = compile_guard_expr_action(ctx, &expr.node, expr, registry, local_scope);
    if guard_has_fallback(&compiled) {
        None
    } else {
        Some(compiled)
    }
}

fn compile_guard_expr(
    ctx: &EvalCtx,
    expr: &Expr,
    full_expr: &Spanned<Expr>,
    registry: &VarRegistry,
    local_scope: &LocalScope,
) -> CompiledGuard {
    match expr {
        // Boolean literals
        Expr::Bool(true) => CompiledGuard::True,
        Expr::Bool(false) => CompiledGuard::False,

        // Conjunction: a /\ b
        Expr::And(a, b) => {
            let mut guards = Vec::new();
            collect_and_guards(ctx, a, registry, local_scope, &mut guards);
            collect_and_guards(ctx, b, registry, local_scope, &mut guards);
            if guards.len() == 1 {
                guards.pop().unwrap()
            } else {
                CompiledGuard::And(guards)
            }
        }

        // Disjunction: a \/ b
        Expr::Or(a, b) => {
            let mut guards = Vec::new();
            collect_or_guards(ctx, a, registry, local_scope, &mut guards);
            collect_or_guards(ctx, b, registry, local_scope, &mut guards);
            if guards.len() == 1 {
                guards.pop().unwrap()
            } else {
                CompiledGuard::Or(guards)
            }
        }

        // Negation: ~a or \lnot a
        Expr::Not(inner) => {
            let compiled = compile_guard(ctx, inner, registry, local_scope);
            match compiled {
                CompiledGuard::True => CompiledGuard::False,
                CompiledGuard::False => CompiledGuard::True,
                CompiledGuard::Not(inner) => *inner,
                other => CompiledGuard::Not(Box::new(other)),
            }
        }

        // Equality: a = b
        Expr::Eq(a, b) => compile_eq_guard(ctx, a, b, registry, local_scope, false),

        // Inequality: a # b or a /= b
        Expr::Neq(a, b) => compile_eq_guard(ctx, a, b, registry, local_scope, true),

        // Less than: a < b
        Expr::Lt(a, b) => compile_cmp_guard(ctx, a, b, CmpOp::Lt, registry, local_scope),

        // Less than or equal: a <= b or a =< b
        Expr::Leq(a, b) => compile_cmp_guard(ctx, a, b, CmpOp::Le, registry, local_scope),

        // Greater than: a > b
        Expr::Gt(a, b) => compile_cmp_guard(ctx, a, b, CmpOp::Gt, registry, local_scope),

        // Greater than or equal: a >= b
        Expr::Geq(a, b) => compile_cmp_guard(ctx, a, b, CmpOp::Ge, registry, local_scope),

        // Set membership: a \in b
        // For sets that need lazy membership (Powerset, FuncSet, RecordSet, Seq),
        // we try to pre-evaluate them if they're constant expressions.
        // This is a major optimization for invariants like `sent \in SUBSET(Proc × M)`
        // where the SUBSET is constant and can be computed once at startup.
        Expr::In(elem, set) => {
            let elem_expr = compile_value_expr(ctx, elem, registry, local_scope);
            if needs_lazy_membership_set(ctx, set) {
                // Try to pre-evaluate the set as a constant
                // This handles cases like SUBSET(Proc × M) where the set is constant
                if let Ok(set_val) = eval(ctx, set) {
                    CompiledGuard::In {
                        elem: elem_expr,
                        set: CompiledExpr::Const(set_val),
                    }
                } else {
                    CompiledGuard::Fallback {
                expr: full_expr.clone(),
                reads_next: expr_contains_any_prime(&full_expr.node),
            }
                }
            } else {
                let set_expr = compile_value_expr(ctx, set, registry, local_scope);
                CompiledGuard::In {
                    elem: elem_expr,
                    set: set_expr,
                }
            }
        }

        // Non-membership: a \notin b -> ~(a \in b)
        // Same pre-evaluation logic as membership
        Expr::NotIn(elem, set) => {
            let elem_expr = compile_value_expr(ctx, elem, registry, local_scope);
            if needs_lazy_membership_set(ctx, set) {
                // Try to pre-evaluate the set as a constant
                if let Ok(set_val) = eval(ctx, set) {
                    CompiledGuard::Not(Box::new(CompiledGuard::In {
                        elem: elem_expr,
                        set: CompiledExpr::Const(set_val),
                    }))
                } else {
                    CompiledGuard::Fallback {
                expr: full_expr.clone(),
                reads_next: expr_contains_any_prime(&full_expr.node),
            }
                }
            } else {
                let set_expr = compile_value_expr(ctx, set, registry, local_scope);
                CompiledGuard::Not(Box::new(CompiledGuard::In {
                    elem: elem_expr,
                    set: set_expr,
                }))
            }
        }

        // Implication: a => b
        Expr::Implies(a, b) => {
            let antecedent = compile_guard(ctx, a, registry, local_scope);
            let consequent = compile_guard(ctx, b, registry, local_scope);
            CompiledGuard::Implies {
                antecedent: Box::new(antecedent),
                consequent: Box::new(consequent),
            }
        }

        // Universal quantifier: \A x \in S : P(x)
        Expr::Forall(bounds, body) => {
            // Currently only support single-variable quantifiers with domain
            if bounds.len() != 1 {
                return CompiledGuard::Fallback {
                expr: full_expr.clone(),
                reads_next: expr_contains_any_prime(&full_expr.node),
            };
            }
            let bound_var = &bounds[0];
            // Must have a domain (x \in S)
            let domain = match &bound_var.domain {
                Some(d) => d,
                None => return CompiledGuard::Fallback {
                expr: full_expr.clone(),
                reads_next: expr_contains_any_prime(&full_expr.node),
            },
            };
            // Must not have pattern destructuring
            if bound_var.pattern.is_some() {
                return CompiledGuard::Fallback {
                expr: full_expr.clone(),
                reads_next: expr_contains_any_prime(&full_expr.node),
            };
            }
            let var_name: Arc<str> = Arc::from(bound_var.name.node.as_str());

            // Add quantifier variable to local scope for body compilation
            let inner_scope = local_scope.with_var(&bound_var.name.node);

            let domain_expr = compile_value_expr(ctx, domain, registry, local_scope);
            let body_guard = compile_guard(ctx, body, registry, &inner_scope);

            CompiledGuard::ForAll {
                var_name,
                domain: domain_expr,
                body: Box::new(body_guard),
            }
        }

        // Existential quantifier: \E x \in S : P(x)
        Expr::Exists(bounds, body) => {
            // Currently only support single-variable quantifiers with domain
            if bounds.len() != 1 {
                return CompiledGuard::Fallback {
                expr: full_expr.clone(),
                reads_next: expr_contains_any_prime(&full_expr.node),
            };
            }
            let bound_var = &bounds[0];
            // Must have a domain (x \in S)
            let domain = match &bound_var.domain {
                Some(d) => d,
                None => return CompiledGuard::Fallback {
                expr: full_expr.clone(),
                reads_next: expr_contains_any_prime(&full_expr.node),
            },
            };
            // Must not have pattern destructuring
            if bound_var.pattern.is_some() {
                return CompiledGuard::Fallback {
                expr: full_expr.clone(),
                reads_next: expr_contains_any_prime(&full_expr.node),
            };
            }
            let var_name: Arc<str> = Arc::from(bound_var.name.node.as_str());

            // Add quantifier variable to local scope for body compilation
            let inner_scope = local_scope.with_var(&bound_var.name.node);

            let domain_expr = compile_value_expr(ctx, domain, registry, local_scope);
            let body_guard = compile_guard(ctx, body, registry, &inner_scope);

            CompiledGuard::Exists {
                var_name,
                domain: domain_expr,
                body: Box::new(body_guard),
            }
        }

        // Equivalence: a <=> b -> (a /\ b) \/ (~a /\ ~b)
        Expr::Equiv(a, b) => {
            let guard_a = compile_guard(ctx, a, registry, local_scope);
            let guard_b = compile_guard(ctx, b, registry, local_scope);
            // a <=> b is equivalent to (a /\ b) \/ (~a /\ ~b)
            CompiledGuard::Or(vec![
                CompiledGuard::And(vec![guard_a.clone(), guard_b.clone()]),
                CompiledGuard::And(vec![
                    CompiledGuard::Not(Box::new(guard_a)),
                    CompiledGuard::Not(Box::new(guard_b)),
                ]),
            ])
        }

        // Subset: a \subseteq b
        Expr::Subseteq(a, b) => {
            let left = compile_value_expr(ctx, a, registry, local_scope);
            let right = compile_value_expr(ctx, b, registry, local_scope);
            CompiledGuard::Subseteq { left, right }
        }

        // Try to inline zero-arg operators
        Expr::Ident(name) => {
            // Resolve operator name through replacements (e.g., Op <- Replacement)
            let resolved_name = ctx.resolve_op_name(name);
            if let Some(def) = ctx.get_op(resolved_name) {
                if def.params.is_empty() {
                    // Inline the operator body
                    return compile_guard_expr(
                        ctx,
                        &def.body.node,
                        &def.body,
                        registry,
                        local_scope,
                    );
                }
            }
            // Fall back for variables used as boolean values
            CompiledGuard::Fallback {
                expr: full_expr.clone(),
                reads_next: expr_contains_any_prime(&full_expr.node),
            }
        }

        // User-defined operator application: Before(i, j), etc.
        // Inline the operator body with parameter substitutions.
        // Note: Uses eval_with_arrays which requires immutable context, so quantifiers
        // in the inlined body will still fall back. But non-quantifier expressions
        // (comparisons, set membership, etc.) will be compiled efficiently.
        Expr::Apply(op_expr, args) => {
            if let Expr::Ident(op_name) = &op_expr.node {
                // Resolve operator name through replacements (e.g., Op <- Replacement)
                let resolved_name = ctx.resolve_op_name(op_name);
                if let Some(def) = ctx.get_op(resolved_name) {
                    // Check that we have the right number of arguments
                    if def.params.len() == args.len()
                        && def.params.iter().all(|p| p.arity == 0)
                        && inlining_substitution_is_capture_safe(def, args)
                    {
                        // Create substitutions for parameters -> arguments
                        let subs: Vec<Substitution> = def
                            .params
                            .iter()
                            .zip(args.iter())
                            .map(|(param, arg)| Substitution {
                                from: param.name.clone(),
                                to: arg.clone(),
                            })
                            .collect();

                        // Apply substitutions to get the expanded body
                        let substituted_body = apply_substitutions(&def.body, &subs);

                        // Recursively compile the substituted body
                        return compile_guard_expr(
                            ctx,
                            &substituted_body.node,
                            &substituted_body,
                            registry,
                            local_scope,
                        );
                    }
                }
            }
            // Fall back if not a simple operator application
            CompiledGuard::Fallback {
                expr: full_expr.clone(),
                reads_next: expr_contains_any_prime(&full_expr.node),
            }
        }

        // LET expression: compile with inlined definitions
        Expr::Let(defs, body) => {
            // Compile each definition and add to local scope for inlining
            let mut extended_scope = local_scope.clone();
            let mut all_compiled = true;

            for def in defs {
                // Only support zero-argument definitions
                if !def.params.is_empty() {
                    all_compiled = false;
                    break;
                }

                // Compile the definition body
                let def_compiled = compile_value_expr_inner(
                    ctx,
                    &def.body.node,
                    &def.body,
                    registry,
                    StateRef::Current,
                    false,
                    &extended_scope,
                );

                if expr_has_fallback(&def_compiled) {
                    all_compiled = false;
                    break;
                }

                // Add to scope for subsequent definitions and body
                extended_scope = extended_scope.with_let_binding(&def.name.node, def_compiled);
            }

            if all_compiled {
                // Compile the body with inlined definitions
                let body_guard =
                    compile_guard_expr(ctx, &body.node, body, registry, &extended_scope);
                // IMPORTANT: If the body has any fallbacks, fall back the entire LET.
                // Otherwise, the fallback sub-expressions won't have access to the
                // LET-defined names (which are only in compile-time let_bindings).
                if !guard_has_fallback(&body_guard) {
                    return body_guard;
                }
            }

            CompiledGuard::Fallback {
                expr: full_expr.clone(),
                reads_next: expr_contains_any_prime(&full_expr.node),
            }
        }

        // Everything else falls back to full eval
        _ => CompiledGuard::Fallback {
                expr: full_expr.clone(),
                reads_next: expr_contains_any_prime(&full_expr.node),
            },
    }
}

/// Compile a guard expression for use with eval_with_values (mutable context).
///
/// This variant compiles quantifiers (Forall, Exists) and Implies to their
/// optimized variants instead of Fallback, since eval_with_values can bind
/// quantifier variables in a mutable context.
fn compile_guard_expr_for_filter(
    ctx: &EvalCtx,
    expr: &Expr,
    full_expr: &Spanned<Expr>,
    registry: &VarRegistry,
    local_scope: &LocalScope,
) -> CompiledGuard {
    match expr {
        Expr::Bool(true) => CompiledGuard::True,
        Expr::Bool(false) => CompiledGuard::False,

        // Conjunction: a /\ b (short-circuit evaluation)
        Expr::And(a, b) => {
            let mut guards = Vec::new();
            collect_and_guards_for_filter(ctx, a, registry, local_scope, &mut guards);
            collect_and_guards_for_filter(ctx, b, registry, local_scope, &mut guards);
            if guards.len() == 1 {
                guards.pop().unwrap()
            } else {
                CompiledGuard::And(guards)
            }
        }

        // Disjunction: a \/ b (short-circuit evaluation)
        Expr::Or(a, b) => {
            let mut guards = Vec::new();
            collect_or_guards_for_filter(ctx, a, registry, local_scope, &mut guards);
            collect_or_guards_for_filter(ctx, b, registry, local_scope, &mut guards);
            if guards.len() == 1 {
                guards.pop().unwrap()
            } else {
                CompiledGuard::Or(guards)
            }
        }

        // Negation: ~a
        Expr::Not(a) => {
            let inner = compile_guard_for_filter(ctx, a, registry, local_scope);
            CompiledGuard::Not(Box::new(inner))
        }

        // Equality comparison
        Expr::Eq(a, b) => {
            let left = compile_value_expr(ctx, a, registry, local_scope);
            let right = compile_value_expr(ctx, b, registry, local_scope);

            // Optimize constant comparisons
            match (&left, &right) {
                (CompiledExpr::Const(l), CompiledExpr::Const(r)) => {
                    if l == r {
                        CompiledGuard::True
                    } else {
                        CompiledGuard::False
                    }
                }
                (_, CompiledExpr::Const(c)) => CompiledGuard::EqConst {
                    expr: left,
                    expected: c.clone(),
                },
                (CompiledExpr::Const(c), _) => CompiledGuard::EqConst {
                    expr: right,
                    expected: c.clone(),
                },
                _ => CompiledGuard::IntCmp {
                    left,
                    op: CmpOp::Eq,
                    right,
                },
            }
        }

        // Inequality comparison
        Expr::Neq(a, b) => {
            let left = compile_value_expr(ctx, a, registry, local_scope);
            let right = compile_value_expr(ctx, b, registry, local_scope);

            match (&left, &right) {
                (CompiledExpr::Const(l), CompiledExpr::Const(r)) => {
                    if l != r {
                        CompiledGuard::True
                    } else {
                        CompiledGuard::False
                    }
                }
                (_, CompiledExpr::Const(c)) => CompiledGuard::NeqConst {
                    expr: left,
                    expected: c.clone(),
                },
                (CompiledExpr::Const(c), _) => CompiledGuard::NeqConst {
                    expr: right,
                    expected: c.clone(),
                },
                _ => CompiledGuard::IntCmp {
                    left,
                    op: CmpOp::Neq,
                    right,
                },
            }
        }

        // Integer comparisons
        Expr::Lt(a, b) => {
            let left = compile_value_expr(ctx, a, registry, local_scope);
            let right = compile_value_expr(ctx, b, registry, local_scope);
            CompiledGuard::IntCmp {
                left,
                op: CmpOp::Lt,
                right,
            }
        }
        Expr::Leq(a, b) => {
            let left = compile_value_expr(ctx, a, registry, local_scope);
            let right = compile_value_expr(ctx, b, registry, local_scope);
            CompiledGuard::IntCmp {
                left,
                op: CmpOp::Le,
                right,
            }
        }
        Expr::Gt(a, b) => {
            let left = compile_value_expr(ctx, a, registry, local_scope);
            let right = compile_value_expr(ctx, b, registry, local_scope);
            CompiledGuard::IntCmp {
                left,
                op: CmpOp::Gt,
                right,
            }
        }
        Expr::Geq(a, b) => {
            let left = compile_value_expr(ctx, a, registry, local_scope);
            let right = compile_value_expr(ctx, b, registry, local_scope);
            CompiledGuard::IntCmp {
                left,
                op: CmpOp::Ge,
                right,
            }
        }

        // Set membership
        Expr::In(elem, set) => {
            // For lazy membership sets (Powerset, FuncSet, RecordSet, Seq), use Fallback
            if needs_lazy_membership_set(ctx, set) {
                return CompiledGuard::Fallback {
                expr: full_expr.clone(),
                reads_next: expr_contains_any_prime(&full_expr.node),
            };
            }

            let elem_expr = compile_value_expr(ctx, elem, registry, local_scope);
            let set_expr = compile_value_expr(ctx, set, registry, local_scope);
            CompiledGuard::In {
                elem: elem_expr,
                set: set_expr,
            }
        }

        Expr::NotIn(elem, set) => {
            if needs_lazy_membership_set(ctx, set) {
                return CompiledGuard::Fallback {
                expr: full_expr.clone(),
                reads_next: expr_contains_any_prime(&full_expr.node),
            };
            }

            let elem_expr = compile_value_expr(ctx, elem, registry, local_scope);
            let set_expr = compile_value_expr(ctx, set, registry, local_scope);
            CompiledGuard::Not(Box::new(CompiledGuard::In {
                elem: elem_expr,
                set: set_expr,
            }))
        }

        // Implication: a => b
        // For filter evaluation (mutable context), compile to optimized Implies variant
        Expr::Implies(a, b) => {
            let antecedent = compile_guard_for_filter(ctx, a, registry, local_scope);
            let consequent = compile_guard_for_filter(ctx, b, registry, local_scope);
            CompiledGuard::Implies {
                antecedent: Box::new(antecedent),
                consequent: Box::new(consequent),
            }
        }

        // Universal quantifier: \A x \in S : P(x)
        // For filter evaluation (mutable context), compile to optimized ForAll variant
        Expr::Forall(bounds, body) => {
            // Currently only support single-variable quantifiers with domain
            if bounds.len() != 1 {
                return CompiledGuard::Fallback {
                expr: full_expr.clone(),
                reads_next: expr_contains_any_prime(&full_expr.node),
            };
            }
            let bound_var = &bounds[0];
            // Must have a domain (x \in S)
            let domain = match &bound_var.domain {
                Some(d) => d,
                None => return CompiledGuard::Fallback {
                expr: full_expr.clone(),
                reads_next: expr_contains_any_prime(&full_expr.node),
            },
            };
            // Must not have pattern destructuring
            if bound_var.pattern.is_some() {
                return CompiledGuard::Fallback {
                expr: full_expr.clone(),
                reads_next: expr_contains_any_prime(&full_expr.node),
            };
            }
            let var_name: Arc<str> = Arc::from(bound_var.name.node.as_str());

            // Add quantifier variable to local scope for body compilation
            let inner_scope = local_scope.with_var(&bound_var.name.node);

            let domain_expr = compile_value_expr(ctx, domain, registry, local_scope);
            let body_guard =
                compile_guard_expr_for_filter(ctx, &body.node, body, registry, &inner_scope);

            CompiledGuard::ForAll {
                var_name,
                domain: domain_expr,
                body: Box::new(body_guard),
            }
        }

        // Existential quantifier: \E x \in S : P(x)
        // For filter evaluation (mutable context), compile to optimized Exists variant
        Expr::Exists(bounds, body) => {
            // Currently only support single-variable quantifiers with domain
            if bounds.len() != 1 {
                return CompiledGuard::Fallback {
                expr: full_expr.clone(),
                reads_next: expr_contains_any_prime(&full_expr.node),
            };
            }
            let bound_var = &bounds[0];
            // Must have a domain (x \in S)
            let domain = match &bound_var.domain {
                Some(d) => d,
                None => return CompiledGuard::Fallback {
                expr: full_expr.clone(),
                reads_next: expr_contains_any_prime(&full_expr.node),
            },
            };
            // Must not have pattern destructuring
            if bound_var.pattern.is_some() {
                return CompiledGuard::Fallback {
                expr: full_expr.clone(),
                reads_next: expr_contains_any_prime(&full_expr.node),
            };
            }
            let var_name: Arc<str> = Arc::from(bound_var.name.node.as_str());

            // Add quantifier variable to local scope for body compilation
            let inner_scope = local_scope.with_var(&bound_var.name.node);

            let domain_expr = compile_value_expr(ctx, domain, registry, local_scope);
            let body_guard =
                compile_guard_expr_for_filter(ctx, &body.node, body, registry, &inner_scope);

            CompiledGuard::Exists {
                var_name,
                domain: domain_expr,
                body: Box::new(body_guard),
            }
        }

        // Equivalence: a <=> b -> (a /\ b) \/ (~a /\ ~b)
        Expr::Equiv(a, b) => {
            let guard_a = compile_guard_for_filter(ctx, a, registry, local_scope);
            let guard_b = compile_guard_for_filter(ctx, b, registry, local_scope);
            CompiledGuard::Or(vec![
                CompiledGuard::And(vec![guard_a.clone(), guard_b.clone()]),
                CompiledGuard::And(vec![
                    CompiledGuard::Not(Box::new(guard_a)),
                    CompiledGuard::Not(Box::new(guard_b)),
                ]),
            ])
        }

        // Subset: a \subseteq b
        Expr::Subseteq(a, b) => {
            let left = compile_value_expr(ctx, a, registry, local_scope);
            let right = compile_value_expr(ctx, b, registry, local_scope);
            CompiledGuard::Subseteq { left, right }
        }

        // Try to inline zero-arg operators
        Expr::Ident(name) => {
            // Resolve operator name through replacements (e.g., Op <- Replacement)
            let resolved_name = ctx.resolve_op_name(name);
            if let Some(def) = ctx.get_op(resolved_name) {
                if def.params.is_empty() {
                    // Inline the operator body
                    return compile_guard_expr_for_filter(
                        ctx,
                        &def.body.node,
                        &def.body,
                        registry,
                        local_scope,
                    );
                }
            }
            // Fall back for variables used as boolean values
            CompiledGuard::Fallback {
                expr: full_expr.clone(),
                reads_next: expr_contains_any_prime(&full_expr.node),
            }
        }

        // User-defined operator application: Before(i, j), etc.
        // Inline the operator body with parameter substitutions.
        Expr::Apply(op_expr, args) => {
            if let Expr::Ident(op_name) = &op_expr.node {
                // Resolve operator name through replacements (e.g., Op <- Replacement)
                let resolved_name = ctx.resolve_op_name(op_name);
                if let Some(def) = ctx.get_op(resolved_name) {
                    // Check that we have the right number of arguments
                    if def.params.len() == args.len()
                        && def.params.iter().all(|p| p.arity == 0)
                        && inlining_substitution_is_capture_safe(def, args)
                    {
                        // Create substitutions for parameters -> arguments
                        let subs: Vec<Substitution> = def
                            .params
                            .iter()
                            .zip(args.iter())
                            .map(|(param, arg)| Substitution {
                                from: param.name.clone(),
                                to: arg.clone(),
                            })
                            .collect();

                        // Apply substitutions to get the expanded body
                        let substituted_body = apply_substitutions(&def.body, &subs);

                        // Recursively compile the substituted body
                        return compile_guard_expr_for_filter(
                            ctx,
                            &substituted_body.node,
                            &substituted_body,
                            registry,
                            local_scope,
                        );
                    }
                }
            }
            // Fall back if not a simple operator application
            CompiledGuard::Fallback {
                expr: full_expr.clone(),
                reads_next: expr_contains_any_prime(&full_expr.node),
            }
        }

        // LET expression: compile with inlined definitions
        Expr::Let(defs, body) => {
            // Compile each definition and add to local scope for inlining
            let mut extended_scope = local_scope.clone();
            let mut all_compiled = true;

            for def in defs {
                // Only support zero-argument definitions
                if !def.params.is_empty() {
                    all_compiled = false;
                    break;
                }

                // Compile the definition body
                let def_compiled = compile_value_expr_inner(
                    ctx,
                    &def.body.node,
                    &def.body,
                    registry,
                    StateRef::Current,
                    false,
                    &extended_scope,
                );

                if expr_has_fallback(&def_compiled) {
                    all_compiled = false;
                    break;
                }

                // Add to scope for subsequent definitions and body
                extended_scope = extended_scope.with_let_binding(&def.name.node, def_compiled);
            }

            if all_compiled {
                // Compile the body with inlined definitions
                let body_guard =
                    compile_guard_expr_for_filter(ctx, &body.node, body, registry, &extended_scope);
                // IMPORTANT: If the body has any fallbacks, fall back the entire LET.
                if !guard_has_fallback(&body_guard) {
                    return body_guard;
                }
            }

            CompiledGuard::Fallback {
                expr: full_expr.clone(),
                reads_next: expr_contains_any_prime(&full_expr.node),
            }
        }

        // Everything else falls back to full eval
        _ => CompiledGuard::Fallback {
                expr: full_expr.clone(),
                reads_next: expr_contains_any_prime(&full_expr.node),
            },
    }
}

/// Helper to collect conjuncts for filter compilation
fn collect_and_guards_for_filter(
    ctx: &EvalCtx,
    expr: &Spanned<Expr>,
    registry: &VarRegistry,
    local_scope: &LocalScope,
    guards: &mut Vec<CompiledGuard>,
) {
    if let Expr::And(a, b) = &expr.node {
        collect_and_guards_for_filter(ctx, a, registry, local_scope, guards);
        collect_and_guards_for_filter(ctx, b, registry, local_scope, guards);
    } else {
        guards.push(compile_guard_expr_for_filter(
            ctx,
            &expr.node,
            expr,
            registry,
            local_scope,
        ));
    }
}

/// Helper to collect disjuncts for filter compilation
fn collect_or_guards_for_filter(
    ctx: &EvalCtx,
    expr: &Spanned<Expr>,
    registry: &VarRegistry,
    local_scope: &LocalScope,
    guards: &mut Vec<CompiledGuard>,
) {
    if let Expr::Or(a, b) = &expr.node {
        collect_or_guards_for_filter(ctx, a, registry, local_scope, guards);
        collect_or_guards_for_filter(ctx, b, registry, local_scope, guards);
    } else {
        guards.push(compile_guard_expr_for_filter(
            ctx,
            &expr.node,
            expr,
            registry,
            local_scope,
        ));
    }
}

#[cfg(test)]
fn compile_guard_expr_action(
    ctx: &EvalCtx,
    expr: &Expr,
    full_expr: &Spanned<Expr>,
    registry: &VarRegistry,
    local_scope: &LocalScope,
) -> CompiledGuard {
    match expr {
        Expr::Bool(true) => CompiledGuard::True,
        Expr::Bool(false) => CompiledGuard::False,

        Expr::And(a, b) => {
            let mut guards = Vec::new();
            collect_and_guards_action(ctx, a, registry, local_scope, &mut guards);
            collect_and_guards_action(ctx, b, registry, local_scope, &mut guards);
            if guards.len() == 1 {
                guards.pop().unwrap()
            } else {
                CompiledGuard::And(guards)
            }
        }

        Expr::Or(a, b) => {
            let mut guards = Vec::new();
            collect_or_guards_action(ctx, a, registry, local_scope, &mut guards);
            collect_or_guards_action(ctx, b, registry, local_scope, &mut guards);
            if guards.len() == 1 {
                guards.pop().unwrap()
            } else {
                CompiledGuard::Or(guards)
            }
        }

        Expr::Not(inner) => {
            let compiled =
                compile_guard_expr_action(ctx, &inner.node, inner, registry, local_scope);
            match compiled {
                CompiledGuard::True => CompiledGuard::False,
                CompiledGuard::False => CompiledGuard::True,
                CompiledGuard::Not(inner) => *inner,
                other => CompiledGuard::Not(Box::new(other)),
            }
        }

        Expr::Eq(a, b) => compile_eq_guard_action(ctx, a, b, registry, local_scope, false),
        Expr::Neq(a, b) => compile_eq_guard_action(ctx, a, b, registry, local_scope, true),

        Expr::Lt(a, b) => compile_cmp_guard_action(ctx, a, b, CmpOp::Lt, registry, local_scope),
        Expr::Leq(a, b) => compile_cmp_guard_action(ctx, a, b, CmpOp::Le, registry, local_scope),
        Expr::Gt(a, b) => compile_cmp_guard_action(ctx, a, b, CmpOp::Gt, registry, local_scope),
        Expr::Geq(a, b) => compile_cmp_guard_action(ctx, a, b, CmpOp::Ge, registry, local_scope),

        Expr::In(elem, set) => {
            let elem_expr = compile_action_value_expr(ctx, elem, registry, local_scope);
            let set_expr = compile_action_value_expr(ctx, set, registry, local_scope);
            CompiledGuard::In {
                elem: elem_expr,
                set: set_expr,
            }
        }

        Expr::NotIn(elem, set) => {
            let elem_expr = compile_action_value_expr(ctx, elem, registry, local_scope);
            let set_expr = compile_action_value_expr(ctx, set, registry, local_scope);
            CompiledGuard::Not(Box::new(CompiledGuard::In {
                elem: elem_expr,
                set: set_expr,
            }))
        }

        Expr::Implies(a, b) => {
            let guard_a = compile_guard_expr_action(ctx, &a.node, a, registry, local_scope);
            let guard_b = compile_guard_expr_action(ctx, &b.node, b, registry, local_scope);
            CompiledGuard::Or(vec![CompiledGuard::Not(Box::new(guard_a)), guard_b])
        }

        Expr::Equiv(a, b) => {
            let guard_a = compile_guard_expr_action(ctx, &a.node, a, registry, local_scope);
            let guard_b = compile_guard_expr_action(ctx, &b.node, b, registry, local_scope);
            CompiledGuard::Or(vec![
                CompiledGuard::And(vec![guard_a.clone(), guard_b.clone()]),
                CompiledGuard::And(vec![
                    CompiledGuard::Not(Box::new(guard_a)),
                    CompiledGuard::Not(Box::new(guard_b)),
                ]),
            ])
        }

        Expr::Subseteq(a, b) => {
            let left = compile_action_value_expr(ctx, a, registry, local_scope);
            let right = compile_action_value_expr(ctx, b, registry, local_scope);
            CompiledGuard::Subseteq { left, right }
        }

        Expr::Ident(name) => {
            // Resolve operator name through replacements (e.g., Op <- Replacement)
            let resolved_name = ctx.resolve_op_name(name);
            if let Some(def) = ctx.get_op(resolved_name) {
                if def.params.is_empty() {
                    return compile_guard_expr_action(
                        ctx,
                        &def.body.node,
                        &def.body,
                        registry,
                        local_scope,
                    );
                }
            }
            CompiledGuard::Fallback {
                expr: full_expr.clone(),
                reads_next: expr_contains_any_prime(&full_expr.node),
            }
        }

        // LET expression: compile with inlined definitions
        Expr::Let(defs, body) => {
            // Compile each definition and add to local scope for inlining
            let mut extended_scope = local_scope.clone();
            let mut all_compiled = true;

            for def in defs {
                // Only support zero-argument definitions
                if !def.params.is_empty() {
                    all_compiled = false;
                    break;
                }

                // Compile the definition body
                let def_compiled = compile_value_expr_inner(
                    ctx,
                    &def.body.node,
                    &def.body,
                    registry,
                    StateRef::Current,
                    false,
                    &extended_scope,
                );

                if expr_has_fallback(&def_compiled) {
                    all_compiled = false;
                    break;
                }

                // Add to scope for subsequent definitions and body
                extended_scope = extended_scope.with_let_binding(&def.name.node, def_compiled);
            }

            if all_compiled {
                // Compile the body with inlined definitions
                let body_guard =
                    compile_guard_expr_action(ctx, &body.node, body, registry, &extended_scope);
                // IMPORTANT: If the body has any fallbacks, fall back the entire LET.
                if !guard_has_fallback(&body_guard) {
                    return body_guard;
                }
            }

            CompiledGuard::Fallback {
                expr: full_expr.clone(),
                reads_next: expr_contains_any_prime(&full_expr.node),
            }
        }

        _ => CompiledGuard::Fallback {
                expr: full_expr.clone(),
                reads_next: expr_contains_any_prime(&full_expr.node),
            },
    }
}

/// Compile an equality/inequality guard with one side being a constant.
fn compile_eq_guard(
    ctx: &EvalCtx,
    a: &Spanned<Expr>,
    b: &Spanned<Expr>,
    registry: &VarRegistry,
    local_scope: &LocalScope,
    is_neq: bool,
) -> CompiledGuard {
    // Try to identify constant vs expression pattern
    let (expr_side, const_side) = match try_eval_const(ctx, b) {
        Some(const_val) => (a, Some(const_val)),
        None => match try_eval_const(ctx, a) {
            Some(const_val) => (b, Some(const_val)),
            None => (a, None),
        },
    };

    if let Some(const_val) = const_side {
        let compiled_expr = compile_value_expr(ctx, expr_side, registry, local_scope);

        if debug_guard() {
            eprintln!("DEBUG compile_eq_guard: compiled_expr={:?}", compiled_expr);
        }

        // Try to recognize FuncApp[LocalVar] = const pattern for specialized guard
        if let CompiledExpr::FuncApp {
            state,
            func_var,
            key,
        } = &compiled_expr
        {
            if let CompiledExpr::LocalVar { depth, .. } = key.as_ref() {
                // Use specialized fused guard
                if debug_guard() {
                    eprintln!("DEBUG compile_eq_guard: creating IntFuncLocalVar{} key_depth={} expected={:?}",
                        if is_neq { "NeqConst" } else { "EqConst" }, depth, const_val);
                }
                if is_neq {
                    return CompiledGuard::IntFuncLocalVarNeqConst {
                        state: *state,
                        func_var: *func_var,
                        key_depth: *depth,
                        expected: const_val,
                    };
                } else {
                    return CompiledGuard::IntFuncLocalVarEqConst {
                        state: *state,
                        func_var: *func_var,
                        key_depth: *depth,
                        expected: const_val,
                    };
                }
            }
        }

        // Generic path
        if is_neq {
            CompiledGuard::NeqConst {
                expr: compiled_expr,
                expected: const_val,
            }
        } else {
            CompiledGuard::EqConst {
                expr: compiled_expr,
                expected: const_val,
            }
        }
    } else {
        // Neither side is constant, compile as comparison
        let left = compile_value_expr(ctx, a, registry, local_scope);
        let right = compile_value_expr(ctx, b, registry, local_scope);
        if is_neq {
            CompiledGuard::IntCmp {
                left,
                op: CmpOp::Neq,
                right,
            }
        } else {
            CompiledGuard::IntCmp {
                left,
                op: CmpOp::Eq,
                right,
            }
        }
    }
}

#[cfg(test)]
fn compile_eq_guard_action(
    ctx: &EvalCtx,
    a: &Spanned<Expr>,
    b: &Spanned<Expr>,
    registry: &VarRegistry,
    local_scope: &LocalScope,
    is_neq: bool,
) -> CompiledGuard {
    let (expr_side, const_side) = match try_eval_const(ctx, b) {
        Some(const_val) => (a, Some(const_val)),
        None => match try_eval_const(ctx, a) {
            Some(const_val) => (b, Some(const_val)),
            None => (a, None),
        },
    };

    if let Some(const_val) = const_side {
        let compiled_expr = compile_action_value_expr(ctx, expr_side, registry, local_scope);

        // Try to recognize FuncApp[LocalVar] = const pattern for specialized guard
        if let CompiledExpr::FuncApp {
            state,
            func_var,
            key,
        } = &compiled_expr
        {
            if let CompiledExpr::LocalVar { depth, .. } = key.as_ref() {
                // Use specialized fused guard
                if is_neq {
                    return CompiledGuard::IntFuncLocalVarNeqConst {
                        state: *state,
                        func_var: *func_var,
                        key_depth: *depth,
                        expected: const_val,
                    };
                } else {
                    return CompiledGuard::IntFuncLocalVarEqConst {
                        state: *state,
                        func_var: *func_var,
                        key_depth: *depth,
                        expected: const_val,
                    };
                }
            }
        }

        // Generic path
        if is_neq {
            CompiledGuard::NeqConst {
                expr: compiled_expr,
                expected: const_val,
            }
        } else {
            CompiledGuard::EqConst {
                expr: compiled_expr,
                expected: const_val,
            }
        }
    } else {
        let left = compile_action_value_expr(ctx, a, registry, local_scope);
        let right = compile_action_value_expr(ctx, b, registry, local_scope);
        if is_neq {
            CompiledGuard::IntCmp {
                left,
                op: CmpOp::Neq,
                right,
            }
        } else {
            CompiledGuard::IntCmp {
                left,
                op: CmpOp::Eq,
                right,
            }
        }
    }
}

/// Compile an integer comparison guard.
fn compile_cmp_guard(
    ctx: &EvalCtx,
    a: &Spanned<Expr>,
    b: &Spanned<Expr>,
    op: CmpOp,
    registry: &VarRegistry,
    local_scope: &LocalScope,
) -> CompiledGuard {
    let left = compile_value_expr(ctx, a, registry, local_scope);
    let right = compile_value_expr(ctx, b, registry, local_scope);
    CompiledGuard::IntCmp { left, op, right }
}

#[cfg(test)]
fn compile_cmp_guard_action(
    ctx: &EvalCtx,
    a: &Spanned<Expr>,
    b: &Spanned<Expr>,
    op: CmpOp,
    registry: &VarRegistry,
    local_scope: &LocalScope,
) -> CompiledGuard {
    let left = compile_action_value_expr(ctx, a, registry, local_scope);
    let right = compile_action_value_expr(ctx, b, registry, local_scope);
    CompiledGuard::IntCmp { left, op, right }
}

/// Compile a value expression (not necessarily boolean).
fn compile_value_expr(
    ctx: &EvalCtx,
    expr: &Spanned<Expr>,
    registry: &VarRegistry,
    local_scope: &LocalScope,
) -> CompiledExpr {
    compile_value_expr_inner(
        ctx,
        &expr.node,
        expr,
        registry,
        StateRef::Current,
        false,
        local_scope,
    )
}

fn compile_action_value_expr(
    ctx: &EvalCtx,
    expr: &Spanned<Expr>,
    registry: &VarRegistry,
    local_scope: &LocalScope,
) -> CompiledExpr {
    compile_value_expr_inner(
        ctx,
        &expr.node,
        expr,
        registry,
        StateRef::Current,
        true,
        local_scope,
    )
}

fn compile_value_expr_inner(
    ctx: &EvalCtx,
    expr: &Expr,
    full_expr: &Spanned<Expr>,
    registry: &VarRegistry,
    state: StateRef,
    allow_prime: bool,
    local_scope: &LocalScope,
) -> CompiledExpr {
    // Track compilation depth to prevent infinite recursion from recursive operators
    let depth = COMPILE_DEPTH.with(|d| {
        let current = d.get();
        d.set(current + 1);
        current + 1
    });

    // Depth check - fall back if we're too deep (likely recursive operator)
    if depth > MAX_COMPILE_DEPTH {
        COMPILE_DEPTH.with(|d| d.set(d.get() - 1));
        return CompiledExpr::Fallback {
            expr: full_expr.clone(),
            reads_next: expr_contains_any_prime(&full_expr.node),
        };
    }

    let result = compile_value_expr_inner_impl(
        ctx,
        expr,
        full_expr,
        registry,
        state,
        allow_prime,
        local_scope,
    );

    COMPILE_DEPTH.with(|d| d.set(d.get() - 1));
    result
}

#[derive(Default)]
struct BoundNameStack {
    names: Vec<String>,
}

impl BoundNameStack {
    fn contains(&self, name: &str) -> bool {
        self.names.iter().rev().any(|n| n == name)
    }

    fn mark(&self) -> usize {
        self.names.len()
    }

    fn pop_to(&mut self, mark: usize) {
        self.names.truncate(mark);
    }

    fn push_names(&mut self, names: impl IntoIterator<Item = String>) {
        self.names.extend(names);
    }
}

fn bound_names_from_bound_var(bound: &tla_core::ast::BoundVar) -> Vec<String> {
    let mut names = vec![bound.name.node.clone()];
    if let Some(pattern) = &bound.pattern {
        match pattern {
            tla_core::ast::BoundPattern::Var(v) => names.push(v.node.clone()),
            tla_core::ast::BoundPattern::Tuple(vs) => {
                names.extend(vs.iter().map(|v| v.node.clone()))
            }
        }
    }
    names
}

fn collect_free_vars(expr: &Expr, bound: &mut BoundNameStack, free: &mut HashSet<String>) {
    match expr {
        Expr::Bool(_) | Expr::Int(_) | Expr::String(_) | Expr::OpRef(_) => {}

        Expr::Ident(name) => {
            if !bound.contains(name.as_str()) {
                free.insert(name.clone());
            }
        }

        Expr::Apply(op, args) => {
            collect_free_vars(&op.node, bound, free);
            for a in args {
                collect_free_vars(&a.node, bound, free);
            }
        }
        Expr::ModuleRef(_, _, args) => {
            for a in args {
                collect_free_vars(&a.node, bound, free);
            }
        }
        Expr::InstanceExpr(_, subs) => {
            for sub in subs {
                collect_free_vars(&sub.to.node, bound, free);
            }
        }
        Expr::Lambda(params, body) => {
            let mark = bound.mark();
            bound.push_names(params.iter().map(|p| p.node.clone()));
            collect_free_vars(&body.node, bound, free);
            bound.pop_to(mark);
        }

        Expr::And(a, b)
        | Expr::Or(a, b)
        | Expr::Implies(a, b)
        | Expr::Equiv(a, b)
        | Expr::In(a, b)
        | Expr::NotIn(a, b)
        | Expr::Subseteq(a, b)
        | Expr::Union(a, b)
        | Expr::Intersect(a, b)
        | Expr::SetMinus(a, b)
        | Expr::LeadsTo(a, b)
        | Expr::WeakFair(a, b)
        | Expr::StrongFair(a, b)
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
        | Expr::FuncSet(a, b) => {
            collect_free_vars(&a.node, bound, free);
            collect_free_vars(&b.node, bound, free);
        }

        Expr::Not(a)
        | Expr::Prime(a)
        | Expr::Always(a)
        | Expr::Eventually(a)
        | Expr::Enabled(a)
        | Expr::Unchanged(a)
        | Expr::Powerset(a)
        | Expr::BigUnion(a)
        | Expr::Domain(a)
        | Expr::Neg(a) => {
            collect_free_vars(&a.node, bound, free);
        }

        Expr::If(cond, then_e, else_e) => {
            collect_free_vars(&cond.node, bound, free);
            collect_free_vars(&then_e.node, bound, free);
            collect_free_vars(&else_e.node, bound, free);
        }

        Expr::Case(arms, other) => {
            for arm in arms {
                collect_free_vars(&arm.guard.node, bound, free);
                collect_free_vars(&arm.body.node, bound, free);
            }
            if let Some(default) = other {
                collect_free_vars(&default.node, bound, free);
            }
        }

        Expr::Forall(bounds, body) | Expr::Exists(bounds, body) | Expr::FuncDef(bounds, body) => {
            for b in bounds {
                if let Some(domain) = &b.domain {
                    collect_free_vars(&domain.node, bound, free);
                }
            }
            let mark = bound.mark();
            bound.push_names(bounds.iter().flat_map(bound_names_from_bound_var));
            collect_free_vars(&body.node, bound, free);
            bound.pop_to(mark);
        }

        Expr::Choose(bound_var, body) => {
            if let Some(domain) = &bound_var.domain {
                collect_free_vars(&domain.node, bound, free);
            }
            let mark = bound.mark();
            bound.push_names(bound_names_from_bound_var(bound_var));
            collect_free_vars(&body.node, bound, free);
            bound.pop_to(mark);
        }

        Expr::SetEnum(elems) | Expr::Tuple(elems) | Expr::Times(elems) => {
            for e in elems {
                collect_free_vars(&e.node, bound, free);
            }
        }

        Expr::SetBuilder(body, bounds) => {
            for b in bounds {
                if let Some(domain) = &b.domain {
                    collect_free_vars(&domain.node, bound, free);
                }
            }
            let mark = bound.mark();
            bound.push_names(bounds.iter().flat_map(bound_names_from_bound_var));
            collect_free_vars(&body.node, bound, free);
            bound.pop_to(mark);
        }

        Expr::SetFilter(bound_var, pred) => {
            if let Some(domain) = &bound_var.domain {
                collect_free_vars(&domain.node, bound, free);
            }
            let mark = bound.mark();
            bound.push_names(bound_names_from_bound_var(bound_var));
            collect_free_vars(&pred.node, bound, free);
            bound.pop_to(mark);
        }

        Expr::FuncApply(func, arg) => {
            collect_free_vars(&func.node, bound, free);
            collect_free_vars(&arg.node, bound, free);
        }

        Expr::Except(base, specs) => {
            collect_free_vars(&base.node, bound, free);
            for spec in specs {
                for elem in &spec.path {
                    if let tla_core::ast::ExceptPathElement::Index(idx_expr) = elem {
                        collect_free_vars(&idx_expr.node, bound, free);
                    }
                }
                collect_free_vars(&spec.value.node, bound, free);
            }
        }

        Expr::Record(fields) => {
            for (_, v) in fields {
                collect_free_vars(&v.node, bound, free);
            }
        }
        Expr::RecordAccess(r, _) => collect_free_vars(&r.node, bound, free),
        Expr::RecordSet(fields) => {
            for (_, v) in fields {
                collect_free_vars(&v.node, bound, free);
            }
        }

        Expr::Let(defs, body) => {
            // LET binds operator names in the body and in def bodies (mutual recursion).
            let mark = bound.mark();
            bound.push_names(defs.iter().map(|d| d.name.node.clone()));
            for def in defs {
                let def_mark = bound.mark();
                bound.push_names(def.params.iter().map(|p| p.name.node.clone()));
                collect_free_vars(&def.body.node, bound, free);
                bound.pop_to(def_mark);
            }
            collect_free_vars(&body.node, bound, free);
            bound.pop_to(mark);
        }
    }
}

fn free_vars(expr: &Expr) -> HashSet<String> {
    let mut free = HashSet::new();
    let mut bound = BoundNameStack::default();
    collect_free_vars(expr, &mut bound, &mut free);
    free
}

fn substitution_would_capture(
    expr: &Expr,
    param_name: &str,
    arg_free: &HashSet<String>,
    bound: &mut BoundNameStack,
) -> bool {
    match expr {
        Expr::Bool(_) | Expr::Int(_) | Expr::String(_) | Expr::OpRef(_) => false,

        Expr::Ident(name) => {
            if name == param_name && !bound.contains(param_name) {
                return arg_free.iter().any(|v| bound.contains(v.as_str()));
            }
            false
        }

        Expr::Apply(op, args) => {
            if substitution_would_capture(&op.node, param_name, arg_free, bound) {
                return true;
            }
            for a in args {
                if substitution_would_capture(&a.node, param_name, arg_free, bound) {
                    return true;
                }
            }
            false
        }
        Expr::ModuleRef(_, _, args) => args
            .iter()
            .any(|a| substitution_would_capture(&a.node, param_name, arg_free, bound)),
        Expr::InstanceExpr(_, subs) => subs
            .iter()
            .any(|s| substitution_would_capture(&s.to.node, param_name, arg_free, bound)),
        Expr::Lambda(params, body) => {
            let mark = bound.mark();
            bound.push_names(params.iter().map(|p| p.node.clone()));
            let risk = substitution_would_capture(&body.node, param_name, arg_free, bound);
            bound.pop_to(mark);
            risk
        }

        Expr::And(a, b)
        | Expr::Or(a, b)
        | Expr::Implies(a, b)
        | Expr::Equiv(a, b)
        | Expr::In(a, b)
        | Expr::NotIn(a, b)
        | Expr::Subseteq(a, b)
        | Expr::Union(a, b)
        | Expr::Intersect(a, b)
        | Expr::SetMinus(a, b)
        | Expr::LeadsTo(a, b)
        | Expr::WeakFair(a, b)
        | Expr::StrongFair(a, b)
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
        | Expr::FuncSet(a, b) => {
            substitution_would_capture(&a.node, param_name, arg_free, bound)
                || substitution_would_capture(&b.node, param_name, arg_free, bound)
        }

        Expr::Not(a)
        | Expr::Prime(a)
        | Expr::Always(a)
        | Expr::Eventually(a)
        | Expr::Enabled(a)
        | Expr::Unchanged(a)
        | Expr::Powerset(a)
        | Expr::BigUnion(a)
        | Expr::Domain(a)
        | Expr::Neg(a) => substitution_would_capture(&a.node, param_name, arg_free, bound),

        Expr::If(cond, then_e, else_e) => {
            substitution_would_capture(&cond.node, param_name, arg_free, bound)
                || substitution_would_capture(&then_e.node, param_name, arg_free, bound)
                || substitution_would_capture(&else_e.node, param_name, arg_free, bound)
        }

        Expr::Case(arms, other) => {
            for arm in arms {
                if substitution_would_capture(&arm.guard.node, param_name, arg_free, bound)
                    || substitution_would_capture(&arm.body.node, param_name, arg_free, bound)
                {
                    return true;
                }
            }
            if let Some(default) = other {
                return substitution_would_capture(&default.node, param_name, arg_free, bound);
            }
            false
        }

        Expr::Forall(bounds, body) | Expr::Exists(bounds, body) | Expr::FuncDef(bounds, body) => {
            for b in bounds {
                if let Some(domain) = &b.domain {
                    if substitution_would_capture(&domain.node, param_name, arg_free, bound) {
                        return true;
                    }
                }
            }
            let mark = bound.mark();
            bound.push_names(bounds.iter().flat_map(bound_names_from_bound_var));
            let risk = substitution_would_capture(&body.node, param_name, arg_free, bound);
            bound.pop_to(mark);
            risk
        }

        Expr::Choose(bound_var, body) => {
            if let Some(domain) = &bound_var.domain {
                if substitution_would_capture(&domain.node, param_name, arg_free, bound) {
                    return true;
                }
            }
            let mark = bound.mark();
            bound.push_names(bound_names_from_bound_var(bound_var));
            let risk = substitution_would_capture(&body.node, param_name, arg_free, bound);
            bound.pop_to(mark);
            risk
        }

        Expr::SetEnum(elems) | Expr::Tuple(elems) | Expr::Times(elems) => elems
            .iter()
            .any(|e| substitution_would_capture(&e.node, param_name, arg_free, bound)),

        Expr::SetBuilder(body, bounds) => {
            for b in bounds {
                if let Some(domain) = &b.domain {
                    if substitution_would_capture(&domain.node, param_name, arg_free, bound) {
                        return true;
                    }
                }
            }
            let mark = bound.mark();
            bound.push_names(bounds.iter().flat_map(bound_names_from_bound_var));
            let risk = substitution_would_capture(&body.node, param_name, arg_free, bound);
            bound.pop_to(mark);
            risk
        }

        Expr::SetFilter(bound_var, pred) => {
            if let Some(domain) = &bound_var.domain {
                if substitution_would_capture(&domain.node, param_name, arg_free, bound) {
                    return true;
                }
            }
            let mark = bound.mark();
            bound.push_names(bound_names_from_bound_var(bound_var));
            let risk = substitution_would_capture(&pred.node, param_name, arg_free, bound);
            bound.pop_to(mark);
            risk
        }

        Expr::FuncApply(func, arg) => {
            substitution_would_capture(&func.node, param_name, arg_free, bound)
                || substitution_would_capture(&arg.node, param_name, arg_free, bound)
        }

        Expr::Except(base, specs) => {
            if substitution_would_capture(&base.node, param_name, arg_free, bound) {
                return true;
            }
            for spec in specs {
                for elem in &spec.path {
                    if let tla_core::ast::ExceptPathElement::Index(idx_expr) = elem {
                        if substitution_would_capture(&idx_expr.node, param_name, arg_free, bound) {
                            return true;
                        }
                    }
                }
                if substitution_would_capture(&spec.value.node, param_name, arg_free, bound) {
                    return true;
                }
            }
            false
        }

        Expr::Record(fields) => fields
            .iter()
            .any(|(_, v)| substitution_would_capture(&v.node, param_name, arg_free, bound)),
        Expr::RecordAccess(r, _) => {
            substitution_would_capture(&r.node, param_name, arg_free, bound)
        }
        Expr::RecordSet(fields) => fields
            .iter()
            .any(|(_, v)| substitution_would_capture(&v.node, param_name, arg_free, bound)),

        Expr::Let(defs, body) => {
            let mark = bound.mark();
            bound.push_names(defs.iter().map(|d| d.name.node.clone()));

            for def in defs {
                let def_mark = bound.mark();
                bound.push_names(def.params.iter().map(|p| p.name.node.clone()));
                if substitution_would_capture(&def.body.node, param_name, arg_free, bound) {
                    return true;
                }
                bound.pop_to(def_mark);
            }

            let risk = substitution_would_capture(&body.node, param_name, arg_free, bound);
            bound.pop_to(mark);
            risk
        }
    }
}

/// Check if inlining an operator would cause variable capture.
///
/// Returns `true` if substitution is safe (no capture), `false` if it would cause capture.
/// This is used to decide whether to inline operator applications during preprocessing.
pub fn inlining_substitution_is_capture_safe(
    def: &tla_core::ast::OperatorDef,
    args: &[Spanned<Expr>],
) -> bool {
    if def.params.len() != args.len() {
        return true;
    }
    for (param, arg) in def.params.iter().zip(args.iter()) {
        if param.arity != 0 {
            // Higher-order parameter substitutions are not inlined in value compilation paths.
            continue;
        }
        let arg_free = free_vars(&arg.node);
        if arg_free.is_empty() {
            continue;
        }
        let mut bound = BoundNameStack::default();
        if substitution_would_capture(
            &def.body.node,
            param.name.node.as_str(),
            &arg_free,
            &mut bound,
        ) {
            return false;
        }
    }
    true
}

fn compile_value_expr_inner_impl(
    ctx: &EvalCtx,
    expr: &Expr,
    full_expr: &Spanned<Expr>,
    registry: &VarRegistry,
    state: StateRef,
    allow_prime: bool,
    local_scope: &LocalScope,
) -> CompiledExpr {
    match expr {
        Expr::Prime(inner) if allow_prime => compile_value_expr_inner(
            ctx,
            &inner.node,
            inner,
            registry,
            StateRef::Next,
            allow_prime,
            local_scope,
        ),

        // Constants
        Expr::Bool(b) => CompiledExpr::Const(Value::Bool(*b)),
        Expr::Int(n) => CompiledExpr::Const(Value::big_int(n.clone())),
        Expr::String(s) => CompiledExpr::Const(Value::String(Arc::from(s.as_str()))),

        // Boolean negation: ~expr
        Expr::Not(inner) => {
            let compiled = compile_value_expr_inner(
                ctx,
                &inner.node,
                inner,
                registry,
                state,
                allow_prime,
                local_scope,
            );
            CompiledExpr::Not(Box::new(compiled))
        }

        // Boolean equality: a = b (for use in IF conditions, etc.)
        Expr::Eq(a, b) => {
            let left = compile_value_expr_inner(
                ctx,
                &a.node,
                a,
                registry,
                state,
                allow_prime,
                local_scope,
            );
            let right = compile_value_expr_inner(
                ctx,
                &b.node,
                b,
                registry,
                state,
                allow_prime,
                local_scope,
            );
            // Only use compiled path if neither side fell back
            if !expr_has_fallback(&left) && !expr_has_fallback(&right) {
                return CompiledExpr::BoolEq {
                    left: Box::new(left),
                    right: Box::new(right),
                };
            }
            CompiledExpr::Fallback {
                expr: full_expr.clone(),
                reads_next: expr_contains_any_prime(&full_expr.node),
            }
        }

        // Variable reference
        Expr::Ident(name) => {
            // Match EvalCtx lookup precedence:
            // 0) LET bindings (compile-time inlined) - highest priority
            if let Some(expr) = local_scope.get_let_binding(name) {
                return expr.clone();
            }

            // 1) local bindings (EXISTS/quantifiers) can shadow state vars and operators
            if let Some(depth) = local_scope.get_depth(name) {
                return CompiledExpr::LocalVar {
                    name: Arc::from(name.as_str()),
                    depth,
                };
            }

            // 2) state variables (current/next)
            if let Some(idx) = registry.get(name) {
                return CompiledExpr::StateVar { state, idx };
            }

            // 3) environment bindings (CONSTANT model values, etc.)
            if let Some(val) = ctx.env.get(name.as_str()) {
                return CompiledExpr::Const(val.clone());
            }

            // 4) operator replacement (CONSTANT Op <- Replacement)
            let resolved_name = ctx.resolve_op_name(name);
            if resolved_name != name.as_str() {
                if let Some(depth) = local_scope.get_depth(resolved_name) {
                    return CompiledExpr::LocalVar {
                        name: Arc::from(resolved_name),
                        depth,
                    };
                }
                if let Some(idx) = registry.get(resolved_name) {
                    return CompiledExpr::StateVar { state, idx };
                }
                if let Some(val) = ctx.env.get(resolved_name) {
                    return CompiledExpr::Const(val.clone());
                }
            }

            // 5) zero-arg operator value (inline body for compilation)
            if let Some(def) = ctx.get_op(resolved_name) {
                if def.params.is_empty() {
                    // Try to evaluate as compile-time constant first.
                    if let Some(val) = try_eval_const(ctx, &def.body) {
                        return CompiledExpr::Const(val);
                    }
                    // Inline the operator body so it can be compiled instead of falling back.
                    return compile_value_expr_inner(
                        ctx,
                        &def.body.node,
                        &def.body,
                        registry,
                        state,
                        allow_prime,
                        local_scope,
                    );
                }
            }

            // Unknown identifier - fall back to AST evaluation
            CompiledExpr::Fallback {
                expr: full_expr.clone(),
                reads_next: expr_contains_any_prime(&full_expr.node),
            }
        }

        // User-defined operator application: Op(a, b, ...)
        //
        // Inline the operator body with parameter substitutions so the resulting expression
        // can be compiled to native CompiledExpr operations instead of falling back.
        Expr::Apply(op_expr, args) => {
            if let Expr::Ident(op_name) = &op_expr.node {
                // If this name is bound as a closure (higher-order operator param), defer to eval().
                if matches!(ctx.lookup(op_name.as_str()), Some(Value::Closure(_))) {
                    return CompiledExpr::Fallback {
                        expr: full_expr.clone(),
                        reads_next: expr_contains_any_prime(&full_expr.node),
                    };
                }

                // Handle built-in sequence operators: Head, Tail, Append, Len
                match op_name.as_str() {
                    "Head" if args.len() == 1 => {
                        let compiled_seq = compile_value_expr_inner(
                            ctx,
                            &args[0].node,
                            &args[0],
                            registry,
                            state,
                            allow_prime,
                            local_scope,
                        );
                        if !expr_has_fallback(&compiled_seq) {
                            return CompiledExpr::SeqHead(Box::new(compiled_seq));
                        }
                    }
                    "Tail" if args.len() == 1 => {
                        let compiled_seq = compile_value_expr_inner(
                            ctx,
                            &args[0].node,
                            &args[0],
                            registry,
                            state,
                            allow_prime,
                            local_scope,
                        );
                        if !expr_has_fallback(&compiled_seq) {
                            return CompiledExpr::SeqTail(Box::new(compiled_seq));
                        }
                    }
                    "Append" if args.len() == 2 => {
                        let compiled_seq = compile_value_expr_inner(
                            ctx,
                            &args[0].node,
                            &args[0],
                            registry,
                            state,
                            allow_prime,
                            local_scope,
                        );
                        let compiled_elem = compile_value_expr_inner(
                            ctx,
                            &args[1].node,
                            &args[1],
                            registry,
                            state,
                            allow_prime,
                            local_scope,
                        );
                        if !expr_has_fallback(&compiled_seq) && !expr_has_fallback(&compiled_elem) {
                            return CompiledExpr::SeqAppend {
                                seq: Box::new(compiled_seq),
                                elem: Box::new(compiled_elem),
                            };
                        }
                    }
                    "Len" if args.len() == 1 => {
                        let compiled_seq = compile_value_expr_inner(
                            ctx,
                            &args[0].node,
                            &args[0],
                            registry,
                            state,
                            allow_prime,
                            local_scope,
                        );
                        if !expr_has_fallback(&compiled_seq) {
                            return CompiledExpr::SeqLen(Box::new(compiled_seq));
                        }
                    }
                    "Cardinality" if args.len() == 1 => {
                        let compiled_set = compile_value_expr_inner(
                            ctx,
                            &args[0].node,
                            &args[0],
                            registry,
                            state,
                            allow_prime,
                            local_scope,
                        );
                        // Always wrap in SetCardinality, even with fallback inside.
                        // This preserves the structure so reads_next_state() propagates correctly.
                        // Previously we fell through to a Fallback that lost reads_next info.
                        return CompiledExpr::SetCardinality(Box::new(compiled_set));
                    }
                    "Range" if args.len() == 1 => {
                        let compiled_seq = compile_value_expr_inner(
                            ctx,
                            &args[0].node,
                            &args[0],
                            registry,
                            state,
                            allow_prime,
                            local_scope,
                        );
                        if !expr_has_fallback(&compiled_seq) {
                            return CompiledExpr::SeqRange(Box::new(compiled_seq));
                        }
                    }
                    _ => {}
                }

                // Resolve operator name through replacements (e.g., Op <- Replacement)
                let resolved_name = ctx.resolve_op_name(op_name);
                if let Some(def) = ctx.get_op(resolved_name) {
                    // Only inline when arity matches and parameters are value params (not operators).
                    if def.params.len() == args.len()
                        && def.params.iter().all(|p| p.arity == 0)
                        && inlining_substitution_is_capture_safe(def, args)
                    {
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
                        return compile_value_expr_inner(
                            ctx,
                            &substituted_body.node,
                            &substituted_body,
                            registry,
                            state,
                            allow_prime,
                            local_scope,
                        );
                    }
                }
            }
            CompiledExpr::Fallback {
                expr: full_expr.clone(),
                reads_next: expr_contains_any_prime(&full_expr.node),
            }
        }

        // Function application: f[x]
        Expr::FuncApply(func, arg) => {
            // Check if func is a state variable (fast path with direct index)
            if let Expr::Ident(name) = &func.node {
                if let Some(idx) = registry.get(name) {
                    let key = compile_value_expr_inner(
                        ctx,
                        &arg.node,
                        arg,
                        registry,
                        state,
                        allow_prime,
                        local_scope,
                    );
                    // Use specialized variant if key is LocalVar (common pattern: c[i], pc[self])
                    if let CompiledExpr::LocalVar { depth, .. } = &key {
                        return CompiledExpr::FuncAppIntFuncLocalVar {
                            state,
                            func_var: idx,
                            key_depth: *depth,
                        };
                    }
                    return CompiledExpr::FuncApp {
                        state,
                        func_var: idx,
                        key: Box::new(key),
                    };
                }
            }
            if allow_prime {
                if let Expr::Prime(inner) = &func.node {
                    if let Expr::Ident(name) = &inner.node {
                        if let Some(idx) = registry.get(name) {
                            let key = compile_value_expr_inner(
                                ctx,
                                &arg.node,
                                arg,
                                registry,
                                state,
                                allow_prime,
                                local_scope,
                            );
                            // Use specialized variant if key is LocalVar
                            if let CompiledExpr::LocalVar { depth, .. } = &key {
                                return CompiledExpr::FuncAppIntFuncLocalVar {
                                    state: StateRef::Next,
                                    func_var: idx,
                                    key_depth: *depth,
                                };
                            }
                            return CompiledExpr::FuncApp {
                                state: StateRef::Next,
                                func_var: idx,
                                key: Box::new(key),
                            };
                        }
                    }
                }
            }
            // General case: compile both func and key expressions
            // This handles local variables, computed values, tuples, etc.
            let compiled_func = compile_value_expr_inner(
                ctx,
                &func.node,
                func,
                registry,
                state,
                allow_prime,
                local_scope,
            );
            let compiled_key = compile_value_expr_inner(
                ctx,
                &arg.node,
                arg,
                registry,
                state,
                allow_prime,
                local_scope,
            );

            // Only use DynFuncApp if compilation didn't result in Fallback
            // (avoid nested fallbacks which provide no benefit)
            if !expr_has_fallback(&compiled_func) && !expr_has_fallback(&compiled_key) {
                CompiledExpr::DynFuncApp {
                    func: Box::new(compiled_func),
                    key: Box::new(compiled_key),
                }
            } else {
                CompiledExpr::Fallback {
                    expr: full_expr.clone(),
                    reads_next: expr_contains_any_prime(&full_expr.node),
                }
            }
        }

        // Tuples: <<a, b, ...>>
        Expr::Tuple(elems) => {
            // Try to evaluate as constant first
            if let Some(val) = try_eval_const(ctx, full_expr) {
                return CompiledExpr::Const(val);
            }
            // Compile all elements
            let compiled: Vec<_> = elems
                .iter()
                .map(|e| {
                    compile_value_expr_inner(
                        ctx,
                        &e.node,
                        e,
                        registry,
                        state,
                        allow_prime,
                        local_scope,
                    )
                })
                .collect();
            CompiledExpr::Tuple(compiled)
        }

        // Set enumeration: {a, b, ...}
        Expr::SetEnum(elems) => {
            // Try to evaluate as constant first
            if let Some(val) = try_eval_const(ctx, full_expr) {
                return CompiledExpr::Const(val);
            }
            // Compile all elements
            let compiled: Vec<_> = elems
                .iter()
                .map(|e| {
                    compile_value_expr_inner(
                        ctx,
                        &e.node,
                        e,
                        registry,
                        state,
                        allow_prime,
                        local_scope,
                    )
                })
                .collect();
            CompiledExpr::SetEnum(compiled)
        }

        // Record constructor: [field1 |-> val1, field2 |-> val2, ...]
        Expr::Record(fields) => {
            // Try to evaluate as constant first
            if let Some(val) = try_eval_const(ctx, full_expr) {
                return CompiledExpr::Const(val);
            }
            // Compile all field values
            let compiled: Vec<(Arc<str>, CompiledExpr)> = fields
                .iter()
                .map(|(name, e)| {
                    let compiled_val = compile_value_expr_inner(
                        ctx,
                        &e.node,
                        e,
                        registry,
                        state,
                        allow_prime,
                        local_scope,
                    );
                    (Arc::from(name.node.as_str()), compiled_val)
                })
                .collect();
            // Only use Record variant if all fields compiled without fallback
            if compiled.iter().all(|(_, e)| !expr_has_fallback(e)) {
                return CompiledExpr::Record(compiled);
            }
            CompiledExpr::Fallback {
                expr: full_expr.clone(),
                reads_next: expr_contains_any_prime(&full_expr.node),
            }
        }

        // Record field access: expr.field
        Expr::RecordAccess(record_expr, field) => {
            let compiled_record = compile_value_expr_inner(
                ctx,
                &record_expr.node,
                record_expr,
                registry,
                state,
                allow_prime,
                local_scope,
            );
            if !expr_has_fallback(&compiled_record) {
                return CompiledExpr::RecordAccess {
                    record: Box::new(compiled_record),
                    field: Arc::from(field.node.as_str()),
                };
            }
            CompiledExpr::Fallback {
                expr: full_expr.clone(),
                reads_next: expr_contains_any_prime(&full_expr.node),
            }
        }

        // Set union: a \cup b
        Expr::Union(a, b) => {
            let left = compile_value_expr_inner(
                ctx,
                &a.node,
                a,
                registry,
                state,
                allow_prime,
                local_scope,
            );
            let right = compile_value_expr_inner(
                ctx,
                &b.node,
                b,
                registry,
                state,
                allow_prime,
                local_scope,
            );
            CompiledExpr::BinOp {
                left: Box::new(left),
                op: BinOp::SetUnion,
                right: Box::new(right),
            }
        }

        // Set difference: a \ b
        Expr::SetMinus(a, b) => {
            let left = compile_value_expr_inner(
                ctx,
                &a.node,
                a,
                registry,
                state,
                allow_prime,
                local_scope,
            );
            let right = compile_value_expr_inner(
                ctx,
                &b.node,
                b,
                registry,
                state,
                allow_prime,
                local_scope,
            );
            CompiledExpr::BinOp {
                left: Box::new(left),
                op: BinOp::SetDiff,
                right: Box::new(right),
            }
        }

        // Integer addition: a + b
        Expr::Add(a, b) => {
            let left = compile_value_expr_inner(
                ctx,
                &a.node,
                a,
                registry,
                state,
                allow_prime,
                local_scope,
            );
            let right = compile_value_expr_inner(
                ctx,
                &b.node,
                b,
                registry,
                state,
                allow_prime,
                local_scope,
            );
            CompiledExpr::BinOp {
                left: Box::new(left),
                op: BinOp::IntAdd,
                right: Box::new(right),
            }
        }

        // Integer subtraction: a - b
        Expr::Sub(a, b) => {
            let left = compile_value_expr_inner(
                ctx,
                &a.node,
                a,
                registry,
                state,
                allow_prime,
                local_scope,
            );
            let right = compile_value_expr_inner(
                ctx,
                &b.node,
                b,
                registry,
                state,
                allow_prime,
                local_scope,
            );
            CompiledExpr::BinOp {
                left: Box::new(left),
                op: BinOp::IntSub,
                right: Box::new(right),
            }
        }

        // Integer multiplication: a * b
        Expr::Mul(a, b) => {
            let left = compile_value_expr_inner(
                ctx,
                &a.node,
                a,
                registry,
                state,
                allow_prime,
                local_scope,
            );
            let right = compile_value_expr_inner(
                ctx,
                &b.node,
                b,
                registry,
                state,
                allow_prime,
                local_scope,
            );
            CompiledExpr::BinOp {
                left: Box::new(left),
                op: BinOp::IntMul,
                right: Box::new(right),
            }
        }

        // Integer division: a \div b
        Expr::IntDiv(a, b) => {
            let left = compile_value_expr_inner(
                ctx,
                &a.node,
                a,
                registry,
                state,
                allow_prime,
                local_scope,
            );
            let right = compile_value_expr_inner(
                ctx,
                &b.node,
                b,
                registry,
                state,
                allow_prime,
                local_scope,
            );
            CompiledExpr::BinOp {
                left: Box::new(left),
                op: BinOp::IntDiv,
                right: Box::new(right),
            }
        }

        // Integer modulo: a % b
        Expr::Mod(a, b) => {
            let left = compile_value_expr_inner(
                ctx,
                &a.node,
                a,
                registry,
                state,
                allow_prime,
                local_scope,
            );
            let right = compile_value_expr_inner(
                ctx,
                &b.node,
                b,
                registry,
                state,
                allow_prime,
                local_scope,
            );
            CompiledExpr::BinOp {
                left: Box::new(left),
                op: BinOp::IntMod,
                right: Box::new(right),
            }
        }

        // Set intersection: a \cap b
        Expr::Intersect(a, b) => {
            let left = compile_value_expr_inner(
                ctx,
                &a.node,
                a,
                registry,
                state,
                allow_prime,
                local_scope,
            );
            let right = compile_value_expr_inner(
                ctx,
                &b.node,
                b,
                registry,
                state,
                allow_prime,
                local_scope,
            );
            CompiledExpr::BinOp {
                left: Box::new(left),
                op: BinOp::SetIntersect,
                right: Box::new(right),
            }
        }

        // EXCEPT: [f EXCEPT ![k] = v] or [f EXCEPT ![k1][k2] = v]
        Expr::Except(func, specs) => {
            // Only handle single-spec EXCEPT (no multiple specs like ![a]=x, ![b]=y)
            if specs.len() == 1 && !specs[0].path.is_empty() {
                // Check that all path elements are Index (not Field)
                let all_indices = specs[0]
                    .path
                    .iter()
                    .all(|p| matches!(p, tla_core::ast::ExceptPathElement::Index(_)));

                if all_indices {
                    let func_compiled = compile_value_expr_inner(
                        ctx,
                        &func.node,
                        func,
                        registry,
                        state,
                        allow_prime,
                        local_scope,
                    );

                    // Compile all keys in the path
                    let keys: Vec<CompiledExpr> = specs[0]
                        .path
                        .iter()
                        .filter_map(|p| {
                            if let tla_core::ast::ExceptPathElement::Index(key_expr) = p {
                                Some(compile_value_expr_inner(
                                    ctx,
                                    &key_expr.node,
                                    key_expr,
                                    registry,
                                    state,
                                    allow_prime,
                                    local_scope,
                                ))
                            } else {
                                None
                            }
                        })
                        .collect();

                    // If the value contains @, substitute it with the path expression
                    // e.g., [f EXCEPT ![k1][k2] = Tail(@)] becomes
                    //       [f EXCEPT ![k1][k2] = Tail(f[k1][k2])]
                    let value_expr = if expr_contains_at(&specs[0].value.node) {
                        // Build the path expression: f[k1][k2]...
                        let path_expr = build_path_expr(func, &specs[0].path);
                        // Substitute @ with the path expression
                        let substituted = substitute_at_in_expr(&specs[0].value.node, &path_expr);
                        Spanned::new(substituted, specs[0].value.span)
                    } else {
                        specs[0].value.clone()
                    };

                    let value_compiled = compile_value_expr_inner(
                        ctx,
                        &value_expr.node,
                        &value_expr,
                        registry,
                        state,
                        allow_prime,
                        local_scope,
                    );

                    // Use simple Except for single-key paths, ExceptNested for multi-key
                    if keys.len() == 1 {
                        return CompiledExpr::Except {
                            func: Box::new(func_compiled),
                            key: Box::new(keys.into_iter().next().unwrap()),
                            value: Box::new(value_compiled),
                        };
                    } else {
                        return CompiledExpr::ExceptNested {
                            func: Box::new(func_compiled),
                            keys,
                            value: Box::new(value_compiled),
                        };
                    }
                }
            }
            CompiledExpr::Fallback {
                expr: full_expr.clone(),
                reads_next: expr_contains_any_prime(&full_expr.node),
            }
        }

        // Conditional expression: IF cond THEN then_branch ELSE else_branch
        Expr::If(cond, then_branch, else_branch) => {
            // Try to evaluate as constant first (handles IF with constant condition)
            if let Some(val) = try_eval_const(ctx, full_expr) {
                return CompiledExpr::Const(val);
            }
            // Compile all three parts
            let cond_compiled = compile_value_expr_inner(
                ctx,
                &cond.node,
                cond,
                registry,
                state,
                allow_prime,
                local_scope,
            );
            let then_compiled = compile_value_expr_inner(
                ctx,
                &then_branch.node,
                then_branch,
                registry,
                state,
                allow_prime,
                local_scope,
            );
            let else_compiled = compile_value_expr_inner(
                ctx,
                &else_branch.node,
                else_branch,
                registry,
                state,
                allow_prime,
                local_scope,
            );
            // Only use compiled path if none of the parts fell back
            if !expr_has_fallback(&cond_compiled)
                && !expr_has_fallback(&then_compiled)
                && !expr_has_fallback(&else_compiled)
            {
                return CompiledExpr::IfThenElse {
                    condition: Box::new(cond_compiled),
                    then_branch: Box::new(then_compiled),
                    else_branch: Box::new(else_compiled),
                };
            }
            // Fall back if any part couldn't be compiled
            CompiledExpr::Fallback {
                expr: full_expr.clone(),
                reads_next: expr_contains_any_prime(&full_expr.node),
            }
        }

        // Function definition: [x \in S |-> e]
        Expr::FuncDef(bounds, body) => {
            // Only handle single-bound-variable functions for now
            // Multi-variable functions like [x \in S, y \in T |-> e] fall back
            if bounds.len() != 1 {
                return CompiledExpr::Fallback {
                    expr: full_expr.clone(),
                    reads_next: expr_contains_any_prime(&full_expr.node),
                };
            }

            let bound = &bounds[0];

            // Need a domain expression
            let domain_expr = match &bound.domain {
                Some(d) => d,
                None => {
                    return CompiledExpr::Fallback {
                        expr: full_expr.clone(),
                        reads_next: expr_contains_any_prime(&full_expr.node),
                    };
                }
            };

            // Compile the domain
            let domain_compiled = compile_value_expr_inner(
                ctx,
                &domain_expr.node,
                domain_expr,
                registry,
                state,
                allow_prime,
                local_scope,
            );

            // Create extended local scope with the bound variable
            let var_name = bound.name.node.as_str();
            let inner_scope = local_scope.with_var(var_name);

            // Compile the body with the extended scope
            let body_compiled = compile_value_expr_inner(
                ctx,
                &body.node,
                body,
                registry,
                state,
                allow_prime,
                &inner_scope,
            );

            // Only use compiled path if neither part fell back
            if !expr_has_fallback(&domain_compiled) && !expr_has_fallback(&body_compiled) {
                return CompiledExpr::FuncDef {
                    var_name: Arc::from(var_name),
                    domain: Box::new(domain_compiled),
                    body: Box::new(body_compiled),
                };
            }

            // Fall back if any part couldn't be compiled
            CompiledExpr::Fallback {
                expr: full_expr.clone(),
                reads_next: expr_contains_any_prime(&full_expr.node),
            }
        }

        // LET expression: LET defs IN body
        // Compile-time inlining: compile each definition and add to local scope,
        // then compile the body. References to LET names will be replaced with
        // the compiled definition expressions.
        Expr::Let(defs, body) => {
            // Try to evaluate as constant first
            if let Some(val) = try_eval_const(ctx, full_expr) {
                return CompiledExpr::Const(val);
            }

            // Compile each definition and add to local scope for inlining
            let mut extended_scope = local_scope.clone();
            let mut all_compiled = true;

            for def in defs {
                // Only support zero-argument definitions for compilation
                if !def.params.is_empty() {
                    // Has parameters - fall back to full eval
                    all_compiled = false;
                    break;
                }

                // Compile the definition body with current scope
                let def_compiled = compile_value_expr_inner(
                    ctx,
                    &def.body.node,
                    &def.body,
                    registry,
                    state,
                    allow_prime,
                    &extended_scope,
                );

                if expr_has_fallback(&def_compiled) {
                    all_compiled = false;
                    break;
                }

                // Add to scope for subsequent definitions and body (compile-time inlining)
                extended_scope = extended_scope.with_let_binding(&def.name.node, def_compiled);
            }

            if all_compiled {
                // Compile the body with all definitions in scope for inlining
                let body_compiled = compile_value_expr_inner(
                    ctx,
                    &body.node,
                    body,
                    registry,
                    state,
                    allow_prime,
                    &extended_scope,
                );

                if !expr_has_fallback(&body_compiled) {
                    // The body already has all LET references inlined
                    return body_compiled;
                }
            }

            // Fall back if any part couldn't be compiled
            CompiledExpr::Fallback {
                expr: full_expr.clone(),
                reads_next: expr_contains_any_prime(&full_expr.node),
            }
        }

        // Everything else - try to evaluate as a constant, otherwise fall back
        _ => {
            // Try to evaluate as compile-time constant (handles things like Proc × M)
            // This may fail if the expression references local variables or state vars
            if let Ok(val) = eval(ctx, full_expr) {
                return CompiledExpr::Const(val);
            }
            CompiledExpr::Fallback {
                expr: full_expr.clone(),
                reads_next: expr_contains_any_prime(&full_expr.node),
            }
        }
    }
}

/// Check if an AST expression contains the `@` identifier (EXCEPT self-reference).
/// Returns true if `@` appears anywhere in the expression.
fn expr_contains_at(expr: &Expr) -> bool {
    use tla_core::ast::ExceptPathElement;
    match expr {
        // @ is represented as Ident("@")
        Expr::Ident(name) => name == "@",
        // Literals never contain @
        Expr::Bool(_) | Expr::Int(_) | Expr::String(_) | Expr::OpRef(_) => false,
        // Unary expressions - check single child
        Expr::Prime(a)
        | Expr::Not(a)
        | Expr::Neg(a)
        | Expr::Powerset(a)
        | Expr::BigUnion(a)
        | Expr::Domain(a)
        | Expr::Always(a)
        | Expr::Eventually(a)
        | Expr::Enabled(a)
        | Expr::Unchanged(a)
        | Expr::RecordAccess(a, _) => expr_contains_at(&a.node),
        // Binary expressions - check both children
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
        | Expr::Add(a, b)
        | Expr::Sub(a, b)
        | Expr::Mul(a, b)
        | Expr::Div(a, b)
        | Expr::IntDiv(a, b)
        | Expr::Mod(a, b)
        | Expr::Pow(a, b)
        | Expr::Union(a, b)
        | Expr::Intersect(a, b)
        | Expr::SetMinus(a, b)
        | Expr::Range(a, b)
        | Expr::FuncSet(a, b)
        | Expr::FuncApply(a, b)
        | Expr::WeakFair(a, b)
        | Expr::StrongFair(a, b)
        | Expr::LeadsTo(a, b) => expr_contains_at(&a.node) || expr_contains_at(&b.node),
        // Ternary IF expression
        Expr::If(c, t, e) => {
            expr_contains_at(&c.node) || expr_contains_at(&t.node) || expr_contains_at(&e.node)
        }
        // Operator application and Lambda
        Expr::Apply(op, args) => {
            expr_contains_at(&op.node) || args.iter().any(|a| expr_contains_at(&a.node))
        }
        Expr::Lambda(_, body) => expr_contains_at(&body.node),
        // List-based expressions (SetEnum, Tuple, Times/cartesian product)
        Expr::SetEnum(elems) | Expr::Tuple(elems) | Expr::Times(elems) => {
            elems.iter().any(|e| expr_contains_at(&e.node))
        }
        // Record expressions
        Expr::Record(fields) | Expr::RecordSet(fields) => {
            fields.iter().any(|(_, e)| expr_contains_at(&e.node))
        }
        // Quantifiers
        Expr::Forall(bounds, body) | Expr::Exists(bounds, body) => {
            bounds
                .iter()
                .any(|b| b.domain.as_ref().is_some_and(|d| expr_contains_at(&d.node)))
                || expr_contains_at(&body.node)
        }
        Expr::Choose(bound, body) => {
            bound
                .domain
                .as_ref()
                .is_some_and(|d| expr_contains_at(&d.node))
                || expr_contains_at(&body.node)
        }
        // Set comprehensions
        Expr::SetBuilder(body, bounds) => {
            expr_contains_at(&body.node)
                || bounds
                    .iter()
                    .any(|b| b.domain.as_ref().is_some_and(|d| expr_contains_at(&d.node)))
        }
        Expr::SetFilter(bound, pred) => {
            bound
                .domain
                .as_ref()
                .is_some_and(|d| expr_contains_at(&d.node))
                || expr_contains_at(&pred.node)
        }
        // Function definition
        Expr::FuncDef(bounds, body) => {
            bounds
                .iter()
                .any(|b| b.domain.as_ref().is_some_and(|d| expr_contains_at(&d.node)))
                || expr_contains_at(&body.node)
        }
        // EXCEPT expression
        Expr::Except(base, specs) => {
            expr_contains_at(&base.node)
                || specs.iter().any(|s| {
                    s.path.iter().any(|p| match p {
                        ExceptPathElement::Index(idx) => expr_contains_at(&idx.node),
                        ExceptPathElement::Field(_) => false,
                    }) || expr_contains_at(&s.value.node)
                })
        }
        // LET expression
        Expr::Let(defs, body) => {
            defs.iter().any(|d| expr_contains_at(&d.body.node)) || expr_contains_at(&body.node)
        }
        // CASE expression
        Expr::Case(arms, other) => {
            arms.iter()
                .any(|a| expr_contains_at(&a.guard.node) || expr_contains_at(&a.body.node))
                || other.as_ref().is_some_and(|o| expr_contains_at(&o.node))
        }
        // Module references - these don't contain @
        Expr::InstanceExpr(_, _) | Expr::ModuleRef(_, _, _) => false,
    }
}

/// Substitute all occurrences of `@` in an expression with a replacement expression.
/// Used to compile EXCEPT expressions that contain `@` (self-reference).
///
/// For `[f EXCEPT ![k1][k2] = Tail(@)]`, we substitute @ with `f[k1][k2]`
/// to get `[f EXCEPT ![k1][k2] = Tail(f[k1][k2])]` which can be compiled normally.
fn substitute_at_in_expr(expr: &Expr, replacement: &Spanned<Expr>) -> Expr {
    use tla_core::ast::{CaseArm, ExceptPathElement, ExceptSpec, OperatorDef};

    match expr {
        // @ is replaced with the replacement expression
        Expr::Ident(name) if name == "@" => replacement.node.clone(),

        // Literals and non-@ identifiers pass through unchanged
        Expr::Bool(_) | Expr::Int(_) | Expr::String(_) | Expr::OpRef(_) | Expr::Ident(_) => {
            expr.clone()
        }

        // Unary expressions - substitute in child
        Expr::Prime(a) => Expr::Prime(Box::new(Spanned::new(
            substitute_at_in_expr(&a.node, replacement),
            a.span,
        ))),
        Expr::Not(a) => Expr::Not(Box::new(Spanned::new(
            substitute_at_in_expr(&a.node, replacement),
            a.span,
        ))),
        Expr::Neg(a) => Expr::Neg(Box::new(Spanned::new(
            substitute_at_in_expr(&a.node, replacement),
            a.span,
        ))),
        Expr::Powerset(a) => Expr::Powerset(Box::new(Spanned::new(
            substitute_at_in_expr(&a.node, replacement),
            a.span,
        ))),
        Expr::BigUnion(a) => Expr::BigUnion(Box::new(Spanned::new(
            substitute_at_in_expr(&a.node, replacement),
            a.span,
        ))),
        Expr::Domain(a) => Expr::Domain(Box::new(Spanned::new(
            substitute_at_in_expr(&a.node, replacement),
            a.span,
        ))),
        Expr::Always(a) => Expr::Always(Box::new(Spanned::new(
            substitute_at_in_expr(&a.node, replacement),
            a.span,
        ))),
        Expr::Eventually(a) => Expr::Eventually(Box::new(Spanned::new(
            substitute_at_in_expr(&a.node, replacement),
            a.span,
        ))),
        Expr::Enabled(a) => Expr::Enabled(Box::new(Spanned::new(
            substitute_at_in_expr(&a.node, replacement),
            a.span,
        ))),
        Expr::Unchanged(a) => Expr::Unchanged(Box::new(Spanned::new(
            substitute_at_in_expr(&a.node, replacement),
            a.span,
        ))),
        Expr::RecordAccess(a, field) => Expr::RecordAccess(
            Box::new(Spanned::new(
                substitute_at_in_expr(&a.node, replacement),
                a.span,
            )),
            field.clone(),
        ),

        // Binary expressions - substitute in both children
        Expr::And(a, b) => Expr::And(
            Box::new(Spanned::new(
                substitute_at_in_expr(&a.node, replacement),
                a.span,
            )),
            Box::new(Spanned::new(
                substitute_at_in_expr(&b.node, replacement),
                b.span,
            )),
        ),
        Expr::Or(a, b) => Expr::Or(
            Box::new(Spanned::new(
                substitute_at_in_expr(&a.node, replacement),
                a.span,
            )),
            Box::new(Spanned::new(
                substitute_at_in_expr(&b.node, replacement),
                b.span,
            )),
        ),
        Expr::Implies(a, b) => Expr::Implies(
            Box::new(Spanned::new(
                substitute_at_in_expr(&a.node, replacement),
                a.span,
            )),
            Box::new(Spanned::new(
                substitute_at_in_expr(&b.node, replacement),
                b.span,
            )),
        ),
        Expr::Equiv(a, b) => Expr::Equiv(
            Box::new(Spanned::new(
                substitute_at_in_expr(&a.node, replacement),
                a.span,
            )),
            Box::new(Spanned::new(
                substitute_at_in_expr(&b.node, replacement),
                b.span,
            )),
        ),
        Expr::Eq(a, b) => Expr::Eq(
            Box::new(Spanned::new(
                substitute_at_in_expr(&a.node, replacement),
                a.span,
            )),
            Box::new(Spanned::new(
                substitute_at_in_expr(&b.node, replacement),
                b.span,
            )),
        ),
        Expr::Neq(a, b) => Expr::Neq(
            Box::new(Spanned::new(
                substitute_at_in_expr(&a.node, replacement),
                a.span,
            )),
            Box::new(Spanned::new(
                substitute_at_in_expr(&b.node, replacement),
                b.span,
            )),
        ),
        Expr::Lt(a, b) => Expr::Lt(
            Box::new(Spanned::new(
                substitute_at_in_expr(&a.node, replacement),
                a.span,
            )),
            Box::new(Spanned::new(
                substitute_at_in_expr(&b.node, replacement),
                b.span,
            )),
        ),
        Expr::Leq(a, b) => Expr::Leq(
            Box::new(Spanned::new(
                substitute_at_in_expr(&a.node, replacement),
                a.span,
            )),
            Box::new(Spanned::new(
                substitute_at_in_expr(&b.node, replacement),
                b.span,
            )),
        ),
        Expr::Gt(a, b) => Expr::Gt(
            Box::new(Spanned::new(
                substitute_at_in_expr(&a.node, replacement),
                a.span,
            )),
            Box::new(Spanned::new(
                substitute_at_in_expr(&b.node, replacement),
                b.span,
            )),
        ),
        Expr::Geq(a, b) => Expr::Geq(
            Box::new(Spanned::new(
                substitute_at_in_expr(&a.node, replacement),
                a.span,
            )),
            Box::new(Spanned::new(
                substitute_at_in_expr(&b.node, replacement),
                b.span,
            )),
        ),
        Expr::In(a, b) => Expr::In(
            Box::new(Spanned::new(
                substitute_at_in_expr(&a.node, replacement),
                a.span,
            )),
            Box::new(Spanned::new(
                substitute_at_in_expr(&b.node, replacement),
                b.span,
            )),
        ),
        Expr::NotIn(a, b) => Expr::NotIn(
            Box::new(Spanned::new(
                substitute_at_in_expr(&a.node, replacement),
                a.span,
            )),
            Box::new(Spanned::new(
                substitute_at_in_expr(&b.node, replacement),
                b.span,
            )),
        ),
        Expr::Subseteq(a, b) => Expr::Subseteq(
            Box::new(Spanned::new(
                substitute_at_in_expr(&a.node, replacement),
                a.span,
            )),
            Box::new(Spanned::new(
                substitute_at_in_expr(&b.node, replacement),
                b.span,
            )),
        ),
        Expr::Add(a, b) => Expr::Add(
            Box::new(Spanned::new(
                substitute_at_in_expr(&a.node, replacement),
                a.span,
            )),
            Box::new(Spanned::new(
                substitute_at_in_expr(&b.node, replacement),
                b.span,
            )),
        ),
        Expr::Sub(a, b) => Expr::Sub(
            Box::new(Spanned::new(
                substitute_at_in_expr(&a.node, replacement),
                a.span,
            )),
            Box::new(Spanned::new(
                substitute_at_in_expr(&b.node, replacement),
                b.span,
            )),
        ),
        Expr::Mul(a, b) => Expr::Mul(
            Box::new(Spanned::new(
                substitute_at_in_expr(&a.node, replacement),
                a.span,
            )),
            Box::new(Spanned::new(
                substitute_at_in_expr(&b.node, replacement),
                b.span,
            )),
        ),
        Expr::Div(a, b) => Expr::Div(
            Box::new(Spanned::new(
                substitute_at_in_expr(&a.node, replacement),
                a.span,
            )),
            Box::new(Spanned::new(
                substitute_at_in_expr(&b.node, replacement),
                b.span,
            )),
        ),
        Expr::IntDiv(a, b) => Expr::IntDiv(
            Box::new(Spanned::new(
                substitute_at_in_expr(&a.node, replacement),
                a.span,
            )),
            Box::new(Spanned::new(
                substitute_at_in_expr(&b.node, replacement),
                b.span,
            )),
        ),
        Expr::Mod(a, b) => Expr::Mod(
            Box::new(Spanned::new(
                substitute_at_in_expr(&a.node, replacement),
                a.span,
            )),
            Box::new(Spanned::new(
                substitute_at_in_expr(&b.node, replacement),
                b.span,
            )),
        ),
        Expr::Pow(a, b) => Expr::Pow(
            Box::new(Spanned::new(
                substitute_at_in_expr(&a.node, replacement),
                a.span,
            )),
            Box::new(Spanned::new(
                substitute_at_in_expr(&b.node, replacement),
                b.span,
            )),
        ),
        Expr::Union(a, b) => Expr::Union(
            Box::new(Spanned::new(
                substitute_at_in_expr(&a.node, replacement),
                a.span,
            )),
            Box::new(Spanned::new(
                substitute_at_in_expr(&b.node, replacement),
                b.span,
            )),
        ),
        Expr::Intersect(a, b) => Expr::Intersect(
            Box::new(Spanned::new(
                substitute_at_in_expr(&a.node, replacement),
                a.span,
            )),
            Box::new(Spanned::new(
                substitute_at_in_expr(&b.node, replacement),
                b.span,
            )),
        ),
        Expr::SetMinus(a, b) => Expr::SetMinus(
            Box::new(Spanned::new(
                substitute_at_in_expr(&a.node, replacement),
                a.span,
            )),
            Box::new(Spanned::new(
                substitute_at_in_expr(&b.node, replacement),
                b.span,
            )),
        ),
        Expr::Range(a, b) => Expr::Range(
            Box::new(Spanned::new(
                substitute_at_in_expr(&a.node, replacement),
                a.span,
            )),
            Box::new(Spanned::new(
                substitute_at_in_expr(&b.node, replacement),
                b.span,
            )),
        ),
        Expr::FuncSet(a, b) => Expr::FuncSet(
            Box::new(Spanned::new(
                substitute_at_in_expr(&a.node, replacement),
                a.span,
            )),
            Box::new(Spanned::new(
                substitute_at_in_expr(&b.node, replacement),
                b.span,
            )),
        ),
        Expr::FuncApply(a, b) => Expr::FuncApply(
            Box::new(Spanned::new(
                substitute_at_in_expr(&a.node, replacement),
                a.span,
            )),
            Box::new(Spanned::new(
                substitute_at_in_expr(&b.node, replacement),
                b.span,
            )),
        ),
        Expr::WeakFair(a, b) => Expr::WeakFair(
            Box::new(Spanned::new(
                substitute_at_in_expr(&a.node, replacement),
                a.span,
            )),
            Box::new(Spanned::new(
                substitute_at_in_expr(&b.node, replacement),
                b.span,
            )),
        ),
        Expr::StrongFair(a, b) => Expr::StrongFair(
            Box::new(Spanned::new(
                substitute_at_in_expr(&a.node, replacement),
                a.span,
            )),
            Box::new(Spanned::new(
                substitute_at_in_expr(&b.node, replacement),
                b.span,
            )),
        ),
        Expr::LeadsTo(a, b) => Expr::LeadsTo(
            Box::new(Spanned::new(
                substitute_at_in_expr(&a.node, replacement),
                a.span,
            )),
            Box::new(Spanned::new(
                substitute_at_in_expr(&b.node, replacement),
                b.span,
            )),
        ),

        // Ternary IF expression
        Expr::If(c, t, e) => Expr::If(
            Box::new(Spanned::new(
                substitute_at_in_expr(&c.node, replacement),
                c.span,
            )),
            Box::new(Spanned::new(
                substitute_at_in_expr(&t.node, replacement),
                t.span,
            )),
            Box::new(Spanned::new(
                substitute_at_in_expr(&e.node, replacement),
                e.span,
            )),
        ),

        // Operator application
        Expr::Apply(op, args) => Expr::Apply(
            Box::new(Spanned::new(
                substitute_at_in_expr(&op.node, replacement),
                op.span,
            )),
            args.iter()
                .map(|a| Spanned::new(substitute_at_in_expr(&a.node, replacement), a.span))
                .collect(),
        ),

        // Lambda - only substitute in body (params are bound names)
        Expr::Lambda(params, body) => Expr::Lambda(
            params.clone(),
            Box::new(Spanned::new(
                substitute_at_in_expr(&body.node, replacement),
                body.span,
            )),
        ),

        // List-based expressions
        Expr::SetEnum(elems) => Expr::SetEnum(
            elems
                .iter()
                .map(|e| Spanned::new(substitute_at_in_expr(&e.node, replacement), e.span))
                .collect(),
        ),
        Expr::Tuple(elems) => Expr::Tuple(
            elems
                .iter()
                .map(|e| Spanned::new(substitute_at_in_expr(&e.node, replacement), e.span))
                .collect(),
        ),
        Expr::Times(elems) => Expr::Times(
            elems
                .iter()
                .map(|e| Spanned::new(substitute_at_in_expr(&e.node, replacement), e.span))
                .collect(),
        ),

        // Record expressions
        Expr::Record(fields) => Expr::Record(
            fields
                .iter()
                .map(|(name, e)| {
                    (
                        name.clone(),
                        Spanned::new(substitute_at_in_expr(&e.node, replacement), e.span),
                    )
                })
                .collect(),
        ),
        Expr::RecordSet(fields) => Expr::RecordSet(
            fields
                .iter()
                .map(|(name, e)| {
                    (
                        name.clone(),
                        Spanned::new(substitute_at_in_expr(&e.node, replacement), e.span),
                    )
                })
                .collect(),
        ),

        // Quantifiers - substitute in domain and body
        Expr::Forall(bounds, body) => Expr::Forall(
            bounds
                .iter()
                .map(|b| substitute_bound(b, replacement))
                .collect(),
            Box::new(Spanned::new(
                substitute_at_in_expr(&body.node, replacement),
                body.span,
            )),
        ),
        Expr::Exists(bounds, body) => Expr::Exists(
            bounds
                .iter()
                .map(|b| substitute_bound(b, replacement))
                .collect(),
            Box::new(Spanned::new(
                substitute_at_in_expr(&body.node, replacement),
                body.span,
            )),
        ),
        Expr::Choose(bound, body) => Expr::Choose(
            substitute_quant_bound(bound, replacement),
            Box::new(Spanned::new(
                substitute_at_in_expr(&body.node, replacement),
                body.span,
            )),
        ),

        // Set comprehensions
        Expr::SetBuilder(body, bounds) => Expr::SetBuilder(
            Box::new(Spanned::new(
                substitute_at_in_expr(&body.node, replacement),
                body.span,
            )),
            bounds
                .iter()
                .map(|b| substitute_bound(b, replacement))
                .collect(),
        ),
        Expr::SetFilter(bound, pred) => Expr::SetFilter(
            substitute_quant_bound(bound, replacement),
            Box::new(Spanned::new(
                substitute_at_in_expr(&pred.node, replacement),
                pred.span,
            )),
        ),

        // Function definition
        Expr::FuncDef(bounds, body) => Expr::FuncDef(
            bounds
                .iter()
                .map(|b| substitute_bound(b, replacement))
                .collect(),
            Box::new(Spanned::new(
                substitute_at_in_expr(&body.node, replacement),
                body.span,
            )),
        ),

        // EXCEPT expression - substitute in base, path indices, and values
        Expr::Except(base, specs) => Expr::Except(
            Box::new(Spanned::new(
                substitute_at_in_expr(&base.node, replacement),
                base.span,
            )),
            specs
                .iter()
                .map(|s| ExceptSpec {
                    path: s
                        .path
                        .iter()
                        .map(|p| match p {
                            ExceptPathElement::Index(idx) => {
                                ExceptPathElement::Index(Spanned::new(
                                    substitute_at_in_expr(&idx.node, replacement),
                                    idx.span,
                                ))
                            }
                            ExceptPathElement::Field(f) => ExceptPathElement::Field(f.clone()),
                        })
                        .collect(),
                    value: Spanned::new(
                        substitute_at_in_expr(&s.value.node, replacement),
                        s.value.span,
                    ),
                })
                .collect(),
        ),

        // LET expression
        Expr::Let(defs, body) => Expr::Let(
            defs.iter()
                .map(|d| OperatorDef {
                    name: d.name.clone(),
                    params: d.params.clone(),
                    body: Spanned::new(
                        substitute_at_in_expr(&d.body.node, replacement),
                        d.body.span,
                    ),
                    local: d.local,
                })
                .collect(),
            Box::new(Spanned::new(
                substitute_at_in_expr(&body.node, replacement),
                body.span,
            )),
        ),

        // CASE expression
        Expr::Case(arms, other) => Expr::Case(
            arms.iter()
                .map(|a| CaseArm {
                    guard: Spanned::new(
                        substitute_at_in_expr(&a.guard.node, replacement),
                        a.guard.span,
                    ),
                    body: Spanned::new(
                        substitute_at_in_expr(&a.body.node, replacement),
                        a.body.span,
                    ),
                })
                .collect(),
            other.as_ref().map(|o| {
                Box::new(Spanned::new(
                    substitute_at_in_expr(&o.node, replacement),
                    o.span,
                ))
            }),
        ),

        // Module references - pass through unchanged (these don't contain @)
        Expr::InstanceExpr(_, _) | Expr::ModuleRef(_, _, _) => expr.clone(),
    }
}

/// Helper to substitute @ in a BoundVar
fn substitute_bound(
    bound: &tla_core::ast::BoundVar,
    replacement: &Spanned<Expr>,
) -> tla_core::ast::BoundVar {
    tla_core::ast::BoundVar {
        name: bound.name.clone(),
        domain: bound.domain.as_ref().map(|d| {
            Box::new(Spanned::new(
                substitute_at_in_expr(&d.node, replacement),
                d.span,
            ))
        }),
        pattern: bound.pattern.clone(),
    }
}

// Alias for single-bound quantifiers that use the same BoundVar type
fn substitute_quant_bound(
    bound: &tla_core::ast::BoundVar,
    replacement: &Spanned<Expr>,
) -> tla_core::ast::BoundVar {
    substitute_bound(bound, replacement)
}

/// Build a path expression: f[k1][k2]...[kn] as nested FuncApply
/// Returns the AST representation of accessing the function at the given path.
fn build_path_expr(
    func: &Spanned<Expr>,
    path: &[tla_core::ast::ExceptPathElement],
) -> Spanned<Expr> {
    let mut result = func.clone();
    for elem in path {
        match elem {
            tla_core::ast::ExceptPathElement::Index(key) => {
                result = Spanned::new(
                    Expr::FuncApply(Box::new(result), Box::new(key.clone())),
                    func.span, // Use func's span for the whole expression
                );
            }
            tla_core::ast::ExceptPathElement::Field(field) => {
                result = Spanned::new(
                    Expr::RecordAccess(Box::new(result), field.clone()),
                    func.span,
                );
            }
        }
    }
    result
}

/// Check if a compiled expression would fall back to full AST evaluation.
///
/// Returns true if the expression contains any `Fallback` variant.
pub fn expr_has_fallback(expr: &CompiledExpr) -> bool {
    match expr {
        CompiledExpr::Const(_)
        | CompiledExpr::StateVar { .. }
        | CompiledExpr::LocalVar { .. }
        | CompiledExpr::FuncAppIntFuncLocalVar { .. } => false,
        CompiledExpr::FuncApp { key, .. } => expr_has_fallback(key),
        CompiledExpr::DynFuncApp { func, key } => expr_has_fallback(func) || expr_has_fallback(key),
        CompiledExpr::BinOp { left, right, .. } => {
            expr_has_fallback(left) || expr_has_fallback(right)
        }
        CompiledExpr::Except { func, key, value } => {
            expr_has_fallback(func) || expr_has_fallback(key) || expr_has_fallback(value)
        }
        CompiledExpr::ExceptNested { func, keys, value } => {
            expr_has_fallback(func)
                || keys.iter().any(expr_has_fallback)
                || expr_has_fallback(value)
        }
        CompiledExpr::SetEnum(elems) | CompiledExpr::Tuple(elems) => {
            elems.iter().any(expr_has_fallback)
        }
        CompiledExpr::Not(inner) => expr_has_fallback(inner),
        CompiledExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => {
            expr_has_fallback(condition)
                || expr_has_fallback(then_branch)
                || expr_has_fallback(else_branch)
        }
        CompiledExpr::FuncDef { domain, body, .. } => {
            expr_has_fallback(domain) || expr_has_fallback(body)
        }
        CompiledExpr::SeqHead(seq)
        | CompiledExpr::SeqTail(seq)
        | CompiledExpr::SeqLen(seq)
        | CompiledExpr::SetCardinality(seq)
        | CompiledExpr::SeqRange(seq) => expr_has_fallback(seq),
        CompiledExpr::SeqAppend { seq, elem } => expr_has_fallback(seq) || expr_has_fallback(elem),
        CompiledExpr::Record(fields) => fields.iter().any(|(_, e)| expr_has_fallback(e)),
        CompiledExpr::BoolEq { left, right } => expr_has_fallback(left) || expr_has_fallback(right),
        CompiledExpr::RecordAccess { record, .. } => expr_has_fallback(record),
        CompiledExpr::Fallback { .. } => true,
    }
}

fn guard_has_fallback(guard: &CompiledGuard) -> bool {
    match guard {
        CompiledGuard::True | CompiledGuard::False => false,
        CompiledGuard::EqConst { expr, .. } | CompiledGuard::NeqConst { expr, .. } => {
            expr_has_fallback(expr)
        }
        // Specialized fused guards have no fallback - they're fully compiled
        CompiledGuard::IntFuncLocalVarEqConst { .. }
        | CompiledGuard::IntFuncLocalVarNeqConst { .. } => false,
        CompiledGuard::IntCmp { left, right, .. } => {
            expr_has_fallback(left) || expr_has_fallback(right)
        }
        CompiledGuard::In { elem, set } => expr_has_fallback(elem) || expr_has_fallback(set),
        CompiledGuard::Subseteq { left, right } => {
            expr_has_fallback(left) || expr_has_fallback(right)
        }
        CompiledGuard::And(guards) | CompiledGuard::Or(guards) => {
            guards.iter().any(guard_has_fallback)
        }
        CompiledGuard::Not(inner) => guard_has_fallback(inner),
        CompiledGuard::ForAll { domain, body, .. } => {
            expr_has_fallback(domain) || guard_has_fallback(body)
        }
        CompiledGuard::Exists { domain, body, .. } => {
            expr_has_fallback(domain) || guard_has_fallback(body)
        }
        CompiledGuard::Implies {
            antecedent,
            consequent,
        } => guard_has_fallback(antecedent) || guard_has_fallback(consequent),
        CompiledGuard::Fallback { .. } => true,
    }
}

/// Check if a compiled guard reads from next-state (contains primed variables).
///
/// Guards that read next-state cannot be short-circuited during enumeration because
/// the next-state values are not yet available. Instead, they must be validated
/// against each generated successor state.
///
/// This catches "hidden primes" - primed variables inside operator bodies that aren't
/// visible syntactically (e.g., `rcvd0(self) >= N` where `rcvd0` uses `rcvd'[self]`).
pub fn guard_reads_next_state(guard: &CompiledGuard) -> bool {
    match guard {
        CompiledGuard::True | CompiledGuard::False => false,
        CompiledGuard::EqConst { expr, .. } | CompiledGuard::NeqConst { expr, .. } => {
            expr.reads_next_state()
        }
        // Specialized fused guards reference state - check their state ref
        CompiledGuard::IntFuncLocalVarEqConst { state, .. }
        | CompiledGuard::IntFuncLocalVarNeqConst { state, .. } => *state == StateRef::Next,
        CompiledGuard::IntCmp { left, right, .. } => {
            left.reads_next_state() || right.reads_next_state()
        }
        CompiledGuard::In { elem, set } => elem.reads_next_state() || set.reads_next_state(),
        CompiledGuard::Subseteq { left, right } => {
            left.reads_next_state() || right.reads_next_state()
        }
        CompiledGuard::And(guards) | CompiledGuard::Or(guards) => {
            guards.iter().any(guard_reads_next_state)
        }
        CompiledGuard::Not(inner) => guard_reads_next_state(inner),
        CompiledGuard::ForAll { domain, body, .. } => {
            domain.reads_next_state() || guard_reads_next_state(body)
        }
        CompiledGuard::Exists { domain, body, .. } => {
            domain.reads_next_state() || guard_reads_next_state(body)
        }
        CompiledGuard::Implies {
            antecedent,
            consequent,
        } => guard_reads_next_state(antecedent) || guard_reads_next_state(consequent),
        CompiledGuard::Fallback { reads_next, .. } => *reads_next,
    }
}

/// Try to evaluate an expression as a compile-time constant.
fn try_eval_const(ctx: &EvalCtx, expr: &Spanned<Expr>) -> Option<Value> {
    // Only try for expressions that are likely constant
    match &expr.node {
        Expr::Bool(b) => Some(Value::Bool(*b)),
        Expr::Int(n) => Some(Value::big_int(n.clone())),
        Expr::String(s) => Some(Value::String(Arc::from(s.as_str()))),
        Expr::SetEnum(elems) if elems.is_empty() => Some(Value::Set(SortedSet::new())),
        Expr::Tuple(elems) => {
            // Try to evaluate all elements as constants
            let mut vals = Vec::with_capacity(elems.len());
            for elem in elems {
                vals.push(try_eval_const(ctx, elem)?);
            }
            Some(Value::Tuple(vals.into()))
        }
        Expr::Ident(name) => {
            // Resolve operator name through replacements (e.g., Op <- Replacement)
            let resolved_name = ctx.resolve_op_name(name);
            // Check if it's a zero-arg operator that evaluates to a constant
            if let Some(def) = ctx.get_op(resolved_name) {
                if def.params.is_empty() {
                    // Try to evaluate (may fail if it references state variables)
                    return eval(ctx, &def.body).ok();
                }
            }
            // Check if it's a CONSTANT value bound in ctx.env (e.g., model values)
            if let Some(val) = ctx.env.get(resolved_name) {
                return Some(val.clone());
            }
            None
        }
        _ => None,
    }
}

/// Collect AND guards into a flat list.
fn collect_and_guards(
    ctx: &EvalCtx,
    expr: &Spanned<Expr>,
    registry: &VarRegistry,
    local_scope: &LocalScope,
    guards: &mut Vec<CompiledGuard>,
) {
    if let Expr::And(a, b) = &expr.node {
        collect_and_guards(ctx, a, registry, local_scope, guards);
        collect_and_guards(ctx, b, registry, local_scope, guards);
    } else {
        let compiled = compile_guard(ctx, expr, registry, local_scope);
        // Skip trivially true guards
        if !matches!(compiled, CompiledGuard::True) {
            guards.push(compiled);
        }
    }
}

/// Collect OR guards into a flat list.
fn collect_or_guards(
    ctx: &EvalCtx,
    expr: &Spanned<Expr>,
    registry: &VarRegistry,
    local_scope: &LocalScope,
    guards: &mut Vec<CompiledGuard>,
) {
    if let Expr::Or(a, b) = &expr.node {
        collect_or_guards(ctx, a, registry, local_scope, guards);
        collect_or_guards(ctx, b, registry, local_scope, guards);
    } else {
        let compiled = compile_guard(ctx, expr, registry, local_scope);
        // Skip trivially false guards
        if !matches!(compiled, CompiledGuard::False) {
            guards.push(compiled);
        }
    }
}

#[cfg(test)]
fn collect_and_guards_action(
    ctx: &EvalCtx,
    expr: &Spanned<Expr>,
    registry: &VarRegistry,
    local_scope: &LocalScope,
    guards: &mut Vec<CompiledGuard>,
) {
    if let Expr::And(a, b) = &expr.node {
        collect_and_guards_action(ctx, a, registry, local_scope, guards);
        collect_and_guards_action(ctx, b, registry, local_scope, guards);
    } else {
        let compiled = compile_guard_expr_action(ctx, &expr.node, expr, registry, local_scope);
        if !matches!(compiled, CompiledGuard::True) {
            guards.push(compiled);
        }
    }
}

#[cfg(test)]
fn collect_or_guards_action(
    ctx: &EvalCtx,
    expr: &Spanned<Expr>,
    registry: &VarRegistry,
    local_scope: &LocalScope,
    guards: &mut Vec<CompiledGuard>,
) {
    if let Expr::Or(a, b) = &expr.node {
        collect_or_guards_action(ctx, a, registry, local_scope, guards);
        collect_or_guards_action(ctx, b, registry, local_scope, guards);
    } else {
        let compiled = compile_guard_expr_action(ctx, &expr.node, expr, registry, local_scope);
        if !matches!(compiled, CompiledGuard::False) {
            guards.push(compiled);
        }
    }
}

// ============================================================================
// COMPILED ASSIGNMENTS - Pre-compile assignments for fast evaluation
// ============================================================================

/// A compiled assignment extracted during preprocessing.
///
/// Assignments are expressions like `x' = expr` or `UNCHANGED x` that modify
/// state variables. Compiling them avoids AST traversal at runtime.
#[derive(Debug, Clone)]
pub enum CompiledAssignment {
    /// x' = compiled_expr
    Assign {
        var_idx: VarIndex,
        var_name: Arc<str>,
        value: CompiledExpr,
    },

    /// x' = x (UNCHANGED)
    Unchanged {
        var_idx: VarIndex,
        var_name: Arc<str>,
    },

    /// x' \in S (choose from set)
    InSet {
        var_idx: VarIndex,
        var_name: Arc<str>,
        set_expr: CompiledExpr,
    },
}

impl CompiledAssignment {
    /// Get the target variable index for this assignment.
    #[inline]
    pub fn var_idx(&self) -> VarIndex {
        match self {
            CompiledAssignment::Assign { var_idx, .. } => *var_idx,
            CompiledAssignment::Unchanged { var_idx, .. } => *var_idx,
            CompiledAssignment::InSet { var_idx, .. } => *var_idx,
        }
    }

    /// Get the target variable name for this assignment.
    #[inline]
    pub fn var_name(&self) -> &Arc<str> {
        match self {
            CompiledAssignment::Assign { var_name, .. } => var_name,
            CompiledAssignment::Unchanged { var_name, .. } => var_name,
            CompiledAssignment::InSet { var_name, .. } => var_name,
        }
    }
}

/// Extract and compile assignments from a list of conjuncts.
///
/// This function identifies primed variable assignments and compiles their
/// RHS expressions for fast evaluation at runtime.
pub fn extract_compiled_assignments(
    ctx: &EvalCtx,
    conjuncts: &[Spanned<Expr>],
    vars: &[Arc<str>],
    registry: &VarRegistry,
    local_scope: &LocalScope,
) -> Vec<CompiledAssignment> {
    let mut assignments = Vec::new();

    for conj in conjuncts {
        extract_compiled_assignment_expr(
            ctx,
            &conj.node,
            conj,
            vars,
            registry,
            local_scope,
            &mut assignments,
        );
    }

    assignments
}

/// Recursively extract compiled assignments from an expression.
#[allow(clippy::only_used_in_recursion)]
fn extract_compiled_assignment_expr(
    ctx: &EvalCtx,
    expr: &Expr,
    _full_expr: &Spanned<Expr>,
    vars: &[Arc<str>],
    registry: &VarRegistry,
    local_scope: &LocalScope,
    assignments: &mut Vec<CompiledAssignment>,
) {
    match expr {
        // Conjunction: extract from both sides
        Expr::And(a, b) => {
            extract_compiled_assignment_expr(
                ctx,
                &a.node,
                a,
                vars,
                registry,
                local_scope,
                assignments,
            );
            extract_compiled_assignment_expr(
                ctx,
                &b.node,
                b,
                vars,
                registry,
                local_scope,
                assignments,
            );
        }

        // Equality with primed variable: x' = expr
        Expr::Eq(lhs, rhs) => {
            // Check if LHS is primed variable: x'
            if let Expr::Prime(inner) = &lhs.node {
                if let Expr::Ident(name) = &inner.node {
                    if let Some(idx) = registry.get(name) {
                        let var_name = Arc::from(name.as_str());
                        // Fast path: x' = x  ==> UNCHANGED x
                        if matches!(&rhs.node, Expr::Ident(rhs_name) if rhs_name == name) {
                            assignments.push(CompiledAssignment::Unchanged {
                                var_idx: idx,
                                var_name,
                            });
                            return;
                        }
                        // Action RHS may reference primed vars (next-state dependencies), so allow primes.
                        let compiled_rhs =
                            compile_action_value_expr(ctx, rhs, registry, local_scope);
                        assignments.push(CompiledAssignment::Assign {
                            var_idx: idx,
                            var_name,
                            value: compiled_rhs,
                        });
                        return;
                    }
                }
            }
            // Check if RHS is primed variable (symmetric case): expr = x'
            if let Expr::Prime(inner) = &rhs.node {
                if let Expr::Ident(name) = &inner.node {
                    if let Some(idx) = registry.get(name) {
                        let var_name = Arc::from(name.as_str());
                        // Fast path: x = x'  ==> UNCHANGED x
                        if matches!(&lhs.node, Expr::Ident(lhs_name) if lhs_name == name) {
                            assignments.push(CompiledAssignment::Unchanged {
                                var_idx: idx,
                                var_name,
                            });
                            return;
                        }
                        // Action LHS expression may reference primed vars; allow primes.
                        let compiled_lhs =
                            compile_action_value_expr(ctx, lhs, registry, local_scope);
                        assignments.push(CompiledAssignment::Assign {
                            var_idx: idx,
                            var_name,
                            value: compiled_lhs,
                        });
                    }
                }
            }
        }

        // UNCHANGED <<x, y>> or UNCHANGED x
        Expr::Unchanged(inner) => {
            extract_unchanged_compiled(&inner.node, registry, assignments);
        }

        // Membership with primed variable: x' \in S
        Expr::In(lhs, rhs) => {
            if let Expr::Prime(inner) = &lhs.node {
                if let Expr::Ident(name) = &inner.node {
                    if let Some(idx) = registry.get(name) {
                        let var_name = Arc::from(name.as_str());
                        // In-set domains can reference primed vars (e.g. SUBSET(opId' × opId')).
                        let compiled_set =
                            compile_action_value_expr(ctx, rhs, registry, local_scope);
                        assignments.push(CompiledAssignment::InSet {
                            var_idx: idx,
                            var_name,
                            set_expr: compiled_set,
                        });
                    }
                }
            }
        }

        // Zero-param operator reference - recurse into operator body
        Expr::Ident(name) => {
            // Resolve operator name through replacements (e.g., Op <- Replacement)
            let resolved_name = ctx.resolve_op_name(name);
            if let Some(def) = ctx.get_op(resolved_name) {
                if def.params.is_empty() {
                    extract_compiled_assignment_expr(
                        ctx,
                        &def.body.node,
                        &def.body,
                        vars,
                        registry,
                        local_scope,
                        assignments,
                    );
                }
            }
        }

        // Operator application - might be a user-defined action
        Expr::Apply(op_expr, args) => {
            if let Expr::Ident(op_name) = &op_expr.node {
                // Resolve operator name through replacements (e.g., F <- MCF)
                let resolved_name = ctx.resolve_op_name(op_name);
                if let Some(def) = ctx.get_op(resolved_name) {
                    // Use AST-level substitution for arguments that may contain primed variables.
                    // This is critical for patterns like Send(x, 1, y, y') where y' cannot be
                    // evaluated to a value but needs to be substituted into the operator body.
                    let subs: Vec<Substitution> = def
                        .params
                        .iter()
                        .zip(args.iter())
                        .map(|(param, arg)| Substitution {
                            from: param.name.clone(),
                            to: arg.clone(),
                        })
                        .collect();

                    // Apply substitutions to get the expanded body
                    let substituted_body = apply_substitutions(&def.body, &subs);

                    // Also create value bindings for arguments that CAN be evaluated
                    // (needed for expressions that reference non-primed parameters)
                    let mut new_ctx = ctx.clone();
                    for (param, arg) in def.params.iter().zip(args.iter()) {
                        if let Ok(arg_val) = eval(&new_ctx, arg) {
                            new_ctx
                                .env
                                .insert(Arc::from(param.name.node.as_str()), arg_val);
                        }
                    }

                    // Recurse into the substituted body
                    extract_compiled_assignment_expr(
                        &new_ctx,
                        &substituted_body.node,
                        &substituted_body,
                        vars,
                        registry,
                        local_scope,
                        assignments,
                    );
                }
            }
        }

        // Module reference (named INSTANCE): Inst!Op(args)
        //
        // The referenced operator body may contain primed assignments even though the
        // ModuleRef syntax does not, so we inline here to compile assignments.
        Expr::ModuleRef(instance_name, op_name, args) => {
            let Some(instance_info) = ctx.get_instance(instance_name.name()) else {
                return;
            };
            let Some(def) = ctx.get_instance_op(&instance_info.module_name, op_name) else {
                return;
            };
            if def.params.len() != args.len() {
                return;
            }

            // Apply INSTANCE ... WITH substitutions first (module-level substitutions)
            let mut inlined = apply_substitutions(&def.body, &instance_info.substitutions);

            // Always apply parameter substitution (call-by-name) so primed arguments
            // remain evaluable during compiled assignment evaluation.
            if !def.params.is_empty() {
                let subs: Vec<Substitution> = def
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

            // Ensure unqualified operator names resolve to the instanced module's defs.
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

            extract_compiled_assignment_expr(
                &scoped_ctx,
                &inlined.node,
                &inlined,
                vars,
                registry,
                local_scope,
                assignments,
            );
        }

        // LET ... IN body
        Expr::Let(defs, body) => {
            let mut new_ctx = ctx.clone();
            // Issue #70 fix: ALL LET operators must stay in local_ops, not shared.ops or env.
            // For zero-arg operators, pre-evaluating and storing in env breaks dependency tracking
            // because when the value is later accessed during operator caching, no state reads
            // are recorded (the value is just retrieved from env).
            let mut local_ops = new_ctx
                .local_ops
                .as_ref()
                .map(|o| (**o).clone())
                .unwrap_or_default();
            for def in defs {
                // Add ALL defs to local_ops for lazy evaluation with proper dep tracking
                local_ops.insert(def.name.node.clone(), def.clone());
            }
            new_ctx = new_ctx.with_local_ops(local_ops);
            extract_compiled_assignment_expr(
                &new_ctx,
                &body.node,
                body,
                vars,
                registry,
                local_scope,
                assignments,
            );
        }

        // IF-THEN-ELSE: evaluate condition and extract from appropriate branch
        Expr::If(cond, then_branch, else_branch) => {
            match eval(ctx, cond) {
                Ok(Value::Bool(true)) => {
                    extract_compiled_assignment_expr(
                        ctx,
                        &then_branch.node,
                        then_branch,
                        vars,
                        registry,
                        local_scope,
                        assignments,
                    );
                }
                Ok(Value::Bool(false)) => {
                    extract_compiled_assignment_expr(
                        ctx,
                        &else_branch.node,
                        else_branch,
                        vars,
                        registry,
                        local_scope,
                        assignments,
                    );
                }
                _ => {
                    // Condition couldn't be evaluated - skip
                }
            }
        }

        // CASE: evaluate guards and extract from matching branch
        Expr::Case(arms, other) => {
            for arm in arms {
                if let Ok(Value::Bool(true)) = eval(ctx, &arm.guard) {
                    extract_compiled_assignment_expr(
                        ctx,
                        &arm.body.node,
                        &arm.body,
                        vars,
                        registry,
                        local_scope,
                        assignments,
                    );
                    return;
                }
            }
            // No arm matched - try OTHER
            if let Some(other_body) = other {
                extract_compiled_assignment_expr(
                    ctx,
                    &other_body.node,
                    other_body,
                    vars,
                    registry,
                    local_scope,
                    assignments,
                );
            }
        }

        // TRUE and other expressions - skip
        _ => {}
    }
}

/// Extract UNCHANGED variables as compiled assignments.
fn extract_unchanged_compiled(
    expr: &Expr,
    registry: &VarRegistry,
    assignments: &mut Vec<CompiledAssignment>,
) {
    match expr {
        Expr::Ident(name) => {
            if let Some(idx) = registry.get(name) {
                let var_name = Arc::from(name.as_str());
                assignments.push(CompiledAssignment::Unchanged {
                    var_idx: idx,
                    var_name,
                });
            }
        }
        Expr::Tuple(elems) => {
            for elem in elems {
                extract_unchanged_compiled(&elem.node, registry, assignments);
            }
        }
        _ => {}
    }
}

/// Evaluate compiled assignments against an ArrayState.
///
/// Returns the evaluated assignments as (VarIndex, Value) pairs, or multiple
/// values for InSet assignments.
///
/// **Conflict detection**: When the same variable receives multiple assignments from
/// different parts of a conjunction (e.g., `UNCHANGED hr` from one action and `hr' = 5`
/// from another), this function detects the conflict and returns an empty result,
/// indicating no valid successor states exist for this branch.
pub fn evaluate_compiled_assignments(
    ctx: &EvalCtx,
    assignments: &[CompiledAssignment],
    current_array: &ArrayState,
) -> EvalResult<Vec<EvaluatedAssignment>> {
    use std::collections::HashMap;

    let mut result = Vec::with_capacity(assignments.len());

    // Track constraints for conflict detection:
    // - None: unconstrained
    // - Some(None): constrained to InSet (values tracked in pending_insets)
    // - Some(Some(v)): constrained to specific value
    let mut constraints: HashMap<VarIndex, Option<Value>> = HashMap::new();

    // Track pending InSet assignments (may be filtered by later constraints)
    let mut pending_insets: HashMap<VarIndex, (Arc<str>, Vec<Value>)> = HashMap::new();

    // Some assignment RHS expressions may reference other primed variables (next-state values),
    // e.g. `diskPos' = file_pointer'` or `buff' = ... lo' ...`. In those cases we must evaluate
    // with a progressive next-state context (TLC-style), so later assignments can observe the
    // primed values computed earlier in the same conjunction.
    let needs_progressive_next = assignments.iter().any(|a| match a {
        CompiledAssignment::Assign { value, .. } => value.reads_next_state(),
        CompiledAssignment::InSet { set_expr, .. } => set_expr.reads_next_state(),
        CompiledAssignment::Unchanged { .. } => false,
    });

    // Scratch "next-state" array used for progressive evaluation. Starts as the current state
    // (so unconstrained vars are treated as UNCHANGED), and is updated as we evaluate
    // deterministic assignments.
    let mut next_scratch = if needs_progressive_next {
        Some(current_array.clone())
    } else {
        None
    };

    for assignment in assignments {
        match assignment {
            CompiledAssignment::Assign {
                var_idx,
                var_name,
                value,
            } => {
                // Fast path: x' = y' (direct primed variable copy). This must be handled as a
                // correlated choice when y' is nondeterministic (e.g. y' \in S), so we preserve
                // it explicitly instead of eagerly evaluating it against the current state.
                if let CompiledExpr::StateVar {
                    state: StateRef::Next,
                    idx: src_idx,
                } = value
                {
                    result.push(EvaluatedAssignment::CopyFromVar {
                        dest_idx: *var_idx,
                        dest_name: var_name.clone(),
                        src_idx: *src_idx,
                    });

                    // Best-effort: update scratch with the current value of src in the partial
                    // next-state (helps subsequent deterministic expressions that reference dest').
                    if let Some(scratch) = next_scratch.as_mut() {
                        let v = scratch.get(*src_idx).clone();
                        scratch.set(*var_idx, v);
                    }

                    continue;
                }

                let computed = if let Some(scratch) = next_scratch.as_ref() {
                    value.eval_with_arrays(ctx, current_array, scratch)?
                } else {
                    value.eval_with_array(ctx, current_array)?
                };

                // Check for conflicts with existing constraints
                if let Some(existing) = constraints.get(var_idx) {
                    match existing {
                        Some(existing_val) => {
                            if existing_val != &computed {
                                // Conflict - same variable assigned different values
                                return Ok(Vec::new());
                            }
                            // Same value - redundant, skip
                            continue;
                        }
                        None => {
                            // Was InSet, filter to this value
                            if let Some((_, inset_values)) = pending_insets.get_mut(var_idx) {
                                if !inset_values.contains(&computed) {
                                    // Value not in set - conflict
                                    return Ok(Vec::new());
                                }
                                *inset_values = vec![computed.clone()];
                            }
                        }
                    }
                }

                constraints.insert(*var_idx, Some(computed.clone()));

                if let Some(scratch) = next_scratch.as_mut() {
                    scratch.set(*var_idx, computed.clone());
                }

                result.push(EvaluatedAssignment::Assign {
                    var_idx: *var_idx,
                    var_name: var_name.clone(),
                    value: computed,
                });
            }
            CompiledAssignment::Unchanged { var_idx, var_name } => {
                let current_val = current_array.get(*var_idx).clone();

                // Check for conflicts
                if let Some(existing) = constraints.get(var_idx) {
                    match existing {
                        Some(existing_val) => {
                            if existing_val != &current_val {
                                // Conflict - UNCHANGED vs different assigned value
                                return Ok(Vec::new());
                            }
                            // Same value - redundant, skip
                            continue;
                        }
                        None => {
                            // Was InSet, filter to current value
                            if let Some((_, inset_values)) = pending_insets.get_mut(var_idx) {
                                if !inset_values.contains(&current_val) {
                                    // Current value not in set - conflict
                                    return Ok(Vec::new());
                                }
                                *inset_values = vec![current_val];
                            }
                        }
                    }
                }

                constraints.insert(*var_idx, Some(current_array.get(*var_idx).clone()));

                result.push(EvaluatedAssignment::Unchanged {
                    var_idx: *var_idx,
                    var_name: var_name.clone(),
                });
            }
            CompiledAssignment::InSet {
                var_idx,
                var_name,
                set_expr,
            } => {
                let set_val = if let Some(scratch) = next_scratch.as_ref() {
                    set_expr.eval_with_arrays(ctx, current_array, scratch)?
                } else {
                    set_expr.eval_with_array(ctx, current_array)?
                };
                if let Some(set) = set_val.to_ord_set() {
                    let mut values: Vec<Value> = set.into_iter().collect();

                    // Check for conflicts with existing constraints
                    if let Some(existing) = constraints.get(var_idx) {
                        match existing {
                            Some(existing_val) => {
                                // Already constrained to a value, filter InSet
                                if !values.contains(existing_val) {
                                    // Conflict - existing value not in this set
                                    return Ok(Vec::new());
                                }
                                // Existing value is in set - no need to add InSet, already constrained
                                continue;
                            }
                            None => {
                                // Was already InSet, intersect
                                if let Some((_, prev_values)) = pending_insets.get(var_idx) {
                                    values.retain(|v| prev_values.contains(v));
                                    if values.is_empty() {
                                        // Empty intersection - conflict
                                        return Ok(Vec::new());
                                    }
                                }
                            }
                        }
                    }

                    // Preserve empty domains so successor construction correctly yields no successors.
                    if !values.is_empty() {
                        constraints.insert(*var_idx, None);
                        pending_insets.insert(*var_idx, (var_name.clone(), values.clone()));
                        // Don't add to result yet - finalize InSets at the end
                    } else {
                        // Empty InSet - no valid successors for this var
                        return Ok(Vec::new());
                    }
                }
            }
        }
    }

    // Finalize pending InSets - add them to result
    for (var_idx, (var_name, values)) in pending_insets {
        if values.len() == 1 {
            // Single value - convert to Assign
            let value = values.into_iter().next().unwrap();
            result.push(EvaluatedAssignment::Assign {
                var_idx,
                var_name,
                value,
            });
        } else if !values.is_empty() {
            result.push(EvaluatedAssignment::InSet {
                var_idx,
                var_name,
                values,
            });
        }
    }

    Ok(result)
}

/// An evaluated assignment ready for building successor states.
#[derive(Debug, Clone)]
pub enum EvaluatedAssignment {
    /// x' = value
    Assign {
        var_idx: VarIndex,
        var_name: Arc<str>,
        value: Value,
    },
    /// x' = x (copy from current state)
    Unchanged {
        var_idx: VarIndex,
        var_name: Arc<str>,
    },
    /// x' \in values (multiple possible values)
    InSet {
        var_idx: VarIndex,
        var_name: Arc<str>,
        values: Vec<Value>,
    },

    /// x' = y' (copy from another primed variable chosen in the same successor).
    ///
    /// This preserves correlation when `y'` is nondeterministic (e.g. `y' \in S`).
    CopyFromVar {
        dest_idx: VarIndex,
        dest_name: Arc<str>,
        src_idx: VarIndex,
    },
}

#[derive(Clone, Copy)]
enum AssignmentChoices<'a> {
    Single(&'a Value),
    Multi(&'a [Value]),
}

/// Count the number of combinations for the given assignment choices.
#[inline]
fn count_combinations(choices: &[Option<AssignmentChoices<'_>>]) -> usize {
    let mut count = 1usize;
    for choice in choices {
        if let Some(AssignmentChoices::Multi(values)) = choice {
            count = count.saturating_mul(values.len());
        }
        // Single or None: multiplier of 1
    }
    count
}

/// Build successor ArrayStates from evaluated assignments.
pub fn build_successor_array_states(
    current_array: &ArrayState,
    vars: &[Arc<str>],
    assignments: &[EvaluatedAssignment],
    registry: &VarRegistry,
) -> Vec<ArrayState> {
    let num_vars = vars.len();
    debug_assert_eq!(registry.len(), num_vars);
    for (i, var) in vars.iter().enumerate() {
        debug_assert_eq!(registry.name(VarIndex(i as u16)), var.as_ref());
    }

    // Build lookup map: var_idx -> value choices.
    //
    // NOTE: We treat explicit UNCHANGED assignments as no-ops here; the default
    // behavior for unassigned variables is to copy from the current state.
    let mut assignment_map: Vec<Option<AssignmentChoices<'_>>> = vec![None; num_vars];
    let mut copy_pairs: Vec<(VarIndex, VarIndex)> = Vec::new();
    for assignment in assignments {
        match assignment {
            EvaluatedAssignment::Assign { var_idx, value, .. } => {
                assignment_map[var_idx.as_usize()] = Some(AssignmentChoices::Single(value));
            }
            EvaluatedAssignment::Unchanged { .. } => {}
            EvaluatedAssignment::InSet {
                var_idx, values, ..
            } => {
                assignment_map[var_idx.as_usize()] = Some(AssignmentChoices::Multi(values));
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
        .any(|c| matches!(c, Some(AssignmentChoices::Multi(values)) if values.is_empty()))
    {
        return Vec::new();
    }

    // Heuristic: incremental fingerprint updates require hashing both the old and new value
    // for each assigned var. If most vars are assigned, it's cheaper to recompute the
    // fingerprint once per successor at the leaf.
    let assigned_vars = assignment_map.iter().filter(|c| c.is_some()).count();
    let use_incremental_fp = assigned_vars.saturating_mul(2) < num_vars;

    // Pre-allocate results vector to avoid reallocations during enumeration.
    let num_combinations = count_combinations(&assignment_map);
    let mut results = Vec::with_capacity(num_combinations);
    // Start from the current ArrayState so unassigned variables are unchanged without
    // per-var copying or redundant fingerprint work.
    let mut array_state = current_array.clone();
    if use_incremental_fp {
        // Ensure the combined-xor cache exists so we can update fingerprints incrementally.
        let _ = array_state.fingerprint(registry);
    }
    enumerate_combinations_array_from_choices(
        &assignment_map,
        0,
        &mut array_state,
        registry,
        &mut results,
        use_incremental_fp,
    );

    // Apply correlated primed-variable copies (x' = y') after choice enumeration so the copy
    // observes the chosen value of `y'` in each successor.
    if !copy_pairs.is_empty() {
        // Support chains like a' = b' /\ b' = c' by applying up to N passes.
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

/// Build successor DiffSuccessors from evaluated assignments.
///
/// This avoids cloning full ArrayStates during enumeration. Only the changed values are cloned.
/// The successor fingerprint is computed incrementally from the base state's cached combined-xor.
pub fn build_successor_diffs(
    current_array: &ArrayState,
    vars: &[Arc<str>],
    assignments: &[EvaluatedAssignment],
    registry: &VarRegistry,
) -> Vec<DiffSuccessor> {
    let num_vars = vars.len();
    debug_assert_eq!(registry.len(), num_vars);
    for (i, var) in vars.iter().enumerate() {
        debug_assert_eq!(registry.name(VarIndex(i as u16)), var.as_ref());
    }

    // Build lookup map: var_idx -> value choices.
    //
    // NOTE: We treat explicit UNCHANGED assignments as no-ops here; the default
    // behavior for unassigned variables is to copy from the current state.
    let mut assignment_map: Vec<Option<AssignmentChoices<'_>>> = vec![None; num_vars];
    let mut copy_pairs: Vec<(VarIndex, VarIndex)> = Vec::new();
    for assignment in assignments {
        match assignment {
            EvaluatedAssignment::Assign { var_idx, value, .. } => {
                assignment_map[var_idx.as_usize()] = Some(AssignmentChoices::Single(value));
            }
            EvaluatedAssignment::Unchanged { .. } => {}
            EvaluatedAssignment::InSet {
                var_idx, values, ..
            } => {
                assignment_map[var_idx.as_usize()] = Some(AssignmentChoices::Multi(values));
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
        .any(|c| matches!(c, Some(AssignmentChoices::Multi(values)) if values.is_empty()))
    {
        return Vec::new();
    }

    let num_combinations = count_combinations(&assignment_map);
    let mut results = Vec::with_capacity(num_combinations);

    let mut changes: Vec<(VarIndex, &Value)> = Vec::new();
    enumerate_combinations_diff_from_choices(
        &assignment_map,
        0,
        current_array,
        registry,
        &mut changes,
        &mut results,
    );

    if !copy_pairs.is_empty() {
        // Post-process diffs to apply correlated primed-variable copies (x' = y') and recompute
        // fingerprints. This keeps the main combination enumerator fast and centralized.
        for diff in results.iter_mut() {
            for _ in 0..copy_pairs.len() {
                for (dest, src) in copy_pairs.iter().copied() {
                    let src_val = diff
                        .changes
                        .iter()
                        .find(|(idx, _)| *idx == src)
                        .map(|(_, v)| v.clone())
                        .unwrap_or_else(|| current_array.get(src).clone());

                    if let Some((_, v)) = diff.changes.iter_mut().find(|(idx, _)| *idx == dest) {
                        *v = src_val;
                    } else {
                        diff.changes.push((dest, src_val));
                    }
                }
            }
            diff.fingerprint = compute_diff_fingerprint(current_array, &diff.changes, registry);
        }
    }

    results
}

/// Build a single successor DiffSuccessor from evaluated assignments by taking ownership.
///
/// This is an optimized fast path for the common case where all assignments are deterministic
/// (no `InSet` nondeterminism) and there's exactly one possible successor state.
/// By taking ownership of the assignments, we avoid cloning the values.
///
/// Returns `None` if there are any `InSet` assignments (use `build_successor_diffs` instead).
pub fn build_successor_diff_owned(
    current_array: &ArrayState,
    assignments: Vec<EvaluatedAssignment>,
    registry: &VarRegistry,
) -> Option<DiffSuccessor> {
    // Collect changes, moving values out of assignments.
    // Use SmallVec to avoid heap allocation for small change sets.
    let mut changes: DiffChanges = DiffChanges::new();

    for assignment in assignments {
        match assignment {
            EvaluatedAssignment::Assign { var_idx, value, .. } => {
                changes.push((var_idx, value));
            }
            EvaluatedAssignment::Unchanged { .. } => {
                // No change needed
            }
            EvaluatedAssignment::InSet { .. } => {
                // Nondeterministic - can't use this fast path
                return None;
            }
            EvaluatedAssignment::CopyFromVar { .. } => {
                // Correlated copy requires observing the chosen source value in the successor.
                // Handle via the general diff builder.
                return None;
            }
        }
    }

    // Compute fingerprint and build diff
    let fp = compute_diff_fingerprint(current_array, &changes, registry);
    Some(DiffSuccessor::from_smallvec(fp, changes))
}

/// Build successor DiffSuccessors from evaluated assignments, keeping only successors that pass `filter`.
///
/// `filter` is evaluated against a mutable next-state ArrayState built by applying assignments.
pub fn build_successor_diffs_filtered<F>(
    current_array: &ArrayState,
    vars: &[Arc<str>],
    assignments: &[EvaluatedAssignment],
    registry: &VarRegistry,
    mut filter: F,
) -> Vec<DiffSuccessor>
where
    F: FnMut(&ArrayState) -> bool,
{
    let num_vars = vars.len();
    debug_assert_eq!(registry.len(), num_vars);
    for (i, var) in vars.iter().enumerate() {
        debug_assert_eq!(registry.name(VarIndex(i as u16)), var.as_ref());
    }

    // Build lookup map: var_idx -> value choices.
    //
    // NOTE: We treat explicit UNCHANGED assignments as no-ops here; the default
    // behavior for unassigned variables is to copy from the current state.
    let mut assignment_map: Vec<Option<AssignmentChoices<'_>>> = vec![None; num_vars];
    let mut copy_pairs: Vec<(VarIndex, VarIndex)> = Vec::new();
    for assignment in assignments {
        match assignment {
            EvaluatedAssignment::Assign { var_idx, value, .. } => {
                assignment_map[var_idx.as_usize()] = Some(AssignmentChoices::Single(value));
            }
            EvaluatedAssignment::Unchanged { .. } => {}
            EvaluatedAssignment::InSet {
                var_idx, values, ..
            } => {
                assignment_map[var_idx.as_usize()] = Some(AssignmentChoices::Multi(values));
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
        .any(|c| matches!(c, Some(AssignmentChoices::Multi(values)) if values.is_empty()))
    {
        return Vec::new();
    }

    // Enumerate all combinations of assignments and emit diffs.
    let num_combinations = count_combinations(&assignment_map);
    let mut results = Vec::with_capacity(num_combinations);

    let mut next_array = current_array.clone();

    let mut changes: Vec<(VarIndex, &Value)> = Vec::new();
    enumerate_combinations_diff_from_choices_filtered(
        &assignment_map,
        0,
        current_array,
        &mut next_array,
        registry,
        &mut changes,
        &mut results,
        &copy_pairs,
        &mut filter,
    );

    if !copy_pairs.is_empty() {
        for diff in results.iter_mut() {
            for _ in 0..copy_pairs.len() {
                for (dest, src) in copy_pairs.iter().copied() {
                    let src_val = diff
                        .changes
                        .iter()
                        .find(|(idx, _)| *idx == src)
                        .map(|(_, v)| v.clone())
                        .unwrap_or_else(|| current_array.get(src).clone());
                    if let Some((_, v)) = diff.changes.iter_mut().find(|(idx, _)| *idx == dest) {
                        *v = src_val;
                    } else {
                        diff.changes.push((dest, src_val));
                    }
                }
            }
            diff.fingerprint = compute_diff_fingerprint(current_array, &diff.changes, registry);
        }
    }

    results
}

fn enumerate_combinations_diff_from_choices<'a>(
    choices: &[Option<AssignmentChoices<'a>>],
    idx: usize,
    base: &ArrayState,
    registry: &VarRegistry,
    changes: &mut Vec<(VarIndex, &'a Value)>,
    results: &mut Vec<DiffSuccessor>,
) {
    if idx >= choices.len() {
        // Clone only changed values into the diff (avoid storing explicit no-ops).
        // Use SmallVec to avoid heap allocation for small change sets.
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
        Some(AssignmentChoices::Single(v)) => {
            changes.push((var_idx, v));
            enumerate_combinations_diff_from_choices(
                choices,
                idx + 1,
                base,
                registry,
                changes,
                results,
            );
            changes.pop();
        }
        Some(AssignmentChoices::Multi(values)) => {
            for v in *values {
                changes.push((var_idx, v));
                enumerate_combinations_diff_from_choices(
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
        None => enumerate_combinations_diff_from_choices(
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
fn enumerate_combinations_diff_from_choices_filtered<'a, F>(
    choices: &[Option<AssignmentChoices<'a>>],
    idx: usize,
    base: &ArrayState,
    next: &mut ArrayState,
    registry: &VarRegistry,
    changes: &mut Vec<(VarIndex, &'a Value)>,
    results: &mut Vec<DiffSuccessor>,
    copy_pairs: &[(VarIndex, VarIndex)],
    filter: &mut F,
) where
    F: FnMut(&ArrayState) -> bool,
{
    if idx >= choices.len() {
        // Apply correlated copies to the candidate next-state before filtering.
        // We restore values afterwards because `next` is reused across recursion.
        let mut saved: Vec<(VarIndex, Value)> = Vec::new();
        if !copy_pairs.is_empty() {
            saved.reserve(copy_pairs.len());
            for (dest, _src) in copy_pairs.iter().copied() {
                saved.push((dest, next.get(dest).clone()));
            }
            for _ in 0..copy_pairs.len() {
                for (dest, src) in copy_pairs.iter().copied() {
                    let v = next.get(src).clone();
                    next.set(dest, v);
                }
            }
        }

        let keep = filter(next);

        if !saved.is_empty() {
            for (dest, old) in saved {
                next.set(dest, old);
            }
        }

        if !keep {
            return;
        }

        // Clone only changed values into the diff (avoid storing explicit no-ops).
        // Use SmallVec to avoid heap allocation for small change sets.
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
        Some(AssignmentChoices::Single(v)) => {
            next.set(var_idx, (*v).clone());
            changes.push((var_idx, v));
            enumerate_combinations_diff_from_choices_filtered(
                choices,
                idx + 1,
                base,
                next,
                registry,
                changes,
                results,
                copy_pairs,
                filter,
            );
            changes.pop();
        }
        Some(AssignmentChoices::Multi(values)) => {
            for v in *values {
                next.set(var_idx, v.clone());
                changes.push((var_idx, v));
                enumerate_combinations_diff_from_choices_filtered(
                    choices,
                    idx + 1,
                    base,
                    next,
                    registry,
                    changes,
                    results,
                    copy_pairs,
                    filter,
                );
                changes.pop();
            }
        }
        None => {
            enumerate_combinations_diff_from_choices_filtered(
                choices,
                idx + 1,
                base,
                next,
                registry,
                changes,
                results,
                copy_pairs,
                filter,
            );
        }
    }
}

/// Enumerate all combinations of assignments into ArrayStates.
fn enumerate_combinations_array_from_choices<'a>(
    choices: &[Option<AssignmentChoices<'a>>],
    idx: usize,
    current: &mut ArrayState,
    registry: &VarRegistry,
    results: &mut Vec<ArrayState>,
    use_incremental_fp: bool,
) {
    if idx >= choices.len() {
        if !use_incremental_fp {
            let _ = current.fingerprint(registry);
        }
        results.push(current.clone());
        return;
    }

    let var_idx = VarIndex(idx as u16);

    match &choices[idx] {
        Some(AssignmentChoices::Single(v)) => {
            if use_incremental_fp {
                current.set_with_registry(var_idx, (*v).clone(), registry);
            } else {
                current.set(var_idx, (*v).clone());
            }
            enumerate_combinations_array_from_choices(
                choices,
                idx + 1,
                current,
                registry,
                results,
                use_incremental_fp,
            );
        }
        Some(AssignmentChoices::Multi(values)) => {
            for v in *values {
                if use_incremental_fp {
                    current.set_with_registry(var_idx, v.clone(), registry);
                } else {
                    current.set(var_idx, v.clone());
                }
                enumerate_combinations_array_from_choices(
                    choices,
                    idx + 1,
                    current,
                    registry,
                    results,
                    use_incremental_fp,
                );
            }
        }
        None => {
            enumerate_combinations_array_from_choices(
                choices,
                idx + 1,
                current,
                registry,
                results,
                use_incremental_fp,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;
    use tla_core::ast::{OpParam, OperatorDef};

    #[test]
    fn test_compiled_guard_true_false() {
        let guard_true = CompiledGuard::True;
        let guard_false = CompiledGuard::False;

        let mut ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        assert!(guard_true.eval_with_array(&mut ctx, &array_state).unwrap());
        assert!(!guard_false.eval_with_array(&mut ctx, &array_state).unwrap());
    }

    #[test]
    fn test_compiled_guard_eq_const() {
        let guard = CompiledGuard::EqConst {
            expr: CompiledExpr::StateVar {
                state: StateRef::Current,
                idx: VarIndex(0),
            },
            expected: Value::SmallInt(42),
        };

        let mut ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![Value::SmallInt(42)]);

        assert!(guard.eval_with_array(&mut ctx, &array_state).unwrap());

        let array_state_different = ArrayState::from_values(vec![Value::SmallInt(43)]);
        assert!(!guard
            .eval_with_array(&mut ctx, &array_state_different)
            .unwrap());
    }

    #[test]
    fn test_compiled_guard_int_cmp() {
        let guard = CompiledGuard::IntCmp {
            left: CompiledExpr::StateVar {
                state: StateRef::Current,
                idx: VarIndex(0),
            },
            op: CmpOp::Lt,
            right: CompiledExpr::Const(Value::SmallInt(10)),
        };

        let mut ctx = EvalCtx::new();

        let state_5 = ArrayState::from_values(vec![Value::SmallInt(5)]);
        assert!(guard.eval_with_array(&mut ctx, &state_5).unwrap());

        let state_15 = ArrayState::from_values(vec![Value::SmallInt(15)]);
        assert!(!guard.eval_with_array(&mut ctx, &state_15).unwrap());
    }

    #[test]
    fn test_compile_prime_guard_next_state_func_apply() {
        let span = tla_core::Span::dummy();

        let prime_rcvd = Spanned::new(
            Expr::Prime(Box::new(Spanned::new(
                Expr::Ident("rcvd".to_string()),
                span,
            ))),
            span,
        );
        let self_ident = Spanned::new(Expr::Ident("self".to_string()), span);
        let lhs = Spanned::new(
            Expr::FuncApply(Box::new(prime_rcvd), Box::new(self_ident)),
            span,
        );
        let rhs = Spanned::new(Expr::SetEnum(vec![]), span);
        let expr = Spanned::new(Expr::Neq(Box::new(lhs), Box::new(rhs)), span);

        let mut registry = VarRegistry::new();
        registry.register("rcvd");

        let mut ctx = EvalCtx::new();
        ctx.push_binding(Arc::from("self"), Value::SmallInt(1));

        // Create a local scope with "self" at depth 0
        let local_scope = LocalScope::new().with_var("self");
        let compiled = compile_prime_guard(&ctx, &expr, &registry, &local_scope)
            .expect("prime guard should compile");

        let current = ArrayState::from_values(vec![Value::empty_set()]);

        // next.rcvd[self] == {} -> guard false
        let key = Value::SmallInt(1);
        let mut domain = im::OrdSet::new();
        domain.insert(key.clone());
        let mut mapping = im::OrdMap::new();
        mapping.insert(key.clone(), Value::empty_set());
        let next_false = ArrayState::from_values(vec![Value::Func(crate::value::FuncValue::new(
            domain.clone(),
            mapping,
        ))]);
        assert!(!compiled
            .eval_with_arrays(&mut ctx, &current, &next_false)
            .unwrap());

        // next.rcvd[self] == {2} -> guard true
        let mut mapping2 = im::OrdMap::new();
        mapping2.insert(key, Value::set([Value::SmallInt(2)]));
        let next_true = ArrayState::from_values(vec![Value::Func(crate::value::FuncValue::new(
            domain, mapping2,
        ))]);
        assert!(compiled
            .eval_with_arrays(&mut ctx, &current, &next_true)
            .unwrap());
    }

    #[test]
    fn test_compiled_expr_binop_int_mul() {
        let expr = CompiledExpr::BinOp {
            left: Box::new(CompiledExpr::Const(Value::SmallInt(6))),
            op: BinOp::IntMul,
            right: Box::new(CompiledExpr::Const(Value::SmallInt(7))),
        };

        let ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        let result = expr.eval_with_array(&ctx, &array_state).unwrap();
        assert_eq!(result, Value::SmallInt(42));
    }

    #[test]
    fn test_compiled_expr_binop_int_div() {
        let expr = CompiledExpr::BinOp {
            left: Box::new(CompiledExpr::Const(Value::SmallInt(17))),
            op: BinOp::IntDiv,
            right: Box::new(CompiledExpr::Const(Value::SmallInt(5))),
        };

        let ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        // TLA+ div: 17 \div 5 = 3 (floor division)
        let result = expr.eval_with_array(&ctx, &array_state).unwrap();
        assert_eq!(result, Value::SmallInt(3));
    }

    #[test]
    fn test_compiled_expr_binop_int_div_negative() {
        let expr = CompiledExpr::BinOp {
            left: Box::new(CompiledExpr::Const(Value::SmallInt(-17))),
            op: BinOp::IntDiv,
            right: Box::new(CompiledExpr::Const(Value::SmallInt(5))),
        };

        let ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        // TLA+ div: -17 \div 5 = -4 (floor division toward negative infinity)
        let result = expr.eval_with_array(&ctx, &array_state).unwrap();
        assert_eq!(result, Value::SmallInt(-4));
    }

    #[test]
    fn test_compiled_expr_binop_int_mod() {
        let expr = CompiledExpr::BinOp {
            left: Box::new(CompiledExpr::Const(Value::SmallInt(17))),
            op: BinOp::IntMod,
            right: Box::new(CompiledExpr::Const(Value::SmallInt(5))),
        };

        let ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        // 17 % 5 = 2
        let result = expr.eval_with_array(&ctx, &array_state).unwrap();
        assert_eq!(result, Value::SmallInt(2));
    }

    #[test]
    fn test_compiled_expr_binop_int_mod_negative() {
        let expr = CompiledExpr::BinOp {
            left: Box::new(CompiledExpr::Const(Value::SmallInt(-17))),
            op: BinOp::IntMod,
            right: Box::new(CompiledExpr::Const(Value::SmallInt(5))),
        };

        let ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        // TLA+ mod: -17 % 5 = 3 (always non-negative when divisor is positive)
        let result = expr.eval_with_array(&ctx, &array_state).unwrap();
        assert_eq!(result, Value::SmallInt(3));
    }

    #[test]
    fn test_compiled_expr_binop_set_intersect() {
        let set_a = Value::set([Value::SmallInt(1), Value::SmallInt(2), Value::SmallInt(3)]);
        let set_b = Value::set([Value::SmallInt(2), Value::SmallInt(3), Value::SmallInt(4)]);

        let expr = CompiledExpr::BinOp {
            left: Box::new(CompiledExpr::Const(set_a)),
            op: BinOp::SetIntersect,
            right: Box::new(CompiledExpr::Const(set_b)),
        };

        let ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        let result = expr.eval_with_array(&ctx, &array_state).unwrap();
        let expected = Value::set([Value::SmallInt(2), Value::SmallInt(3)]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_compile_value_expr_inlines_zero_arg_operator() {
        let span = tla_core::Span::dummy();

        // Op == x + 1
        let op_body = Spanned::new(
            Expr::Add(
                Box::new(Spanned::new(Expr::Ident("x".to_string()), span)),
                Box::new(Spanned::new(Expr::Int(BigInt::from(1)), span)),
            ),
            span,
        );

        let mut ctx = EvalCtx::new();
        ctx.define_op(
            "Op".to_string(),
            OperatorDef {
                name: Spanned::new("Op".to_string(), span),
                params: vec![],
                body: op_body,
                local: false,
            },
        );

        let mut registry = VarRegistry::new();
        registry.register("x");
        let local_scope = LocalScope::new();

        let expr = Spanned::new(Expr::Ident("Op".to_string()), span);
        let compiled = compile_value_expr(&ctx, &expr, &registry, &local_scope);
        assert!(!expr_has_fallback(&compiled));

        let array_state = ArrayState::from_values(vec![Value::SmallInt(41)]);
        let result = compiled.eval_with_array(&ctx, &array_state).unwrap();
        assert_eq!(result, Value::SmallInt(42));
    }

    #[test]
    fn test_compile_value_expr_inlines_operator_application() {
        let span = tla_core::Span::dummy();

        // AddOne(a) == a + 1
        let op_body = Spanned::new(
            Expr::Add(
                Box::new(Spanned::new(Expr::Ident("a".to_string()), span)),
                Box::new(Spanned::new(Expr::Int(BigInt::from(1)), span)),
            ),
            span,
        );

        let mut ctx = EvalCtx::new();
        ctx.define_op(
            "AddOne".to_string(),
            OperatorDef {
                name: Spanned::new("AddOne".to_string(), span),
                params: vec![OpParam {
                    name: Spanned::new("a".to_string(), span),
                    arity: 0,
                }],
                body: op_body,
                local: false,
            },
        );

        let mut registry = VarRegistry::new();
        registry.register("x");
        let local_scope = LocalScope::new();

        let expr = Spanned::new(
            Expr::Apply(
                Box::new(Spanned::new(Expr::Ident("AddOne".to_string()), span)),
                vec![Spanned::new(Expr::Ident("x".to_string()), span)],
            ),
            span,
        );

        let compiled = compile_value_expr(&ctx, &expr, &registry, &local_scope);
        assert!(!expr_has_fallback(&compiled));

        let array_state = ArrayState::from_values(vec![Value::SmallInt(41)]);
        let result = compiled.eval_with_array(&ctx, &array_state).unwrap();
        assert_eq!(result, Value::SmallInt(42));
    }

    #[test]
    fn test_compile_value_expr_local_var_shadows_state_var() {
        let span = tla_core::Span::dummy();

        let mut registry = VarRegistry::new();
        registry.register("x");

        let mut ctx = EvalCtx::new();
        ctx.push_binding(Arc::from("x"), Value::SmallInt(99));

        let local_scope = LocalScope::new().with_var("x");

        let expr = Spanned::new(Expr::Ident("x".to_string()), span);
        let compiled = compile_value_expr(&ctx, &expr, &registry, &local_scope);

        match &compiled {
            CompiledExpr::LocalVar { depth, .. } => assert_eq!(*depth, 0),
            other => panic!("expected LocalVar, got {other:?}"),
        }

        let array_state = ArrayState::from_values(vec![Value::SmallInt(41)]);
        let result = compiled.eval_with_array(&ctx, &array_state).unwrap();
        assert_eq!(result, Value::SmallInt(99));
    }

    #[test]
    fn test_compiled_guard_and() {
        // Test AND with two true guards
        let guard_and_true = CompiledGuard::And(vec![CompiledGuard::True, CompiledGuard::True]);

        let mut ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        assert!(guard_and_true.eval_with_array(&mut ctx, &array_state).unwrap());

        // Test AND with one false guard (short-circuit)
        let guard_and_false = CompiledGuard::And(vec![CompiledGuard::False, CompiledGuard::True]);
        assert!(!guard_and_false.eval_with_array(&mut ctx, &array_state).unwrap());

        // Test AND with false second (ensures both evaluated when first is true)
        let guard_and_false2 = CompiledGuard::And(vec![CompiledGuard::True, CompiledGuard::False]);
        assert!(!guard_and_false2.eval_with_array(&mut ctx, &array_state).unwrap());

        // Test empty AND (should be vacuously true)
        let guard_and_empty = CompiledGuard::And(vec![]);
        assert!(guard_and_empty.eval_with_array(&mut ctx, &array_state).unwrap());
    }

    #[test]
    fn test_compiled_guard_or() {
        // Test OR with two false guards
        let guard_or_false = CompiledGuard::Or(vec![CompiledGuard::False, CompiledGuard::False]);

        let mut ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        assert!(!guard_or_false.eval_with_array(&mut ctx, &array_state).unwrap());

        // Test OR with one true guard (short-circuit)
        let guard_or_true = CompiledGuard::Or(vec![CompiledGuard::True, CompiledGuard::False]);
        assert!(guard_or_true.eval_with_array(&mut ctx, &array_state).unwrap());

        // Test OR with true second (ensures both evaluated when first is false)
        let guard_or_true2 = CompiledGuard::Or(vec![CompiledGuard::False, CompiledGuard::True]);
        assert!(guard_or_true2.eval_with_array(&mut ctx, &array_state).unwrap());

        // Test empty OR (should be vacuously false)
        let guard_or_empty = CompiledGuard::Or(vec![]);
        assert!(!guard_or_empty.eval_with_array(&mut ctx, &array_state).unwrap());
    }

    #[test]
    fn test_compiled_guard_in() {
        let set = Value::set([Value::SmallInt(1), Value::SmallInt(2), Value::SmallInt(3)]);

        let guard_in_true = CompiledGuard::In {
            elem: CompiledExpr::Const(Value::SmallInt(2)),
            set: CompiledExpr::Const(set.clone()),
        };

        let mut ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        assert!(guard_in_true.eval_with_array(&mut ctx, &array_state).unwrap());

        let guard_in_false = CompiledGuard::In {
            elem: CompiledExpr::Const(Value::SmallInt(42)),
            set: CompiledExpr::Const(set),
        };
        assert!(!guard_in_false.eval_with_array(&mut ctx, &array_state).unwrap());
    }

    #[test]
    fn test_compiled_guard_neq_const() {
        let guard = CompiledGuard::NeqConst {
            expr: CompiledExpr::StateVar {
                state: StateRef::Current,
                idx: VarIndex(0),
            },
            expected: Value::SmallInt(42),
        };

        let mut ctx = EvalCtx::new();

        // Value is different from expected -> true
        let array_state_different = ArrayState::from_values(vec![Value::SmallInt(99)]);
        assert!(guard.eval_with_array(&mut ctx, &array_state_different).unwrap());

        // Value equals expected -> false
        let array_state_same = ArrayState::from_values(vec![Value::SmallInt(42)]);
        assert!(!guard.eval_with_array(&mut ctx, &array_state_same).unwrap());
    }

    #[test]
    fn test_compiled_guard_not() {
        let guard_not_true = CompiledGuard::Not(Box::new(CompiledGuard::True));
        let guard_not_false = CompiledGuard::Not(Box::new(CompiledGuard::False));

        let mut ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        assert!(!guard_not_true.eval_with_array(&mut ctx, &array_state).unwrap());
        assert!(guard_not_false.eval_with_array(&mut ctx, &array_state).unwrap());
    }

    #[test]
    fn test_compiled_guard_subseteq() {
        let set_small = Value::set([Value::SmallInt(1), Value::SmallInt(2)]);
        let set_large = Value::set([Value::SmallInt(1), Value::SmallInt(2), Value::SmallInt(3)]);

        // small \subseteq large -> true
        let guard_true = CompiledGuard::Subseteq {
            left: CompiledExpr::Const(set_small.clone()),
            right: CompiledExpr::Const(set_large.clone()),
        };

        let mut ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        assert!(guard_true.eval_with_array(&mut ctx, &array_state).unwrap());

        // large \subseteq small -> false
        let guard_false = CompiledGuard::Subseteq {
            left: CompiledExpr::Const(set_large),
            right: CompiledExpr::Const(set_small),
        };
        assert!(!guard_false.eval_with_array(&mut ctx, &array_state).unwrap());
    }

    #[test]
    fn test_compiled_guard_implies() {
        let mut ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        // True => True -> True
        let tt = CompiledGuard::Implies {
            antecedent: Box::new(CompiledGuard::True),
            consequent: Box::new(CompiledGuard::True),
        };
        assert!(tt.eval_with_array(&mut ctx, &array_state).unwrap());

        // True => False -> False
        let tf = CompiledGuard::Implies {
            antecedent: Box::new(CompiledGuard::True),
            consequent: Box::new(CompiledGuard::False),
        };
        assert!(!tf.eval_with_array(&mut ctx, &array_state).unwrap());

        // False => True -> True (vacuously)
        let ft = CompiledGuard::Implies {
            antecedent: Box::new(CompiledGuard::False),
            consequent: Box::new(CompiledGuard::True),
        };
        assert!(ft.eval_with_array(&mut ctx, &array_state).unwrap());

        // False => False -> True (vacuously)
        let ff = CompiledGuard::Implies {
            antecedent: Box::new(CompiledGuard::False),
            consequent: Box::new(CompiledGuard::False),
        };
        assert!(ff.eval_with_array(&mut ctx, &array_state).unwrap());
    }

    #[test]
    fn test_compiled_expr_set_enum() {
        let expr = CompiledExpr::SetEnum(vec![
            CompiledExpr::Const(Value::SmallInt(1)),
            CompiledExpr::Const(Value::SmallInt(2)),
            CompiledExpr::Const(Value::SmallInt(3)),
        ]);

        let ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        let result = expr.eval_with_array(&ctx, &array_state).unwrap();
        let expected = Value::set([Value::SmallInt(1), Value::SmallInt(2), Value::SmallInt(3)]);
        assert_eq!(result, expected);

        // Test empty set
        let expr_empty = CompiledExpr::SetEnum(vec![]);
        let result_empty = expr_empty.eval_with_array(&ctx, &array_state).unwrap();
        assert_eq!(result_empty, Value::empty_set());
    }

    #[test]
    fn test_compiled_expr_tuple() {
        let expr = CompiledExpr::Tuple(vec![
            CompiledExpr::Const(Value::SmallInt(1)),
            CompiledExpr::Const(Value::SmallInt(2)),
            CompiledExpr::Const(Value::SmallInt(3)),
        ]);

        let ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        let result = expr.eval_with_array(&ctx, &array_state).unwrap();
        let expected = Value::tuple([Value::SmallInt(1), Value::SmallInt(2), Value::SmallInt(3)]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_compiled_expr_not() {
        let ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        let expr_not_true = CompiledExpr::Not(Box::new(CompiledExpr::Const(Value::Bool(true))));
        let expr_not_false = CompiledExpr::Not(Box::new(CompiledExpr::Const(Value::Bool(false))));

        assert_eq!(
            expr_not_true.eval_with_array(&ctx, &array_state).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            expr_not_false.eval_with_array(&ctx, &array_state).unwrap(),
            Value::Bool(true)
        );
    }

    #[test]
    fn test_compiled_expr_if_then_else() {
        let ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        // IF TRUE THEN 1 ELSE 2
        let expr_true = CompiledExpr::IfThenElse {
            condition: Box::new(CompiledExpr::Const(Value::Bool(true))),
            then_branch: Box::new(CompiledExpr::Const(Value::SmallInt(1))),
            else_branch: Box::new(CompiledExpr::Const(Value::SmallInt(2))),
        };
        assert_eq!(
            expr_true.eval_with_array(&ctx, &array_state).unwrap(),
            Value::SmallInt(1)
        );

        // IF FALSE THEN 1 ELSE 2
        let expr_false = CompiledExpr::IfThenElse {
            condition: Box::new(CompiledExpr::Const(Value::Bool(false))),
            then_branch: Box::new(CompiledExpr::Const(Value::SmallInt(1))),
            else_branch: Box::new(CompiledExpr::Const(Value::SmallInt(2))),
        };
        assert_eq!(
            expr_false.eval_with_array(&ctx, &array_state).unwrap(),
            Value::SmallInt(2)
        );
    }

    #[test]
    fn test_compiled_expr_binop_set_union() {
        let set_a = Value::set([Value::SmallInt(1), Value::SmallInt(2)]);
        let set_b = Value::set([Value::SmallInt(2), Value::SmallInt(3)]);

        let expr = CompiledExpr::BinOp {
            left: Box::new(CompiledExpr::Const(set_a)),
            op: BinOp::SetUnion,
            right: Box::new(CompiledExpr::Const(set_b)),
        };

        let ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        let result = expr.eval_with_array(&ctx, &array_state).unwrap();
        let expected = Value::set([Value::SmallInt(1), Value::SmallInt(2), Value::SmallInt(3)]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_compiled_expr_binop_set_diff() {
        let set_a = Value::set([Value::SmallInt(1), Value::SmallInt(2), Value::SmallInt(3)]);
        let set_b = Value::set([Value::SmallInt(2)]);

        let expr = CompiledExpr::BinOp {
            left: Box::new(CompiledExpr::Const(set_a)),
            op: BinOp::SetDiff,
            right: Box::new(CompiledExpr::Const(set_b)),
        };

        let ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        let result = expr.eval_with_array(&ctx, &array_state).unwrap();
        let expected = Value::set([Value::SmallInt(1), Value::SmallInt(3)]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_compiled_expr_bool_eq() {
        let ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        // 42 = 42 -> true
        let expr_eq = CompiledExpr::BoolEq {
            left: Box::new(CompiledExpr::Const(Value::SmallInt(42))),
            right: Box::new(CompiledExpr::Const(Value::SmallInt(42))),
        };
        assert_eq!(
            expr_eq.eval_with_array(&ctx, &array_state).unwrap(),
            Value::Bool(true)
        );

        // 42 = 43 -> false
        let expr_neq = CompiledExpr::BoolEq {
            left: Box::new(CompiledExpr::Const(Value::SmallInt(42))),
            right: Box::new(CompiledExpr::Const(Value::SmallInt(43))),
        };
        assert_eq!(
            expr_neq.eval_with_array(&ctx, &array_state).unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn test_compiled_expr_set_cardinality() {
        let set = Value::set([Value::SmallInt(1), Value::SmallInt(2), Value::SmallInt(3)]);

        let expr = CompiledExpr::SetCardinality(Box::new(CompiledExpr::Const(set)));

        let ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        let result = expr.eval_with_array(&ctx, &array_state).unwrap();
        assert_eq!(result, Value::SmallInt(3));

        // Empty set cardinality
        let expr_empty = CompiledExpr::SetCardinality(Box::new(CompiledExpr::Const(Value::empty_set())));
        let result_empty = expr_empty.eval_with_array(&ctx, &array_state).unwrap();
        assert_eq!(result_empty, Value::SmallInt(0));
    }

    #[test]
    fn test_compiled_guard_int_cmp_all_ops() {
        let mut ctx = EvalCtx::new();

        // Test all comparison operators
        let ops = [
            (CmpOp::Lt, 5, 10, true),
            (CmpOp::Lt, 10, 5, false),
            (CmpOp::Lt, 5, 5, false),
            (CmpOp::Le, 5, 10, true),
            (CmpOp::Le, 10, 5, false),
            (CmpOp::Le, 5, 5, true),
            (CmpOp::Gt, 10, 5, true),
            (CmpOp::Gt, 5, 10, false),
            (CmpOp::Gt, 5, 5, false),
            (CmpOp::Ge, 10, 5, true),
            (CmpOp::Ge, 5, 10, false),
            (CmpOp::Ge, 5, 5, true),
        ];

        for (op, left, right, expected) in ops {
            let guard = CompiledGuard::IntCmp {
                left: CompiledExpr::Const(Value::SmallInt(left)),
                op,
                right: CompiledExpr::Const(Value::SmallInt(right)),
            };
            let array_state = ArrayState::from_values(vec![]);
            let result = guard.eval_with_array(&mut ctx, &array_state).unwrap();
            assert_eq!(
                result, expected,
                "Failed for {:?}: {} vs {} expected {}",
                op, left, right, expected
            );
        }
    }

    #[test]
    fn test_compiled_guard_forall_all_true() {
        // \A x \in {1, 2, 3} : x > 0
        let guard = CompiledGuard::ForAll {
            var_name: Arc::from("x"),
            domain: CompiledExpr::Const(Value::set([
                Value::SmallInt(1),
                Value::SmallInt(2),
                Value::SmallInt(3),
            ])),
            body: Box::new(CompiledGuard::IntCmp {
                left: CompiledExpr::LocalVar {
                    name: Arc::from("x"),
                    depth: 0,
                },
                op: CmpOp::Gt,
                right: CompiledExpr::Const(Value::SmallInt(0)),
            }),
        };

        let mut ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        // All elements > 0, so result is true
        assert!(guard.eval_with_array(&mut ctx, &array_state).unwrap());
    }

    #[test]
    fn test_compiled_guard_forall_one_false() {
        // \A x \in {1, 2, 3} : x > 1
        let guard = CompiledGuard::ForAll {
            var_name: Arc::from("x"),
            domain: CompiledExpr::Const(Value::set([
                Value::SmallInt(1),
                Value::SmallInt(2),
                Value::SmallInt(3),
            ])),
            body: Box::new(CompiledGuard::IntCmp {
                left: CompiledExpr::LocalVar {
                    name: Arc::from("x"),
                    depth: 0,
                },
                op: CmpOp::Gt,
                right: CompiledExpr::Const(Value::SmallInt(1)),
            }),
        };

        let mut ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        // 1 is not > 1, so result is false
        assert!(!guard.eval_with_array(&mut ctx, &array_state).unwrap());
    }

    #[test]
    fn test_compiled_guard_forall_empty_domain() {
        // \A x \in {} : x > 0 (vacuously true)
        let guard = CompiledGuard::ForAll {
            var_name: Arc::from("x"),
            domain: CompiledExpr::Const(Value::empty_set()),
            body: Box::new(CompiledGuard::IntCmp {
                left: CompiledExpr::LocalVar {
                    name: Arc::from("x"),
                    depth: 0,
                },
                op: CmpOp::Gt,
                right: CompiledExpr::Const(Value::SmallInt(0)),
            }),
        };

        let mut ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        // Vacuously true
        assert!(guard.eval_with_array(&mut ctx, &array_state).unwrap());
    }

    #[test]
    fn test_compiled_guard_exists_one_true() {
        // \E x \in {1, 2, 3} : x > 2
        let guard = CompiledGuard::Exists {
            var_name: Arc::from("x"),
            domain: CompiledExpr::Const(Value::set([
                Value::SmallInt(1),
                Value::SmallInt(2),
                Value::SmallInt(3),
            ])),
            body: Box::new(CompiledGuard::IntCmp {
                left: CompiledExpr::LocalVar {
                    name: Arc::from("x"),
                    depth: 0,
                },
                op: CmpOp::Gt,
                right: CompiledExpr::Const(Value::SmallInt(2)),
            }),
        };

        let mut ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        // 3 > 2, so result is true
        assert!(guard.eval_with_array(&mut ctx, &array_state).unwrap());
    }

    #[test]
    fn test_compiled_guard_exists_none_true() {
        // \E x \in {1, 2, 3} : x > 10
        let guard = CompiledGuard::Exists {
            var_name: Arc::from("x"),
            domain: CompiledExpr::Const(Value::set([
                Value::SmallInt(1),
                Value::SmallInt(2),
                Value::SmallInt(3),
            ])),
            body: Box::new(CompiledGuard::IntCmp {
                left: CompiledExpr::LocalVar {
                    name: Arc::from("x"),
                    depth: 0,
                },
                op: CmpOp::Gt,
                right: CompiledExpr::Const(Value::SmallInt(10)),
            }),
        };

        let mut ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        // No element > 10, so result is false
        assert!(!guard.eval_with_array(&mut ctx, &array_state).unwrap());
    }

    #[test]
    fn test_compiled_guard_exists_empty_domain() {
        // \E x \in {} : x > 0 (vacuously false)
        let guard = CompiledGuard::Exists {
            var_name: Arc::from("x"),
            domain: CompiledExpr::Const(Value::empty_set()),
            body: Box::new(CompiledGuard::IntCmp {
                left: CompiledExpr::LocalVar {
                    name: Arc::from("x"),
                    depth: 0,
                },
                op: CmpOp::Gt,
                right: CompiledExpr::Const(Value::SmallInt(0)),
            }),
        };

        let mut ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        // Vacuously false
        assert!(!guard.eval_with_array(&mut ctx, &array_state).unwrap());
    }

    #[test]
    fn test_compiled_guard_nested_quantifiers() {
        // \A x \in {1, 2} : \E y \in {1, 2, 3} : y > x
        // For each x, there exists y > x
        // x=1: y=2,3 work; x=2: y=3 works -> true
        let guard = CompiledGuard::ForAll {
            var_name: Arc::from("x"),
            domain: CompiledExpr::Const(Value::set([Value::SmallInt(1), Value::SmallInt(2)])),
            body: Box::new(CompiledGuard::Exists {
                var_name: Arc::from("y"),
                domain: CompiledExpr::Const(Value::set([
                    Value::SmallInt(1),
                    Value::SmallInt(2),
                    Value::SmallInt(3),
                ])),
                body: Box::new(CompiledGuard::IntCmp {
                    left: CompiledExpr::LocalVar {
                        name: Arc::from("y"),
                        depth: 0,
                    }, // y (innermost)
                    op: CmpOp::Gt,
                    right: CompiledExpr::LocalVar {
                        name: Arc::from("x"),
                        depth: 1,
                    }, // x (outer)
                }),
            }),
        };

        let mut ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        assert!(guard.eval_with_array(&mut ctx, &array_state).unwrap());
    }

    // === FuncApp tests (Re: #53) ===

    #[test]
    fn test_compiled_expr_func_app_basic() {
        // Test f[key] where f is a state variable containing a function
        // State: f = [1 |-> "a", 2 |-> "b", 3 |-> "c"]
        let mut domain = im::OrdSet::new();
        domain.insert(Value::SmallInt(1));
        domain.insert(Value::SmallInt(2));
        domain.insert(Value::SmallInt(3));
        let mut mapping = im::OrdMap::new();
        mapping.insert(Value::SmallInt(1), Value::string("a"));
        mapping.insert(Value::SmallInt(2), Value::string("b"));
        mapping.insert(Value::SmallInt(3), Value::string("c"));
        let func = Value::Func(crate::value::FuncValue::new(domain, mapping));

        let expr = CompiledExpr::FuncApp {
            state: StateRef::Current,
            func_var: VarIndex(0),
            key: Box::new(CompiledExpr::Const(Value::SmallInt(2))),
        };

        let ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![func]);

        let result = expr.eval_with_array(&ctx, &array_state).unwrap();
        assert_eq!(result, Value::string("b"));
    }

    #[test]
    fn test_compiled_expr_func_app_with_local_var_key() {
        // Test f[x] where x is a local variable
        // State: f = [1 |-> 10, 2 |-> 20]
        let mut domain = im::OrdSet::new();
        domain.insert(Value::SmallInt(1));
        domain.insert(Value::SmallInt(2));
        let mut mapping = im::OrdMap::new();
        mapping.insert(Value::SmallInt(1), Value::SmallInt(10));
        mapping.insert(Value::SmallInt(2), Value::SmallInt(20));
        let func = Value::Func(crate::value::FuncValue::new(domain, mapping));

        let expr = CompiledExpr::FuncApp {
            state: StateRef::Current,
            func_var: VarIndex(0),
            key: Box::new(CompiledExpr::LocalVar {
                name: Arc::from("x"),
                depth: 0,
            }),
        };

        let mut ctx = EvalCtx::new();
        ctx.push_binding(Arc::from("x"), Value::SmallInt(2));
        let array_state = ArrayState::from_values(vec![func]);

        let result = expr.eval_with_array(&ctx, &array_state).unwrap();
        assert_eq!(result, Value::SmallInt(20));
    }

    #[test]
    fn test_compiled_expr_dyn_func_app() {
        // Test func[key] where func is a compiled expression (not a state var)
        // This tests DynFuncApp
        let mut domain = im::OrdSet::new();
        domain.insert(Value::SmallInt(1));
        domain.insert(Value::SmallInt(2));
        let mut mapping = im::OrdMap::new();
        mapping.insert(Value::SmallInt(1), Value::string("first"));
        mapping.insert(Value::SmallInt(2), Value::string("second"));
        let func = Value::Func(crate::value::FuncValue::new(domain, mapping));

        let expr = CompiledExpr::DynFuncApp {
            func: Box::new(CompiledExpr::Const(func)),
            key: Box::new(CompiledExpr::Const(Value::SmallInt(1))),
        };

        let ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        let result = expr.eval_with_array(&ctx, &array_state).unwrap();
        assert_eq!(result, Value::string("first"));
    }

    #[test]
    fn test_compiled_expr_dyn_func_app_with_local_var() {
        // Test localVar[key] where localVar is a function bound in context
        let mut domain = im::OrdSet::new();
        domain.insert(Value::string("a"));
        domain.insert(Value::string("b"));
        let mut mapping = im::OrdMap::new();
        mapping.insert(Value::string("a"), Value::SmallInt(100));
        mapping.insert(Value::string("b"), Value::SmallInt(200));
        let func = Value::Func(crate::value::FuncValue::new(domain, mapping));

        let expr = CompiledExpr::DynFuncApp {
            func: Box::new(CompiledExpr::LocalVar {
                name: Arc::from("f"),
                depth: 0,
            }),
            key: Box::new(CompiledExpr::Const(Value::string("b"))),
        };

        let mut ctx = EvalCtx::new();
        ctx.push_binding(Arc::from("f"), func);
        let array_state = ArrayState::from_values(vec![]);

        let result = expr.eval_with_array(&ctx, &array_state).unwrap();
        assert_eq!(result, Value::SmallInt(200));
    }

    // === EXCEPT tests (Re: #53) ===

    #[test]
    fn test_compiled_expr_except_basic() {
        // [f EXCEPT ![2] = 99] where f = [1 |-> 10, 2 |-> 20]
        let mut domain = im::OrdSet::new();
        domain.insert(Value::SmallInt(1));
        domain.insert(Value::SmallInt(2));
        let mut mapping = im::OrdMap::new();
        mapping.insert(Value::SmallInt(1), Value::SmallInt(10));
        mapping.insert(Value::SmallInt(2), Value::SmallInt(20));
        let func = Value::Func(crate::value::FuncValue::new(domain.clone(), mapping));

        let expr = CompiledExpr::Except {
            func: Box::new(CompiledExpr::Const(func)),
            key: Box::new(CompiledExpr::Const(Value::SmallInt(2))),
            value: Box::new(CompiledExpr::Const(Value::SmallInt(99))),
        };

        let ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        let result = expr.eval_with_array(&ctx, &array_state).unwrap();

        // Verify result is a function with updated mapping
        if let Value::Func(fv) = result {
            assert_eq!(fv.apply(&Value::SmallInt(1)), Some(&Value::SmallInt(10)));
            assert_eq!(fv.apply(&Value::SmallInt(2)), Some(&Value::SmallInt(99)));
        } else {
            panic!("Expected function value");
        }
    }

    #[test]
    fn test_compiled_expr_except_with_state_var() {
        // [var EXCEPT ![key] = val] where var is a state variable
        let mut domain = im::OrdSet::new();
        domain.insert(Value::string("x"));
        domain.insert(Value::string("y"));
        let mut mapping = im::OrdMap::new();
        mapping.insert(Value::string("x"), Value::SmallInt(1));
        mapping.insert(Value::string("y"), Value::SmallInt(2));
        let func = Value::Func(crate::value::FuncValue::new(domain, mapping));

        let expr = CompiledExpr::Except {
            func: Box::new(CompiledExpr::StateVar {
                state: StateRef::Current,
                idx: VarIndex(0),
            }),
            key: Box::new(CompiledExpr::Const(Value::string("x"))),
            value: Box::new(CompiledExpr::Const(Value::SmallInt(42))),
        };

        let ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![func]);

        let result = expr.eval_with_array(&ctx, &array_state).unwrap();

        if let Value::Func(fv) = result {
            assert_eq!(fv.apply(&Value::string("x")), Some(&Value::SmallInt(42)));
            assert_eq!(fv.apply(&Value::string("y")), Some(&Value::SmallInt(2)));
        } else {
            panic!("Expected function value");
        }
    }

    #[test]
    fn test_compiled_expr_except_nested() {
        // [f EXCEPT ![1][2] = 99] where f = [1 |-> [2 |-> 20, 3 |-> 30]]
        // Result: [1 |-> [2 |-> 99, 3 |-> 30]]

        // Inner function: [2 |-> 20, 3 |-> 30]
        let mut inner_domain = im::OrdSet::new();
        inner_domain.insert(Value::SmallInt(2));
        inner_domain.insert(Value::SmallInt(3));
        let mut inner_mapping = im::OrdMap::new();
        inner_mapping.insert(Value::SmallInt(2), Value::SmallInt(20));
        inner_mapping.insert(Value::SmallInt(3), Value::SmallInt(30));
        let inner_func = Value::Func(crate::value::FuncValue::new(inner_domain, inner_mapping));

        // Outer function: [1 |-> inner_func]
        let mut outer_domain = im::OrdSet::new();
        outer_domain.insert(Value::SmallInt(1));
        let mut outer_mapping = im::OrdMap::new();
        outer_mapping.insert(Value::SmallInt(1), inner_func);
        let outer_func = Value::Func(crate::value::FuncValue::new(outer_domain, outer_mapping));

        let expr = CompiledExpr::ExceptNested {
            func: Box::new(CompiledExpr::Const(outer_func)),
            keys: vec![
                CompiledExpr::Const(Value::SmallInt(1)),
                CompiledExpr::Const(Value::SmallInt(2)),
            ],
            value: Box::new(CompiledExpr::Const(Value::SmallInt(99))),
        };

        let ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        let result = expr.eval_with_array(&ctx, &array_state).unwrap();

        // Navigate: result[1][2] should be 99, result[1][3] should be 30
        if let Value::Func(outer_fv) = result {
            let inner = outer_fv.apply(&Value::SmallInt(1)).unwrap();
            if let Value::Func(inner_fv) = inner {
                assert_eq!(inner_fv.apply(&Value::SmallInt(2)), Some(&Value::SmallInt(99)));
                assert_eq!(inner_fv.apply(&Value::SmallInt(3)), Some(&Value::SmallInt(30)));
            } else {
                panic!("Expected inner function value");
            }
        } else {
            panic!("Expected outer function value");
        }
    }

    // === Sequence operation tests (Re: #53) ===

    #[test]
    fn test_compiled_expr_seq_head() {
        // Head(<<1, 2, 3>>) = 1
        let seq = Value::Tuple(Arc::from(vec![
            Value::SmallInt(1),
            Value::SmallInt(2),
            Value::SmallInt(3),
        ]));

        let expr = CompiledExpr::SeqHead(Box::new(CompiledExpr::Const(seq)));

        let ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        let result = expr.eval_with_array(&ctx, &array_state).unwrap();
        assert_eq!(result, Value::SmallInt(1));
    }

    #[test]
    fn test_compiled_expr_seq_tail() {
        // Tail(<<1, 2, 3>>) = <<2, 3>>
        let seq = Value::Tuple(Arc::from(vec![
            Value::SmallInt(1),
            Value::SmallInt(2),
            Value::SmallInt(3),
        ]));

        let expr = CompiledExpr::SeqTail(Box::new(CompiledExpr::Const(seq)));

        let ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        let result = expr.eval_with_array(&ctx, &array_state).unwrap();
        assert_eq!(
            result,
            Value::Tuple(Arc::from(vec![Value::SmallInt(2), Value::SmallInt(3)]))
        );
    }

    #[test]
    fn test_compiled_expr_seq_append() {
        // Append(<<1, 2>>, 3) = <<1, 2, 3>>
        let seq = Value::Tuple(Arc::from(vec![Value::SmallInt(1), Value::SmallInt(2)]));

        let expr = CompiledExpr::SeqAppend {
            seq: Box::new(CompiledExpr::Const(seq)),
            elem: Box::new(CompiledExpr::Const(Value::SmallInt(3))),
        };

        let ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        let result = expr.eval_with_array(&ctx, &array_state).unwrap();
        assert_eq!(
            result,
            Value::Tuple(Arc::from(vec![
                Value::SmallInt(1),
                Value::SmallInt(2),
                Value::SmallInt(3)
            ]))
        );
    }

    #[test]
    fn test_compiled_expr_seq_len() {
        // Len(<<1, 2, 3, 4>>) = 4
        let seq = Value::Tuple(Arc::from(vec![
            Value::SmallInt(1),
            Value::SmallInt(2),
            Value::SmallInt(3),
            Value::SmallInt(4),
        ]));

        let expr = CompiledExpr::SeqLen(Box::new(CompiledExpr::Const(seq)));

        let ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        let result = expr.eval_with_array(&ctx, &array_state).unwrap();
        assert_eq!(result, Value::SmallInt(4));
    }

    #[test]
    fn test_compiled_expr_seq_len_empty() {
        // Len(<<>>) = 0
        let seq = Value::Tuple(Arc::from(vec![]));

        let expr = CompiledExpr::SeqLen(Box::new(CompiledExpr::Const(seq)));

        let ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        let result = expr.eval_with_array(&ctx, &array_state).unwrap();
        assert_eq!(result, Value::SmallInt(0));
    }

    // === FuncDef tests (Re: #53) ===

    #[test]
    fn test_compiled_expr_func_def_basic() {
        // [x \in {0, 1, 2} |-> x * 2]  (non-sequence domain to get Func not Seq)
        // Result: [0 |-> 0, 1 |-> 2, 2 |-> 4]
        let expr = CompiledExpr::FuncDef {
            var_name: Arc::from("x"),
            domain: Box::new(CompiledExpr::Const(Value::set([
                Value::SmallInt(0),
                Value::SmallInt(1),
                Value::SmallInt(2),
            ]))),
            body: Box::new(CompiledExpr::BinOp {
                left: Box::new(CompiledExpr::LocalVar {
                    name: Arc::from("x"),
                    depth: 0,
                }),
                op: BinOp::IntMul,
                right: Box::new(CompiledExpr::Const(Value::SmallInt(2))),
            }),
        };

        let ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        let result = expr.eval_with_array(&ctx, &array_state).unwrap();

        if let Value::Func(fv) = result {
            assert_eq!(fv.apply(&Value::SmallInt(0)), Some(&Value::SmallInt(0)));
            assert_eq!(fv.apply(&Value::SmallInt(1)), Some(&Value::SmallInt(2)));
            assert_eq!(fv.apply(&Value::SmallInt(2)), Some(&Value::SmallInt(4)));
        } else {
            panic!("Expected function value, got {:?}", result);
        }
    }

    #[test]
    fn test_compiled_expr_func_def_seq_domain() {
        // [x \in {1, 2, 3} |-> x * 2]  (1..n domain becomes Seq)
        // Result: <<2, 4, 6>> (TLA+ sequences are 1-indexed)
        let expr = CompiledExpr::FuncDef {
            var_name: Arc::from("x"),
            domain: Box::new(CompiledExpr::Const(Value::set([
                Value::SmallInt(1),
                Value::SmallInt(2),
                Value::SmallInt(3),
            ]))),
            body: Box::new(CompiledExpr::BinOp {
                left: Box::new(CompiledExpr::LocalVar {
                    name: Arc::from("x"),
                    depth: 0,
                }),
                op: BinOp::IntMul,
                right: Box::new(CompiledExpr::Const(Value::SmallInt(2))),
            }),
        };

        let ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        let result = expr.eval_with_array(&ctx, &array_state).unwrap();

        // Domain {1, 2, 3} is sequence domain, returns Seq
        if let Value::Seq(seq) = result {
            assert_eq!(seq.len(), 3);
            assert_eq!(seq[0], Value::SmallInt(2));
            assert_eq!(seq[1], Value::SmallInt(4));
            assert_eq!(seq[2], Value::SmallInt(6));
        } else {
            panic!("Expected sequence value for 1..n domain, got {:?}", result);
        }
    }

    #[test]
    fn test_compiled_expr_func_def_with_conditional() {
        // [x \in {0, 1, 2} |-> IF x = 1 THEN 100 ELSE x]
        // Result: [0 |-> 0, 1 |-> 100, 2 |-> 2]
        let expr = CompiledExpr::FuncDef {
            var_name: Arc::from("x"),
            domain: Box::new(CompiledExpr::Const(Value::set([
                Value::SmallInt(0),
                Value::SmallInt(1),
                Value::SmallInt(2),
            ]))),
            body: Box::new(CompiledExpr::IfThenElse {
                condition: Box::new(CompiledExpr::BoolEq {
                    left: Box::new(CompiledExpr::LocalVar {
                        name: Arc::from("x"),
                        depth: 0,
                    }),
                    right: Box::new(CompiledExpr::Const(Value::SmallInt(1))),
                }),
                then_branch: Box::new(CompiledExpr::Const(Value::SmallInt(100))),
                else_branch: Box::new(CompiledExpr::LocalVar {
                    name: Arc::from("x"),
                    depth: 0,
                }),
            }),
        };

        let ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        let result = expr.eval_with_array(&ctx, &array_state).unwrap();

        if let Value::Func(fv) = result {
            assert_eq!(fv.apply(&Value::SmallInt(0)), Some(&Value::SmallInt(0)));
            assert_eq!(fv.apply(&Value::SmallInt(1)), Some(&Value::SmallInt(100)));
            assert_eq!(fv.apply(&Value::SmallInt(2)), Some(&Value::SmallInt(2)));
        } else {
            panic!("Expected function value, got {:?}", result);
        }
    }

    #[test]
    fn test_compiled_expr_func_def_empty_domain() {
        // [x \in {} |-> x]
        // Result: empty sequence (empty 1..n domain is detected as sequence)
        let expr = CompiledExpr::FuncDef {
            var_name: Arc::from("x"),
            domain: Box::new(CompiledExpr::Const(Value::empty_set())),
            body: Box::new(CompiledExpr::LocalVar {
                name: Arc::from("x"),
                depth: 0,
            }),
        };

        let ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        let result = expr.eval_with_array(&ctx, &array_state).unwrap();

        // Empty domain produces empty Seq (since {} is considered a valid 1..0 sequence)
        if let Value::Seq(seq) = result {
            assert!(seq.is_empty());
        } else {
            panic!("Expected empty sequence value for empty domain, got {:?}", result);
        }
    }

    #[test]
    fn test_compiled_expr_seq_range() {
        // Range(<<10, 20, 30>>) = {10, 20, 30}
        let seq = Value::Tuple(Arc::from(vec![
            Value::SmallInt(10),
            Value::SmallInt(20),
            Value::SmallInt(30),
        ]));
        let expr = CompiledExpr::SeqRange(Box::new(CompiledExpr::Const(seq)));

        let ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        let result = expr.eval_with_array(&ctx, &array_state).unwrap();
        let expected = Value::set([Value::SmallInt(10), Value::SmallInt(20), Value::SmallInt(30)]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_compiled_expr_seq_range_with_duplicates() {
        // Range(<<1, 2, 1, 3, 2>>) = {1, 2, 3}
        let seq = Value::Tuple(Arc::from(vec![
            Value::SmallInt(1),
            Value::SmallInt(2),
            Value::SmallInt(1),
            Value::SmallInt(3),
            Value::SmallInt(2),
        ]));
        let expr = CompiledExpr::SeqRange(Box::new(CompiledExpr::Const(seq)));

        let ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        let result = expr.eval_with_array(&ctx, &array_state).unwrap();
        let expected = Value::set([Value::SmallInt(1), Value::SmallInt(2), Value::SmallInt(3)]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_compiled_expr_seq_range_empty() {
        // Range(<<>>) = {}
        let seq = Value::Tuple(Arc::from(vec![]));
        let expr = CompiledExpr::SeqRange(Box::new(CompiledExpr::Const(seq)));

        let ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        let result = expr.eval_with_array(&ctx, &array_state).unwrap();
        assert_eq!(result, Value::empty_set());
    }

    #[test]
    fn test_compiled_expr_record() {
        // [a |-> 1, b |-> 2, c |-> 3]
        let expr = CompiledExpr::Record(vec![
            (Arc::from("a"), CompiledExpr::Const(Value::SmallInt(1))),
            (Arc::from("b"), CompiledExpr::Const(Value::SmallInt(2))),
            (Arc::from("c"), CompiledExpr::Const(Value::SmallInt(3))),
        ]);

        let ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        let result = expr.eval_with_array(&ctx, &array_state).unwrap();
        // Verify by accessing fields
        if let Value::Record(rec) = result {
            assert_eq!(rec.get("a"), Some(&Value::SmallInt(1)));
            assert_eq!(rec.get("b"), Some(&Value::SmallInt(2)));
            assert_eq!(rec.get("c"), Some(&Value::SmallInt(3)));
        } else {
            panic!("Expected Record value, got {:?}", result);
        }
    }

    #[test]
    fn test_compiled_expr_record_with_computed_values() {
        // [x |-> 2 + 3, y |-> 4 * 5]
        let expr = CompiledExpr::Record(vec![
            (
                Arc::from("x"),
                CompiledExpr::BinOp {
                    left: Box::new(CompiledExpr::Const(Value::SmallInt(2))),
                    op: BinOp::IntAdd,
                    right: Box::new(CompiledExpr::Const(Value::SmallInt(3))),
                },
            ),
            (
                Arc::from("y"),
                CompiledExpr::BinOp {
                    left: Box::new(CompiledExpr::Const(Value::SmallInt(4))),
                    op: BinOp::IntMul,
                    right: Box::new(CompiledExpr::Const(Value::SmallInt(5))),
                },
            ),
        ]);

        let ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        let result = expr.eval_with_array(&ctx, &array_state).unwrap();
        if let Value::Record(rec) = result {
            assert_eq!(rec.get("x"), Some(&Value::SmallInt(5)));
            assert_eq!(rec.get("y"), Some(&Value::SmallInt(20)));
        } else {
            panic!("Expected Record value, got {:?}", result);
        }
    }

    #[test]
    fn test_compiled_expr_record_access() {
        // [a |-> 10, b |-> 20].b = 20
        let record = Value::Record(crate::value::RecordValue::from_sorted_entries(vec![
            (Arc::from("a"), Value::SmallInt(10)),
            (Arc::from("b"), Value::SmallInt(20)),
        ]));
        let expr = CompiledExpr::RecordAccess {
            record: Box::new(CompiledExpr::Const(record)),
            field: Arc::from("b"),
        };

        let ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![]);

        let result = expr.eval_with_array(&ctx, &array_state).unwrap();
        assert_eq!(result, Value::SmallInt(20));
    }

    #[test]
    fn test_compiled_expr_record_access_from_state_var() {
        // state_var.field where state_var is a record
        let record = Value::Record(crate::value::RecordValue::from_sorted_entries(vec![
            (Arc::from("count"), Value::SmallInt(42)),
            (Arc::from("status"), Value::string("active")),
        ]));
        let expr = CompiledExpr::RecordAccess {
            record: Box::new(CompiledExpr::StateVar {
                state: StateRef::Current,
                idx: VarIndex(0),
            }),
            field: Arc::from("status"),
        };

        let ctx = EvalCtx::new();
        let array_state = ArrayState::from_values(vec![record]);

        let result = expr.eval_with_array(&ctx, &array_state).unwrap();
        assert_eq!(result, Value::string("active"));
    }

    #[test]
    fn test_compiled_expr_func_app_int_func_local_var() {
        // pc[self] where pc is IntFunc, self is local var
        // State: pc = [0 |-> "init", 1 |-> "done", 2 |-> "wait"]
        // Local: self = 1
        // Result: "done"
        use crate::value::IntIntervalFunc;

        let pc = Value::IntFunc(IntIntervalFunc::new(
            0,
            2,
            vec![
                Value::string("init"),
                Value::string("done"),
                Value::string("wait"),
            ],
        ));

        let expr = CompiledExpr::FuncAppIntFuncLocalVar {
            state: StateRef::Current,
            func_var: VarIndex(0),
            key_depth: 0, // most recent local var
        };

        let mut ctx = EvalCtx::new();
        ctx.push_binding(Arc::from("self"), Value::SmallInt(1));
        let array_state = ArrayState::from_values(vec![pc]);

        let result = expr.eval_with_array(&ctx, &array_state).unwrap();
        assert_eq!(result, Value::string("done"));
    }

    #[test]
    fn test_compiled_expr_func_app_int_func_local_var_nested_depth() {
        // Inner local var: f[outer] where f uses depth=1 (second most recent)
        // Locals: outer=2, inner=99 (inner is depth=0, outer is depth=1)
        // State: f = [0 |-> "a", 1 |-> "b", 2 |-> "c"]
        // Result: f[outer] = f[2] = "c"
        use crate::value::IntIntervalFunc;

        let f = Value::IntFunc(IntIntervalFunc::new(
            0,
            2,
            vec![Value::string("a"), Value::string("b"), Value::string("c")],
        ));

        let expr = CompiledExpr::FuncAppIntFuncLocalVar {
            state: StateRef::Current,
            func_var: VarIndex(0),
            key_depth: 1, // second most recent (outer)
        };

        let mut ctx = EvalCtx::new();
        ctx.push_binding(Arc::from("outer"), Value::SmallInt(2));
        ctx.push_binding(Arc::from("inner"), Value::SmallInt(99));
        let array_state = ArrayState::from_values(vec![f]);

        let result = expr.eval_with_array(&ctx, &array_state).unwrap();
        assert_eq!(result, Value::string("c"));
    }

    #[test]
    fn test_compiled_guard_int_func_local_var_eq_const() {
        // pc[self] = "ready" where pc is IntFunc
        // State: pc = [0 |-> "init", 1 |-> "ready", 2 |-> "done"]
        // Local: self = 1
        // Result: true
        use crate::value::IntIntervalFunc;

        let pc = Value::IntFunc(IntIntervalFunc::new(
            0,
            2,
            vec![
                Value::string("init"),
                Value::string("ready"),
                Value::string("done"),
            ],
        ));

        let guard = CompiledGuard::IntFuncLocalVarEqConst {
            state: StateRef::Current,
            func_var: VarIndex(0),
            key_depth: 0,
            expected: Value::string("ready"),
        };

        let mut ctx = EvalCtx::new();
        ctx.push_binding(Arc::from("self"), Value::SmallInt(1));
        let array_state = ArrayState::from_values(vec![pc]);

        assert!(guard.eval_with_array(&mut ctx, &array_state).unwrap());
    }

    #[test]
    fn test_compiled_guard_int_func_local_var_eq_const_false() {
        // pc[self] = "done" when pc[self] = "ready"
        // Result: false
        use crate::value::IntIntervalFunc;

        let pc = Value::IntFunc(IntIntervalFunc::new(
            0,
            2,
            vec![
                Value::string("init"),
                Value::string("ready"),
                Value::string("done"),
            ],
        ));

        let guard = CompiledGuard::IntFuncLocalVarEqConst {
            state: StateRef::Current,
            func_var: VarIndex(0),
            key_depth: 0,
            expected: Value::string("done"), // expecting "done" but will get "ready"
        };

        let mut ctx = EvalCtx::new();
        ctx.push_binding(Arc::from("self"), Value::SmallInt(1));
        let array_state = ArrayState::from_values(vec![pc]);

        assert!(!guard.eval_with_array(&mut ctx, &array_state).unwrap());
    }

    #[test]
    fn test_compiled_guard_int_func_local_var_neq_const() {
        // pc[self] # "init" where pc[self] = "ready"
        // Result: true
        use crate::value::IntIntervalFunc;

        let pc = Value::IntFunc(IntIntervalFunc::new(
            0,
            2,
            vec![
                Value::string("init"),
                Value::string("ready"),
                Value::string("done"),
            ],
        ));

        let guard = CompiledGuard::IntFuncLocalVarNeqConst {
            state: StateRef::Current,
            func_var: VarIndex(0),
            key_depth: 0,
            expected: Value::string("init"), // not "init", and we have "ready"
        };

        let mut ctx = EvalCtx::new();
        ctx.push_binding(Arc::from("self"), Value::SmallInt(1));
        let array_state = ArrayState::from_values(vec![pc]);

        assert!(guard.eval_with_array(&mut ctx, &array_state).unwrap());
    }

    #[test]
    fn test_compiled_guard_int_func_local_var_neq_const_false() {
        // pc[self] # "ready" when pc[self] = "ready"
        // Result: false
        use crate::value::IntIntervalFunc;

        let pc = Value::IntFunc(IntIntervalFunc::new(
            0,
            2,
            vec![
                Value::string("init"),
                Value::string("ready"),
                Value::string("done"),
            ],
        ));

        let guard = CompiledGuard::IntFuncLocalVarNeqConst {
            state: StateRef::Current,
            func_var: VarIndex(0),
            key_depth: 0,
            expected: Value::string("ready"), // # "ready", but we have "ready"
        };

        let mut ctx = EvalCtx::new();
        ctx.push_binding(Arc::from("self"), Value::SmallInt(1));
        let array_state = ArrayState::from_values(vec![pc]);

        assert!(!guard.eval_with_array(&mut ctx, &array_state).unwrap());
    }
}
